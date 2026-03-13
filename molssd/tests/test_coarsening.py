"""Tests for the spectral graph coarsening module (molssd.core.coarsening)."""
import torch
import pytest

from molssd.core.coarsening import (
    compute_graph_laplacian,
    compute_eigendecomposition,
    spectral_clustering,
    build_coarsening_matrix,
    build_coarsened_adjacency,
    build_coarsening_hierarchy,
    coarsen_positions,
    lift_positions,
)


# -------------------------------------------------------------------
# Graph Laplacian
# -------------------------------------------------------------------


class TestGraphLaplacian:
    """Tests for compute_graph_laplacian."""

    def test_graph_laplacian_shape(self, small_molecule):
        """L has correct shape (N, N)."""
        adj = small_molecule["adj"]
        n = small_molecule["num_atoms"]
        L = compute_graph_laplacian(adj, normalized=False)
        assert L.shape == (n, n)

    def test_graph_laplacian_properties(self, small_molecule):
        """L is symmetric, rows sum to zero, and is positive semi-definite."""
        adj = small_molecule["adj"]
        L = compute_graph_laplacian(adj, normalized=False)

        # Symmetric
        assert torch.allclose(L, L.T, atol=1e-6), "Laplacian must be symmetric"

        # Row sums = 0
        row_sums = L.sum(dim=1)
        assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-6), (
            "Row sums of the Laplacian must be zero"
        )

        # Positive semi-definite: all eigenvalues >= 0
        eigenvalues = torch.linalg.eigvalsh(L)
        assert (eigenvalues >= -1e-6).all(), (
            f"Laplacian must be PSD, got eigenvalues {eigenvalues}"
        )

    def test_normalized_laplacian(self, medium_molecule):
        """Normalized Laplacian has eigenvalues in [0, 2]."""
        adj = medium_molecule["adj"]
        L_norm = compute_graph_laplacian(adj, normalized=True)

        eigenvalues = torch.linalg.eigvalsh(L_norm)
        assert (eigenvalues >= -1e-6).all(), (
            f"Normalized Laplacian eigenvalues must be >= 0, got min {eigenvalues.min()}"
        )
        assert (eigenvalues <= 2.0 + 1e-6).all(), (
            f"Normalized Laplacian eigenvalues must be <= 2, got max {eigenvalues.max()}"
        )


# -------------------------------------------------------------------
# Eigendecomposition
# -------------------------------------------------------------------


class TestEigendecomposition:

    def test_eigendecomposition(self, small_molecule):
        """Eigenvalues are non-negative and smallest is approximately zero."""
        adj = small_molecule["adj"]
        L = compute_graph_laplacian(adj)
        eigenvalues, eigenvectors = compute_eigendecomposition(L)

        # Non-negative (allow small numerical error)
        assert (eigenvalues >= -1e-5).all(), (
            f"Eigenvalues must be non-negative, got {eigenvalues}"
        )

        # Smallest eigenvalue is ~0 for a connected graph
        assert eigenvalues[0].abs() < 1e-5, (
            f"Smallest eigenvalue of connected graph should be ~0, got {eigenvalues[0]}"
        )


# -------------------------------------------------------------------
# Spectral clustering
# -------------------------------------------------------------------


class TestSpectralClustering:

    def test_spectral_clustering(self, medium_molecule, seed):
        """Spectral clustering produces the correct number of clusters and
        assigns all nodes."""
        adj = medium_molecule["adj"]
        n = medium_molecule["num_atoms"]
        n_clusters = 4

        L = compute_graph_laplacian(adj)
        eigenvalues, eigenvectors = compute_eigendecomposition(L, k=n_clusters)
        labels = spectral_clustering(eigenvectors, n_clusters)

        # Correct number of assignments
        assert labels.shape == (n,)

        # All labels in valid range
        assert labels.min() >= 0
        assert labels.max() < n_clusters

        # All clusters are used (for a reasonable graph)
        unique_labels = labels.unique()
        assert len(unique_labels) == n_clusters, (
            f"Expected {n_clusters} distinct clusters, got {len(unique_labels)}"
        )


# -------------------------------------------------------------------
# Coarsening matrix
# -------------------------------------------------------------------


class TestCoarseningMatrix:

    def _get_simple_clustering(self, n, n_clusters):
        """Helper: deterministic cluster assignment for testing."""
        return torch.arange(n) % n_clusters

    def test_coarsening_matrix_shape(self, small_molecule):
        """C has shape (n_clusters, N)."""
        n = small_molecule["num_atoms"]
        n_clusters = 2
        labels = self._get_simple_clustering(n, n_clusters)
        C = build_coarsening_matrix(labels, n_clusters)
        assert C.shape == (n_clusters, n)

    def test_coarsening_matrix_row_sums(self, small_molecule):
        """Each row of C sums to 1 (mass-weighted averaging within cluster)."""
        n = small_molecule["num_atoms"]
        masses = small_molecule["atomic_masses"]
        n_clusters = 2
        labels = self._get_simple_clustering(n, n_clusters)
        C = build_coarsening_matrix(labels, n_clusters, atomic_masses=masses)

        row_sums = C.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(n_clusters), atol=1e-6), (
            f"Row sums should be 1, got {row_sums}"
        )

    def test_coarsening_matrix_mass_weighted(self):
        """Mass weighting is correct: C_{I,j} = m_j / sum_cluster m."""
        masses = torch.tensor([2.0, 3.0, 5.0])
        labels = torch.tensor([0, 0, 1])  # first two in cluster 0, third in cluster 1
        C = build_coarsening_matrix(labels, 2, atomic_masses=masses)

        # Cluster 0: atoms 0, 1 with masses 2, 3. Total = 5.
        assert torch.isclose(C[0, 0], torch.tensor(2.0 / 5.0), atol=1e-6)
        assert torch.isclose(C[0, 1], torch.tensor(3.0 / 5.0), atol=1e-6)
        assert torch.isclose(C[0, 2], torch.tensor(0.0), atol=1e-6)

        # Cluster 1: atom 2 with mass 5. Total = 5.
        assert torch.isclose(C[1, 2], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(C[1, 0], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(C[1, 1], torch.tensor(0.0), atol=1e-6)


# -------------------------------------------------------------------
# Coarsened adjacency
# -------------------------------------------------------------------


class TestCoarsenedAdjacency:

    def test_build_coarsened_adjacency(self, medium_molecule, seed):
        """Coarsened adjacency is binary, symmetric, and has no self-loops."""
        adj = medium_molecule["adj"]
        n = medium_molecule["num_atoms"]
        n_clusters = 4

        L = compute_graph_laplacian(adj)
        _, eigvecs = compute_eigendecomposition(L, k=n_clusters)
        labels = spectral_clustering(eigvecs, n_clusters)

        A_coarse = build_coarsened_adjacency(adj, labels, n_clusters)

        # Shape
        assert A_coarse.shape == (n_clusters, n_clusters)

        # Binary
        unique_vals = A_coarse.unique()
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist()), (
            f"Adjacency must be binary, got unique values {unique_vals}"
        )

        # Symmetric
        assert torch.allclose(A_coarse, A_coarse.T, atol=1e-6)

        # No self-loops
        assert torch.allclose(A_coarse.diag(), torch.zeros(n_clusters), atol=1e-6)


# -------------------------------------------------------------------
# Hierarchy building
# -------------------------------------------------------------------


class TestHierarchy:

    def test_build_hierarchy_small(self, small_molecule, seed):
        """Small molecule (3 atoms) gets a valid hierarchy."""
        adj = small_molecule["adj"]
        n = small_molecule["num_atoms"]
        masses = small_molecule["atomic_masses"]

        hierarchy = build_coarsening_hierarchy(
            adj, n, atomic_masses=masses,
        )

        # Should produce at least one coarsening level
        assert len(hierarchy) >= 1

        # Every level must have fewer nodes than the previous
        prev_n = n
        for level in hierarchy:
            assert level.num_nodes < prev_n, (
                f"Level must reduce node count: {level.num_nodes} >= {prev_n}"
            )
            assert level.coarsening_matrix.shape == (level.num_nodes, prev_n)
            prev_n = level.num_nodes

        # Final level should reach 1 node
        assert hierarchy[-1].num_nodes == 1

    def test_build_hierarchy_medium(self, medium_molecule, seed):
        """Medium molecule (12 atoms) gets a multi-level hierarchy."""
        adj = medium_molecule["adj"]
        n = medium_molecule["num_atoms"]
        masses = medium_molecule["atomic_masses"]

        hierarchy = build_coarsening_hierarchy(
            adj, n, atomic_masses=masses,
        )

        # For 12 atoms with 3-fold reduction we expect at least 2 levels
        # 12 -> 4 -> 1  (or similar)
        assert len(hierarchy) >= 2, (
            f"Expected >= 2 levels for 12 atoms, got {len(hierarchy)}"
        )

        # Check monotonically decreasing node counts
        prev_n = n
        for level in hierarchy:
            assert level.num_nodes < prev_n
            prev_n = level.num_nodes


# -------------------------------------------------------------------
# Position coarsening and lifting, equivariance
# -------------------------------------------------------------------


class TestPositionOps:

    def test_coarsen_positions_equivariance(self, medium_molecule, random_rotation, seed):
        """C @ (R @ X) == R @ (C @ X) for rotation R.

        Coarsening via mass-weighted center-of-mass commutes with rotations
        because C is a linear operator that only mixes atoms within clusters,
        and rotation acts identically on all atoms.
        """
        adj = medium_molecule["adj"]
        n = medium_molecule["num_atoms"]
        masses = medium_molecule["atomic_masses"]
        X = medium_molecule["positions"]
        R = random_rotation

        hierarchy = build_coarsening_hierarchy(adj, n, atomic_masses=masses)
        C = hierarchy[0].coarsening_matrix

        # Rotate then coarsen
        X_rotated = (R @ X.T).T  # (N, 3)
        coarse_of_rotated = coarsen_positions(C, X_rotated)

        # Coarsen then rotate
        coarse_X = coarsen_positions(C, X)
        rotated_of_coarse = (R @ coarse_X.T).T

        assert torch.allclose(coarse_of_rotated, rotated_of_coarse, atol=1e-5), (
            "Coarsening must commute with rotation"
        )

    def test_lift_positions(self, medium_molecule, seed):
        """lift(coarsen(X)) has the same center of mass as X."""
        adj = medium_molecule["adj"]
        n = medium_molecule["num_atoms"]
        masses = medium_molecule["atomic_masses"]
        X = medium_molecule["positions"]

        hierarchy = build_coarsening_hierarchy(adj, n, atomic_masses=masses)
        C = hierarchy[0].coarsening_matrix

        X_coarse = coarsen_positions(C, X)
        X_lifted = lift_positions(C, X_coarse)

        # Center of mass should be preserved
        total_mass = masses.sum()
        com_original = (masses.unsqueeze(1) * X).sum(dim=0) / total_mass
        com_lifted = (masses.unsqueeze(1) * X_lifted).sum(dim=0) / total_mass

        assert torch.allclose(com_original, com_lifted, atol=1e-4), (
            f"Center of mass mismatch: original {com_original} vs lifted {com_lifted}"
        )
