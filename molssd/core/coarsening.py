"""Spectral graph coarsening operators for multi-scale molecular hierarchies.

This module implements the foundation of the MolSSD framework: spectral graph
coarsening that creates a hierarchy of progressively coarser molecular graph
representations. The coarsening is based on spectral clustering of the graph
Laplacian eigenvectors, producing mass-weighted aggregation operators that
preserve SE(3) equivariance through center-of-mass pooling.

The key mathematical objects are:

- Graph Laplacian L = D - A (combinatorial) or D^{-1/2} L D^{-1/2} (normalized)
- Coarsening matrix C^{(k)} in R^{N_k x N_{k-1}}, mapping fine to coarse via
  mass-weighted averaging within spectral clusters
- Multi-level hierarchy: atoms -> super-atoms -> fragments -> ... -> centroid

All operations are differentiable through PyTorch autograd so that the
vector-Jacobian products (VJPs) needed by the SSD reverse process can be
computed automatically.

References:
    - SSD paper (arXiv:2603.08709): Scale-space diffusion framework
    - Spectral clustering: Luxburg, "A Tutorial on Spectral Clustering" (2007)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------

def compute_graph_laplacian(
    adj: torch.Tensor,
    normalized: bool = False,
) -> torch.Tensor:
    """Build the graph Laplacian from an adjacency matrix.

    Computes either the combinatorial (unnormalized) Laplacian L = D - A
    or the symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.

    The combinatorial Laplacian is the default because it preserves
    center-of-mass properties under coarsening, which is essential for
    maintaining SE(3) equivariance in the MolSSD framework.

    Args:
        adj: Adjacency matrix of shape ``(N, N)``. Can be weighted or binary.
            Must be symmetric. Self-loops are permitted but not required.
        normalized: If ``True``, return the symmetric normalized Laplacian.
            If ``False`` (default), return the combinatorial Laplacian.

    Returns:
        Laplacian matrix of shape ``(N, N)``, same dtype and device as *adj*.

    Raises:
        ValueError: If *adj* is not 2-D or not square.
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(
            f"Adjacency matrix must be square 2-D tensor, got shape {adj.shape}"
        )

    # Degree vector
    deg = adj.sum(dim=1)  # (N,)

    if not normalized:
        # L = D - A
        return torch.diag(deg) - adj
    else:
        # L_sym = I - D^{-1/2} A D^{-1/2}
        # Handle isolated nodes (degree 0) by setting their inverse sqrt to 0.
        deg_inv_sqrt = torch.zeros_like(deg)
        nonzero = deg > 0
        deg_inv_sqrt[nonzero] = deg[nonzero].pow(-0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        n = adj.shape[0]
        identity = torch.eye(n, dtype=adj.dtype, device=adj.device)
        return identity - D_inv_sqrt @ adj @ D_inv_sqrt


# ---------------------------------------------------------------------------
# Eigendecomposition
# ---------------------------------------------------------------------------

def compute_eigendecomposition(
    L: torch.Tensor,
    k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the eigendecomposition of a graph Laplacian.

    Uses ``torch.linalg.eigh`` which exploits the symmetry of the Laplacian
    and returns eigenvalues in ascending order. For small molecules (as in
    QM9 with at most 29 atoms), full dense eigendecomposition is efficient.

    Args:
        L: Symmetric Laplacian matrix of shape ``(N, N)``.
        k: Number of smallest eigenvalues/eigenvectors to return. If ``None``,
            return all *N* eigenpairs.

    Returns:
        eigenvalues: Tensor of shape ``(k,)`` (or ``(N,)``), ascending order.
        eigenvectors: Tensor of shape ``(N, k)`` (or ``(N, N)``), columns are
            the corresponding eigenvectors.

    Raises:
        ValueError: If *k* is larger than *N*.
    """
    n = L.shape[0]
    if k is not None and k > n:
        raise ValueError(
            f"Requested k={k} eigenpairs but Laplacian has size {n}"
        )

    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    if k is not None:
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]

    return eigenvalues, eigenvectors


# ---------------------------------------------------------------------------
# Spectral clustering
# ---------------------------------------------------------------------------

def spectral_clustering(
    eigenvectors: torch.Tensor,
    n_clusters: int,
    random_state: int = 0,
    n_init: int = 10,
) -> torch.Tensor:
    """Cluster graph nodes using spectral embedding from Laplacian eigenvectors.

    Takes the bottom-*n_clusters* eigenvectors of the graph Laplacian,
    row-normalizes them, and runs k-means clustering to produce a partition
    of the nodes into *n_clusters* groups.

    The bottom eigenvectors (corresponding to the smallest eigenvalues)
    capture the low-frequency structure of the graph, so clustering on them
    tends to produce well-connected groups that respect the molecular topology.

    Args:
        eigenvectors: Matrix of shape ``(N, k)`` where ``k >= n_clusters``.
            Columns are the eigenvectors corresponding to the smallest
            eigenvalues of the graph Laplacian.
        n_clusters: Number of clusters (super-nodes) to produce.
        random_state: Random seed for k-means reproducibility.
        n_init: Number of k-means restarts (default 10).

    Returns:
        Cluster assignment tensor of shape ``(N,)`` with integer labels in
        ``[0, n_clusters)``, on the same device as *eigenvectors*.

    Raises:
        ValueError: If *n_clusters* exceeds the number of nodes or available
            eigenvectors.
    """
    n, num_eigvecs = eigenvectors.shape
    if n_clusters > n:
        raise ValueError(
            f"Cannot create {n_clusters} clusters from {n} nodes"
        )
    if n_clusters > num_eigvecs:
        raise ValueError(
            f"Need at least {n_clusters} eigenvectors for {n_clusters} "
            f"clusters, but only {num_eigvecs} provided"
        )

    # Select the bottom n_clusters eigenvectors (skip the constant Fiedler
    # vector at index 0 only if it's truly constant; in practice we keep it
    # because for disconnected graphs the multiplicity of eigenvalue 0
    # encodes connectivity information).
    U = eigenvectors[:, :n_clusters]  # (N, n_clusters)

    # Row-normalize so each node's embedding lives on the unit sphere.
    # This improves k-means performance on spectral embeddings.
    row_norms = U.norm(dim=1, keepdim=True).clamp(min=1e-12)
    U_normalized = U / row_norms

    # Move to CPU/numpy for sklearn k-means.
    U_np = U_normalized.detach().cpu().numpy()

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(U_np)

    return torch.tensor(labels, dtype=torch.long, device=eigenvectors.device)


# ---------------------------------------------------------------------------
# Coarsening matrix construction
# ---------------------------------------------------------------------------

def build_coarsening_matrix(
    cluster_assignment: torch.Tensor,
    num_clusters: int,
    atomic_masses: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the mass-weighted coarsening matrix C^{(k)}.

    The coarsening matrix C maps fine-level signals to coarse-level signals
    via mass-weighted averaging within each cluster (super-node)::

        C_{Ij} = m_j / (sum_{i in S_I} m_i)    if j in S_I
        C_{Ij} = 0                              otherwise

    where S_I is the set of fine-level nodes assigned to super-node I and
    m_j is the mass (weight) of node j. When applied to 3D positions,
    ``C @ X`` yields the center-of-mass of each cluster, which preserves
    SE(3) equivariance.

    If *atomic_masses* is ``None``, uniform weights are used (simple averaging
    where each atom contributes equally).

    Args:
        cluster_assignment: Integer tensor of shape ``(N,)`` with values in
            ``[0, num_clusters)``.
        num_clusters: Number of super-nodes (rows of C).
        atomic_masses: Optional tensor of shape ``(N,)`` with positive masses
            for each fine-level node.

    Returns:
        Coarsening matrix of shape ``(num_clusters, N)``, dtype ``float32``
        (or matching *atomic_masses* dtype), on the same device as
        *cluster_assignment*.
    """
    device = cluster_assignment.device
    n = cluster_assignment.shape[0]

    if atomic_masses is not None:
        masses = atomic_masses.to(device=device)
    else:
        masses = torch.ones(n, device=device)

    # Build C as a dense matrix for differentiability.
    C = torch.zeros(num_clusters, n, dtype=masses.dtype, device=device)

    for I in range(num_clusters):
        mask = cluster_assignment == I  # (N,) boolean
        if mask.any():
            cluster_masses = masses[mask]
            total_mass = cluster_masses.sum()
            # C_{I, j} = m_j / total_mass  for j in S_I
            C[I, mask] = masses[mask] / total_mass

    return C


# ---------------------------------------------------------------------------
# Coarsened adjacency
# ---------------------------------------------------------------------------

def build_coarsened_adjacency(
    adj: torch.Tensor,
    cluster_assignment: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """Build the adjacency matrix for the coarsened graph.

    An edge exists between super-nodes I and J (I != J) if any atom in
    cluster I is bonded to any atom in cluster J in the original graph.
    Self-loops are excluded.

    Args:
        adj: Original adjacency matrix of shape ``(N, N)``.
        cluster_assignment: Integer tensor of shape ``(N,)`` mapping each
            atom to its super-node index.
        num_clusters: Number of super-nodes.

    Returns:
        Binary adjacency matrix of shape ``(num_clusters, num_clusters)``
        for the coarsened graph, same device as *adj*.
    """
    device = adj.device
    n = adj.shape[0]

    # Build an indicator matrix P of shape (num_clusters, N) where
    # P[I, j] = 1 if atom j belongs to cluster I.
    P = torch.zeros(num_clusters, n, dtype=adj.dtype, device=device)
    for I in range(num_clusters):
        mask = cluster_assignment == I
        P[I, mask] = 1.0

    # Coarsened adjacency: A_coarse = P @ A @ P^T
    # Entry (I, J) counts the number of edges between clusters I and J.
    A_coarse = P @ adj @ P.t()

    # Binarize (edge exists or not) and remove self-loops.
    A_coarse = (A_coarse > 0).to(adj.dtype)
    A_coarse.fill_diagonal_(0.0)

    return A_coarse


# ---------------------------------------------------------------------------
# CoarseningLevel dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoarseningLevel:
    """Stores all data for one level in the coarsening hierarchy.

    Attributes:
        coarsening_matrix: The matrix C^{(k)} of shape ``(N_k, N_{k-1})``
            that maps fine-level signals to this coarse level.
        cluster_assignment: Integer tensor of shape ``(N_{k-1},)`` mapping
            each node at the previous (finer) level to its super-node index
            at this level.
        coarsened_adj: Adjacency matrix of shape ``(N_k, N_k)`` for the
            graph at this coarsening level.
        num_nodes: Number of nodes (super-nodes) at this level, i.e. N_k.
        eigenvalues: Eigenvalues of the Laplacian at the *previous* level
            that were used to determine the clustering.
        eigenvectors: Eigenvectors of the Laplacian at the *previous* level
            that were used for spectral clustering.
    """
    coarsening_matrix: torch.Tensor
    cluster_assignment: torch.Tensor
    coarsened_adj: torch.Tensor
    num_nodes: int
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor


# ---------------------------------------------------------------------------
# Multi-level hierarchy
# ---------------------------------------------------------------------------

def _compute_target_sizes(num_atoms: int, fold: int = 3) -> List[int]:
    """Compute default target sizes for a coarsening hierarchy.

    Produces approximately *fold*-fold reduction at each level::

        N, N/fold, N/fold^2, ..., 1

    Each level has at least 1 node. Levels where the target size equals
    the previous level (both round to the same integer) are skipped.

    Args:
        num_atoms: Number of atoms at the finest level.
        fold: Reduction factor per level (default 3).

    Returns:
        List of target sizes *excluding* the finest level, in descending
        order. The last entry is always 1.
    """
    sizes: List[int] = []
    current = num_atoms
    while current > 1:
        next_size = max(1, round(current / fold))
        if next_size >= current:
            # Avoid infinite loop if current is already very small.
            next_size = current - 1
            if next_size < 1:
                next_size = 1
        sizes.append(next_size)
        current = next_size
    return sizes


def build_coarsening_hierarchy(
    adj: torch.Tensor,
    num_atoms: int,
    target_sizes: Optional[Sequence[int]] = None,
    atomic_masses: Optional[torch.Tensor] = None,
) -> List[CoarseningLevel]:
    """Build a full multi-level spectral coarsening hierarchy.

    Starting from the finest-level molecular graph, this function
    iteratively:

    1. Computes the graph Laplacian at the current level.
    2. Computes its eigendecomposition.
    3. Performs spectral clustering to determine the next (coarser) level's
       super-node partition.
    4. Builds the mass-weighted coarsening matrix and the coarsened
       adjacency matrix.

    The result is a list of ``CoarseningLevel`` objects from finest-to-coarsest
    (level 0 is the first coarsening, i.e. the mapping from full atoms to the
    first set of super-nodes).

    For QM9 molecules (up to 29 atoms including H, 9 heavy atoms), a typical
    hierarchy has 2-3 levels.

    If *target_sizes* is not provided, sizes are computed automatically using
    approximately 3-fold reduction per level: N -> N/3 -> N/9 -> ... -> 1.

    Args:
        adj: Adjacency matrix of the finest-level molecular graph, shape
            ``(N, N)``.
        num_atoms: Number of atoms at the finest level (should equal
            ``adj.shape[0]``).
        target_sizes: Optional sequence of target super-node counts for each
            coarsening level, in order from fine to coarse. Each entry must
            be smaller than the previous level's node count.
        atomic_masses: Optional tensor of shape ``(N,)`` with atomic masses.
            These are used for mass-weighted center-of-mass coarsening. If
            ``None``, uniform weights are used.

    Returns:
        List of ``CoarseningLevel`` objects, one per coarsening step, ordered
        from finest to coarsest.

    Raises:
        ValueError: If *num_atoms* does not match ``adj.shape[0]``, or if
            any target size is invalid.
    """
    if adj.shape[0] != num_atoms:
        raise ValueError(
            f"adj has {adj.shape[0]} nodes but num_atoms={num_atoms}"
        )

    if num_atoms <= 1:
        return []

    if target_sizes is None:
        target_sizes = _compute_target_sizes(num_atoms)

    hierarchy: List[CoarseningLevel] = []
    current_adj = adj.float()
    current_n = num_atoms
    # Track cumulative masses: at each level, the "mass" of a super-node is
    # the sum of masses of all original atoms it contains.
    current_masses = atomic_masses.float() if atomic_masses is not None else None

    for level_idx, n_clusters in enumerate(target_sizes):
        if n_clusters >= current_n:
            # Skip this level: cannot coarsen to the same or larger size.
            continue
        if n_clusters < 1:
            raise ValueError(
                f"Target size at level {level_idx} must be >= 1, got {n_clusters}"
            )

        # 1. Laplacian of the current-level graph.
        L = compute_graph_laplacian(current_adj, normalized=False)

        # 2. Eigendecomposition. We need at least n_clusters eigenvectors.
        k_eig = min(n_clusters, current_n)
        eigenvalues, eigenvectors = compute_eigendecomposition(L, k=k_eig)

        # 3. Spectral clustering.
        #    Handle edge case: if current_n == n_clusters, each node is its
        #    own cluster (identity coarsening). We skip to avoid trivial work.
        cluster_assignment = spectral_clustering(
            eigenvectors, n_clusters, random_state=level_idx
        )

        # 4. Coarsening matrix.
        C = build_coarsening_matrix(
            cluster_assignment, n_clusters, atomic_masses=current_masses
        )

        # 5. Coarsened adjacency.
        coarsened_adj = build_coarsened_adjacency(
            current_adj, cluster_assignment, n_clusters
        )

        level = CoarseningLevel(
            coarsening_matrix=C,
            cluster_assignment=cluster_assignment,
            coarsened_adj=coarsened_adj,
            num_nodes=n_clusters,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
        )
        hierarchy.append(level)

        # Prepare for next iteration.
        current_adj = coarsened_adj
        current_n = n_clusters

        # Propagate masses: each super-node's mass is the total mass of
        # its constituent atoms.
        if current_masses is not None:
            new_masses = torch.zeros(
                n_clusters, dtype=current_masses.dtype,
                device=current_masses.device,
            )
            for I in range(n_clusters):
                mask = cluster_assignment == I
                new_masses[I] = current_masses[mask].sum()
            current_masses = new_masses
        # If masses were None, stay None (uniform weighting at all levels).

        if current_n <= 1:
            break

    return hierarchy


# ---------------------------------------------------------------------------
# Position coarsening and lifting
# ---------------------------------------------------------------------------

def coarsen_positions(
    C: torch.Tensor,
    X: torch.Tensor,
) -> torch.Tensor:
    """Project fine-level positions to coarse level via the coarsening matrix.

    Computes ``X_coarse = C @ X``. When C is mass-weighted, this yields the
    center-of-mass of each cluster, preserving SE(3) equivariance.

    Args:
        C: Coarsening matrix of shape ``(N_coarse, N_fine)``.
        X: Fine-level positions of shape ``(N_fine, D)`` where D is the
            spatial dimension (typically 3).

    Returns:
        Coarse-level positions of shape ``(N_coarse, D)``.
    """
    return C @ X


def lift_positions(
    C: torch.Tensor,
    X_coarse: torch.Tensor,
) -> torch.Tensor:
    """Lift coarse-level positions back to fine level using the pseudoinverse.

    Computes ``X_fine = C^+ @ X_coarse`` where C^+ is the Moore-Penrose
    pseudoinverse of C. For a mass-weighted coarsening matrix, this assigns
    each atom the position of its parent super-node (the center-of-mass of
    its cluster), which is the minimum-norm lifting that preserves SE(3)
    equivariance.

    This is equivalent to ``C^T @ (C @ C^T)^{-1} @ X_coarse`` when C has
    full row rank.

    Args:
        C: Coarsening matrix of shape ``(N_coarse, N_fine)``.
        X_coarse: Coarse-level positions of shape ``(N_coarse, D)``.

    Returns:
        Fine-level positions of shape ``(N_fine, D)``.
    """
    C_pinv = torch.linalg.pinv(C)  # (N_fine, N_coarse)
    return C_pinv @ X_coarse


# ---------------------------------------------------------------------------
# Batched helpers (PyTorch Geometric compatibility)
# ---------------------------------------------------------------------------

def build_coarsening_hierarchy_batched(
    adj_list: List[torch.Tensor],
    num_atoms_list: List[int],
    target_sizes_list: Optional[List[Optional[Sequence[int]]]] = None,
    atomic_masses_list: Optional[List[Optional[torch.Tensor]]] = None,
) -> List[List[CoarseningLevel]]:
    """Build coarsening hierarchies for a batch of molecules.

    This is a convenience wrapper that calls :func:`build_coarsening_hierarchy`
    for each molecule in the batch independently. In PyTorch Geometric, batched
    graphs are typically represented as a single large disconnected graph with a
    ``batch`` vector; this function expects the per-molecule adjacency matrices
    to be extracted beforehand.

    Args:
        adj_list: List of adjacency matrices, one per molecule.
        num_atoms_list: List of atom counts, one per molecule.
        target_sizes_list: Optional list of target-size sequences, one per
            molecule. If ``None``, automatic sizing is used for all molecules.
        atomic_masses_list: Optional list of mass tensors, one per molecule.
            If ``None``, uniform weights are used for all.

    Returns:
        List of coarsening hierarchies (each is a ``List[CoarseningLevel]``),
        one per molecule.
    """
    batch_size = len(adj_list)

    if target_sizes_list is None:
        target_sizes_list = [None] * batch_size
    if atomic_masses_list is None:
        atomic_masses_list = [None] * batch_size

    hierarchies: List[List[CoarseningLevel]] = []
    for i in range(batch_size):
        hierarchy = build_coarsening_hierarchy(
            adj=adj_list[i],
            num_atoms=num_atoms_list[i],
            target_sizes=target_sizes_list[i],
            atomic_masses=atomic_masses_list[i],
        )
        hierarchies.append(hierarchy)

    return hierarchies


def coarsen_positions_batched(
    C_list: List[torch.Tensor],
    X_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Coarsen positions for a batch of molecules.

    Args:
        C_list: List of coarsening matrices, one per molecule.
        X_list: List of position tensors, one per molecule.

    Returns:
        List of coarsened position tensors.
    """
    return [coarsen_positions(C, X) for C, X in zip(C_list, X_list)]


def lift_positions_batched(
    C_list: List[torch.Tensor],
    X_coarse_list: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Lift positions for a batch of molecules.

    Args:
        C_list: List of coarsening matrices, one per molecule.
        X_coarse_list: List of coarse-level position tensors.

    Returns:
        List of fine-level position tensors.
    """
    return [lift_positions(C, X) for C, X in zip(C_list, X_coarse_list)]
