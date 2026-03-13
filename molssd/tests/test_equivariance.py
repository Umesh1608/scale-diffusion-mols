"""Tests for SE(3) equivariance of the MolSSD pipeline."""
import torch
import pytest

from molssd.core.coarsening import (
    build_coarsening_hierarchy,
    coarsen_positions,
)
from molssd.core.degradation import DegradationOperator
from molssd.core.diffusion import MolSSDDiffusion
from molssd.core.noise_schedules import CosineSchedule, ResolutionSchedule


T = 100  # Fast tests


@pytest.fixture
def equivariance_setup(medium_molecule, seed):
    """Build hierarchy and operators for equivariance tests."""
    adj = medium_molecule["adj"]
    n = medium_molecule["num_atoms"]
    masses = medium_molecule["atomic_masses"]
    positions = medium_molecule["positions"]
    atom_types = medium_molecule["atom_types"]

    hierarchy = build_coarsening_hierarchy(adj, n, atomic_masses=masses)
    noise_sched = CosineSchedule(T=T)

    num_levels = len(hierarchy) + 1
    num_atoms_per_level = [n] + [level.num_nodes for level in hierarchy]

    res_sched = ResolutionSchedule(
        T=T,
        num_levels=num_levels,
        num_atoms_per_level=num_atoms_per_level,
        schedule_type="convex_decay",
        gamma=0.5,
    )

    deg_op = DegradationOperator(hierarchy, noise_sched, res_sched)

    diffusion = MolSSDDiffusion(
        noise_schedule=noise_sched,
        resolution_schedule=res_sched,
        num_atom_types=2,
    )

    return {
        "hierarchy": hierarchy,
        "deg_op": deg_op,
        "diffusion": diffusion,
        "noise_sched": noise_sched,
        "positions": positions,
        "atom_types": atom_types,
        "num_atoms": n,
    }


# -------------------------------------------------------------------
# Coarsening equivariance
# -------------------------------------------------------------------


class TestCoarseningEquivariance:

    def test_coarsening_rotation_equivariance(
        self, equivariance_setup, random_rotation
    ):
        """C @ (R @ X) == R @ (C @ X) -- coarsening commutes with rotation."""
        hierarchy = equivariance_setup["hierarchy"]
        X = equivariance_setup["positions"]
        R = random_rotation

        for level in hierarchy:
            C = level.coarsening_matrix

            # Rotate then coarsen
            X_rot = (R @ X.T).T
            coarse_of_rot = coarsen_positions(C, X_rot)

            # Coarsen then rotate
            coarse_X = coarsen_positions(C, X)
            rot_of_coarse = (R @ coarse_X.T).T

            assert torch.allclose(coarse_of_rot, rot_of_coarse, atol=1e-5), (
                f"Rotation equivariance broken at level with "
                f"{level.num_nodes} nodes: "
                f"max diff = {(coarse_of_rot - rot_of_coarse).abs().max().item()}"
            )

            # Use coarsened positions for next level
            X = coarse_X

    def test_coarsening_translation_equivariance(
        self, equivariance_setup, random_translation
    ):
        """C @ (X + t) == (C @ X) + t for translation t.

        This holds because each row of C sums to 1, so
        C @ (X + t1^T) = C@X + (C @ 1) t^T = C@X + 1 t^T = (C@X) + t.
        """
        hierarchy = equivariance_setup["hierarchy"]
        X = equivariance_setup["positions"]
        t_vec = random_translation  # shape (3,)

        for level in hierarchy:
            C = level.coarsening_matrix

            # Translate then coarsen
            X_translated = X + t_vec.unsqueeze(0)  # broadcast (N, 3)
            coarse_of_translated = coarsen_positions(C, X_translated)

            # Coarsen then translate
            coarse_X = coarsen_positions(C, X)
            translated_of_coarse = coarse_X + t_vec.unsqueeze(0)

            assert torch.allclose(
                coarse_of_translated, translated_of_coarse, atol=1e-5
            ), (
                f"Translation equivariance broken at level with "
                f"{level.num_nodes} nodes: "
                f"max diff = "
                f"{(coarse_of_translated - translated_of_coarse).abs().max().item()}"
            )

            # Use coarsened positions for next level
            X = coarse_X


# -------------------------------------------------------------------
# Forward process equivariance
# -------------------------------------------------------------------


class TestForwardProcessEquivariance:

    def test_forward_process_rotation_equivariance(
        self, equivariance_setup, random_rotation, seed
    ):
        """Applying rotation before/after forward process gives the same
        result, up to noise matching.

        The deterministic part of the forward process is:
            signal = alpha_bar_t * C^{r(t)} @ x0

        Since C commutes with rotation R:
            C @ (R @ X) == R @ (C @ X)

        The signal of the rotated input should equal the rotation of
        the original signal.
        """
        d = equivariance_setup
        X = d["positions"]
        R = random_rotation
        deg_op = d["deg_op"]
        noise_sched = d["noise_sched"]

        # Test at several timesteps
        test_timesteps = [0, T // 4, T // 2, 3 * T // 4, T - 1]

        for t in test_timesteps:
            # Signal from original positions
            signal_original = deg_op.apply_M1t(X, t)

            # Signal from rotated positions
            X_rot = (R @ X.T).T
            signal_rotated = deg_op.apply_M1t(X_rot, t)

            # Rotate the original signal
            rotated_signal_original = (R @ signal_original.T).T

            assert torch.allclose(
                signal_rotated, rotated_signal_original, atol=1e-4
            ), (
                f"Forward process rotation equivariance broken at t={t}: "
                f"max diff = "
                f"{(signal_rotated - rotated_signal_original).abs().max().item()}"
            )
