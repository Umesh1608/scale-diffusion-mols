"""Tests for the full diffusion process (molssd.core.diffusion)."""
import torch
import pytest

from molssd.core.coarsening import build_coarsening_hierarchy
from molssd.core.degradation import DegradationOperator
from molssd.core.diffusion import MolSSDDiffusion
from molssd.core.noise_schedules import CosineSchedule, ResolutionSchedule


T = 100  # Fast tests


@pytest.fixture
def diffusion_setup(medium_molecule, seed):
    """Build a MolSSDDiffusion instance from the medium molecule fixture."""
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

    diffusion = MolSSDDiffusion(
        noise_schedule=noise_sched,
        resolution_schedule=res_sched,
        num_atom_types=2,  # 0=H, 1=C in the fixture
    )

    deg_op = DegradationOperator(hierarchy, noise_sched, res_sched)

    return {
        "diffusion": diffusion,
        "deg_op": deg_op,
        "noise_sched": noise_sched,
        "res_sched": res_sched,
        "hierarchy": hierarchy,
        "positions": positions,
        "atom_types": atom_types,
        "num_atoms": n,
    }


# -------------------------------------------------------------------
# Forward process
# -------------------------------------------------------------------


class TestForwardProcess:

    def test_forward_process_shape(self, diffusion_setup):
        """Output shapes are correct for the forward process."""
        d = diffusion_setup
        x0 = d["positions"]
        atom_types = d["atom_types"]
        hierarchy = d["hierarchy"]
        deg_op = d["deg_op"]

        for t in [0, T // 2, T - 1]:
            x_t, eps, coarsened_types = d["diffusion"].forward_process(
                x0, atom_types, t, hierarchy
            )
            r_t = deg_op.get_resolution_at(t)
            C_rt = deg_op._get_composed_C(r_t)
            expected_n = C_rt.shape[0]

            assert x_t.shape == (expected_n, 3), (
                f"x_t shape mismatch at t={t}: {x_t.shape} vs ({expected_n}, 3)"
            )
            assert eps.shape == (expected_n, 3), (
                f"eps shape mismatch at t={t}"
            )
            assert coarsened_types.shape == (expected_n,), (
                f"coarsened_types shape mismatch at t={t}"
            )

    def test_forward_process_at_t0(self, diffusion_setup):
        """At t=0, x_t is close to alpha_bar(0) * x0 (low noise)."""
        d = diffusion_setup
        x0 = d["positions"]
        atom_types = d["atom_types"]
        hierarchy = d["hierarchy"]
        noise_sched = d["noise_sched"]

        x_t, eps, _ = d["diffusion"].forward_process(x0, atom_types, 0, hierarchy)

        # At t=0, x_t = alpha_bar(0) * x0 + sigma(0) * eps
        # sigma(0) is very small, so x_t ~ alpha_bar(0) * x0
        ab_0 = noise_sched.alpha_bar(0)
        sigma_0 = noise_sched.sigma(0)
        signal = ab_0 * x0

        # The noise contribution is sigma_0 * eps
        noise_contribution = sigma_0 * eps
        # Verify the decomposition
        reconstructed = signal + noise_contribution
        assert torch.allclose(x_t, reconstructed, atol=1e-5), (
            "x_t should equal alpha_bar(0)*x0 + sigma(0)*eps at t=0"
        )

        # Also verify that sigma_0 is small so signal dominates
        assert sigma_0.item() < 0.1, (
            f"sigma(0) should be small, got {sigma_0.item()}"
        )


# -------------------------------------------------------------------
# x0 prediction roundtrip
# -------------------------------------------------------------------


class TestPredictX0:

    def test_predict_x0_roundtrip(self, diffusion_setup):
        """Forward then predict_x0 recovers ~x0 when epsilon is known."""
        d = diffusion_setup
        x0 = d["positions"]
        atom_types = d["atom_types"]
        hierarchy = d["hierarchy"]
        diffusion = d["diffusion"]
        deg_op = d["deg_op"]

        # Use t=0 where resolution is full (identity coarsening)
        t = 0
        x_t, eps, _ = diffusion.forward_process(x0, atom_types, t, hierarchy)

        x0_hat = diffusion.predict_x0(x_t, t, eps, deg_op)

        assert torch.allclose(x0_hat, x0, atol=1e-3), (
            f"predict_x0 should recover x0 at t=0 with known eps: "
            f"max diff = {(x0_hat - x0).abs().max().item()}"
        )


# -------------------------------------------------------------------
# Atom type coarsening
# -------------------------------------------------------------------


class TestCoarsenAtomTypes:

    def test_coarsen_atom_types_majority(self, diffusion_setup):
        """Majority vote assigns the most common type in each cluster."""
        diffusion = diffusion_setup["diffusion"]

        # 6 atoms: [0, 0, 0, 1, 1, 0]
        atom_types = torch.tensor([0, 0, 0, 1, 1, 0])
        # 2 clusters: cluster 0 = atoms {0,1,2}, cluster 1 = atoms {3,4,5}
        cluster_assignment = torch.tensor([0, 0, 0, 1, 1, 1])

        coarsened = diffusion.coarsen_atom_types(atom_types, cluster_assignment, 2)

        # Cluster 0: types [0, 0, 0] -> majority = 0
        assert coarsened[0].item() == 0
        # Cluster 1: types [1, 1, 0] -> majority = 1
        assert coarsened[1].item() == 1


# -------------------------------------------------------------------
# Isotropic posterior at same resolution
# -------------------------------------------------------------------


class TestPosterior:

    def test_isotropic_posterior_at_same_resolution(self, diffusion_setup):
        """Returns is_isotropic=True when no resolution change occurs."""
        d = diffusion_setup
        x0 = d["positions"]
        deg_op = d["deg_op"]
        diffusion = d["diffusion"]

        # Find a step with no resolution change (t >= 1)
        no_change_t = None
        for t in range(1, T):
            if not deg_op.is_resolution_change(t):
                no_change_t = t
                break

        if no_change_t is None:
            pytest.skip("No non-resolution-change timestep found (t >= 1)")

        # Build x_t at that timestep
        x_t_signal = deg_op.apply_M1t(x0, no_change_t)
        sigma_t = d["noise_sched"].sigma(no_change_t)
        x_t = x_t_signal + sigma_t * torch.randn_like(x_t_signal)

        posterior = diffusion.compute_posterior_params(x_t, x0, no_change_t, deg_op)

        assert posterior["is_isotropic"] is True, (
            "Posterior should be isotropic when resolution does not change"
        )
        assert "mu" in posterior
        assert "sigma" in posterior
