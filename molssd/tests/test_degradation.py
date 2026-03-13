"""Tests for the degradation operator (molssd.core.degradation)."""
import torch
import pytest

from molssd.core.coarsening import build_coarsening_hierarchy
from molssd.core.degradation import DegradationOperator, apply_Mt_vjp
from molssd.core.noise_schedules import CosineSchedule, ResolutionSchedule


T = 100  # Fast tests


@pytest.fixture
def setup_degradation(medium_molecule, seed):
    """Build a DegradationOperator from the medium molecule fixture."""
    adj = medium_molecule["adj"]
    n = medium_molecule["num_atoms"]
    masses = medium_molecule["atomic_masses"]
    positions = medium_molecule["positions"]

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

    return {
        "deg_op": deg_op,
        "noise_sched": noise_sched,
        "res_sched": res_sched,
        "positions": positions,
        "hierarchy": hierarchy,
        "num_atoms": n,
    }


class TestApplyM1t:

    def test_apply_M1t_at_t0(self, setup_degradation):
        """At t=0 (full resolution), M_{1:0} x ~ alpha_bar(0) * x.

        At t=0 the resolution level is 0, so C^{r(0)} = I,
        and M_{1:0} x = alpha_bar(0) * I @ x = alpha_bar(0) * x.
        """
        deg_op = setup_degradation["deg_op"]
        noise_sched = setup_degradation["noise_sched"]
        x0 = setup_degradation["positions"]

        result = deg_op.apply_M1t(x0, t=0)
        alpha_bar_0 = noise_sched.alpha_bar(0)
        expected = alpha_bar_0 * x0

        assert torch.allclose(result, expected, atol=1e-5), (
            "M_{1:0} x should equal alpha_bar(0) * x at t=0"
        )

    def test_apply_M1t_coarsens(self, setup_degradation):
        """At a coarse timestep, output has fewer rows than input."""
        deg_op = setup_degradation["deg_op"]
        x0 = setup_degradation["positions"]
        n = setup_degradation["num_atoms"]

        # Find a timestep where resolution level > 0
        coarse_t = None
        for t in range(T):
            if deg_op.get_resolution_at(t) > 0:
                coarse_t = t
                break

        if coarse_t is None:
            pytest.skip("No coarse timestep found in schedule")

        result = deg_op.apply_M1t(x0, t=coarse_t)
        assert result.shape[0] < n, (
            f"At coarse timestep {coarse_t}, expected fewer rows than {n}, "
            f"got {result.shape[0]}"
        )
        assert result.shape[1] == 3


class TestApplyMt:

    def test_apply_Mt_no_resolution_change(self, setup_degradation):
        """When r(t)==r(t-1), M_t is just a_t * I (scaling only)."""
        deg_op = setup_degradation["deg_op"]
        x0 = setup_degradation["positions"]

        # Find a timestep with no resolution change (skip t=0)
        no_change_t = None
        for t in range(1, T):
            if not deg_op.is_resolution_change(t):
                no_change_t = t
                break

        if no_change_t is None:
            pytest.skip("No non-resolution-change timestep found")

        # At this timestep, x lives at the current resolution.
        # We'll use M_{1:t-1} x0 as the input at the correct resolution.
        x_prev = deg_op.apply_M1t(x0, no_change_t - 1)

        result = deg_op.apply_Mt(x_prev, no_change_t)

        # Expected: a_t * x_prev
        ab_t = deg_op.noise_schedule.alpha_bar(no_change_t)
        ab_prev = deg_op.noise_schedule.alpha_bar(no_change_t - 1)
        a_t = ab_t / ab_prev
        expected = a_t * x_prev

        assert torch.allclose(result, expected, atol=1e-5), (
            "M_t should be a_t * I when resolution does not change"
        )


class TestMtTranspose:

    def test_apply_Mt_transpose_vjp_vs_explicit(self, setup_degradation):
        """VJP-based transpose matches explicit transpose computation."""
        deg_op = setup_degradation["deg_op"]
        x0 = setup_degradation["positions"]

        # Find a resolution-changing timestep for a non-trivial test
        change_t = None
        for t in range(1, T):
            if deg_op.is_resolution_change(t):
                change_t = t
                break

        if change_t is None:
            # Fall back to a non-changing step
            change_t = 1

        # Create a v vector in the output space of M_t
        r_t = deg_op.get_resolution_at(change_t)
        n_out = deg_op._get_composed_C(r_t).shape[0]
        v = torch.randn(n_out, 3)

        vjp_result = deg_op.apply_Mt_transpose(v, change_t)
        explicit_result = deg_op.apply_Mt_transpose_explicit(v, change_t)

        assert torch.allclose(vjp_result, explicit_result, atol=1e-4), (
            f"VJP transpose and explicit transpose differ: "
            f"max diff = {(vjp_result - explicit_result).abs().max().item()}"
        )


class TestMtTMt:

    def test_MtT_Mt_symmetric(self, setup_degradation):
        """M_t^T M_t is symmetric."""
        deg_op = setup_degradation["deg_op"]

        # Use a resolution-changing step if available; otherwise any step
        test_t = 1
        for t in range(1, T):
            if deg_op.is_resolution_change(t):
                test_t = t
                break

        MtTMt = deg_op.compute_MtT_Mt(test_t)
        assert torch.allclose(MtTMt, MtTMt.T, atol=1e-5), (
            "M_t^T M_t must be symmetric"
        )

    def test_MtT_Mt_matvec_matches_full(self, setup_degradation):
        """Matvec version matches full matrix multiply."""
        deg_op = setup_degradation["deg_op"]

        test_t = 1
        for t in range(1, T):
            if deg_op.is_resolution_change(t):
                test_t = t
                break

        MtTMt = deg_op.compute_MtT_Mt(test_t)

        # Create a vector in the input space
        n_input = MtTMt.shape[0]
        v = torch.randn(n_input, 3)

        matvec_result = deg_op.compute_MtT_Mt_matvec(v, test_t)
        full_result = MtTMt @ v

        assert torch.allclose(matvec_result, full_result, atol=1e-4), (
            f"Matvec and full matrix multiply differ: "
            f"max diff = {(matvec_result - full_result).abs().max().item()}"
        )
