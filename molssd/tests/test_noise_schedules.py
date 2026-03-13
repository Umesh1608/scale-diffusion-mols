"""Tests for noise schedules and resolution schedule (molssd.core.noise_schedules)."""
import torch
import pytest

from molssd.core.noise_schedules import (
    CosineSchedule,
    LinearSchedule,
    ResolutionSchedule,
    get_noise_schedule,
)


T = 100  # Use small T for fast tests


# -------------------------------------------------------------------
# Cosine schedule
# -------------------------------------------------------------------


class TestCosineSchedule:

    @pytest.fixture
    def sched(self):
        return CosineSchedule(T=T)

    def test_cosine_alpha_bar_range(self, sched):
        """alpha_bar is in (0, 1] for all t."""
        for t in range(T):
            ab = sched.alpha_bar(t).item()
            assert 0.0 < ab <= 1.0 + 1e-6, (
                f"alpha_bar({t}) = {ab} not in (0, 1]"
            )

    def test_cosine_alpha_bar_monotone(self, sched):
        """alpha_bar decreases monotonically with t."""
        vals = torch.tensor([sched.alpha_bar(t).item() for t in range(T)])
        diffs = vals[1:] - vals[:-1]
        assert (diffs <= 1e-7).all(), (
            f"alpha_bar must be non-increasing, but found increases: "
            f"max increase = {diffs.max().item()}"
        )

    def test_cosine_vp_condition(self, sched):
        """Variance-preserving condition: alpha_bar^2 + sigma^2 ~ 1."""
        for t in range(T):
            ab = sched.alpha_bar(t)
            s2 = sched.sigma_squared(t)
            total = ab ** 2 + s2
            assert torch.isclose(total, torch.tensor(1.0), atol=1e-4), (
                f"VP condition violated at t={t}: alpha_bar^2 + sigma^2 = {total.item()}"
            )


# -------------------------------------------------------------------
# Linear schedule
# -------------------------------------------------------------------


class TestLinearSchedule:

    @pytest.fixture
    def sched(self):
        return LinearSchedule(T=T)

    def test_linear_alpha_bar_range(self, sched):
        """alpha_bar is in (0, 1] for all t."""
        for t in range(T):
            ab = sched.alpha_bar(t).item()
            assert 0.0 < ab <= 1.0 + 1e-6, (
                f"alpha_bar({t}) = {ab} not in (0, 1]"
            )

    def test_linear_alpha_bar_monotone(self, sched):
        """alpha_bar decreases monotonically with t."""
        vals = torch.tensor([sched.alpha_bar(t).item() for t in range(T)])
        diffs = vals[1:] - vals[:-1]
        assert (diffs <= 1e-7).all(), "alpha_bar must be non-increasing"

    def test_linear_vp_condition(self, sched):
        """VP condition: alpha_bar^2 + sigma^2 ~ 1."""
        for t in range(T):
            ab = sched.alpha_bar(t)
            s2 = sched.sigma_squared(t)
            total = ab ** 2 + s2
            assert torch.isclose(total, torch.tensor(1.0), atol=1e-4), (
                f"VP condition violated at t={t}: {total.item()}"
            )


# -------------------------------------------------------------------
# SNR
# -------------------------------------------------------------------


class TestSNR:

    @pytest.mark.parametrize("ScheduleClass", [CosineSchedule, LinearSchedule])
    def test_snr_monotone(self, ScheduleClass):
        """SNR decreases (or stays same) with t."""
        sched = ScheduleClass(T=T)
        snr_vals = torch.tensor([sched.snr(t).item() for t in range(T)])
        diffs = snr_vals[1:] - snr_vals[:-1]
        assert (diffs <= 1e-5).all(), (
            f"SNR must be non-increasing, max increase = {diffs.max().item()}"
        )


# -------------------------------------------------------------------
# Resolution schedule
# -------------------------------------------------------------------


class TestResolutionSchedule:

    @pytest.fixture
    def res_sched(self):
        return ResolutionSchedule(
            T=T,
            num_levels=4,
            num_atoms_per_level=[12, 4, 2, 1],
            schedule_type="convex_decay",
            gamma=0.5,
        )

    def test_resolution_schedule_levels(self, res_sched):
        """resolution_level(t) is in the valid range [0, num_levels-1]."""
        for t in range(T):
            level = res_sched.resolution_level(t).item()
            assert 0 <= level < res_sched.num_levels, (
                f"Level {level} at t={t} out of range [0, {res_sched.num_levels - 1}]"
            )

    def test_resolution_schedule_monotone(self, res_sched):
        """Resolution levels increase (or stay same) with t.

        As t grows, the model should move to coarser resolutions.
        """
        levels = [res_sched.resolution_level(t).item() for t in range(T)]
        for i in range(1, len(levels)):
            assert levels[i] >= levels[i - 1], (
                f"Level went from {levels[i-1]} at t={i-1} to {levels[i]} at t={i}"
            )

    def test_resolution_schedule_change_detection(self, res_sched):
        """is_resolution_change correctly identifies transitions."""
        levels = [res_sched.resolution_level(t).item() for t in range(T)]

        # t=0 should never be a resolution change
        assert not res_sched.is_resolution_change(0).item()

        for t in range(1, T):
            expected = levels[t] != levels[t - 1]
            actual = res_sched.is_resolution_change(t).item()
            assert actual == expected, (
                f"Mismatch at t={t}: level went {levels[t-1]}->{levels[t]}, "
                f"change flag = {actual}"
            )


# -------------------------------------------------------------------
# Factory function
# -------------------------------------------------------------------


class TestFactory:

    def test_factory_cosine(self):
        """get_noise_schedule('cosine') returns a CosineSchedule."""
        sched = get_noise_schedule("cosine", T=50)
        assert isinstance(sched, CosineSchedule)
        assert sched.T == 50

    def test_factory_linear(self):
        """get_noise_schedule('linear') returns a LinearSchedule."""
        sched = get_noise_schedule("linear", T=50)
        assert isinstance(sched, LinearSchedule)
        assert sched.T == 50

    def test_factory_unknown_raises(self):
        """Unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown noise schedule"):
            get_noise_schedule("unknown_schedule")
