"""Noise schedules and resolution schedule for MolSSD.

Implements diffusion noise schedules (cosine, linear) and the resolution
schedule that maps diffusion timesteps to coarsening levels. The resolution
schedule follows the Scale Space Diffusion (SSD) framework, controlling when
the model operates at each level of the molecular hierarchy.

References:
    - Nichol & Dhariwal (2021): cosine schedule for improved DDPM.
    - Ho et al. (2020): linear beta schedule for DDPM.
    - SSD (arXiv:2603.08709): ConvexDecay resolution schedule with gamma.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class NoiseSchedule(ABC, nn.Module):
    """Abstract base class for diffusion noise schedules.

    All schedule values are precomputed at ``__init__`` and stored as
    registered buffers so they live on the same device as the model and
    are included in ``state_dict`` without requiring gradients.

    Parameters
    ----------
    T : int
        Total number of diffusion timesteps (default 1000). Timesteps are
        indexed ``0 .. T-1`` where ``t=0`` is the clean data and ``t=T-1``
        is (approximately) pure noise.
    """

    def __init__(self, T: int = 1000) -> None:
        super().__init__()
        self.T = T

    # -- helpers for indexing precomputed tensors --

    @staticmethod
    def _to_index(t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Convert *t* to a long tensor suitable for indexing buffers."""
        if isinstance(t, int):
            return torch.tensor([t], dtype=torch.long)
        return t.long()

    # -- public API (all accept scalar int or batched tensor) --

    @abstractmethod
    def alpha_bar(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Cumulative signal-scaling factor at timestep *t*.

        For the variance-preserving formulation this satisfies
        ``alpha_bar(t)^2 + sigma(t)^2 == 1``.
        """

    @abstractmethod
    def sigma(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Noise standard deviation at timestep *t*."""

    @abstractmethod
    def sigma_squared(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Noise variance ``sigma(t)^2`` at timestep *t*."""

    @abstractmethod
    def beta(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Per-step noise coefficient (forward-process variance) at *t*."""

    def snr(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Signal-to-noise ratio: ``alpha_bar(t)^2 / sigma(t)^2``.

        Useful for the Min-SNR-gamma loss weighting strategy.
        """
        ab = self.alpha_bar(t)
        s2 = self.sigma_squared(t)
        return ab ** 2 / s2.clamp(min=1e-20)


# ---------------------------------------------------------------------------
# Cosine schedule (Nichol & Dhariwal, 2021)
# ---------------------------------------------------------------------------


class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule from *Improved Denoising Diffusion Probabilistic
    Models* (Nichol & Dhariwal, 2021).

    .. math::

        \\bar{\\alpha}_t = \\cos^2\\!\\left(
            \\frac{t/T + s}{1 + s} \\cdot \\frac{\\pi}{2}
        \\right)

    with offset ``s = 0.008`` to prevent ``alpha_bar`` from being too small
    near ``t = 0``.  The schedule is **variance-preserving**:
    ``sigma_t = sqrt(1 - alpha_bar_t^2)``.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    s : float
        Small offset to avoid singularity near ``t = 0``.
    """

    def __init__(self, T: int = 1000, s: float = 0.008) -> None:
        super().__init__(T=T)
        self.s = s

        # Precompute alpha_bar for t = 0 .. T-1
        steps = torch.arange(T, dtype=torch.float64)
        f_t = torch.cos(((steps / T) + s) / (1.0 + s) * (math.pi / 2.0)) ** 2

        # Normalise so that alpha_bar(0) == 1 (approximately)
        alpha_bar_vals = f_t / f_t[0]

        # Clamp to avoid numerical issues
        alpha_bar_vals = alpha_bar_vals.clamp(min=1e-8, max=1.0)

        # Derive sigma, sigma_squared from VP condition
        sigma_sq_vals = 1.0 - alpha_bar_vals ** 2
        sigma_sq_vals = sigma_sq_vals.clamp(min=0.0)
        sigma_vals = torch.sqrt(sigma_sq_vals)

        # Derive betas:  alpha_bar_t = prod_{i=1}^{t} (1 - beta_i)
        # => alpha_t = alpha_bar_t / alpha_bar_{t-1},  beta_t = 1 - alpha_t
        alpha_bar_prev = torch.cat([torch.tensor([1.0], dtype=torch.float64), alpha_bar_vals[:-1]])
        alpha_t = alpha_bar_vals / alpha_bar_prev
        beta_vals = (1.0 - alpha_t).clamp(min=0.0, max=0.999)

        # Register as float32 buffers
        self.register_buffer("_alpha_bar", alpha_bar_vals.float())
        self.register_buffer("_sigma", sigma_vals.float())
        self.register_buffer("_sigma_sq", sigma_sq_vals.float())
        self.register_buffer("_beta", beta_vals.float())

    # -- interface implementation --

    def alpha_bar(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._alpha_bar[idx]

    def sigma(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._sigma[idx]

    def sigma_squared(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._sigma_sq[idx]

    def beta(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._beta[idx]


# ---------------------------------------------------------------------------
# Linear schedule (Ho et al., 2020)
# ---------------------------------------------------------------------------


class LinearSchedule(NoiseSchedule):
    r"""Linear beta schedule from *Denoising Diffusion Probabilistic Models*
    (Ho et al., 2020).

    ``beta_t`` increases linearly from ``beta_min`` to ``beta_max`` over ``T``
    steps:

    .. math::

        \beta_t = \beta_{\min} + \frac{t}{T-1}(\beta_{\max} - \beta_{\min})

    Then ``alpha_t = 1 - beta_t`` and
    ``\bar{\alpha}_t = \prod_{i=0}^{t} \alpha_i``.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps.
    beta_min : float
        Starting (smallest) beta value.
    beta_max : float
        Ending (largest) beta value.
    """

    def __init__(
        self,
        T: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
    ) -> None:
        super().__init__(T=T)
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Precompute betas linearly spaced
        beta_vals = torch.linspace(beta_min, beta_max, T, dtype=torch.float64)

        alpha_t = 1.0 - beta_vals
        alpha_bar_vals = torch.cumprod(alpha_t, dim=0)

        # Clamp for safety
        alpha_bar_vals = alpha_bar_vals.clamp(min=1e-8, max=1.0)

        # Derive sigma from VP condition: sigma^2 = 1 - alpha_bar^2
        sigma_sq_vals = (1.0 - alpha_bar_vals ** 2).clamp(min=0.0)
        sigma_vals = torch.sqrt(sigma_sq_vals)

        self.register_buffer("_alpha_bar", alpha_bar_vals.float())
        self.register_buffer("_sigma", sigma_vals.float())
        self.register_buffer("_sigma_sq", sigma_sq_vals.float())
        self.register_buffer("_beta", beta_vals.float())

    # -- interface implementation --

    def alpha_bar(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._alpha_bar[idx]

    def sigma(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._sigma[idx]

    def sigma_squared(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._sigma_sq[idx]

    def beta(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        idx = self._to_index(t)
        return self._beta[idx]


# ---------------------------------------------------------------------------
# Resolution schedule (SSD paper)
# ---------------------------------------------------------------------------


class ResolutionSchedule(nn.Module):
    """Maps diffusion timesteps to molecular resolution levels.

    In the SSD framework the forward process operates at progressively coarser
    resolutions as *t* increases from 0 to *T*.  This class precomputes the
    mapping ``r(t)`` and identifies the exact timesteps at which a resolution
    change occurs (where the non-isotropic posterior must be used instead of
    the standard isotropic Gaussian posterior).

    Parameters
    ----------
    T : int
        Total diffusion timesteps.
    num_levels : int
        Number of distinct resolution levels ``0 .. num_levels-1``, where
        level 0 is full atomic resolution and ``num_levels-1`` is the coarsest
        (e.g., single centroid).
    num_atoms_per_level : List[int]
        Number of (super-)atoms at each resolution level, e.g.
        ``[N, N_1, N_2, ..., 1]``.  Must have length ``num_levels``.
    schedule_type : str
        One of ``'convex_decay'`` (default from SSD paper),
        ``'equal'``, or ``'sigmoid'``.
    gamma : float
        Exponent for convex-decay schedule.  ``gamma=0.5`` (square-root
        decay) is the default recommended by the SSD paper.
    sigmoid_k : float
        Steepness of the sigmoid transition (only used when
        ``schedule_type='sigmoid'``).
    """

    SCHEDULE_TYPES = ("convex_decay", "equal", "sigmoid")

    def __init__(
        self,
        T: int = 1000,
        num_levels: int = 4,
        num_atoms_per_level: Optional[List[int]] = None,
        schedule_type: str = "convex_decay",
        gamma: float = 0.5,
        sigmoid_k: float = 10.0,
    ) -> None:
        super().__init__()
        if schedule_type not in self.SCHEDULE_TYPES:
            raise ValueError(
                f"Unknown schedule_type '{schedule_type}'. "
                f"Must be one of {self.SCHEDULE_TYPES}."
            )
        if num_atoms_per_level is not None and len(num_atoms_per_level) != num_levels:
            raise ValueError(
                f"len(num_atoms_per_level) = {len(num_atoms_per_level)} "
                f"must equal num_levels = {num_levels}."
            )

        self.T = T
        self.num_levels = num_levels
        self.schedule_type = schedule_type
        self.gamma = gamma
        self.sigmoid_k = sigmoid_k

        # If no atom counts given, assume uniform spacing of fractional sizes
        if num_atoms_per_level is not None:
            N = num_atoms_per_level[0]  # full resolution atom count
            fractions = torch.tensor(
                [n / N for n in num_atoms_per_level], dtype=torch.float32
            )
        else:
            # Evenly spaced fractional sizes from 1.0 down to 1/num_levels
            fractions = torch.linspace(1.0, 1.0 / num_levels, num_levels)

        self.register_buffer("_fractions", fractions)

        # Precompute info_content and resolution level for every timestep
        info_vals = torch.zeros(T, dtype=torch.float32)
        level_vals = torch.zeros(T, dtype=torch.long)

        for t in range(T):
            info_t = self._compute_info(t, T, schedule_type, gamma, sigmoid_k)
            info_vals[t] = info_t
            # Map info -> resolution level:  find the level whose fractional
            # size is closest to info_t
            level_vals[t] = (fractions - info_t).abs().argmin().long()

        self.register_buffer("_info", info_vals)
        self.register_buffer("_levels", level_vals)

        # Precompute resolution-change flags
        change_flags = torch.zeros(T, dtype=torch.bool)
        for t in range(1, T):
            if level_vals[t] != level_vals[t - 1]:
                change_flags[t] = True
        self.register_buffer("_change_flags", change_flags)

    # -- internal helpers --

    @staticmethod
    def _compute_info(
        t: int,
        T: int,
        schedule_type: str,
        gamma: float,
        sigmoid_k: float,
    ) -> float:
        """Compute the molecular information content ``Info_mol(t)`` in [0, 1].

        * ``convex_decay``:  ``(1 - t/T)^gamma``
        * ``equal``:         piecewise-constant at equal intervals
        * ``sigmoid``:       ``1 - sigmoid(k * (t/T - 0.5))``  (rescaled to [0, 1])
        """
        frac = t / T  # in [0, 1)

        if schedule_type == "convex_decay":
            return (1.0 - frac) ** gamma

        if schedule_type == "equal":
            # Still monotonically decreasing; just return linear decay
            return 1.0 - frac

        if schedule_type == "sigmoid":
            # S-shaped drop from ~1 at t=0 to ~0 at t=T
            raw = 1.0 / (1.0 + math.exp(sigmoid_k * (frac - 0.5)))
            # Rescale so that f(0)=1 and f(1)=0 exactly
            f0 = 1.0 / (1.0 + math.exp(sigmoid_k * (0.0 - 0.5)))
            f1 = 1.0 / (1.0 + math.exp(sigmoid_k * (1.0 - 0.5)))
            return (raw - f1) / (f0 - f1)

        raise ValueError(f"Unknown schedule_type '{schedule_type}'")

    # -- public API --

    def resolution_level(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Return the resolution level ``r(t)`` for the given timestep(s).

        Parameters
        ----------
        t : int or Tensor of long
            Diffusion timesteps (0-indexed).

        Returns
        -------
        Tensor of long
            Resolution level(s), in ``0 .. num_levels - 1``.
        """
        if isinstance(t, int):
            idx = torch.tensor([t], dtype=torch.long)
        else:
            idx = t.long()
        return self._levels[idx]

    def is_resolution_change(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Return ``True`` where ``r(t) != r(t-1)``.

        At these timesteps the non-isotropic posterior should be used.

        Parameters
        ----------
        t : int or Tensor of long
            Diffusion timesteps.  For ``t=0`` the result is always ``False``.

        Returns
        -------
        Tensor of bool
        """
        if isinstance(t, int):
            idx = torch.tensor([t], dtype=torch.long)
        else:
            idx = t.long()
        return self._change_flags[idx]

    def get_resolution_change_steps(self) -> List[int]:
        """Return a sorted list of all timesteps where the resolution changes.

        These are the timesteps ``t`` for which ``r(t) != r(t-1)``.
        """
        return self._change_flags.nonzero(as_tuple=False).squeeze(-1).tolist()

    def info_content(self, t: Union[int, torch.Tensor]) -> torch.Tensor:
        """Return the molecular information content ``Info_mol(t)`` at *t*.

        Values are in ``[0, 1]`` with 1 meaning full information (clean data)
        and 0 meaning no information (pure noise / maximally coarse).

        Parameters
        ----------
        t : int or Tensor of long
            Diffusion timesteps.

        Returns
        -------
        Tensor of float
        """
        if isinstance(t, int):
            idx = torch.tensor([t], dtype=torch.long)
        else:
            idx = t.long()
        return self._info[idx]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def get_noise_schedule(name: str, T: int = 1000, **kwargs) -> NoiseSchedule:
    """Create a noise schedule by name.

    Parameters
    ----------
    name : str
        ``'cosine'`` or ``'linear'``.
    T : int
        Number of diffusion timesteps.
    **kwargs
        Additional keyword arguments forwarded to the schedule constructor
        (e.g., ``s`` for cosine, ``beta_min`` / ``beta_max`` for linear).

    Returns
    -------
    NoiseSchedule
        Instantiated noise schedule.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    name_lower = name.lower()
    if name_lower == "cosine":
        return CosineSchedule(T=T, **kwargs)
    if name_lower == "linear":
        return LinearSchedule(T=T, **kwargs)
    raise ValueError(
        f"Unknown noise schedule '{name}'. Choose from 'cosine' or 'linear'."
    )
