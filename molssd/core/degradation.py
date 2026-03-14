"""Linear degradation operator M_t for the MolSSD forward process.

This module implements the per-step degradation operator M_t and its composed
version M_{1:t}, which form the core of how MolSSD combines spectral coarsening
with diffusion.

Mathematical framework
----------------------
Per-step degradation:

    M_t = a_t * C^{r(t)} @ pinv(C^{r(t-1)})

where:
- a_t is a scalar scaling factor derived from the noise schedule
  (a_t = alpha_bar(t) / alpha_bar(t-1))
- r(t) maps timestep t to resolution level via the ResolutionSchedule
- C^{(k)} is the coarsening matrix at level k
- pinv(C^{(k)}) is its Moore-Penrose pseudoinverse

Key simplification (from SSD paper): the composed operator telescopes:

    M_{1:t} = alpha_bar_t * C^{r(t)}

so we just need the coarsening matrix at the current resolution level, scaled
by the cumulative signal factor.

When r(t) == r(t-1) (no resolution change): M_t = a_t * I  (standard DDPM
scaling at the current resolution).

References:
    - SSD paper (arXiv:2603.08709): Scale-space diffusion framework
"""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

from molssd.core.coarsening import CoarseningLevel
from molssd.core.noise_schedules import NoiseSchedule, ResolutionSchedule


# ---------------------------------------------------------------------------
# Standalone VJP helper
# ---------------------------------------------------------------------------


def apply_Mt_vjp(
    Mt_fn: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Compute M_t^T @ v via vector-Jacobian product.

    Uses ``torch.autograd.grad`` to evaluate the transpose action without
    ever materializing M_t explicitly.  The identity exploited is:

        M_t^T @ v = grad_x (v^T @ M_t @ x)

    This works because M_t is a linear operator in x, so the gradient of the
    scalar v^T M_t x with respect to x is exactly M_t^T v.

    Args:
        Mt_fn: A callable that maps ``x -> M_t @ x``.  Must be differentiable
            with respect to its input through PyTorch autograd.
        v: Vector to left-multiply by M_t^T, shape matching the output of
            ``Mt_fn``.
        x: Input vector at which to evaluate the VJP.  Its value does not
            affect the result (since M_t is linear), but it determines the
            input shape and device.

    Returns:
        M_t^T @ v, same shape as *x*.
    """
    x_req = x.detach().requires_grad_(True)
    Mx = Mt_fn(x_req)
    # Scalar: v^T @ Mx (element-wise product then sum, handles arbitrary shapes)
    loss = (v * Mx).sum()
    (Mt_T_v,) = torch.autograd.grad(loss, x_req)
    return Mt_T_v


# ---------------------------------------------------------------------------
# DegradationOperator
# ---------------------------------------------------------------------------


class DegradationOperator(nn.Module):
    """Degradation operator M_t for the MolSSD forward process.

    Precomputes and caches the coarsening matrices and their pseudoinverses
    at each level so that applying M_t and M_{1:t} is efficient at runtime.

    The operator acts on 3D positions of shape ``(N, 3)`` but generalises to
    ``(N, D)`` for arbitrary feature dimension D.

    Parameters
    ----------
    coarsening_hierarchy : List[CoarseningLevel]
        Multi-level coarsening hierarchy produced by
        :func:`~molssd.core.coarsening.build_coarsening_hierarchy`.
        Level 0 maps the full atomic graph to the first set of super-nodes,
        level 1 maps those super-nodes to the next coarser level, etc.
    noise_schedule : NoiseSchedule
        Provides ``alpha_bar(t)`` and ``beta(t)`` for the diffusion process.
    resolution_schedule : ResolutionSchedule
        Provides ``resolution_level(t)`` and ``is_resolution_change(t)``
        mapping timesteps to coarsening levels.
    """

    def __init__(
        self,
        coarsening_hierarchy: List[CoarseningLevel],
        noise_schedule: NoiseSchedule,
        resolution_schedule: ResolutionSchedule,
    ) -> None:
        super().__init__()
        self.noise_schedule = noise_schedule
        self.resolution_schedule = resolution_schedule
        self.num_levels = len(coarsening_hierarchy) + 1  # +1 for level 0 (atoms)

        # Build *composed* coarsening matrices from level 0 (atoms) to level k.
        #   C_composed[0] = I                          (level 0: identity)
        #   C_composed[1] = C^{(1)}                    (atoms -> super-atoms)
        #   C_composed[2] = C^{(2)} @ C^{(1)}          (atoms -> fragments)
        #   ...
        # These are what M_{1:t} uses: M_{1:t} = alpha_bar_t * C_composed[r(t)]
        #
        # We also precompute pseudoinverses for each composed matrix.

        composed_matrices: List[torch.Tensor] = []
        composed_pinvs: List[torch.Tensor] = []

        # Level 0: identity (full atomic resolution)
        if len(coarsening_hierarchy) > 0:
            n_atoms = coarsening_hierarchy[0].coarsening_matrix.shape[1]
        else:
            # Degenerate case: no coarsening hierarchy at all
            n_atoms = 1

        I = torch.eye(n_atoms, dtype=torch.float32)
        composed_matrices.append(I)
        composed_pinvs.append(I)  # pinv(I) = I

        # Build composed matrices iteratively
        current_composed = I
        for level in coarsening_hierarchy:
            C_k = level.coarsening_matrix  # (N_k, N_{k-1})
            current_composed = C_k @ current_composed  # (N_k, N_atoms)
            composed_matrices.append(current_composed)
            composed_pinvs.append(torch.linalg.pinv(current_composed))

        # Register as buffers (non-trainable, move with model device)
        for k, (C, C_pinv) in enumerate(zip(composed_matrices, composed_pinvs)):
            self.register_buffer(f"_C_composed_{k}", C)
            self.register_buffer(f"_C_pinv_{k}", C_pinv)

        self._num_composed = len(composed_matrices)

    # -- internal helpers --

    def _get_composed_C(self, level: int) -> torch.Tensor:
        """Get the composed coarsening matrix C_composed[level]."""
        return getattr(self, f"_C_composed_{level}")

    def _get_composed_pinv(self, level: int) -> torch.Tensor:
        """Get the pseudoinverse of C_composed[level]."""
        return getattr(self, f"_C_pinv_{level}")

    def _scalar_t(self, t: Union[int, torch.Tensor]) -> int:
        """Ensure t is a plain Python int for indexing."""
        if isinstance(t, torch.Tensor):
            return t.item()
        return int(t)

    def _get_alpha_t(self, t: int) -> torch.Tensor:
        """Per-step signal scaling: a_t = alpha_bar(t) / alpha_bar(t-1).

        At t=0 we define a_0 = alpha_bar(0) (no previous step).
        """
        ab_t = self.noise_schedule.alpha_bar(t)
        if t == 0:
            return ab_t
        ab_prev = self.noise_schedule.alpha_bar(t - 1)
        return ab_t / ab_prev.clamp(min=1e-12)

    # -- public API --

    def get_resolution_at(self, t: Union[int, torch.Tensor]) -> int:
        """Get the resolution level at timestep t.

        The returned level is clamped to the maximum available level in this
        molecule's coarsening hierarchy, so small molecules with fewer
        coarsening levels are handled gracefully.

        Returns
        -------
        int
            Resolution level in ``[0, num_composed - 1]``.
        """
        t_int = self._scalar_t(t)
        level_tensor = self.resolution_schedule.resolution_level(t_int)
        level = level_tensor.item()
        # Clamp to available hierarchy depth
        max_level = self._num_composed - 1
        return min(level, max_level)

    def is_resolution_change(self, t: Union[int, torch.Tensor]) -> bool:
        """Whether the resolution changes at timestep t (i.e. r(t) != r(t-1)).

        At ``t=0`` this is always ``False``.

        Returns
        -------
        bool
        """
        t_int = self._scalar_t(t)
        return bool(self.resolution_schedule.is_resolution_change(t_int).item())

    def apply_Mt(
        self,
        x: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the per-step degradation operator M_t to x.

        If the resolution changes at t (r(t) != r(t-1)):

            M_t @ x = a_t * C^{r(t)} @ pinv(C^{r(t-1)}) @ x

        where C^{r(k)} denotes the *composed* coarsening matrix from level 0
        to level r(k).

        If the resolution does NOT change at t:

            M_t @ x = a_t * x

        Args:
            x: Input tensor of shape ``(N, D)`` where N is the number of
                nodes at the current resolution level (r(t-1)) and D is the
                feature dimension.
            t: Diffusion timestep (scalar).

        Returns:
            Degraded tensor. Shape is ``(N_{r(t)}, D)`` -- the number of rows
            may decrease when a resolution change occurs.
        """
        t_int = self._scalar_t(t)
        a_t = self._get_alpha_t(t_int)

        if not self.is_resolution_change(t_int):
            # No resolution change: simple scaling
            return a_t * x

        # Resolution change: coarsen then scale
        r_t = self.get_resolution_at(t_int)
        r_prev = self.get_resolution_at(t_int - 1) if t_int > 0 else 0

        C_rt = self._get_composed_C(r_t)          # (N_{r(t)}, N_atoms)
        C_pinv_prev = self._get_composed_pinv(r_prev)  # (N_atoms, N_{r(t-1)})

        # M_t @ x = a_t * C^{r(t)} @ pinv(C^{r(t-1)}) @ x
        return a_t * (C_rt @ (C_pinv_prev @ x))

    def apply_M1t(
        self,
        x0: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Apply the composed degradation M_{1:t} to clean data x0.

        Thanks to the telescoping property:

            M_{1:t} @ x0 = alpha_bar_t * C^{r(t)} @ x0

        This is the main operator used in the forward process to obtain the
        deterministic (signal) part of the noisy sample at timestep t.

        Args:
            x0: Clean data of shape ``(N_atoms, D)`` at full atomic resolution
                (level 0).
            t: Diffusion timestep (scalar).

        Returns:
            Degraded tensor of shape ``(N_{r(t)}, D)``.
        """
        t_int = self._scalar_t(t)
        alpha_bar_t = self.noise_schedule.alpha_bar(t_int)
        r_t = self.get_resolution_at(t_int)

        C_rt = self._get_composed_C(r_t)  # (N_{r(t)}, N_atoms)

        return alpha_bar_t * (C_rt @ x0)

    def apply_Mt_transpose(
        self,
        v: torch.Tensor,
        t: Union[int, torch.Tensor],
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute M_t^T @ v via vector-Jacobian product (VJP).

        Uses ``torch.autograd.grad`` to compute the transpose action without
        forming M_t explicitly.  This is the preferred method during training
        because it naturally handles any future modifications to M_t (e.g.,
        making it learnable).

        If *x* is not provided, a dummy tensor of appropriate shape is created
        (the VJP result is independent of x for linear operators).

        Args:
            v: Vector of shape ``(N_{r(t)}, D)`` -- lives in the output space
                of M_t.
            t: Diffusion timestep (scalar).
            x: Optional input tensor of shape ``(N_{r(t-1)}, D)``.  Used only
                to determine the input shape and device.

        Returns:
            M_t^T @ v, of shape ``(N_{r(t-1)}, D)``.
        """
        t_int = self._scalar_t(t)

        # Determine input dimension
        r_prev = self.get_resolution_at(t_int - 1) if t_int > 0 else 0
        n_input = self._get_composed_C(r_prev).shape[0]

        if x is None:
            D = v.shape[1] if v.ndim > 1 else 1
            x = torch.zeros(n_input, D, dtype=v.dtype, device=v.device)

        # Wrap apply_Mt as a function of x only
        def Mt_fn(x_in: torch.Tensor) -> torch.Tensor:
            return self.apply_Mt(x_in, t_int)

        return apply_Mt_vjp(Mt_fn, v, x)

    def apply_Mt_transpose_explicit(
        self,
        v: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute M_t^T @ v by explicitly transposing the matrix.

        This is a validation/debugging method that materializes the full M_t
        matrix and computes its transpose.  For production use, prefer
        :meth:`apply_Mt_transpose` which uses VJP and avoids large matrices.

        Args:
            v: Vector of shape ``(N_{r(t)}, D)``.
            t: Diffusion timestep (scalar).

        Returns:
            M_t^T @ v, of shape ``(N_{r(t-1)}, D)``.
        """
        t_int = self._scalar_t(t)
        a_t = self._get_alpha_t(t_int)

        if not self.is_resolution_change(t_int):
            # M_t = a_t * I  =>  M_t^T = a_t * I
            return a_t * v

        r_t = self.get_resolution_at(t_int)
        r_prev = self.get_resolution_at(t_int - 1) if t_int > 0 else 0

        C_rt = self._get_composed_C(r_t)               # (N_{r(t)}, N_atoms)
        C_pinv_prev = self._get_composed_pinv(r_prev)   # (N_atoms, N_{r(t-1)})

        # M_t = a_t * C_rt @ C_pinv_prev
        # M_t^T = a_t * C_pinv_prev^T @ C_rt^T
        return a_t * (C_pinv_prev.t() @ (C_rt.t() @ v))

    def compute_MtT_Mt(
        self,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the matrix M_t^T @ M_t (needed for posterior covariance).

        When r(t) != r(t-1):

            M_t^T M_t = a_t^2 * pinv(C^{r(t-1)})^T @ C^{r(t)}^T
                        @ C^{r(t)} @ pinv(C^{r(t-1)})

        When r(t) == r(t-1):

            M_t^T M_t = a_t^2 * I

        Args:
            t: Diffusion timestep (scalar).

        Returns:
            Square matrix of shape ``(N_{r(t-1)}, N_{r(t-1)})``.
        """
        t_int = self._scalar_t(t)
        a_t = self._get_alpha_t(t_int)
        a_t_sq = a_t ** 2

        if not self.is_resolution_change(t_int):
            # M_t^T M_t = a_t^2 * I
            r = self.get_resolution_at(t_int)
            n = self._get_composed_C(r).shape[0]
            device = self._get_composed_C(r).device
            return a_t_sq * torch.eye(n, dtype=torch.float32, device=device)

        r_t = self.get_resolution_at(t_int)
        r_prev = self.get_resolution_at(t_int - 1) if t_int > 0 else 0

        C_rt = self._get_composed_C(r_t)               # (N_{r(t)}, N_atoms)
        C_pinv_prev = self._get_composed_pinv(r_prev)   # (N_atoms, N_{r(t-1)})

        # A = C_rt @ C_pinv_prev   shape: (N_{r(t)}, N_{r(t-1)})
        A = C_rt @ C_pinv_prev
        # M_t^T M_t = a_t^2 * A^T @ A
        return a_t_sq * (A.t() @ A)

    def compute_MtT_Mt_matvec(
        self,
        v: torch.Tensor,
        t: Union[int, torch.Tensor],
    ) -> torch.Tensor:
        """Compute (M_t^T M_t) @ v without materializing the full matrix.

        This is more memory-efficient than :meth:`compute_MtT_Mt` followed by
        a matrix-vector product, especially for large molecules.  It is the
        preferred interface for iterative solvers such as Lanczos.

        When r(t) == r(t-1): returns ``a_t^2 * v``.

        When r(t) != r(t-1):
            1. w  = pinv(C^{r(t-1)}) @ v
            2. w  = C^{r(t)} @ w                 (project to coarser level)
            3. w  = C^{r(t)}^T @ w               (project back via transpose)
            4. w  = pinv(C^{r(t-1)})^T @ w       (lift back via pinv transpose)
            5. return a_t^2 * w

        Args:
            v: Vector of shape ``(N_{r(t-1)}, D)``.
            t: Diffusion timestep (scalar).

        Returns:
            (M_t^T M_t) @ v, same shape as *v*.
        """
        t_int = self._scalar_t(t)
        a_t = self._get_alpha_t(t_int)
        a_t_sq = a_t ** 2

        if not self.is_resolution_change(t_int):
            return a_t_sq * v

        r_t = self.get_resolution_at(t_int)
        r_prev = self.get_resolution_at(t_int - 1) if t_int > 0 else 0

        C_rt = self._get_composed_C(r_t)               # (N_{r(t)}, N_atoms)
        C_pinv_prev = self._get_composed_pinv(r_prev)   # (N_atoms, N_{r(t-1)})

        # Forward pass: A @ v = C_rt @ C_pinv_prev @ v
        w = C_pinv_prev @ v                # (N_atoms, D)
        w = C_rt @ w                       # (N_{r(t)}, D)

        # Transpose pass: A^T @ (A @ v) = C_pinv_prev^T @ C_rt^T @ w
        w = C_rt.t() @ w                   # (N_atoms, D)
        w = C_pinv_prev.t() @ w            # (N_{r(t-1)}, D)

        return a_t_sq * w
