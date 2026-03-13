"""Full diffusion process for MolSSD (Molecular Scale-Space Diffusion).

This module ties together the coarsening hierarchy, degradation operator,
noise schedules, and Lanczos-based posterior sampling into a complete
forward/reverse diffusion process for 3D molecular generation.

Forward process:
    x_t = M_{1:t} x_0 + sigma_t epsilon,   epsilon ~ N(0, I)

    where M_{1:t} = alpha_bar_t C^{r(t)} is the composed degradation operator
    that simultaneously coarsens (via spectral graph coarsening) and scales
    (via the noise schedule) the clean coordinates.

Reverse process:
    At non-resolution-changing steps (standard DDPM posterior):
        mu = (sqrt(alpha_bar_{t-1}) beta_t) / (1 - alpha_bar_t) x0_hat
             + (sqrt(alpha_t) (1 - alpha_bar_{t-1})) / (1 - alpha_bar_t) x_t
        sigma = sqrt(beta_tilde_t)

    At resolution-changing steps (non-isotropic posterior via Lanczos):
        Sigma = sigma_{t-1}^2 I - (sigma_{t-1}^4 / sigma_t^2) M_t^T M_t
        mu = M_{1:t-1} x0_hat + (sigma_{t-1}^2 / sigma_t^2) M_t^T (x_t - M_t M_{1:t-1} x0_hat)

References:
    - SSD paper (arXiv:2603.08709): Scale-space diffusion framework
    - Ho et al. (2020): DDPM forward/reverse process
    - Nichol & Dhariwal (2021): Improved DDPM with cosine schedule
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from molssd.core.coarsening import CoarseningLevel, build_coarsening_hierarchy
from molssd.core.degradation import DegradationOperator
from molssd.core.lanczos import posterior_covariance_eigendecomp, sample_non_isotropic
from molssd.core.noise_schedules import NoiseSchedule, ResolutionSchedule


class MolSSDDiffusion(nn.Module):
    """Full diffusion process for MolSSD molecular generation.

    Combines spectral graph coarsening with DDPM-style diffusion to produce
    a multi-scale generative model for 3D molecular structures. The forward
    process progressively coarsens and noises the molecular coordinates,
    while the reverse process denoises and refines them, using Lanczos-based
    sampling at resolution-changing steps.

    Parameters
    ----------
    noise_schedule : NoiseSchedule
        Provides alpha_bar(t), sigma(t), beta(t) for the diffusion process.
    resolution_schedule : ResolutionSchedule
        Maps timesteps to resolution levels and identifies resolution changes.
    num_atom_types : int
        Number of discrete atom types (e.g. 5 for {H, C, N, O, F} in QM9).
    """

    def __init__(
        self,
        noise_schedule: NoiseSchedule,
        resolution_schedule: ResolutionSchedule,
        num_atom_types: int = 5,
    ) -> None:
        super().__init__()
        self.noise_schedule = noise_schedule
        self.resolution_schedule = resolution_schedule
        self.num_atom_types = num_atom_types

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def forward_process(
        self,
        x0: torch.Tensor,
        atom_types: torch.Tensor,
        t: int,
        coarsening_hierarchy: List[CoarseningLevel],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the forward (noising + coarsening) process to produce x_t.

        Applies the composed degradation operator M_{1:t} to the clean
        coordinates x0 and adds Gaussian noise scaled by sigma_t:

            x_t = M_{1:t} x_0 + sigma_t * epsilon,  epsilon ~ N(0, I)

        Also coarsens the discrete atom types to the target resolution
        via majority vote within each super-node cluster.

        Args:
            x0: Clean atomic coordinates of shape ``(N, 3)`` at full atomic
                resolution.
            atom_types: Integer tensor of shape ``(N,)`` with atom type
                indices in ``[0, num_atom_types)``.
            t: Diffusion timestep (0-indexed).
            coarsening_hierarchy: List of CoarseningLevel objects defining
                the multi-scale hierarchy for this molecule.

        Returns:
            x_t: Noised, possibly coarsened coordinates of shape
                ``(N_{r(t)}, 3)``.
            epsilon: The noise sample that was added, shape ``(N_{r(t)}, 3)``.
                Needed for training the denoiser to predict epsilon.
            coarsened_types: Atom types at the target resolution, shape
                ``(N_{r(t)},)``.  At full resolution (r(t)=0) this equals
                the input atom_types.
        """
        device = x0.device
        dtype = x0.dtype

        # Build the degradation operator for this hierarchy
        degradation_op = DegradationOperator(
            coarsening_hierarchy=coarsening_hierarchy,
            noise_schedule=self.noise_schedule,
            resolution_schedule=self.resolution_schedule,
        )
        # Move degradation op buffers to the same device
        degradation_op = degradation_op.to(device=device)

        # Apply composed degradation: M_{1:t} x_0 = alpha_bar_t C^{r(t)} x_0
        signal = degradation_op.apply_M1t(x0, t)  # (N_{r(t)}, 3)
        N_rt = signal.shape[0]

        # Sample noise
        epsilon = torch.randn(N_rt, 3, device=device, dtype=dtype)

        # Noise level
        sigma_t = self.noise_schedule.sigma(t).to(device=device, dtype=dtype)

        # Forward sample
        x_t = signal + sigma_t * epsilon

        # Coarsen atom types via majority vote
        r_t = degradation_op.get_resolution_at(t)
        if r_t == 0:
            coarsened_types = atom_types
        else:
            # Apply coarsening through all levels up to r(t)
            current_types = atom_types
            for level_idx in range(r_t):
                level = coarsening_hierarchy[level_idx]
                current_types = self.coarsen_atom_types(
                    current_types,
                    level.cluster_assignment,
                    level.num_nodes,
                )
            coarsened_types = current_types

        return x_t, epsilon, coarsened_types

    # ------------------------------------------------------------------
    # Posterior computation
    # ------------------------------------------------------------------

    def compute_posterior_params(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t: int,
        degradation_op: DegradationOperator,
    ) -> Dict[str, Any]:
        """Compute posterior distribution parameters for the reverse step.

        At non-resolution-changing steps, returns the standard DDPM posterior
        (isotropic Gaussian).  At resolution-changing steps, returns the
        non-isotropic posterior parameters that require Lanczos sampling.

        Standard DDPM posterior (r(t) == r(t-1)):
            mu = (sqrt(alpha_bar_{t-1}) beta_t) / (1 - alpha_bar_t) x0_hat
                 + (sqrt(alpha_t) (1 - alpha_bar_{t-1})) / (1 - alpha_bar_t) x_t
            sigma^2 = beta_tilde_t = beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)

        Non-isotropic posterior (r(t) != r(t-1)):
            mu = M_{1:t-1} x0_hat + (sigma_{t-1}^2 / sigma_t^2) M_t^T (x_t - M_t M_{1:t-1} x0_hat)
            Sigma eigendecomposed via Lanczos on M_t^T M_t

        Args:
            x_t: Current noisy sample of shape ``(N_{r(t)}, 3)``.
            x0_hat: Predicted clean data of shape ``(N, 3)`` at full atomic
                resolution.
            t: Current diffusion timestep. Must be >= 1 (not called at t=0).
            degradation_op: DegradationOperator for this molecule's hierarchy.

        Returns:
            Dictionary with keys:
            - ``'is_isotropic'``: bool flag.
            - ``'mu'``: Posterior mean, shape ``(N_{r(t-1)}, 3)``.
            - ``'sigma'``: (isotropic only) Posterior std dev, scalar tensor.
            - ``'eigenvalues'``: (non-isotropic only) Eigenvalues of Sigma,
              shape ``(k,)``.
            - ``'ritz_vectors'``: (non-isotropic only) Eigenvectors, shape
              ``(N_{r(t-1)}, k)``.
            - ``'sigma_t_minus_1'``: (non-isotropic only) sigma at t-1.
        """
        device = x_t.device
        dtype = x_t.dtype

        is_res_change = degradation_op.is_resolution_change(t)

        if not is_res_change:
            # ---- Standard DDPM posterior ----
            return self._compute_isotropic_posterior(x_t, x0_hat, t, degradation_op)
        else:
            # ---- Non-isotropic posterior (resolution change) ----
            return self._compute_non_isotropic_posterior(
                x_t, x0_hat, t, degradation_op
            )

    def _compute_isotropic_posterior(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t: int,
        degradation_op: DegradationOperator,
    ) -> Dict[str, Any]:
        """Compute the standard DDPM isotropic Gaussian posterior.

        mu_t = (sqrt(alpha_bar_{t-1}) beta_t) / (1 - alpha_bar_t^2) x0_hat_coarsened
               + (sqrt(alpha_t) (1 - alpha_bar_{t-1}^2)) / (1 - alpha_bar_t^2) x_t

        beta_tilde_t = beta_t (1 - alpha_bar_{t-1}^2) / (1 - alpha_bar_t^2)

        Note: In the variance-preserving formulation where alpha_bar^2 + sigma^2 = 1,
        we use sigma^2 in place of (1 - alpha_bar^2) since they are the same quantity.
        """
        device = x_t.device
        dtype = x_t.dtype

        alpha_bar_t = self.noise_schedule.alpha_bar(t).to(device=device, dtype=dtype)
        sigma_sq_t = self.noise_schedule.sigma_squared(t).to(device=device, dtype=dtype)
        beta_t = self.noise_schedule.beta(t).to(device=device, dtype=dtype)

        if t > 0:
            alpha_bar_tm1 = self.noise_schedule.alpha_bar(t - 1).to(device=device, dtype=dtype)
            sigma_sq_tm1 = self.noise_schedule.sigma_squared(t - 1).to(device=device, dtype=dtype)
        else:
            alpha_bar_tm1 = torch.ones(1, device=device, dtype=dtype)
            sigma_sq_tm1 = torch.zeros(1, device=device, dtype=dtype)

        # alpha_t = alpha_bar_t / alpha_bar_{t-1}
        alpha_t = alpha_bar_t / alpha_bar_tm1.clamp(min=1e-12)

        # At the same resolution, x0_hat needs to be projected to the
        # current resolution level
        r_t = degradation_op.get_resolution_at(t)
        C_rt = degradation_op._get_composed_C(r_t)
        x0_hat_coarsened = C_rt @ x0_hat  # (N_{r(t)}, 3)

        # DDPM posterior mean
        # mu = (sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t^2)) * x0_hat_coarsened
        #    + (sqrt(alpha_t) * (1 - alpha_bar_{t-1}^2) / (1 - alpha_bar_t^2)) * x_t
        coeff_x0 = (torch.sqrt(alpha_bar_tm1) * beta_t) / sigma_sq_t.clamp(min=1e-12)
        coeff_xt = (torch.sqrt(alpha_t) * sigma_sq_tm1) / sigma_sq_t.clamp(min=1e-12)

        mu = coeff_x0 * x0_hat_coarsened + coeff_xt * x_t

        # DDPM posterior variance: beta_tilde = beta_t * sigma^2_{t-1} / sigma^2_t
        beta_tilde = (beta_t * sigma_sq_tm1) / sigma_sq_t.clamp(min=1e-12)
        sigma_posterior = torch.sqrt(beta_tilde.clamp(min=1e-12))

        return {
            "is_isotropic": True,
            "mu": mu,
            "sigma": sigma_posterior,
        }

    def _compute_non_isotropic_posterior(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t: int,
        degradation_op: DegradationOperator,
    ) -> Dict[str, Any]:
        """Compute the non-isotropic posterior at a resolution-changing step.

        mu = M_{1:t-1} x0_hat + (sigma_{t-1}^2 / sigma_t^2) M_t^T (x_t - M_t M_{1:t-1} x0_hat)
        Sigma = sigma_{t-1}^2 I - (sigma_{t-1}^4 / sigma_t^2) M_t^T M_t
        """
        device = x_t.device
        dtype = x_t.dtype

        sigma_t = self.noise_schedule.sigma(t).to(device=device, dtype=dtype)
        sigma_sq_t = self.noise_schedule.sigma_squared(t).to(device=device, dtype=dtype)

        if t > 0:
            sigma_tm1 = self.noise_schedule.sigma(t - 1).to(device=device, dtype=dtype)
            sigma_sq_tm1 = self.noise_schedule.sigma_squared(t - 1).to(device=device, dtype=dtype)
        else:
            sigma_tm1 = torch.zeros(1, device=device, dtype=dtype)
            sigma_sq_tm1 = torch.zeros(1, device=device, dtype=dtype)

        # M_{1:t-1} x0_hat: project x0_hat to resolution at t-1
        M1tm1_x0hat = degradation_op.apply_M1t(x0_hat, t - 1) if t > 0 else x0_hat

        # M_t M_{1:t-1} x0_hat: further project to resolution at t
        Mt_M1tm1_x0hat = degradation_op.apply_Mt(M1tm1_x0hat, t)

        # Residual at coarser level
        residual = x_t - Mt_M1tm1_x0hat  # (N_{r(t)}, 3)

        # M_t^T residual: lift back to resolution at t-1
        # Use explicit transpose when grad is disabled (e.g., during sampling)
        # because VJP requires autograd to be active.
        if torch.is_grad_enabled():
            MtT_residual = degradation_op.apply_Mt_transpose(residual, t)
        else:
            MtT_residual = degradation_op.apply_Mt_transpose_explicit(residual, t)

        # Posterior mean
        # mu = M_{1:t-1} x0_hat + (sigma_{t-1}^2 / sigma_t^2) M_t^T residual
        scale = sigma_sq_tm1 / sigma_sq_t.clamp(min=1e-12)
        mu = M1tm1_x0hat + scale * MtT_residual  # (N_{r(t-1)}, 3)

        # Eigendecomposition of posterior covariance via Lanczos
        N_rtm1 = mu.shape[0]
        r_t = degradation_op.get_resolution_at(t)
        # k = number of nodes at coarser resolution (rank of M_t^T M_t)
        k = degradation_op._get_composed_C(r_t).shape[0]

        # Build matvec function for M_t^T M_t operating on vectors (not matrices)
        def MtT_Mt_matvec(v: torch.Tensor) -> torch.Tensor:
            # v has shape (N_{r(t-1)},) -- need to reshape to (N_{r(t-1)}, 1) for degradation op
            v_2d = v.unsqueeze(1)
            result = degradation_op.compute_MtT_Mt_matvec(v_2d, t)
            return result.squeeze(1)

        eigenvalues, ritz_vectors = posterior_covariance_eigendecomp(
            MtT_Mt_matvec=MtT_Mt_matvec,
            sigma_t=sigma_t,
            sigma_t_minus_1=sigma_tm1,
            n=N_rtm1,
            k=k,
            device=device,
            dtype=dtype,
        )

        return {
            "is_isotropic": False,
            "mu": mu,
            "eigenvalues": eigenvalues,
            "ritz_vectors": ritz_vectors,
            "sigma_t_minus_1": sigma_tm1,
        }

    # ------------------------------------------------------------------
    # Reverse step
    # ------------------------------------------------------------------

    def reverse_step(
        self,
        x_t: torch.Tensor,
        t: int,
        posterior_params: Dict[str, Any],
    ) -> torch.Tensor:
        """Sample x_{t-1} from the posterior distribution.

        For isotropic posteriors (standard DDPM steps):
            x_{t-1} = mu + sigma * z,  z ~ N(0, I)

        For non-isotropic posteriors (resolution-changing steps):
            Uses Lanczos-based sample_non_isotropic to efficiently sample
            from the rank-deficient Gaussian.

        Args:
            x_t: Current noisy sample of shape ``(N_{r(t)}, 3)``.  Not
                directly used in the sampling formula (already encoded in
                posterior_params['mu']), but kept for interface consistency.
            t: Current diffusion timestep.
            posterior_params: Dictionary returned by
                :meth:`compute_posterior_params`.

        Returns:
            x_{t-1} of shape ``(N_{r(t-1)}, 3)``.
        """
        mu = posterior_params["mu"]
        device = mu.device
        dtype = mu.dtype

        if t == 0:
            # At t=0, no noise is added; return the mean directly
            return mu

        if posterior_params["is_isotropic"]:
            sigma = posterior_params["sigma"].to(device=device, dtype=dtype)
            z = torch.randn_like(mu)
            return mu + sigma * z
        else:
            return sample_non_isotropic(
                mu=mu,
                eigenvalues=posterior_params["eigenvalues"],
                ritz_vectors=posterior_params["ritz_vectors"],
                sigma_t_minus_1=posterior_params["sigma_t_minus_1"],
            )

    # ------------------------------------------------------------------
    # x0 prediction from epsilon prediction
    # ------------------------------------------------------------------

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: int,
        eps_pred: torch.Tensor,
        degradation_op: DegradationOperator,
    ) -> torch.Tensor:
        """Recover a prediction of x_0 from the network's epsilon prediction.

        The forward process is:

            x_t = alpha_bar_t C^{r(t)} x_0 + sigma_t epsilon

        Rearranging:

            C^{r(t)} x_0 = (x_t - sigma_t eps_pred) / alpha_bar_t

        Lifting back to full atomic resolution:

            x0_hat = pinv(C^{r(t)}) @ (x_t - sigma_t eps_pred) / alpha_bar_t

        Args:
            x_t: Noisy sample at timestep t, shape ``(N_{r(t)}, 3)``.
            t: Diffusion timestep.
            eps_pred: Predicted noise from the denoiser, shape ``(N_{r(t)}, 3)``.
            degradation_op: DegradationOperator for this molecule's hierarchy.

        Returns:
            x0_hat: Predicted clean coordinates at full atomic resolution,
                shape ``(N, 3)``.
        """
        device = x_t.device
        dtype = x_t.dtype

        alpha_bar_t = self.noise_schedule.alpha_bar(t).to(device=device, dtype=dtype)
        sigma_t = self.noise_schedule.sigma(t).to(device=device, dtype=dtype)

        # Recover coarsened x0: C^{r(t)} x0 = (x_t - sigma_t eps) / alpha_bar_t
        x0_coarse = (x_t - sigma_t * eps_pred) / alpha_bar_t.clamp(min=1e-12)

        # Lift to full resolution using pseudoinverse
        r_t = degradation_op.get_resolution_at(t)
        if r_t == 0:
            # Already at full resolution, no lifting needed
            return x0_coarse

        C_pinv = degradation_op._get_composed_pinv(r_t)  # (N_atoms, N_{r(t)})
        x0_hat = C_pinv @ x0_coarse  # (N_atoms, 3)

        return x0_hat

    # ------------------------------------------------------------------
    # Atom type coarsening
    # ------------------------------------------------------------------

    def coarsen_atom_types(
        self,
        atom_types: torch.Tensor,
        cluster_assignment: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """Coarsen discrete atom types via majority vote within clusters.

        Each super-node (cluster) is assigned the atom type that occurs most
        frequently among its constituent atoms.  Ties are broken by taking the
        smallest type index.

        Args:
            atom_types: Integer tensor of shape ``(N,)`` with atom type indices
                in ``[0, num_atom_types)``.
            cluster_assignment: Integer tensor of shape ``(N,)`` mapping each
                atom to its cluster (super-node) index in ``[0, num_clusters)``.
            num_clusters: Number of super-nodes (clusters).

        Returns:
            Coarsened atom types of shape ``(num_clusters,)`` as a long tensor.
        """
        device = atom_types.device
        coarsened = torch.zeros(num_clusters, dtype=torch.long, device=device)

        for I in range(num_clusters):
            mask = cluster_assignment == I
            if mask.any():
                cluster_types = atom_types[mask]
                # Count occurrences of each type
                counts = torch.zeros(
                    self.num_atom_types, dtype=torch.long, device=device
                )
                for type_idx in range(self.num_atom_types):
                    counts[type_idx] = (cluster_types == type_idx).sum()
                # Majority vote (argmax picks smallest index on ties)
                coarsened[I] = counts.argmax()
            else:
                # Empty cluster -- should not happen in practice
                coarsened[I] = 0

        return coarsened
