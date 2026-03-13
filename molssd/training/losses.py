"""Loss functions for MolSSD training.

Implements Min-SNR-gamma weighted position loss, cross-entropy type loss,
and the combined MolSSD objective following the EDM and SSD training protocols.

References:
    - Min-SNR weighting: Hang et al., "Efficient Diffusion Training via
      Min-SNR Weighting Strategy" (ICLR 2024)
    - SSD loss formulation: arXiv:2603.08709
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MinSNRWeighting:
    """Min-SNR-gamma loss weighting strategy.

    Reduces the weight of high-noise (low-SNR) timesteps to stabilize
    training and reduce gradient variance. The weight is defined as:

        w(t) = min(SNR(t), gamma) / SNR(t)

    For high-SNR timesteps (low noise), w(t) ~ 1.
    For low-SNR timesteps (high noise), w(t) ~ gamma / SNR(t) < 1.

    Args:
        gamma: Clamping threshold for SNR values. Default is 5.0,
            which is the recommended value from the Min-SNR paper.
    """

    def __init__(self, gamma: float = 5.0) -> None:
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma

    def __call__(self, snr_values: torch.Tensor) -> torch.Tensor:
        """Compute Min-SNR weights for a batch of SNR values.

        Args:
            snr_values: Tensor of shape (B,) containing SNR(t) values
                for each sample in the batch. Must be positive.

        Returns:
            Tensor of shape (B,) with per-sample weights in (0, 1].
        """
        # Clamp SNR to avoid division by zero for very small values
        snr_clamped = snr_values.clamp(min=1e-8)
        weights = torch.clamp(snr_clamped, max=self.gamma) / snr_clamped
        return weights

    def __repr__(self) -> str:
        return f"MinSNRWeighting(gamma={self.gamma})"


class PositionLoss(nn.Module):
    """MSE loss on predicted vs true noise for atom positions.

    Computes the mean squared error between predicted and true noise
    vectors, weighted by Min-SNR-gamma per-sample weights. Operates
    on 3D position coordinates with shape (B, N, 3) or flattened
    batches from PyG-style data.

    Args:
        snr_gamma: Gamma parameter for Min-SNR weighting. Default 5.0.
    """

    def __init__(self, snr_gamma: float = 5.0) -> None:
        super().__init__()
        self.snr_weighting = MinSNRWeighting(gamma=snr_gamma)

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        snr_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Min-SNR weighted position MSE loss.

        Args:
            eps_pred: Predicted noise, shape (B, N, 3) or (B, N_total, 3)
                where N is the number of atoms per molecule.
            eps_true: True noise, same shape as eps_pred.
            snr_values: SNR values for each sample, shape (B,).

        Returns:
            Scalar loss tensor (mean over batch, atoms, and coordinates).
        """
        # Per-sample MSE: average over atoms and coordinates
        # Shape: (B, N, 3) -> (B,) after mean over last two dims
        per_sample_mse = (eps_pred - eps_true).pow(2).mean(dim=list(range(1, eps_pred.dim())))

        # Apply Min-SNR weighting
        weights = self.snr_weighting(snr_values)

        # Weighted mean over batch
        loss = (weights * per_sample_mse).mean()
        return loss


class TypeLoss(nn.Module):
    """Cross-entropy loss on atom type predictions.

    Computes cross-entropy between predicted atom type logits and
    ground truth atom type indices. Handles coarsened types where
    a supernode's type is determined by majority vote over its
    constituent atoms.

    Args:
        num_classes: Number of atom type classes. If None, inferred
            from logits dimension.
        label_smoothing: Label smoothing factor. Default 0.0.
    """

    def __init__(
        self,
        num_classes: int | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing

    def forward(
        self,
        type_logits: torch.Tensor,
        type_targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for atom types.

        Args:
            type_logits: Predicted logits, shape (B, N, C) or (N_total, C)
                where C is the number of atom type classes.
            type_targets: Ground truth atom type indices, shape (B, N)
                or (N_total,). Values should be in [0, C-1].

        Returns:
            Scalar loss tensor (mean over all atoms).
        """
        # Flatten to 2D for cross-entropy: (N_total, C) and (N_total,)
        if type_logits.dim() == 3:
            B, N, C = type_logits.shape
            type_logits = type_logits.reshape(-1, C)
            type_targets = type_targets.reshape(-1)
        elif type_logits.dim() == 2:
            pass  # Already in (N_total, C) format
        else:
            raise ValueError(
                f"type_logits must be 2D or 3D, got {type_logits.dim()}D"
            )

        loss = F.cross_entropy(
            type_logits,
            type_targets,
            label_smoothing=self.label_smoothing,
        )
        return loss


class MolSSDLoss(nn.Module):
    """Combined MolSSD training objective.

    Computes the joint loss:

        L = L_pos + lambda_type * L_type

    where L_pos is the Min-SNR weighted MSE on position noise predictions
    and L_type is the cross-entropy on atom type predictions.

    Args:
        lambda_type: Weight for the atom type loss term. Default 0.1.
        snr_gamma: Gamma parameter for Min-SNR weighting. Default 5.0.
        label_smoothing: Label smoothing for type cross-entropy. Default 0.0.
    """

    def __init__(
        self,
        lambda_type: float = 0.1,
        snr_gamma: float = 5.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_type = lambda_type
        self.position_loss = PositionLoss(snr_gamma=snr_gamma)
        self.type_loss = TypeLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        type_logits: torch.Tensor,
        type_targets: torch.Tensor,
        snr_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the combined MolSSD loss.

        Args:
            eps_pred: Predicted position noise, shape (B, N, 3).
            eps_true: True position noise, shape (B, N, 3).
            type_logits: Predicted atom type logits, shape (B, N, C) or (N_total, C).
            type_targets: Ground truth atom types, shape (B, N) or (N_total,).
            snr_values: Per-sample SNR values, shape (B,).

        Returns:
            Tuple of:
                - total_loss: Scalar combined loss tensor.
                - loss_dict: Dictionary with individual loss components:
                    - "loss_pos": Position noise MSE (weighted).
                    - "loss_type": Atom type cross-entropy.
                    - "loss_total": Combined loss value.
        """
        l_pos = self.position_loss(eps_pred, eps_true, snr_values)
        l_type = self.type_loss(type_logits, type_targets)

        total_loss = l_pos + self.lambda_type * l_type

        loss_dict = {
            "loss_pos": l_pos.detach(),
            "loss_type": l_type.detach(),
            "loss_total": total_loss.detach(),
        }

        return total_loss, loss_dict
