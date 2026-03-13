"""Optimizer and learning-rate scheduler utilities for MolSSD.

Provides factory functions for creating an AdamW optimizer and a
linear-warmup + cosine-decay learning rate schedule, which is the
standard recipe for training diffusion models.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


def get_optimizer(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Create an AdamW optimizer for the model.

    Args:
        model: The model whose parameters to optimise.
        lr: Learning rate.
        weight_decay: L2 regularisation coefficient.
        betas: Adam beta coefficients (momentum, variance).
        eps: Term added to the denominator for numerical stability.

    Returns:
        Configured ``AdamW`` optimizer.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 5000,
    total_steps: int = 500_000,
) -> LambdaLR:
    """Create a linear-warmup + cosine-decay learning rate scheduler.

    The schedule has two phases:

    1. **Linear warmup** (steps 0 .. warmup_steps - 1): LR increases
       linearly from 0 to the base LR.
    2. **Cosine decay** (steps warmup_steps .. total_steps): LR decays
       following a cosine curve from base LR to 0.

    After ``total_steps`` the LR stays at 0.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps (warmup + decay).

    Returns:
        A ``LambdaLR`` scheduler that should be stepped once per
        training step (not per epoch).
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: 0 -> 1
            return step / max(1, warmup_steps)
        # Cosine decay: 1 -> 0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)  # clamp after total_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
