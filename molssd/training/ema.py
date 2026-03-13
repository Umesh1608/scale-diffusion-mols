"""Exponential Moving Average (EMA) of model parameters.

Maintains shadow copies of model parameters that are updated as an
exponential moving average during training. The EMA parameters typically
produce better generation quality and are used for evaluation and sampling.

Usage::

    model = MyModel()
    ema = ExponentialMovingAverage(model, decay=0.9999)

    for batch in dataloader:
        loss = model(batch)
        loss.backward()
        optimizer.step()
        ema.update()

    # Evaluate with EMA parameters
    with ema.average_parameters():
        model.eval()
        evaluate(model)
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of each model parameter, updated at each
    training step as:

        shadow = decay * shadow + (1 - decay) * param

    The shadow parameters can be temporarily applied to the model for
    evaluation via :meth:`apply_shadow` / :meth:`restore` or via the
    :meth:`average_parameters` context manager.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor. Values close to 1.0 (e.g. 0.9999)
            produce smoother averages. Must be in [0, 1).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")

        self.model = model
        self.decay = decay
        self.num_updates = 0

        # Shadow parameters: deep copy of initial model parameters
        self.shadow_params: list[torch.Tensor] = [
            p.clone().detach() for p in model.parameters()
        ]

        # Backup storage for original parameters during apply_shadow
        self._backup_params: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters.

        Applies the EMA update rule:
            shadow = decay * shadow + (1 - decay) * param

        Should be called once after each optimizer step.
        """
        self.num_updates += 1
        for shadow, param in zip(self.shadow_params, self.model.parameters()):
            if param.requires_grad:
                shadow.lerp_(param.data, 1.0 - self.decay)

    def apply_shadow(self) -> None:
        """Copy shadow (EMA) parameters into the model.

        The original parameters are saved internally so they can be
        restored later with :meth:`restore`. This is useful for
        evaluation with EMA parameters.

        Raises:
            RuntimeError: If shadow parameters are already applied
                (i.e., :meth:`apply_shadow` was called without a
                matching :meth:`restore`).
        """
        if self._backup_params is not None:
            raise RuntimeError(
                "Shadow parameters are already applied. "
                "Call restore() before calling apply_shadow() again."
            )

        self._backup_params = [
            p.data.clone() for p in self.model.parameters()
        ]

        for shadow, param in zip(self.shadow_params, self.model.parameters()):
            param.data.copy_(shadow)

    def restore(self) -> None:
        """Restore the original model parameters after :meth:`apply_shadow`.

        Raises:
            RuntimeError: If :meth:`apply_shadow` was not called first.
        """
        if self._backup_params is None:
            raise RuntimeError(
                "No backup parameters to restore. "
                "Call apply_shadow() before calling restore()."
            )

        for backup, param in zip(self._backup_params, self.model.parameters()):
            param.data.copy_(backup)

        self._backup_params = None

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Context manager to temporarily use EMA parameters.

        Applies shadow parameters on entry and restores original
        parameters on exit. Safe to use even if an exception occurs.

        Example::

            with ema.average_parameters():
                model.eval()
                val_loss = evaluate(model)
        """
        self.apply_shadow()
        try:
            yield
        finally:
            self.restore()

    def state_dict(self) -> Dict[str, Any]:
        """Return the EMA state for checkpointing.

        Returns:
            Dictionary containing:
                - decay: The EMA decay factor.
                - num_updates: Number of update steps performed.
                - shadow_params: List of shadow parameter tensors.
        """
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow_params": [p.clone() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from a checkpoint.

        Args:
            state_dict: Dictionary previously returned by :meth:`state_dict`.

        Raises:
            ValueError: If the number of shadow parameters does not match
                the number of model parameters.
        """
        shadow_params = state_dict["shadow_params"]
        if len(shadow_params) != len(self.shadow_params):
            raise ValueError(
                f"Number of shadow parameters ({len(shadow_params)}) does not "
                f"match model parameters ({len(self.shadow_params)})"
            )

        self.decay = state_dict["decay"]
        self.num_updates = state_dict["num_updates"]
        self.shadow_params = [p.clone().detach() for p in shadow_params]

    def __repr__(self) -> str:
        return (
            f"ExponentialMovingAverage("
            f"decay={self.decay}, "
            f"num_updates={self.num_updates}, "
            f"num_params={len(self.shadow_params)})"
        )
