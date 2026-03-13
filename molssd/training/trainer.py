"""MolSSD training loop.

Implements the full training process for MolSSD, including per-molecule
forward diffusion, batched model inference, loss computation, EMA tracking,
mixed-precision training, and checkpointing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from molssd.core.coarsening import CoarseningLevel
from molssd.core.degradation import DegradationOperator
from molssd.core.diffusion import MolSSDDiffusion
from molssd.core.noise_schedules import NoiseSchedule, ResolutionSchedule
from molssd.models.flexi_net import MolecularFlexiNet
from molssd.training.ema import ExponentialMovingAverage
from molssd.training.losses import MolSSDLoss
from molssd.training.optimizers import get_optimizer, get_scheduler

logger = logging.getLogger(__name__)

try:
    import wandb

    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def _edge_index_from_adj(adj: torch.Tensor) -> torch.Tensor:
    """Extract COO edge_index ``(2, E)`` from a dense adjacency matrix."""
    row, col = torch.where(adj > 0)
    return torch.stack([row, col], dim=0)


def _select_amp_dtype() -> torch.dtype:
    """Pick bf16 if the GPU supports it, otherwise fp16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


class MolSSDTrainer:
    """Full training harness for MolSSD.

    Args:
        model: MolecularFlexiNet denoising network.
        diffusion: MolSSDDiffusion forward/reverse process.
        noise_schedule: Noise schedule (shared with diffusion).
        resolution_schedule: Resolution schedule (shared with diffusion).
        loss_fn: Combined MolSSD loss (position MSE + type CE).
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup duration.
        max_steps: Total training steps.
        ema_decay: EMA decay factor.
        grad_clip_norm: Max gradient norm for clipping.
        log_every: Logging interval (steps).
        sample_every: Sampling interval (steps).
        checkpoint_every: Checkpoint interval (steps).
        eval_every: Validation interval (steps).
        device: Training device (auto-detected if None).
        use_wandb: Log to wandb if available.
        checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: MolecularFlexiNet,
        diffusion: MolSSDDiffusion,
        noise_schedule: NoiseSchedule,
        resolution_schedule: ResolutionSchedule,
        loss_fn: Optional[MolSSDLoss] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 5000,
        max_steps: int = 500_000,
        ema_decay: float = 0.9999,
        grad_clip_norm: float = 1.0,
        log_every: int = 100,
        sample_every: int = 5000,
        checkpoint_every: int = 10_000,
        eval_every: int = 10_000,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        self.noise_schedule = noise_schedule.to(self.device)
        self.resolution_schedule = resolution_schedule.to(self.device)
        self.loss_fn = (loss_fn or MolSSDLoss()).to(self.device)

        self.max_steps = max_steps
        self.grad_clip_norm = grad_clip_norm
        self.log_every = log_every
        self.sample_every = sample_every
        self.checkpoint_every = checkpoint_every
        self.eval_every = eval_every
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb and _HAS_WANDB

        # Optimizer, scheduler, EMA
        self.optimizer = get_optimizer(model, lr=lr, weight_decay=weight_decay)
        self.scheduler = get_scheduler(
            self.optimizer, warmup_steps=warmup_steps, total_steps=max_steps
        )
        self.ema = ExponentialMovingAverage(model, decay=ema_decay)

        # Mixed precision
        self.amp_dtype = _select_amp_dtype()
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        self.global_step = 0

    # ------------------------------------------------------------------
    # Per-molecule forward diffusion + batching
    # ------------------------------------------------------------------

    def _forward_diffuse_batch(
        self, batch: Dict[str, Any], t: int
    ) -> Dict[str, Any]:
        """Run per-molecule forward diffusion and reassemble into a batch.

        For each molecule, builds a DegradationOperator from its coarsening
        hierarchy, runs ``diffusion.forward_process`` to get ``(x_t, epsilon,
        coarsened_types)``, and extracts the edge_index at the target
        resolution level. The results are concatenated with appropriate
        node-offset shifts to form a single batched graph.

        Returns dict with: x_t, epsilon, coarsened_types, edge_index,
        batch_vector, resolution_level, num_nodes_list.
        """
        positions = batch["positions"]
        atom_types = batch["atom_types"]
        num_atoms_list = batch["num_atoms_list"]
        adj_list = batch["adj_list"]
        hierarchies = batch["coarsening_hierarchies"]
        batch_size = batch["batch_size"]

        x_t_parts: List[torch.Tensor] = []
        eps_parts: List[torch.Tensor] = []
        types_parts: List[torch.Tensor] = []
        ei_parts: List[torch.Tensor] = []
        bvec_parts: List[torch.Tensor] = []

        node_offset = 0       # into the original concatenated positions
        coarse_offset = 0     # for the coarsened batched graph

        resolution_level: int = 0

        for i in range(batch_size):
            n_i = num_atoms_list[i]
            x0_i = positions[node_offset: node_offset + n_i].to(self.device)
            types_i = atom_types[node_offset: node_offset + n_i].to(self.device)
            adj_i = adj_list[i].to(self.device)
            hierarchy_i: List[CoarseningLevel] = hierarchies[i]

            # Build degradation operator (needed to query resolution level)
            degradation_op = DegradationOperator(
                coarsening_hierarchy=hierarchy_i,
                noise_schedule=self.noise_schedule,
                resolution_schedule=self.resolution_schedule,
            ).to(self.device)

            # Forward process -> x_t, epsilon, coarsened_types
            x_t_i, eps_i, ctypes_i = self.diffusion.forward_process(
                x0_i, types_i, t, hierarchy_i
            )

            # Resolution level (consistent across molecules for the same t)
            r_t = degradation_op.get_resolution_at(t)
            resolution_level = r_t

            # Edge index at the target resolution
            if r_t == 0:
                adj_at_level = adj_i
            elif r_t <= len(hierarchy_i):
                adj_at_level = hierarchy_i[r_t - 1].coarsened_adj.to(self.device)
            else:
                # Hierarchy is shallower than the requested level; use coarsest
                adj_at_level = hierarchy_i[-1].coarsened_adj.to(self.device)

            ei_i = _edge_index_from_adj(adj_at_level) + coarse_offset
            n_coarse = x_t_i.shape[0]

            x_t_parts.append(x_t_i)
            eps_parts.append(eps_i)
            types_parts.append(ctypes_i)
            ei_parts.append(ei_i)
            bvec_parts.append(
                torch.full((n_coarse,), i, dtype=torch.long, device=self.device)
            )

            node_offset += n_i
            coarse_offset += n_coarse

        return {
            "x_t": torch.cat(x_t_parts, dim=0),
            "epsilon": torch.cat(eps_parts, dim=0),
            "coarsened_types": torch.cat(types_parts, dim=0),
            "edge_index": torch.cat(ei_parts, dim=1),
            "batch_vector": torch.cat(bvec_parts, dim=0),
            "resolution_level": resolution_level,
            "num_nodes_list": [x.shape[0] for x in x_t_parts],
        }

    # ------------------------------------------------------------------
    # Padding helper
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_to_batch(
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        num_nodes_list: List[int],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reshape flat ``(N_total, 3)`` tensors to padded ``(B, max_N, 3)``.

        Padding positions are filled with matching values so they contribute
        zero to the per-sample MSE.
        """
        max_n = max(num_nodes_list)
        device = eps_pred.device
        dtype = eps_pred.dtype

        pred_padded = torch.zeros(batch_size, max_n, 3, device=device, dtype=dtype)
        true_padded = torch.zeros(batch_size, max_n, 3, device=device, dtype=dtype)

        offset = 0
        for i, n_i in enumerate(num_nodes_list):
            pred_padded[i, :n_i] = eps_pred[offset: offset + n_i]
            true_padded[i, :n_i] = eps_true[offset: offset + n_i]
            offset += n_i

        return pred_padded, true_padded

    # ------------------------------------------------------------------
    # Single train / val step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one training step. Returns a dict of scalar loss values."""
        self.model.train()

        T = self.noise_schedule.T
        batch_size = batch["batch_size"]

        # Same timestep for all molecules in the batch (keeps a single
        # resolution level so the model runs one level's EGNN stack).
        t = torch.randint(0, T, (1,)).item()

        # Forward diffusion per molecule, then batch
        fwd = self._forward_diffuse_batch(batch, t)

        x_t = fwd["x_t"]
        epsilon = fwd["epsilon"]
        coarsened_types = fwd["coarsened_types"]
        edge_index = fwd["edge_index"]
        batch_vector = fwd["batch_vector"]
        resolution_level = fwd["resolution_level"]
        num_nodes_list = fwd["num_nodes_list"]

        # Timestep tensor for the model: shape (B,)
        t_tensor = torch.full(
            (batch_size,), t, dtype=torch.long, device=self.device
        )

        # SNR values for Min-SNR loss weighting: shape (B,)
        snr_values = self.noise_schedule.snr(t).expand(batch_size).to(self.device)

        # ---------- forward + loss under mixed precision ----------
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=self.device.type, dtype=self.amp_dtype):
            eps_pred, type_logits = self.model(
                x_t,
                coarsened_types,
                t_tensor,
                resolution_level,
                edge_index,
                edge_attr=None,
                batch=batch_vector,
            )

            # Pad eps tensors from flat (N_total, 3) -> (B, max_N, 3)
            eps_pred_padded, eps_true_padded = self._pad_to_batch(
                eps_pred, epsilon, num_nodes_list, batch_size
            )

            total_loss, loss_dict = self.loss_fn(
                eps_pred_padded,
                eps_true_padded,
                type_logits,
                coarsened_types,
                snr_values,
            )

        # ---------- backward + optimiser ----------
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.ema.update()
        self.scheduler.step()

        self.global_step += 1

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def val_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute one validation step (no gradients). Returns loss dict."""
        self.model.eval()

        T = self.noise_schedule.T
        batch_size = batch["batch_size"]

        t = torch.randint(0, T, (1,)).item()

        fwd = self._forward_diffuse_batch(batch, t)

        x_t = fwd["x_t"]
        epsilon = fwd["epsilon"]
        coarsened_types = fwd["coarsened_types"]
        edge_index = fwd["edge_index"]
        batch_vector = fwd["batch_vector"]
        resolution_level = fwd["resolution_level"]
        num_nodes_list = fwd["num_nodes_list"]

        t_tensor = torch.full(
            (batch_size,), t, dtype=torch.long, device=self.device
        )
        snr_values = self.noise_schedule.snr(t).expand(batch_size).to(self.device)

        with autocast(device_type=self.device.type, dtype=self.amp_dtype):
            eps_pred, type_logits = self.model(
                x_t,
                coarsened_types,
                t_tensor,
                resolution_level,
                edge_index,
                edge_attr=None,
                batch=batch_vector,
            )

            eps_pred_padded, eps_true_padded = self._pad_to_batch(
                eps_pred, epsilon, num_nodes_list, batch_size
            )

            _, loss_dict = self.loss_fn(
                eps_pred_padded,
                eps_true_padded,
                type_logits,
                coarsened_types,
                snr_values,
            )

        return {k: v.item() for k, v in loss_dict.items()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, step: Optional[int] = None) -> None:
        """Save a training checkpoint to *path*."""
        if step is None:
            step = self.global_step
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint at step %d to %s", step, path)

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint and restore all state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.global_step = checkpoint["step"]
        logger.info("Loaded checkpoint from %s (step %d)", path, self.global_step)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def _save_latest(self) -> None:
        """Save a 'latest' checkpoint that is always overwritten."""
        latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.pt")
        self.save_checkpoint(latest_path, step=self.global_step)

    def train(
        self,
        train_loader,
        val_loader=None,
        max_steps: Optional[int] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        """Run the full training loop with logging, validation, and checkpointing.

        Handles ``KeyboardInterrupt`` gracefully by saving a checkpoint
        before exiting, so training can always be resumed with
        ``--resume checkpoints/checkpoint_latest.pt``.

        Args:
            train_loader: DataLoader yielding collated batch dicts.
            val_loader: Optional DataLoader for periodic validation.
            max_steps: Override for ``self.max_steps``.
            resume_from: Path to a checkpoint to resume from.
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)

        max_steps = max_steps or self.max_steps
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info(
            "Starting training from step %d to %d on %s",
            self.global_step, max_steps, self.device,
        )

        train_iter = iter(train_loader)
        running_loss: Dict[str, float] = {}
        running_count = 0
        t_start = time.time()

        try:
            while self.global_step < max_steps:
                # Fetch next batch, cycling through the loader
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                loss_dict = self.train_step(batch)

                for k, v in loss_dict.items():
                    running_loss[k] = running_loss.get(k, 0.0) + v
                running_count += 1

                step = self.global_step

                # --- Logging ---
                if step % self.log_every == 0 and running_count > 0:
                    avg = {k: v / running_count for k, v in running_loss.items()}
                    elapsed = time.time() - t_start
                    steps_per_sec = self.log_every / max(elapsed, 1e-9)
                    lr = self.optimizer.param_groups[0]["lr"]

                    msg = (
                        f"[step {step}]"
                        f"  loss={avg.get('loss_total', 0.0):.4f}"
                        f"  pos={avg.get('loss_pos', 0.0):.4f}"
                        f"  type={avg.get('loss_type', 0.0):.4f}"
                        f"  lr={lr:.2e}"
                        f"  steps/s={steps_per_sec:.1f}"
                    )
                    logger.info(msg)

                    if self.use_wandb:
                        wandb.log({"train/" + k: v for k, v in avg.items()}, step=step)
                        wandb.log({"lr": lr, "steps_per_sec": steps_per_sec}, step=step)

                    running_loss = {}
                    running_count = 0
                    t_start = time.time()

                # --- Validation ---
                if (
                    val_loader is not None
                    and step % self.eval_every == 0
                    and step > 0
                ):
                    val_metrics = self._run_validation(val_loader)
                    msg = (
                        f"[val step {step}]"
                        f"  loss={val_metrics.get('loss_total', 0.0):.4f}"
                        f"  pos={val_metrics.get('loss_pos', 0.0):.4f}"
                        f"  type={val_metrics.get('loss_type', 0.0):.4f}"
                    )
                    logger.info(msg)

                    if self.use_wandb:
                        wandb.log(
                            {"val/" + k: v for k, v in val_metrics.items()}, step=step
                        )

                # --- Checkpoint ---
                if step % self.checkpoint_every == 0 and step > 0:
                    ckpt_path = os.path.join(
                        self.checkpoint_dir, f"checkpoint_{step}.pt"
                    )
                    self.save_checkpoint(ckpt_path, step=step)
                    self._save_latest()

        except KeyboardInterrupt:
            logger.info(
                "Training interrupted at step %d. Saving checkpoint ...",
                self.global_step,
            )
            self._save_latest()
            logger.info(
                "Saved checkpoint_latest.pt. Resume with: "
                "--resume %s/checkpoint_latest.pt",
                self.checkpoint_dir,
            )
            return

        # Final checkpoint
        final_path = os.path.join(self.checkpoint_dir, "checkpoint_final.pt")
        self.save_checkpoint(final_path, step=self.global_step)
        self._save_latest()
        logger.info("Training complete at step %d.", self.global_step)

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def _run_validation(
        self,
        val_loader,
        max_batches: int = 50,
    ) -> Dict[str, float]:
        """Run validation with EMA parameters over at most *max_batches*."""
        accum: Dict[str, float] = {}
        count = 0

        with self.ema.average_parameters():
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                loss_dict = self.val_step(batch)
                for k, v in loss_dict.items():
                    accum[k] = accum.get(k, 0.0) + v
                count += 1

        if count == 0:
            return {}
        return {k: v / count for k, v in accum.items()}
