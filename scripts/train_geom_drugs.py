#!/usr/bin/env python
"""Training entry point for MolSSD on GEOM-Drugs (unconditional generation).

Handles larger molecules (up to 181 atoms) with 10 atom types and deeper
coarsening hierarchies. Supports size-aware dynamic batching to maintain
constant GPU memory usage across varying molecule sizes.

Usage::

    python scripts/train_geom_drugs.py --batch-size 32 --max-steps 500000
    python scripts/train_geom_drugs.py --resume checkpoints_geom/latest.pt
    python scripts/train_geom_drugs.py --max-atoms 80 --batch-size 64  # smaller subset
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from molssd.core.noise_schedules import get_noise_schedule, ResolutionSchedule
from molssd.core.diffusion import MolSSDDiffusion
from molssd.data.geom_drugs_loader import (
    GEOMDrugsMolSSD,
    geom_drugs_collate_fn,
    RandomRotation,
    get_geom_drugs_splits,
    NUM_ATOM_TYPES,
)
from molssd.models.flexi_net import MolecularFlexiNet, DEFAULT_LEVELS
from molssd.training.losses import MolSSDLoss
from molssd.training.trainer import MolSSDTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_geom_drugs")


# ---------------------------------------------------------------------------
# 5-level architecture for GEOM-Drugs (deeper than QM9's 4-level)
# ---------------------------------------------------------------------------

GEOM_DRUGS_LEVELS = [
    {"num_blocks": 4, "hidden_dim": 128},   # Level 0 (finest, full atomic)
    {"num_blocks": 3, "hidden_dim": 256},   # Level 1
    {"num_blocks": 3, "hidden_dim": 384},   # Level 2
    {"num_blocks": 2, "hidden_dim": 512},   # Level 3
    {"num_blocks": 2, "hidden_dim": 512},   # Level 4 (coarsest)
]


# ---------------------------------------------------------------------------
# GPU-cached loader (same as train_qm9.py)
# ---------------------------------------------------------------------------

class GPUCachedLoader:
    """Pre-collates all batches and caches GPU tensors."""

    def __init__(self, dataset, batch_size, collate_fn, device, shuffle=True):
        self.device = device
        self.shuffle = shuffle

        logger.info("Pre-caching %d batches on GPU ...", len(dataset) // batch_size)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0, drop_last=True,
        )

        self.batches = []
        for batch in loader:
            gpu_batch = self._to_device(batch)
            self.batches.append(gpu_batch)

        gpu_mb = torch.cuda.memory_allocated(device) / 1e6
        logger.info("Cached %d batches on GPU (%.0f MB VRAM)", len(self.batches), gpu_mb)

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                out[k] = [
                    {kk: vv.to(self.device, non_blocking=True)
                     if isinstance(vv, torch.Tensor) else vv
                     for kk, vv in d.items()} if d is not None else None
                    for d in v
                ]
            else:
                out[k] = v
        return out

    def __iter__(self):
        indices = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(indices)
        for i in indices:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train MolSSD on GEOM-Drugs for unconditional 3D molecule generation.",
    )

    # Data
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--max-atoms", type=int, default=181,
                    help="Max atoms per molecule (default: 181)")
    p.add_argument("--max-molecules", type=int, default=None,
                    help="Cap dataset size (for dev/debug)")

    # Checkpointing
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints_geom")

    # Training
    p.add_argument("--max-steps", type=int, default=500_000)
    p.add_argument("--batch-size", type=int, default=32,
                    help="Batch size (default: 32, smaller for larger molecules)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-steps", type=int, default=5000)

    # Diffusion
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--noise-schedule", type=str, default="cosine",
                    choices=["cosine", "linear"])

    # Resolution
    p.add_argument("--resolution-schedule", type=str, default="convex_decay")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--max-levels", type=int, default=5,
                    help="Max coarsening levels (default: 5 for drugs)")

    # EMA / optimisation
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Logging
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--checkpoint-every", type=int, default=5000)

    # Device / workers
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpu-cache", action="store_true",
                    help="Pre-load all batches onto GPU")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    # Resume
    p.add_argument("--resume", type=str, default=None)

    # Fine-tune from QM9 checkpoint
    p.add_argument("--pretrained", type=str, default=None,
                    help="Path to a QM9 checkpoint to initialize from (transfer learning)")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="molssd-geom-drugs")

    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Using device: %s", device)
    seed_everything(args.seed)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled cuDNN benchmark + TF32 for %s", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # 1. Datasets
    # ------------------------------------------------------------------
    logger.info("Loading GEOM-Drugs datasets from %s ...", args.data_dir)
    train_ds, val_ds, _ = get_geom_drugs_splits(
        root=args.data_dir,
        max_levels=args.max_levels,
        max_atoms=args.max_atoms,
        max_molecules=args.max_molecules,
        train_transform=RandomRotation(),
    )

    use_cuda = device.type == "cuda"
    if args.gpu_cache and use_cuda:
        train_loader = GPUCachedLoader(
            train_ds, batch_size=args.batch_size,
            collate_fn=geom_drugs_collate_fn, device=device, shuffle=True,
        )
        val_loader = GPUCachedLoader(
            val_ds, batch_size=args.batch_size,
            collate_fn=geom_drugs_collate_fn, device=device, shuffle=False,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=geom_drugs_collate_fn,
            pin_memory=use_cuda, drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=geom_drugs_collate_fn,
            pin_memory=use_cuda,
        )
    logger.info("Train: %d molecules, Val: %d molecules", len(train_ds), len(val_ds))

    # ------------------------------------------------------------------
    # 2. Noise schedule and resolution schedule
    # ------------------------------------------------------------------
    noise_schedule = get_noise_schedule(name=args.noise_schedule, T=args.T)

    # Representative per-level atom counts for GEOM-Drugs
    # ~181 -> ~60 -> ~20 -> ~7 -> ~2 -> 1
    num_atoms_per_level = [181, 60, 20, 7, 2, 1]
    num_levels = args.max_levels + 1
    num_atoms_per_level = num_atoms_per_level[:num_levels]

    resolution_schedule = ResolutionSchedule(
        T=args.T,
        num_levels=num_levels,
        num_atoms_per_level=num_atoms_per_level,
        schedule_type=args.resolution_schedule,
        gamma=args.gamma,
    )

    # ------------------------------------------------------------------
    # 3. Model
    # ------------------------------------------------------------------
    model = MolecularFlexiNet(
        level_configs=GEOM_DRUGS_LEVELS[:num_levels],
        num_atom_types=NUM_ATOM_TYPES,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("MolecularFlexiNet: %d trainable parameters", num_params)

    # Optional: initialize from a QM9-pretrained checkpoint (transfer learning)
    if args.pretrained:
        logger.info("Loading pretrained weights from %s ...", args.pretrained)
        ckpt = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        # Load what we can (shapes may differ for atom embeddings, output heads)
        model_state = model.state_dict()
        pretrained_state = ckpt["model_state_dict"]
        loaded = 0
        for k, v in pretrained_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state)
        logger.info("Loaded %d / %d parameters from pretrained checkpoint", loaded, len(model_state))

    # ------------------------------------------------------------------
    # 4. Diffusion
    # ------------------------------------------------------------------
    diffusion = MolSSDDiffusion(
        noise_schedule=noise_schedule,
        resolution_schedule=resolution_schedule,
        num_atom_types=NUM_ATOM_TYPES,
    )

    # ------------------------------------------------------------------
    # 5. W&B
    # ------------------------------------------------------------------
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            use_wandb = True
            logger.info("W&B run: %s", wandb.run.url)
        except ImportError:
            logger.warning("wandb not installed -- logging disabled")

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer = MolSSDTrainer(
        model=model,
        diffusion=diffusion,
        noise_schedule=noise_schedule,
        resolution_schedule=resolution_schedule,
        loss_fn=MolSSDLoss(lambda_type=0.1, snr_gamma=5.0),
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        ema_decay=args.ema_decay,
        grad_clip_norm=args.grad_clip,
        log_every=args.log_every,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        device=device,
        use_wandb=use_wandb,
        checkpoint_dir=args.checkpoint_dir,
    )

    # ------------------------------------------------------------------
    # 7. Auto-resume
    # ------------------------------------------------------------------
    resume_path = args.resume
    if resume_path is None:
        latest = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(latest):
            resume_path = latest
            logger.info("Auto-resuming from %s", latest)

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=args.max_steps,
        resume_from=resume_path,
    )


if __name__ == "__main__":
    main()
