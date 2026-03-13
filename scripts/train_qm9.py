#!/usr/bin/env python
"""Training entry point for MolSSD on QM9 (unconditional generation).

Parses command-line arguments, builds the dataset / model / diffusion
pipeline, and launches training via the MolSSDTrainer.

Usage::

    python scripts/train_qm9.py --batch-size 64 --max-steps 300000 --wandb
    python scripts/train_qm9.py --resume checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `molssd` is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# MolSSD imports
# ---------------------------------------------------------------------------
from molssd.core.noise_schedules import get_noise_schedule, ResolutionSchedule
from molssd.core.diffusion import MolSSDDiffusion
from molssd.data.qm9_loader import (
    QM9MolSSD,
    qm9_collate_fn,
    RandomRotation,
    get_qm9_splits,
)
from molssd.models.flexi_net import MolecularFlexiNet
from molssd.training.losses import MolSSDLoss
from molssd.training.trainer import MolSSDTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_qm9")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MolSSD on QM9 for unconditional 3D molecule generation.",
    )

    # Data
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root directory for QM9 data (default: ./data)")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints (default: ./checkpoints)")

    # Training hyper-parameters
    parser.add_argument("--max-steps", type=int, default=300_000,
                        help="Total training steps (default: 300000)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate (default: 3e-4)")

    # Diffusion
    parser.add_argument("--T", type=int, default=1000,
                        help="Number of diffusion timesteps (default: 1000)")
    parser.add_argument("--noise-schedule", type=str, default="cosine",
                        choices=["cosine", "linear"],
                        help="Noise schedule type (default: cosine)")

    # Resolution schedule
    parser.add_argument("--resolution-schedule", type=str, default="convex_decay",
                        choices=["convex_decay", "equal", "sigmoid"],
                        help="Resolution schedule type (default: convex_decay)")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Gamma for convex-decay resolution schedule (default: 0.5)")
    parser.add_argument("--max-levels", type=int, default=3,
                        help="Max coarsening levels in the hierarchy (default: 3)")

    # Model
    parser.add_argument("--num-atom-types", type=int, default=5,
                        help="Number of atom type classes (default: 5 for QM9)")

    # EMA / optimisation
    parser.add_argument("--ema-decay", type=float, default=0.9999,
                        help="EMA decay rate (default: 0.9999)")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Max gradient norm for clipping (default: 1.0)")
    parser.add_argument("--warmup-steps", type=int, default=5000,
                        help="LR linear warmup steps (default: 5000)")

    # Logging / evaluation cadence
    parser.add_argument("--log-every", type=int, default=100,
                        help="Log training metrics every N steps (default: 100)")
    parser.add_argument("--eval-every", type=int, default=10_000,
                        help="Run validation every N steps (default: 10000)")
    parser.add_argument("--checkpoint-every", type=int, default=10_000,
                        help="Save checkpoint every N steps (default: 10000)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (default: cuda if available else cpu)")

    # Resumption
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="molssd-qm9",
                        help="W&B project name (default: molssd-qm9)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Seed everything
# ---------------------------------------------------------------------------

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

    # Device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # Reproducibility
    seed_everything(args.seed)

    # GPU optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Enabled cuDNN benchmark + TF32 for %s", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # 1. Datasets and data loaders
    # ------------------------------------------------------------------
    logger.info("Loading QM9 datasets from %s ...", args.data_dir)
    train_ds, val_ds, _ = get_qm9_splits(
        root=args.data_dir,
        max_levels=args.max_levels,
        ratio=3,
        train_transform=RandomRotation(),
    )

    use_cuda = device.type == "cuda"
    num_workers = min(8, os.cpu_count() or 1) if use_cuda else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=qm9_collate_fn,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(num_workers // 2, 1) if num_workers > 0 else 0,
        collate_fn=qm9_collate_fn,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
    )
    logger.info("Train: %d molecules, Val: %d molecules", len(train_ds), len(val_ds))

    # ------------------------------------------------------------------
    # 2. Noise schedule and resolution schedule
    # ------------------------------------------------------------------
    noise_schedule = get_noise_schedule(name=args.noise_schedule, T=args.T)

    # Representative per-level atom counts for QM9 (max ~29 atoms, 3-fold
    # reduction per level):  [29, 10, 3, 1]
    num_atoms_per_level = [29, 10, 3, 1]
    # Trim to max_levels + 1 (including full-resolution level 0)
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
        level_configs=None,  # use DEFAULT_LEVELS
        num_atom_types=args.num_atom_types,
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("MolecularFlexiNet: %d trainable parameters", num_params)

    # ------------------------------------------------------------------
    # 4. Diffusion process
    # ------------------------------------------------------------------
    diffusion = MolSSDDiffusion(
        noise_schedule=noise_schedule,
        resolution_schedule=resolution_schedule,
        num_atom_types=args.num_atom_types,
    )

    # ------------------------------------------------------------------
    # 5. Optional W&B initialisation (before trainer so flag is ready)
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
    # 6. Trainer (creates its own optimizer, scheduler, EMA internally)
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
    # 7. Auto-resume: if no explicit --resume but checkpoint_latest.pt
    #    exists, resume from it automatically.
    # ------------------------------------------------------------------
    resume_path = args.resume
    if resume_path is None:
        latest = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
        if os.path.exists(latest):
            resume_path = latest
            logger.info("Auto-resuming from %s", latest)

    # ------------------------------------------------------------------
    # 8. Launch training (handles epochs, validation, checkpointing)
    # ------------------------------------------------------------------
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_steps=args.max_steps,
        resume_from=resume_path,
    )


if __name__ == "__main__":
    main()
