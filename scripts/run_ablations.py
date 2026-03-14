#!/usr/bin/env python
"""Step A4: Ablation studies for MolSSD.

Runs ablation experiments by training variants and evaluating them.
Each ablation isolates one contribution's impact.

Ablations:
    1. Isotropic vs non-isotropic posterior
    2. Spectral vs random coarsening
    3. 2-level vs 3-level vs 4-level hierarchy
    4. Full Flexi-Net vs fixed-width network

Usage::

    python scripts/run_ablations.py --base-checkpoint checkpoints/checkpoint_final.pt
    python scripts/run_ablations.py --ablation isotropic --max-steps 100000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ablations")

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATIONS = {
    # Ablation 1: Coarsening method (Contribution 1)
    "spectral": {
        "description": "Spectral coarsening (default, our method)",
        "args": [],  # baseline = default settings
    },
    # Ablation 2: Number of hierarchy levels (Contribution 5)
    "levels_2": {
        "description": "2-level hierarchy (full + 1 coarse)",
        "args": ["--max-levels", "1"],
    },
    "levels_3": {
        "description": "3-level hierarchy (full + 2 coarse)",
        "args": ["--max-levels", "2"],
    },
    "levels_4": {
        "description": "4-level hierarchy (full + 3 coarse) [default]",
        "args": ["--max-levels", "3"],
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Run MolSSD ablation studies.")
    p.add_argument("--ablation", type=str, default=None,
                    choices=list(ABLATIONS.keys()) + ["all"],
                    help="Which ablation to run (default: all)")
    p.add_argument("--max-steps", type=int, default=100_000,
                    help="Training steps per ablation (default: 100000)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-eval-molecules", type=int, default=5000,
                    help="Molecules to generate for evaluation (default: 5000)")
    p.add_argument("--output-dir", type=str, default="./ablation_results")
    return p.parse_args()


def run_ablation(name: str, config: dict, args):
    """Train and evaluate one ablation variant."""
    logger.info("=" * 60)
    logger.info("  Ablation: %s", name)
    logger.info("  %s", config["description"])
    logger.info("=" * 60)

    ckpt_dir = os.path.join(args.output_dir, name, "checkpoints")
    eval_dir = os.path.join(args.output_dir, name, "evaluation")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Train
    train_cmd = [
        sys.executable, "scripts/train_qm9.py",
        "--batch-size", str(args.batch_size),
        "--max-steps", str(args.max_steps),
        "--checkpoint-dir", ckpt_dir,
        "--checkpoint-every", "10000",
        "--log-every", "500",
        "--eval-every", "10000",
        "--num-workers", "0",
    ] + config["args"]

    logger.info("Training command: %s", " ".join(train_cmd))
    result = subprocess.run(train_cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Training failed for ablation '%s'", name)
        return None

    # 2. Find best checkpoint
    ckpt_path = os.path.join(ckpt_dir, "checkpoint_latest.pt")
    if not os.path.exists(ckpt_path):
        logger.error("No checkpoint found for ablation '%s'", name)
        return None

    # 3. Evaluate
    eval_cmd = [
        sys.executable, "scripts/run_evaluation.py",
        "--checkpoint", ckpt_path,
        "--num-molecules", str(args.num_eval_molecules),
        "--output-dir", eval_dir,
    ]

    logger.info("Evaluation command: %s", " ".join(eval_cmd))
    result = subprocess.run(eval_cmd, capture_output=False)
    if result.returncode != 0:
        logger.error("Evaluation failed for ablation '%s'", name)
        return None

    # 4. Load metrics
    metrics_path = os.path.join(eval_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def print_ablation_table(results: dict):
    """Print ablation comparison table."""
    header = f"{'Ablation':<25} {'Atom Stab%':>10} {'Mol Stab%':>10} {'Valid%':>8} {'Time(min)':>10}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  Step A4: Ablation Study Results")
    print("=" * len(header))
    print(header)
    print(sep)

    for name, metrics in results.items():
        if metrics is None:
            print(f"{name:<25} {'FAILED':>10}")
            continue
        row = f"{name:<25}"
        row += f" {metrics.get('atom_stability', 0) * 100:>10.1f}"
        row += f" {metrics.get('mol_stability', 0) * 100:>10.1f}"
        row += f" {metrics.get('validity', 0) * 100:>8.1f}"
        t = metrics.get("sampling_time_min")
        row += f" {t:>10.1f}" if t else f" {'—':>10}"
        print(row)

    print("=" * len(header))
    print()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ablation is None or args.ablation == "all":
        ablations_to_run = ABLATIONS
    else:
        ablations_to_run = {args.ablation: ABLATIONS[args.ablation]}

    results = {}
    for name, config in ablations_to_run.items():
        metrics = run_ablation(name, config, args)
        results[name] = metrics

    # Save combined results
    combined_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Combined results saved to %s", combined_path)

    print_ablation_table(results)


if __name__ == "__main__":
    main()
