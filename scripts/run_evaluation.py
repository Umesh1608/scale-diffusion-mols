#!/usr/bin/env python
"""One-command evaluation for Step A3: sample molecules and compare with baselines.

Usage::

    python scripts/run_evaluation.py --checkpoint checkpoints/checkpoint_final.pt
    python scripts/run_evaluation.py --checkpoint checkpoints/checkpoint_300000.pt --num-molecules 10000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from molssd.core.noise_schedules import get_noise_schedule, ResolutionSchedule
from molssd.core.diffusion import MolSSDDiffusion
from molssd.core.coarsening import build_coarsening_hierarchy
from molssd.core.degradation import DegradationOperator
from molssd.models.flexi_net import MolecularFlexiNet
from molssd.training.ema import ExponentialMovingAverage
from molssd.evaluation.metrics import (
    build_molecule,
    compute_generation_metrics,
    bond_length_js_divergence,
    bond_angle_js_divergence,
)
from molssd.evaluation.batched_sampling import sample_molecules_batched
from molssd.data.qm9_loader import QM9MolSSD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_evaluation")

# ---------------------------------------------------------------------------
# Baseline results from published papers
# ---------------------------------------------------------------------------

BASELINES = {
    "EDM": {
        "atom_stability": 98.7, "mol_stability": 82.0,
        "validity": 98.7, "uniqueness": 99.5,
        "sampling_time_min": 28.4,
    },
    "GeoLDM": {
        "atom_stability": 98.9, "mol_stability": 89.4,
        "validity": 93.8, "uniqueness": 99.4,
    },
    "MiDi": {
        "atom_stability": 99.0, "mol_stability": 83.3,
        "validity": 99.0,
    },
    "GCDM": {
        "atom_stability": 99.0, "mol_stability": 85.7,
    },
    "EQGAT-diff": {
        "atom_stability": 98.7, "mol_stability": 81.3,
        "validity": 98.5, "uniqueness": 99.4,
    },
    "MolDiff": {
        "atom_stability": 98.2, "mol_stability": 78.5,
        "validity": 96.1, "uniqueness": 99.3,
    },
}


# Approximate atom-count distribution for QM9 (from EDM)
_QM9_ATOM_COUNT_DIST = {
    5: 0.005, 6: 0.008, 7: 0.015, 8: 0.025, 9: 0.04,
    10: 0.045, 11: 0.055, 12: 0.06, 13: 0.065, 14: 0.07,
    15: 0.075, 16: 0.075, 17: 0.075, 18: 0.07, 19: 0.065,
    20: 0.055, 21: 0.045, 22: 0.035, 23: 0.03, 24: 0.025,
    25: 0.02, 26: 0.015, 27: 0.012, 28: 0.008, 29: 0.007,
}


def sample_atom_counts(n: int) -> list[int]:
    counts = list(_QM9_ATOM_COUNT_DIST.keys())
    probs = np.array(list(_QM9_ATOM_COUNT_DIST.values()), dtype=np.float64)
    probs /= probs.sum()
    return list(np.random.choice(counts, size=n, p=probs))


def parse_args():
    p = argparse.ArgumentParser(description="Full MolSSD evaluation pipeline.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-molecules", type=int, default=10_000)
    p.add_argument("--T-sample", type=int, default=1000,
                    help="Sampling steps (default: 1000 = full)")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./evaluation_results")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_comparison_table(molssd_metrics: dict):
    """Print the comparison table (Step A3)."""
    header = f"{'Model':<15} {'Atom Stab%':>10} {'Mol Stab%':>10} {'Valid%':>8} {'Unique%':>8} {'Time(min)':>10}"
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("  Step A3: QM9 Benchmark Comparison")
    print("=" * len(header))
    print(header)
    print(sep)

    for name, metrics in BASELINES.items():
        row = f"{name:<15}"
        row += f" {metrics.get('atom_stability', '-'):>10}"
        row += f" {metrics.get('mol_stability', '-'):>10}"
        row += f" {metrics.get('validity', '-'):>8}"
        row += f" {metrics.get('uniqueness', '-'):>8}"
        t = metrics.get("sampling_time_min")
        row += f" {t:>10.1f}" if t else f" {'—':>10}"
        print(row)

    print(sep)
    m = molssd_metrics
    row = f"{'MolSSD (ours)':<15}"
    row += f" {m.get('atom_stability', 0) * 100:>10.1f}"
    row += f" {m.get('mol_stability', 0) * 100:>10.1f}"
    row += f" {m.get('validity', 0) * 100:>8.1f}"
    row += f" {m.get('uniqueness', 0) * 100:>8.1f}"
    t = m.get("sampling_time_min")
    row += f" {t:>10.1f}" if t else f" {'—':>10}"
    print(row)
    print("=" * len(header))
    print()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    logger.info("Device: %s", device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model from checkpoint
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    step = ckpt.get("step", "?")
    logger.info("Checkpoint step: %s", step)

    noise_schedule = get_noise_schedule(name="cosine", T=1000)
    resolution_schedule = ResolutionSchedule(
        T=1000, num_levels=4, num_atoms_per_level=[29, 10, 3, 1],
        schedule_type="convex_decay", gamma=0.5,
    )
    diffusion = MolSSDDiffusion(noise_schedule, resolution_schedule, num_atom_types=5).to(device)
    model = MolecularFlexiNet(num_atom_types=5).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    if "ema_state_dict" in ckpt:
        ema = ExponentialMovingAverage(model)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply_shadow()
        logger.info("Applied EMA parameters")

    model.eval()

    # ------------------------------------------------------------------
    # 2. Sample molecules (timed)
    # ------------------------------------------------------------------
    num_atoms_list = sample_atom_counts(args.num_molecules)
    logger.info(
        "Generating %d molecules (T_sample=%d) ...",
        args.num_molecules, args.T_sample,
    )

    t_start = time.time()
    results = sample_molecules_batched(
        model=model,
        noise_schedule=noise_schedule,
        resolution_schedule=resolution_schedule,
        num_atoms_list=num_atoms_list,
        num_atom_types=5,
        device=device,
        T_sample=args.T_sample,
        batch_size=64,
    )
    sampling_time = time.time() - t_start
    sampling_time_min = sampling_time / 60.0
    logger.info(
        "Generated %d molecules in %.1f sec (%.1f min)",
        len(results), sampling_time, sampling_time_min,
    )

    # Save generated molecules
    gen_path = os.path.join(args.output_dir, "generated_molecules.pt")
    gen_positions = [r["positions"].cpu() for r in results]
    gen_types = [r["atom_types"].cpu() for r in results]
    torch.save({"positions": gen_positions, "atom_types": gen_types}, gen_path)
    logger.info("Saved generated molecules to %s", gen_path)

    # ------------------------------------------------------------------
    # 3. Compute metrics
    # ------------------------------------------------------------------
    logger.info("Computing generation quality metrics ...")
    gen_pos_np = [p.numpy().astype(np.float64) for p in gen_positions]
    gen_types_np = [t.numpy().astype(np.int64) for t in gen_types]

    # Build training SMILES for novelty
    logger.info("Loading training set for novelty computation ...")
    train_ds = QM9MolSSD(root=args.data_dir, split="train", transform=None)
    training_smiles = set()
    try:
        from rdkit import Chem
        for i in range(len(train_ds)):
            sample = train_ds[i]
            mol = build_molecule(
                sample["positions"].numpy().astype(np.float64),
                sample["atom_types"].numpy().astype(np.int64),
            )
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    training_smiles.add(smi)
        logger.info("Training set: %d unique SMILES", len(training_smiles))
    except Exception as e:
        logger.warning("Could not build training SMILES: %s", e)

    metrics = compute_generation_metrics(
        generated_positions=gen_pos_np,
        generated_types=gen_types_np,
        training_smiles=training_smiles if training_smiles else None,
    )

    # Distributional metrics
    logger.info("Computing distributional metrics ...")
    test_ds = QM9MolSSD(root=args.data_dir, split="test", transform=None)
    ref_mols = []
    for i in range(min(10000, len(test_ds))):
        s = test_ds[i]
        m = build_molecule(
            s["positions"].numpy().astype(np.float64),
            s["atom_types"].numpy().astype(np.int64),
        )
        if m is not None:
            ref_mols.append(m)

    gen_mols = [
        build_molecule(p, t) for p, t in zip(gen_pos_np, gen_types_np)
    ]
    gen_mols_valid = [m for m in gen_mols if m is not None]

    metrics["bond_length_jsd"] = bond_length_js_divergence(gen_mols_valid, ref_mols)
    metrics["bond_angle_jsd"] = bond_angle_js_divergence(gen_mols_valid, ref_mols)
    metrics["sampling_time_sec"] = sampling_time
    metrics["sampling_time_min"] = sampling_time_min
    metrics["checkpoint_step"] = step
    metrics["num_generated"] = args.num_molecules
    metrics["T_sample"] = args.T_sample

    # ------------------------------------------------------------------
    # 4. Save and print results
    # ------------------------------------------------------------------
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    json_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            json_metrics[k] = float(v) if not np.isnan(v) else None
        elif isinstance(v, (np.integer, int)):
            json_metrics[k] = int(v)
        else:
            json_metrics[k] = v
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    print_comparison_table(metrics)


if __name__ == "__main__":
    main()
