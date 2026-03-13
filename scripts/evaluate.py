#!/usr/bin/env python
"""Evaluation script for MolSSD-generated molecules.

Loads generated molecules from a ``.pt`` file, loads reference molecules
from the QM9 test set, and computes the standard EDM evaluation metrics:
atom stability, molecule stability, validity, uniqueness, novelty, and
bond-length / bond-angle Jensen-Shannon divergences.

Usage::

    python scripts/evaluate.py --generated generated_molecules.pt
    python scripts/evaluate.py --generated gen.pt --data-dir ./data --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure the project root is on sys.path so `molssd` is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from molssd.data.qm9_loader import QM9MolSSD
from molssd.evaluation.metrics import (
    build_molecule,
    bond_angle_js_divergence,
    bond_length_js_divergence,
    compute_generation_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated molecules against QM9 reference set.",
    )
    parser.add_argument("--generated", type=str, required=True,
                        help="Path to generated molecules .pt file (required)")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Root directory for QM9 data (default: ./data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save metrics as JSON")
    parser.add_argument("--num-reference", type=int, default=10_000,
                        help="Number of reference molecules from the test set (default: 10000)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_generated_molecules(
    path: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load generated molecules from a .pt file.

    Expects a dict with keys ``'positions'`` (list of tensors) and
    ``'atom_types'`` (list of tensors).

    Returns:
        Tuple of (positions_list, types_list) as numpy arrays.
    """
    data = torch.load(path, map_location="cpu", weights_only=False)

    positions_list: list[np.ndarray] = []
    types_list: list[np.ndarray] = []

    raw_positions = data["positions"]
    raw_types = data["atom_types"]

    for pos, types in zip(raw_positions, raw_types):
        if isinstance(pos, torch.Tensor):
            pos = pos.numpy()
        if isinstance(types, torch.Tensor):
            types = types.numpy()
        positions_list.append(np.asarray(pos, dtype=np.float64))
        types_list.append(np.asarray(types, dtype=np.int64))

    return positions_list, types_list


def load_reference_molecules(
    data_dir: str,
    num_reference: int,
) -> tuple[list[np.ndarray], list[np.ndarray], set[str]]:
    """Load reference molecules from the QM9 test set.

    Also extracts training-set SMILES for novelty computation.

    Returns:
        Tuple of (ref_positions, ref_types, training_smiles).
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("RDKit not available; some metrics will be skipped.")
        Chem = None

    # Load QM9 test split
    logger.info("Loading QM9 test split from %s ...", data_dir)
    test_ds = QM9MolSSD(root=data_dir, split="test", transform=None)

    num_ref = min(num_reference, len(test_ds))
    logger.info("Using %d reference molecules from the test set.", num_ref)

    ref_positions: list[np.ndarray] = []
    ref_types: list[np.ndarray] = []

    for i in range(num_ref):
        sample = test_ds[i]
        ref_positions.append(sample["positions"].numpy().astype(np.float64))
        ref_types.append(sample["atom_types"].numpy().astype(np.int64))

    # Extract training SMILES for novelty computation
    training_smiles: set[str] = set()
    if Chem is not None:
        logger.info("Building training SMILES set for novelty computation ...")
        train_ds = QM9MolSSD(root=data_dir, split="train", transform=None)
        for i in range(len(train_ds)):
            sample = train_ds[i]
            mol = build_molecule(
                sample["positions"].numpy().astype(np.float64),
                sample["atom_types"].numpy().astype(np.int64),
            )
            if mol is not None:
                try:
                    smi = Chem.MolToSmiles(mol)
                    if smi:
                        training_smiles.add(smi)
                except Exception:
                    pass
        logger.info("Training set: %d unique SMILES", len(training_smiles))

    return ref_positions, ref_types, training_smiles


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_results(metrics: dict[str, float]) -> None:
    """Print evaluation metrics in a readable table."""
    print()
    print("=" * 60)
    print("  MolSSD Evaluation Results")
    print("=" * 60)
    print()

    # Generation summary
    print(f"  Generated molecules:     {metrics.get('num_generated', 'N/A')}")
    print(f"  Valid molecules:         {metrics.get('num_valid', 'N/A')}")
    print(f"  Unique valid molecules:  {metrics.get('num_unique', 'N/A')}")
    print()

    # Core metrics
    print("  --- Core Metrics ---")
    fmt = "  {:<25s} {:.4f}"
    for key in ["atom_stability", "mol_stability", "validity", "uniqueness", "novelty"]:
        val = metrics.get(key)
        if val is not None:
            if np.isnan(val):
                print(f"  {key:<25s} N/A (no training SMILES)")
            else:
                print(fmt.format(key, val))

    # Distributional metrics
    print()
    print("  --- Distributional Metrics ---")
    for key in ["bond_length_jsd", "bond_angle_jsd"]:
        val = metrics.get(key)
        if val is not None:
            if np.isnan(val):
                print(f"  {key:<25s} N/A")
            else:
                print(fmt.format(key, val))

    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load generated molecules
    # ------------------------------------------------------------------
    logger.info("Loading generated molecules from %s ...", args.generated)
    gen_positions, gen_types = load_generated_molecules(args.generated)
    logger.info("Loaded %d generated molecules.", len(gen_positions))

    # ------------------------------------------------------------------
    # 2. Load reference molecules
    # ------------------------------------------------------------------
    ref_positions, ref_types, training_smiles = load_reference_molecules(
        data_dir=args.data_dir,
        num_reference=args.num_reference,
    )

    # ------------------------------------------------------------------
    # 3. Compute core generation metrics
    # ------------------------------------------------------------------
    logger.info("Computing generation quality metrics ...")
    metrics = compute_generation_metrics(
        generated_positions=gen_positions,
        generated_types=gen_types,
        training_smiles=training_smiles if len(training_smiles) > 0 else None,
    )

    # ------------------------------------------------------------------
    # 4. Compute distributional metrics (bond length / angle JSD)
    # ------------------------------------------------------------------
    logger.info("Building RDKit molecules for distributional metrics ...")
    gen_mols = [
        build_molecule(pos, types)
        for pos, types in zip(gen_positions, gen_types)
    ]
    ref_mols = [
        build_molecule(pos, types)
        for pos, types in zip(ref_positions, ref_types)
    ]

    # Filter out None entries
    gen_mols_valid = [m for m in gen_mols if m is not None]
    ref_mols_valid = [m for m in ref_mols if m is not None]

    logger.info(
        "Built %d / %d generated mols, %d / %d reference mols for distributional metrics.",
        len(gen_mols_valid), len(gen_mols),
        len(ref_mols_valid), len(ref_mols),
    )

    bl_jsd = bond_length_js_divergence(gen_mols_valid, ref_mols_valid)
    ba_jsd = bond_angle_js_divergence(gen_mols_valid, ref_mols_valid)

    metrics["bond_length_jsd"] = bl_jsd
    metrics["bond_angle_jsd"] = ba_jsd

    # ------------------------------------------------------------------
    # 5. Print results
    # ------------------------------------------------------------------
    print_results(metrics)

    # ------------------------------------------------------------------
    # 6. Optionally save to JSON
    # ------------------------------------------------------------------
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert any numpy/tensor values to plain Python types for JSON
        json_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, float)):
                json_metrics[k] = float(v) if not np.isnan(v) else None
            elif isinstance(v, (np.integer, int)):
                json_metrics[k] = int(v)
            else:
                json_metrics[k] = v

        with open(output_path, "w") as f:
            json.dump(json_metrics, f, indent=2)
        logger.info("Metrics saved to %s", output_path)


if __name__ == "__main__":
    main()
