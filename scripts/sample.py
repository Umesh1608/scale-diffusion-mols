#!/usr/bin/env python
"""Molecule generation (sampling) script for MolSSD.

Loads a trained checkpoint, reconstructs the model and schedules, then
generates molecules via the reverse diffusion process and saves the
results to a ``.pt`` file.

Usage::

    python scripts/sample.py --checkpoint checkpoints/molssd_final.pt --num-molecules 1000
    python scripts/sample.py --checkpoint checkpoints/molssd_final.pt --T-sample 250 --output gen.pt
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `molssd` is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from molssd.core.noise_schedules import get_noise_schedule, ResolutionSchedule
from molssd.core.diffusion import MolSSDDiffusion
from molssd.core.coarsening import CoarseningLevel, build_coarsening_hierarchy
from molssd.core.degradation import DegradationOperator
from molssd.models.flexi_net import MolecularFlexiNet
from molssd.training.ema import ExponentialMovingAverage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sample")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate molecules with a trained MolSSD model.",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a trained model checkpoint (.pt)")
    parser.add_argument("--num-molecules", type=int, default=100,
                        help="Number of molecules to generate (default: 100)")
    parser.add_argument("--output", type=str, default="./generated_molecules.pt",
                        help="Output file path (default: ./generated_molecules.pt)")
    parser.add_argument("--T-sample", type=int, default=250,
                        help="Number of reverse-diffusion steps for sampling (default: 250)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available else cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
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
# Sample atom counts from the QM9 distribution
# ---------------------------------------------------------------------------

# Approximate atom-count distribution for QM9 (from EDM).
# Keys are number of atoms, values are approximate relative frequency.
_QM9_ATOM_COUNT_DIST = {
    5: 0.005, 6: 0.008, 7: 0.015, 8: 0.025, 9: 0.04,
    10: 0.045, 11: 0.055, 12: 0.06, 13: 0.065, 14: 0.07,
    15: 0.075, 16: 0.075, 17: 0.075, 18: 0.07, 19: 0.065,
    20: 0.055, 21: 0.045, 22: 0.035, 23: 0.03, 24: 0.025,
    25: 0.02, 26: 0.015, 27: 0.012, 28: 0.008, 29: 0.007,
}


def sample_atom_counts(num_molecules: int) -> list[int]:
    """Sample atom counts from the approximate QM9 distribution."""
    counts = list(_QM9_ATOM_COUNT_DIST.keys())
    probs = np.array(list(_QM9_ATOM_COUNT_DIST.values()), dtype=np.float64)
    probs /= probs.sum()
    return list(np.random.choice(counts, size=num_molecules, p=probs))


# ---------------------------------------------------------------------------
# Build a dummy fully-connected graph + coarsening hierarchy for sampling
# ---------------------------------------------------------------------------

def build_sampling_hierarchy(
    num_atoms: int,
    max_levels: int = 3,
    ratio: int = 3,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, list[CoarseningLevel]]:
    """Create a fully-connected adjacency and coarsening hierarchy.

    During sampling we do not have ground-truth bonds, so we use a
    fully-connected graph (self-loops excluded). The coarsening hierarchy
    is built on top of this graph using the standard spectral coarsening
    routine.

    Returns:
        Tuple of (adj, hierarchy) where adj is ``(N, N)`` and hierarchy
        is a list of CoarseningLevel objects.
    """
    adj = torch.ones(num_atoms, num_atoms, device=device) - torch.eye(num_atoms, device=device)
    # Uniform masses (we don't know atom types yet)
    masses = torch.ones(num_atoms, device=device)

    try:
        hierarchy = build_coarsening_hierarchy(
            adj=adj,
            num_atoms=num_atoms,
            target_sizes=None,
            atomic_masses=masses,
        )
    except Exception:
        hierarchy = []

    if len(hierarchy) > max_levels:
        hierarchy = hierarchy[:max_levels]

    return adj, hierarchy


# ---------------------------------------------------------------------------
# Reverse diffusion sampling loop (single molecule)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_single_molecule(
    model: MolecularFlexiNet,
    diffusion: MolSSDDiffusion,
    noise_schedule,
    resolution_schedule: ResolutionSchedule,
    num_atoms: int,
    T_sample: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a single molecule via the full reverse diffusion process.

    Args:
        model: Trained denoising network.
        diffusion: MolSSDDiffusion instance.
        noise_schedule: Noise schedule for alpha_bar / sigma.
        resolution_schedule: Resolution schedule mapping t -> level.
        num_atoms: Number of atoms in the generated molecule.
        T_sample: Number of reverse steps to use.
        device: Computation device.

    Returns:
        Tuple of (positions, atom_types) where positions is ``(N, 3)``
        and atom_types is ``(N,)`` of predicted integer type indices.
    """
    model.eval()

    # Build adjacency + coarsening hierarchy for this molecule size
    adj, hierarchy = build_sampling_hierarchy(num_atoms, device=device)

    # Determine the coarsest resolution at t = T_sample - 1
    r_max = resolution_schedule.resolution_level(T_sample - 1).item()
    r_max = min(r_max, len(hierarchy))

    # Figure out how many nodes at the starting (coarsest) resolution
    if r_max == 0 or len(hierarchy) == 0:
        n_start = num_atoms
    else:
        # Compose coarsening to get the number of super-nodes at level r_max
        C = hierarchy[0].coarsening_matrix
        for k in range(1, min(r_max, len(hierarchy))):
            C = hierarchy[k].coarsening_matrix @ C
        n_start = C.shape[0]

    # Start from pure noise at the coarsest resolution
    x_t = torch.randn(n_start, 3, device=device)

    # Random initial atom types (will be refined during denoising)
    atom_types = torch.randint(0, diffusion.num_atom_types, (n_start,), device=device)

    # Build edge index for the starting resolution
    if r_max > 0 and len(hierarchy) >= r_max:
        coarsened_adj = hierarchy[r_max - 1].coarsened_adj.to(device)
    else:
        coarsened_adj = adj.to(device)
    edge_index = (coarsened_adj > 0).nonzero(as_tuple=False).t().contiguous()

    # Map sampling steps to diffusion timesteps (uniform sub-sampling)
    timesteps = torch.linspace(T_sample - 1, 0, T_sample).long()

    for i, t_val in enumerate(timesteps):
        t = int(t_val.item())
        r_t = resolution_schedule.resolution_level(t).item()
        r_t = min(r_t, len(hierarchy))

        # Build per-graph time tensor (batch size 1)
        t_tensor = torch.tensor([t], dtype=torch.float32, device=device)

        # Model forward: predict noise and type logits
        eps_pred, type_logits = model(
            x_t=x_t,
            atom_types=atom_types,
            t=t_tensor,
            resolution_level=r_t,
            edge_index=edge_index,
            edge_attr=None,
            batch=None,
        )

        # Update atom type predictions from logits
        atom_types = type_logits.argmax(dim=-1)

        # Simple DDPM reverse step (isotropic)
        if t > 0:
            alpha_bar_t = noise_schedule.alpha_bar(t).to(device)
            sigma_t = noise_schedule.sigma(t).to(device)

            t_prev = int(timesteps[min(i + 1, len(timesteps) - 1)].item())
            alpha_bar_prev = noise_schedule.alpha_bar(t_prev).to(device)
            sigma_prev = noise_schedule.sigma(t_prev).to(device)

            # Predict x0 from epsilon prediction
            x0_hat = (x_t - sigma_t * eps_pred) / alpha_bar_t.clamp(min=1e-8)

            # DDPM posterior mean
            beta_t = noise_schedule.beta(t).to(device)
            sigma_sq_t = noise_schedule.sigma_squared(t).to(device)
            sigma_sq_prev = noise_schedule.sigma_squared(t_prev).to(device)

            coeff_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / sigma_sq_t.clamp(min=1e-8)
            coeff_xt = (torch.sqrt(alpha_bar_t / alpha_bar_prev.clamp(min=1e-8))
                        * sigma_sq_prev) / sigma_sq_t.clamp(min=1e-8)

            mu = coeff_x0 * x0_hat + coeff_xt * x_t

            # Posterior variance
            beta_tilde = (beta_t * sigma_sq_prev) / sigma_sq_t.clamp(min=1e-8)
            sigma_post = torch.sqrt(beta_tilde.clamp(min=1e-12))

            z = torch.randn_like(x_t) if t_prev > 0 else torch.zeros_like(x_t)
            x_t = mu + sigma_post * z

            # Handle resolution change: if the next step is at a finer
            # resolution, lift x_t back to that finer resolution.
            if t_prev > 0:
                r_prev = resolution_schedule.resolution_level(t_prev).item()
                r_prev = min(r_prev, len(hierarchy))
                if r_prev < r_t and r_t > 0 and len(hierarchy) > 0:
                    # Lift from coarser to finer via pseudoinverse
                    level = hierarchy[min(r_t - 1, len(hierarchy) - 1)]
                    C = level.coarsening_matrix.to(device)
                    C_pinv = torch.linalg.pinv(C)
                    x_t = C_pinv @ x_t
                    # Expand atom types correspondingly
                    atom_types = atom_types[level.cluster_assignment.to(device)]
                    # Rebuild edge index for the finer resolution
                    if r_prev > 0 and r_prev <= len(hierarchy):
                        fine_adj = hierarchy[r_prev - 1].coarsened_adj.to(device)
                    else:
                        fine_adj = adj.to(device)
                    edge_index = (fine_adj > 0).nonzero(as_tuple=False).t().contiguous()
        else:
            # t == 0: use the clean prediction directly
            x0_hat = (x_t - noise_schedule.sigma(0).to(device) * eps_pred) / (
                noise_schedule.alpha_bar(0).to(device).clamp(min=1e-8)
            )
            x_t = x0_hat

    # If we ended at a coarsened resolution, lift to full atomic resolution
    current_n = x_t.shape[0]
    if current_n < num_atoms and len(hierarchy) > 0:
        # Compose coarsening matrices and use pseudoinverse to lift
        for k in range(len(hierarchy) - 1, -1, -1):
            if x_t.shape[0] >= num_atoms:
                break
            C = hierarchy[k].coarsening_matrix.to(device)
            if C.shape[0] == x_t.shape[0]:
                C_pinv = torch.linalg.pinv(C)
                x_t = C_pinv @ x_t
                atom_types = atom_types[hierarchy[k].cluster_assignment.to(device)]

    # Ensure we have exactly num_atoms
    if x_t.shape[0] != num_atoms:
        # Truncate or pad to match
        if x_t.shape[0] > num_atoms:
            x_t = x_t[:num_atoms]
            atom_types = atom_types[:num_atoms]
        else:
            pad_n = num_atoms - x_t.shape[0]
            x_t = torch.cat([x_t, torch.randn(pad_n, 3, device=device)], dim=0)
            atom_types = torch.cat([
                atom_types,
                torch.randint(0, diffusion.num_atom_types, (pad_n,), device=device),
            ])

    # Center the molecule (subtract center of mass)
    x_t = x_t - x_t.mean(dim=0, keepdim=True)

    return x_t.cpu(), atom_types.cpu()


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

    seed_everything(args.seed)

    # ------------------------------------------------------------------
    # 1. Load checkpoint
    # ------------------------------------------------------------------
    logger.info("Loading checkpoint: %s", args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Extract config from checkpoint if available
    ckpt_config = checkpoint.get("config", None)
    T_train = getattr(ckpt_config, "T", 1000) if ckpt_config is not None else 1000
    num_atom_types = getattr(ckpt_config, "num_atom_types", 5) if ckpt_config is not None else 5

    # ------------------------------------------------------------------
    # 2. Reconstruct model and schedules
    # ------------------------------------------------------------------
    noise_schedule = get_noise_schedule(name="cosine", T=T_train)

    num_atoms_per_level = [29, 10, 3, 1]
    resolution_schedule = ResolutionSchedule(
        T=T_train,
        num_levels=4,
        num_atoms_per_level=num_atoms_per_level,
        schedule_type="convex_decay",
        gamma=0.5,
    )

    diffusion = MolSSDDiffusion(
        noise_schedule=noise_schedule,
        resolution_schedule=resolution_schedule,
        num_atom_types=num_atom_types,
    ).to(device)

    model = MolecularFlexiNet(
        level_configs=None,  # DEFAULT_LEVELS
        num_atom_types=num_atom_types,
    ).to(device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # If EMA state is available, apply it for better generation quality
    if "ema_state_dict" in checkpoint:
        ema = ExponentialMovingAverage(model)
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema.apply_shadow()
        logger.info("Applied EMA parameters for sampling")

    model.eval()

    # ------------------------------------------------------------------
    # 3. Sample molecules
    # ------------------------------------------------------------------
    num_atoms_list = sample_atom_counts(args.num_molecules)
    logger.info(
        "Generating %d molecules (atom counts %d-%d, T_sample=%d) ...",
        args.num_molecules,
        min(num_atoms_list),
        max(num_atoms_list),
        args.T_sample,
    )

    # Clamp T_sample to not exceed training T
    T_sample = min(args.T_sample, T_train)

    all_positions = []
    all_atom_types = []

    for i in range(args.num_molecules):
        n_atoms = num_atoms_list[i]
        positions, atom_types = sample_single_molecule(
            model=model,
            diffusion=diffusion,
            noise_schedule=noise_schedule,
            resolution_schedule=resolution_schedule,
            num_atoms=n_atoms,
            T_sample=T_sample,
            device=device,
        )
        all_positions.append(positions)
        all_atom_types.append(atom_types)

        if (i + 1) % 10 == 0 or (i + 1) == args.num_molecules:
            logger.info("  Generated %d / %d molecules", i + 1, args.num_molecules)

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "positions": all_positions,
            "atom_types": all_atom_types,
            "num_atoms_list": num_atoms_list,
            "num_molecules": args.num_molecules,
            "T_sample": T_sample,
            "checkpoint": args.checkpoint,
            "seed": args.seed,
        },
        str(output_path),
    )
    logger.info("Saved %d generated molecules to %s", args.num_molecules, output_path)


if __name__ == "__main__":
    main()
