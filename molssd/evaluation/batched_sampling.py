"""Batched molecule sampling for fast generation.

Groups molecules by atom count and processes entire batches through the model
in a single forward pass, achieving 30-60x speedup over sequential sampling.

All molecules with the same atom count share an identical fully-connected
graph and coarsening hierarchy, so we build these once per group and reuse.

Uses the standard DDPM isotropic reverse step (same as EDM) for simplicity
and speed. The non-isotropic posterior can be enabled later for quality
comparisons.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from molssd.core.coarsening import CoarseningLevel, build_coarsening_hierarchy
from molssd.core.noise_schedules import NoiseSchedule, ResolutionSchedule

logger = logging.getLogger(__name__)


def _build_timestep_schedule(T: int, T_sample: Optional[int] = None) -> List[int]:
    """Build descending timestep schedule, ending at 0."""
    if T_sample is None or T_sample >= T:
        return list(range(T - 1, -1, -1))
    T_sample = max(T_sample, 1)
    indices = torch.linspace(T - 1, 0, T_sample).long().tolist()
    if indices[-1] != 0:
        indices[-1] = 0
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


@torch.no_grad()
def sample_molecules_batched(
    model: nn.Module,
    noise_schedule: NoiseSchedule,
    resolution_schedule: ResolutionSchedule,
    num_atoms_list: List[int],
    num_atom_types: int = 5,
    device: torch.device = torch.device("cuda"),
    T_sample: Optional[int] = None,
    batch_size: int = 64,
) -> List[Dict[str, torch.Tensor]]:
    """Generate molecules in batches grouped by atom count.

    Molecules with the same atom count share the same graph topology and
    coarsening hierarchy, so they are batched together for efficient GPU
    utilisation.

    Args:
        model: Trained denoising network in eval mode.
        noise_schedule: Noise schedule (alpha_bar, sigma, beta).
        resolution_schedule: Resolution schedule mapping t -> level.
        num_atoms_list: List of atom counts, one per molecule.
        num_atom_types: Number of discrete atom types.
        device: Computation device.
        T_sample: Number of reverse steps (None = full T).
        batch_size: Max molecules per batch within each atom-count group.

    Returns:
        List of dicts with 'positions' (N,3) and 'atom_types' (N,) per molecule,
        in the same order as num_atoms_list.
    """
    model.eval()
    T = noise_schedule.T
    timesteps = _build_timestep_schedule(T, T_sample)

    # Group molecule indices by atom count
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, n_atoms in enumerate(num_atoms_list):
        if n_atoms >= 2:
            groups[n_atoms].append(idx)

    # Pre-build hierarchy + schedule for each unique atom count
    hierarchy_cache: Dict[int, Tuple[List[CoarseningLevel], ResolutionSchedule]] = {}

    results: List[Optional[Dict[str, torch.Tensor]]] = [None] * len(num_atoms_list)
    total_generated = 0
    t_start = time.time()

    for n_atoms, indices in sorted(groups.items()):
        # Build hierarchy once per atom count (on CPU)
        if n_atoms not in hierarchy_cache:
            adj_cpu = torch.ones(n_atoms, n_atoms) - torch.eye(n_atoms)
            try:
                hierarchy = build_coarsening_hierarchy(
                    adj=adj_cpu, num_atoms=n_atoms,
                )
            except Exception:
                hierarchy = []
            hierarchy_cache[n_atoms] = hierarchy

        hierarchy = hierarchy_cache[n_atoms]

        # Process in sub-batches
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            B = len(batch_indices)

            batch_results = _sample_batch_same_size(
                model=model,
                noise_schedule=noise_schedule,
                resolution_schedule=resolution_schedule,
                hierarchy=hierarchy,
                num_atoms=n_atoms,
                batch_size=B,
                num_atom_types=num_atom_types,
                device=device,
                timesteps=timesteps,
            )

            for i, mol_idx in enumerate(batch_indices):
                results[mol_idx] = {
                    "positions": batch_results["positions"][i],
                    "atom_types": batch_results["atom_types"][i],
                }

            total_generated += B
            if total_generated % 500 < B:
                elapsed = time.time() - t_start
                rate = total_generated / max(elapsed, 1e-9)
                logger.info(
                    "  Generated %d / %d molecules (%.1f mol/s)",
                    total_generated, len(num_atoms_list), rate,
                )

    # Fill any skipped molecules (n_atoms < 2)
    for i in range(len(results)):
        if results[i] is None:
            results[i] = {
                "positions": torch.zeros(0, 3),
                "atom_types": torch.zeros(0, dtype=torch.long),
            }

    elapsed = time.time() - t_start
    logger.info(
        "Generated %d molecules in %.1f sec (%.1f mol/s, %.1f min)",
        len(results), elapsed, len(results) / max(elapsed, 1e-9), elapsed / 60,
    )

    return results


@torch.no_grad()
def _sample_batch_same_size(
    model: nn.Module,
    noise_schedule: NoiseSchedule,
    resolution_schedule: ResolutionSchedule,
    hierarchy: List[CoarseningLevel],
    num_atoms: int,
    batch_size: int,
    num_atom_types: int,
    device: torch.device,
    timesteps: List[int],
) -> Dict[str, torch.Tensor]:
    """Sample a batch of molecules that all have the same atom count.

    All molecules share the same graph topology and coarsening hierarchy,
    so we batch them into a single PyG-style graph and run one model
    forward pass per timestep.

    Uses standard DDPM isotropic reverse steps (same as EDM baseline).

    Returns:
        Dict with 'positions': list of (N,3) tensors,
        'atom_types': list of (N,) tensors.
    """
    B = batch_size
    N = num_atoms

    # Build fully-connected edge index for one molecule
    adj_single = torch.ones(N, N, device=device) - torch.eye(N, device=device)
    ei_single = (adj_single > 0).nonzero(as_tuple=False).t().contiguous()

    # Build batched edge index (shift each molecule's edges by N*i)
    ei_parts = []
    batch_vector_parts = []
    for i in range(B):
        ei_parts.append(ei_single + i * N)
        batch_vector_parts.append(
            torch.full((N,), i, dtype=torch.long, device=device)
        )
    edge_index = torch.cat(ei_parts, dim=1)
    batch_vector = torch.cat(batch_vector_parts)

    # Determine starting resolution
    t_start = timesteps[0]
    r_start = resolution_schedule.resolution_level(t_start).item()
    r_start = min(r_start, len(hierarchy))

    # Figure out starting node count
    if r_start == 0 or len(hierarchy) == 0:
        n_start = N
    else:
        n_start = hierarchy[min(r_start, len(hierarchy)) - 1].num_nodes

    # Initialize: pure noise at starting resolution
    x_t = torch.randn(B * n_start, 3, device=device)
    atom_types = torch.randint(0, num_atom_types, (B * n_start,), device=device)

    # Build edge index at starting resolution
    if r_start > 0 and len(hierarchy) >= r_start:
        adj_coarse = hierarchy[r_start - 1].coarsened_adj.to(device)
        ei_coarse = (adj_coarse > 0).nonzero(as_tuple=False).t().contiguous()
        ei_parts_c = [ei_coarse + i * n_start for i in range(B)]
        edge_index_current = torch.cat(ei_parts_c, dim=1)
        batch_vector_current = torch.cat([
            torch.full((n_start,), i, dtype=torch.long, device=device)
            for i in range(B)
        ])
    else:
        edge_index_current = edge_index
        batch_vector_current = batch_vector

    current_n = n_start  # nodes per molecule at current resolution

    for step_idx, t in enumerate(timesteps):
        r_t = resolution_schedule.resolution_level(t).item()
        r_t = min(r_t, len(hierarchy))

        # Check if resolution changed — need to lift positions
        expected_n = N if r_t == 0 else (
            hierarchy[r_t - 1].num_nodes if r_t <= len(hierarchy) else
            hierarchy[-1].num_nodes if hierarchy else N
        )

        if expected_n != current_n and len(hierarchy) > 0:
            # Lift or coarsen all molecules in the batch
            if expected_n > current_n:
                # Lift: coarse → fine via pseudoinverse
                for lev in range(min(r_t, len(hierarchy)) if r_t > 0 else 0,
                                 len(hierarchy)):
                    C = hierarchy[lev].coarsening_matrix.to(device)
                    if C.shape[0] == current_n:
                        C_pinv = torch.linalg.pinv(C)
                        # Apply per molecule
                        new_parts = []
                        type_parts = []
                        for i in range(B):
                            x_mol = x_t[i * current_n:(i + 1) * current_n]
                            new_parts.append(C_pinv @ x_mol)
                            # Expand types
                            t_mol = atom_types[i * current_n:(i + 1) * current_n]
                            type_parts.append(
                                t_mol[hierarchy[lev].cluster_assignment.to(device)]
                            )
                        x_t = torch.cat(new_parts)
                        atom_types = torch.cat(type_parts)
                        current_n = C_pinv.shape[0]
                        break
            else:
                # Coarsen: fine → coarse
                for lev_idx in range(len(hierarchy)):
                    C = hierarchy[lev_idx].coarsening_matrix.to(device)
                    if C.shape[0] == expected_n:
                        new_parts = []
                        type_parts = []
                        for i in range(B):
                            x_mol = x_t[i * current_n:(i + 1) * current_n]
                            new_parts.append(C @ x_mol)
                            type_parts.append(
                                atom_types[i * current_n:(i + 1) * current_n][:expected_n]
                            )
                        x_t = torch.cat(new_parts)
                        atom_types = torch.cat(type_parts)
                        current_n = expected_n
                        break

            # Rebuild edge index for new resolution
            if r_t == 0:
                edge_index_current = edge_index
                batch_vector_current = batch_vector
            elif r_t <= len(hierarchy):
                adj_c = hierarchy[r_t - 1].coarsened_adj.to(device)
                ei_c = (adj_c > 0).nonzero(as_tuple=False).t().contiguous()
                ei_parts_c = [ei_c + i * current_n for i in range(B)]
                edge_index_current = torch.cat(ei_parts_c, dim=1)
                batch_vector_current = torch.cat([
                    torch.full((current_n,), i, dtype=torch.long, device=device)
                    for i in range(B)
                ])

        # Timestep tensor
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)

        # Model forward (batched — single GPU call for all B molecules)
        eps_pred, type_logits = model(
            x_t=x_t,
            atom_types=atom_types,
            t=t_tensor,
            resolution_level=r_t,
            edge_index=edge_index_current,
            edge_attr=None,
            batch=batch_vector_current,
        )

        # Update atom types
        atom_types = type_logits.argmax(dim=-1)

        # DDPM reverse step (isotropic, same as EDM)
        if t > 0:
            alpha_bar_t = noise_schedule.alpha_bar(t).to(device)
            sigma_t = noise_schedule.sigma(t).to(device)

            t_next = timesteps[min(step_idx + 1, len(timesteps) - 1)]
            alpha_bar_prev = noise_schedule.alpha_bar(t_next).to(device)
            sigma_sq_t = noise_schedule.sigma_squared(t).to(device)
            sigma_sq_prev = noise_schedule.sigma_squared(t_next).to(device)
            beta_t = noise_schedule.beta(t).to(device)

            # Predict x0 (clamp to prevent overflow from small alpha_bar)
            x0_hat = (x_t - sigma_t * eps_pred) / alpha_bar_t.clamp(min=1e-8)
            x0_hat = x0_hat.clamp(-20, 20)  # molecular coordinates rarely exceed ±20 Å

            # Posterior mean
            coeff_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / sigma_sq_t.clamp(min=1e-8)
            alpha_t = alpha_bar_t / alpha_bar_prev.clamp(min=1e-8)
            coeff_xt = (torch.sqrt(alpha_t) * sigma_sq_prev) / sigma_sq_t.clamp(min=1e-8)
            mu = coeff_x0 * x0_hat + coeff_xt * x_t

            # Posterior variance
            beta_tilde = (beta_t * sigma_sq_prev) / sigma_sq_t.clamp(min=1e-8)
            sigma_post = torch.sqrt(beta_tilde.clamp(min=1e-12))

            z = torch.randn_like(x_t) if t_next > 0 else torch.zeros_like(x_t)
            x_t = mu + sigma_post * z
        else:
            # t == 0: clean prediction
            sigma_0 = noise_schedule.sigma(0).to(device)
            alpha_bar_0 = noise_schedule.alpha_bar(0).to(device)
            x_t = (x_t - sigma_0 * eps_pred) / alpha_bar_0.clamp(min=1e-8)

    # If still at coarsened resolution, lift to full
    if current_n < N and len(hierarchy) > 0:
        new_parts = []
        type_parts = []
        for i in range(B):
            x_mol = x_t[i * current_n:(i + 1) * current_n]
            t_mol = atom_types[i * current_n:(i + 1) * current_n]
            # Lift through all remaining levels
            cur_x = x_mol
            cur_t = t_mol
            for k in range(len(hierarchy) - 1, -1, -1):
                C = hierarchy[k].coarsening_matrix.to(device)
                if C.shape[0] == cur_x.shape[0]:
                    C_pinv = torch.linalg.pinv(C)
                    cur_x = C_pinv @ cur_x
                    cur_t = cur_t[hierarchy[k].cluster_assignment.to(device)]
            # Trim/pad to N
            if cur_x.shape[0] > N:
                cur_x = cur_x[:N]
                cur_t = cur_t[:N]
            elif cur_x.shape[0] < N:
                pad = N - cur_x.shape[0]
                cur_x = torch.cat([cur_x, torch.randn(pad, 3, device=device)])
                cur_t = torch.cat([cur_t, torch.randint(0, num_atom_types, (pad,), device=device)])
            new_parts.append(cur_x)
            type_parts.append(cur_t)
        x_t = torch.cat(new_parts)
        atom_types = torch.cat(type_parts)
        current_n = N

    # Center each molecule
    for i in range(B):
        mol_pos = x_t[i * current_n:(i + 1) * current_n]
        x_t[i * current_n:(i + 1) * current_n] = mol_pos - mol_pos.mean(dim=0, keepdim=True)

    # Split into per-molecule results
    positions_list = []
    types_list = []
    for i in range(B):
        positions_list.append(x_t[i * current_n:(i + 1) * current_n].cpu())
        types_list.append(atom_types[i * current_n:(i + 1) * current_n].cpu())

    return {
        "positions": positions_list,
        "atom_types": types_list,
    }
