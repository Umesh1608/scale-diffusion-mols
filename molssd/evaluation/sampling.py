"""Reverse-process sampling loop for generating molecules with MolSSD.

Implements the full reverse diffusion sampling procedure that starts from pure
Gaussian noise at the coarsest resolution level and progressively denoises and
refines the molecular representation back to full atomic resolution.

The sampling procedure follows the SSD framework (arXiv:2603.08709):
  1. Begin at t = T-1 (coarsest resolution) with x_{T-1} ~ N(0, I).
  2. At each step, predict noise with the denoiser network, recover x0_hat,
     compute the posterior, and sample x_{t-1}.
  3. At resolution-changing steps the posterior is non-isotropic and the
     number of nodes increases (coarse -> fine).
  4. At t = 0, return x0_hat directly as the final prediction.

Supports reduced-step sampling (T_sample < T) via a linearly spaced
sub-schedule for faster generation with minimal quality loss.

Functions
---------
sample_single_molecule
    Generate one molecule given a pre-built coarsening hierarchy.
sample_molecules
    Generate a batch of molecules with varying atom counts.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from molssd.core.coarsening import CoarseningLevel, build_coarsening_hierarchy
from molssd.core.degradation import DegradationOperator
from molssd.core.diffusion import MolSSDDiffusion
from molssd.core.noise_schedules import NoiseSchedule, ResolutionSchedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adj_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """Convert a dense adjacency matrix to COO edge_index.

    Args:
        adj: Dense adjacency matrix of shape ``(N, N)``.

    Returns:
        Edge index tensor of shape ``(2, E)`` in COO format.
    """
    src, dst = torch.where(adj > 0)
    return torch.stack([src, dst], dim=0)


def _get_adj_at_resolution(
    resolution_level: int,
    coarsening_hierarchy: List[CoarseningLevel],
    num_atoms: int,
    device: torch.device,
) -> torch.Tensor:
    """Get the adjacency matrix at the given resolution level.

    At level 0 (full atomic resolution), returns a fully connected adjacency
    matrix (no self-loops) which is appropriate for unconditional generation
    where the true molecular graph is unknown.

    At level k > 0, returns the coarsened adjacency stored in the hierarchy
    at index k-1 (since hierarchy[0] maps atoms -> first coarsening).

    Args:
        resolution_level: Resolution level index (0 = finest).
        coarsening_hierarchy: Pre-built coarsening hierarchy.
        num_atoms: Number of atoms at full resolution.
        device: Target device.

    Returns:
        Adjacency matrix of shape ``(N_k, N_k)`` at the requested level.
    """
    if resolution_level == 0:
        # Fully connected graph (no self-loops) at atomic resolution
        adj = torch.ones(num_atoms, num_atoms, device=device)
        adj.fill_diagonal_(0.0)
        return adj

    # hierarchy[k-1] stores the coarsened adjacency at level k
    return coarsening_hierarchy[resolution_level - 1].coarsened_adj.to(device)


def _get_num_nodes_at_resolution(
    resolution_level: int,
    coarsening_hierarchy: List[CoarseningLevel],
    num_atoms: int,
) -> int:
    """Get the number of nodes at the given resolution level.

    Args:
        resolution_level: Resolution level index (0 = finest).
        coarsening_hierarchy: Pre-built coarsening hierarchy.
        num_atoms: Number of atoms at full resolution.

    Returns:
        Number of (super-)nodes at the requested level.
    """
    if resolution_level == 0:
        return num_atoms
    return coarsening_hierarchy[resolution_level - 1].num_nodes


def _infer_resolution_from_nodes(
    n_nodes: int,
    coarsening_hierarchy: List[CoarseningLevel],
    num_atoms: int,
) -> int:
    """Infer the resolution level from the number of nodes."""
    if n_nodes == num_atoms:
        return 0
    for i, level in enumerate(coarsening_hierarchy):
        if level.num_nodes == n_nodes:
            return i + 1
    # Fallback: find the closest level
    diffs = [(abs(num_atoms - n_nodes), 0)]
    for i, level in enumerate(coarsening_hierarchy):
        diffs.append((abs(level.num_nodes - n_nodes), i + 1))
    return min(diffs)[1]


def _lift_positions(
    x: torch.Tensor,
    coarsening_hierarchy: List[CoarseningLevel],
    from_level: int,
    to_level: int,
) -> torch.Tensor:
    """Lift positions from a coarser to a finer level using pseudoinverses."""
    current = x
    for level in range(from_level, to_level, -1):
        C = coarsening_hierarchy[level - 1].coarsening_matrix.to(x.device)
        C_pinv = torch.linalg.pinv(C)
        current = C_pinv @ current
    return current


def _coarsen_positions(
    x: torch.Tensor,
    coarsening_hierarchy: List[CoarseningLevel],
    from_level: int,
    to_level: int,
) -> torch.Tensor:
    """Coarsen positions from a finer to a coarser level."""
    current = x
    for level in range(from_level + 1, to_level + 1):
        C = coarsening_hierarchy[level - 1].coarsening_matrix.to(x.device)
        current = C @ current
    return current


def _expand_atom_types(
    atom_types_coarse: torch.Tensor,
    coarsening_hierarchy: List[CoarseningLevel],
    from_level: int,
    to_level: int,
) -> torch.Tensor:
    """Expand atom types from a coarser to a finer resolution level.

    Each fine-level node inherits the atom type of its parent super-node at
    the coarser level. This is applied iteratively through intermediate
    levels if from_level - to_level > 1.

    Args:
        atom_types_coarse: Integer atom types at the coarser level, shape
            ``(N_{from_level},)``.
        coarsening_hierarchy: Pre-built coarsening hierarchy.
        from_level: Starting (coarser) resolution level.
        to_level: Target (finer) resolution level. Must be <= from_level.

    Returns:
        Atom types at the target level, shape ``(N_{to_level},)``.
    """
    current_types = atom_types_coarse
    # Walk from coarser to finer: level k -> level k-1 uses
    # hierarchy[k-1].cluster_assignment to broadcast types
    for level in range(from_level, to_level, -1):
        cluster_assignment = coarsening_hierarchy[level - 1].cluster_assignment
        # Each fine-level node gets the type of its parent super-node
        current_types = current_types[cluster_assignment.to(current_types.device)]
    return current_types


def _build_timestep_schedule(T: int, T_sample: Optional[int] = None) -> List[int]:
    """Build the sequence of timesteps for sampling.

    If ``T_sample`` is None or >= T, returns the full schedule
    ``[T-1, T-2, ..., 1, 0]``.  Otherwise, selects ``T_sample`` evenly
    spaced timesteps from the full range and always includes 0.

    Args:
        T: Total number of diffusion timesteps.
        T_sample: Number of sampling steps. If None, use all T steps.

    Returns:
        List of integer timesteps in descending order, ending at 0.
    """
    if T_sample is None or T_sample >= T:
        return list(range(T - 1, -1, -1))

    T_sample = max(T_sample, 1)
    # Linearly spaced indices from T-1 down to 0
    indices = torch.linspace(T - 1, 0, T_sample).long().tolist()
    # Ensure 0 is included and no duplicates
    if indices[-1] != 0:
        indices[-1] = 0
    # Remove duplicates while preserving order (descending)
    seen = set()
    unique = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


# ---------------------------------------------------------------------------
# Single molecule sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_single_molecule(
    model: nn.Module,
    diffusion: MolSSDDiffusion,
    degradation_op: DegradationOperator,
    num_atoms: int,
    coarsening_hierarchy: List[CoarseningLevel],
    device: torch.device,
    T_sample: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Generate a single molecule via the reverse diffusion process.

    Starts from Gaussian noise at the coarsest resolution and iteratively
    denoises and refines, with resolution increasing (coarse -> fine) as
    the timestep decreases.

    Args:
        model: Trained denoising network (e.g. MolecularFlexiNet). Must be
            in eval mode.
        diffusion: MolSSDDiffusion instance holding the noise schedule and
            providing ``predict_x0``, ``compute_posterior_params``, and
            ``reverse_step``.
        degradation_op: DegradationOperator built from the coarsening
            hierarchy for this molecule.
        num_atoms: Number of atoms at full atomic resolution.
        coarsening_hierarchy: Pre-built coarsening hierarchy for this
            molecule.
        device: Device to run sampling on (e.g. ``torch.device('cuda')``).
        T_sample: Number of sampling steps. If None, uses the full schedule
            (all T timesteps). A smaller value speeds up generation.

    Returns:
        Dictionary with:
            - ``'positions'``: Final atomic positions, shape ``(num_atoms, 3)``.
            - ``'atom_types'``: Predicted atom type indices, shape
              ``(num_atoms,)``.
    """
    noise_schedule = diffusion.noise_schedule
    T = noise_schedule.T

    # Build the timestep schedule
    timesteps = _build_timestep_schedule(T, T_sample)

    # Determine the initial (coarsest) resolution at t = T-1
    t_start = timesteps[0]
    resolution_level = degradation_op.get_resolution_at(t_start)
    n_nodes = _get_num_nodes_at_resolution(
        resolution_level, coarsening_hierarchy, num_atoms
    )

    # Sample x_{T-1} ~ N(0, I) at the coarsest resolution
    x_t = torch.randn(n_nodes, 3, device=device)

    # Initialize atom types uniformly at the coarsest resolution
    atom_types = torch.randint(
        0, diffusion.num_atom_types, (n_nodes,), device=device
    )

    # Store final type_logits for the last update
    final_type_logits = None

    for step_idx, t in enumerate(timesteps):
        # Current resolution level and adjacency
        resolution_level = degradation_op.get_resolution_at(t)

        # When using reduced sampling steps, x_t may be at a different
        # resolution than what this timestep expects (because the posterior
        # steps from t to t-1, not to the next scheduled timestep).
        # Lift or coarsen x_t to match the expected resolution.
        expected_nodes = _get_num_nodes_at_resolution(
            resolution_level, coarsening_hierarchy, num_atoms
        )
        if x_t.shape[0] != expected_nodes:
            # Determine the current resolution of x_t based on its node count
            x_t_res = _infer_resolution_from_nodes(
                x_t.shape[0], coarsening_hierarchy, num_atoms
            )
            if expected_nodes > x_t.shape[0]:
                # Need to lift (coarse -> fine): use pseudoinverse
                x_t = _lift_positions(
                    x_t, coarsening_hierarchy, x_t_res, resolution_level
                )
                atom_types = _expand_atom_types(
                    atom_types, coarsening_hierarchy, x_t_res, resolution_level
                )
            else:
                # Need to coarsen (fine -> coarse): use coarsening matrix
                x_t = _coarsen_positions(
                    x_t, coarsening_hierarchy, x_t_res, resolution_level
                )
                atom_types = atom_types[:expected_nodes]

        adj = _get_adj_at_resolution(
            resolution_level, coarsening_hierarchy, num_atoms, device
        )
        edge_index = _adj_to_edge_index(adj)

        # Build the timestep tensor (scalar, shape (1,) for unbatched)
        t_tensor = torch.tensor([t], dtype=torch.long, device=device)

        # Run the denoiser
        eps_hat, type_logits = model(
            x_t, atom_types, t_tensor, resolution_level, edge_index
        )

        # Predict x0 at full atomic resolution
        x0_hat = diffusion.predict_x0(x_t, t, eps_hat, degradation_op)

        if t == 0:
            # At the final step, use x0_hat directly
            x_0 = x0_hat
            final_type_logits = type_logits
            break

        # Compute posterior parameters
        posterior_params = diffusion.compute_posterior_params(
            x_t, x0_hat, t, degradation_op
        )

        # Sample x_{t-1}
        x_t_minus_1 = diffusion.reverse_step(x_t, t, posterior_params)

        # Update atom types from current predictions
        atom_types = type_logits.argmax(dim=-1)

        # Determine the resolution at the next timestep
        if step_idx + 1 < len(timesteps):
            t_next = timesteps[step_idx + 1]
        else:
            t_next = 0

        res_next = degradation_op.get_resolution_at(t_next)

        # If the next scheduled step has a different resolution from t-1,
        # we'll handle the mismatch at the top of the next iteration.
        if res_next != resolution_level:
            coarse_types = type_logits.argmax(dim=-1)
            atom_types = _expand_atom_types(
                coarse_types, coarsening_hierarchy, resolution_level, res_next
            )

        x_t = x_t_minus_1
        final_type_logits = type_logits

    # Final atom types from the last type_logits
    if final_type_logits is not None:
        # If we ended at t=0 at full resolution, type_logits has N_0 nodes
        final_atom_types = final_type_logits.argmax(dim=-1)
        # If the final resolution is not 0, expand to full resolution
        final_res = degradation_op.get_resolution_at(0)
        if final_atom_types.shape[0] != num_atoms:
            current_res = degradation_op.get_resolution_at(timesteps[-1])
            final_atom_types = _expand_atom_types(
                final_atom_types, coarsening_hierarchy, current_res, 0
            )
    else:
        final_atom_types = atom_types

    return {
        "positions": x_0,
        "atom_types": final_atom_types,
    }


# ---------------------------------------------------------------------------
# Batch molecule sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_molecules(
    model: nn.Module,
    diffusion: MolSSDDiffusion,
    noise_schedule: NoiseSchedule,
    resolution_schedule: ResolutionSchedule,
    num_molecules: int,
    num_atoms_list: List[int],
    device: torch.device,
    T_sample: Optional[int] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Generate multiple molecules via the MolSSD reverse diffusion process.

    Each molecule is generated independently with its own coarsening
    hierarchy built from a fully connected graph of the specified size.
    Molecules are sampled sequentially (one at a time) since each may have
    a different number of atoms and thus a different hierarchy.

    Args:
        model: Trained denoising network (e.g. MolecularFlexiNet). Must be
            in eval mode. Moved to ``device`` by the caller.
        diffusion: MolSSDDiffusion instance.
        noise_schedule: Noise schedule (provides T and alpha_bar/sigma).
        resolution_schedule: Resolution schedule mapping timesteps to levels.
        num_molecules: Number of molecules to generate.
        num_atoms_list: List of atom counts, one per molecule. If shorter
            than ``num_molecules``, entries are cycled.
        device: Device to run sampling on.
        T_sample: Number of sampling steps per molecule. If None, uses the
            full T-step schedule.

    Returns:
        List of dictionaries, each containing:
            - ``'positions'``: Tensor of shape ``(N_i, 3)`` with atomic
              coordinates.
            - ``'atom_types'``: Tensor of shape ``(N_i,)`` with integer
              atom type indices.
    """
    model.eval()
    results: List[Dict[str, torch.Tensor]] = []

    for mol_idx in range(num_molecules):
        # Determine atom count (cycle through list if necessary)
        num_atoms = num_atoms_list[mol_idx % len(num_atoms_list)]

        if num_atoms < 1:
            # Skip degenerate molecules
            results.append({
                "positions": torch.zeros(0, 3, device=device),
                "atom_types": torch.zeros(0, dtype=torch.long, device=device),
            })
            continue

        # Build a fully connected adjacency matrix (no self-loops) as the
        # dummy molecular graph for generation.
        # Build on CPU first — coarsening uses eigendecomposition which
        # must happen on CPU, then move the DegradationOperator to device.
        adj_cpu = torch.ones(num_atoms, num_atoms) - torch.eye(num_atoms)

        # Build the coarsening hierarchy (on CPU)
        coarsening_hierarchy = build_coarsening_hierarchy(
            adj=adj_cpu,
            num_atoms=num_atoms,
        )

        # Determine num_atoms_per_level for the resolution schedule
        # Level 0 = full atoms, then each hierarchy level
        num_atoms_per_level = [num_atoms]
        for level in coarsening_hierarchy:
            num_atoms_per_level.append(level.num_nodes)

        # Build a molecule-specific resolution schedule matching the hierarchy
        mol_resolution_schedule = ResolutionSchedule(
            T=noise_schedule.T,
            num_levels=len(num_atoms_per_level),
            num_atoms_per_level=num_atoms_per_level,
            schedule_type=resolution_schedule.schedule_type,
            gamma=resolution_schedule.gamma,
        ).to(device)

        # Build a molecule-specific diffusion object with matching schedule
        mol_diffusion = MolSSDDiffusion(
            noise_schedule=noise_schedule,
            resolution_schedule=mol_resolution_schedule,
            num_atom_types=diffusion.num_atom_types,
        ).to(device)

        # Build the degradation operator
        degradation_op = DegradationOperator(
            coarsening_hierarchy=coarsening_hierarchy,
            noise_schedule=noise_schedule,
            resolution_schedule=mol_resolution_schedule,
        ).to(device)

        # Sample the molecule
        result = sample_single_molecule(
            model=model,
            diffusion=mol_diffusion,
            degradation_op=degradation_op,
            num_atoms=num_atoms,
            coarsening_hierarchy=coarsening_hierarchy,
            device=device,
            T_sample=T_sample,
        )
        results.append(result)

    return results
