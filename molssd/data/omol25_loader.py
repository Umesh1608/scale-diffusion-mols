"""OMol25 dataset loader for MolSSD pre-training.

Loads from Meta's OMol25 dataset (83M molecules, up to 350 atoms) stored in
ASE-compatible LMDB format. Supports the full dataset and the "Neutral" subset
(charge-neutral singlets — ideal for organic/drug-like molecules).

Data access:
    OMol25 requires Globus access. Apply at:
    https://fair-chem.github.io/molecules/datasets/omol25.html

    Alternatively, use the 4M subset for faster iteration.

Format:
    LMDB files loadable via ``fairchem.core.datasets.AseDBDataset``.
    Each entry is an ASE Atoms object with positions, atomic numbers,
    total energy, and forces.

References:
    - OMol25: Batatia et al., "OMol25: A Molecular Dataset for Molecular
      Foundation Models" (arXiv:2505.08762)
    - FairChem: https://github.com/facebookresearch/fairchem
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from molssd.core.coarsening import (
    CoarseningLevel,
    build_coarsening_hierarchy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — extended atom types for OMol25 (organic subset)
# ---------------------------------------------------------------------------

# For the "Neutral" organic subset, we use 16 common atom types.
# Rare elements are mapped to a catch-all index.
ATOMIC_NUMBER_TO_INDEX = {
    1: 0,    # H
    6: 1,    # C
    7: 2,    # N
    8: 3,    # O
    9: 4,    # F
    16: 5,   # S
    15: 6,   # P
    17: 7,   # Cl
    35: 8,   # Br
    53: 9,   # I
    5: 10,   # B
    14: 11,  # Si
    34: 12,  # Se
    33: 13,  # As
    52: 14,  # Te
}
# Index 15 reserved for "other" / unknown atoms
OTHER_INDEX = 15
NUM_ATOM_TYPES = 16

INDEX_TO_SYMBOL = {
    0: "H", 1: "C", 2: "N", 3: "O", 4: "F", 5: "S", 6: "P",
    7: "Cl", 8: "Br", 9: "I", 10: "B", 11: "Si", 12: "Se",
    13: "As", 14: "Te", 15: "X",
}

ATOMIC_MASSES = {
    0: 1.008, 1: 12.011, 2: 14.007, 3: 15.999, 4: 18.998,
    5: 32.065, 6: 30.974, 7: 35.453, 8: 79.904, 9: 126.904,
    10: 10.811, 11: 28.086, 12: 78.960, 13: 74.922, 14: 127.60,
    15: 12.011,  # fallback mass for "other"
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coarsen_types_majority(
    atom_types: torch.Tensor,
    cluster_assignment: torch.Tensor,
    num_clusters: int,
    num_atom_types: int = NUM_ATOM_TYPES,
) -> torch.Tensor:
    one_hot = torch.zeros(atom_types.shape[0], num_atom_types)
    one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
    counts = torch.zeros(num_clusters, num_atom_types)
    counts.index_add_(0, cluster_assignment, one_hot)
    return counts.argmax(dim=1)


@dataclass
class CoarsenedData:
    positions: torch.Tensor
    atom_types: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int


def _precompute_coarsened(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    hierarchy: List[CoarseningLevel],
) -> List[CoarsenedData]:
    result: List[CoarsenedData] = []
    cur_pos = positions
    cur_types = atom_types
    for level in hierarchy:
        C = level.coarsening_matrix
        coarsened_pos = C @ cur_pos
        coarsened_types = _coarsen_types_majority(
            cur_types, level.cluster_assignment, level.num_nodes,
        )
        row, col = torch.where(level.coarsened_adj > 0)
        ei = torch.stack([row, col], dim=0)
        result.append(CoarsenedData(
            positions=coarsened_pos, atom_types=coarsened_types,
            edge_index=ei, num_nodes=level.num_nodes,
        ))
        cur_pos = coarsened_pos
        cur_types = coarsened_types
    return result


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class OMol25Data:
    """Container for a single OMol25 molecule."""
    positions: torch.Tensor
    atom_types: torch.Tensor
    edge_index: torch.Tensor
    adj: torch.Tensor
    atomic_masses: torch.Tensor
    num_atoms: int
    energy: Optional[float] = None
    coarsening_hierarchy: List[CoarseningLevel] = field(default_factory=list)
    coarsened_data: List[CoarsenedData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# OMol25 Dataset
# ---------------------------------------------------------------------------

class OMol25MolSSD(Dataset):
    """OMol25 dataset preprocessed for MolSSD pre-training.

    Loads OMol25 from LMDB files via fairchem's AseDBDataset, filters for
    organic molecules, and preprocesses for MolSSD training.

    Args:
        lmdb_path: Path to the OMol25 LMDB directory (e.g., ``train_4M/``).
        max_coarsening_levels: Max coarsening hierarchy depth.
        max_atoms: Skip molecules larger than this.
        max_molecules: Cap dataset size.
        transform: Optional transform at getitem time.
        filter_organic: If True, skip molecules with unsupported atom types.
        cache_path: Path to save/load processed cache. If None, processes
            on-the-fly (slower but uses less disk).
    """

    def __init__(
        self,
        lmdb_path: str,
        max_coarsening_levels: int = 5,
        max_atoms: int = 200,
        max_molecules: Optional[int] = None,
        transform: Optional[Callable] = None,
        filter_organic: bool = True,
        cache_path: Optional[str] = None,
    ) -> None:
        self.lmdb_path = lmdb_path
        self.max_coarsening_levels = max_coarsening_levels
        self.max_atoms = max_atoms
        self.max_molecules = max_molecules
        self.transform = transform
        self.filter_organic = filter_organic

        if cache_path and os.path.exists(cache_path):
            logger.info("Loading cached OMol25 data from %s", cache_path)
            self.data_list: List[OMol25Data] = torch.load(
                cache_path, weights_only=False
            )
            if max_molecules and len(self.data_list) > max_molecules:
                self.data_list = self.data_list[:max_molecules]
        else:
            self.data_list = self._process()
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self.data_list, cache_path)
                logger.info("Saved processed OMol25 data to %s", cache_path)

    def _process(self) -> List[OMol25Data]:
        """Load and preprocess OMol25 molecules from LMDB."""
        try:
            from fairchem.core.datasets import AseDBDataset
        except ImportError:
            raise ImportError(
                "fairchem is required for OMol25. Install with: "
                "pip install fairchem-core"
            )

        logger.info("Loading OMol25 from %s ...", self.lmdb_path)
        ase_dataset = AseDBDataset({"src": self.lmdb_path})
        total = len(ase_dataset)
        logger.info("Found %d entries in OMol25 LMDB", total)

        if self.max_molecules:
            total = min(total, self.max_molecules)

        data_list: List[OMol25Data] = []
        skipped = 0

        for i in range(total):
            if (i + 1) % 50000 == 0:
                logger.info(
                    "  Processing %d / %d (kept %d, skipped %d)",
                    i + 1, total, len(data_list), skipped,
                )

            try:
                atoms = ase_dataset.get_atoms(i)
            except Exception:
                skipped += 1
                continue

            result = self._process_ase_atoms(atoms)
            if result is not None:
                data_list.append(result)
            else:
                skipped += 1

        logger.info(
            "Processed %d molecules (%d skipped)", len(data_list), skipped
        )
        return data_list

    def _process_ase_atoms(self, atoms: Any) -> Optional[OMol25Data]:
        """Process a single ASE Atoms object into OMol25Data."""
        import numpy as np

        num_atoms = len(atoms)
        if num_atoms < 2 or num_atoms > self.max_atoms:
            return None

        # 1. Map atomic numbers
        atomic_numbers = atoms.get_atomic_numbers()
        atom_types = torch.zeros(num_atoms, dtype=torch.long)
        for i, z in enumerate(atomic_numbers):
            idx = ATOMIC_NUMBER_TO_INDEX.get(int(z))
            if idx is not None:
                atom_types[i] = idx
            elif self.filter_organic:
                return None  # Skip molecules with unsupported atoms
            else:
                atom_types[i] = OTHER_INDEX

        # 2. Positions
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float32)

        # 3. Atomic masses
        atomic_masses = torch.tensor(
            [ATOMIC_MASSES[int(at.item())] for at in atom_types],
            dtype=torch.float32,
        )

        # 4. Build adjacency from distance cutoff (no bonds in OMol25)
        # Use 1.8 Angstrom cutoff for bonded atoms
        dists = torch.cdist(pos, pos)
        # Element-aware cutoff: sum of covalent radii + tolerance
        adj = (dists < 1.8) & (dists > 0.4)
        adj = adj.float()
        adj = torch.max(adj, adj.t())

        # Ensure connectivity — if isolated atoms exist, use larger cutoff
        connected = adj.sum(dim=1) > 0
        if not connected.all():
            adj_fallback = (dists < 2.5) & (dists > 0.4)
            adj = torch.max(adj, adj_fallback.float())
            adj = torch.max(adj, adj.t())

        # Build edge_index
        src, dst = torch.where(adj > 0)
        if len(src) == 0:
            return None
        edge_index = torch.stack([src, dst], dim=0).long()

        # 5. Center of mass subtraction
        total_mass = atomic_masses.sum()
        com = (atomic_masses.unsqueeze(1) * pos).sum(dim=0) / total_mass
        positions = pos - com.unsqueeze(0)

        # 6. Energy (if available)
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            energy = None

        # 7. Coarsening hierarchy
        try:
            hierarchy = build_coarsening_hierarchy(
                adj=adj, num_atoms=num_atoms,
                target_sizes=None, atomic_masses=atomic_masses,
            )
        except Exception:
            hierarchy = []

        if len(hierarchy) > self.max_coarsening_levels:
            hierarchy = hierarchy[:self.max_coarsening_levels]

        coarsened_data = _precompute_coarsened(positions, atom_types, hierarchy)

        return OMol25Data(
            positions=positions,
            atom_types=atom_types,
            edge_index=edge_index,
            adj=adj,
            atomic_masses=atomic_masses,
            num_atoms=num_atoms,
            energy=energy,
            coarsening_hierarchy=hierarchy,
            coarsened_data=coarsened_data,
        )

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mol = self.data_list[idx]
        data = {
            "positions": mol.positions.clone(),
            "atom_types": mol.atom_types.clone(),
            "edge_index": mol.edge_index.clone(),
            "adj": mol.adj.clone(),
            "atomic_masses": mol.atomic_masses.clone(),
            "num_atoms": mol.num_atoms,
            "coarsening_hierarchy": mol.coarsening_hierarchy,
            "coarsened_data": mol.coarsened_data,
        }
        if self.transform is not None:
            data = self.transform(data)
        return data
