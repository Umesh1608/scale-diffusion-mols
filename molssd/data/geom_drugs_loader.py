"""GEOM-Drugs dataset loader for MolSSD training.

Loads the GEOM-Drugs dataset (drug-like molecules, up to 181 atoms) and
preprocesses each molecule for the MolSSD framework. Supports downloading
from Harvard Dataverse, selecting lowest-energy conformers, and precomputing
spectral coarsening hierarchies.

Key differences from QM9:
    - 10 atom types: H, C, N, O, F, S, P, Cl, Br, I
    - Much larger molecules (avg ~44 atoms, max 181)
    - Deeper coarsening hierarchies (up to 5 levels)
    - ~304K molecules (vs 130K for QM9)

Data source:
    Harvard Dataverse: doi:10.7910/DVN/JNGTDF
    File: ``rdkit_folder.tar.gz`` (recommended for Python)

References:
    - GEOM dataset: Axelrod & Gomez-Bombarelli, "GEOM: Energy-annotated
      molecular conformations" (Scientific Data, 2022)
    - EDM: Hoogeboom et al., "Equivariant Diffusion for Molecule
      Generation in 3D" (ICML 2022)
"""

from __future__ import annotations

import logging
import os
import pickle
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from molssd.core.coarsening import (
    CoarseningLevel,
    build_coarsening_hierarchy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GEOM-Drugs atom type mapping (10 atom types for drug-like molecules)
ATOMIC_NUMBER_TO_INDEX = {
    1: 0,   # H
    6: 1,   # C
    7: 2,   # N
    8: 3,   # O
    9: 4,   # F
    16: 5,  # S
    15: 6,  # P
    17: 7,  # Cl
    35: 8,  # Br
    53: 9,  # I
}
INDEX_TO_ATOMIC_NUMBER = {v: k for k, v in ATOMIC_NUMBER_TO_INDEX.items()}
INDEX_TO_SYMBOL = {
    0: "H", 1: "C", 2: "N", 3: "O", 4: "F",
    5: "S", 6: "P", 7: "Cl", 8: "Br", 9: "I",
}
NUM_ATOM_TYPES = 10

# Atomic masses in Daltons
ATOMIC_MASSES = {
    0: 1.008,    # H
    1: 12.011,   # C
    2: 14.007,   # N
    3: 15.999,   # O
    4: 18.998,   # F
    5: 32.065,   # S
    6: 30.974,   # P
    7: 35.453,   # Cl
    8: 79.904,   # Br
    9: 126.904,  # I
}

# Default split sizes (60/20/20 following recent benchmarks)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2

# Maximum atoms to include (skip very large molecules for memory)
MAX_ATOMS_DEFAULT = 181


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coarsen_types_majority(
    atom_types: torch.Tensor,
    cluster_assignment: torch.Tensor,
    num_clusters: int,
    num_atom_types: int = NUM_ATOM_TYPES,
) -> torch.Tensor:
    """Coarsen atom types via majority vote (vectorised)."""
    one_hot = torch.zeros(atom_types.shape[0], num_atom_types)
    one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
    counts = torch.zeros(num_clusters, num_atom_types)
    counts.index_add_(0, cluster_assignment, one_hot)
    return counts.argmax(dim=1)


def _precompute_coarsened(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    hierarchy: List[CoarseningLevel],
) -> List["CoarsenedData"]:
    """Precompute coarsened positions, types, and edges at each level."""
    result: List[CoarsenedData] = []
    cur_pos = positions
    cur_types = atom_types

    for level in hierarchy:
        C = level.coarsening_matrix
        coarsened_pos = C @ cur_pos
        coarsened_types = _coarsen_types_majority(
            cur_types, level.cluster_assignment, level.num_nodes,
        )
        adj_c = level.coarsened_adj
        row, col = torch.where(adj_c > 0)
        ei = torch.stack([row, col], dim=0)

        result.append(CoarsenedData(
            positions=coarsened_pos,
            atom_types=coarsened_types,
            edge_index=ei,
            num_nodes=level.num_nodes,
        ))
        cur_pos = coarsened_pos
        cur_types = coarsened_types

    return result


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CoarsenedData:
    """Precomputed data at one coarsened resolution level."""
    positions: torch.Tensor
    atom_types: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int


@dataclass
class GEOMDrugsData:
    """Container for a single preprocessed drug-like molecule.

    Attributes:
        positions: Atom coordinates (CoM-subtracted), shape ``(N, 3)``.
        atom_types: Integer atom type indices (0-9), shape ``(N,)``.
        edge_index: Bond connectivity in COO format, shape ``(2, E)``.
        adj: Dense adjacency matrix, shape ``(N, N)``.
        atomic_masses: Per-atom masses, shape ``(N,)``.
        num_atoms: Number of atoms.
        coarsening_hierarchy: Multi-level coarsening hierarchy.
        coarsened_data: Precomputed coarsened data per level.
    """
    positions: torch.Tensor
    atom_types: torch.Tensor
    edge_index: torch.Tensor
    adj: torch.Tensor
    atomic_masses: torch.Tensor
    num_atoms: int
    coarsening_hierarchy: List[CoarseningLevel] = field(default_factory=list)
    coarsened_data: List[CoarsenedData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Random rotation (data augmentation)
# ---------------------------------------------------------------------------

class RandomRotation:
    """Applies a random SO(3) rotation to atom positions."""

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        positions = data["positions"]
        R = self._random_rotation_matrix(device=positions.device, dtype=positions.dtype)
        data["positions"] = positions @ R.T
        if "coarsened_data" in data:
            for cd in data["coarsened_data"]:
                if hasattr(cd, "positions"):
                    cd.positions = cd.positions @ R.T
        return data

    @staticmethod
    def _random_rotation_matrix(
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        mat = torch.randn(3, 3, device=device, dtype=dtype)
        q, r = torch.linalg.qr(mat)
        d = torch.diag(torch.sign(torch.diag(r)))
        q = q @ d
        if torch.det(q) < 0:
            q[:, 0] *= -1
        return q


# ---------------------------------------------------------------------------
# GEOM-Drugs Dataset
# ---------------------------------------------------------------------------

class GEOMDrugsMolSSD(Dataset):
    """GEOM-Drugs dataset preprocessed for MolSSD training.

    Preprocessing steps:
        1. Load molecules from GEOM rdkit_folder pickle files
        2. Select lowest-energy conformer per molecule
        3. Extract positions, atom types (10 types for drugs)
        4. Build molecular graph from RDKit bonds
        5. Subtract center of mass
        6. Precompute coarsening hierarchy (deeper for larger molecules)
        7. Cache everything for fast loading

    Args:
        root: Root directory for data storage.
        split: One of ``'train'``, ``'val'``, ``'test'``.
        transform: Optional transform applied at ``__getitem__`` time.
        max_coarsening_levels: Maximum coarsening hierarchy depth.
        max_atoms: Skip molecules larger than this.
        max_molecules: Cap dataset size (for faster iteration during dev).
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_coarsening_levels: int = 5,
        max_atoms: int = MAX_ATOMS_DEFAULT,
        max_molecules: Optional[int] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.root = root
        self.split = split
        self.transform = transform
        self.max_coarsening_levels = max_coarsening_levels
        self.max_atoms = max_atoms
        self.max_molecules = max_molecules

        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            logger.info("Loading cached %s split from %s", split, cache_path)
            self.data_list: List[GEOMDrugsData] = torch.load(
                cache_path, weights_only=False
            )
            if max_molecules and len(self.data_list) > max_molecules:
                self.data_list = self.data_list[:max_molecules]
        else:
            logger.info(
                "Processing GEOM-Drugs %s split (this may take a while)...",
                split,
            )
            self.data_list = self._process()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.data_list, cache_path)
            logger.info("Saved processed %s split to %s", split, cache_path)

    def _cache_path(self) -> str:
        return os.path.join(
            self.root,
            "molssd_processed",
            f"geom_drugs_{self.split}_levels{self.max_coarsening_levels}"
            f"_maxatoms{self.max_atoms}.pt",
        )

    def _find_raw_dir(self) -> str:
        """Locate the extracted GEOM rdkit_folder/drugs directory."""
        candidates = [
            os.path.join(self.root, "geom_drugs", "rdkit_folder", "drugs"),
            os.path.join(self.root, "geom_drugs", "drugs"),
            os.path.join(self.root, "rdkit_folder", "drugs"),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path

        # Try to extract from tarball
        tar_candidates = [
            os.path.join(self.root, "geom_drugs", "rdkit_folder.tar.gz"),
            os.path.join(self.root, "rdkit_folder.tar.gz"),
        ]
        for tar_path in tar_candidates:
            if os.path.exists(tar_path):
                extract_dir = os.path.join(self.root, "geom_drugs")
                logger.info("Extracting %s ...", tar_path)
                with tarfile.open(tar_path, "r:gz") as tar:
                    tar.extractall(path=extract_dir)
                # Re-check
                for path in candidates:
                    if os.path.isdir(path):
                        return path

        raise FileNotFoundError(
            "GEOM-Drugs raw data not found. Download rdkit_folder.tar.gz from "
            "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF "
            f"and place it in {self.root}/geom_drugs/"
        )

    def _process(self) -> List[GEOMDrugsData]:
        """Load and preprocess GEOM-Drugs molecules for this split."""
        raw_dir = self._find_raw_dir()

        # List all pickle files
        pkl_files = sorted([
            f for f in os.listdir(raw_dir) if f.endswith(".pickle")
        ])
        total = len(pkl_files)
        logger.info("Found %d molecule files in %s", total, raw_dir)

        # Determine split indices (60/20/20)
        train_end = int(total * TRAIN_FRAC)
        val_end = int(total * (TRAIN_FRAC + VAL_FRAC))

        if self.split == "train":
            files = pkl_files[:train_end]
        elif self.split == "val":
            files = pkl_files[train_end:val_end]
        else:
            files = pkl_files[val_end:]

        if self.max_molecules:
            files = files[:self.max_molecules]

        logger.info(
            "Split '%s': %d files (indices %s)",
            self.split, len(files),
            f"[0, {train_end})" if self.split == "train"
            else f"[{train_end}, {val_end})" if self.split == "val"
            else f"[{val_end}, {total})",
        )

        data_list: List[GEOMDrugsData] = []
        skipped = 0
        skipped_size = 0

        for i, fname in enumerate(files):
            if (i + 1) % 10000 == 0:
                logger.info(
                    "  Processing %d / %d (kept %d, skipped %d) ...",
                    i + 1, len(files), len(data_list), skipped,
                )

            fpath = os.path.join(raw_dir, fname)
            result = self._process_single_file(fpath)
            if result is not None:
                data_list.append(result)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(
                "Skipped %d molecules (unsupported atoms, too large, or "
                "degenerate graphs).", skipped,
            )

        logger.info(
            "Processed %d molecules for '%s' split.", len(data_list), self.split,
        )
        return data_list

    def _process_single_file(self, fpath: str) -> Optional[GEOMDrugsData]:
        """Process a single GEOM pickle file into GEOMDrugsData.

        Selects the lowest-energy conformer and extracts positions,
        atom types, bonds, and precomputes the coarsening hierarchy.
        """
        from rdkit import Chem

        try:
            with open(fpath, "rb") as f:
                mol_data = pickle.load(f)
        except Exception:
            return None

        # mol_data is typically a dict with 'conformers' key
        # Each conformer has 'rd_mol' (RDKit mol) and 'totalenergy' or 'boltzmannweight'
        if isinstance(mol_data, dict):
            conformers = mol_data.get("conformers", [])
            if not conformers:
                return None
            # Select lowest energy conformer
            best = min(
                conformers,
                key=lambda c: c.get("totalenergy", c.get("energy", float("inf"))),
            )
            mol = best.get("rd_mol")
            if mol is None:
                return None
        elif isinstance(mol_data, list):
            # Some formats store a list of RDKit mols directly
            if not mol_data:
                return None
            mol = mol_data[0]
        else:
            return None

        return self._process_rdkit_mol(mol)

    def _process_rdkit_mol(self, mol: Any) -> Optional[GEOMDrugsData]:
        """Process a single RDKit Mol object."""
        from rdkit import Chem

        if mol is None:
            return None

        num_atoms = mol.GetNumAtoms()
        if num_atoms < 2 or num_atoms > self.max_atoms:
            return None

        # 1. Extract positions
        try:
            conf = mol.GetConformer()
        except Exception:
            return None
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float32)

        # 2. Map atomic numbers to indices
        atom_types = torch.zeros(num_atoms, dtype=torch.long)
        for i in range(num_atoms):
            z_i = mol.GetAtomWithIdx(i).GetAtomicNum()
            if z_i not in ATOMIC_NUMBER_TO_INDEX:
                return None  # Unsupported atom type
            atom_types[i] = ATOMIC_NUMBER_TO_INDEX[z_i]

        # 3. Atomic masses
        atomic_masses = torch.tensor(
            [ATOMIC_MASSES[int(at.item())] for at in atom_types],
            dtype=torch.float32,
        )

        # 4. Build edge_index and adjacency from bonds
        bonds = mol.GetBonds()
        src, dst = [], []
        for bond in bonds:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([i, j])
            dst.extend([j, i])

        if len(src) == 0:
            # No bonds — fully connected fallback
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        src.append(i)
                        dst.append(j)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        adj = torch.zeros(num_atoms, num_atoms, dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = torch.max(adj, adj.t())

        # 5. Subtract center of mass
        total_mass = atomic_masses.sum()
        com = (atomic_masses.unsqueeze(1) * pos).sum(dim=0) / total_mass
        positions = pos - com.unsqueeze(0)

        # 6. Precompute coarsening hierarchy
        try:
            hierarchy = build_coarsening_hierarchy(
                adj=adj,
                num_atoms=num_atoms,
                target_sizes=None,
                atomic_masses=atomic_masses,
            )
        except Exception as e:
            logger.debug(
                "Coarsening failed for molecule with %d atoms: %s",
                num_atoms, e,
            )
            hierarchy = []

        if len(hierarchy) > self.max_coarsening_levels:
            hierarchy = hierarchy[:self.max_coarsening_levels]

        # 7. Precompute coarsened data
        coarsened_data = _precompute_coarsened(
            positions, atom_types, hierarchy
        )

        return GEOMDrugsData(
            positions=positions,
            atom_types=atom_types,
            edge_index=edge_index.long(),
            adj=adj,
            atomic_masses=atomic_masses,
            num_atoms=num_atoms,
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


# ---------------------------------------------------------------------------
# Collate function (reuses the same pattern as QM9)
# ---------------------------------------------------------------------------

def geom_drugs_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for GEOM-Drugs batches.

    Same structure as ``qm9_collate_fn`` — concatenates precomputed
    coarsened data at each level for vectorised forward diffusion.
    """
    positions_list = []
    atom_types_list = []
    edge_index_list = []
    batch_idx_list = []
    num_atoms_list = []
    adj_list = []
    atomic_masses_list = []
    coarsening_hierarchies = []

    # Determine max coarsening depth
    max_depth = 0
    for data in batch:
        cd = data.get("coarsened_data", [])
        max_depth = max(max_depth, len(cd))

    level_positions: List[List[torch.Tensor]] = [[] for _ in range(max_depth)]
    level_types: List[List[torch.Tensor]] = [[] for _ in range(max_depth)]
    level_edge_index: List[List[torch.Tensor]] = [[] for _ in range(max_depth)]
    level_batch_idx: List[List[torch.Tensor]] = [[] for _ in range(max_depth)]
    level_num_nodes: List[List[int]] = [[] for _ in range(max_depth)]
    level_offsets: List[int] = [0] * max_depth

    node_offset = 0
    for i, data in enumerate(batch):
        n = data["num_atoms"]
        positions_list.append(data["positions"])
        atom_types_list.append(data["atom_types"])

        ei = data["edge_index"].clone() + node_offset
        edge_index_list.append(ei)

        batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
        num_atoms_list.append(n)
        adj_list.append(data["adj"])
        atomic_masses_list.append(data["atomic_masses"])
        coarsening_hierarchies.append(data.get("coarsening_hierarchy", []))

        cd = data.get("coarsened_data", [])
        for lev in range(max_depth):
            if lev < len(cd):
                cdata = cd[lev]
                n_c = cdata.num_nodes
                level_positions[lev].append(cdata.positions)
                level_types[lev].append(cdata.atom_types)
                ei_c = cdata.edge_index + level_offsets[lev]
                level_edge_index[lev].append(ei_c)
            else:
                if len(cd) > 0:
                    last = cd[-1]
                    n_c = last.num_nodes
                    level_positions[lev].append(last.positions)
                    level_types[lev].append(last.atom_types)
                    ei_c = last.edge_index + level_offsets[lev]
                    level_edge_index[lev].append(ei_c)
                else:
                    n_c = n
                    level_positions[lev].append(data["positions"])
                    level_types[lev].append(data["atom_types"])
                    ei_c = data["edge_index"].clone() + level_offsets[lev]
                    level_edge_index[lev].append(ei_c)

            level_batch_idx[lev].append(
                torch.full((n_c,), i, dtype=torch.long)
            )
            level_num_nodes[lev].append(n_c)
            level_offsets[lev] += n_c

        node_offset += n

    coarsened_levels = []
    for lev in range(max_depth):
        if len(level_positions[lev]) > 0:
            coarsened_levels.append({
                "positions": torch.cat(level_positions[lev], dim=0),
                "atom_types": torch.cat(level_types[lev], dim=0),
                "edge_index": torch.cat(level_edge_index[lev], dim=1),
                "batch_idx": torch.cat(level_batch_idx[lev], dim=0),
                "num_nodes_list": level_num_nodes[lev],
            })
        else:
            coarsened_levels.append(None)

    return {
        "positions": torch.cat(positions_list, dim=0),
        "atom_types": torch.cat(atom_types_list, dim=0),
        "edge_index": torch.cat(edge_index_list, dim=1),
        "batch_idx": torch.cat(batch_idx_list, dim=0),
        "num_atoms_list": num_atoms_list,
        "adj_list": adj_list,
        "atomic_masses_list": atomic_masses_list,
        "coarsening_hierarchies": coarsening_hierarchies,
        "coarsened_levels": coarsened_levels,
        "batch_size": len(batch),
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def get_geom_drugs_splits(
    root: str = "./data",
    max_levels: int = 5,
    max_atoms: int = MAX_ATOMS_DEFAULT,
    max_molecules: Optional[int] = None,
    train_transform: Optional[Callable] = None,
) -> Tuple[GEOMDrugsMolSSD, GEOMDrugsMolSSD, GEOMDrugsMolSSD]:
    """Create GEOM-Drugs train/val/test datasets."""
    train_ds = GEOMDrugsMolSSD(
        root=root, split="train", transform=train_transform,
        max_coarsening_levels=max_levels, max_atoms=max_atoms,
        max_molecules=max_molecules,
    )
    val_ds = GEOMDrugsMolSSD(
        root=root, split="val", transform=None,
        max_coarsening_levels=max_levels, max_atoms=max_atoms,
        max_molecules=max_molecules,
    )
    test_ds = GEOMDrugsMolSSD(
        root=root, split="test", transform=None,
        max_coarsening_levels=max_levels, max_atoms=max_atoms,
        max_molecules=max_molecules,
    )
    return train_ds, val_ds, test_ds
