"""QM9 dataset loader for MolSSD training.

Loads the QM9 dataset from PyTorch Geometric, preprocesses each molecule
for the MolSSD framework (center-of-mass subtraction, adjacency matrix
construction, coarsening hierarchy precomputation), and provides standard
train/val/test splits.

Preprocessing steps per molecule:
    1. Load from ``torch_geometric.datasets.QM9``
    2. Extract positions (``data.pos``) and atomic numbers (``data.z``)
    3. Map atomic numbers to the MolSSD 0-4 index: H(1)->0, C(6)->1,
       N(7)->2, O(8)->3, F(9)->4
    4. Build a dense adjacency matrix from ``data.edge_index``
    5. Subtract center of mass (translation invariance)
    6. Precompute the spectral coarsening hierarchy
    7. Cache the processed data for fast loading

References:
    - QM9 dataset: Ramakrishnan et al., "Quantum chemistry structures and
      properties of 134 kilo molecules" (2014)
    - EDM splits: Hoogeboom et al., "Equivariant Diffusion for Molecule
      Generation in 3D" (ICML 2022)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from molssd.core.coarsening import (
    CoarseningLevel,
    build_coarsening_hierarchy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Mapping from QM9 atomic numbers to MolSSD 0-4 index
ATOMIC_NUMBER_TO_INDEX = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
INDEX_TO_ATOMIC_NUMBER = {v: k for k, v in ATOMIC_NUMBER_TO_INDEX.items()}
INDEX_TO_SYMBOL = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}
NUM_ATOM_TYPES = 5

# Atomic masses in atomic mass units (Daltons)
ATOMIC_MASSES = {
    0: 1.008,    # H
    1: 12.011,   # C
    2: 14.007,   # N
    3: 15.999,   # O
    4: 18.998,   # F
}

# Standard EDM-style split sizes for QM9 (~130,831 molecules total)
# First 100K for training, next 17,831 for validation, last 13,000 for test
TRAIN_SIZE = 100_000
VAL_SIZE = 17_831


# ---------------------------------------------------------------------------
# RandomRotation transform
# ---------------------------------------------------------------------------

class RandomRotation:
    """Applies a random SO(3) rotation to atom positions.

    Generates a uniformly random rotation matrix using QR decomposition
    of a random Gaussian matrix, then multiplies positions by it.
    This is a data augmentation transform that leverages the rotational
    equivariance of the model.
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random rotation to the ``positions`` field.

        Args:
            data: Dictionary containing at least a ``positions`` key
                with a tensor of shape ``(N, 3)``.

        Returns:
            The same dictionary with rotated positions.
        """
        positions = data["positions"]
        R = self._random_rotation_matrix(device=positions.device, dtype=positions.dtype)
        data["positions"] = positions @ R.T
        # Also rotate coarsened positions in the hierarchy if they exist
        if "coarsening_hierarchy" in data:
            for level in data["coarsening_hierarchy"]:
                if hasattr(level, "coarsened_positions"):
                    level.coarsened_positions = level.coarsened_positions @ R.T
        return data

    @staticmethod
    def _random_rotation_matrix(
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate a uniformly random SO(3) rotation matrix.

        Uses QR decomposition of a random Gaussian matrix, with sign
        correction to ensure a proper rotation (det = +1).

        Returns:
            Rotation matrix of shape ``(3, 3)``.
        """
        mat = torch.randn(3, 3, device=device, dtype=dtype)
        q, r = torch.linalg.qr(mat)
        # Ensure proper rotation (det = +1) by correcting signs
        d = torch.diag(torch.sign(torch.diag(r)))
        q = q @ d
        if torch.det(q) < 0:
            q[:, 0] *= -1
        return q


# ---------------------------------------------------------------------------
# MolSSD Data container
# ---------------------------------------------------------------------------

@dataclass
class CoarsenedData:
    """Precomputed data at one coarsened resolution level.

    Attributes:
        positions: Coarsened positions, shape ``(N_k, 3)``.
        atom_types: Majority-vote atom types, shape ``(N_k,)``.
        edge_index: COO edges, shape ``(2, E_k)``.
        num_nodes: Number of super-nodes at this level.
    """
    positions: torch.Tensor
    atom_types: torch.Tensor
    edge_index: torch.Tensor
    num_nodes: int


@dataclass
class MolSSDData:
    """Container for a single preprocessed molecule.

    Attributes:
        positions: Atom coordinates with center of mass subtracted,
            shape ``(N, 3)``.
        atom_types: Integer atom type indices (0-4), shape ``(N,)``.
        edge_index: Bond connectivity in COO format, shape ``(2, E)``.
        adj: Dense adjacency matrix, shape ``(N, N)``.
        atomic_masses: Per-atom masses, shape ``(N,)``.
        num_atoms: Number of atoms in this molecule.
        coarsening_hierarchy: Precomputed multi-level coarsening hierarchy.
        coarsened_data: Precomputed coarsened positions/types/edges per level.
    """
    positions: torch.Tensor
    atom_types: torch.Tensor
    edge_index: torch.Tensor
    adj: torch.Tensor
    atomic_masses: torch.Tensor
    num_atoms: int
    coarsening_hierarchy: List[CoarseningLevel] = field(default_factory=list)
    coarsened_data: List[CoarsenedData] = field(default_factory=list)


def _precompute_coarsened(
    positions: torch.Tensor,
    atom_types: torch.Tensor,
    hierarchy: List[CoarseningLevel],
    num_atom_types: int = 5,
) -> List[CoarsenedData]:
    """Precompute coarsened positions, types, and edges at each level.

    Done once during dataset preprocessing so the collate function and
    trainer don't need to redo it every batch.
    """
    result: List[CoarsenedData] = []
    cur_pos = positions
    cur_types = atom_types

    for level in hierarchy:
        C = level.coarsening_matrix
        coarsened_pos = C @ cur_pos
        coarsened_types = _coarsen_types_majority(
            cur_types, level.cluster_assignment, level.num_nodes, num_atom_types
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


def _coarsen_types_majority(
    atom_types: torch.Tensor,
    cluster_assignment: torch.Tensor,
    num_clusters: int,
    num_atom_types: int = 5,
) -> torch.Tensor:
    """Coarsen atom types via majority vote (vectorised, no loop)."""
    device = atom_types.device
    one_hot = torch.zeros(atom_types.shape[0], num_atom_types, device=device)
    one_hot.scatter_(1, atom_types.unsqueeze(1), 1.0)
    counts = torch.zeros(num_clusters, num_atom_types, device=device)
    counts.index_add_(0, cluster_assignment, one_hot)
    return counts.argmax(dim=1)


# ---------------------------------------------------------------------------
# QM9 Dataset
# ---------------------------------------------------------------------------

class QM9MolSSD(Dataset):
    """QM9 dataset preprocessed for MolSSD training.

    Preprocessing steps:
        1. Load QM9 from ``torch_geometric.datasets.QM9``
        2. Extract positions, atom types (H=0, C=1, N=2, O=3, F=4)
        3. Build molecular graph (edge_index from bonds)
        4. Subtract center of mass (translation invariance)
        5. Precompute coarsening hierarchy for each molecule
        6. Cache everything for fast loading

    Args:
        root: Root directory for the raw/processed data.
        split: One of ``'train'``, ``'val'``, ``'test'``.
        transform: Optional callable transform applied to each sample
            at ``__getitem__`` time (e.g., :class:`RandomRotation`).
        max_coarsening_levels: Maximum number of coarsening levels to
            compute in the hierarchy.
        coarsening_ratio: Fold-reduction per coarsening level (default 3).
    """

    def __init__(
        self,
        root: str = "./data",
        split: str = "train",
        transform: Optional[Callable] = None,
        max_coarsening_levels: int = 3,
        coarsening_ratio: int = 3,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{split}'"
            )

        self.root = root
        self.split = split
        self.transform = transform
        self.max_coarsening_levels = max_coarsening_levels
        self.coarsening_ratio = coarsening_ratio

        # Build or load processed data
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            logger.info("Loading cached %s split from %s", split, cache_path)
            self.data_list: List[MolSSDData] = torch.load(
                cache_path, weights_only=False
            )
        else:
            logger.info("Processing QM9 %s split (this may take a few minutes)...", split)
            self.data_list = self._process()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.data_list, cache_path)
            logger.info("Saved processed %s split to %s", split, cache_path)

    def _cache_path(self) -> str:
        """Return the path for the cached processed data file."""
        return os.path.join(
            self.root,
            "molssd_processed",
            f"qm9_{self.split}_levels{self.max_coarsening_levels}"
            f"_ratio{self.coarsening_ratio}.pt",
        )

    def _process(self) -> List[MolSSDData]:
        """Load raw QM9 and preprocess all molecules for this split.

        Reads the SDF file directly with RDKit (bypassing PyG's QM9 class
        which can crash on unparseable molecules) and applies the standard
        EDM-style splits.

        Returns:
            List of :class:`MolSSDData` objects.
        """
        from rdkit import Chem

        sdf_path = os.path.join(self.root, "qm9_raw", "raw", "gdb9.sdf")
        unchar_path = os.path.join(self.root, "qm9_raw", "raw", "uncharacterized.txt")

        if not os.path.exists(sdf_path):
            # Trigger PyG download only (will fail on process, but raw files
            # will already be on disk for us).
            try:
                from torch_geometric.datasets import QM9 as PyGQM9
                PyGQM9(root=os.path.join(self.root, "qm9_raw"))
            except Exception:
                pass
            if not os.path.exists(sdf_path):
                raise FileNotFoundError(
                    f"QM9 SDF file not found at {sdf_path}. "
                    "Download QM9 manually or ensure PyG can download it."
                )

        # Read the skip list (uncharacterized molecules)
        skip_indices: set = set()
        if os.path.exists(unchar_path):
            with open(unchar_path) as f:
                for line in f.read().split("\n")[9:-2]:
                    parts = line.split()
                    if parts:
                        skip_indices.add(int(parts[0]) - 1)

        # Read all valid molecules from the SDF
        logger.info("Reading QM9 SDF with RDKit ...")
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

        all_mols = []
        for i, mol in enumerate(suppl):
            if i in skip_indices:
                continue
            if mol is None:
                continue
            all_mols.append(mol)

        total = len(all_mols)
        logger.info("Read %d valid molecules from QM9 SDF.", total)

        # Determine split indices (EDM-style)
        train_end = min(TRAIN_SIZE, total)
        val_end = min(TRAIN_SIZE + VAL_SIZE, total)

        if self.split == "train":
            indices = range(0, train_end)
        elif self.split == "val":
            indices = range(train_end, val_end)
        else:  # test
            indices = range(val_end, total)

        logger.info(
            "Split '%s': indices [%d, %d) (%d molecules)",
            self.split, indices.start, indices.stop, len(indices),
        )

        data_list: List[MolSSDData] = []
        skipped = 0

        for idx in indices:
            result = self._process_single_rdkit(all_mols[idx])
            if result is not None:
                data_list.append(result)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(
                "Skipped %d molecules during processing (unsupported atoms "
                "or degenerate graphs).", skipped,
            )

        logger.info("Processed %d molecules for '%s' split.", len(data_list), self.split)
        return data_list

    def _process_single_rdkit(self, mol: Any) -> Optional[MolSSDData]:
        """Process a single RDKit Mol into MolSSDData.

        Args:
            mol: An ``rdkit.Chem.Mol`` object.

        Returns:
            :class:`MolSSDData` or ``None`` if the molecule contains
            unsupported atom types.
        """
        from rdkit import Chem

        num_atoms = mol.GetNumAtoms()
        if num_atoms < 2:
            return None

        # 1. Extract atomic numbers and positions
        conf = mol.GetConformer()
        positions_np = conf.GetPositions()  # (N, 3) numpy
        pos = torch.tensor(positions_np, dtype=torch.float32)

        atom_types = torch.zeros(num_atoms, dtype=torch.long)
        for i in range(num_atoms):
            z_i = mol.GetAtomWithIdx(i).GetAtomicNum()
            if z_i not in ATOMIC_NUMBER_TO_INDEX:
                return None
            atom_types[i] = ATOMIC_NUMBER_TO_INDEX[z_i]

        # 2. Build atomic masses tensor
        atomic_masses = torch.tensor(
            [ATOMIC_MASSES[int(at.item())] for at in atom_types],
            dtype=torch.float32,
        )

        # 3. Build edge_index and dense adjacency from bonds
        bonds = mol.GetBonds()
        src, dst = [], []
        for bond in bonds:
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([i, j])
            dst.extend([j, i])

        if len(src) == 0:
            # No bonds -- build fully connected graph as fallback
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        src.append(i)
                        dst.append(j)

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        adj = torch.zeros(num_atoms, num_atoms, dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = torch.max(adj, adj.t())

        # 4. Subtract center of mass (mass-weighted)
        total_mass = atomic_masses.sum()
        com = (atomic_masses.unsqueeze(1) * pos).sum(dim=0) / total_mass
        positions = pos - com.unsqueeze(0)

        # 5. Precompute coarsening hierarchy
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
            hierarchy = hierarchy[: self.max_coarsening_levels]

        # 6. Precompute coarsened data at each level
        coarsened_data = _precompute_coarsened(
            positions.float(), atom_types, hierarchy
        )

        return MolSSDData(
            positions=positions.float(),
            atom_types=atom_types,
            edge_index=edge_index.long(),
            adj=adj,
            atomic_masses=atomic_masses,
            num_atoms=num_atoms,
            coarsening_hierarchy=hierarchy,
            coarsened_data=coarsened_data,
        )

    def _process_single(self, pyg_data: Any) -> Optional[MolSSDData]:
        """Process a single PyG Data object into MolSSDData.

        Args:
            pyg_data: A ``torch_geometric.data.Data`` object from the QM9
                dataset.

        Returns:
            :class:`MolSSDData` or ``None`` if the molecule contains
            unsupported atom types.
        """
        # 1. Extract atomic numbers and positions
        z = pyg_data.z  # (N,) int tensor of atomic numbers
        pos = pyg_data.pos  # (N, 3) float tensor

        if z is None or pos is None:
            return None

        num_atoms = z.shape[0]
        if num_atoms < 2:
            return None

        # 2. Map atomic numbers to 0-4 index
        atom_types = torch.zeros(num_atoms, dtype=torch.long)
        for i in range(num_atoms):
            z_i = int(z[i].item())
            if z_i not in ATOMIC_NUMBER_TO_INDEX:
                # Unsupported atom type
                return None
            atom_types[i] = ATOMIC_NUMBER_TO_INDEX[z_i]

        # 3. Build atomic masses tensor
        atomic_masses = torch.tensor(
            [ATOMIC_MASSES[int(at.item())] for at in atom_types],
            dtype=torch.float32,
        )

        # 4. Build edge_index and dense adjacency matrix
        edge_index = pyg_data.edge_index  # (2, E)
        if edge_index is None or edge_index.shape[1] == 0:
            # No bonds -- build fully connected graph as fallback
            src = []
            dst = []
            for i in range(num_atoms):
                for j in range(num_atoms):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            edge_index = torch.tensor([src, dst], dtype=torch.long)

        adj = torch.zeros(num_atoms, num_atoms, dtype=torch.float32)
        adj[edge_index[0], edge_index[1]] = 1.0
        # Ensure symmetry
        adj = torch.max(adj, adj.t())

        # 5. Subtract center of mass (mass-weighted)
        total_mass = atomic_masses.sum()
        com = (atomic_masses.unsqueeze(1) * pos).sum(dim=0) / total_mass
        positions = pos - com.unsqueeze(0)

        # 6. Precompute coarsening hierarchy
        try:
            hierarchy = build_coarsening_hierarchy(
                adj=adj,
                num_atoms=num_atoms,
                target_sizes=None,  # Use automatic 3-fold reduction
                atomic_masses=atomic_masses,
            )
        except Exception as e:
            logger.debug(
                "Coarsening failed for molecule with %d atoms: %s",
                num_atoms, e,
            )
            # Still include the molecule but with an empty hierarchy
            hierarchy = []

        # Truncate hierarchy to max_coarsening_levels
        if len(hierarchy) > self.max_coarsening_levels:
            hierarchy = hierarchy[: self.max_coarsening_levels]

        coarsened_data = _precompute_coarsened(
            positions.float(), atom_types, hierarchy
        )

        return MolSSDData(
            positions=positions.float(),
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
        """Return a preprocessed molecule as a dictionary.

        Returns:
            Dictionary with keys:
                - ``positions``: ``(N, 3)`` float tensor
                - ``atom_types``: ``(N,)`` long tensor (0-4)
                - ``edge_index``: ``(2, E)`` long tensor
                - ``adj``: ``(N, N)`` float tensor
                - ``atomic_masses``: ``(N,)`` float tensor
                - ``num_atoms``: int
                - ``coarsening_hierarchy``: ``List[CoarseningLevel]``
        """
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
# Collate function
# ---------------------------------------------------------------------------

def qm9_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for QM9MolSSD batches.

    Concatenates precomputed coarsened data (positions, types, edges) at
    each hierarchy level from the dataset so the trainer uses fully
    vectorised forward diffusion at all resolutions.

    Args:
        batch: List of dictionaries, each from :meth:`QM9MolSSD.__getitem__`.

    Returns:
        Dictionary with level-0 data and ``coarsened_levels`` list.
    """
    positions_list = []
    atom_types_list = []
    edge_index_list = []
    batch_idx_list = []
    num_atoms_list = []
    adj_list = []
    atomic_masses_list = []
    coarsening_hierarchies = []

    # Determine max coarsening depth across the batch
    max_depth = 0
    for data in batch:
        cd = data.get("coarsened_data", [])
        max_depth = max(max_depth, len(cd))

    # Per-level accumulators (just concatenation — no computation)
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

        # Concatenate precomputed coarsened data per level
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
                # Molecule has fewer levels; use last available
                if len(cd) > 0:
                    last = cd[-1]
                    n_c = last.num_nodes
                    level_positions[lev].append(last.positions)
                    level_types[lev].append(last.atom_types)
                    ei_c = last.edge_index + level_offsets[lev]
                    level_edge_index[lev].append(ei_c)
                else:
                    # No coarsening at all; use full resolution
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

    # Build coarsened level dicts
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
# Convenience: get all splits
# ---------------------------------------------------------------------------

def get_qm9_splits(
    root: str = "./data",
    max_levels: int = 3,
    ratio: int = 3,
    train_transform: Optional[Callable] = None,
) -> Tuple[QM9MolSSD, QM9MolSSD, QM9MolSSD]:
    """Create QM9 train/val/test datasets for MolSSD.

    Args:
        root: Root directory for data storage.
        max_levels: Maximum coarsening hierarchy depth.
        ratio: Coarsening fold-reduction ratio.
        train_transform: Optional transform for the training set (e.g.,
            :class:`RandomRotation`). Validation and test sets receive
            no transform.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    train_ds = QM9MolSSD(
        root=root,
        split="train",
        transform=train_transform,
        max_coarsening_levels=max_levels,
        coarsening_ratio=ratio,
    )
    val_ds = QM9MolSSD(
        root=root,
        split="val",
        transform=None,
        max_coarsening_levels=max_levels,
        coarsening_ratio=ratio,
    )
    test_ds = QM9MolSSD(
        root=root,
        split="test",
        transform=None,
        max_coarsening_levels=max_levels,
        coarsening_ratio=ratio,
    )
    return train_ds, val_ds, test_ds
