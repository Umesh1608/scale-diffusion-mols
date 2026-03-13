"""Molecular generation quality metrics following the EDM evaluation protocol.

Provides functions to assess generated molecule quality including atom/molecule
stability, validity, uniqueness, novelty, and distributional metrics (bond length
and bond angle Jensen-Shannon divergences).

References:
    - EDM evaluation: Hoogeboom et al., "Equivariant Diffusion for Molecule
      Generation in 3D" (ICML 2022)
    - Atom stability: based on standard valence rules for H, C, N, O, F
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdmolops
except ImportError:
    warnings.warn(
        "RDKit is not installed. Molecular metrics will not be available. "
        "Install with: conda install -c conda-forge rdkit"
    )
    Chem = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Atomic numbers for the standard atom types used in QM9 / EDM
ATOM_TYPE_TO_ELEMENT = {0: "H", 1: "C", 2: "N", 3: "O", 4: "F"}
ELEMENT_TO_ATOMIC_NUM = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}

# Allowed valences per element (standard organic chemistry valences)
ALLOWED_VALENCES: Dict[str, List[int]] = {
    "H": [1],
    "C": [4],
    "N": [3],
    "O": [2],
    "F": [1],
}

# Bond length thresholds (in Angstroms) for inferring bond orders from
# 3D coordinates. Based on typical covalent bond lengths +/- tolerance.
# Format: (element1, element2): [(max_distance, bond_order), ...]
# Ordered from shortest (highest bond order) to longest (single bond).
BOND_THRESHOLDS: Dict[Tuple[str, str], List[Tuple[float, int]]] = {
    # Carbon-Carbon
    ("C", "C"): [(1.22, 3), (1.40, 2), (1.78, 1)],   # triple, aromatic/double, single
    # Carbon-Nitrogen
    ("C", "N"): [(1.20, 3), (1.36, 2), (1.70, 1)],
    # Carbon-Oxygen
    ("C", "O"): [(1.16, 3), (1.32, 2), (1.60, 1)],
    # Carbon-Hydrogen
    ("C", "H"): [(1.30, 1)],
    # Carbon-Fluorine
    ("C", "F"): [(1.60, 1)],
    # Nitrogen-Nitrogen
    ("N", "N"): [(1.16, 3), (1.30, 2), (1.60, 1)],
    # Nitrogen-Oxygen
    ("N", "O"): [(1.24, 2), (1.55, 1)],
    # Nitrogen-Hydrogen
    ("N", "H"): [(1.20, 1)],
    # Oxygen-Oxygen
    ("O", "O"): [(1.30, 2), (1.55, 1)],
    # Oxygen-Hydrogen
    ("O", "H"): [(1.15, 1)],
    # Fluorine-Hydrogen
    ("F", "H"): [(1.10, 1)],
    # Nitrogen-Fluorine
    ("N", "F"): [(1.55, 1)],
    # Oxygen-Fluorine
    ("O", "F"): [(1.55, 1)],
    # Hydrogen-Hydrogen (rare but handle gracefully)
    ("H", "H"): [(0.90, 1)],
    # Fluorine-Fluorine
    ("F", "F"): [(1.55, 1)],
}

# Fallback maximum distance for any pair not in BOND_THRESHOLDS
_DEFAULT_MAX_BOND_DIST = 1.80


# ---------------------------------------------------------------------------
# Bond inference and molecule building
# ---------------------------------------------------------------------------

def _get_bond_order(
    elem1: str,
    elem2: str,
    distance: float,
) -> int:
    """Determine bond order from element pair and interatomic distance.

    Args:
        elem1: Element symbol of the first atom.
        elem2: Element symbol of the second atom.
        distance: Interatomic distance in Angstroms.

    Returns:
        Bond order (1, 2, or 3), or 0 if no bond is expected.
    """
    key = (elem1, elem2) if (elem1, elem2) in BOND_THRESHOLDS else (elem2, elem1)
    thresholds = BOND_THRESHOLDS.get(key)

    if thresholds is None:
        # Unknown pair: use generic distance cutoff for single bond
        return 1 if distance < _DEFAULT_MAX_BOND_DIST else 0

    for max_dist, order in thresholds:
        if distance < max_dist:
            return order
    return 0


def build_molecule(
    positions: np.ndarray,
    atom_types: np.ndarray,
    threshold: float = 1.8,
    atom_type_map: Optional[Dict[int, str]] = None,
) -> Optional[Any]:
    """Build an RDKit Mol object from 3D positions and atom types.

    Infers bonds from interatomic distances using element-pair-specific
    distance thresholds. Falls back to a generic single-bond threshold
    for unknown element pairs.

    Args:
        positions: Atom positions, shape (N, 3), in Angstroms.
        atom_types: Integer atom type indices, shape (N,).
        threshold: Generic maximum bond distance (used as fallback), in
            Angstroms. Default 1.8.
        atom_type_map: Mapping from integer type to element symbol.
            Defaults to the standard QM9 mapping {0: H, 1: C, 2: N, 3: O, 4: F}.

    Returns:
        RDKit Mol object with 3D coordinates, or None if construction fails.
    """
    if Chem is None:
        warnings.warn("RDKit not available, returning None")
        return None

    if atom_type_map is None:
        atom_type_map = ATOM_TYPE_TO_ELEMENT

    n_atoms = len(positions)
    if n_atoms == 0:
        return None

    # Map integer types to element symbols
    try:
        elements = [atom_type_map[int(t)] for t in atom_types]
    except KeyError as e:
        warnings.warn(f"Unknown atom type {e}, cannot build molecule")
        return None

    # Create editable molecule
    mol = Chem.RWMol()

    # Add atoms
    for elem in elements:
        atomic_num = ELEMENT_TO_ATOMIC_NUM.get(elem)
        if atomic_num is None:
            # Try generic periodic table lookup
            atom_obj = Chem.Atom(elem)
            atomic_num = atom_obj.GetAtomicNum()
            if atomic_num == 0:
                warnings.warn(f"Unknown element '{elem}', skipping molecule")
                return None
        mol.AddAtom(Chem.Atom(atomic_num))

    # Add bonds based on pairwise distances
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            bond_order = _get_bond_order(elements[i], elements[j], dist)

            if bond_order > 0:
                bond_type_map = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                }
                mol.AddBond(i, j, bond_type_map.get(bond_order, Chem.BondType.SINGLE))

    # Set 3D coordinates
    conf = Chem.Conformer(n_atoms)
    for i in range(n_atoms):
        conf.SetAtomPosition(i, positions[i].tolist())
    mol.AddConformer(conf, assignId=True)

    # Convert to regular Mol
    try:
        mol = mol.GetMol()
    except Exception:
        return None

    # Try to sanitize; return None if it fails
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Return the unsanitized mol -- caller can decide whether to use it
        pass

    return mol


# ---------------------------------------------------------------------------
# Atom and molecule stability
# ---------------------------------------------------------------------------

def check_atom_stability(
    mol: Any,
    allowed_valences: Optional[Dict[str, List[int]]] = None,
) -> Tuple[int, int]:
    """Check how many atoms in a molecule have chemically valid valence.

    An atom is considered stable if its total valence (sum of bond orders
    plus implicit/explicit hydrogens) matches one of its allowed valences.

    Args:
        mol: RDKit Mol object.
        allowed_valences: Mapping from element symbol to list of allowed
            total valences. Defaults to standard organic valences.

    Returns:
        Tuple of (num_stable_atoms, num_total_atoms).
    """
    if allowed_valences is None:
        allowed_valences = ALLOWED_VALENCES

    if mol is None:
        return (0, 0)

    num_stable = 0
    num_total = mol.GetNumAtoms()

    for atom in mol.GetAtoms():
        elem = atom.GetSymbol()
        valence = atom.GetTotalValence()
        valid_vals = allowed_valences.get(elem)
        if valid_vals is not None:
            if valence in valid_vals:
                num_stable += 1
        else:
            # Unknown element: consider stable if valence > 0
            if valence > 0:
                num_stable += 1

    return (num_stable, num_total)


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def compute_generation_metrics(
    generated_positions: List[np.ndarray],
    generated_types: List[np.ndarray],
    training_smiles: Optional[Set[str]] = None,
    atom_type_map: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """Compute comprehensive generation quality metrics.

    Follows the EDM evaluation protocol: atom stability, molecule stability,
    validity, uniqueness, and novelty.

    Args:
        generated_positions: List of position arrays, each shape (N_i, 3).
        generated_types: List of atom type arrays, each shape (N_i,).
        training_smiles: Set of canonical SMILES from the training set,
            used to compute novelty. If None, novelty is not computed.
        atom_type_map: Mapping from integer type to element symbol.

    Returns:
        Dictionary with keys:
            - atom_stability: Fraction of atoms with valid valence.
            - mol_stability: Fraction of molecules where all atoms are stable.
            - validity: Fraction of molecules parseable by RDKit.
            - uniqueness: Fraction of unique canonical SMILES among valid mols.
            - novelty: Fraction of valid unique SMILES not in training set
              (NaN if training_smiles is None).
            - num_generated: Total number of generated molecules.
            - num_valid: Number of valid molecules.
            - num_unique: Number of unique valid molecules.
    """
    if Chem is None:
        warnings.warn("RDKit not available; returning empty metrics")
        return {}

    n_generated = len(generated_positions)
    if n_generated == 0:
        return {
            "atom_stability": 0.0,
            "mol_stability": 0.0,
            "validity": 0.0,
            "uniqueness": 0.0,
            "novelty": float("nan"),
            "num_generated": 0,
            "num_valid": 0,
            "num_unique": 0,
        }

    total_stable_atoms = 0
    total_atoms = 0
    n_mol_stable = 0
    valid_smiles: List[str] = []

    for pos, types in zip(generated_positions, generated_types):
        mol = build_molecule(pos, types, atom_type_map=atom_type_map)

        if mol is None:
            continue

        # Atom stability
        n_stable, n_total = check_atom_stability(mol)
        total_stable_atoms += n_stable
        total_atoms += n_total

        if n_total > 0 and n_stable == n_total:
            n_mol_stable += 1

        # Validity: try to get canonical SMILES
        try:
            smiles = Chem.MolToSmiles(mol)
            if smiles and Chem.MolFromSmiles(smiles) is not None:
                valid_smiles.append(smiles)
        except Exception:
            pass

    # Compute metrics
    atom_stability = total_stable_atoms / max(total_atoms, 1)
    mol_stability = n_mol_stable / n_generated
    validity = len(valid_smiles) / n_generated

    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / max(len(valid_smiles), 1)

    if training_smiles is not None and len(unique_smiles) > 0:
        novel = unique_smiles - training_smiles
        novelty = len(novel) / len(unique_smiles)
    else:
        novelty = float("nan")

    return {
        "atom_stability": atom_stability,
        "mol_stability": mol_stability,
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "num_generated": n_generated,
        "num_valid": len(valid_smiles),
        "num_unique": len(unique_smiles),
    }


# ---------------------------------------------------------------------------
# Distributional metrics
# ---------------------------------------------------------------------------

def _extract_bond_lengths(mols: List[Any]) -> List[float]:
    """Extract all bond lengths from a list of RDKit Mol objects.

    Args:
        mols: List of RDKit Mol objects (None entries are skipped).

    Returns:
        List of bond lengths in Angstroms.
    """
    lengths: List[float] = []
    for mol in mols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
        except Exception:
            continue
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pos_i = np.array(conf.GetAtomPosition(i))
            pos_j = np.array(conf.GetAtomPosition(j))
            lengths.append(float(np.linalg.norm(pos_i - pos_j)))
    return lengths


def _extract_bond_angles(mols: List[Any]) -> List[float]:
    """Extract all bond angles from a list of RDKit Mol objects.

    For each atom with at least two neighbors, computes all pairwise
    bond angles (in degrees).

    Args:
        mols: List of RDKit Mol objects (None entries are skipped).

    Returns:
        List of bond angles in degrees.
    """
    angles: List[float] = []
    for mol in mols:
        if mol is None:
            continue
        try:
            conf = mol.GetConformer()
        except Exception:
            continue

        for atom in mol.GetAtoms():
            neighbors = atom.GetNeighbors()
            if len(neighbors) < 2:
                continue

            center_pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
            neighbor_positions = [
                np.array(conf.GetAtomPosition(n.GetIdx())) for n in neighbors
            ]

            # All pairwise angles through the central atom
            for ii in range(len(neighbor_positions)):
                for jj in range(ii + 1, len(neighbor_positions)):
                    v1 = neighbor_positions[ii] - center_pos
                    v2 = neighbor_positions[jj] - center_pos
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 < 1e-8 or norm2 < 1e-8:
                        continue
                    cos_angle = np.clip(
                        np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0
                    )
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    angles.append(float(angle_deg))
    return angles


def _histogram_js_divergence(
    values_p: List[float],
    values_q: List[float],
    n_bins: int = 100,
    range_min: Optional[float] = None,
    range_max: Optional[float] = None,
) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    Both distributions are discretized into histograms with the same
    bin edges before computing JSD.

    Args:
        values_p: Samples from the first distribution.
        values_q: Samples from the second distribution.
        n_bins: Number of histogram bins.
        range_min: Minimum value for histogram range. If None, uses
            the minimum across both distributions.
        range_max: Maximum value for histogram range. If None, uses
            the maximum across both distributions.

    Returns:
        Jensen-Shannon divergence (base 2). Returns NaN if either
        distribution is empty.
    """
    if len(values_p) == 0 or len(values_q) == 0:
        return float("nan")

    all_values = values_p + values_q
    if range_min is None:
        range_min = min(all_values)
    if range_max is None:
        range_max = max(all_values)

    # Histograms (density=True normalizes to a probability distribution)
    hist_p, bin_edges = np.histogram(
        values_p, bins=n_bins, range=(range_min, range_max), density=False
    )
    hist_q, _ = np.histogram(
        values_q, bins=n_bins, range=(range_min, range_max), density=False
    )

    # Convert to probability distributions (add small epsilon for stability)
    eps = 1e-10
    p = hist_p.astype(np.float64) + eps
    q = hist_q.astype(np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()

    # scipy's jensenshannon returns the JS *distance* (sqrt of divergence)
    # We square it to get the divergence
    js_dist = jensenshannon(p, q, base=2.0)
    return float(js_dist ** 2)


def bond_length_js_divergence(
    generated_mols: List[Any],
    reference_mols: List[Any],
    n_bins: int = 100,
) -> float:
    """Compute Jensen-Shannon divergence of bond length distributions.

    Compares the distribution of bond lengths between generated and
    reference molecules. Lower values indicate better agreement.

    Args:
        generated_mols: List of generated RDKit Mol objects.
        reference_mols: List of reference RDKit Mol objects.
        n_bins: Number of histogram bins for discretization.

    Returns:
        Jensen-Shannon divergence (base 2). Returns NaN if either
        set has no valid bonds.
    """
    gen_lengths = _extract_bond_lengths(generated_mols)
    ref_lengths = _extract_bond_lengths(reference_mols)

    return _histogram_js_divergence(
        gen_lengths,
        ref_lengths,
        n_bins=n_bins,
        range_min=0.5,
        range_max=3.0,
    )


def bond_angle_js_divergence(
    generated_mols: List[Any],
    reference_mols: List[Any],
    n_bins: int = 100,
) -> float:
    """Compute Jensen-Shannon divergence of bond angle distributions.

    Compares the distribution of bond angles (in degrees) between
    generated and reference molecules. Lower values indicate better
    agreement with reference geometry distributions.

    Args:
        generated_mols: List of generated RDKit Mol objects.
        reference_mols: List of reference RDKit Mol objects.
        n_bins: Number of histogram bins for discretization.

    Returns:
        Jensen-Shannon divergence (base 2). Returns NaN if either
        set has no valid bond angles.
    """
    gen_angles = _extract_bond_angles(generated_mols)
    ref_angles = _extract_bond_angles(reference_mols)

    return _histogram_js_divergence(
        gen_angles,
        ref_angles,
        n_bins=n_bins,
        range_min=0.0,
        range_max=180.0,
    )
