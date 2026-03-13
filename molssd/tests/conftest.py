"""Shared test fixtures for MolSSD tests."""
import torch
import pytest
import numpy as np


@pytest.fixture
def seed():
    """Set random seeds for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def small_molecule():
    """A small molecule graph (water-like, 3 atoms: O, H, H).

    Returns dict with positions, atom_types, adjacency, atomic_masses.
    """
    positions = torch.tensor([
        [0.0, 0.0, 0.0],     # O
        [0.757, 0.586, 0.0],  # H
        [-0.757, 0.586, 0.0], # H
    ], dtype=torch.float32)

    atom_types = torch.tensor([1, 0, 0], dtype=torch.long)  # O=1, H=0

    adj = torch.tensor([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=torch.float32)

    atomic_masses = torch.tensor([15.999, 1.008, 1.008], dtype=torch.float32)

    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=torch.long)

    return {
        'positions': positions,
        'atom_types': atom_types,
        'adj': adj,
        'atomic_masses': atomic_masses,
        'edge_index': edge_index,
        'num_atoms': 3,
    }


@pytest.fixture
def medium_molecule():
    """A medium molecule graph (benzene-like, 12 atoms: 6C + 6H).

    Returns dict with positions, atom_types, adjacency, atomic_masses.
    """
    angles = torch.linspace(0, 2 * np.pi, 7)[:-1]
    r_c = 1.4
    r_h = 2.48

    c_pos = torch.stack([r_c * torch.cos(angles), r_c * torch.sin(angles),
                         torch.zeros(6)], dim=1)
    h_pos = torch.stack([r_h * torch.cos(angles), r_h * torch.sin(angles),
                         torch.zeros(6)], dim=1)

    positions = torch.cat([c_pos, h_pos], dim=0)
    atom_types = torch.cat([torch.ones(6, dtype=torch.long),
                            torch.zeros(6, dtype=torch.long)])

    n = 12
    adj = torch.zeros(n, n)
    for i in range(6):
        adj[i, (i + 1) % 6] = 1
        adj[(i + 1) % 6, i] = 1
        adj[i, i + 6] = 1
        adj[i + 6, i] = 1

    atomic_masses = torch.cat([
        torch.full((6,), 12.011),
        torch.full((6,), 1.008),
    ])

    src, dst = torch.where(adj > 0)
    edge_index = torch.stack([src, dst], dim=0)

    return {
        'positions': positions,
        'atom_types': atom_types,
        'adj': adj,
        'atomic_masses': atomic_masses,
        'edge_index': edge_index,
        'num_atoms': 12,
    }


@pytest.fixture
def random_rotation():
    """Generate a random 3D rotation matrix."""
    mat = torch.randn(3, 3)
    q, r = torch.linalg.qr(mat)
    d = torch.diag(torch.sign(torch.diag(r)))
    q = q @ d
    if torch.det(q) < 0:
        q[:, 0] *= -1
    return q


@pytest.fixture
def random_translation():
    """Generate a random 3D translation vector."""
    return torch.randn(3)


@pytest.fixture
def batch_of_molecules(small_molecule, medium_molecule):
    """A batch containing both the small and medium molecule."""
    return [small_molecule, medium_molecule]
