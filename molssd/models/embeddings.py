"""Embedding modules for MolSSD.

Provides time, resolution-level, and atom-type embeddings used to
condition the E(n)-equivariant denoising network.

Modules
-------
SinusoidalTimeEmbedding
    Deterministic sinusoidal positional encoding for the diffusion
    timestep, following the Transformer convention (Vaswani et al., 2017).
ResolutionEmbedding
    Learned embedding table for discrete coarsening resolution levels.
AtomTypeEmbedding
    Learned embedding table for atom element types.
TimestepResolutionEmbedding
    Combines timestep and resolution level into a single conditioning
    vector via concatenation and MLP projection.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# SinusoidalTimeEmbedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous diffusion timesteps.

    Produces a fixed (non-learned) embedding of shape ``(batch, dim)``
    using interleaved sine and cosine functions at geometrically spaced
    frequencies, identical to the scheme in *Attention Is All You Need*
    (Vaswani et al., 2017) and widely adopted in diffusion models
    (Ho et al., 2020; Song et al., 2021).

    Parameters
    ----------
    dim : int
        Embedding dimension.  Must be divisible by 2 so that the even
        indices can hold sine values and the odd indices cosine values.

    Raises
    ------
    ValueError
        If *dim* is not divisible by 2.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(
                f"SinusoidalTimeEmbedding requires dim divisible by 2, got {dim}."
            )
        self.dim = dim

        # Pre-compute log-spaced frequency denominators and register as a
        # buffer so they move with the module to the correct device/dtype.
        half = dim // 2
        freq_exponent = torch.arange(half, dtype=torch.float32) * (
            -math.log(10_000.0) / half
        )
        self.register_buffer("freq_exponent", freq_exponent, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        """Compute sinusoidal embedding for timesteps.

        Parameters
        ----------
        t : Tensor, shape ``(batch,)`` or ``(batch, 1)``
            Continuous timestep values (typically in ``[0, 1]`` or
            ``[0, T]``).

        Returns
        -------
        Tensor, shape ``(batch, dim)``
            Sinusoidal embedding vectors.
        """
        t = t.view(-1, 1).float()  # (B, 1)
        freqs = torch.exp(self.freq_exponent)  # (dim/2,)
        args = t * freqs.unsqueeze(0)  # (B, dim/2)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---------------------------------------------------------------------------
# ResolutionEmbedding
# ---------------------------------------------------------------------------

class ResolutionEmbedding(nn.Module):
    """Learned embedding for discrete coarsening resolution levels.

    Maps an integer resolution level ``k in {0, 1, ..., num_levels - 1}``
    to a dense vector of size ``dim``.

    Parameters
    ----------
    num_levels : int
        Total number of resolution levels in the coarsening hierarchy.
    dim : int
        Embedding vector dimension.
    """

    def __init__(self, num_levels: int, dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_levels, dim)

    def forward(self, level: Tensor) -> Tensor:
        """Look up resolution embeddings.

        Parameters
        ----------
        level : Tensor, shape ``(batch,)``
            Integer resolution level indices.

        Returns
        -------
        Tensor, shape ``(batch, dim)``
            Resolution embedding vectors.
        """
        return self.embedding(level)


# ---------------------------------------------------------------------------
# AtomTypeEmbedding
# ---------------------------------------------------------------------------

class AtomTypeEmbedding(nn.Module):
    """Learned embedding for atom element types.

    For QM9 the canonical type vocabulary is:
    ``{0: H, 1: C, 2: N, 3: O, 4: F, 5: coarsened/mixed}``.

    The extra "coarsened/mixed" type represents super-nodes produced by
    spectral graph coarsening that aggregate multiple element types.

    Parameters
    ----------
    num_types : int
        Size of the atom-type vocabulary.  Default ``6`` covers the five
        QM9 elements plus one coarsened/mixed type.
    dim : int
        Embedding vector dimension.
    """

    def __init__(self, num_types: int = 6, dim: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_types, dim)

    def forward(self, atom_types: Tensor) -> Tensor:
        """Look up atom-type embeddings.

        Parameters
        ----------
        atom_types : Tensor, shape ``(num_atoms,)``
            Integer atom-type indices.

        Returns
        -------
        Tensor, shape ``(num_atoms, dim)``
            Atom-type embedding vectors.
        """
        return self.embedding(atom_types)


# ---------------------------------------------------------------------------
# TimestepResolutionEmbedding
# ---------------------------------------------------------------------------

class TimestepResolutionEmbedding(nn.Module):
    """Combined timestep + resolution conditioning module.

    Produces a single conditioning vector by:

    1. Encoding the continuous timestep ``t`` with
       :class:`SinusoidalTimeEmbedding`.
    2. Encoding the discrete resolution level with
       :class:`ResolutionEmbedding`.
    3. Concatenating both embeddings and projecting through a 2-layer MLP
       with SiLU activation to produce a vector of size ``out_dim``.

    Parameters
    ----------
    time_dim : int
        Internal dimension of the sinusoidal time embedding (must be even).
    res_levels : int
        Number of discrete resolution levels.
    out_dim : int
        Dimension of the final conditioning vector.
    """

    def __init__(self, time_dim: int, res_levels: int, out_dim: int) -> None:
        super().__init__()

        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.res_emb = ResolutionEmbedding(res_levels, time_dim)

        # Projection MLP: concat(time_emb, res_emb) -> out_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: Tensor, res_level: Tensor) -> Tensor:
        """Compute the combined conditioning vector.

        Parameters
        ----------
        t : Tensor, shape ``(batch,)`` or ``(batch, 1)``
            Continuous diffusion timesteps.
        res_level : Tensor, shape ``(batch,)``
            Integer resolution level indices.

        Returns
        -------
        Tensor, shape ``(batch, out_dim)``
            Conditioning vector ready to be broadcast to per-node
            representations and injected into EGNN blocks.
        """
        t_enc = self.time_emb(t)  # (B, time_dim)
        r_enc = self.res_emb(res_level)  # (B, time_dim)
        combined = torch.cat([t_enc, r_enc], dim=-1)  # (B, 2 * time_dim)
        return self.mlp(combined)  # (B, out_dim)
