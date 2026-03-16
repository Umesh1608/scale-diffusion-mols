"""E(n)-Equivariant Graph Neural Network (EGNN) for MolSSD.

Implements the EGNN architecture following the EDM (Hoogeboom et al., 2022)
implementation with critical fixes for diffusion training:
  - coord_diff normalization (prevents scale issues)
  - Xavier init with gain=0.001 on final coord layer (prevents collapse)
  - coords_range per layer (controls update magnitude)
  - phi_x receives raw features, not processed messages (independent gradient path)
  - No division by num_neighbours for coordinate updates

References:
    - Satorras et al. (2021), "E(n) Equivariant Graph Neural Networks"
    - Hoogeboom et al. (2022), "Equivariant Diffusion for Molecule Generation in 3D"
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter


# ---------------------------------------------------------------------------
# Activation lookup
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "mish": nn.Mish,
}


def _get_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS.keys())}."
        )
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# coord2diff: normalized coordinate differences (following EDM)
# ---------------------------------------------------------------------------

def _coord2diff(
    x: Tensor,
    edge_index: Tensor,
    norm_constant: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute normalized coordinate differences and squared distances.

    Following EDM: diff = (x_i - x_j) / (||x_i - x_j|| + norm_constant).
    This makes diff roughly unit-length for typical bond distances (~1-2 Å),
    preventing large coordinate shifts from distant atom pairs.

    Returns:
        dist_sq: Squared distances, shape (E, 1).
        diff_norm: Normalized difference vectors, shape (E, 3).
    """
    src, dst = edge_index
    diff = x[src] - x[dst]  # (E, 3)
    dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)
    norm = torch.sqrt(dist_sq + 1e-8)  # (E, 1)
    diff_norm = diff / (norm + norm_constant)  # (E, 3)
    return dist_sq, diff_norm


# ---------------------------------------------------------------------------
# EGNNBlock (following EDM architecture)
# ---------------------------------------------------------------------------

class EGNNBlock(nn.Module):
    """Single E(n)-equivariant message-passing block.

    Key differences from naive EGNN (matching EDM):
      - phi_x receives raw [h_src, h_dst, edge_attr], not processed messages
      - phi_x is 3-layer MLP with xavier_init(gain=0.001) on final layer
      - coord_diff is normalized by (||diff|| + norm_constant)
      - No division by num_neighbours for coordinate aggregation
      - coords_range parameter controls max coordinate shift per layer
      - tanh applied externally, then multiplied by coords_range

    Parameters
    ----------
    hidden_dim : int
        Node feature dimension.
    edge_dim : int
        Edge attribute dimension (0 = no edge attributes).
    time_dim : int
        Time/conditioning embedding dimension (0 = no conditioning).
    act_fn : str
        Activation function name.
    attention : bool
        Use attention gating on messages.
    coords_range : float
        Maximum coordinate shift per layer (EDM uses 15/n_layers).
    norm_constant : float
        Denominator offset for coord_diff normalization (EDM default: 1.0).
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        time_dim: int = 0,
        act_fn: str = "silu",
        attention: bool = True,
        coords_range: float = 2.5,
        norm_constant: float = 1.0,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.attention = attention
        self.coords_range = coords_range
        self.norm_constant = norm_constant

        # ----- Edge message network phi_e -----
        # Input: h_src || h_dst || dist^2 || edge_attr || t_emb
        edge_input_dim = 2 * hidden_dim + 1 + edge_dim + time_dim
        self.phi_e = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            _get_activation(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ----- Attention gate -----
        if self.attention:
            self.phi_att = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        # ----- Coordinate update network phi_x -----
        # Following EDM: 3-layer MLP on raw [h_src, h_dst, edge_attr]
        # Outputs scalar weight per edge. Final layer: xavier_init(gain=0.001), no bias.
        coord_input_dim = 2 * hidden_dim + edge_dim + time_dim
        final_layer = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.xavier_uniform_(final_layer.weight, gain=0.001)
        self.phi_x = nn.Sequential(
            nn.Linear(coord_input_dim, hidden_dim),
            _get_activation(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
            _get_activation(act_fn),
            final_layer,
        )

        # ----- Node update network phi_h -----
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            _get_activation(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        t_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of one EGNN block.

        Returns (h_out, x_out) with updated features and coordinates.
        """
        src, dst = edge_index

        # --- Normalized coordinate differences (EDM-style) ---
        dist_sq, diff_norm = _coord2diff(x, edge_index, self.norm_constant)

        # --- Build edge message input ---
        msg_parts = [h[src], h[dst], dist_sq]
        if edge_attr is not None:
            msg_parts.append(edge_attr)
        if t_emb is not None:
            msg_parts.append(t_emb[src])
        msg_input = torch.cat(msg_parts, dim=-1)

        # --- Compute messages ---
        m_ij = self.phi_e(msg_input)  # (E, hidden_dim)

        if self.attention:
            att = self.phi_att(m_ij)
            m_ij = att * m_ij

        # --- Coordinate update (EDM-style) ---
        # phi_x receives raw features [h_src, h_dst, edge_attr, t_emb]
        # (independent gradient path from message network)
        coord_parts = [h[src], h[dst]]
        if edge_attr is not None:
            coord_parts.append(edge_attr)
        if t_emb is not None:
            coord_parts.append(t_emb[src])
        coord_input = torch.cat(coord_parts, dim=-1)

        # tanh applied externally, scaled by coords_range
        coord_weight = torch.tanh(self.phi_x(coord_input))  # (E, 1) in [-1, 1]
        coord_shift = diff_norm * coord_weight * self.coords_range  # (E, 3)

        # Aggregate (sum, NO division by num_neighbours — following EDM)
        coord_agg = scatter(
            coord_shift, dst, dim=0, dim_size=x.size(0), reduce="sum"
        )

        x_out = x + coord_agg

        # --- Feature update (invariant, with residual) ---
        msg_agg = scatter(
            m_ij, dst, dim=0, dim_size=h.size(0), reduce="sum"
        )
        h_input = torch.cat([h, msg_agg], dim=-1)
        h_out = h + self.phi_h(h_input)

        return h_out, x_out


# ---------------------------------------------------------------------------
# EGNNStack
# ---------------------------------------------------------------------------

class EGNNStack(nn.Module):
    """Stack of EGNNBlock layers.

    Parameters
    ----------
    hidden_dim : int
        Node feature dimension.
    num_blocks : int
        Number of sequential EGNN blocks.
    edge_dim : int
        Edge attribute dimension.
    time_dim : int
        Time/conditioning embedding dimension.
    act_fn : str
        Activation function name.
    attention : bool
        Use attention gating.
    normalize : bool
        Ignored (kept for backward compat). EDM-style normalization is
        always used (coord_diff normalization, not neighbour division).
    total_coords_range : float
        Total coordinate range budget, divided equally among layers.
        EDM default: 15.0 with 6 layers → 2.5 per layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        edge_dim: int = 0,
        time_dim: int = 0,
        act_fn: str = "silu",
        attention: bool = True,
        normalize: bool = True,  # ignored, kept for compat
        total_coords_range: float = 15.0,
    ) -> None:
        super().__init__()

        coords_range_per_layer = total_coords_range / max(num_blocks, 1)

        self.blocks = nn.ModuleList([
            EGNNBlock(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                time_dim=time_dim,
                act_fn=act_fn,
                attention=attention,
                coords_range=coords_range_per_layer,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        t_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the full stack of EGNN blocks."""
        for block in self.blocks:
            h, x = block(h, x, edge_index, edge_attr=edge_attr, t_emb=t_emb)
        return h, x
