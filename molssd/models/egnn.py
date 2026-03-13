"""E(n)-Equivariant Graph Neural Network (EGNN) for MolSSD.

Implements the EGNN architecture from Satorras et al. (2021),
"E(n) Equivariant Graph Neural Networks" (arXiv:2102.09844),
adapted for score-based diffusion on molecular graphs with
time and resolution conditioning.

Key equivariance properties:
  - Position updates use difference vectors (x_i - x_j), preserving
    translation equivariance.
  - Feature updates use squared distances ||x_i - x_j||^2 (invariant
    scalars), not raw positions.
  - The network is equivariant to rotations, reflections, and translations
    of the input coordinates.
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
    """Return an activation module by name."""
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {list(_ACTIVATIONS.keys())}."
        )
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# Helper: 2-layer MLP
# ---------------------------------------------------------------------------

def _build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    act_fn: str = "silu",
) -> nn.Sequential:
    """Construct a 2-layer MLP: Linear -> Act -> Linear."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        _get_activation(act_fn),
        nn.Linear(hidden_dim, out_dim),
    )


# ---------------------------------------------------------------------------
# EGNNBlock
# ---------------------------------------------------------------------------

class EGNNBlock(nn.Module):
    """Single E(n)-equivariant message-passing block.

    Message computation::

        m_ij = phi_e(h_i || h_j || ||x_i - x_j||^2 || edge_attr || t_emb)

    Optional attention::

        att_ij = sigmoid(phi_att(m_ij))
        m_ij   = att_ij * m_ij

    Coordinate update (equivariant)::

        x_i' = x_i + (1 / (|N(i)| + 1)) * sum_j (x_i - x_j) * phi_x(m_ij)

    Feature update (invariant, with residual)::

        agg_i = sum_j m_ij
        h_i'  = h_i + phi_h(h_i || agg_i)

    Parameters
    ----------
    hidden_dim : int
        Dimension of node feature vectors *h*.
    edge_dim : int
        Dimension of edge attribute vectors. 0 means no edge attributes.
    time_dim : int
        Dimension of the time/conditioning embedding. 0 means no conditioning.
    act_fn : str
        Activation function name (``'silu'``, ``'relu'``, ``'gelu'``, etc.).
    attention : bool
        If ``True``, apply a learned scalar attention gate to messages.
    normalize : bool
        If ``True``, normalize the coordinate update by the number of
        neighbours plus one, which prevents exploding position shifts.
    """

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int = 0,
        time_dim: int = 0,
        act_fn: str = "silu",
        attention: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.time_dim = time_dim
        self.attention = attention
        self.normalize = normalize

        # ----- Edge message network phi_e -----
        # Input: h_i || h_j || dist^2 || edge_attr || t_emb
        edge_input_dim = 2 * hidden_dim + 1 + edge_dim + time_dim
        self.phi_e = _build_mlp(edge_input_dim, hidden_dim, hidden_dim, act_fn)

        # ----- Attention gate phi_att -----
        if self.attention:
            self.phi_att = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        # ----- Coordinate update network phi_x -----
        # Outputs a *scalar* weight per message, gated by tanh to [-1, 1].
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _get_activation(act_fn),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # ----- Node update network phi_h -----
        # Input: h_i || agg_i  ->  residual added to h_i
        self.phi_h = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            _get_activation(act_fn),
            nn.Linear(hidden_dim, hidden_dim),
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        t_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass of one EGNN block.

        Parameters
        ----------
        h : Tensor, shape ``(N, hidden_dim)``
            Node feature vectors.
        x : Tensor, shape ``(N, 3)``
            Node 3-D coordinates.
        edge_index : Tensor, shape ``(2, E)``
            Edge indices in COO format ``[src, dst]``.
        edge_attr : Tensor or None, shape ``(E, edge_dim)``
            Optional edge attributes (e.g., bond type encoding).
        t_emb : Tensor or None, shape ``(N, time_dim)``
            Optional per-node time/resolution conditioning embedding.
            When the conditioning is per-graph, the caller should expand
            it to per-node before passing it here.

        Returns
        -------
        h_out : Tensor, shape ``(N, hidden_dim)``
            Updated node features.
        x_out : Tensor, shape ``(N, 3)``
            Updated node coordinates.
        """
        src, dst = edge_index  # src -> dst (messages flow src -> dst)

        # --- Pairwise difference vectors and squared distances ---
        diff = x[src] - x[dst]  # (E, 3)
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)

        # --- Build edge message input ---
        msg_input_parts = [h[src], h[dst], dist_sq]

        if edge_attr is not None:
            msg_input_parts.append(edge_attr)

        if t_emb is not None:
            # Broadcast per-node time embedding onto edges (use source node).
            msg_input_parts.append(t_emb[src])

        msg_input = torch.cat(msg_input_parts, dim=-1)  # (E, edge_input_dim)

        # --- Compute messages ---
        m_ij = self.phi_e(msg_input)  # (E, hidden_dim)

        # --- Optional attention gating ---
        if self.attention:
            att = self.phi_att(m_ij)  # (E, 1)
            m_ij = att * m_ij  # (E, hidden_dim)

        # --- Coordinate update (equivariant) ---
        coord_weight = self.phi_x(m_ij)  # (E, 1)
        coord_shift = diff * coord_weight  # (E, 3)

        # Aggregate coordinate shifts per destination node.
        coord_agg = scatter(
            coord_shift, dst, dim=0, dim_size=x.size(0), reduce="sum"
        )  # (N, 3)

        if self.normalize:
            # Count number of incoming edges per node + 1 (avoid div-by-zero
            # and dampen the update).
            num_neighbours = scatter(
                torch.ones(dst.size(0), 1, device=x.device, dtype=x.dtype),
                dst,
                dim=0,
                dim_size=x.size(0),
                reduce="sum",
            )  # (N, 1)
            coord_agg = coord_agg / (num_neighbours + 1.0)

        x_out = x + coord_agg  # (N, 3)

        # --- Feature update (invariant, with residual) ---
        msg_agg = scatter(
            m_ij, dst, dim=0, dim_size=h.size(0), reduce="sum"
        )  # (N, hidden_dim)

        h_input = torch.cat([h, msg_agg], dim=-1)  # (N, 2 * hidden_dim)
        h_out = h + self.phi_h(h_input)  # residual connection

        return h_out, x_out


# ---------------------------------------------------------------------------
# EGNNStack
# ---------------------------------------------------------------------------

class EGNNStack(nn.Module):
    """Stack of :class:`EGNNBlock` layers with residual connections.

    Each block receives the *same* time embedding ``t_emb`` so that
    the conditioning signal is available at every depth.

    Parameters
    ----------
    hidden_dim : int
        Node feature dimension shared across all blocks.
    num_blocks : int
        Number of sequential EGNN blocks.
    edge_dim : int
        Dimension of edge attributes.
    time_dim : int
        Dimension of time/resolution conditioning.
    act_fn : str
        Activation function name.
    attention : bool
        Whether to use attention gating inside each block.
    normalize : bool
        Whether to normalize coordinate updates by neighbour count.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        edge_dim: int = 0,
        time_dim: int = 0,
        act_fn: str = "silu",
        attention: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                EGNNBlock(
                    hidden_dim=hidden_dim,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    act_fn=act_fn,
                    attention=attention,
                    normalize=normalize,
                )
                for _ in range(num_blocks)
            ]
        )

    # ------------------------------------------------------------------ #

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        t_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the full stack of EGNN blocks.

        Parameters
        ----------
        h : Tensor, shape ``(N, hidden_dim)``
            Initial node features.
        x : Tensor, shape ``(N, 3)``
            Initial node coordinates.
        edge_index : Tensor, shape ``(2, E)``
            Edge connectivity in COO format.
        edge_attr : Tensor or None, shape ``(E, edge_dim)``
            Optional edge attributes.
        t_emb : Tensor or None, shape ``(N, time_dim)``
            Optional time/resolution conditioning (per-node).

        Returns
        -------
        h : Tensor, shape ``(N, hidden_dim)``
            Final node features after all blocks.
        x : Tensor, shape ``(N, 3)``
            Final node coordinates after all blocks.
        """
        for block in self.blocks:
            # Each EGNNBlock already applies internal residual connections
            # (h_out = h + phi_h(...) and x_out = x + coord_agg), so we
            # pass the outputs directly to the next block.
            h, x = block(h, x, edge_index, edge_attr=edge_attr, t_emb=t_emb)

        return h, x
