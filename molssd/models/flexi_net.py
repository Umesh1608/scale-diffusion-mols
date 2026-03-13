"""Molecular Flexi-Net: hierarchical E(n)-equivariant denoising network for MolSSD.

Implements the denoising backbone of the Scale-Space Diffusion framework.
Molecular Flexi-Net is a multi-resolution architecture where each coarsening
level has its own dedicated EGNN block stack with level-specific capacity
(hidden dimension and depth). At each diffusion timestep, only the level
corresponding to the current resolution is active, achieving computational
savings by processing fewer nodes at coarser resolutions while using larger
hidden dimensions for richer representations.

Architecture overview::

    Level 0 (finest, full atomic):   4 blocks, hidden_dim=128
    Level 1:                         3 blocks, hidden_dim=256
    Level 2:                         3 blocks, hidden_dim=384
    Level 3 (coarsest, centroid):    2 blocks, hidden_dim=512

Key design principles:
    - **Dynamic activation**: at timestep t with resolution k, only level k's
      EGNN stack is executed.
    - **E(n)-equivariant**: all operations (message passing, pooling, unpooling)
      preserve equivariance to rotations, reflections, and translations.
    - **Resolution-conditioned**: the timestep+resolution embedding is injected
      into every EGNN block so the network knows which scale it operates at.

Classes
-------
PoolingLayer
    Pools features and positions from a fine level to a coarser level using
    the coarsening matrix (center-of-mass for positions, scatter aggregation
    for features).
UnpoolingLayer
    Lifts features and positions from a coarse level back to a finer level
    using the pseudoinverse of the coarsening matrix.
ZeroSkipConnection
    Learnable gated skip connection initialized to zero, preventing
    discontinuities at resolution-changing steps.
FlexiNetLevel
    Wraps an EGNNStack for one resolution level with a specific hidden
    dimension and block count.
MolecularFlexiNet
    Main denoising network that routes inputs to the appropriate resolution
    level and produces noise and atom-type predictions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter

from molssd.models.egnn import EGNNStack
from molssd.models.embeddings import AtomTypeEmbedding, TimestepResolutionEmbedding


# ---------------------------------------------------------------------------
# Default level configurations
# ---------------------------------------------------------------------------

DEFAULT_LEVELS: List[Dict[str, int]] = [
    {"num_blocks": 4, "hidden_dim": 128},   # Level 0 (finest, full atomic)
    {"num_blocks": 3, "hidden_dim": 256},   # Level 1
    {"num_blocks": 3, "hidden_dim": 384},   # Level 2
    {"num_blocks": 2, "hidden_dim": 512},   # Level 3 (coarsest, centroid)
]


# ---------------------------------------------------------------------------
# PoolingLayer
# ---------------------------------------------------------------------------

class PoolingLayer(nn.Module):
    """Pool features and positions from a fine level to a coarser level.

    Positions are pooled via center-of-mass: ``X_coarse = C @ X_fine``, which
    is E(n)-equivariant when C is the mass-weighted coarsening matrix.

    Features are aggregated using scatter-based mean pooling guided by the
    cluster assignment vector, then projected to the coarse level's hidden
    dimension via a 1x1 linear layer.

    Parameters
    ----------
    d_fine : int
        Feature dimension at the fine level.
    d_coarse : int
        Feature dimension at the coarse level.
    """

    def __init__(self, d_fine: int, d_coarse: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_fine, d_coarse)

    def forward(
        self,
        h_fine: Tensor,
        x_fine: Tensor,
        coarsening_matrix: Tensor,
        cluster_assignment: Tensor,
        num_coarse_nodes: int,
    ) -> Tuple[Tensor, Tensor]:
        """Pool from fine to coarse resolution.

        Parameters
        ----------
        h_fine : Tensor, shape ``(N_fine, d_fine)``
            Node features at the fine level.
        x_fine : Tensor, shape ``(N_fine, 3)``
            Node positions at the fine level.
        coarsening_matrix : Tensor, shape ``(N_coarse, N_fine)``
            Mass-weighted coarsening matrix C. Applied to positions to yield
            center-of-mass coordinates.
        cluster_assignment : Tensor, shape ``(N_fine,)``
            Integer cluster labels mapping each fine node to its coarse
            super-node index in ``[0, N_coarse)``.
        num_coarse_nodes : int
            Number of nodes at the coarse level (N_coarse).

        Returns
        -------
        h_coarse : Tensor, shape ``(N_coarse, d_coarse)``
            Pooled and projected node features at the coarse level.
        x_coarse : Tensor, shape ``(N_coarse, 3)``
            Center-of-mass positions at the coarse level.
        """
        # Positions: center-of-mass via coarsening matrix (equivariant).
        x_coarse = coarsening_matrix @ x_fine  # (N_coarse, 3)

        # Features: scatter mean over cluster assignments, then project.
        h_agg = scatter(
            h_fine,
            cluster_assignment,
            dim=0,
            dim_size=num_coarse_nodes,
            reduce="mean",
        )  # (N_coarse, d_fine)
        h_coarse = self.proj(h_agg)  # (N_coarse, d_coarse)

        return h_coarse, x_coarse


# ---------------------------------------------------------------------------
# UnpoolingLayer
# ---------------------------------------------------------------------------

class UnpoolingLayer(nn.Module):
    """Unpool features and positions from a coarse level to a finer level.

    Positions are lifted using the pseudoinverse of the coarsening matrix:
    ``X_fine = C^+ @ X_coarse``, which assigns each atom its parent
    super-node's position (the cluster center-of-mass). This preserves
    E(n)-equivariance.

    Features are broadcast from each coarse super-node to all of its
    constituent fine-level atoms via the cluster assignment, then projected
    to the fine level's hidden dimension.

    Parameters
    ----------
    d_coarse : int
        Feature dimension at the coarse level.
    d_fine : int
        Feature dimension at the fine level.
    """

    def __init__(self, d_coarse: int, d_fine: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_coarse, d_fine)

    def forward(
        self,
        h_coarse: Tensor,
        x_coarse: Tensor,
        coarsening_matrix: Tensor,
        cluster_assignment: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Unpool from coarse to fine resolution.

        Parameters
        ----------
        h_coarse : Tensor, shape ``(N_coarse, d_coarse)``
            Node features at the coarse level.
        x_coarse : Tensor, shape ``(N_coarse, 3)``
            Node positions at the coarse level.
        coarsening_matrix : Tensor, shape ``(N_coarse, N_fine)``
            Coarsening matrix C. Its pseudoinverse is used to lift positions.
        cluster_assignment : Tensor, shape ``(N_fine,)``
            Integer cluster labels mapping each fine node to its parent
            coarse super-node.

        Returns
        -------
        h_fine : Tensor, shape ``(N_fine, d_fine)``
            Broadcast and projected features at the fine level.
        x_fine : Tensor, shape ``(N_fine, 3)``
            Lifted positions at the fine level (each atom gets its parent
            super-node's position).
        """
        # Positions: pseudoinverse lifting (equivariant).
        C_pinv = torch.linalg.pinv(coarsening_matrix)  # (N_fine, N_coarse)
        x_fine = C_pinv @ x_coarse  # (N_fine, 3)

        # Features: broadcast coarse features to constituent atoms, then project.
        h_broadcast = h_coarse[cluster_assignment]  # (N_fine, d_coarse)
        h_fine = self.proj(h_broadcast)  # (N_fine, d_fine)

        return h_fine, x_fine


# ---------------------------------------------------------------------------
# ZeroSkipConnection
# ---------------------------------------------------------------------------

class ZeroSkipConnection(nn.Module):
    """Gated skip connection initialized to zero.

    Implements::

        h_out = gate * W_skip(h_encoder) + (1 - gate) * h_decoder

    where ``gate`` is a learnable scalar parameter initialized to 0. At the
    start of training, the skip connection contributes nothing, which prevents
    discontinuities at resolution-changing steps. As training progresses, the
    gate learns to incorporate encoder information where beneficial.

    Parameters
    ----------
    dim : int
        Feature dimension of both the encoder and decoder representations.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h_encoder: Tensor, h_decoder: Tensor) -> Tensor:
        """Apply the gated skip connection.

        Parameters
        ----------
        h_encoder : Tensor, shape ``(N, dim)``
            Features from the encoder path.
        h_decoder : Tensor, shape ``(N, dim)``
            Features from the decoder path.

        Returns
        -------
        Tensor, shape ``(N, dim)``
            Blended features with learned gating.
        """
        gate = torch.sigmoid(self.gate)
        return gate * self.linear(h_encoder) + (1.0 - gate) * h_decoder


# ---------------------------------------------------------------------------
# FlexiNetLevel
# ---------------------------------------------------------------------------

class FlexiNetLevel(nn.Module):
    """One resolution level of the Molecular Flexi-Net.

    Wraps an :class:`~molssd.models.egnn.EGNNStack` with a specific hidden
    dimension and number of blocks for processing at a particular coarsening
    resolution.

    Parameters
    ----------
    hidden_dim : int
        Feature dimension for all EGNN blocks at this level.
    num_blocks : int
        Number of sequential EGNN blocks in the stack.
    edge_dim : int
        Dimension of edge attributes passed to each EGNN block.
    time_dim : int
        Dimension of the time/resolution conditioning embedding.
    act_fn : str
        Activation function name (e.g. ``'silu'``).
    attention : bool
        Whether to use attention gating inside each EGNN block.
    normalize : bool
        Whether to normalize coordinate updates by neighbour count.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        edge_dim: int = 0,
        time_dim: int = 128,
        act_fn: str = "silu",
        attention: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.egnn_stack = EGNNStack(
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            edge_dim=edge_dim,
            time_dim=time_dim,
            act_fn=act_fn,
            attention=attention,
            normalize=normalize,
        )

    def forward(
        self,
        h: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        t_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run this level's EGNN stack.

        Parameters
        ----------
        h : Tensor, shape ``(N, hidden_dim)``
            Node features.
        x : Tensor, shape ``(N, 3)``
            Node positions.
        edge_index : Tensor, shape ``(2, E)``
            Edge indices in COO format.
        edge_attr : Tensor or None, shape ``(E, edge_dim)``
            Optional edge attributes.
        t_emb : Tensor or None, shape ``(N, time_dim)``
            Per-node time/resolution conditioning embedding.

        Returns
        -------
        h_out : Tensor, shape ``(N, hidden_dim)``
            Updated node features.
        x_out : Tensor, shape ``(N, 3)``
            Updated node positions.
        """
        return self.egnn_stack(h, x, edge_index, edge_attr=edge_attr, t_emb=t_emb)


# ---------------------------------------------------------------------------
# MolecularFlexiNet
# ---------------------------------------------------------------------------

class MolecularFlexiNet(nn.Module):
    """Molecular Flexi-Net: the denoising backbone of MolSSD.

    A multi-resolution E(n)-equivariant denoising network that dynamically
    activates different EGNN stacks depending on the current diffusion
    resolution level. Each resolution level has its own dedicated EGNN stack
    with level-specific capacity (hidden dimension and depth).

    At timestep *t* with resolution level *k*:
      1. Atom types are embedded and projected to level *k*'s hidden dimension.
      2. A combined timestep + resolution conditioning vector is computed.
      3. Level *k*'s EGNN stack processes the (possibly coarsened) graph.
      4. Output heads predict position noise and atom type logits.

    This "dynamic activation" design achieves computational savings at coarser
    resolutions (fewer nodes) while using larger hidden dimensions for richer
    feature representations.

    Parameters
    ----------
    level_configs : list of dict, optional
        Per-level configuration. Each dict must contain ``'num_blocks'`` (int)
        and ``'hidden_dim'`` (int). Level 0 is finest (full atomic), level L
        is coarsest. If ``None``, :data:`DEFAULT_LEVELS` is used.
    num_atom_types : int
        Size of the atom-type vocabulary (including any coarsened/mixed type).
        Default is 6 for QM9 (H, C, N, O, F, coarsened).
    time_dim : int
        Internal dimension for the sinusoidal time embedding and the
        resolution embedding. Default 128.
    edge_dim : int
        Dimension of edge attribute vectors. Default 4.
    act_fn : str
        Activation function name for all EGNN blocks. Default ``'silu'``.
    attention : bool
        Whether to use attention gating inside EGNN blocks. Default ``True``.
    normalize : bool
        Whether to normalize coordinate updates. Default ``True``.

    Attributes
    ----------
    num_levels : int
        Total number of resolution levels in the architecture.
    levels : nn.ModuleList
        List of :class:`FlexiNetLevel` modules, one per resolution level.
    atom_embed : AtomTypeEmbedding
        Shared atom-type embedding lookup.
    time_res_embed : TimestepResolutionEmbedding
        Combined timestep + resolution conditioning module.
    input_proj : nn.ModuleList
        Per-level linear projections from atom embedding dim to level hidden dim.
    output_pos_head : nn.ModuleList
        Per-level linear layers mapping hidden features to 3D noise predictions.
    output_type_head : nn.ModuleList
        Per-level linear layers mapping hidden features to atom type logits.
    pool_layers : nn.ModuleList
        Pooling layers between consecutive levels (for future U-Net extension).
    unpool_layers : nn.ModuleList
        Unpooling layers between consecutive levels (for future U-Net extension).
    skip_connections : nn.ModuleList
        Zero-initialized skip connections for each level (for future U-Net extension).
    """

    def __init__(
        self,
        level_configs: Optional[List[Dict[str, int]]] = None,
        num_atom_types: int = 6,
        time_dim: int = 128,
        edge_dim: int = 0,
        act_fn: str = "silu",
        attention: bool = True,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        if level_configs is None:
            level_configs = DEFAULT_LEVELS

        self.level_configs = level_configs
        self.num_levels = len(level_configs)
        self.time_dim = time_dim
        self.edge_dim = edge_dim
        self.num_atom_types = num_atom_types

        # --- Shared embeddings ---
        # The atom embedding dimension matches level 0's hidden dim so that
        # at the finest resolution no projection is needed (identity proj).
        atom_embed_dim = level_configs[0]["hidden_dim"]
        self.atom_embed = AtomTypeEmbedding(
            num_types=num_atom_types, dim=atom_embed_dim
        )

        # Time + resolution conditioning. The output dimension of the
        # conditioning MLP is set to the *maximum* hidden dim across levels;
        # per-level time projection layers then map it to each level's dim.
        self.time_res_embed = TimestepResolutionEmbedding(
            time_dim=time_dim,
            res_levels=self.num_levels,
            out_dim=time_dim,
        )

        # --- Per-level modules ---
        self.levels = nn.ModuleList()
        self.input_proj = nn.ModuleList()
        self.time_proj = nn.ModuleList()
        self.output_pos_head = nn.ModuleList()
        self.output_type_head = nn.ModuleList()

        for cfg in level_configs:
            hidden_dim = cfg["hidden_dim"]
            num_blocks = cfg["num_blocks"]

            # EGNN stack for this level.
            self.levels.append(
                FlexiNetLevel(
                    hidden_dim=hidden_dim,
                    num_blocks=num_blocks,
                    edge_dim=edge_dim,
                    time_dim=time_dim,
                    act_fn=act_fn,
                    attention=attention,
                    normalize=normalize,
                )
            )

            # Project atom embeddings (atom_embed_dim) -> level hidden_dim.
            self.input_proj.append(nn.Linear(atom_embed_dim, hidden_dim))

            # Project time/resolution conditioning (time_dim) -> level time_dim.
            # The EGNN blocks expect time_dim input, and we keep time_dim
            # consistent across levels so this is an identity-sized projection
            # that allows per-level adaptation of the conditioning signal.
            self.time_proj.append(nn.Linear(time_dim, time_dim))

            # Output heads for this level.
            self.output_pos_head.append(nn.Linear(hidden_dim, 3))
            self.output_type_head.append(nn.Linear(hidden_dim, num_atom_types))

        # --- Pooling / unpooling / skip layers (for U-Net extension) ---
        self.pool_layers = nn.ModuleList()
        self.unpool_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()

        for i in range(self.num_levels - 1):
            d_fine = level_configs[i]["hidden_dim"]
            d_coarse = level_configs[i + 1]["hidden_dim"]
            self.pool_layers.append(PoolingLayer(d_fine, d_coarse))
            self.unpool_layers.append(UnpoolingLayer(d_coarse, d_fine))
            self.skip_connections.append(ZeroSkipConnection(d_fine))

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x_t: Tensor,
        atom_types: Tensor,
        t: Tensor,
        resolution_level: int,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass: predict noise and atom type logits at the given resolution.

        At resolution level *k*, only the level-*k* EGNN stack is executed.
        The input graph should already be coarsened to level *k* (i.e.,
        ``x_t`` has ``N_k`` nodes and ``edge_index`` connects super-nodes at
        level *k*).

        Parameters
        ----------
        x_t : Tensor, shape ``(N_k, 3)``
            Noisy node positions at the current resolution level.
        atom_types : Tensor, shape ``(N_k,)``
            Integer atom-type indices (or coarsened/mixed type for super-nodes).
        t : Tensor, shape ``(B,)`` or ``(B, 1)``
            Continuous diffusion timesteps for each graph in the batch.
        resolution_level : int
            Current coarsening resolution level (0 = finest, L = coarsest).
        edge_index : Tensor, shape ``(2, E)``
            Edge connectivity at the current resolution level in COO format.
        edge_attr : Tensor or None, shape ``(E, edge_dim)``
            Optional edge attributes at the current resolution level.
        batch : Tensor or None, shape ``(N_k,)``
            Batch assignment vector for PyTorch Geometric batched graphs.
            Maps each node to its graph index in ``[0, B)``. Required when
            ``B > 1`` to correctly broadcast per-graph embeddings to per-node.

        Returns
        -------
        eps_hat : Tensor, shape ``(N_k, 3)``
            Predicted position noise (score) at the current resolution.
        type_logits : Tensor, shape ``(N_k, num_atom_types)``
            Predicted atom type logits at the current resolution.

        Raises
        ------
        ValueError
            If ``resolution_level`` is out of range ``[0, num_levels)``.
        """
        if resolution_level < 0 or resolution_level >= self.num_levels:
            raise ValueError(
                f"resolution_level must be in [0, {self.num_levels}), "
                f"got {resolution_level}"
            )

        k = resolution_level
        num_nodes = x_t.size(0)

        # 1. Embed atom types and project to level k's hidden dimension.
        h = self.atom_embed(atom_types)         # (N_k, atom_embed_dim)
        h = self.input_proj[k](h)               # (N_k, hidden_dim_k)

        # 2. Compute timestep + resolution conditioning.
        # Build a resolution level tensor matching the batch dimension.
        if batch is not None:
            batch_size = int(batch.max().item()) + 1
        else:
            batch_size = 1

        res_level_tensor = torch.full(
            (batch_size,), resolution_level,
            dtype=torch.long, device=x_t.device,
        )
        t_emb_graph = self.time_res_embed(t, res_level_tensor)  # (B, time_dim)

        # Apply per-level time projection.
        t_emb_graph = self.time_proj[k](t_emb_graph)            # (B, time_dim)

        # Broadcast per-graph conditioning to per-node.
        if batch is not None:
            t_emb = t_emb_graph[batch]          # (N_k, time_dim)
        else:
            t_emb = t_emb_graph.expand(num_nodes, -1)  # (N_k, time_dim)

        # 3. Run level k's EGNN stack.
        h_out, x_out = self.levels[k](
            h, x_t, edge_index, edge_attr=edge_attr, t_emb=t_emb
        )

        # 4. Output heads.
        eps_hat = self.output_pos_head[k](h_out)        # (N_k, 3)
        type_logits = self.output_type_head[k](h_out)   # (N_k, num_atom_types)

        return eps_hat, type_logits
