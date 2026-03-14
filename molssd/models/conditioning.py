"""Property and spectrum conditioning modules for MolSSD.

Implements the conditioning pipeline for conditional molecular generation:

    Pre-training (OMol25):
        scalar HOMO-LUMO gap → ScalarPropertyEncoder → c ∈ R^d → adaLN → EGNN

    Fine-tuning (UV absorbers):
        UV spectrum (600-dim) → SpectrumEncoder → c ∈ R^d → adaLN → EGNN

The key design: both encoders produce the SAME dimension conditioning vector,
injected at the SAME point (adaLN). The "swap" is just replacing which encoder
is active. The EGNN backbone doesn't change.

Classifier-free guidance:
    During training, the conditioning is randomly dropped (replaced with a
    learned null embedding) with probability p. During sampling:
        ε_guided = (1 + w) · ε_cond − w · ε_uncond

Modules
-------
ScalarPropertyEncoder
    Encodes one or more scalar properties (HOMO-LUMO gap, logP, etc.)
    into a conditioning vector via sinusoidal embedding + MLP.
SpectrumEncoder
    Encodes a discretized UV-Vis spectrum (1D signal) into a conditioning
    vector via 1D CNN + global pooling.
MultiPropertyEncoder
    Combines scalar properties and optional spectrum into a single
    conditioning vector.
PropertyConditioner
    Top-level module that wraps encoding + classifier-free guidance dropout.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Scalar property encoder
# ---------------------------------------------------------------------------

class ScalarPropertyEncoder(nn.Module):
    """Encode scalar molecular properties into a conditioning vector.

    Each scalar property gets a sinusoidal embedding (like timestep encoding),
    then all are concatenated and projected through an MLP.

    Supported properties for UV-absorber work:
        - homo_lumo_gap (eV): primary proxy for λ_max
        - lambda_max (nm): actual absorption wavelength (from TD-DFT)
        - epsilon_max: molar extinction coefficient
        - logP: lipophilicity
        - sa_score: synthetic accessibility
        - mw: molecular weight
        - photostability: 0-1 score

    Parameters
    ----------
    property_names : list of str
        Names of scalar properties to encode. Order matters — must match
        the order in the input dict.
    embed_dim : int
        Dimension of each property's sinusoidal embedding.
    out_dim : int
        Final conditioning vector dimension.
    """

    def __init__(
        self,
        property_names: List[str],
        embed_dim: int = 64,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        self.property_names = property_names
        self.num_properties = len(property_names)
        self.embed_dim = embed_dim

        # Sinusoidal embedding frequencies (shared across properties)
        half = embed_dim // 2
        freq_exp = torch.arange(half, dtype=torch.float32) * (
            -math.log(10_000.0) / half
        )
        self.register_buffer("freq_exp", freq_exp)

        # Per-property learned scaling (normalizes different property ranges)
        self.prop_scale = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1))
            for name in property_names
        })
        self.prop_bias = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in property_names
        })

        # MLP: concat of all property embeddings → out_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.num_properties * embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _sinusoidal_embed(self, x: Tensor) -> Tensor:
        """Sinusoidal embedding for a scalar tensor (B,) → (B, embed_dim)."""
        x = x.view(-1, 1).float()
        freqs = torch.exp(self.freq_exp)
        args = x * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, properties: Dict[str, Tensor]) -> Tensor:
        """Encode scalar properties into a conditioning vector.

        Parameters
        ----------
        properties : dict of str → Tensor
            Each value has shape ``(B,)`` or ``(B, 1)``.

        Returns
        -------
        Tensor, shape ``(B, out_dim)``
        """
        embeddings = []
        for name in self.property_names:
            val = properties[name].view(-1).float()
            # Learned normalization
            val = val * self.prop_scale[name] + self.prop_bias[name]
            emb = self._sinusoidal_embed(val)
            embeddings.append(emb)

        combined = torch.cat(embeddings, dim=-1)  # (B, num_props * embed_dim)
        return self.mlp(combined)  # (B, out_dim)


# ---------------------------------------------------------------------------
# Spectrum encoder (1D CNN)
# ---------------------------------------------------------------------------

class SpectrumEncoder(nn.Module):
    """Encode a discretized UV-Vis absorption spectrum into a conditioning vector.

    Architecture: 1D CNN with global average pooling, producing a fixed-size
    vector regardless of spectrum length.

    Input: discretized spectrum of shape (B, num_bins) where num_bins
    corresponds to wavelength bins (e.g., 200-800nm at 1nm → 600 bins).

    Parameters
    ----------
    num_bins : int
        Number of wavelength bins in the input spectrum.
    out_dim : int
        Final conditioning vector dimension.
    channels : list of int
        Hidden channel sizes for the 1D CNN layers.
    """

    def __init__(
        self,
        num_bins: int = 600,
        out_dim: int = 256,
        channels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 256]

        layers = []
        in_ch = 1  # single-channel input (absorption intensity)
        for ch in channels:
            layers.extend([
                nn.Conv1d(in_ch, ch, kernel_size=7, padding=3),
                nn.BatchNorm1d(ch),
                nn.SiLU(),
                nn.MaxPool1d(2),
            ])
            in_ch = ch

        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Linear(channels[-1], out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, spectrum: Tensor) -> Tensor:
        """Encode a spectrum into a conditioning vector.

        Parameters
        ----------
        spectrum : Tensor, shape ``(B, num_bins)``
            Discretized absorption spectrum (intensity values).

        Returns
        -------
        Tensor, shape ``(B, out_dim)``
        """
        x = spectrum.unsqueeze(1)  # (B, 1, num_bins)
        x = self.cnn(x)           # (B, C, L')
        x = x.mean(dim=-1)        # (B, C) — global average pooling
        return self.proj(x)        # (B, out_dim)


# ---------------------------------------------------------------------------
# Multi-property encoder (scalars + optional spectrum)
# ---------------------------------------------------------------------------

class MultiPropertyEncoder(nn.Module):
    """Combines scalar properties and optional spectrum into one conditioning vector.

    This is the unified encoder that handles both pre-training (scalars only)
    and fine-tuning (scalars + spectrum) seamlessly.

    Parameters
    ----------
    scalar_properties : list of str
        Names of scalar properties to encode.
    use_spectrum : bool
        Whether to include the spectrum encoder.
    embed_dim : int
        Per-property sinusoidal embedding dimension.
    out_dim : int
        Final conditioning vector dimension.
    spectrum_bins : int
        Number of wavelength bins (if using spectrum).
    """

    def __init__(
        self,
        scalar_properties: List[str],
        use_spectrum: bool = False,
        embed_dim: int = 64,
        out_dim: int = 256,
        spectrum_bins: int = 600,
    ) -> None:
        super().__init__()
        self.use_spectrum = use_spectrum
        self.out_dim = out_dim

        self.scalar_encoder = ScalarPropertyEncoder(
            property_names=scalar_properties,
            embed_dim=embed_dim,
            out_dim=out_dim,
        )

        if use_spectrum:
            self.spectrum_encoder = SpectrumEncoder(
                num_bins=spectrum_bins,
                out_dim=out_dim,
            )
            # Fusion: concatenate scalar + spectrum embeddings → project
            self.fusion = nn.Sequential(
                nn.Linear(2 * out_dim, out_dim),
                nn.SiLU(),
                nn.Linear(out_dim, out_dim),
            )

    def forward(
        self,
        properties: Dict[str, Tensor],
        spectrum: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode properties (+ optional spectrum) into conditioning vector.

        Parameters
        ----------
        properties : dict of str → Tensor(B,)
            Scalar property values.
        spectrum : Tensor or None, shape ``(B, num_bins)``
            Optional absorption spectrum.

        Returns
        -------
        Tensor, shape ``(B, out_dim)``
        """
        c_scalar = self.scalar_encoder(properties)  # (B, out_dim)

        if self.use_spectrum and spectrum is not None:
            c_spectrum = self.spectrum_encoder(spectrum)  # (B, out_dim)
            c = self.fusion(torch.cat([c_scalar, c_spectrum], dim=-1))
        else:
            c = c_scalar

        return c


# ---------------------------------------------------------------------------
# Property conditioner with classifier-free guidance
# ---------------------------------------------------------------------------

class PropertyConditioner(nn.Module):
    """Top-level conditioning module with classifier-free guidance support.

    Wraps a MultiPropertyEncoder and handles:
    1. Encoding properties/spectrum into a conditioning vector
    2. Random dropout of conditioning during training (for CFG)
    3. Providing null (unconditional) embeddings during guided sampling

    During training with dropout_prob > 0:
        - With probability p, replace c with a learned null embedding
        - The model learns both conditional and unconditional generation

    During sampling with guidance:
        ε_guided = (1 + w) · ε(x_t, t, c) − w · ε(x_t, t, ∅)

    Parameters
    ----------
    scalar_properties : list of str
        Names of scalar properties.
    use_spectrum : bool
        Whether to include spectrum conditioning.
    out_dim : int
        Conditioning vector dimension (must match model's time_dim or
        conditioning injection dimension).
    dropout_prob : float
        Probability of dropping conditioning during training (CFG).
    """

    def __init__(
        self,
        scalar_properties: List[str],
        use_spectrum: bool = False,
        out_dim: int = 256,
        dropout_prob: float = 0.15,
        spectrum_bins: int = 600,
    ) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob
        self.out_dim = out_dim

        self.encoder = MultiPropertyEncoder(
            scalar_properties=scalar_properties,
            use_spectrum=use_spectrum,
            out_dim=out_dim,
            spectrum_bins=spectrum_bins,
        )

        # Learned null embedding for unconditional generation (CFG)
        self.null_embedding = nn.Parameter(torch.randn(out_dim) * 0.02)

    def forward(
        self,
        properties: Dict[str, Tensor],
        spectrum: Optional[Tensor] = None,
        force_unconditional: bool = False,
    ) -> Tensor:
        """Encode conditioning with optional CFG dropout.

        Parameters
        ----------
        properties : dict of str → Tensor(B,)
            Scalar property values.
        spectrum : Tensor or None
            Optional absorption spectrum.
        force_unconditional : bool
            If True, always return null embedding (used during guided sampling
            for the unconditional branch).

        Returns
        -------
        Tensor, shape ``(B, out_dim)``
        """
        if force_unconditional:
            B = next(iter(properties.values())).shape[0]
            device = next(iter(properties.values())).device
            return self.null_embedding.unsqueeze(0).expand(B, -1).to(device)

        c = self.encoder(properties, spectrum)  # (B, out_dim)

        # Classifier-free guidance: random dropout during training
        if self.training and self.dropout_prob > 0:
            B = c.shape[0]
            mask = torch.rand(B, 1, device=c.device) < self.dropout_prob
            null = self.null_embedding.unsqueeze(0).expand(B, -1)
            c = torch.where(mask, null, c)

        return c

    @torch.no_grad()
    def guided_combine(
        self,
        eps_cond: Tensor,
        eps_uncond: Tensor,
        guidance_weight: float = 2.5,
    ) -> Tensor:
        """Apply classifier-free guidance to noise predictions.

        ε_guided = (1 + w) · ε_cond − w · ε_uncond

        Parameters
        ----------
        eps_cond : Tensor
            Noise prediction with conditioning.
        eps_uncond : Tensor
            Noise prediction without conditioning (null embedding).
        guidance_weight : float
            Guidance strength w. Higher = stronger conditioning adherence
            but lower diversity. Typical range: 1.0-5.0.

        Returns
        -------
        Tensor
            Guided noise prediction.
        """
        return (1.0 + guidance_weight) * eps_cond - guidance_weight * eps_uncond
