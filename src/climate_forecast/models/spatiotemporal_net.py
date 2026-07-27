"""Spherical Harmonic Spatial Embedding and SpatioTemporal Forecasting Network."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from climate_forecast.config import (
    DEFAULT_QUANTILES,
    FORECAST_HORIZON_DAYS,
    NUM_SPHERICAL_HARMONIC_FREQS,
    ModelConfig,
)
from climate_forecast.models.heads import (
    ExtremeHazardClassificationHead,
    MultiHorizonQuantileHead,
)


class SphericalHarmonicEmbedding(nn.Module):
    """Encodes continuous latitude, longitude, and elevation on the spherical manifold using multi-scale Fourier harmonics."""

    def __init__(self, num_freqs: int = NUM_SPHERICAL_HARMONIC_FREQS, embed_dim: int = 64):
        super().__init__()
        self.num_freqs = num_freqs
        # Frequencies: 2^0, 2^1, ... 2^(L-1)
        freq_bands = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        self.register_buffer("freq_bands", freq_bands)

        # Raw coordinate input dimension:
        # For lat: num_freqs * 2 (sin, cos)
        # For lon: num_freqs * 2 (sin, cos)
        # For elevation: 1
        raw_dim = (num_freqs * 4) + 1
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x_spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spatial: [B, 3] where [:, 0] is lat in [-1, 1], [:, 1] is lon in [-1, 1], [:, 2] is norm elevation.
        Returns:
            embed: [B, embed_dim]
        """
        lat = x_spatial[:, 0:1] * math.pi      # [-pi, pi]
        lon = x_spatial[:, 1:2] * math.pi      # [-pi, pi]
        elev = x_spatial[:, 2:3]

        # Multi-scale sinusoidal features
        lat_freqs = lat * self.freq_bands.unsqueeze(0)   # [B, num_freqs]
        lon_freqs = lon * self.freq_bands.unsqueeze(0)   # [B, num_freqs]

        sin_lat = torch.sin(lat_freqs)
        cos_lat = torch.cos(lat_freqs)
        sin_lon = torch.sin(lon_freqs)
        cos_lon = torch.cos(lon_freqs)

        fourier_feats = torch.cat([sin_lat, cos_lat, sin_lon, cos_lon, elev], dim=-1)
        return self.proj(fourier_feats)


class TemporalTCNBlock(nn.Module):
    """Dilated Causal Temporal Convolutional Block with Residual and LayerNorm."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, padding=self.padding
        )
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C]
        """
        B, T, C = x.shape
        res = self.residual(x)

        # Transpose to [B, C, T] for 1D Conv
        out = x.transpose(1, 2)
        out = self.conv1(out)[:, :, :T]  # Trim padding to maintain exact T
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)

        out = out.transpose(1, 2)
        out = self.conv2(out)[:, :, :T]
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = F.gelu(out)
        out = self.dropout(out)

        return out + res


class ClimateSpatiotemporalNet(nn.Module):
    """Deep SpatioTemporal Neural Forecaster with Spherical Pos-Encoding, TCN, Attention, and Multi-Horizon Quantile Heads."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()

        # Spatial Spherical Harmonics Embedding
        self.spatial_encoder = SphericalHarmonicEmbedding(
            num_freqs=NUM_SPHERICAL_HARMONIC_FREQS,
            embed_dim=self.config.spatial_embed_dim,
        )

        # Input feature projection
        # x_seq: [B, T=30, 7] + broadcasted spatial_embed [B, T, spatial_embed_dim]
        combined_in_dim = self.config.input_features_per_step + self.config.spatial_embed_dim
        self.in_proj = nn.Sequential(
            nn.Linear(combined_in_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
        )

        # Dilated Residual TCN sequence backbone
        tcn_layers = []
        dilations = [1, 2, 4, 8][: self.config.tcn_num_layers]
        for dil in dilations:
            tcn_layers.append(
                TemporalTCNBlock(
                    in_channels=self.config.hidden_dim,
                    out_channels=self.config.hidden_dim,
                    kernel_size=self.config.tcn_kernel_size,
                    dilation=dil,
                    dropout=self.config.dropout,
                )
            )
        self.tcn_backbone = nn.Sequential(*tcn_layers)

        # Multi-Head Temporal Self-Attention over context window
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_dim,
            num_heads=self.config.num_attention_heads,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.config.hidden_dim)

        # Future temporal conditioner: maps [sin_doy, cos_doy] for 14 horizon steps + context summary
        self.future_temporal_proj = nn.Sequential(
            nn.Linear(2, self.config.hidden_dim // 2),
            nn.GELU(),
        )

        # Multi-horizon decoder MLP
        decoder_in_dim = self.config.hidden_dim + (self.config.hidden_dim // 2)
        self.decoder_backbone = nn.Sequential(
            nn.Linear(decoder_in_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.GELU(),
        )

        # Quantile & Extreme Classification Heads
        self.quantile_head = MultiHorizonQuantileHead(
            hidden_dim=self.config.hidden_dim,
            quantiles=self.config.quantiles,
            num_targets=self.config.num_targets,
        )

        self.hazard_head = ExtremeHazardClassificationHead(
            hidden_dim=self.config.hidden_dim,
            num_hazards=self.config.num_extreme_heads,
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        x_spatial: torch.Tensor,
        x_future_time: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_seq: [B, T=30, C=7] historical continuous sequence
            x_spatial: [B, 3] (lat, lon, elevation)
            x_future_time: [B, H=14, 2] (sin_doy, cos_doy for future 14 days)
        Returns:
            Dict containing:
                "quantiles": [B, 14, 3, 3] -> (TMAX, TMIN, PRCP) x (P10, P50, P90)
                "hazard_probs": [B, 14, 3] -> (Heatwave, Frost, Deluge)
                "latent_repr": [B, 14, hidden_dim]
        """
        B, T, _ = x_seq.shape
        H = x_future_time.shape[1]

        # 1. Spatial Harmonic Embedding
        sp_embed = self.spatial_encoder(x_spatial)  # [B, spatial_embed_dim]

        # Broadcast spatial embedding across history steps
        sp_expanded = sp_embed.unsqueeze(1).expand(B, T, -1)  # [B, T, spatial_embed_dim]
        combined_seq = torch.cat([x_seq, sp_expanded], dim=-1)  # [B, T, 7 + spatial_embed_dim]

        # 2. Sequence projection & TCN extraction
        h = self.in_proj(combined_seq)  # [B, T, hidden_dim]
        h = self.tcn_backbone(h)        # [B, T, hidden_dim]

        # 3. Temporal Self-Attention
        attn_out, _ = self.temporal_attention(h, h, h)
        h_seq = self.attn_norm(h + attn_out)  # [B, T, hidden_dim]

        # Aggregate context summary (last step + mean pool)
        ctx_last = h_seq[:, -1, :]             # [B, hidden_dim]
        ctx_mean = torch.mean(h_seq, dim=1)    # [B, hidden_dim]
        ctx_summary = 0.6 * ctx_last + 0.4 * ctx_mean  # [B, hidden_dim]

        # 4. Multi-Horizon Rollout conditioning with future calendar harmonics
        ctx_expanded = ctx_summary.unsqueeze(1).expand(B, H, -1)  # [B, H, hidden_dim]
        fut_time_feat = self.future_temporal_proj(x_future_time)   # [B, H, hidden_dim // 2]
        decoder_in = torch.cat([ctx_expanded, fut_time_feat], dim=-1)  # [B, H, hidden_dim + hidden_dim//2]

        latent_rollout = self.decoder_backbone(decoder_in)  # [B, H, hidden_dim]

        # 5. Output Heads
        quantiles = self.quantile_head(latent_rollout)      # [B, H, 3, 3]
        hazard_probs = self.hazard_head(latent_rollout)     # [B, H, 3]

        return {
            "quantiles": quantiles,
            "hazard_probs": hazard_probs,
            "latent_repr": latent_rollout,
        }
