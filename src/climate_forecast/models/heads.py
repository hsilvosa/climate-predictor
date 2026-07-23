"""Multi-horizon probabilistic quantile and extreme hazard prediction heads."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from climate_forecast.config import DEFAULT_QUANTILES


class MultiHorizonQuantileHead(nn.Module):
    """Outputs monotonic probabilistic quantiles [P10, P50, P90] for TMAX, TMIN, and PRCP.

    Uses structural softplus delta parametrization to strictly guarantee P10 <= P50 <= P90.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        quantiles: List[float] = DEFAULT_QUANTILES,
        num_targets: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.quantiles = quantiles
        self.num_targets = num_targets  # 0: TMAX, 1: TMIN, 2: PRCP

        # For each target: predicts median (P50), lower delta (P50 - P10 >= 0), upper delta (P90 - P50 >= 0)
        self.head_tmax = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # [p50, delta_low, delta_high]
        )
        self.head_tmin = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.head_prcp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(self, h_rollout: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_rollout: [B, H=14, hidden_dim]
        Returns:
            quantiles: [B, H=14, num_targets=3, num_quantiles=3]
                       where quantiles[:, :, target, 0] = P10
                             quantiles[:, :, target, 1] = P50
                             quantiles[:, :, target, 2] = P90
        """
        # TMAX Head
        raw_tmax = self.head_tmax(h_rollout)  # [B, H, 3]
        p50_tmax = raw_tmax[..., 0]
        d_low_tmax = F.softplus(raw_tmax[..., 1])
        d_high_tmax = F.softplus(raw_tmax[..., 2])
        p10_tmax = p50_tmax - d_low_tmax
        p90_tmax = p50_tmax + d_high_tmax
        q_tmax = torch.stack([p10_tmax, p50_tmax, p90_tmax], dim=-1)  # [B, H, 3]

        # TMIN Head
        raw_tmin = self.head_tmin(h_rollout)
        p50_tmin = raw_tmin[..., 0]
        d_low_tmin = F.softplus(raw_tmin[..., 1])
        d_high_tmin = F.softplus(raw_tmin[..., 2])
        p10_tmin = p50_tmin - d_low_tmin
        p90_tmin = p50_tmin + d_high_tmin
        q_tmin = torch.stack([p10_tmin, p50_tmin, p90_tmin], dim=-1)  # [B, H, 3]

        # PRCP Head (enforce non-negative rain)
        raw_prcp = self.head_prcp(h_rollout)
        p50_prcp = F.softplus(raw_prcp[..., 0])
        d_low_prcp = F.softplus(raw_prcp[..., 1])
        d_high_prcp = F.softplus(raw_prcp[..., 2])
        p10_prcp = torch.clamp(p50_prcp - d_low_prcp, min=0.0)
        p90_prcp = p50_prcp + d_high_prcp
        q_prcp = torch.stack([p10_prcp, p50_prcp, p90_prcp], dim=-1)  # [B, H, 3]

        # Stack targets: [B, H, 3, 3]
        return torch.stack([q_tmax, q_tmin, q_prcp], dim=2)


class ExtremeHazardClassificationHead(nn.Module):
    """Outputs calibrated event probabilities for extreme heatwaves, frost freezes, and deluge rainfall."""

    def __init__(self, hidden_dim: int = 128, num_hazards: int = 3):
        super().__init__()
        self.num_hazards = num_hazards
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_hazards),
        )

    def forward(self, h_rollout: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_rollout: [B, H=14, hidden_dim]
        Returns:
            probs: [B, H=14, 3] (0: Heatwave, 1: Frost, 2: Deluge)
        """
        logits = self.head(h_rollout)
        return torch.sigmoid(logits)
