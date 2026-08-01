"""Multi-Horizon Pinball Loss and Weighted Hazard Cross-Entropy with robust numerical guards."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from climate_forecast.config import DEFAULT_QUANTILES, TrainingConfig


class MultiHorizonPinballLoss(nn.Module):
    """Computes Pinball (Quantile) Loss across multiple forecast horizons and targets."""

    def __init__(self, quantiles: List[float] = DEFAULT_QUANTILES):
        super().__init__()
        self.quantiles = quantiles
        self.register_buffer("q_tensor", torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, q_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_pred: [B, H=14, num_targets=3, num_quantiles=3]
            y_true: [B, H=14, num_targets=3]
        Returns:
            loss: Scalar tensor
        """
        y_expanded = y_true.unsqueeze(-1).expand_as(q_pred)
        error = torch.nan_to_num(y_expanded - q_pred, nan=0.0, posinf=100.0, neginf=-100.0)

        q = self.q_tensor.view(1, 1, 1, -1)
        pinball = torch.maximum(q * error, (q - 1.0) * error)

        weights = torch.tensor([1.0, 1.0, 0.5], device=q_pred.device).view(1, 1, 3, 1)
        weighted_pinball = pinball * weights

        return torch.mean(weighted_pinball)


class ExtremeHazardLoss(nn.Module):
    """Weighted Binary Cross-Entropy for imbalanced extreme hazard classification."""

    def __init__(self, pos_weights: Optional[List[float]] = None):
        super().__init__()
        weights = pos_weights or [4.0, 3.5, 5.0]
        self.register_buffer("pos_weight", torch.tensor(weights, dtype=torch.float32))

    def forward(self, prob_pred: torch.Tensor, y_hazard: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        p = torch.clamp(prob_pred, min=eps, max=1.0 - eps)
        pos_w = self.pos_weight.view(1, 1, 3)

        y_safe = torch.nan_to_num(y_hazard, nan=0.0)
        bce = - (pos_w * y_safe * torch.log(p) + (1.0 - y_safe) * torch.log(1.0 - p))
        return torch.mean(bce)


class ClimateCompositeLoss(nn.Module):
    """Joint composite loss balancing quantile trajectory estimation and extreme hazard detection."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        super().__init__()
        self.config = config or TrainingConfig()
        self.pinball_loss = MultiHorizonPinballLoss()
        self.hazard_loss = ExtremeHazardLoss()

    def forward(
        self,
        q_pred: torch.Tensor,
        prob_pred: torch.Tensor,
        y_targets: torch.Tensor,
        y_hazards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        l_pinball = self.pinball_loss(q_pred, y_targets)
        l_hazard = self.hazard_loss(prob_pred, y_hazards)

        total_loss = (
            self.config.pinball_loss_weight * l_pinball
            + self.config.hazard_bce_loss_weight * l_hazard
        )

        return {
            "total_loss": total_loss,
            "pinball_loss": l_pinball,
            "hazard_loss": l_hazard,
        }
