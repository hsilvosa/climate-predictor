"""Tests for neural network components, forward pass, and composite loss."""

import pytest
import torch

from climate_forecast.config import ModelConfig
from climate_forecast.models.loss import ClimateCompositeLoss, MultiHorizonPinballLoss
from climate_forecast.models.spatiotemporal_net import (
    ClimateSpatiotemporalNet,
    SphericalHarmonicEmbedding,
)


def test_spherical_harmonic_embedding():
    embedder = SphericalHarmonicEmbedding(num_freqs=8, embed_dim=64)
    x_spatial = torch.tensor([[0.5, -0.2, 0.1], [-0.8, 0.9, -0.5]])  # [B=2, 3]
    out = embedder(x_spatial)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()


def test_spatiotemporal_net_forward():
    cfg = ModelConfig(hidden_dim=64, tcn_num_layers=2, num_attention_heads=2)
    model = ClimateSpatiotemporalNet(cfg)

    B = 4
    x_seq = torch.randn(B, 30, 7)
    x_spatial = torch.randn(B, 3)
    x_future_time = torch.randn(B, 14, 2)

    outputs = model(x_seq, x_spatial, x_future_time)

    assert "quantiles" in outputs
    assert "hazard_probs" in outputs
    assert outputs["quantiles"].shape == (B, 14, 3, 3)
    assert outputs["hazard_probs"].shape == (B, 14, 3)

    # Check structural quantile monotonicity: P10 <= P50 <= P90
    q = outputs["quantiles"]
    p10 = q[..., 0]
    p50 = q[..., 1]
    p90 = q[..., 2]

    assert (p50 >= p10 - 1e-5).all(), "P50 must be >= P10"
    assert (p90 >= p50 - 1e-5).all(), "P90 must be >= P50"


def test_loss_backward():
    cfg = ModelConfig(hidden_dim=32, tcn_num_layers=2, num_attention_heads=2)
    model = ClimateSpatiotemporalNet(cfg)
    loss_fn = ClimateCompositeLoss()

    x_seq = torch.randn(2, 30, 7)
    x_spatial = torch.randn(2, 3)
    x_future = torch.randn(2, 14, 2)

    y_targets = torch.randn(2, 14, 3)
    y_hazards = torch.randint(0, 2, (2, 14, 3)).float()

    outputs = model(x_seq, x_spatial, x_future)
    loss_dict = loss_fn(outputs["quantiles"], outputs["hazard_probs"], y_targets, y_hazards)

    assert "total_loss" in loss_dict
    loss = loss_dict["total_loss"]
    assert not torch.isnan(loss)

    loss.backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
