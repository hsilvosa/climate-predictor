"""Step 2: Train Deep SpatioTemporal Climate Forecaster with CUDA acceleration."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from climate_forecast.config import (
    CHECKPOINTS_DIR,
    PROCESSED_DATA_DIR,
    ModelConfig,
    TrainingConfig,
)
from climate_forecast.data.climatology import ClimatologyEngine
from climate_forecast.data.dataset import create_dataloaders
from climate_forecast.models.spatiotemporal_net import ClimateSpatiotemporalNet
from climate_forecast.training.trainer import ClimateForecasterTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Spatiotemporal Climate Forecaster")
    parser.add_argument("--parquet-path", default=str(PROCESSED_DATA_DIR / "observations.parquet"))
    parser.add_argument("--climatology-path", default=str(PROCESSED_DATA_DIR / "stations_climatology.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    logger.info(f"Loading observations from {args.parquet_path}...")
    df_obs = pd.read_parquet(args.parquet_path)
    logger.info(f"Loaded {len(df_obs)} observation rows across {df_obs['station_id'].nunique()} stations.")

    logger.info(f"Loading climatology from {args.climatology_path}...")
    climatology = ClimatologyEngine.load_json(args.climatology_path)

    training_cfg = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
    )

    logger.info("Constructing sliding window spatiotemporal datasets (train <= 2021, val 2022, test >= 2023)...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders(
        df=df_obs,
        climatology=climatology,
        training_cfg=training_cfg,
        stations_meta=climatology.stations_clim,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    # Save scaler
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    scaler.save_json(CHECKPOINTS_DIR / "scaler.json")

    logger.info("Initializing ClimateSpatiotemporalNet model...")
    model = ClimateSpatiotemporalNet(model_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model trainable parameters: {total_params:,}")

    trainer = ClimateForecasterTrainer(
        model=model,
        training_cfg=training_cfg,
        model_cfg=model_cfg,
        device=device,
    )

    logger.info("Starting training loop...")
    best_model, history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=CHECKPOINTS_DIR,
        scaler=scaler,
    )

    # Save training history
    with open(CHECKPOINTS_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Evaluating best model on held-out test set (2023-2026)...")
    test_losses, test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Total Loss: {test_losses['total_loss']:.4f}")
    logger.info(f"Test TMAX MAE (Overall): {test_metrics.tmax_metrics.mae_overall:.2f} C (Day 1: {test_metrics.tmax_metrics.mae_day1:.2f} C, Day 7: {test_metrics.tmax_metrics.mae_day7:.2f} C)")
    logger.info(f"Test TMIN MAE (Overall): {test_metrics.tmin_metrics.mae_overall:.2f} C")
    logger.info(f"Test PRCP MAE (Overall): {test_metrics.prcp_metrics.mae_overall:.2f} mm")
    logger.info(f"Test Heatwave Hazard F1: {test_metrics.heatwave_hazard.f1:.3f} (ROC-AUC: {test_metrics.heatwave_hazard.roc_auc:.3f})")
    logger.info(f"Test Frost Hazard F1: {test_metrics.frost_hazard.f1:.3f} (ROC-AUC: {test_metrics.frost_hazard.roc_auc:.3f})")
    logger.info(f"Test Deluge Hazard F1: {test_metrics.deluge_hazard.f1:.3f} (ROC-AUC: {test_metrics.deluge_hazard.roc_auc:.3f})")
    logger.info(f"Test Mean CRPS: {test_metrics.mean_crps:.4f}")

    # Save test metrics
    with open(CHECKPOINTS_DIR / "test_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics.to_dict(), f, indent=2)

    logger.info("Training and evaluation completed successfully!")


if __name__ == "__main__":
    main()
