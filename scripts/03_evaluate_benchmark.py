"""Step 3: Generate detailed benchmark report comparing model against Persistence and Climatology baselines."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
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
from climate_forecast.training.metrics import calculate_hazard_metrics, calculate_target_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate forecasting benchmarks")
    parser.add_argument("--parquet-path", default=str(PROCESSED_DATA_DIR / "observations.parquet"))
    parser.add_argument("--climatology-path", default=str(PROCESSED_DATA_DIR / "stations_climatology.json"))
    parser.add_argument("--model-path", default=str(CHECKPOINTS_DIR / "best_climate_forecaster.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_obs = pd.read_parquet(args.parquet_path)
    climatology = ClimatologyEngine.load_json(args.climatology_path)

    _, _, test_loader, scaler = create_dataloaders(
        df=df_obs,
        climatology=climatology,
        stations_meta=climatology.stations_clim,
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    cfg = ModelConfig(**checkpoint["model_config"])
    model = ClimateSpatiotemporalNet(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_q_preds = []
    all_prob_preds = []
    all_y_targets = []
    all_y_hazards = []
    all_x_seqs = []

    with torch.no_grad():
        for batch in test_loader:
            x_seq = batch["x_seq"].to(device)
            x_sp = batch["x_spatial"].to(device)
            x_ft = batch["x_future_time"].to(device)
            y_tgt = batch["y_targets"].to(device)
            y_haz = batch["y_hazards"].to(device)

            out = model(x_seq, x_sp, x_ft)
            all_q_preds.append(out["quantiles"].cpu().numpy())
            all_prob_preds.append(out["hazard_probs"].cpu().numpy())
            all_y_targets.append(y_tgt.cpu().numpy())
            all_y_hazards.append(y_haz.cpu().numpy())
            all_x_seqs.append(x_seq.cpu().numpy())

    q_preds = np.concatenate(all_q_preds, axis=0)
    prob_preds = np.concatenate(all_prob_preds, axis=0)
    y_targets = np.concatenate(all_y_targets, axis=0)
    y_hazards = np.concatenate(all_y_hazards, axis=0)
    x_seqs = np.concatenate(all_x_seqs, axis=0)

    # Model metrics
    tmax_m = calculate_target_metrics(q_preds[:, :, 0, :], y_targets[:, :, 0])
    tmin_m = calculate_target_metrics(q_preds[:, :, 1, :], y_targets[:, :, 1])
    prcp_m = calculate_target_metrics(q_preds[:, :, 2, :], y_targets[:, :, 2])

    hw_m = calculate_hazard_metrics(prob_preds[:, :, 0], y_hazards[:, :, 0])
    frost_m = calculate_hazard_metrics(prob_preds[:, :, 1], y_hazards[:, :, 1])
    deluge_m = calculate_hazard_metrics(prob_preds[:, :, 2], y_hazards[:, :, 2])

    # Persistence Baseline: Last observed day repeats for all 14 future days
    last_tmax = scaler.inverse_transform_tmax(x_seqs[:, -1, 0])
    last_tmin = scaler.inverse_transform_tmin(x_seqs[:, -1, 1])
    persist_tmax = np.repeat(last_tmax[:, np.newaxis], 14, axis=1)
    persist_tmin = np.repeat(last_tmin[:, np.newaxis], 14, axis=1)

    persist_tmax_mae = float(np.mean(np.abs(persist_tmax - y_targets[:, :, 0])))
    persist_tmin_mae = float(np.mean(np.abs(persist_tmin - y_targets[:, :, 1])))

    # Skill Score = 1 - (MAE_model / MAE_persistence)
    tmax_skill = (1.0 - (tmax_m.mae_overall / max(0.01, persist_tmax_mae))) * 100.0
    tmin_skill = (1.0 - (tmin_m.mae_overall / max(0.01, persist_tmin_mae))) * 100.0

    report = {
        "model_architecture": "ClimateSpatiotemporalNet (Spherical Harmonic + TCN + Attention)",
        "evaluation_samples": int(len(y_targets)),
        "tmax": {
            "model_mae_overall_c": tmax_m.mae_overall,
            "model_rmse_c": tmax_m.rmse_overall,
            "mae_day1": tmax_m.mae_day1,
            "mae_day3": tmax_m.mae_day3,
            "mae_day7": tmax_m.mae_day7,
            "mae_day14": tmax_m.mae_day14,
            "coverage_80pct": tmax_m.coverage_80pct,
            "persistence_mae_c": round(persist_tmax_mae, 3),
            "skill_score_pct": round(tmax_skill, 1),
        },
        "tmin": {
            "model_mae_overall_c": tmin_m.mae_overall,
            "model_rmse_c": tmin_m.rmse_overall,
            "mae_day1": tmin_m.mae_day1,
            "mae_day3": tmin_m.mae_day3,
            "mae_day7": tmin_m.mae_day7,
            "mae_day14": tmin_m.mae_day14,
            "coverage_80pct": tmin_m.coverage_80pct,
            "persistence_mae_c": round(persist_tmin_mae, 3),
            "skill_score_pct": round(tmin_skill, 1),
        },
        "prcp": {
            "model_mae_overall_mm": prcp_m.mae_overall,
            "coverage_80pct": prcp_m.coverage_80pct,
        },
        "extreme_hazards": {
            "heatwave_f1": hw_m.f1,
            "heatwave_roc_auc": hw_m.roc_auc,
            "frost_f1": frost_m.f1,
            "frost_roc_auc": frost_m.roc_auc,
            "deluge_f1": deluge_m.f1,
            "deluge_roc_auc": deluge_m.roc_auc,
        },
    }

    report_path = CHECKPOINTS_DIR / "benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 60)
    logger.info("BENCHMARK EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info(f"TMAX MAE: {tmax_m.mae_overall:.2f} C (Skill vs Persistence: +{tmax_skill:.1f}%)")
    logger.info(f"TMIN MAE: {tmin_m.mae_overall:.2f} C (Skill vs Persistence: +{tmin_skill:.1f}%)")
    logger.info(f"Heatwave Alert F1: {hw_m.f1:.3f} | ROC-AUC: {hw_m.roc_auc:.3f}")
    logger.info(f"Frost Alert F1: {frost_m.f1:.3f} | ROC-AUC: {frost_m.roc_auc:.3f}")
    logger.info(f"Deluge Alert F1: {deluge_m.f1:.3f} | ROC-AUC: {deluge_m.roc_auc:.3f}")
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
