"""Comprehensive meteorological evaluation metrics for probabilistic forecasting and extreme events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class TargetMetrics:
    mae_overall: float
    rmse_overall: float
    mae_day1: float
    mae_day3: float
    mae_day7: float
    mae_day14: float
    coverage_80pct: float  # Percentage of true values falling between P10 and P90 (ideal = 80%)


@dataclass
class HazardMetrics:
    precision: float
    recall: float
    f1: float
    roc_auc: float
    num_positive_events: int


@dataclass
class ClimateEvaluationMetrics:
    tmax_metrics: TargetMetrics
    tmin_metrics: TargetMetrics
    prcp_metrics: TargetMetrics
    heatwave_hazard: HazardMetrics
    frost_hazard: HazardMetrics
    deluge_hazard: HazardMetrics
    mean_crps: float

    def to_dict(self) -> Dict:
        return asdict(self)


def calculate_target_metrics(
    q_pred: np.ndarray,  # [N, 14, 3] (P10, P50, P90)
    y_true: np.ndarray,  # [N, 14]
) -> TargetMetrics:
    """Compute MAE, RMSE, and 80% coverage interval calibration."""
    p50 = q_pred[..., 1]
    p10 = q_pred[..., 0]
    p90 = q_pred[..., 2]

    err = p50 - y_true
    mae_overall = float(np.mean(np.abs(err)))
    rmse_overall = float(np.sqrt(np.mean(err ** 2)))

    mae_day1 = float(np.mean(np.abs(err[:, 0]))) if err.shape[1] > 0 else mae_overall
    mae_day3 = float(np.mean(np.abs(err[:, 2]))) if err.shape[1] > 2 else mae_overall
    mae_day7 = float(np.mean(np.abs(err[:, 6]))) if err.shape[1] > 6 else mae_overall
    mae_day14 = float(np.mean(np.abs(err[:, 13]))) if err.shape[1] > 13 else mae_overall

    in_band = (y_true >= p10) & (y_true <= p90)
    coverage_80 = float(np.mean(in_band)) * 100.0

    return TargetMetrics(
        mae_overall=round(mae_overall, 3),
        rmse_overall=round(rmse_overall, 3),
        mae_day1=round(mae_day1, 3),
        mae_day3=round(mae_day3, 3),
        mae_day7=round(mae_day7, 3),
        mae_day14=round(mae_day14, 3),
        coverage_80pct=round(coverage_80, 2),
    )


def calculate_hazard_metrics(
    prob_pred: np.ndarray,  # [N, 14] or [N * 14]
    y_true: np.ndarray,     # [N, 14] or [N * 14]
    threshold: float = 0.5,
) -> HazardMetrics:
    """Compute Precision, Recall, F1, and ROC-AUC for binary hazard prediction."""
    y_flat = y_true.reshape(-1)
    p_flat = prob_pred.reshape(-1)
    pred_binary = (p_flat >= threshold).astype(int)

    num_pos = int(np.sum(y_flat))

    if num_pos == 0 or num_pos == len(y_flat):
        return HazardMetrics(precision=0.0, recall=0.0, f1=0.0, roc_auc=0.5, num_positive_events=num_pos)

    prec = float(precision_score(y_flat, pred_binary, zero_division=0))
    rec = float(recall_score(y_flat, pred_binary, zero_division=0))
    f1 = float(f1_score(y_flat, pred_binary, zero_division=0))

    try:
        auc = float(roc_auc_score(y_flat, p_flat))
    except Exception:
        auc = 0.5

    return HazardMetrics(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        roc_auc=round(auc, 4),
        num_positive_events=num_pos,
    )


def estimate_crps(
    q_pred: np.ndarray,  # [N, 14, 3, 3] (quantiles P10, P50, P90)
    y_true: np.ndarray,  # [N, 14, 3]
) -> float:
    """Approximate Continuous Ranked Probability Score (CRPS) from discrete quantiles."""
    # CRPS approx via quantile loss average across tau in [0.10, 0.50, 0.90]
    taus = [0.10, 0.50, 0.90]
    total_pinball = 0.0
    for q_idx, tau in enumerate(taus):
        diff = y_true - q_pred[..., q_idx]
        pinball = np.maximum(tau * diff, (tau - 1.0) * diff)
        total_pinball += np.mean(pinball)
    return round(float(total_pinball / len(taus)), 4)


def evaluate_predictions(
    q_preds: np.ndarray,    # [N, 14, 3, 3]
    prob_preds: np.ndarray, # [N, 14, 3]
    y_targets: np.ndarray,  # [N, 14, 3]
    y_hazards: np.ndarray,  # [N, 14, 3]
) -> ClimateEvaluationMetrics:
    """Comprehensive evaluation of model predictions."""
    tmax_m = calculate_target_metrics(q_preds[:, :, 0, :], y_targets[:, :, 0])
    tmin_m = calculate_target_metrics(q_preds[:, :, 1, :], y_targets[:, :, 1])
    prcp_m = calculate_target_metrics(q_preds[:, :, 2, :], y_targets[:, :, 2])

    hw_m = calculate_hazard_metrics(prob_preds[:, :, 0], y_hazards[:, :, 0])
    frost_m = calculate_hazard_metrics(prob_preds[:, :, 1], y_hazards[:, :, 1])
    deluge_m = calculate_hazard_metrics(prob_preds[:, :, 2], y_hazards[:, :, 2])

    crps_val = estimate_crps(q_preds, y_targets)

    return ClimateEvaluationMetrics(
        tmax_metrics=tmax_m,
        tmin_metrics=tmin_m,
        prcp_metrics=prcp_m,
        heatwave_hazard=hw_m,
        frost_hazard=frost_m,
        deluge_hazard=deluge_m,
        mean_crps=crps_val,
    )
