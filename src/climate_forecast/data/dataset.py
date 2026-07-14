"""PyTorch SpatioTemporal dataset, sliding window generation, and data scaling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from climate_forecast.config import (
    CONTEXT_WINDOW_DAYS,
    FORECAST_HORIZON_DAYS,
    TrainingConfig,
)
from climate_forecast.data.climatology import ClimatologyEngine


@dataclass
class ScalerParams:
    tmax_mean: float
    tmax_std: float
    tmin_mean: float
    tmin_std: float
    prcp_mean: float
    prcp_std: float
    elev_mean: float
    elev_std: float


class ClimateDataScaler:
    """Standardizes weather variables and spatial elevation."""

    def __init__(self, params: Optional[ScalerParams] = None):
        self.params = params

    def fit(self, df: pd.DataFrame, stations_meta: Optional[Dict] = None) -> ClimateDataScaler:
        tmax = df["TMAX"].dropna().values if "TMAX" in df else np.array([20.0])
        tmin = df["TMIN"].dropna().values if "TMIN" in df else np.array([10.0])
        prcp = np.log1p(np.maximum(0.0, df["PRCP"].dropna().values)) if "PRCP" in df else np.array([1.0])

        elevs = [s.elevation for s in stations_meta.values()] if stations_meta else [200.0]

        self.params = ScalerParams(
            tmax_mean=float(np.mean(tmax)) if len(tmax) > 0 else 20.0,
            tmax_std=float(max(np.std(tmax), 1.0)) if len(tmax) > 0 else 10.0,
            tmin_mean=float(np.mean(tmin)) if len(tmin) > 0 else 10.0,
            tmin_std=float(max(np.std(tmin), 1.0)) if len(tmin) > 0 else 8.0,
            prcp_mean=float(np.mean(prcp)) if len(prcp) > 0 else 1.0,
            prcp_std=float(max(np.std(prcp), 0.5)) if len(prcp) > 0 else 1.0,
            elev_mean=float(np.mean(elevs)) if len(elevs) > 0 else 200.0,
            elev_std=float(max(np.std(elevs), 50.0)) if len(elevs) > 0 else 200.0,
        )
        return self

    def transform_tmax(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (x - self.params.tmax_mean) / self.params.tmax_std

    def inverse_transform_tmax(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (x * self.params.tmax_std) + self.params.tmax_mean

    def transform_tmin(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (x - self.params.tmin_mean) / self.params.tmin_std

    def inverse_transform_tmin(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return (x * self.params.tmin_std) + self.params.tmin_mean

    def transform_prcp(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(x, torch.Tensor):
            log_x = torch.log1p(torch.clamp(x, min=0.0))
        else:
            log_x = np.log1p(np.maximum(0.0, x))
        return (log_x - self.params.prcp_mean) / self.params.prcp_std

    def inverse_transform_prcp(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        scaled = (x * self.params.prcp_std) + self.params.prcp_mean
        if isinstance(x, torch.Tensor):
            return torch.clamp(torch.expm1(scaled), min=0.0)
        else:
            return np.maximum(0.0, np.expm1(scaled))

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self.params), f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> ClimateDataScaler:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(ScalerParams(**data))


class SpatiotemporalClimateDataset(Dataset):
    """PyTorch Dataset for multi-step spatiotemporal forecasting."""

    def __init__(
        self,
        samples_x_seq: np.ndarray,          # [N, 30, 7]
        samples_x_spatial: np.ndarray,      # [N, 3] (lat, lon, elev)
        samples_x_future_time: np.ndarray,  # [N, 14, 2] (sin_doy, cos_doy for t+1..t+14)
        samples_y_targets: np.ndarray,      # [N, 14, 3] (TMAX, TMIN, PRCP)
        samples_y_hazards: np.ndarray,      # [N, 14, 3] (heatwave, frost, deluge binary)
        station_ids: List[str],
        start_dates: List[str],
    ):
        self.x_seq = torch.from_numpy(samples_x_seq).float()
        self.x_spatial = torch.from_numpy(samples_x_spatial).float()
        self.x_future_time = torch.from_numpy(samples_x_future_time).float()
        self.y_targets = torch.from_numpy(samples_y_targets).float()
        self.y_hazards = torch.from_numpy(samples_y_hazards).float()
        self.station_ids = station_ids
        self.start_dates = start_dates

    def __len__(self) -> int:
        return len(self.x_seq)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_seq": self.x_seq[idx],
            "x_spatial": self.x_spatial[idx],
            "x_future_time": self.x_future_time[idx],
            "y_targets": self.y_targets[idx],
            "y_hazards": self.y_hazards[idx],
        }


def build_spatiotemporal_samples(
    df: pd.DataFrame,
    climatology: ClimatologyEngine,
    scaler: ClimateDataScaler,
    context_days: int = CONTEXT_WINDOW_DAYS,
    horizon_days: int = FORECAST_HORIZON_DAYS,
    stride_days: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    """Construct sliding window samples with strict NaN checking."""
    x_seq_list = []
    x_spatial_list = []
    x_future_time_list = []
    y_targets_list = []
    y_hazards_list = []
    station_id_list = []
    start_date_list = []

    seq_len = context_days + horizon_days

    for st_id, st_df in df.groupby("station_id"):
        st_meta = climatology.get_station(st_id)
        if st_meta is None:
            continue

        st_df = st_df.sort_values("date").copy()
        st_df["TMAX"] = st_df["TMAX"].interpolate(method="linear", limit=7).ffill().bfill()
        st_df["TMIN"] = st_df["TMIN"].interpolate(method="linear", limit=7).ffill().bfill()
        st_df["PRCP"] = st_df["PRCP"].fillna(0.0)

        # Drop any remaining unfillable rows
        st_df = st_df.dropna(subset=["TMAX", "TMIN", "PRCP"])
        n_rows = len(st_df)
        if n_rows < seq_len:
            continue

        dates = pd.to_datetime(st_df["date"]).values
        tmax = st_df["TMAX"].values.astype(np.float32)
        tmin = st_df["TMIN"].values.astype(np.float32)
        prcp = st_df["PRCP"].values.astype(np.float32)
        doy = st_df["day_of_year"].values

        # Normalize features
        tmax_norm = scaler.transform_tmax(tmax)
        tmin_norm = scaler.transform_tmin(tmin)
        prcp_norm = scaler.transform_prcp(prcp)

        sin_doy = np.sin(2.0 * np.pi * doy / 365.25).astype(np.float32)
        cos_doy = np.cos(2.0 * np.pi * doy / 365.25).astype(np.float32)

        # Anomalies & Hazards
        tmax_anom = np.zeros(n_rows, dtype=np.float32)
        tmin_anom = np.zeros(n_rows, dtype=np.float32)
        is_hw = np.zeros(n_rows, dtype=np.float32)
        is_frost = np.zeros(n_rows, dtype=np.float32)
        is_deluge = np.zeros(n_rows, dtype=np.float32)

        for i in range(n_rows):
            ta, tma, hw, fr, de = climatology.compute_anomalies_and_hazards(
                st_id, int(doy[i]), float(tmax[i]), float(tmin[i]), float(prcp[i])
            )
            tmax_anom[i] = ta
            tmin_anom[i] = tma
            is_hw[i] = 1.0 if hw else 0.0
            is_frost[i] = 1.0 if fr else 0.0
            is_deluge[i] = 1.0 if de else 0.0

        features = np.column_stack([
            tmax_norm, tmin_norm, prcp_norm, sin_doy, cos_doy, tmax_anom, tmin_anom
        ]).astype(np.float32)

        elev_norm = (st_meta.elevation - scaler.params.elev_mean) / scaler.params.elev_std
        spatial_vec = np.array([st_meta.latitude / 90.0, st_meta.longitude / 180.0, elev_norm], dtype=np.float32)

        for start_idx in range(0, n_rows - seq_len + 1, stride_days):
            dt_start = dates[start_idx]
            dt_end = dates[start_idx + seq_len - 1]
            days_span = (dt_end - dt_start) / np.timedelta64(1, "D")
            if days_span > (seq_len + 3):
                continue

            ctx_end = start_idx + context_days
            f_end = ctx_end + horizon_days

            x_seq = features[start_idx:ctx_end]
            x_future_time = np.column_stack([sin_doy[ctx_end:f_end], cos_doy[ctx_end:f_end]]).astype(np.float32)
            y_tgt = np.column_stack([tmax[ctx_end:f_end], tmin[ctx_end:f_end], prcp[ctx_end:f_end]]).astype(np.float32)
            y_haz = np.column_stack([is_hw[ctx_end:f_end], is_frost[ctx_end:f_end], is_deluge[ctx_end:f_end]]).astype(np.float32)

            # Strict NaN guard
            if not np.isfinite(x_seq).all() or not np.isfinite(y_tgt).all() or not np.isfinite(y_haz).all():
                continue

            x_seq_list.append(x_seq)
            x_spatial_list.append(spatial_vec)
            x_future_time_list.append(x_future_time)
            y_targets_list.append(y_tgt)
            y_hazards_list.append(y_haz)
            station_id_list.append(st_id)
            start_date_list.append(str(pd.to_datetime(dt_start).date()))

    if not x_seq_list:
        return (
            np.zeros((0, context_days, 7), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, horizon_days, 2), dtype=np.float32),
            np.zeros((0, horizon_days, 3), dtype=np.float32),
            np.zeros((0, horizon_days, 3), dtype=np.float32),
            [],
            [],
        )

    return (
        np.array(x_seq_list, dtype=np.float32),
        np.array(x_spatial_list, dtype=np.float32),
        np.array(x_future_time_list, dtype=np.float32),
        np.array(y_targets_list, dtype=np.float32),
        np.array(y_hazards_list, dtype=np.float32),
        station_id_list,
        start_date_list,
    )


def create_dataloaders(
    df: pd.DataFrame,
    climatology: ClimatologyEngine,
    scaler: Optional[ClimateDataScaler] = None,
    training_cfg: Optional[TrainingConfig] = None,
    stations_meta: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, ClimateDataScaler]:
    """Create train, validation, and test PyTorch DataLoaders with zero-leakage temporal split."""
    cfg = training_cfg or TrainingConfig()

    df_train = df[df["year"] <= cfg.train_end_year].copy()
    df_val = df[df["year"] == cfg.val_year].copy()
    df_test = df[df["year"] >= cfg.test_start_year].copy()

    if df_val.empty:
        years = sorted(df["year"].unique())
        df_train = df[df["year"] < years[-2]].copy()
        df_val = df[df["year"] == years[-2]].copy()
        df_test = df[df["year"] == years[-1]].copy()

    if scaler is None:
        scaler = ClimateDataScaler().fit(df_train, stations_meta=stations_meta)

    # Use stride 7 for efficient training over large corpus
    x_seq_tr, x_sp_tr, x_ft_tr, y_tgt_tr, y_haz_tr, st_tr, dt_tr = build_spatiotemporal_samples(
        df_train, climatology, scaler, stride_days=7
    )
    x_seq_val, x_sp_val, x_ft_val, y_tgt_val, y_haz_val, st_val, dt_val = build_spatiotemporal_samples(
        df_val, climatology, scaler, stride_days=7
    )
    x_seq_te, x_sp_te, x_ft_te, y_tgt_te, y_haz_te, st_te, dt_te = build_spatiotemporal_samples(
        df_test, climatology, scaler, stride_days=7
    )

    train_ds = SpatiotemporalClimateDataset(x_seq_tr, x_sp_tr, x_ft_tr, y_tgt_tr, y_haz_tr, st_tr, dt_tr)
    val_ds = SpatiotemporalClimateDataset(x_seq_val, x_sp_val, x_ft_val, y_tgt_val, y_haz_val, st_val, dt_val)
    test_ds = SpatiotemporalClimateDataset(x_seq_te, x_sp_te, x_ft_te, y_tgt_te, y_haz_te, st_te, dt_te)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler
