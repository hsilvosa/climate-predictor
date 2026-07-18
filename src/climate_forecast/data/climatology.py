"""Climatological baselines, 30-year normals, and extreme threshold calculators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from climate_forecast.config import (
    CLIMATOLOGY_END_YEAR,
    CLIMATOLOGY_START_YEAR,
    DELUGE_ABSOLUTE_THRESHOLD_MM,
    DELUGE_PERCENTILE_THRESHOLD,
    FROST_FREEZE_THRESHOLD_C,
    HEATWAVE_ABSOLUTE_THRESHOLD_C,
    HEATWAVE_PERCENTILE_THRESHOLD,
    PROCESSED_DATA_DIR,
)


@dataclass
class DayClimatology:
    day_of_year: int
    tmax_mean: float
    tmax_std: float
    tmax_p10: float
    tmax_p50: float
    tmax_p90: float
    tmax_p95: float
    tmin_mean: float
    tmin_std: float
    tmin_p05: float
    tmin_p10: float
    tmin_p50: float
    tmin_p90: float
    prcp_mean: float
    prcp_p90: float
    prcp_p95: float
    prcp_p99: float


@dataclass
class StationClimatology:
    station_id: str
    latitude: float
    longitude: float
    elevation: float
    name: str
    country_name: str
    climate_zone: str
    all_time_tmax_record: float
    all_time_tmin_record: float
    all_time_prcp_record: float
    daily_normals: Dict[int, DayClimatology] = field(default_factory=dict)


class ClimatologyEngine:
    """Computes, serializes, and queries 30-year daily climatological normals."""

    def __init__(self, stations_clim: Optional[Dict[str, StationClimatology]] = None):
        self.stations_clim: Dict[str, StationClimatology] = stations_clim or {}

    @classmethod
    def fit_from_dataframe(
        cls,
        df: pd.DataFrame,
        stations_metadata: Optional[Dict] = None,
        start_year: int = CLIMATOLOGY_START_YEAR,
        end_year: int = CLIMATOLOGY_END_YEAR,
        smoothing_window_days: int = 7,
    ) -> ClimatologyEngine:
        """Calculate 30-year daily normals with temporal rolling smoothing."""
        df_base = df[(df["year"] >= start_year) & (df["year"] <= end_year)].copy()
        if df_base.empty:
            df_base = df.copy()

        stations_dict: Dict[str, StationClimatology] = {}

        for station_id, st_group in df_base.groupby("station_id"):
            meta = stations_metadata.get(station_id) if stations_metadata else None
            lat = meta.latitude if meta else float(st_group.get("latitude", pd.Series([0.0])).iloc[0])
            lon = meta.longitude if meta else float(st_group.get("longitude", pd.Series([0.0])).iloc[0])
            elev = meta.elevation if meta else float(st_group.get("elevation", pd.Series([0.0])).iloc[0])
            name = meta.name if meta else station_id
            cname = meta.country_name if meta else ""
            czone = meta.climate_zone if meta else "Temperate"

            # All-time records across entire available series
            full_st = df[df["station_id"] == station_id]
            tmax_rec = float(full_st["TMAX"].max()) if "TMAX" in full_st and full_st["TMAX"].notna().any() else 45.0
            tmin_rec = float(full_st["TMIN"].min()) if "TMIN" in full_st and full_st["TMIN"].notna().any() else -25.0
            prcp_rec = float(full_st["PRCP"].max()) if "PRCP" in full_st and full_st["PRCP"].notna().any() else 100.0

            daily_norms: Dict[int, DayClimatology] = {}

            # Pre-extract arrays for speed
            doy_arr = st_group["day_of_year"].values
            tmax_arr = st_group["TMAX"].dropna().values if "TMAX" in st_group else np.array([])
            tmin_arr = st_group["TMIN"].dropna().values if "TMIN" in st_group else np.array([])
            prcp_arr = st_group["PRCP"].dropna().values if "PRCP" in st_group else np.array([])

            # Calculate daily normals for DOY 1..366
            for doy in range(1, 367):
                # Circular distance window around day-of-year
                # Window accounts for end-of-year wrapping (e.g. Dec 31 to Jan 2)
                diff = np.abs(st_group["day_of_year"].values - doy)
                circ_dist = np.minimum(diff, 365 - diff)
                mask = circ_dist <= smoothing_window_days

                win_tmax = st_group.loc[mask, "TMAX"].dropna().values if "TMAX" in st_group else np.array([])
                win_tmin = st_group.loc[mask, "TMIN"].dropna().values if "TMIN" in st_group else np.array([])
                win_prcp = st_group.loc[mask, "PRCP"].dropna().values if "PRCP" in st_group else np.array([])

                # Default fallback values if sparse
                tmax_mean = float(np.mean(win_tmax)) if len(win_tmax) > 0 else 20.0
                tmax_std = float(np.std(win_tmax)) if len(win_tmax) > 1 else 4.0
                tmax_p10 = float(np.percentile(win_tmax, 10)) if len(win_tmax) > 0 else tmax_mean - 1.28 * tmax_std
                tmax_p50 = float(np.median(win_tmax)) if len(win_tmax) > 0 else tmax_mean
                tmax_p90 = float(np.percentile(win_tmax, 90)) if len(win_tmax) > 0 else tmax_mean + 1.28 * tmax_std
                tmax_p95 = float(np.percentile(win_tmax, 95)) if len(win_tmax) > 0 else tmax_mean + 1.645 * tmax_std

                tmin_mean = float(np.mean(win_tmin)) if len(win_tmin) > 0 else 10.0
                tmin_std = float(np.std(win_tmin)) if len(win_tmin) > 1 else 4.0
                tmin_p05 = float(np.percentile(win_tmin, 5)) if len(win_tmin) > 0 else tmin_mean - 1.645 * tmin_std
                tmin_p10 = float(np.percentile(win_tmin, 10)) if len(win_tmin) > 0 else tmin_mean - 1.28 * tmin_std
                tmin_p50 = float(np.median(win_tmin)) if len(win_tmin) > 0 else tmin_mean
                tmin_p90 = float(np.percentile(win_tmin, 90)) if len(win_tmin) > 0 else tmin_mean + 1.28 * tmin_std

                prcp_mean = float(np.mean(win_prcp)) if len(win_prcp) > 0 else 2.5
                prcp_p90 = float(np.percentile(win_prcp, 90)) if len(win_prcp) > 0 else 8.0
                prcp_p95 = float(np.percentile(win_prcp, 95)) if len(win_prcp) > 0 else 15.0
                prcp_p99 = float(np.percentile(win_prcp, 99)) if len(win_prcp) > 0 else 30.0

                daily_norms[doy] = DayClimatology(
                    day_of_year=doy,
                    tmax_mean=round(tmax_mean, 2),
                    tmax_std=round(max(tmax_std, 0.5), 2),
                    tmax_p10=round(tmax_p10, 2),
                    tmax_p50=round(tmax_p50, 2),
                    tmax_p90=round(tmax_p90, 2),
                    tmax_p95=round(tmax_p95, 2),
                    tmin_mean=round(tmin_mean, 2),
                    tmin_std=round(max(tmin_std, 0.5), 2),
                    tmin_p05=round(tmin_p05, 2),
                    tmin_p10=round(tmin_p10, 2),
                    tmin_p50=round(tmin_p50, 2),
                    tmin_p90=round(tmin_p90, 2),
                    prcp_mean=round(prcp_mean, 2),
                    prcp_p90=round(prcp_p90, 2),
                    prcp_p95=round(prcp_p95, 2),
                    prcp_p99=round(prcp_p99, 2),
                )

            stations_dict[station_id] = StationClimatology(
                station_id=station_id,
                latitude=round(lat, 4),
                longitude=round(lon, 4),
                elevation=round(elev, 1),
                name=name,
                country_name=cname,
                climate_zone=czone,
                all_time_tmax_record=round(tmax_rec, 2),
                all_time_tmin_record=round(tmin_rec, 2),
                all_time_prcp_record=round(prcp_rec, 2),
                daily_normals=daily_norms,
            )

        return cls(stations_dict)

    def get_station(self, station_id: str) -> Optional[StationClimatology]:
        return self.stations_clim.get(station_id)

    def get_norm(self, station_id: str, day_of_year: int) -> Optional[DayClimatology]:
        st = self.get_station(station_id)
        if not st:
            return None
        doy = max(1, min(366, int(day_of_year)))
        return st.daily_normals.get(doy)

    def compute_anomalies_and_hazards(
        self,
        station_id: str,
        day_of_year: int,
        tmax: float,
        tmin: float,
        prcp: float,
    ) -> Tuple[float, float, bool, bool, bool]:
        """Compute (tmax_anom, tmin_anom, is_heatwave, is_frost, is_deluge)."""
        norm = self.get_norm(station_id, day_of_year)
        if norm is None:
            # Fallback global standard
            return 0.0, 0.0, (tmax >= HEATWAVE_ABSOLUTE_THRESHOLD_C), (tmin <= FROST_FREEZE_THRESHOLD_C), (prcp >= DELUGE_ABSOLUTE_THRESHOLD_MM)

        tmax_anom = (tmax - norm.tmax_mean) / (norm.tmax_std + 1e-4)
        tmin_anom = (tmin - norm.tmin_mean) / (norm.tmin_std + 1e-4)

        is_heatwave = (tmax >= norm.tmax_p95) or (tmax >= HEATWAVE_ABSOLUTE_THRESHOLD_C)
        is_frost = (tmin <= FROST_FREEZE_THRESHOLD_C) or (tmin <= norm.tmin_p05)
        is_deluge = (prcp >= norm.prcp_p99) or (prcp >= DELUGE_ABSOLUTE_THRESHOLD_MM)

        return float(tmax_anom), float(tmin_anom), bool(is_heatwave), bool(is_frost), bool(is_deluge)

    def save_json(self, output_path: Path | str) -> None:
        """Serialize station catalog and climatology to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for st_id, st in self.stations_clim.items():
            st_dict = asdict(st)
            # Ensure day keys are converted to strings in json
            st_dict["daily_normals"] = {str(k): asdict(v) for k, v in st.daily_normals.items()}
            data[st_id] = st_dict

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, input_path: Path | str) -> ClimatologyEngine:
        """Load station catalog and climatology from JSON."""
        path = Path(input_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        stations_dict: Dict[str, StationClimatology] = {}
        for st_id, st_raw in data.items():
            norms = {}
            for doy_str, norm_raw in st_raw.get("daily_normals", {}).items():
                norms[int(doy_str)] = DayClimatology(**norm_raw)

            stations_dict[st_id] = StationClimatology(
                station_id=st_raw["station_id"],
                latitude=st_raw["latitude"],
                longitude=st_raw["longitude"],
                elevation=st_raw["elevation"],
                name=st_raw["name"],
                country_name=st_raw["country_name"],
                climate_zone=st_raw.get("climate_zone", "Temperate"),
                all_time_tmax_record=st_raw.get("all_time_tmax_record", 45.0),
                all_time_tmin_record=st_raw.get("all_time_tmin_record", -25.0),
                all_time_prcp_record=st_raw.get("all_time_prcp_record", 100.0),
                daily_normals=norms,
            )
        return cls(stations_dict)
