"""Meteorological indices, Excess Heat Factor (EHF), Frost risk, and Spatial Interpolation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


def calculate_excess_heat_factor(
    tmax_3day_avg: float,
    tmin_3day_avg: float,
    t95_threshold: float,
    past_30day_mean: float,
) -> Tuple[float, str]:
    """Calculate Excess Heat Factor (EHF) index and alert severity tier.

    Reference: Nairn & Fawcett (2014) Excess Heat Factor.
    """
    t_3day = (tmax_3day_avg + tmin_3day_avg) / 2.0
    ehi_sig = t_3day - t95_threshold
    ehi_accl = t_3day - past_30day_mean

    ehf = max(0.0, ehi_sig) * max(1.0, ehi_accl)

    if ehf <= 0.0:
        tier = "Normal"
    elif ehf < 2.0:
        tier = "Advisory"
    elif ehf < 6.0:
        tier = "Watch"
    elif ehf < 14.0:
        tier = "Warning"
    else:
        tier = "Emergency"

    return round(float(ehf), 2), tier


def classify_heatwave_tier(
    prob_heatwave: float,
    tmax_val: float,
    t95_norm: float,
    all_time_tmax: float,
) -> Dict[str, any]:
    """Classify heatwave risk level and record proximity."""
    diff_record = all_time_tmax - tmax_val
    is_record_threat = diff_record <= 1.5

    if prob_heatwave >= 0.85 or tmax_val >= (all_time_tmax - 0.5):
        level = "Extreme Danger"
        color = "#ef4444"
    elif prob_heatwave >= 0.65 or tmax_val >= t95_norm:
        level = "High Danger"
        color = "#f97316"
    elif prob_heatwave >= 0.40:
        level = "Moderate Alert"
        color = "#eab308"
    else:
        level = "Normal"
        color = "#10b981"

    return {
        "level": level,
        "color": color,
        "probability": round(float(prob_heatwave), 3),
        "is_record_threat": bool(is_record_threat),
        "degrees_to_record": round(float(diff_record), 1),
    }


def classify_frost_risk(
    prob_frost: float,
    tmin_val: float,
    all_time_tmin: float,
) -> Dict[str, any]:
    """Classify frost / cold wave risk and freezing hazard level."""
    diff_record = tmin_val - all_time_tmin
    is_record_threat = diff_record <= 2.0

    if tmin_val <= -10.0 or prob_frost >= 0.85:
        level = "Severe Freeze"
        color = "#6366f1"
    elif tmin_val <= 0.0 or prob_frost >= 0.50:
        level = "Frost Warning"
        color = "#38bdf8"
    elif tmin_val <= 3.0:
        level = "Frost Advisory"
        color = "#a5f3fc"
    else:
        level = "No Frost"
        color = "#10b981"

    return {
        "level": level,
        "color": color,
        "probability": round(float(prob_frost), 3),
        "is_record_threat": bool(is_record_threat),
        "degrees_above_record": round(float(diff_record), 1),
    }


def classify_precipitation_hazard(
    prcp_val: float,
    prob_deluge: float,
    p99_norm: float,
) -> Dict[str, any]:
    """Classify rainfall hazard and estimated return interval."""
    if prcp_val >= 60.0 or (prcp_val >= p99_norm and prob_deluge >= 0.70):
        level = "Torrential Deluge"
        return_period = "1-in-20+ Year Event"
        color = "#8b5cf6"
    elif prcp_val >= 30.0 or prob_deluge >= 0.50:
        level = "Heavy Rain Warning"
        return_period = "1-in-5 Year Event"
        color = "#3b82f6"
    elif prcp_val >= 10.0:
        level = "Moderate Rain"
        return_period = "Seasonal"
        color = "#60a5fa"
    else:
        level = "Dry / Light"
        return_period = "Normal"
        color = "#10b981"

    return {
        "level": level,
        "return_period": return_period,
        "color": color,
        "probability": round(float(prob_deluge), 3),
        "rainfall_mm": round(float(prcp_val), 1),
    }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def spatial_idw_interpolation(
    target_lat: float,
    target_lon: float,
    target_elevation: float,
    stations_data: List[Dict],  # Each dict has: lat, lon, elev, values: np.ndarray [14, 3]
    power: float = 2.0,
    k_neighbors: int = 6,
    lapse_rate_c_per_1000m: float = 6.5,
) -> Tuple[np.ndarray, List[Dict]]:
    """Spatial Inverse Distance Weighting with atmospheric lapse rate altitude correction.

    Returns interpolated [14, 3] forecast array and list of contributing neighbor stations.
    """
    distances = []
    for item in stations_data:
        d = haversine_distance(target_lat, target_lon, item["lat"], item["lon"])
        distances.append((d, item))

    # Sort by nearest distance
    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k_neighbors]

    weights = []
    adjusted_arrays = []
    contributing_stations = []

    for dist_km, item in k_nearest:
        w = 1.0 / (max(dist_km, 5.0) ** power)
        weights.append(w)

        # Elevation lapse rate correction: temperature drops 6.5 C per 1000m higher
        elev_diff_km = (target_elevation - item["elev"]) / 1000.0
        temp_delta = - (lapse_rate_c_per_1000m * elev_diff_km)

        val_arr = np.copy(item["forecast"])  # [14, 3] or [14, 3, 3]
        if val_arr.ndim == 3:
            # Adjust TMAX (idx 0) and TMIN (idx 1)
            val_arr[:, 0, :] += temp_delta
            val_arr[:, 1, :] += temp_delta
        else:
            val_arr[:, 0] += temp_delta
            val_arr[:, 1] += temp_delta

        adjusted_arrays.append(val_arr)
        contributing_stations.append({
            "station_id": item["station_id"],
            "name": item["name"],
            "distance_km": round(dist_km, 1),
            "weight_pct": 0.0,  # updated below
        })

    total_w = sum(weights)
    for idx, w in enumerate(weights):
        contributing_stations[idx]["weight_pct"] = round((w / total_w) * 100.0, 1)

    # Weighted combination
    interpolated = sum(w * arr for w, arr in zip(weights, adjusted_arrays)) / total_w
    return interpolated, contributing_stations
