"""Global configuration, hyperparameters, and climatological constants."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
HF_EXPORT_DIR = PROJECT_ROOT / "hf_export"
STATIC_DIR = PROJECT_ROOT / "src" / "climate_forecast" / "web" / "static"

# External Dataset Source
RAW_DATA_SOURCE_DIR = Path(r"D:\datasets\noaa-ghcn-daily\data\raw\2026-08-16")
RAW_TAR_ARCHIVE = RAW_DATA_SOURCE_DIR / "ghcnd_all.tar.gz"
STATIONS_FILE = RAW_DATA_SOURCE_DIR / "ghcnd-stations.txt"
INVENTORY_FILE = RAW_DATA_SOURCE_DIR / "ghcnd-inventory.txt"
COUNTRIES_FILE = RAW_DATA_SOURCE_DIR / "ghcnd-countries.txt"

# Core Target Elements in NOAA GHCN-Daily
TARGET_ELEMENTS = ["TMAX", "TMIN", "PRCP"]

# Scale Factors (NOAA values are stored in tenths)
# TMAX, TMIN: tenths of degrees C -> degrees C (divide by 10.0)
# PRCP: tenths of mm -> mm (divide by 10.0)
VALUE_SCALE_FACTOR = 0.10

# Sequence Lengths & Forecasting Horizon
CONTEXT_WINDOW_DAYS = 30   # History lookback
FORECAST_HORIZON_DAYS = 14 # Multi-step rollout horizon (1..14 days)

# Quantiles for Probabilistic Forecasting
DEFAULT_QUANTILES = [0.10, 0.50, 0.90]

# Climatology Baseline Period (WMO Standard 30-year normal)
CLIMATOLOGY_START_YEAR = 1991
CLIMATOLOGY_END_YEAR = 2020

# Extreme Event Definitions
HEATWAVE_ABSOLUTE_THRESHOLD_C = 38.0   # Extreme heat threshold (°C)
HEATWAVE_PERCENTILE_THRESHOLD = 95.0   # Station-specific climatological 95th percentile
FROST_FREEZE_THRESHOLD_C = 0.0         # Freezing frost threshold (°C)
SEVERE_FROST_THRESHOLD_C = -10.0       # Severe freeze threshold (°C)
DELUGE_ABSOLUTE_THRESHOLD_MM = 50.0    # Torrential rainfall threshold (mm/day)
DELUGE_PERCENTILE_THRESHOLD = 99.0     # Station-specific 99th percentile

# Spatial Spherical Harmonics
NUM_SPHERICAL_HARMONIC_FREQS = 8  # 2^0, 2^1, ... 2^7


@dataclass
class ModelConfig:
    """Hyperparameters for SpatioTemporal Climate Forecaster."""
    input_features_per_step: int = 7       # [tmax, tmin, prcp, sin_doy, cos_doy, tmax_anom, tmin_anom]
    spatial_embed_dim: int = 64            # Spherical harmonic + elevation + region embedding
    hidden_dim: int = 128                  # Backbone latent width
    tcn_num_layers: int = 4                # Dilated residual TCN blocks
    tcn_kernel_size: int = 3               # Conv kernel size
    num_attention_heads: int = 4           # Multi-head self-attention
    dropout: float = 0.15                  # Regularization dropout
    forecast_horizon: int = FORECAST_HORIZON_DAYS  # 14 days
    quantiles: List[float] = field(default_factory=lambda: [0.10, 0.50, 0.90])
    num_targets: int = 3                   # [TMAX, TMIN, PRCP]
    num_extreme_heads: int = 3             # [Heatwave, Frost, Deluge]


@dataclass
class TrainingConfig:
    """Training and optimization hyperparameters."""
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 25
    early_stopping_patience: int = 6
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    pinball_loss_weight: float = 1.0
    hazard_bce_loss_weight: float = 0.35
    grad_clip_norm: float = 1.0
    train_end_year: int = 2021
    val_year: int = 2022
    test_start_year: int = 2023
