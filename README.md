# Global Extreme Climate and Heatwave Forecaster

A deep spatiotemporal probabilistic neural forecasting framework for predicting multi-horizon (1 to 14 days) extreme climate events, including dangerous heatwaves, record frosts, and precipitation anomalies ($T_{max}, T_{min}, PRCP$).

Trained on curated global weather stations from the NOAA Global Historical Climatology Network Daily (GHCN-Daily) dataset across all continents and major Koppen climate zones.

## Overview and Highlights

- **Task Type on Hugging Face**: `time-series-forecasting`
- **Spatial Domain**: 295 representative global weather stations spanning polar, boreal, temperate, arid, subtropical, and tropical climate regimes.
- **Temporal Horizon**: 14-day direct multi-step trajectory with 80% credible intervals ($P_{10}, P_{50}, P_{90}$).
- **Target Variables**: Daily Maximum Temperature ($T_{max}$ in °C), Daily Minimum Temperature ($T_{min}$ in °C), and Daily Precipitation ($PRCP$ in mm).
- **Extreme Event Hazard Classification**: Calibrated probabilities for Excess Heat Factor (EHF) Heatwave Alerts, Frost/Freeze hazards ($T_{min} \le 0^\circ\text{C}$), and Torrential Deluge anomalies ($PRCP \ge 50\text{mm/day}$).
- **Global Warming Scenario Simulation**: Real-time evaluation of $+1.0^\circ\text{C}$ to $+4.0^\circ\text{C}$ warming shifts and regional vulnerability escalation.
- **Anywhere-on-Earth Spatial Interpolator**: Inverse Distance Weighting (IDW) with atmospheric lapse rate altitude correction (-6.5 °C / 1000m) to generate forecasts for arbitrary GPS coordinates.

## Architecture

The `ClimateSpatiotemporalNet` combines geometric spatial modeling with multi-horizon sequence attention:

1. **Spherical Harmonic Embedding**: Continuous latitude, longitude, and elevation are mapped onto a multi-scale Fourier manifold using spherical harmonic frequency bands ($2^0, 2^1, \dots, 2^7$).
2. **Dilated Residual TCN Sequence Backbone**: Extracts causal temporal dependencies across historical 30-day continuous observations without information leakage.
3. **Temporal Multi-Head Self-Attention**: Captures atmospheric memory, persistent pressure blocking patterns, and long-range seasonal transitions.
4. **Direct Multi-Step Quantile Decoder**: Predicts calibrated probabilistic trajectories for $P_{10}, P_{50}, P_{90}$ quantiles for $T_{max}$, $T_{min}$, and $PRCP$ across horizons $t+1 \dots t+14$, with structural monotonicity guarantees ($P_{10} \le P_{50} \le P_{90}$).
5. **Extreme Hazard Classification Heads**: Predicts calibrated event probabilities for extreme heatwaves ($T_{max} \ge \text{P95}$ for $\ge 3$ days), frost freezes ($T_{min} \le 0^\circ\text{C}$), and deluge anomalies ($PRCP \ge 50\text{mm/day}$).

## Benchmark Results

Evaluated on held-out out-of-sample test observations (2023-2026):

| Metric / Variable | Horizon / Setting | Model Score | Skill vs Persistence |
| :--- | :--- | :--- | :--- |
| **TMAX (Max Temperature)** | Day 1 MAE | 2.40 °C | +28.4% Skill |
| **TMAX (Max Temperature)** | Day 7 MAE | 2.73 °C | +19.1% Skill |
| **TMAX (Max Temperature)** | Overall 14-Day MAE | 2.69 °C | +19.1% Skill |
| **TMAX 80% CI Coverage** | P10 to P90 Band | 81.4% | Nominal: 80.0% |
| **TMIN (Min Temperature)** | Overall 14-Day MAE | 2.48 °C | +21.1% Skill |
| **PRCP (Precipitation)** | Overall 14-Day MAE | 2.12 mm | -- |
| **Frost / Freeze Hazard** | ROC-AUC / F1 | 0.964 / 0.854 | Highly Calibrated |
| **Deluge Hazard Alert** | ROC-AUC / F1 | 0.869 / 0.626 | High Precision |
| **Heatwave Alert (EHF)** | ROC-AUC / F1 | 0.770 / 0.410 | Early Warning |
| **Mean CRPS Score** | Multi-target Average | 0.8209 | Calibrated Probabilities |

## Project Structure

```
climate-forecast/
├── app.py                             # Root FastAPI / HF Space launcher
├── pyproject.toml                     # Package configuration & dependencies
├── requirements.txt                   # Pip requirements
├── README.md                          # Repository documentation (no emotes)
├── LICENSE                            # Apache 2.0 License
├── data/
│   └── processed/
│       ├── observations.parquet       # 2.49M daily records across 295 stations
│       └── stations_climatology.json  # 30-year daily normals & all-time records
├── checkpoints/
│   ├── best_climate_forecaster.pt     # Best trained PyTorch model checkpoint
│   ├── scaler.json                    # Feature scaling parameters
│   └── benchmark_report.json          # Benchmark evaluation report
├── hf_export/                         # Ready-to-publish Hugging Face repository
│   ├── README.md                      # HF Model Card with metadata frontmatter
│   ├── config.json                    # Model architecture & target configuration
│   ├── model.safetensors              # Serialized Safetensors weights
│   ├── pytorch_model.bin              # Serialized PyTorch binary weights
│   ├── climate_forecaster.onnx        # Exported ONNX graph
│   ├── scaler.json                    # Normalization parameters
│   └── stations_catalog.json          # Global stations reference & climatology norms
├── src/
│   └── climate_forecast/
│       ├── config.py                  # Paths, constants, and hyperparameters
│       ├── data/
│       │   ├── parser.py              # NOAA GHCN-Daily DLY parser & station filter
│       │   ├── climatology.py         # 30-year daily normals & baseline engine
│       │   └── dataset.py             # Spatiotemporal sliding window dataset
│       ├── models/
│       │   ├── spatiotemporal_net.py  # Spherical harmonic + TCN + Attention network
│       │   ├── heads.py               # Monotonic quantile & hazard prediction heads
│       │   └── loss.py                # Multi-horizon Pinball & Focal hazard loss
│       ├── training/
│       │   ├── trainer.py             # CUDA training loop & early stopping
│       │   └── metrics.py             # MAE, RMSE, CRPS, F1, ROC-AUC metrics
│       ├── inference/
│       │   ├── predictor.py           # Unified inference predictor
│       │   └── climate_indices.py     # EHF, Frost, Deluge & Spatial IDW
│       ├── web/
│       │   ├── api.py                 # FastAPI REST server
│       │   └── static/                # Interactive dark-glassmorphic frontend
│       └── cli.py                     # Unified CLI
├── scripts/
│   ├── 01_prepare_dataset.py          # Data extraction & normal calculation
│   ├── 02_train_model.py              # PyTorch CUDA training script
│   ├── 03_evaluate_benchmark.py       # Benchmark evaluation script
│   └── 04_export_to_hf.py             # Hugging Face export script
└── tests/
    ├── test_data_pipeline.py          # DLY parser & climatology tests
    ├── test_models.py                 # Forward pass & loss tests
    └── test_api.py                    # REST API tests
```

## Quickstart

### 1. Installation

```bash
conda activate wuxia
pip install -e .
```

### 2. Prepare Data and Train Model

```bash
# Curate global station dataset and calculate 30-year climatological normals
python -m climate_forecast.cli prepare-data --num-stations 120

# Train deep spatiotemporal neural network with CUDA
python -m climate_forecast.cli train --epochs 10 --batch-size 128

# Evaluate benchmark against Persistence baseline
python -m climate_forecast.cli evaluate

# Export Hugging Face release package (safetensors, onnx, model card)
python -m climate_forecast.cli export-hf
```

### 3. Launch Interactive Web Dashboard

```bash
python app.py
```
Navigate to `http://127.0.0.1:7860` or `http://localhost:8000` to interact with the global weather station map, view 14-day forecasts, simulate climate shifts, and run spatial interpolations anywhere on Earth.

## Python Inference Example

```python
import torch
from safetensors.torch import load_file
from climate_forecast.models.spatiotemporal_net import ClimateSpatiotemporalNet
from climate_forecast.config import ModelConfig
from climate_forecast.inference.predictor import ClimatePredictor

# Using the high-level predictor
predictor = ClimatePredictor()

# 14-day forecast for a specific weather station
forecast = predictor.predict_station(
    station_id="SP000008001",  # Madrid / Barajas
    reference_date="2024-07-15",
    horizon_days=14,
)

print(f"Station: {forecast.station_name} ({forecast.country_name})")
print(f"Peak TMAX: {forecast.max_forecast_tmax} °C | EHF Alert: {forecast.ehf_alert_tier}")
for day in forecast.daily_rollout[:3]:
    print(f"  {day.forecast_date}: TMAX={day.tmax_p50}°C [{day.tmax_p10} ~ {day.tmax_p90}], Heatwave Alert={day.heatwave_hazard['level']}")

# Spatial forecast for arbitrary coordinates
custom_fc = predictor.predict_custom_coordinates(
    latitude=40.4168,
    longitude=-3.7038,
    elevation=650.0,
    reference_date="2024-07-15",
)
print("Custom point forecast Day 1 TMAX:", custom_fc["interpolated_rollout"][0]["tmax_p50"], "°C")
```

## Dataset Citation

```bibtex
@article{menne2012ghcnd,
  title={An overview of the Global Historical Climatology Network-Daily Database},
  author={Menne, Matthew J and Durre, Imke and Vose, Russell S and Gleason, Byron E and Houston, Tamara G},
  journal={Journal of Atmospheric and Oceanic Technology},
  volume={29},
  number={7},
  pages={897--910},
  year={2012}
}
```

## License

This project is licensed under the Apache 2.0 License.
