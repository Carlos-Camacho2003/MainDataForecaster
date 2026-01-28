# N-BEATS Models

This folder contains N-BEATS-specific model files, scripts, and outputs.

## Features

- **Monte Carlo Dropout** for uncertainty quantification
- **Extended horizon forecasting** (2 days, 5 days, 15 days, 1 month)
- **Anomaly detection integration** - cleans data before training AND forecasting
- **Continuity adjustment** - ensures forecast starts at last known value

## Anomaly Integration

Both training and forecasting automatically detect and interpolate anomalies:

```powershell
# Training with anomaly cleaning (default, recommended)
python -m nbeats.nbeats_train --data processed/DESF/variable.parquet

# Training without anomaly cleaning
python -m nbeats.nbeats_train --data processed/DESF/variable.parquet --no-clean-anomalies

# Forecasting with anomaly cleaning (default)
python -m nbeats.nbeats_forecast --model models/DESF/variable_nbeats.pth --data processed/DESF/variable.parquet

# Forecasting without anomaly cleaning  
python -m nbeats.nbeats_forecast --model ... --data ... --no-clean-anomalies
```

Model checkpoints include metadata about anomaly cleaning:
- `trained_on_clean_data`: Whether model was trained on cleaned data
- `anomalies_cleaned`: Number of anomalies interpolated during training

## Organization

```
nbeats/
├── __init__.py
├── nbeats_model.py      # Model architecture
├── nbeats_train.py      # Training with anomaly cleaning
├── nbeats_forecast.py   # Forecasting with uncertainty
├── nbeats_visualize.py  # Visualization tools
├── forecast_config.py   # Horizon configurations
└── README.md
```

## Model Artifacts

N-BEATS models include:
- `.pth` - PyTorch model checkpoint (with anomaly metadata)
- `_history.json` - Training loss curves
- `_forecast.csv` - Forecast outputs with uncertainty quantification
- `_summary.json` - Forecast metadata, alerts, and anomalies cleaned count

## Forecast Horizons

| Horizon | Hours | Expected Accuracy | Reliability |
|---------|-------|-------------------|-------------|
| 2 days  | 48    | ~85%              | HIGH        |
| 5 days  | 120   | ~72%              | MEDIUM      |
| 15 days | 360   | ~55%              | LOW         |
| 1 month | 720   | ~42%              | TREND ONLY  |

## Hardware Requirements

- **CPU Training:** ~2 min/epoch (medium model)
- **GPU Training:** ~10 sec/epoch (medium model)
- **Memory:** 4GB RAM minimum, 8GB recommended
- **GPU:** CUDA-compatible GPU optional but recommended

## Model Sizes

| Size   | Parameters | Training Time | Best For              |
|--------|------------|---------------|------------------------|
| Small  | ~100K      | Fast          | Quick experiments      |
| Medium | ~1.5M      | Balanced      | Production (default)   |
| Large  | ~2M        | Slow          | Maximum accuracy       |

## Usage

```powershell
# Train single variable
python -m nbeats.nbeats_train --data processed/DESF/corriente_motor_a.parquet

# Batch train all variables for a machine
python -m nbeats.nbeats_train --batch --machine DESF

# Generate 5-day forecast with uncertainty
python -m nbeats.nbeats_forecast --model models/DESF/corriente_motor_a_nbeats.pth --data processed/DESF/corriente_motor_a.parquet --horizon 5_days

# Batch forecast all variables
python -m nbeats.nbeats_forecast --batch --machine DESF --horizon 2_days
```

## PyTorch Notes

N-BEATS requires PyTorch. Install with:
```powershell
# CPU only
pip install torch

# GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Check GPU availability:
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
```
