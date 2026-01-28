# Anomaly Detection Module

Independent module for detecting and handling anomalies in industrial time series data.

## Features

- **Multiple Detection Methods**:
  - Z-score (standard and modified/robust)
  - IQR (Interquartile Range)
  - Rolling window statistics
  - Isolation Forest (ML-based)
  - Combined voting ensemble (recommended)

- **Interpolation Methods**:
  - Linear interpolation
  - Cubic spline interpolation
  - Rolling mean/median

- **Configurable**:
  - Sensor-specific presets
  - Conservative/aggressive modes
  - Customizable thresholds

## Usage

### Command Line

```bash
# Single file with default combined method
python -m anomaly.anomaly_detector --input processed/DESF/corriente_motor_a.parquet

# Batch processing for a machine
python -m anomaly.anomaly_detector --batch --machine DESF

# Custom threshold
python -m anomaly.anomaly_detector --input data.parquet --method modified_zscore --threshold 4.0

# Conservative mode (fewer false positives)
python -m anomaly.anomaly_detector --batch --machine DESF --conservative

# Aggressive mode (catch more anomalies)
python -m anomaly.anomaly_detector --batch --machine DESF --aggressive
```

### Python API

```python
from anomaly import AnomalyDetector, AnomalyConfig, AnomalyMethod

# Basic usage
detector = AnomalyDetector()
df_with_flags = detector.detect(df)
df_clean = detector.interpolate_anomalies(df_with_flags)

# Custom configuration
config = AnomalyConfig(
    method=AnomalyMethod.COMBINED,
    zscore_threshold=3.5,
    modified_zscore_threshold=4.0,
    min_anomaly_votes=2
)
detector = AnomalyDetector(config)

# Sensor-specific configuration
config = AnomalyConfig.sensor_specific("vibration")
detector = AnomalyDetector(config)

# Convenience functions
from anomaly import detect_anomalies, interpolate_anomalies

df_flagged, result = detect_anomalies(df, method="combined")
df_clean = interpolate_anomalies(df, method="combined")
```

## Detection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `zscore` | Standard z-score | Normally distributed data |
| `modified_zscore` | Uses MAD instead of std | Robust to outliers |
| `iqr` | Interquartile range | Boxplot-style detection |
| `rolling` | Rolling window stats | Non-stationary data |
| `isolation_forest` | ML-based isolation | Complex patterns |
| `combined` | Voting ensemble | General purpose (recommended) |

## Configuration Presets

- **Conservative**: High thresholds, more votes required. Minimizes false positives.
- **Aggressive**: Lower thresholds, fewer votes. Catches more anomalies.
- **Sensor-specific**: Optimized for temperature, vibration, current, acceleration.

## Output

- `*_anomaly_summary.json`: Detection statistics
- `*_anomalies.png`: Visualization
- `*_clean.parquet`: Cleaned data with interpolated values

## Integration with N-BEATS

The anomaly module is independent but integrates with N-BEATS forecasting:

```python
# In nbeats_forecast.py, anomalies are automatically detected and interpolated
# before generating forecasts, reducing their influence on predictions.
```
