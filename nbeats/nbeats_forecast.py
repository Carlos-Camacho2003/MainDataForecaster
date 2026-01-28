"""
N-BEATS Forecasting Script

Generates forecasts using trained N-BEATS model with uncertainty quantification
and ETA (Estimated Time to Alarm) calculation.

Supports multiple forecast horizons:
    - 2 days (48h)  - High accuracy (~85%)
    - 5 days (120h) - Medium accuracy (~72%)
    - 15 days (360h) - Low accuracy (~55%), trend indication
    - 1 month (720h) - Trend indication only (~42%)

Usage:
    # Standard 24h forecast
    python nbeats_forecast.py --model models/corriente_motor_a_nbeats.pth --data processed/corriente_motor_a.parquet
    
    # Extended horizon forecasts
    python nbeats_forecast.py --model ... --data ... --horizon 2_days
    python nbeats_forecast.py --model ... --data ... --horizon 5_days
    python nbeats_forecast.py --model ... --data ... --horizon 15_days
    python nbeats_forecast.py --model ... --data ... --horizon 1_month
    
    # With anomaly interpolation (recommended)
    python nbeats_forecast.py --model ... --data ... --clean-anomalies
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm

try:
    # Running as package
    from nbeats.nbeats_model import create_nbeats_model
    from nbeats.forecast_config import (
        ForecastHorizon, HorizonConfig, get_horizon_config, 
        get_comparison_table, HORIZON_CONFIGS
    )
except Exception:
    # Fallback for direct script execution
    from nbeats_model import create_nbeats_model
    from forecast_config import (
        ForecastHorizon, HorizonConfig, get_horizon_config,
        get_comparison_table, HORIZON_CONFIGS
    )

# Import anomaly detection module
try:
    from anomaly import AnomalyDetector, AnomalyConfig, AnomalyMethod
    HAS_ANOMALY = True
except ImportError:
    HAS_ANOMALY = False


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_eta(conf_hi: np.ndarray, L: float, freq: pd.Timedelta) -> Optional[int]:
    """
    Calculate ETA (Estimated Time to Alarm) - first hour where upper CI exceeds threshold.
    
    Args:
        conf_hi: Upper confidence interval values
        L: Threshold value (alarma or critico)
        freq: Time frequency between observations
    
    Returns:
        Hours until threshold crossing, or None if no crossing detected
    """
    if L is None or np.isnan(L):
        return None
    exceed_mask = conf_hi >= L
    if not exceed_mask.any():
        return None
    first_exceed_idx = exceed_mask.argmax()
    return int(first_exceed_idx * freq.total_seconds() / 3600)


def load_model_checkpoint(
    model_path: str,
    device: str = "cpu"
) -> tuple:
    """
    Load trained N-BEATS model from checkpoint.
    
    Returns:
        model, checkpoint_data
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create model with same architecture
    model = create_nbeats_model(
        input_size=checkpoint["lookback"],
        horizon=checkpoint["horizon"],
        model_size=checkpoint["model_size"]
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_with_uncertainty(
    model: torch.nn.Module,
    x_input: np.ndarray,
    device: str,
    n_samples: int = 100,
    dropout_rate: float = 0.1
) -> tuple:
    """
    Generate predictions with uncertainty using Monte Carlo dropout.
    
    Args:
        model: Trained N-BEATS model
        x_input: Input data (lookback,)
        device: Device to run on
        n_samples: Number of MC samples
        dropout_rate: Dropout rate for MC sampling
    
    Returns:
        mean, std, lower_ci, upper_ci
    """
    model.train()  # Enable dropout
    
    # Add dropout if not already present
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()
    
    model.apply(apply_dropout)
    
    # Generate multiple predictions
    predictions = []
    x_tensor = torch.FloatTensor(x_input).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(n_samples):
            pred = model(x_tensor)
            predictions.append(pred.cpu().numpy()[0])
    
    predictions = np.array(predictions)
    
    # Calculate statistics
    mean = np.mean(predictions, axis=0)
    std = np.std(predictions, axis=0)
    
    # 80% confidence interval
    z_score = norm.ppf(0.9)  # 80% CI
    lower_ci = mean - z_score * std
    upper_ci = mean + z_score * std
    
    return mean, std, lower_ci, upper_ci


def rolling_forecast(
    model: torch.nn.Module,
    initial_input: np.ndarray,
    device: str,
    total_steps: int,
    model_horizon: int,
    n_mc_samples: int = 100,
    horizon_config: Optional[HorizonConfig] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate extended forecasts using rolling predictions.
    
    Uses the trained model iteratively to predict beyond its native horizon.
    Each iteration predicts `model_horizon` steps ahead, then rolls the
    input window forward using the predictions.
    
    Args:
        model: Trained N-BEATS model
        initial_input: Normalized input data (lookback,)
        device: Device to run on
        total_steps: Total forecast steps needed
        model_horizon: Native model horizon (typically 24)
        n_mc_samples: MC samples for uncertainty
        horizon_config: Optional config for CI expansion
    
    Returns:
        Tuple of (mean, std, lower_ci, upper_ci, accuracy_by_step)
    """
    lookback = len(initial_input)
    current_input = initial_input.copy()
    
    all_means = []
    all_stds = []
    all_lower = []
    all_upper = []
    all_accuracy = []
    
    steps_predicted = 0
    iteration = 0
    
    while steps_predicted < total_steps:
        # Predict next chunk
        mean, std, lower_ci, upper_ci = predict_with_uncertainty(
            model, current_input, device, n_samples=n_mc_samples
        )
        
        # Determine how many steps to take from this prediction
        steps_to_take = min(model_horizon, total_steps - steps_predicted)
        
        # Apply CI expansion for longer horizons
        if horizon_config:
            for i in range(steps_to_take):
                step = steps_predicted + i + 1
                ci_mult = horizon_config.get_ci_multiplier(step)
                
                # Expand CI based on forecast distance
                expanded_std = std[i] * ci_mult
                z_score = norm.ppf(0.9)  # 80% CI base
                
                # For very long horizons, widen to 90% CI
                if step > 120:
                    z_score = norm.ppf(0.95)  # 90% CI
                
                all_means.append(mean[i])
                all_stds.append(expanded_std)
                all_lower.append(mean[i] - z_score * expanded_std)
                all_upper.append(mean[i] + z_score * expanded_std)
                all_accuracy.append(horizon_config.get_step_accuracy(step))
        else:
            # No config, use standard expansion
            for i in range(steps_to_take):
                step = steps_predicted + i + 1
                # Simple linear expansion
                ci_mult = 1.0 + (step / 100.0)  # 1% wider per hour
                expanded_std = std[i] * ci_mult
                z_score = norm.ppf(0.9)
                
                all_means.append(mean[i])
                all_stds.append(expanded_std)
                all_lower.append(mean[i] - z_score * expanded_std)
                all_upper.append(mean[i] + z_score * expanded_std)
                all_accuracy.append(max(30, 95 - step * 0.1))
        
        # Roll the input window forward
        # Append predictions and drop oldest values
        current_input = np.concatenate([
            current_input[steps_to_take:],
            mean[:steps_to_take]
        ])
        
        steps_predicted += steps_to_take
        iteration += 1
    
    return (
        np.array(all_means),
        np.array(all_stds),
        np.array(all_lower),
        np.array(all_upper),
        np.array(all_accuracy)
    )


def clean_data_anomalies(
    df: pd.DataFrame,
    column: str = "y",
    config: Optional[AnomalyConfig] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Detect and interpolate anomalies in the data before forecasting.
    
    Args:
        df: DataFrame with time series data
        column: Column containing values
        config: Optional anomaly detection config
    
    Returns:
        Tuple of (cleaned DataFrame, number of anomalies found)
    """
    if not HAS_ANOMALY:
        return df, 0
    
    if config is None:
        # Use conservative settings for forecasting - we want to catch
        # obvious spikes/errors but preserve genuine trends
        config = AnomalyConfig(
            method=AnomalyMethod.COMBINED,
            zscore_threshold=3.5,
            modified_zscore_threshold=4.0,
            iqr_multiplier=1.5,
            rolling_window=24,
            rolling_std_multiplier=3.5,
            min_anomaly_votes=2,
            interpolation_method=AnomalyConfig().interpolation_method,
            max_consecutive_anomalies=24  # Don't interpolate gaps > 24h
        )
    
    detector = AnomalyDetector(config)
    
    # Detect anomalies
    df_detected, result = detector.detect(df, column=column, return_details=True)
    
    if result.n_anomalies == 0:
        return df, 0
    
    # Interpolate anomalies
    df_clean = detector.interpolate_anomalies(df_detected, column=column)
    
    # Replace original column with cleaned version
    df_clean[column] = df_clean[f"{column}_clean"]
    
    return df_clean, result.n_anomalies


def forecast_nbeats(
    model_path: str,
    data_path: str,
    meta_path: Optional[str] = None,
    steps: int = 24,
    horizon_preset: Optional[str] = None,
    n_mc_samples: int = 100,
    outdir: str = "forecast_visuals",
    data_outdir: Optional[str] = None,
    device: Optional[str] = None,
    show_accuracy: bool = True,
    clean_anomalies: bool = True,
    anomaly_config: Optional[AnomalyConfig] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Generate forecasts using N-BEATS model.
    
    Supports both standard single-step forecasts and extended rolling forecasts
    for longer horizons (2 days, 5 days, 15 days, 1 month).
    
    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to preprocessed data
        meta_path: Path to metadata JSON (optional)
        steps: Forecast steps (used if horizon_preset is None)
        horizon_preset: Preset horizon ('2_days', '5_days', '15_days', '1_month')
        n_mc_samples: MC samples for uncertainty
        outdir: Output directory
        device: Device to use
        show_accuracy: Whether to show accuracy estimates in output
        clean_anomalies: Whether to detect and interpolate anomalies before forecasting
        anomaly_config: Optional custom anomaly detection configuration
        seed: Random seed for reproducibility (affects MC dropout sampling)
    
    Returns:
        Dictionary with forecast results and metadata
    """
    # Set random seed if specified
    if seed is not None:
        set_seed(seed)
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, checkpoint = load_model_checkpoint(model_path, device)
    
    lookback = checkpoint["lookback"]
    model_horizon = checkpoint["horizon"]
    y_mean = checkpoint["y_mean"]
    y_std = checkpoint["y_std"]
    
    print(f"  Model: Lookback={lookback}h, Native Horizon={model_horizon}h")
    
    # Determine forecast configuration
    horizon_config = None
    if horizon_preset:
        try:
            forecast_horizon = ForecastHorizon(horizon_preset)
            horizon_config = get_horizon_config(forecast_horizon)
            steps = horizon_config.horizon_hours
            n_mc_samples = horizon_config.mc_samples
            
            print(f"\n{'='*60}")
            print(f"FORECAST HORIZON: {horizon_config.horizon_label}")
            print(f"{'='*60}")
            print(f"  Steps: {steps} hours ({steps // 24} days)")
            print(f"  Expected Accuracy: {horizon_config.expected_accuracy_pct:.0f}%")
            print(f"  Reliability: {horizon_config.get_reliability_stars()} ({horizon_config.reliability_label})")
            
            if horizon_config.warnings:
                print(f"\n  ⚠️  WARNINGS:")
                for w in horizon_config.warnings:
                    print(f"      {w}")
            print(f"{'='*60}\n")
            
        except ValueError:
            print(f"Warning: Unknown horizon preset '{horizon_preset}'. Using steps={steps}.")
    
    # Check if we need rolling forecast
    use_rolling = steps > model_horizon
    
    if use_rolling:
        print(f"  Extended forecast: {steps}h requires rolling predictions")
        print(f"  Rolling iterations: {(steps + model_horizon - 1) // model_horizon}")
    
    # Load data (support both CSV and Parquet)
    print(f"Loading data from {data_path}...")
    data_path_obj = Path(data_path)
    if data_path_obj.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)
    
    # Preserve raw series for anchor comparison (before anomaly cleaning)
    raw_y = df["y"].to_numpy()
    
    # Detect and interpolate anomalies before forecasting
    n_anomalies_cleaned = 0
    if clean_anomalies and HAS_ANOMALY:
        df, n_anomalies_cleaned = clean_data_anomalies(df, column="y", config=anomaly_config)
        if n_anomalies_cleaned > 0:
            print(f"  Anomaly cleaning: {n_anomalies_cleaned} anomalies interpolated")
    elif clean_anomalies and not HAS_ANOMALY:
        print("  Warning: Anomaly module not available, skipping anomaly cleaning")
    
    y = df["y"].values
    timestamps = pd.to_datetime(df["timestamp"])
    
    # Get last lookback window
    if len(y) < lookback:
        raise ValueError(f"Insufficient data: need {lookback} points, got {len(y)}")
    
    x_input = y[-lookback:]
    
    # Normalize input (same as training)
    x_input_norm = (x_input - y_mean) / (y_std + 1e-8)
    x_input_norm = np.nan_to_num(x_input_norm, nan=0.0)
    
    # Generate forecast
    print(f"Generating forecast ({n_mc_samples} MC samples)...")
    
    if use_rolling:
        # Rolling forecast for extended horizons
        mean_norm, std_norm, lower_ci_norm, upper_ci_norm, accuracy_by_step = rolling_forecast(
            model=model,
            initial_input=x_input_norm,
            device=device,
            total_steps=steps,
            model_horizon=model_horizon,
            n_mc_samples=n_mc_samples,
            horizon_config=horizon_config
        )
    else:
        # Standard single-step forecast
        mean_norm, std_norm, lower_ci_norm, upper_ci_norm = predict_with_uncertainty(
            model, x_input_norm, device, n_samples=n_mc_samples
        )
        mean_norm = mean_norm[:steps]
        std_norm = std_norm[:steps]
        lower_ci_norm = lower_ci_norm[:steps]
        upper_ci_norm = upper_ci_norm[:steps]
        
        # Calculate accuracy for each step
        if horizon_config:
            accuracy_by_step = np.array([
                horizon_config.get_step_accuracy(i + 1) for i in range(steps)
            ])
        else:
            accuracy_by_step = np.array([max(30, 95 - i * 0.2) for i in range(steps)])
    
    # Denormalize
    mean = mean_norm * y_std + y_mean
    std = std_norm * y_std
    lower_ci = lower_ci_norm * y_std + y_mean
    upper_ci = upper_ci_norm * y_std + y_mean
    
    # Apply continuity adjustment to eliminate discontinuity
    # The first forecast value should be anchored to the last historical value
    # MODIFIED: Use average of last 4 days (96h) instead of single last point
    # to avoid anchoring to noise/outliers
    anchor_window = 96
    
    def compute_anchor(arr: np.ndarray) -> float:
        """Compute anchor using last `anchor_window` points (nan-safe)."""
        if len(arr) == 0:
            return float("nan")
        window = arr[-anchor_window:] if len(arr) >= anchor_window else arr
        return float(np.nanmean(window))
    
    # Anchor on the cleaned series used for forecasting
    last_historical_value = compute_anchor(y)
    raw_anchor_value = compute_anchor(raw_y)
    anchor_source = "cleaned" if (clean_anomalies and HAS_ANOMALY) else "raw"
    
    if len(y) >= anchor_window:
        print(f"  Using 4-day average anchor: {last_historical_value:.4f} (vs last point: {y[-1]:.4f})")
    else:
        print(f"  Using last point anchor: {last_historical_value:.4f} (insufficient data for 4-day avg)")
    
    # Surface differences when anomaly cleaning shifts the anchor
    if anchor_source == "cleaned" and not np.isclose(last_historical_value, raw_anchor_value, atol=1e-6):
        diff = last_historical_value - raw_anchor_value
        print(f"  Anchor uses CLEANED data: cleaned={last_historical_value:.4f}, raw={raw_anchor_value:.4f}, diff={diff:+.4f}")

    first_forecast_value = mean[0]
    continuity_offset = last_historical_value - first_forecast_value
    
    # Apply offset to mean and confidence intervals
    mean = mean + continuity_offset
    lower_ci = lower_ci + continuity_offset
    upper_ci = upper_ci + continuity_offset
    
    print(f"  Continuity adjustment: {continuity_offset:.4f} (anchor={last_historical_value:.4f}, first_pred={first_forecast_value:.4f})")
    
    # Load metadata
    meta = {}
    if meta_path and Path(meta_path).exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    
    L_A = meta.get("alarma")
    L_C = meta.get("critico")
    unidad = meta.get("unidad", "")
    
    # Calculate probabilities of exceeding thresholds
    def p_exceed(L):
        if L is None or np.isnan(L):
            return None
        z = (L - mean) / (std + 1e-8)
        return 1 - norm.cdf(z)
    
    pA = p_exceed(L_A)
    pC = p_exceed(L_C)
    
    # Create forecast dataframe
    # Forecast starts from the last timestamp in historical data
    freq = timestamps.diff().dropna().iloc[-1] if len(timestamps) >= 2 else pd.Timedelta("1h")
    idx_future = pd.date_range(start=timestamps.iloc[-1] + freq, periods=steps, freq=freq)
    
    out_df = pd.DataFrame({
        "timestamp": idx_future,
        "yhat": mean,
        "yhat_lo": lower_ci,
        "yhat_hi": upper_ci,
        "expected_accuracy_pct": accuracy_by_step,
        "forecast_day": [(i // 24) + 1 for i in range(steps)]
    })
    
    if pA is not None:
        out_df["p_exceed_alarma"] = pA
    if pC is not None:
        out_df["p_exceed_critico"] = pC
    
    # Determine output paths
    model_path_obj = Path(model_path)
    machine_type = 'UNKNOWN'
    if 'DESF' in str(model_path_obj).upper():
        machine_type = 'DESF'
    elif 'PICADORA' in str(model_path_obj).upper():
        machine_type = 'PICADORA'
    elif 'PLANT' in str(model_path_obj).upper():
        machine_type = 'PLANT'
    
    outdir_path = Path(outdir) / machine_type
    outdir_path.mkdir(parents=True, exist_ok=True)
    
    # Determine data output path
    if data_outdir:
        data_outdir_path = Path(data_outdir) / machine_type
        data_outdir_path.mkdir(parents=True, exist_ok=True)
    else:
        data_outdir_path = outdir_path
    
    target_name = model_path_obj.stem.replace("_nbeats", "")
    
    # Add horizon info to filename for extended forecasts
    horizon_suffix = f"_{horizon_preset}" if horizon_preset else ""
    
    out_csv = data_outdir_path / f"{target_name}_nbeats_forecast{horizon_suffix}.csv"
    out_df.to_csv(out_csv, index=False)
    
    # Calculate ETA
    eta_alarma = calculate_eta(upper_ci, L_A, freq)
    eta_critico = calculate_eta(upper_ci, L_C, freq)
    
    # Create summary
    max_p_alarma = float(np.nanmax(pA)) if pA is not None else None
    max_p_critico = float(np.nanmax(pC)) if pC is not None else None
    
    summary = {
        "forecast_horizon_hours": steps,
        "forecast_horizon_days": steps // 24,
        "horizon_preset": horizon_preset,
        "expected_accuracy_pct": float(np.mean(accuracy_by_step)),
        "accuracy_day_1": float(accuracy_by_step[min(23, len(accuracy_by_step)-1)]),
        "accuracy_final_day": float(accuracy_by_step[-1]),
        "reliability": horizon_config.reliability_label if horizon_config else "STANDARD",
        "max_p_exceed_alarma": max_p_alarma,
        "max_p_exceed_critico": max_p_critico,
        "eta_alarma_hours": eta_alarma,
        "eta_critico_hours": eta_critico,
        "trigger_amber": bool(max_p_alarma >= 0.20) if max_p_alarma is not None else False,
        "trigger_red": bool(max_p_critico >= 0.50) if max_p_critico is not None else False,
        "threshold_alarma": L_A,
        "threshold_critico": L_C,
        "unidad": unidad,
        "model_type": "N-BEATS",
        "mc_samples": n_mc_samples,
        "rolling_forecast": use_rolling,
        "anomalies_cleaned": int(n_anomalies_cleaned),
        "anchor_value": float(last_historical_value),
        "anchor_value_raw": float(raw_anchor_value),
        "anchor_window_hours": anchor_window,
        "anchor_source": anchor_source,
        "first_forecast_before_anchor": float(first_forecast_value),
        "warnings": horizon_config.warnings if horizon_config else []
    }
    
    out_summary = outdir_path / f"{target_name}_nbeats_summary{horizon_suffix}.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Create enhanced visualization
    create_extended_visualization(
        timestamps=timestamps,
        historical_y=y,
        forecast_timestamps=idx_future,
        mean=mean,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        accuracy_by_step=accuracy_by_step,
        L_A=L_A,
        L_C=L_C,
        unidad=unidad,
        target_name=target_name,
        horizon_config=horizon_config,
        output_path=outdir_path / f"{target_name}_nbeats_forecast{horizon_suffix}.png"
    )
    
    # Print results
    print_forecast_summary(
        target_name=target_name,
        summary=summary,
        horizon_config=horizon_config,
        out_csv=out_csv,
        out_png=outdir_path / f"{target_name}_nbeats_forecast{horizon_suffix}.png",
        out_summary=out_summary
    )
    
    return {
        "forecast_df": out_df,
        "summary": summary,
        "horizon_config": horizon_config.to_dict() if horizon_config else None
    }


def create_extended_visualization(
    timestamps: pd.DatetimeIndex,
    historical_y: np.ndarray,
    forecast_timestamps: pd.DatetimeIndex,
    mean: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    accuracy_by_step: np.ndarray,
    L_A: Optional[float],
    L_C: Optional[float],
    unidad: str,
    target_name: str,
    horizon_config: Optional[HorizonConfig],
    output_path: Path
):
    """Create enhanced visualization for extended forecasts."""
    
    # Determine figure size based on horizon
    n_days = len(mean) // 24
    fig_width = min(20, max(12, 8 + n_days))
    
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, 8), 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Top plot: Forecast
    ax1 = axes[0]
    
    # Plot historical data (last 200 points or less)
    hist_len = min(200, len(historical_y))
    ax1.plot(
        timestamps.iloc[-hist_len:],
        historical_y[-hist_len:],
        label="Historical",
        color="steelblue",
        linewidth=1.5
    )
    
    # Color-code forecast by accuracy
    # High accuracy (>80%): green
    # Medium accuracy (60-80%): orange
    # Low accuracy (<60%): red
    
    for i in range(len(mean) - 1):
        acc = accuracy_by_step[i]
        if acc >= 80:
            color = 'forestgreen'
        elif acc >= 60:
            color = 'darkorange'
        else:
            color = 'darkred'
        
        ax1.plot(
            forecast_timestamps[i:i+2],
            mean[i:i+2],
            color=color,
            linewidth=2
        )
    
    # Add confidence interval with gradient
    # Darker at start, lighter at end
    for i in range(0, len(mean), max(1, len(mean) // 10)):
        end_i = min(i + len(mean) // 10, len(mean))
        alpha = 0.3 * (1 - i / len(mean))  # Fade out
        ax1.fill_between(
            forecast_timestamps[i:end_i],
            lower_ci[i:end_i],
            upper_ci[i:end_i],
            alpha=max(0.1, alpha),
            color='gray'
        )
    
    # Plot thresholds
    if L_A is not None:
        ax1.axhline(L_A, linestyle="--", color="orange", 
                   label=f"Alarma ({L_A} {unidad})", alpha=0.7)
    if L_C is not None:
        ax1.axhline(L_C, linestyle=":", color="red", 
                   label=f"Crítico ({L_C} {unidad})", alpha=0.7)
    
    # Add day markers
    for day in range(1, n_days + 1):
        day_idx = day * 24 - 1
        if day_idx < len(forecast_timestamps):
            ax1.axvline(forecast_timestamps[day_idx], color='gray', 
                       linestyle=':', alpha=0.3)
            ax1.text(forecast_timestamps[day_idx], ax1.get_ylim()[1], 
                    f'Day {day}', ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # Title and labels
    horizon_label = horizon_config.horizon_label if horizon_config else f"{len(mean)}h"
    reliability = horizon_config.reliability_label if horizon_config else "STANDARD"
    
    ax1.set_title(f"{target_name} - N-BEATS Forecast ({horizon_label})\nReliability: {reliability}", 
                 fontsize=12, fontweight='bold')
    ax1.set_ylabel(f"Value ({unidad})" if unidad else "Value")
    ax1.grid(True, alpha=0.3)
    
    # Format x-axis for dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, n_days // 7)))
    
    # Bottom plot: Accuracy indicator
    ax2 = axes[1]
    
    # Create accuracy bars
    # Plot as a line with color gradient
    ax2.fill_between(forecast_timestamps, 0, accuracy_by_step, 
                    alpha=0.3, color='steelblue')
    ax2.plot(forecast_timestamps, accuracy_by_step, color='steelblue', linewidth=1.5)
    
    # Add threshold lines
    ax2.axhline(80, color='green', linestyle='--', alpha=0.5, label='High (80%)')
    ax2.axhline(60, color='orange', linestyle='--', alpha=0.5, label='Medium (60%)')
    ax2.axhline(40, color='red', linestyle='--', alpha=0.5, label='Low (40%)')
    
    ax2.set_ylabel("Expected\nAccuracy %", fontsize=9)
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Build combined legend so historical/threshold labels are preserved
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    base_handles, _ = ax1.get_legend_handles_labels()
    legend_elements = [
        Line2D([0, 1], [0, 1], color='gray', linewidth=2, label='Forecast (color-coded by accuracy)'),
        Patch(facecolor='forestgreen', label='High accuracy (>80%)'),
        Patch(facecolor='darkorange', label='Medium accuracy (60-80%)'),
        Patch(facecolor='darkred', label='Low accuracy (<60%)'),
        Patch(facecolor='gray', alpha=0.3, label='Confidence interval')
    ]
    ax1.legend(handles=base_handles + legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_forecast_summary(
    target_name: str,
    summary: Dict,
    horizon_config: Optional[HorizonConfig],
    out_csv: Path,
    out_png: Path,
    out_summary: Path
):
    """Print formatted forecast summary."""
    
    print(f"\n{'='*70}")
    print(f"N-BEATS FORECAST: {target_name.upper()}")
    print(f"{'='*70}")
    
    # Horizon info
    days = summary['forecast_horizon_days']
    hours = summary['forecast_horizon_hours']
    print(f"  Horizon:          {hours} hours ({days} days)")
    print(f"  Reliability:      {summary['reliability']}")
    print(f"  Avg Accuracy:     {summary['expected_accuracy_pct']:.1f}%")
    print(f"  Day 1 Accuracy:   {summary['accuracy_day_1']:.1f}%")
    print(f"  Final Day Accuracy: {summary['accuracy_final_day']:.1f}%")
    print(f"  Anomalies cleaned: {summary.get('anomalies_cleaned', 0)}")
    
    anchor_val = summary.get("anchor_value")
    if anchor_val is not None and not np.isnan(anchor_val):
        anchor_src = summary.get("anchor_source", "raw")
        anchor_hours = summary.get("anchor_window_hours", 96)
        print(f"  Anchor ({anchor_src}): {anchor_val:.2f} over last {anchor_hours}h")
        raw_anchor = summary.get("anchor_value_raw")
        if raw_anchor is not None and not np.isnan(raw_anchor) and not np.isclose(anchor_val, raw_anchor):
            diff = anchor_val - raw_anchor
            print(f"    Raw anchor: {raw_anchor:.2f} (diff {diff:+.2f})")
    
    # Files
    print(f"\n  Output Files:")
    print(f"     CSV:     {out_csv}")
    print(f"     Plot:    {out_png}")
    print(f"     Summary: {out_summary}")
    
    # Alert status
    print(f"\n{'='*70}")
    print(f"ALERT STATUS")
    print(f"{'='*70}")
    
    if summary["trigger_red"]:
        print("  [RED] ALERT: Critical threshold likely to be exceeded!")
        if summary["max_p_exceed_critico"]:
            print(f"     Max P(critico): {summary['max_p_exceed_critico']:.1%}")
        if summary["eta_critico_hours"]:
            print(f"     ETA to critical: {summary['eta_critico_hours']} hours")
    elif summary["trigger_amber"]:
        print("  [AMBER] ALERT: Alarm threshold likely to be exceeded")
        if summary["max_p_exceed_alarma"]:
            print(f"     Max P(alarma): {summary['max_p_exceed_alarma']:.1%}")
        if summary["eta_alarma_hours"]:
            print(f"     ETA to alarm: {summary['eta_alarma_hours']} hours")
    else:
        print("  [OK] Normal operation expected")
        if summary["max_p_exceed_alarma"]:
            print(f"     Max P(alarma): {summary['max_p_exceed_alarma']:.1%}")
    
    # Warnings
    if summary.get("warnings"):
        print(f"\n{'='*70}")
        print("IMPORTANT NOTES:")
        for w in summary["warnings"]:
            print(f"    - {w}")
    
    print(f"{'='*70}\n")


def batch_forecast_to_data(
    horizon_preset: str = "2_days",
    models_dir: str = "models",
    data_dir: str = "data",
    n_mc_samples: int = 100,
    device: Optional[str] = None,
    machine: Optional[str] = None,
    clean_anomalies: bool = True,
    seed: Optional[int] = None
) -> Dict[str, Path]:
    """
    Generate forecasts for all available models and save combined CSVs to data folder.
    
    This creates one CSV per machine (DESF, PICADORA) containing forecasts for all
    variables with trained models. Useful for downstream analysis and reporting.
    
    Args:
        horizon_preset: Forecast horizon (2_days, 5_days, 15_days, 1_month)
        models_dir: Directory containing trained models
        data_dir: Directory to save output CSVs
        n_mc_samples: Monte Carlo samples for uncertainty
        device: Device to run on (cpu/cuda)
        machine: Specific machine to forecast (DESF, PICADORA, PLANT, ALL, or None for all)
        clean_anomalies: Whether to detect and interpolate anomalies before forecasting
        seed: Random seed for reproducibility (affects MC dropout sampling)
    
    Returns:
        Dictionary mapping machine names to output file paths
    """
    # Set random seed if specified
    if seed is not None:
        set_seed(seed)
        print(f"Using seed: {seed}")
    
    models_path = Path(models_dir)
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Output base directory for combined forecast CSVs
    forecasts_output_base = data_path / "forecasts"
    forecasts_output_base.mkdir(parents=True, exist_ok=True)
    
    # Output base directory for individual visuals
    visuals_output_base = Path("forecast_visuals")
    visuals_output_base.mkdir(parents=True, exist_ok=True)
    
    # Create machine-specific subdirectories
    desf_output = forecasts_output_base / "DESF"
    picadora_output = forecasts_output_base / "PICADORA"
    plant_output = forecasts_output_base / "PLANT"
    desf_output.mkdir(parents=True, exist_ok=True)
    picadora_output.mkdir(parents=True, exist_ok=True)
    plant_output.mkdir(parents=True, exist_ok=True)
    
    # Create machine-specific subdirectories for visuals
    (visuals_output_base / "DESF").mkdir(parents=True, exist_ok=True)
    (visuals_output_base / "PICADORA").mkdir(parents=True, exist_ok=True)
    (visuals_output_base / "PLANT").mkdir(parents=True, exist_ok=True)
    
    # EPI data is stored in data/epi/ subfolder
    epi_path = data_path / "epi"
    
    # Machine configurations
    machines = {
        "DESF": {
            "data_file": epi_path / "DESF" / "DATOS_EPI_DESF_HOURLY.csv",
            "models_subdir": models_path / "DESF",
            "output_file": desf_output / f"FORECAST_DESF_{horizon_preset.upper()}.csv"
        },
        "PICADORA": {
            "data_file": epi_path / "PICADORA" / "DATOS_EPI_PICADORA_HOURLY.csv",
            "models_subdir": models_path / "PICADORA",
            "output_file": picadora_output / f"FORECAST_PICADORA_{horizon_preset.upper()}.csv"
        },
        "PLANT": {
            "data_file": epi_path / "PLANT" / "DATOS_EPI_PLANT_HOURLY.csv",
            "models_subdir": models_path / "PLANT",
            "output_file": plant_output / f"FORECAST_PLANT_{horizon_preset.upper()}.csv"
        }
    }
    
    # Filter machines if specified
    if machine and machine.upper() != "ALL":
        machine_upper = machine.upper()
        if machine_upper in machines:
            machines = {machine_upper: machines[machine_upper]}
        else:
            print(f"  Warning: Unknown machine: {machine}. Valid options: DESF, PICADORA, PLANT, ALL")
            return {}
    
    output_files = {}
    
    for machine_name, config in machines.items():
        print(f"\n{'='*70}")
        print(f"BATCH FORECAST: {machine_name}")
        print(f"{'='*70}")
        if clean_anomalies:
            print(f"  Anomaly cleaning: ENABLED")
        
        # Check if data file exists
        if not config["data_file"].exists():
            print(f"  Warning: Data file not found: {config['data_file']}")
            print(f"     Run 'python performance/run_performance_pipeline.py' first")
            continue
        
        # Find all N-BEATS models for this machine
        models_subdir = config["models_subdir"]
        if not models_subdir.exists():
            print(f"  ⚠️  Models directory not found: {models_subdir}")
            continue
        
        model_files = list(models_subdir.glob("*_nbeats.pth"))
        if not model_files:
            print(f"  ⚠️  No trained models found in {models_subdir}")
            continue
        
        print(f"  Found {len(model_files)} model(s): {[m.stem.replace('_nbeats', '') for m in model_files]}")
        
        # Load base data for timestamps
        df_base = pd.read_csv(config["data_file"])
        base_timestamps = pd.to_datetime(df_base["timestamp"])
        
        # Collect all forecasts
        all_forecasts = []
        
        # Define processed data directory for individual variable parquet files
        processed_dir = Path("processed") / machine_name
        
        for model_file in model_files:
            variable_name = model_file.stem.replace("_nbeats", "")
            print(f"\n  Forecasting: {variable_name}")
            
            # Determine the correct data path - prefer individual parquet files
            individual_data_path = processed_dir / f"{variable_name}.parquet"
            if individual_data_path.exists():
                data_path_to_use = str(individual_data_path)
            else:
                # Fallback to combined hourly CSV
                data_path_to_use = str(config["data_file"])
                print(f"    Warning: Using combined data (individual parquet not found)")
            
            try:
                # Pass the base visuals directory. 
                # forecast_nbeats will append the machine type (e.g. DESF) automatically.
                
                result = forecast_nbeats(
                    model_path=str(model_file),
                    data_path=data_path_to_use,
                    meta_path=None,
                    steps=24,  # Will be overridden by horizon_preset
                    horizon_preset=horizon_preset,
                    n_mc_samples=n_mc_samples,
                    outdir=str(visuals_output_base),  # Use visuals base dir
                    data_outdir=str(forecasts_output_base), # Use data base dir
                    device=device,
                    clean_anomalies=clean_anomalies,
                    seed=seed
                )
                
                # Extract forecast data
                forecast_df = result["forecast_df"]
                forecast_df = forecast_df.rename(columns={
                    "yhat": f"{variable_name}_forecast",
                    "yhat_lo": f"{variable_name}_lower_ci",
                    "yhat_hi": f"{variable_name}_upper_ci"
                })
                
                # Keep only relevant columns
                cols_to_keep = ["timestamp", f"{variable_name}_forecast", 
                               f"{variable_name}_lower_ci", f"{variable_name}_upper_ci"]
                forecast_df = forecast_df[cols_to_keep]
                
                all_forecasts.append(forecast_df)
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        if not all_forecasts:
            print(f"  ⚠️  No successful forecasts for {machine_name}")
            continue
        
        # Merge all forecasts on timestamp
        combined_df = all_forecasts[0]
        for df in all_forecasts[1:]:
            combined_df = combined_df.merge(df, on="timestamp", how="outer")
        
        # Sort by timestamp
        combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)
        
        # Save to data folder
        output_file = config["output_file"]
        
        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save CSV
        print(f"\n  💾 Guardando: {output_file}")
        print(f"     Directorio: {output_file.parent}")
        print(f"     Existe directorio: {output_file.parent.exists()}")
        
        try:
            combined_df.to_csv(output_file, index=False)
            
            # Verify file was written
            if output_file.exists():
                file_size = output_file.stat().st_size
                print(f"  ✅ Archivo guardado exitosamente")
                print(f"     Ruta: {output_file}")
                print(f"     Tamaño: {file_size / 1024:.1f} KB")
                print(f"     Shape: {combined_df.shape}")
                print(f"     Variables: {len(all_forecasts)}")
                print(f"     Horizon: {horizon_preset}")
                output_files[machine_name] = output_file
            else:
                print(f"  ❌ ERROR: Archivo no existe después de guardar!")
                print(f"     Ruta esperada: {output_file}")
        except PermissionError as e:
            print(f"  ❌ ERROR DE PERMISOS: {e}")
            print(f"     El archivo puede estar abierto en otro programa")
        except Exception as e:
            print(f"  ❌ ERROR al guardar: {e}")
            import traceback
            traceback.print_exc()

        # Cleanup: Delete individual variable forecast CSVs
        # Keep only:
        # 1. The combined summary: output_file (FORECAST_{MACHINE}_{HORIZON}.csv)
        # 2. The EPI forecast: DATOS_EPI_{MACHINE}_...
        print(f"\n  🧹 Limpiando archivos individuales...")
        
        # Safety check: only cleanup if output_file exists
        if output_file not in output_files.values():
            print(f"    ⚠️ SALTANDO limpieza - archivo principal no guardado")
        elif not output_file.exists():
            print(f"    ⚠️ SALTANDO limpieza - archivo principal no encontrado: {output_file}")
        else:
            count_deleted = 0
            for f in forecasts_output_base.glob(f"{machine_name}/*.csv"):
                # CRITICAL: Skip the combined output file
                if f.resolve() == output_file.resolve():
                    print(f"    ✓ Conservando: {f.name} (archivo principal)")
                    continue
                
                # Skip FORECAST_ files (should be kept)
                if f.name.startswith("FORECAST_"):
                    print(f"    ✓ Conservando: {f.name} (archivo FORECAST)")
                    continue
                
                # Skip EPI forecast files
                if "DATOS_EPI" in f.name:
                    print(f"    ✓ Conservando: {f.name} (archivo EPI)")
                    continue
                
                try:
                    print(f"    🗑 Eliminando: {f.name}")
                    f.unlink()
                    count_deleted += 1
                except Exception as e:
                    print(f"    ❌ Error eliminando {f.name}: {e}")
            
            print(f"  ✅ Limpieza completa: {count_deleted} archivos eliminados.")
    
    # -------------------------------------------------------------------------
    # Global Cleanup: Ensure all machine folders are clean, even if skipped
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("FINAL CLEANUP CHECK")
    print(f"{'='*70}")
    
    for m_name in ["DESF", "PICADORA", "PLANT"]:
        m_dir = forecasts_output_base / m_name
        if not m_dir.exists():
            continue
            
        print(f"  Scanning {m_name}...")
        deleted_count = 0
        for f in m_dir.glob("*.csv"):
            # Preserve SUMMARIES (FORECAST_*.csv)
            if f.name.startswith("FORECAST_"):
                continue
            
            # Preserve EPI (DATOS_EPI_*.csv)
            if f.name.startswith("DATOS_EPI"):
                continue
                
            # If we just generated this file (it's in output_files), keep it (redundant check but safe)
            if m_name in output_files and f.resolve() == output_files[m_name].resolve():
                continue

            try:
                f.unlink()
                deleted_count += 1
            except Exception as e:
                pass
        
        if deleted_count > 0:
            print(f"    Cleaned {deleted_count} stale files.")
        else:
            print(f"    Clean.")

    # Summary
    print(f"\n{'='*70}")
    print("BATCH FORECAST COMPLETE")
    print(f"{'='*70}")
    print(f"  Output files saved to: {forecasts_output_base}/")
    for machine, path in output_files.items():
        print(f"    • {machine}: {path.name}")
    print(f"{'='*70}\n")
    
    return output_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate N-BEATS forecasts with multiple horizon options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Horizon Options:
  2_days   - 48 hours,  ~85% accuracy, HIGH reliability
  5_days   - 120 hours, ~72% accuracy, MEDIUM reliability  
  15_days  - 360 hours, ~55% accuracy, LOW reliability (trend indication)
  1_month  - 720 hours, ~42% accuracy, TREND ONLY

Examples:
  # Standard 24-hour forecast
  python nbeats_forecast.py --model models/DESF/corriente_motor_a_nbeats.pth \\
      --data processed/DESF/corriente_motor_a.parquet
  
  # 5-day forecast
  python nbeats_forecast.py --model models/DESF/corriente_motor_a_nbeats.pth \\
      --data processed/DESF/corriente_motor_a.parquet --horizon 5_days
  
  # Batch forecast all models to data folder
  python nbeats_forecast.py --batch --horizon 2_days
  
  # Without anomaly cleaning
  python nbeats_forecast.py --batch --horizon 5_days --no-clean-anomalies

Note: Accuracy improves as more training data becomes available.
        """
    )
    parser.add_argument("--model", help="Path to trained model (.pth)")
    parser.add_argument("--data", help="Path to preprocessed data")
    parser.add_argument("--meta", help="Path to metadata JSON")
    parser.add_argument(
        "--horizon",
        choices=["2_days", "5_days", "7_days", "15_days", "1_month", "2_months", "3_months"],
        default="2_days",
        help="Forecast horizon preset (default: 2_days)"
    )
    parser.add_argument("--steps", type=int, default=24, 
                       help="Custom forecast steps (if --horizon not specified)")
    parser.add_argument("--mc-samples", type=int, default=100, 
                       help="MC samples for uncertainty")
    parser.add_argument("--outdir", default="forecast_visuals", help="Output directory for individual forecast visualizations")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--list-horizons", action="store_true",
                       help="Show available horizon options and exit")
    parser.add_argument("--batch", action="store_true",
                       help="Batch forecast all models and save CSVs to data folder")
    parser.add_argument("--machine", choices=["DESF", "PICADORA", "PLANT", "ALL"],
                       default="ALL",
                       help="Machine to forecast (for --batch mode, default: ALL)")
    parser.add_argument("--models-dir", default="models",
                       help="Directory containing trained models (for --batch)")
    parser.add_argument("--data-dir", default="data",
                       help="Directory for input/output data (for --batch)")
    parser.add_argument("--no-clean-anomalies", action="store_true",
                       help="Disable anomaly detection and interpolation before forecasting")
    
    args = parser.parse_args()
    
    # Show horizon options if requested
    if args.list_horizons:
        print("\n" + "=" * 75)
        print("AVAILABLE FORECAST HORIZONS")
        print("=" * 75)
        print(get_comparison_table())
        print("\nUse --horizon <option> to select a preset.")
        print("Example: python nbeats_forecast.py --model ... --data ... --horizon 5_days")
        print("\nAccuracy by Day (example for each horizon):")
        for horizon in ForecastHorizon:
            config = get_horizon_config(horizon)
            print(f"\n  {config.horizon_label}:")
            acc_by_day = config.get_accuracy_by_day()
            for day, acc in list(acc_by_day.items())[:min(5, len(acc_by_day))]:
                bar = "=" * int(acc / 5) + "-" * (20 - int(acc / 5))
                print(f"    Day {day:>2}: {acc:>5.1f}% {bar}")
            if len(acc_by_day) > 5:
                print(f"    ... (total {len(acc_by_day)} days)")
        print("\n" + "=" * 75 + "\n")
        return
    
    # Determine if anomaly cleaning is enabled
    clean_anomalies = not args.no_clean_anomalies
    
    # Batch mode: forecast all models and save to data folder
    if args.batch:
        batch_forecast_to_data(
            horizon_preset=args.horizon,
            models_dir=args.models_dir,
            data_dir=args.data_dir,
            n_mc_samples=args.mc_samples,
            device=args.device,
            machine=args.machine,
            clean_anomalies=clean_anomalies
        )
        return
    
    # Single model mode: validate required arguments
    if not args.model or not args.data:
        parser.error("--model and --data are required for single-model forecasting (or use --batch)")
    
    forecast_nbeats(
        model_path=args.model,
        data_path=args.data,
        meta_path=args.meta,
        steps=args.steps,
        horizon_preset=args.horizon,
        n_mc_samples=args.mc_samples,
        outdir=args.outdir,
        device=args.device,
        clean_anomalies=clean_anomalies
    )


if __name__ == "__main__":
    main()
