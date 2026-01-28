"""
Anomaly Detection Module

Provides comprehensive anomaly detection for industrial time series data,
with multiple detection methods and interpolation strategies.

This module is independent of N-BEATS and can be used standalone for
data quality assessment and cleaning.

Usage:
    # Standalone usage
    python -m anomaly.anomaly_detector --input processed/DESF/corriente_motor_a.parquet
    
    # Batch processing
    python -m anomaly.anomaly_detector --batch --machine DESF
    
    # With specific method
    python -m anomaly.anomaly_detector --input data.parquet --method combined --threshold 3.0
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from scipy.interpolate import CubicSpline

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from anomaly.anomaly_config import AnomalyConfig, AnomalyMethod, InterpolationMethod
except ImportError:
    from anomaly_config import AnomalyConfig, AnomalyMethod, InterpolationMethod


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    n_anomalies: int
    anomaly_pct: float
    anomaly_indices: np.ndarray
    anomaly_mask: np.ndarray
    scores: Optional[np.ndarray] = None
    method_votes: Optional[Dict[str, np.ndarray]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_anomalies": int(self.n_anomalies),
            "anomaly_pct": float(round(self.anomaly_pct, 4)),
            "n_total": int(len(self.anomaly_mask)),
            "anomaly_indices": [int(x) for x in self.anomaly_indices.tolist()] if len(self.anomaly_indices) < 1000 else "too_many"
        }


class AnomalyDetector:
    """
    Multi-method anomaly detector for time series data.
    
    Supports multiple detection algorithms that can be used individually
    or combined via voting for robust anomaly detection.
    
    Attributes:
        config: AnomalyConfig with detection parameters
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """
        Initialize the anomaly detector.
        
        Args:
            config: Configuration object. If None, uses default config.
        """
        self.config = config or AnomalyConfig()
    
    def detect(
        self,
        df: pd.DataFrame,
        column: str = "y",
        return_details: bool = False
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, AnomalyResult]]:
        """
        Detect anomalies in the dataframe.
        
        Args:
            df: DataFrame with time series data
            column: Column name containing values to check
            return_details: Whether to return detailed results
        
        Returns:
            DataFrame with 'is_anomaly' column added
            If return_details=True, also returns AnomalyResult
        """
        df = df.copy()
        y = df[column].values
        
        if self.config.method == AnomalyMethod.COMBINED:
            mask, votes = self._detect_combined(y)
            result = AnomalyResult(
                n_anomalies=mask.sum(),
                anomaly_pct=mask.mean() * 100,
                anomaly_indices=np.where(mask)[0],
                anomaly_mask=mask,
                method_votes=votes
            )
        else:
            mask = self._detect_single_method(y, self.config.method)
            result = AnomalyResult(
                n_anomalies=mask.sum(),
                anomaly_pct=mask.mean() * 100,
                anomaly_indices=np.where(mask)[0],
                anomaly_mask=mask
            )
        
        df["is_anomaly"] = mask
        
        if return_details:
            return df, result
        return df
    
    def _detect_single_method(self, y: np.ndarray, method: AnomalyMethod) -> np.ndarray:
        """Apply a single detection method."""
        if method == AnomalyMethod.ZSCORE:
            return self._detect_zscore(y)
        elif method == AnomalyMethod.MODIFIED_ZSCORE:
            return self._detect_modified_zscore(y)
        elif method == AnomalyMethod.IQR:
            return self._detect_iqr(y)
        elif method == AnomalyMethod.ROLLING:
            return self._detect_rolling(y)
        elif method == AnomalyMethod.ISOLATION_FOREST:
            return self._detect_isolation_forest(y)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _detect_zscore(self, y: np.ndarray) -> np.ndarray:
        """
        Standard Z-score based anomaly detection.
        
        Flags points where |z-score| > threshold.
        Simple but sensitive to outliers in mean/std calculation.
        """
        y_clean = np.nan_to_num(y, nan=np.nanmean(y))
        mean = np.mean(y_clean)
        std = np.std(y_clean)
        
        if std < 1e-10:
            return np.zeros(len(y), dtype=bool)
        
        z_scores = np.abs((y_clean - mean) / std)
        return z_scores > self.config.zscore_threshold
    
    def _detect_modified_zscore(self, y: np.ndarray) -> np.ndarray:
        """
        Modified Z-score using median and MAD (robust to outliers).
        
        Uses Median Absolute Deviation instead of standard deviation
        for more robust outlier detection.
        """
        y_clean = np.nan_to_num(y, nan=np.nanmedian(y))
        median = np.median(y_clean)
        mad = np.median(np.abs(y_clean - median))
        
        if mad < 1e-10:
            # Use standard deviation if MAD is zero
            mad = np.std(y_clean) * 0.6745
        
        if mad < 1e-10:
            return np.zeros(len(y), dtype=bool)
        
        # Modified z-score formula
        modified_z = 0.6745 * (y_clean - median) / mad
        return np.abs(modified_z) > self.config.modified_zscore_threshold
    
    def _detect_iqr(self, y: np.ndarray) -> np.ndarray:
        """
        Interquartile Range (IQR) based detection.
        
        Classic boxplot method: outliers are outside [Q1 - k*IQR, Q3 + k*IQR]
        """
        y_clean = np.nan_to_num(y, nan=np.nanmedian(y))
        q1 = np.percentile(y_clean, 25)
        q3 = np.percentile(y_clean, 75)
        iqr = q3 - q1
        
        if iqr < 1e-10:
            return np.zeros(len(y), dtype=bool)
        
        k = self.config.iqr_multiplier
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        
        return (y_clean < lower) | (y_clean > upper)
    
    def _detect_rolling(self, y: np.ndarray) -> np.ndarray:
        """
        Rolling window statistics for local anomaly detection.
        
        Computes rolling mean and std, flags points outside
        rolling_mean +/- k * rolling_std.
        Good for detecting local anomalies in non-stationary data.
        """
        window = self.config.rolling_window
        k = self.config.rolling_std_multiplier
        
        # Use pandas for efficient rolling computation
        s = pd.Series(y)
        rolling_mean = s.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = s.rolling(window=window, center=True, min_periods=1).std()
        
        # Handle zero std
        rolling_std = rolling_std.replace(0, np.nan).fillna(s.std())
        
        lower = rolling_mean - k * rolling_std
        upper = rolling_mean + k * rolling_std
        
        return ((s < lower) | (s > upper)).values
    
    def _detect_isolation_forest(self, y: np.ndarray) -> np.ndarray:
        """
        Isolation Forest based anomaly detection.
        
        Uses decision trees to isolate anomalies. Points that are
        easier to isolate (shorter path in tree) are more anomalous.
        """
        if not HAS_SKLEARN:
            print("Warning: sklearn not available, falling back to modified z-score")
            return self._detect_modified_zscore(y)
        
        y_clean = np.nan_to_num(y, nan=np.nanmedian(y))
        
        # Reshape for sklearn
        X = y_clean.reshape(-1, 1)
        
        # Create time-based features for better detection
        n = len(y)
        time_feature = np.arange(n).reshape(-1, 1) / n
        
        # Add rolling statistics as features
        s = pd.Series(y_clean)
        rolling_mean = s.rolling(window=12, min_periods=1).mean().values.reshape(-1, 1)
        rolling_std = s.rolling(window=12, min_periods=1).std().fillna(0).values.reshape(-1, 1)
        diff = np.diff(y_clean, prepend=y_clean[0]).reshape(-1, 1)
        
        # Combine features
        features = np.hstack([X, rolling_mean, rolling_std, diff])
        
        # Fit Isolation Forest
        iso = IsolationForest(
            n_estimators=self.config.isolation_n_estimators,
            contamination=self.config.isolation_contamination,
            random_state=self.config.isolation_random_state,
            n_jobs=-1
        )
        
        predictions = iso.fit_predict(features)
        return predictions == -1
    
    def _detect_combined(self, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Combined detection using voting from multiple methods.
        
        Each method votes on whether a point is an anomaly.
        Points with >= min_anomaly_votes are flagged.
        """
        votes = {}
        
        for method in self.config.methods_to_combine:
            mask = self._detect_single_method(y, method)
            votes[method.value] = mask
        
        # Count votes
        vote_array = np.stack(list(votes.values()), axis=0)
        vote_count = vote_array.sum(axis=0)
        
        # Flag anomalies with enough votes
        final_mask = vote_count >= self.config.min_anomaly_votes
        
        return final_mask, votes
    
    def interpolate_anomalies(
        self,
        df: pd.DataFrame,
        column: str = "y",
        anomaly_column: str = "is_anomaly"
    ) -> pd.DataFrame:
        """
        Interpolate anomalous values to create clean data.
        
        Args:
            df: DataFrame with anomaly flags
            column: Column to interpolate
            anomaly_column: Column with anomaly flags
        
        Returns:
            DataFrame with interpolated values in new column '{column}_clean'
        """
        df = df.copy()
        y = df[column].values.copy()
        mask = df[anomaly_column].values if anomaly_column in df.columns else np.zeros(len(y), dtype=bool)
        
        # Check for consecutive anomalies exceeding max threshold
        # Split into groups and only interpolate small gaps
        y_clean = self._interpolate_values(y, mask)
        
        df[f"{column}_clean"] = y_clean
        df["was_interpolated"] = mask
        
        return df
    
    def _interpolate_values(self, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Perform the actual interpolation.
        
        Handles edge cases and applies the configured interpolation method.
        """
        y_clean = y.copy()
        
        if not mask.any():
            return y_clean
        
        # Find consecutive anomaly groups
        groups = self._find_anomaly_groups(mask)
        
        for start, end in groups:
            length = end - start
            
            # Skip if gap is too large
            if length > self.config.max_consecutive_anomalies:
                continue
            
            # Get surrounding valid values
            left_idx = start - 1 if start > 0 else None
            right_idx = end if end < len(y) else None
            
            if left_idx is None and right_idx is None:
                continue
            elif left_idx is None:
                # Fill with right value
                y_clean[start:end] = y[right_idx]
            elif right_idx is None:
                # Fill with left value
                y_clean[start:end] = y[left_idx]
            else:
                # Interpolate between left and right
                y_clean[start:end] = self._interpolate_gap(
                    y, start, end, left_idx, right_idx
                )
        
        return y_clean
    
    def _interpolate_gap(
        self,
        y: np.ndarray,
        start: int,
        end: int,
        left_idx: int,
        right_idx: int
    ) -> np.ndarray:
        """Interpolate a single gap using configured method."""
        method = self.config.interpolation_method
        gap_size = end - start
        
        if method == InterpolationMethod.LINEAR:
            return np.linspace(y[left_idx], y[right_idx], gap_size + 2)[1:-1]
        
        elif method == InterpolationMethod.CUBIC:
            # Use more points for cubic interpolation
            window = min(self.config.interpolation_window, left_idx, len(y) - right_idx - 1)
            if window < 2:
                return np.linspace(y[left_idx], y[right_idx], gap_size + 2)[1:-1]
            
            # Get points around the gap
            left_points = max(0, left_idx - window)
            right_points = min(len(y), right_idx + window + 1)
            
            x_known = np.concatenate([
                np.arange(left_points, start),
                np.arange(end, right_points)
            ])
            y_known = np.concatenate([
                y[left_points:start],
                y[end:right_points]
            ])
            
            if len(x_known) < 4:
                return np.linspace(y[left_idx], y[right_idx], gap_size + 2)[1:-1]
            
            try:
                cs = CubicSpline(x_known, y_known)
                x_interp = np.arange(start, end)
                return cs(x_interp)
            except Exception:
                return np.linspace(y[left_idx], y[right_idx], gap_size + 2)[1:-1]
        
        elif method == InterpolationMethod.ROLLING_MEAN:
            window = self.config.interpolation_window
            # Use rolling mean of surrounding values
            left_start = max(0, start - window)
            right_end = min(len(y), end + window)
            surrounding = np.concatenate([y[left_start:start], y[end:right_end]])
            fill_value = np.mean(surrounding) if len(surrounding) > 0 else y[left_idx]
            return np.full(gap_size, fill_value)
        
        elif method == InterpolationMethod.ROLLING_MEDIAN:
            window = self.config.interpolation_window
            left_start = max(0, start - window)
            right_end = min(len(y), end + window)
            surrounding = np.concatenate([y[left_start:start], y[end:right_end]])
            fill_value = np.median(surrounding) if len(surrounding) > 0 else y[left_idx]
            return np.full(gap_size, fill_value)
        
        else:
            return np.linspace(y[left_idx], y[right_idx], gap_size + 2)[1:-1]
    
    def _find_anomaly_groups(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find consecutive groups of anomalies."""
        groups = []
        in_group = False
        start = 0
        
        for i, is_anomaly in enumerate(mask):
            if is_anomaly and not in_group:
                start = i
                in_group = True
            elif not is_anomaly and in_group:
                groups.append((start, i))
                in_group = False
        
        if in_group:
            groups.append((start, len(mask)))
        
        return groups
    
    def plot_anomalies(
        self,
        df: pd.DataFrame,
        column: str = "y",
        timestamp_column: str = "timestamp",
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create visualization of detected anomalies.
        
        Args:
            df: DataFrame with anomaly flags
            column: Value column
            timestamp_column: Timestamp column
            save_path: Path to save the plot
            title: Plot title
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Ensure timestamp is datetime
        if timestamp_column in df.columns:
            x = pd.to_datetime(df[timestamp_column])
        else:
            x = np.arange(len(df))
        
        y = df[column].values
        mask = df["is_anomaly"].values if "is_anomaly" in df.columns else np.zeros(len(y), dtype=bool)
        
        # Top plot: Original data with anomalies highlighted
        ax1 = axes[0]
        ax1.plot(x, y, label="Original", color="steelblue", linewidth=0.8, alpha=0.8)
        
        if mask.any():
            ax1.scatter(
                x[mask] if isinstance(x, pd.DatetimeIndex) else np.array(x)[mask],
                y[mask],
                color="red",
                s=30,
                label=f"Anomalies ({mask.sum()})",
                zorder=5,
                alpha=0.7
            )
        
        # Add clean data if available
        clean_col = f"{column}_clean"
        if clean_col in df.columns:
            ax1.plot(x, df[clean_col].values, label="Interpolated", 
                    color="forestgreen", linewidth=1, alpha=0.8, linestyle="--")
        
        ax1.set_ylabel(f"{column}")
        ax1.legend(loc="upper right")
        ax1.set_title(title or f"Anomaly Detection: {column}")
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Anomaly indicator
        ax2 = axes[1]
        ax2.fill_between(x, 0, mask.astype(int), alpha=0.5, color="red", step="mid")
        ax2.set_ylabel("Anomaly")
        ax2.set_xlabel("Time")
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Normal", "Anomaly"])
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis for dates
        if isinstance(x, pd.DatetimeIndex):
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved anomaly plot to: {save_path}")
        
        return fig


def detect_anomalies(
    df: pd.DataFrame,
    column: str = "y",
    method: str = "combined",
    config: Optional[AnomalyConfig] = None,
    **kwargs
) -> Tuple[pd.DataFrame, AnomalyResult]:
    """
    Convenience function to detect anomalies.
    
    Args:
        df: Input DataFrame
        column: Column to analyze
        method: Detection method name
        config: Optional config object
        **kwargs: Additional config parameters
    
    Returns:
        Tuple of (DataFrame with flags, AnomalyResult)
    """
    if config is None:
        config = AnomalyConfig(method=AnomalyMethod(method))
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    detector = AnomalyDetector(config)
    return detector.detect(df, column=column, return_details=True)


def interpolate_anomalies(
    df: pd.DataFrame,
    column: str = "y",
    method: str = "combined",
    config: Optional[AnomalyConfig] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to detect and interpolate anomalies.
    
    Args:
        df: Input DataFrame
        column: Column to clean
        method: Detection method
        config: Optional config
        **kwargs: Additional parameters
    
    Returns:
        DataFrame with clean column added
    """
    if config is None:
        config = AnomalyConfig(method=AnomalyMethod(method))
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    detector = AnomalyDetector(config)
    df_detected = detector.detect(df, column=column)
    df_clean = detector.interpolate_anomalies(df_detected, column=column)
    
    return df_clean


def run_anomaly_detection(
    input_path: str,
    output_dir: str = "anomaly_results",
    method: str = "combined",
    column: str = "y",
    save_plot: bool = True,
    save_clean: bool = True,
    config: Optional[AnomalyConfig] = None
) -> Dict[str, Any]:
    """
    Run anomaly detection on a file and save results.
    
    Args:
        input_path: Path to input data (CSV or Parquet)
        output_dir: Directory for outputs
        method: Detection method
        column: Column to analyze
        save_plot: Whether to save visualization
        save_clean: Whether to save cleaned data
        config: Optional config
    
    Returns:
        Dictionary with results summary
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)
    
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    variable_name = input_path.stem
    
    print(f"\n{'='*60}")
    print(f"ANOMALY DETECTION: {variable_name}")
    print(f"{'='*60}")
    print(f"  Method: {method}")
    print(f"  Data points: {len(df)}")
    
    # Detect anomalies
    if config is None:
        config = AnomalyConfig(method=AnomalyMethod(method))
    
    detector = AnomalyDetector(config)
    df_detected, result = detector.detect(df, column=column, return_details=True)
    
    print(f"  Anomalies found: {result.n_anomalies} ({result.anomaly_pct:.2f}%)")
    
    # Interpolate
    df_clean = detector.interpolate_anomalies(df_detected, column=column)
    
    # Save results
    results = {
        "variable": variable_name,
        "method": method,
        "config": config.to_dict(),
        "result": result.to_dict()
    }
    
    # Save summary JSON
    summary_path = output_dir / f"{variable_name}_anomaly_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save plot
    if save_plot:
        plot_path = output_dir / f"{variable_name}_anomalies.png"
        detector.plot_anomalies(
            df_clean,
            column=column,
            save_path=str(plot_path),
            title=f"Anomaly Detection: {variable_name}"
        )
        plt.close()
    
    # Save cleaned data
    if save_clean:
        clean_path = output_dir / f"{variable_name}_clean.parquet"
        df_clean.to_parquet(clean_path, index=False)
        print(f"  Saved cleaned data: {clean_path}")
    
    print(f"{'='*60}\n")
    
    return results


def batch_anomaly_detection(
    processed_dir: str = "processed",
    output_dir: str = "anomaly_results",
    machine: str = "DESF",
    method: str = "combined",
    config: Optional[AnomalyConfig] = None
) -> Dict[str, Dict]:
    """
    Run anomaly detection on all variables for a machine.
    
    Args:
        processed_dir: Directory with processed parquet files
        output_dir: Directory for outputs
        machine: Machine name (DESF, PICADORA, PLANT)
        method: Detection method
        config: Optional config
    
    Returns:
        Dictionary mapping variable names to results
    """
    processed_path = Path(processed_dir) / machine
    output_path = Path(output_dir) / machine
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not processed_path.exists():
        print(f"Error: Directory not found: {processed_path}")
        return {}
    
    parquet_files = list(processed_path.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {processed_path}")
        return {}
    
    print(f"\n{'='*70}")
    print(f"BATCH ANOMALY DETECTION: {machine}")
    print(f"{'='*70}")
    print(f"  Found {len(parquet_files)} variables")
    print(f"  Method: {method}")
    
    all_results = {}
    
    for parquet_file in parquet_files:
        try:
            results = run_anomaly_detection(
                input_path=str(parquet_file),
                output_dir=str(output_path),
                method=method,
                config=config
            )
            all_results[parquet_file.stem] = results
        except Exception as e:
            print(f"  Error processing {parquet_file.stem}: {e}")
    
    # Save combined summary
    summary_path = output_path / "batch_anomaly_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    print(f"  Processed: {len(all_results)} variables")
    print(f"  Output: {output_path}")
    print(f"{'='*70}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Anomaly Detection for Industrial Time Series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Detection Methods:
  zscore           - Standard z-score (simple, sensitive to outliers)
  modified_zscore  - Modified z-score using MAD (robust)
  iqr              - Interquartile range method
  rolling          - Rolling window statistics (good for non-stationary)
  isolation_forest - Isolation Forest (ML-based)
  combined         - Voting ensemble of multiple methods (recommended)

Examples:
  # Single file with default combined method
  python -m anomaly.anomaly_detector --input processed/DESF/corriente_motor_a.parquet
  
  # Batch processing for a machine
  python -m anomaly.anomaly_detector --batch --machine DESF
  
  # Custom threshold
  python -m anomaly.anomaly_detector --input data.parquet --method modified_zscore --threshold 4.0
        """
    )
    
    parser.add_argument("--input", "-i", help="Input file (CSV or Parquet)")
    parser.add_argument("--output", "-o", default="anomaly_results", help="Output directory")
    parser.add_argument(
        "--method", "-m",
        choices=["zscore", "modified_zscore", "iqr", "rolling", "isolation_forest", "combined"],
        default="combined",
        help="Detection method (default: combined)"
    )
    parser.add_argument("--column", "-c", default="y", help="Column to analyze")
    parser.add_argument("--threshold", "-t", type=float, help="Detection threshold (method-dependent)")
    parser.add_argument("--batch", action="store_true", help="Batch process all files for a machine")
    parser.add_argument("--machine", choices=["DESF", "PICADORA", "PLANT"], default="DESF",
                       help="Machine for batch processing")
    parser.add_argument("--conservative", action="store_true", help="Use conservative detection settings")
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive detection settings")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-clean", action="store_true", help="Skip saving cleaned data")
    
    args = parser.parse_args()
    
    # Create config
    if args.conservative:
        config = AnomalyConfig.conservative()
    elif args.aggressive:
        config = AnomalyConfig.aggressive()
    else:
        config = AnomalyConfig(method=AnomalyMethod(args.method))
    
    if args.threshold:
        if args.method == "zscore":
            config.zscore_threshold = args.threshold
        elif args.method == "modified_zscore":
            config.modified_zscore_threshold = args.threshold
        elif args.method == "iqr":
            config.iqr_multiplier = args.threshold
        elif args.method == "rolling":
            config.rolling_std_multiplier = args.threshold
    
    # Run detection
    if args.batch:
        batch_anomaly_detection(
            output_dir=args.output,
            machine=args.machine,
            method=args.method,
            config=config
        )
    elif args.input:
        run_anomaly_detection(
            input_path=args.input,
            output_dir=args.output,
            method=args.method,
            column=args.column,
            save_plot=not args.no_plot,
            save_clean=not args.no_clean,
            config=config
        )
    else:
        parser.error("Either --input or --batch is required")


if __name__ == "__main__":
    main()
