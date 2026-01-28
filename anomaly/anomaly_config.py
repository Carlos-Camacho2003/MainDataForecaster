"""
Anomaly Detection Configuration

Defines configuration classes and enums for anomaly detection methods.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any


class AnomalyMethod(Enum):
    """Available anomaly detection methods."""
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    IQR = "iqr"
    ROLLING = "rolling"
    ISOLATION_FOREST = "isolation_forest"
    COMBINED = "combined"


class InterpolationMethod(Enum):
    """Available interpolation methods for anomaly replacement."""
    LINEAR = "linear"
    CUBIC = "cubic"
    SPLINE = "spline"
    ROLLING_MEAN = "rolling_mean"
    ROLLING_MEDIAN = "rolling_median"


@dataclass
class AnomalyConfig:
    """
    Configuration for anomaly detection.
    
    Attributes:
        method: Primary detection method to use
        zscore_threshold: Threshold for z-score method (default: 3.0)
        modified_zscore_threshold: Threshold for modified z-score (default: 3.5)
        iqr_multiplier: IQR multiplier for outlier bounds (default: 1.5)
        rolling_window: Window size for rolling statistics (default: 24 hours)
        rolling_std_multiplier: Multiplier for rolling std threshold (default: 3.0)
        isolation_contamination: Expected contamination rate for Isolation Forest
        min_anomaly_votes: Minimum votes required for combined method
        interpolation_method: Method to use for interpolating anomalies
        interpolation_window: Window size for rolling interpolation methods
        preserve_extremes: Whether to preserve extreme values (e.g., alarm conditions)
        extreme_percentile: Percentile above which to preserve values
        min_gap_hours: Minimum gap size to consider for interpolation
    """
    method: AnomalyMethod = AnomalyMethod.COMBINED
    
    # Z-score parameters
    zscore_threshold: float = 3.0
    modified_zscore_threshold: float = 3.5
    
    # IQR parameters
    iqr_multiplier: float = 1.5
    
    # Rolling parameters
    rolling_window: int = 24
    rolling_std_multiplier: float = 3.0
    
    # Isolation Forest parameters
    isolation_contamination: float = 0.05
    isolation_n_estimators: int = 100
    isolation_random_state: int = 42
    
    # Combined method parameters
    min_anomaly_votes: int = 2
    methods_to_combine: List[AnomalyMethod] = field(default_factory=lambda: [
        AnomalyMethod.MODIFIED_ZSCORE,
        AnomalyMethod.IQR,
        AnomalyMethod.ROLLING
    ])
    
    # Interpolation parameters
    interpolation_method: InterpolationMethod = InterpolationMethod.CUBIC
    interpolation_window: int = 12
    
    # Preservation parameters
    preserve_extremes: bool = False
    extreme_percentile: float = 99.0
    
    # Gap handling
    min_gap_hours: int = 1
    max_consecutive_anomalies: int = 48  # Don't interpolate gaps > 48 hours
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "method": self.method.value,
            "zscore_threshold": self.zscore_threshold,
            "modified_zscore_threshold": self.modified_zscore_threshold,
            "iqr_multiplier": self.iqr_multiplier,
            "rolling_window": self.rolling_window,
            "rolling_std_multiplier": self.rolling_std_multiplier,
            "isolation_contamination": self.isolation_contamination,
            "isolation_n_estimators": self.isolation_n_estimators,
            "min_anomaly_votes": self.min_anomaly_votes,
            "interpolation_method": self.interpolation_method.value,
            "interpolation_window": self.interpolation_window,
            "preserve_extremes": self.preserve_extremes,
            "extreme_percentile": self.extreme_percentile,
            "max_consecutive_anomalies": self.max_consecutive_anomalies
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AnomalyConfig":
        """Create config from dictionary."""
        config = cls()
        if "method" in d:
            config.method = AnomalyMethod(d["method"])
        if "zscore_threshold" in d:
            config.zscore_threshold = d["zscore_threshold"]
        if "modified_zscore_threshold" in d:
            config.modified_zscore_threshold = d["modified_zscore_threshold"]
        if "iqr_multiplier" in d:
            config.iqr_multiplier = d["iqr_multiplier"]
        if "rolling_window" in d:
            config.rolling_window = d["rolling_window"]
        if "rolling_std_multiplier" in d:
            config.rolling_std_multiplier = d["rolling_std_multiplier"]
        if "isolation_contamination" in d:
            config.isolation_contamination = d["isolation_contamination"]
        if "min_anomaly_votes" in d:
            config.min_anomaly_votes = d["min_anomaly_votes"]
        if "interpolation_method" in d:
            config.interpolation_method = InterpolationMethod(d["interpolation_method"])
        if "interpolation_window" in d:
            config.interpolation_window = d["interpolation_window"]
        if "preserve_extremes" in d:
            config.preserve_extremes = d["preserve_extremes"]
        if "extreme_percentile" in d:
            config.extreme_percentile = d["extreme_percentile"]
        if "max_consecutive_anomalies" in d:
            config.max_consecutive_anomalies = d["max_consecutive_anomalies"]
        return config
    
    @classmethod
    def conservative(cls) -> "AnomalyConfig":
        """
        Create a conservative configuration that only flags extreme outliers.
        Good for preserving genuine spikes while removing obvious errors.
        """
        return cls(
            method=AnomalyMethod.COMBINED,
            zscore_threshold=4.0,
            modified_zscore_threshold=4.5,
            iqr_multiplier=2.0,
            rolling_std_multiplier=4.0,
            min_anomaly_votes=3,
            isolation_contamination=0.01
        )
    
    @classmethod
    def aggressive(cls) -> "AnomalyConfig":
        """
        Create an aggressive configuration that flags more potential anomalies.
        Good for data with known quality issues.
        """
        return cls(
            method=AnomalyMethod.COMBINED,
            zscore_threshold=2.5,
            modified_zscore_threshold=3.0,
            iqr_multiplier=1.0,
            rolling_std_multiplier=2.5,
            min_anomaly_votes=2,
            isolation_contamination=0.10
        )
    
    @classmethod
    def sensor_specific(cls, sensor_type: str) -> "AnomalyConfig":
        """
        Create configuration optimized for specific sensor types.
        
        Args:
            sensor_type: Type of sensor (temperature, vibration, current, etc.)
        """
        sensor_configs = {
            "temperature": cls(
                # Temperature tends to be smooth, aggressive detection
                zscore_threshold=3.0,
                rolling_window=48,
                rolling_std_multiplier=2.5,
                interpolation_method=InterpolationMethod.CUBIC
            ),
            "vibration": cls(
                # Vibration can spike legitimately, be conservative
                zscore_threshold=4.0,
                modified_zscore_threshold=4.5,
                rolling_window=12,
                rolling_std_multiplier=4.0,
                preserve_extremes=True,
                extreme_percentile=98.0
            ),
            "current": cls(
                # Current can spike during startup, moderate detection
                zscore_threshold=3.5,
                rolling_window=24,
                rolling_std_multiplier=3.0,
                interpolation_method=InterpolationMethod.ROLLING_MEDIAN
            ),
            "acceleration": cls(
                # Acceleration is similar to vibration
                zscore_threshold=4.0,
                modified_zscore_threshold=4.0,
                rolling_window=12,
                rolling_std_multiplier=3.5,
                preserve_extremes=True
            )
        }
        return sensor_configs.get(sensor_type.lower(), cls())
