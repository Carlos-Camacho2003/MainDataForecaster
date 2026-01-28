"""
Anomaly Detection Module for Industrial Time Series

This module provides multiple anomaly detection methods for industrial sensor data,
specifically designed for predictive maintenance applications.

Methods:
- Statistical: Z-score, IQR, Modified Z-score (robust)
- Rolling: Rolling statistics with adaptive thresholds
- Isolation Forest: Unsupervised anomaly detection
- Combined: Ensemble of multiple methods

Usage:
    from anomaly import AnomalyDetector
    
    detector = AnomalyDetector(method='combined')
    df_with_anomalies = detector.detect(df)
    df_clean = detector.interpolate_anomalies(df)
"""

from anomaly.anomaly_detector import (
    AnomalyDetector,
    detect_anomalies,
    interpolate_anomalies,
    run_anomaly_detection
)
from anomaly.anomaly_config import AnomalyConfig, AnomalyMethod

__all__ = [
    'AnomalyDetector',
    'AnomalyConfig',
    'AnomalyMethod',
    'detect_anomalies',
    'interpolate_anomalies',
    'run_anomaly_detection'
]
