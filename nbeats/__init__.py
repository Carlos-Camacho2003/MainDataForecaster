"""
N-BEATS Model Package

Neural Basis Expansion Analysis for Time Series forecasting.
This is the primary forecasting model for the IDC_RIOP pipeline.

Modules:
- nbeats_model.py: N-BEATS architecture implementation
- nbeats_train.py: Model training with PyTorch
- nbeats_forecast.py: Forecast generation with uncertainty quantification
- nbeats_visualize.py: Visualization tools

Features:
- Deep learning-based time series forecasting
- Monte Carlo dropout for uncertainty quantification
- Support for DESFIBRADORA and PICADORA machines
- Hourly maximum aggregation for all variables
"""

__version__ = "2.0.0"

try:
    from .nbeats_model import NBeatsModel, create_nbeats_model
    from .nbeats_train import train_nbeats, TimeSeriesDataset
    from .nbeats_forecast import forecast_nbeats, load_model_checkpoint
    
    __all__ = [
        "NBeatsModel",
        "create_nbeats_model",
        "train_nbeats",
        "TimeSeriesDataset",
        "forecast_nbeats",
        "load_model_checkpoint",
    ]
except ImportError:
    # PyTorch may not be installed
    __all__ = []
