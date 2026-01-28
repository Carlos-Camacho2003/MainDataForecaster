"""
Common utilities for IDC_RIOP N-BEATS forecasting pipeline.

Modules:
- data_prep: Secure data preprocessing with hourly max aggregation
- sharepoint_uploader: Upload forecasts to SharePoint
- utils: Shared utility functions
"""

from .sharepoint_uploader import SharePointUploader
from .data_prep import (
    build_train_matrix,
    resample_df,
    validate_file_path,
    validate_column_name,
    CATALOGO,
)
from .utils import (
    detect_machine_type,
    find_input_for_variable,
    validate_forecast_csv,
    create_version_info,
    save_metadata,
)

__all__ = [
    "SharePointUploader",
    "build_train_matrix",
    "construir_tidy",
    "resample_df",
    "validate_file_path",
    "validate_column_name",
    "CATALOGO",
    "detect_machine_type",
    "find_input_for_variable",
    "validate_forecast_csv",
    "create_version_info",
    "save_metadata",
]
