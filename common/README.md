# Common Utilities

Shared modules and utilities for the IDC_RIOP forecasting pipeline.

## Modules

### `data_prep.py` ⭐ NEW
Secure data preparation module with maximum aggregation for safety-critical monitoring.

**Features:**
- **Maximum aggregation per hour** - captures peak values for safety
- **Path validation** - prevents directory traversal attacks
- **Column name sanitization** - prevents injection attacks
- **File integrity verification** - SHA-256 hash tracking
- **Machine type auto-detection** - DESF and PICADORA support

**Usage:**
```python
from common.data_prep import build_train_matrix, resample_df, validate_file_path

# Validate input path
safe_path = validate_file_path("data/DATOS_DESF_CLEANED.csv")

# Build training matrix with MAX aggregation
train_df, meta = build_train_matrix(
    df_raw, 
    target_col="corriente_motor_a",
    resample_rule="1h"
)

print(f"Aggregation: {meta['aggregation_method']}")  # "max"
print(f"Model type: {meta['model_type']}")  # "N-BEATS"
```

### `sharepoint_uploader.py`
SharePoint integration for uploading forecast outputs to Power BI-accessible locations.

**Features:**
- Automatic authentication
- Batch file uploads
- Upload logging and tracking
- Folder organization by model type and date
- File integrity verification (MD5 checksums)

**Usage:**
```python
from common.sharepoint_uploader import SharePointUploader

uploader = SharePointUploader()
if uploader.connect():
    uploader.upload_forecast("forecast_visuals/DESF/corriente_motor_a_forecast.csv", "NBEATS")
    uploader.save_upload_log()
```

**Configuration:**
Create a `.env` file in the project root:
```
SHAREPOINT_SITE_URL=https://company.sharepoint.com/sites/yoursite
SHAREPOINT_USERNAME=your.email@company.com
SHAREPOINT_PASSWORD=your_password
```

### `utils.py`
General utility functions for the pipeline.

**Functions:**
- `sanitize_path()` - Validate and sanitize file paths
- `validate_filename()` - Check filename security
- `validate_forecast_csv()` - Validate Power BI compatibility
- `find_input_for_variable()` - Locate appropriate data file

**Usage:**
```python
from common.utils import validate_forecast_csv, sanitize_path

# Validate path
safe_path = sanitize_path("data/DATOS_DESF_CLEANED.csv")

# Validate forecast format
if validate_forecast_csv("forecast_visuals/DESF/corriente_motor_a_forecast.csv"):
    print("Ready for Power BI import")
```

## Installation

Install SharePoint dependencies:
```powershell
pip install Office365-REST-Python-Client
```

## Dependencies

- `office365-rest-python-client` - SharePoint API client
- `pandas` - Data validation
- `python-dotenv` (optional) - Environment variable management

## Configuration Template

Generate a configuration template:
```powershell
python -c "from common.sharepoint_uploader import SharePointUploader; SharePointUploader.create_config_template()"
```

This creates `.env.template` which you can copy to `.env` and fill in.

## Security Notes

- Never commit `.env` files to version control
- Use environment variables or secure credential storage in production
- Consider using Azure Key Vault for production deployments
- Rotate passwords regularly

## Folder Structure

Uploaded files are organized as:
```
SharePoint/
└── IDC_RIOP_Forecasts/
    └── NBEATS/
        ├── 2025-01/
        │   ├── DESF/
        │   │   ├── corriente_motor_a_forecast.csv
        │   │   ├── corriente_motor_a_summary.json
        │   │   └── corriente_motor_a_version_*.json
        │   └── PICADORA/
        │       └── velocidad_chum_ll_h_forecast.csv
        └── 2025-02/
            └── ...
```

## Power BI Integration

The CSV files uploaded by this module are structured for direct Power BI import:

**Required columns:**
- `timestamp` - ISO 8601 format datetime
- `yhat` - Point forecast
- `yhat_lo` - Lower confidence bound
- `yhat_hi` - Upper confidence bound

**Optional columns:**
- `p_exceed_alarma` - Probability of exceeding alarm threshold
- `p_exceed_critico` - Probability of exceeding critical threshold

## Troubleshooting

### Authentication Issues
```
Error: Authentication failed
```
**Solution:** Verify credentials in `.env` and check SharePoint permissions.

### Upload Failures
```
Error: Failed to upload file
```
**Solution:** 
1. Check network connectivity
2. Verify folder permissions in SharePoint
3. Ensure document library name is correct

### Module Not Found
```
ImportError: No module named 'office365'
```
**Solution:** Install dependencies: `pip install Office365-REST-Python-Client`
