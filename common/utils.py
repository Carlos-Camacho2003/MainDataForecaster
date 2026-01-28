"""
Shared utility functions for the N-BEATS forecasting pipeline.

This module provides secure utility functions for file handling,
validation, and data processing.
"""

from __future__ import annotations
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd


# =============================================================================
# SECURITY: Input Validation
# =============================================================================

def sanitize_path(filepath: str | Path) -> Path:
    """
    Sanitize and validate file paths to prevent path traversal attacks.
    
    Args:
        filepath: Path to sanitize
    
    Returns:
        Sanitized Path object
    
    Raises:
        ValueError: If path contains suspicious patterns
    """
    path = Path(filepath)
    
    # Check for path traversal attempts
    if '..' in str(filepath):
        raise ValueError(f"Path traversal detected: {filepath}")
    
    # Resolve to absolute path
    return path.resolve()


def validate_filename(filename: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validate filename for security.
    
    Args:
        filename: Filename to validate
        allowed_extensions: List of allowed extensions (e.g., ['.csv', '.json'])
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If filename is invalid
    """
    if allowed_extensions is None:
        allowed_extensions = ['.csv', '.parquet', '.json', '.pth', '.png', '.html']
    
    # Check for null bytes or other dangerous characters
    if '\x00' in filename or '\n' in filename or '\r' in filename:
        raise ValueError(f"Invalid characters in filename: {filename}")
    
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise ValueError(f"Invalid extension: {ext}. Allowed: {allowed_extensions}")
    
    return True


# =============================================================================
# MACHINE TYPE DETECTION
# =============================================================================

def detect_machine_type(filepath: str | Path) -> str:
    """
    Detect machine type (DESF or PICADORA) from input file path or contents.
    
    Args:
        filepath: Path to input CSV file
    
    Returns:
        Machine type: 'DESF', 'PICADORA', or 'UNKNOWN'
    """
    filename = Path(filepath).name.upper()
    
    if 'DESF' in filename:
        return 'DESF'
    elif 'PICADORA' in filename:
        return 'PICADORA'
    else:
        # Try to detect from column names if filename is ambiguous
        try:
            df_cols = pd.read_csv(filepath, nrows=0).columns
            cols_lower = [c.lower() for c in df_cols]
            
            # DESF has corriente_motor_a/b, PICADORA has corriente_picadora
            if 'corriente_motor_a' in cols_lower or 'temp_chum_lado_a' in cols_lower:
                return 'DESF'
            elif 'corriente_picadora' in cols_lower or 'temperatura_lado_der' in cols_lower:
                return 'PICADORA'
        except Exception:
            pass
    
    return 'UNKNOWN'


# =============================================================================
# FILE UTILITIES
# =============================================================================

def add_timestamp_to_filename(
    filepath: str | Path,
    format: str = "%Y%m%d_%H%M%S"
) -> Path:
    """
    Add timestamp to filename before extension.
    
    Args:
        filepath: Original file path
        format: Timestamp format string
    
    Returns:
        Path with timestamp inserted
    """
    filepath = Path(filepath)
    timestamp = datetime.now().strftime(format)
    stem = filepath.stem
    suffix = filepath.suffix
    
    new_name = f"{stem}_{timestamp}{suffix}"
    return filepath.parent / new_name


def compute_file_hash(filepath: Path, algorithm: str = 'sha256') -> str:
    """
    Compute hash of a file for integrity verification.
    
    Args:
        filepath: Path to file
        algorithm: Hash algorithm ('sha256', 'md5')
    
    Returns:
        Hex string of file hash
    """
    if algorithm == 'sha256':
        hash_obj = hashlib.sha256()
    elif algorithm == 'md5':
        hash_obj = hashlib.md5()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_obj.update(byte_block)
    
    return hash_obj.hexdigest()


# =============================================================================
# VERSION AND METADATA
# =============================================================================

def create_version_info(
    model_type: str,
    variable: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create version information for outputs.
    
    Args:
        model_type: Model type (e.g., 'N-BEATS')
        variable: Target variable name
        metadata: Additional metadata to include
    
    Returns:
        Dictionary with version info
    """
    import sys
    
    version_info = {
        "model_type": model_type,
        "variable": variable,
        "generated_at": datetime.now().isoformat(),
        "pipeline_version": "2.0.0",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "aggregation_method": "max"  # Always max for this pipeline
    }
    
    if metadata:
        version_info.update(metadata)
    
    return version_info


def save_metadata(
    output_path: str | Path,
    metadata: Dict[str, Any],
    include_version: bool = True
):
    """
    Save metadata to JSON file with optional versioning.
    
    Args:
        output_path: Path to save metadata
        metadata: Metadata dictionary
        include_version: Whether to add version info
    """
    output_path = sanitize_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if include_version:
        metadata["_version_info"] = {
            "created_at": datetime.now().isoformat(),
            "pipeline_version": "2.0.0"
        }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# =============================================================================
# FORECAST VALIDATION
# =============================================================================

def validate_forecast_csv(csv_path: str | Path) -> bool:
    """
    Validate forecast CSV format for consistency and correctness.
    
    Args:
        csv_path: Path to forecast CSV
    
    Returns:
        True if valid, False otherwise
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"❌ File not found: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_cols = ["timestamp", "yhat"]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            print(f"❌ Missing required columns: {missing}")
            return False
        
        # Check timestamp format
        pd.to_datetime(df["timestamp"])
        
        # Check for NaN values in forecast
        nan_count = df["yhat"].isna().sum()
        if nan_count > 0:
            print(f"⚠️ Warning: {nan_count} NaN values in forecast column")
        
        # Check for negative values (if applicable)
        neg_count = (df["yhat"] < 0).sum()
        if neg_count > 0:
            print(f"⚠️ Warning: {neg_count} negative values in forecast")
        
        print(f"✅ Forecast CSV is valid: {csv_path.name}")
        print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False


# =============================================================================
# FILE ORGANIZATION
# =============================================================================

def organize_outputs_by_date(
    source_dir: str | Path,
    target_dir: str | Path,
    pattern: str = "**/*_forecast.csv"
):
    """
    Organize forecast outputs into date-based folders.
    
    Args:
        source_dir: Source directory containing forecasts
        target_dir: Target directory for organized files
        pattern: File pattern to match
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    files_moved = 0
    
    for file_path in source_dir.glob(pattern):
        # Get file creation date
        timestamp = datetime.fromtimestamp(file_path.stat().st_mtime)
        date_folder = timestamp.strftime("%Y-%m")
        
        # Create target folder
        dest_folder = target_dir / date_folder
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy file (don't move to preserve original)
        import shutil
        dest_path = dest_folder / file_path.name
        shutil.copy2(file_path, dest_path)
        files_moved += 1
        
        print(f"  Copied: {file_path.name} → {date_folder}/")
    
    print(f"\n✅ Organized {files_moved} files by date")


# =============================================================================
# INPUT FILE DETECTION
# =============================================================================

def find_input_for_variable(
    variable: str,
    data_dir: str | Path = "data",
    candidate_files: Optional[List[Path]] = None,
    default_candidates: Optional[List[str]] = None,
    return_machine_type: bool = False,
) -> Optional[Path] | Optional[Tuple[Path, str]]:
    """
    Search CSV files for the first file containing the requested variable.
    
    Looks for `variable` in column names (case-insensitive). Reads only headers.
    
    Args:
        variable: Target variable name
        data_dir: Directory to search for CSV files
        candidate_files: Explicit list of files to check
        default_candidates: Default files to check first
        return_machine_type: If True, return tuple (path, machine_type)
    
    Returns:
        Path to the first matching file or None if not found.
        If return_machine_type=True, returns tuple (Path, machine_type) or None.
    """
    data_path = Path(data_dir)
    var = variable.strip().lower()
    
    if default_candidates is None:
        default_candidates = [
            str(data_path / "DATOS_DESF_CLEANED.csv"),
            str(data_path / "DATOS_PICADORA_CLEANED.csv")
        ]
    
    # Try default candidates first
    for d in default_candidates:
        p = Path(d)
        if p.exists():
            try:
                cols = pd.read_csv(p, nrows=0).columns
                cols = [c.strip().lower() for c in cols]
                if var in cols:
                    if return_machine_type:
                        return (p, detect_machine_type(p))
                    return p
            except Exception:
                continue
    
    # Scan all CSVs in data_dir
    if candidate_files is None:
        candidate_files = sorted(list(data_path.glob("*.csv")))
    
    matches: List[Path] = []
    for f in candidate_files:
        try:
            cols = pd.read_csv(f, nrows=0).columns
            cols = [c.strip().lower() for c in cols]
            if var in cols:
                matches.append(Path(f))
        except Exception:
            print(f"⚠️ Skipping unreadable file: {f}")
    
    if not matches:
        return None
    
    if len(matches) > 1:
        print(f"⚠️ Variable '{variable}' found in multiple files: {matches}")
        print(f"   Using the first match.")
    
    result_path = matches[0]
    
    if return_machine_type:
        machine_type = detect_machine_type(result_path)
        return (result_path, machine_type)
    
    return result_path


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_training_data(df: pd.DataFrame, min_samples: int = 100) -> Tuple[bool, List[str]]:
    """
    Validate training data for N-BEATS model.
    
    Args:
        df: Training DataFrame
        min_samples: Minimum required samples
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check minimum samples
    if len(df) < min_samples:
        issues.append(f"Insufficient samples: {len(df)} < {min_samples}")
    
    # Check for required columns
    if 'y' not in df.columns:
        issues.append("Missing 'y' column (target variable)")
    
    if 'timestamp' not in df.columns:
        issues.append("Missing 'timestamp' column")
    
    # Check for NaN in target
    if 'y' in df.columns:
        nan_pct = df['y'].isna().mean() * 100
        if nan_pct > 10:
            issues.append(f"High NaN percentage in target: {nan_pct:.1f}%")
    
    # Check for constant values
    if 'y' in df.columns:
        if df['y'].std() == 0:
            issues.append("Target has zero variance (constant value)")
    
    return (len(issues) == 0, issues)


def get_data_summary(df: pd.DataFrame, target_col: str = 'y') -> Dict[str, Any]:
    """
    Generate summary statistics for training data.
    
    Args:
        df: Training DataFrame
        target_col: Target column name
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "columns": list(df.columns),
    }
    
    if target_col in df.columns:
        y = df[target_col]
        summary["target"] = {
            "mean": float(y.mean()),
            "std": float(y.std()),
            "min": float(y.min()),
            "max": float(y.max()),
            "median": float(y.median()),
            "nan_count": int(y.isna().sum()),
            "nan_pct": float(y.isna().mean() * 100)
        }
    
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'])
        summary["time_range"] = {
            "start": str(ts.min()),
            "end": str(ts.max()),
            "duration_hours": float((ts.max() - ts.min()).total_seconds() / 3600)
        }
    
    return summary
