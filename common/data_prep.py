# data_prep.py
"""
Data Preparation Module for N-BEATS Forecasting Pipeline

This module provides secure data preprocessing with hourly maximum aggregation
for predictive maintenance time series forecasting.

Features:
- Secure input validation and sanitization
- Hourly maximum aggregation for all variables
- Support for both DESFIBRADORA and PICADORA machines
- Comprehensive metadata generation

Usage:
    python -m common.data_prep --input data/DATOS_DESF_CLEANED.csv --target corriente_motor_a
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import argparse
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# =============================================================================
# SECURITY: Input Validation
# =============================================================================

def validate_file_path(filepath: str, allowed_extensions: List[str] = None) -> Path:
    """
    Securely validate and sanitize file paths.
    
    Args:
        filepath: Path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.parquet'])
    
    Returns:
        Validated Path object
    
    Raises:
        ValueError: If path is invalid or contains suspicious patterns
    """
    if allowed_extensions is None:
        allowed_extensions = ['.csv', '.parquet', '.json']
    
    # Convert to Path object
    path = Path(filepath).resolve()
    
    # Check for path traversal attempts
    if '..' in str(filepath):
        raise ValueError(f"Path traversal detected in: {filepath}")
    
    # Validate extension
    if path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Invalid file extension: {path.suffix}. "
            f"Allowed: {allowed_extensions}"
        )
    
    return path


def validate_column_name(col_name: str) -> str:
    """
    Validate and sanitize column names to prevent injection attacks.
    
    Args:
        col_name: Column name to validate
    
    Returns:
        Sanitized column name
    
    Raises:
        ValueError: If column name contains invalid characters
    """
    # Allow only alphanumeric, underscores, and limited special chars
    pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$'
    
    clean_name = col_name.strip().lower()
    
    if not re.match(pattern, clean_name):
        raise ValueError(
            f"Invalid column name: '{col_name}'. "
            f"Column names must start with a letter and contain only "
            f"alphanumeric characters and underscores."
        )
    
    return clean_name


def validate_resample_rule(rule: str) -> str:
    """
    Validate resample rule to ensure it's a valid pandas offset alias.
    
    Args:
        rule: Resample rule string (e.g., '1h', '30min')
    
    Returns:
        Validated rule
    
    Raises:
        ValueError: If rule is invalid
    """
    # Allowed patterns for time-based resampling
    allowed_patterns = [
        r'^\d+[hH]$',        # Hours: 1h, 2H
        r'^\d+[mM]in$',      # Minutes: 30min, 15Min
        r'^\d+[sS]$',        # Seconds: 30s, 60S
        r'^\d+[dD]$',        # Days: 1d, 7D
    ]
    
    for pattern in allowed_patterns:
        if re.match(pattern, rule):
            return rule.lower()
    
    raise ValueError(
        f"Invalid resample rule: '{rule}'. "
        f"Use formats like '1h', '30min', '30s', '1d'."
    )


def compute_file_hash(filepath: Path) -> str:
    """
    Compute SHA-256 hash of a file for integrity verification.
    
    Args:
        filepath: Path to file
    
    Returns:
        Hex string of file hash
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# =============================================================================
# VARIABLE CATALOG
# =============================================================================

CATALOGO = {
    # Condition variables
    "aceleracion": dict(familia="Condición", unidad="g", alarma=1.5, critico=2.5),
    "velocidad": dict(familia="Condición", unidad="mm/s", alarma=5.0, critico=7.5),
    "envolvente": dict(familia="Condición", unidad="gE", alarma=3.0, critico=5.0),
    
    # Pathology variables
    "desalineacion_rad": dict(familia="Patología", unidad="mm/s", alarma=3.0, critico=4.73),
    "desalineacion_ang": dict(familia="Patología", unidad="mm/s", alarma=3.0, critico=4.73),
    "desbalance": dict(familia="Patología", unidad="mm/s", alarma=3.0, critico=4.73),
    "soltura": dict(familia="Patología", unidad="mm/s", alarma=3.0, critico=4.73),
    "soltura_chum": dict(familia="Patología", unidad="mm/s", alarma=3.0, critico=4.73),
    "rodamiento_chum": dict(familia="Patología", unidad="mm/s", alarma=2.0, critico=3.0),
    "desalineacion": dict(familia="Patología", unidad="mm/s", alarma=3.0, critico=4.73),
    
    # Operation variables
    "corriente_motor": dict(familia="Operación", unidad="A", alarma=250.0, critico=300.0),
    "temp_chum": dict(familia="Operación", unidad="°C", alarma=65.0, critico=80.0),
}

CANAL_MAP = {
    "la_h": "LA-H",
    "la_a": "LA-A",
    "ll_h": "LL-H",
    "lb_h": "LB-H",
    "lado_a": "LADO-A",
    "lado_b": "LADO-B"
}


# =============================================================================
# COLUMN PARSING
# =============================================================================

def _parse_col(col: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Parse column name to extract variable base, channel, and operation flag.
    
    Args:
        col: Column name to parse
    
    Returns:
        Tuple of (base_variable, channel, is_operation)
    """
    # Operation variables - DESFIBRADORA naming
    if col in ("corriente_motor_a", "corriente_motor_b"):
        return "corriente_motor", CANAL_MAP["lado_a"] if "a" in col else CANAL_MAP["lado_b"], True
    if col in ("temp_chum_lado_a", "temp_chum_lado_b"):
        return "temp_chum", CANAL_MAP["lado_a"] if "a" in col else CANAL_MAP["lado_b"], True
    
    # Operation variables - PICADORA naming
    if col == "corriente_picadora":
        return "corriente_motor", "PICADORA", True
    if col in ("temperatura_lado_der", "temperatura_lado_izq"):
        return "temp_chum", "LADO-DER" if "der" in col else "LADO-IZQ", True
    
    # Condition variables: aceleracion, velocidad, envolvente
    m = re.match(r"^(aceleracion|velocidad|envolvente)_chum_(la_h|la_a|ll_h|lb_h)$", col)
    if m:
        return m.group(1), CANAL_MAP[m.group(2)], False
    
    # Pathology variables with lado_a or lado_b suffix (DESFIBRADORA)
    m = re.match(
        r"^(desbalance|desalineacion_rad|desalineacion_ang|soltura|rodamiento_chum|desalineacion)_(lado_a|lado_b)$",
        col
    )
    if m:
        base = m.group(1)
        canal = CANAL_MAP[m.group(2)]
        return base, canal, False
    
    # Pathology variables for PICADORA (la/ll suffix)
    # Fixed to handle optional _chum_ part between variable and channel
    m = re.match(
        r"^(desbalance|desalineacion_rad|desalineacion_ang|soltura|rodamiento)(?:_chum)?_(la|ll)$",
        col
    )
    if m:
        base = m.group(1)
        canal = "LA" if m.group(2) == "la" else "LL"
        
        # Normalize base to match CATALOGO keys
        if base == "rodamiento":
            base = "rodamiento_chum"
            
        return base, canal, False
    
    return None, None, False


# =============================================================================
# DATA PROCESSING FUNCTIONS
# =============================================================================

def detect_machine_type(df: pd.DataFrame) -> str:
    """
    Detect machine type from DataFrame columns.
    
    Args:
        df: DataFrame with columns to analyze
    
    Returns:
        Machine type: 'DESF', 'PICADORA', or 'UNKNOWN'
    """
    cols_lower = [c.lower() for c in df.columns]
    
    # DESF has corriente_motor_a/b
    if 'corriente_motor_a' in cols_lower or 'temp_chum_lado_a' in cols_lower:
        return 'DESF'
    # PICADORA has corriente_picadora
    elif 'corriente_picadora' in cols_lower or 'temperatura_lado_der' in cols_lower:
        return 'PICADORA'
    
    return 'UNKNOWN'


def agregar_features_operacion(
    df: pd.DataFrame,
    off_thr: float = 20.0,
    on_thr: float = 50.0,
    corriente_col: str = "corriente_motor_a"
) -> pd.DataFrame:
    """
    Add operational features to the DataFrame.
    
    Args:
        df: Input DataFrame
        off_thr: Current threshold for OFF state
        on_thr: Current threshold for ON state
        corriente_col: Name of current column
    
    Returns:
        DataFrame with added operational features
    """
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.sort_values("timestamp").set_index("timestamp")
    
    # Detect current column - support both DESFIBRADORA and PICADORA naming
    if corriente_col not in out.columns:
        current_cols = [
            "corriente_motor_a",
            "corriente_motor_b",
            "corriente_motor",
            "corriente_picadora"
        ]
        for col in current_cols:
            if col in out.columns:
                corriente_col = col
                break
        else:
            raise ValueError(
                f"No current column found. Searched for: {current_cols}"
            )
    
    c = out[corriente_col].astype(float)
    estado = np.where(c < off_thr, "OFF", np.where(c < on_thr, "IDLE", "ON"))
    
    out["estado"] = estado
    out["is_off"] = (out["estado"] == "OFF").astype(int)
    out["is_idle"] = (out["estado"] == "IDLE").astype(int)
    out["is_on"] = (out["estado"] == "ON").astype(int)
    out["hora"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["turno"] = pd.cut(
        out["hora"],
        bins=[-0.1, 8, 16, 24],
        labels=[0, 1, 2]
    ).astype(int)
    
    return out.reset_index()


def resample_df(
    df: pd.DataFrame,
    rule: str = "1h",
    num_agg: str = "max",  # CHANGED: Default to max instead of median
    cat_agg: str = "last"
) -> pd.DataFrame:
    """
    Resample DataFrame to specified frequency using hourly maximum aggregation.
    
    IMPORTANT: All numeric variables are aggregated by MAXIMUM value per hour
    to capture peak values for predictive maintenance.
    
    Args:
        df: Input DataFrame with 'timestamp' column
        rule: Pandas offset alias (e.g., '1h' for hourly)
        num_agg: Aggregation function for numeric columns (default: 'max')
        cat_agg: Aggregation function for categorical columns (default: 'last')
    
    Returns:
        Resampled DataFrame
    """
    # Validate resample rule
    rule = validate_resample_rule(rule)
    
    df2 = df.copy()
    df2["timestamp"] = pd.to_datetime(df2["timestamp"], utc=True, errors="coerce")
    df2 = df2.set_index("timestamp")
    
    num_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df2.columns if c not in num_cols]
    
    # Numeric resample using MAX aggregation
    if num_cols:
        num_rs = df2[num_cols].resample(rule).agg(num_agg)
    else:
        num_rs = pd.DataFrame(index=pd.DatetimeIndex([]))
    
    # Categorical / non-numeric resample
    if cat_cols:
        if cat_agg == "mode":
            cat_rs = df2[cat_cols].resample(rule).agg(
                lambda x: x.mode().iat[0] if not x.mode().empty else np.nan
            )
        else:
            cat_rs = df2[cat_cols].resample(rule).agg(cat_agg)
    else:
        cat_rs = pd.DataFrame(index=num_rs.index)
    
    out = pd.concat([num_rs, cat_rs], axis=1).reset_index().sort_values("timestamp")
    return out


def build_train_matrix(
    df_raw: pd.DataFrame,
    target_col: str,
    resample_rule: str = "1h"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build training matrix for N-BEATS model with hourly maximum aggregation.
    
    Args:
        df_raw: Raw DataFrame
        target_col: Target variable column name
        resample_rule: Resampling frequency (default: '1h')
    
    Returns:
        Tuple of (training_dataframe, metadata_dict)
    """
    # Validate inputs
    target_col = validate_column_name(target_col)
    resample_rule = validate_resample_rule(resample_rule)
    
    df = df_raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    
    ts_col = "timestamp_grid" if "timestamp_grid" in df.columns else "timestamp"
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    
    if target_col not in df.columns:
        available_cols = [c for c in df.columns if c not in ("timestamp", "maquina")][:10]
        raise ValueError(
            f"Target '{target_col}' not found. "
            f"Available columns include: {available_cols}"
        )
    
    # Parse target column
    base, canal, es_op = _parse_col(target_col)
    if base is None:
        base, canal, es_op = target_col, None, True
    
    meta_base = CATALOGO.get(base, {
        "familia": "Unknown",
        "unidad": "",
        "alarma": None,
        "critico": None
    })
    
    meta = {
        "variable": base,
        "canal": canal,
        "familia": meta_base["familia"],
        "unidad": meta_base["unidad"],
        "alarma": meta_base["alarma"],
        "critico": meta_base["critico"],
        "target_col": target_col,
        "model_type": "N-BEATS",
        "aggregation_method": "max",  # Explicitly document aggregation method
    }
    
    # Add operational features
    feat = agregar_features_operacion(df)
    
    # Exog columns - support both DESFIBRADORA and PICADORA
    exog_cols_desf = [
        "corriente_motor_a", "corriente_motor_b",
        "temp_chum_lado_a", "temp_chum_lado_b"
    ]
    exog_cols_pic = [
        "corriente_picadora", "corriente_motor",
        "temperatura_lado_der", "temperatura_lado_izq"
    ]
    
    exog_cols = []
    if any(col in feat.columns for col in exog_cols_desf):
        exog_cols.extend([c for c in exog_cols_desf if c in feat.columns])
    if any(col in feat.columns for col in exog_cols_pic):
        exog_cols.extend([c for c in exog_cols_pic if c in feat.columns])
    
    # Add operational state and temporal features
    exog_cols.extend(["is_off", "is_idle", "is_on", "turno", "hora", "dow"])
    
    # Remove target from exog to avoid duplication
    exog_cols = [c for c in exog_cols if c != target_col and c in feat.columns]
    
    cols = ["maquina", "timestamp", target_col] + exog_cols
    
    # Resample using MAXIMUM aggregation
    feat_rs = resample_df(feat[cols], rule=resample_rule, num_agg="max")
    
    # Filter to ON state only
    if "is_on" in feat_rs.columns:
        feat_rs = feat_rs[feat_rs["is_on"] == 1].copy()
    
    feat_rs = feat_rs.rename(columns={target_col: "y"})
    feat_rs["y"] = feat_rs["y"].interpolate(limit=6)
    
    meta.update({
        "resample_rule": resample_rule,
        "exog_cols": [c for c in feat_rs.columns if c not in ("timestamp", "maquina", "y")],
        "n_samples": len(feat_rs),
        "date_range": {
            "start": str(feat_rs["timestamp"].min()) if len(feat_rs) > 0 else None,
            "end": str(feat_rs["timestamp"].max()) if len(feat_rs) > 0 else None,
        }
    })
    
    return feat_rs, meta


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for data preparation."""
    ap = argparse.ArgumentParser(
        description="Data preparation for N-BEATS forecasting pipeline"
    )
    ap.add_argument("--input", required=True, help="Input CSV file path")
    ap.add_argument("--sep", default=",", help="CSV separator (default: ',')")
    ap.add_argument("--target", required=True, help="Target variable column name")
    ap.add_argument("--resample", default="1h", help="Resample rule (default: '1h')")
    ap.add_argument("--outdir", default="processed", help="Output directory")
    ap.add_argument(
        "--machine",
        default=None,
        choices=["DESF", "PICADORA"],
        help="Machine type (auto-detected if not provided)"
    )
    
    args = ap.parse_args()
    
    # Validate input file path
    input_path = validate_file_path(args.input, ['.csv'])
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Detect machine type from filename if not provided
    machine_type = args.machine
    if not machine_type:
        filename = input_path.name.upper()
        if 'DESF' in filename:
            machine_type = 'DESF'
        elif 'PICADORA' in filename:
            machine_type = 'PICADORA'
        else:
            machine_type = 'UNKNOWN'
    
    # Create machine-specific output directory
    outdir = Path(args.outdir) / machine_type
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_path}...")
    df_raw = pd.read_csv(input_path, sep=args.sep)
    
    # Parse timestamps
    if "timestamp_grid" in df_raw.columns:
        df_raw["timestamp_grid"] = pd.to_datetime(
            df_raw["timestamp_grid"], errors="coerce"
        )
    elif "timestamp" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(
            df_raw["timestamp"], errors="coerce"
        )
    
    # Auto-detect machine type from data if still unknown
    if machine_type == 'UNKNOWN':
        machine_type = detect_machine_type(df_raw)
        outdir = Path(args.outdir) / machine_type
        outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Machine type: {machine_type}")
    print(f"Target variable: {args.target}")
    print(f"Aggregation: MAXIMUM per hour")
    
    # Build training matrix with MAX aggregation
    train_df, meta = build_train_matrix(
        df_raw,
        target_col=args.target,
        resample_rule=args.resample
    )
    
    # Add machine type and input file info to metadata
    meta["machine_type"] = machine_type
    meta["input_file"] = str(input_path)
    meta["input_file_hash"] = compute_file_hash(input_path)
    meta["generated_at"] = datetime.now().isoformat()
    meta["pipeline_version"] = "2.0.0"
    
    # Save outputs
    out_data = outdir / f"{args.target}.parquet"
    out_meta = outdir / f"{args.target}_meta.json"
    
    train_df.to_parquet(out_data, index=False)
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"Training data: {out_data}")
    print(f"Metadata: {out_meta}")
    print(f"Samples: {len(train_df)}")
    print(f"Aggregation: Maximum per hour")
    print(f"{'='*60}\n")
    
    # Print paths for pipeline integration
    print(str(out_data))
    print(str(out_meta))


if __name__ == "__main__":
    main()
