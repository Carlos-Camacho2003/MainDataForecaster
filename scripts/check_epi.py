"""
Script: check_epi.py
Quick script to report min/max timestamps and numeric values (EPI) for EPI CSV files.

Usage:
    python scripts/check_epi.py  # default checks DATOS_EPI_*_HOURLY.csv files
    python scripts/check_epi.py data/epi/DATOS_EPI_PLANT_HOURLY.csv  # single file

This script uses pandas and prints per-file and combined statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

DEFAULT_FILES = [
    Path("data/epi/DESF/DATOS_EPI_DESF_HOURLLY.csv"),  # fallback for local name typos
    Path("data/epi/DESF/DATOS_EPI_DESF_HOURLY.csv"),
    Path("data/epi/PICADORA/DATOS_EPI_PICADORA_HOURLY.csv"),
    Path("data/epi/PLANT/DATOS_EPI_PLANT_HOURLY.csv")
]

FALLBACK_TS_NAMES = ["timestamp", "ts", "time", "date"]

def find_timestamp_col(df: pd.DataFrame) -> str | None:
    for name in FALLBACK_TS_NAMES:
        if name in df.columns:
            return name
    # no timestamp column found
    return None


def find_numeric_col(df: pd.DataFrame) -> str | None:
    # prefer columns that contain 'epi' or 'value' or 'y'
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numcols:
        return None

    pref_candidates = [c for c in numcols if "epi" in c.lower() or c.lower() in ("value", "y")]
    if pref_candidates:
        return pref_candidates[0]
    return numcols[0]


def read_file_stats(path: Path):
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"error": str(e)}

    stats = {"file": str(path)}
    ts_col = find_timestamp_col(df)
    if ts_col:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
            stats["min_timestamp"] = df[ts_col].min()
            stats["max_timestamp"] = df[ts_col].max()
        except Exception as e:
            stats["ts_error"] = str(e)
    else:
        stats["min_timestamp"] = None
        stats["max_timestamp"] = None

    val_col = find_numeric_col(df)
    if val_col:
        stats["value_col"] = val_col
        stats["min_value"] = df[val_col].min()
        stats["max_value"] = df[val_col].max()
    else:
        stats["value_col"] = None
        stats["min_value"] = None
        stats["max_value"] = None

    stats["rows"] = len(df)

    return stats


def print_stats(stats_list: list[dict]):
    print("\nPer-file EPI checks:\n")
    header = "File | Rows | MinTimestamp | MaxTimestamp | ValueCol | MinValue | MaxValue"
    print(header)
    print("-" * len(header))
    all_timestamps = []
    all_values = []
    total_rows = 0

    for s in stats_list:
        if s.get("error"):
            print(f"{Path(s['file']).name} | ERROR: {s['error']}")
            continue
        file_name = Path(s["file"]).name
        rows = s["rows"]
        min_ts = s["min_timestamp"]
        max_ts = s["max_timestamp"]
        val_col = s.get("value_col") or "-"
        min_val = s.get("min_value")
        max_val = s.get("max_value")
        print(f"{file_name} | {rows} | {min_ts} | {max_ts} | {val_col} | {min_val} | {max_val}")

        if min_ts is not None and not pd.isna(min_ts):
            all_timestamps.append(min_ts)
        if max_ts is not None and not pd.isna(max_ts):
            all_timestamps.append(max_ts)

        if min_val is not None and not (isinstance(min_val, float) and np.isnan(min_val)):
            all_values.append(min_val)
        if max_val is not None and not (isinstance(max_val, float) and np.isnan(max_val)):
            all_values.append(max_val)

        total_rows += rows

    print("\nCombined stats:")
    if all_timestamps:
        try:
            print("  MinTimestamp:", min(all_timestamps))
            print("  MaxTimestamp:", max(all_timestamps))
        except Exception:
            print("  Could not compute combined timestamps")
    else:
        print("  MinTimestamp: -")
        print("  MaxTimestamp: -")

    if all_values:
        try:
            print("  MinValue:", min(all_values))
            print("  MaxValue:", max(all_values))
        except Exception:
            print("  Could not compute combined values")
    else:
        print("  MinValue: -")
        print("  MaxValue: -")

    print("  Total rows:", total_rows)


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Quick EPI CSV checks: min/max timestamps and numeric values.")
    p.add_argument("files", nargs="*", help="Paths to EPI CSV files to check", default=None)
    args = p.parse_args(argv)

    files_arg = args.files if args.files else None
    if not files_arg:
        # default list
        files = [f for f in DEFAULT_FILES if f.exists()]
        if not files:
            print("No default files found in data/epi/; provide file paths as arguments.")
            sys.exit(1)
    else:
        files = [Path(f) for f in files_arg]

    stats = []
    for f in files:
        if not Path(f).exists():
            stats.append({"file": str(f), "error": "file does not exist"})
            continue
        stats.append(read_file_stats(Path(f)))

    print_stats(stats)


if __name__ == "__main__":
    main()
