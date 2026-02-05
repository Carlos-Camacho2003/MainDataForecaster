"""
Script: analyze_historic_data.py

Standalone tool to visualize and analyze historic data availability and gaps 
across raw, processed, and EPI datasets.

Usage:
    python scripts/analyze_historic_data.py
    python scripts/analyze_historic_data.py --machine DESF

Generates an interactive HTML timeline visualization.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import webbrowser
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def scan_data_availability(machine="DESF"):
    """
    Scan all relevant data folders for the machine and return time ranges.
    """
    data_ranges = []
    
    print(f"Scanning data for {machine}...")
    
    # 1. Raw Data (if CSVs exist)
    raw_dir = PROJECT_ROOT / "data" / "raw" / machine
    if raw_dir.exists():
        for csv_file in sorted(raw_dir.glob("*.csv")):
            try:
                # Optimized read: just get first and last row if possible, 
                # but for CSV we might need to parse. reading header + tail is hard without full read.
                # Let's read columns only.
                # Actually, 'timestamp' might be named differently.
                df = pd.read_csv(csv_file, nrows=5)
                ts_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
                
                if ts_col:
                    # Read full column only for speed
                    df_idx = pd.read_csv(csv_file, usecols=[ts_col])
                    df_idx[ts_col] = pd.to_datetime(df_idx[ts_col])
                    
                    data_ranges.append({
                        "Category": "Raw CSV",
                        "Name": csv_file.name,
                        "Start": df_idx[ts_col].min(),
                        "End": df_idx[ts_col].max(),
                        "Count": len(df_idx)
                    })
                    print(f"  [Raw] {csv_file.name}: {len(df_idx)} rows")
            except Exception as e:
                print(f"  Error reading {csv_file.name}: {e}")

    # 2. Processed Parquet (Sensor Data)
    processed_dir = PROJECT_ROOT / "processed" / machine
    if processed_dir.exists():
        for pq_file in sorted(processed_dir.glob("*.parquet")):
            try:
                # Parquet metadata is fast
                df = pd.read_parquet(pq_file, columns=['timestamp'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                data_ranges.append({
                    "Category": "Processed (Sensor)",
                    "Name": pq_file.stem,
                    "Start": df['timestamp'].min(),
                    "End": df['timestamp'].max(),
                    "Count": len(df)
                })
                print(f"  [Processed] {pq_file.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {pq_file.name}: {e}")

    # 3. EPI Data
    # EPI is usually in data/epi/ or data/epi/{MACHINE}/
    epi_paths = [
        PROJECT_ROOT / "data" / "epi" / f"DATOS_EPI_{machine}_HOURLY.csv",
        PROJECT_ROOT / "data" / "epi" / machine / f"DATOS_EPI_{machine}_HOURLY.csv"
    ]
    
    for epi_path in epi_paths:
        if epi_path.exists():
            try:
                df = pd.read_csv(epi_path)
                ts_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
                df[ts_col] = pd.to_datetime(df[ts_col], dayfirst=True) # Assuming potential format issues
                
                data_ranges.append({
                    "Category": "EPI Data",
                    "Name": "EPI Hourly",
                    "Start": df[ts_col].min(),
                    "End": df[ts_col].max(),
                    "Count": len(df)
                })
                print(f"  [EPI] {epi_path.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {epi_path.name}: {e}")

    # 4. Forecasts
    forecast_paths = [
        PROJECT_ROOT / "data" / "forecasts" / machine / f"FORECAST_{machine}_2_DAYS.csv"
    ]
    
    for f_path in forecast_paths:
        if f_path.exists():
            try:
                df = pd.read_csv(f_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    data_ranges.append({
                        "Category": "Forecast Output",
                        "Name": f_path.name,
                        "Start": df['timestamp'].min(),
                        "End": df['timestamp'].max(),
                        "Count": len(df)
                    })
                    print(f"  [Forecast] {f_path.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error reading {f_path.name}: {e}")
                
    return pd.DataFrame(data_ranges)

def create_visualization(df_ranges, machine):
    if df_ranges.empty:
        print("No data found to visualize.")
        return

    # Ensure dates are datetime objects
    df_ranges["Start"] = pd.to_datetime(df_ranges["Start"], errors='coerce')
    df_ranges["End"] = pd.to_datetime(df_ranges["End"], errors='coerce')

    # Drop invalid rows
    initial_len = len(df_ranges)
    df_ranges = df_ranges.dropna(subset=["Start", "End"])
    if len(df_ranges) < initial_len:
        print(f"Dropped {initial_len - len(df_ranges)} rows with invalid dates")

    if df_ranges.empty:
        print("No valid data ranges to visualize after cleaning.")
        return

    # Sort by Start time
    df_ranges = df_ranges.sort_values("Start")
    
    fig = px.timeline(
        df_ranges, 
        x_start="Start", 
        x_end="End", 
        y="Name", 
        color="Category",
        hover_data=["Count", "Start", "End"],
        title=f"Historic Data Availability Timeline - {machine}"
    )
    
    # Customize layout
    fig.update_yaxes(autorange="reversed") # Top to bottom
    fig.update_layout(
        xaxis_title="Timeline",
        yaxis_title="File / Variable",
        legend_title="Data Source",
        template="plotly_white",
        height=max(600, len(df_ranges) * 30),
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    
    # Save to HTML
    output_file = PROJECT_ROOT / f"historic_analysis_{machine}.html"
    fig.write_html(str(output_file))
    print(f"\nVisualization saved to: {output_file}")
    
    # Attempt to open
    try:
        webbrowser.open(f"file://{output_file}")
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description="Analyze historic data ranges.")
    parser.add_argument("--machine", default="DESF", choices=["DESF", "PICADORA", "PLANT"], help="Machine to analyze")
    args = parser.parse_args()
    
    print(f"Analyzing historic data for {args.machine}...")
    
    df = scan_data_availability(args.machine)
    
    if not df.empty:
        print("\nSummary of Data Ranges:")
        print(df[["Category", "Name", "Start", "End", "Count"]].to_string())
        
        create_visualization(df, args.machine)
    else:
        print("No data found.")

if __name__ == "__main__":
    main()
