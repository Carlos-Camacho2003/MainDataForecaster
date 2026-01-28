#!/usr/bin/env python
"""
Full SARIMAX Pipeline Orchestrator

Runs the complete workflow:
1. Preprocess data (sarimax_prep.py)
2. Train SARIMAX model (sarimax_train.py)
3. Generate forecasts (sarimax_forecast.py)
4. Create visualizations (sarimax_visualize.py)

Usage:
    python run_full_pipeline.py --variable corriente_motor_a --steps 24
    python run_full_pipeline.py --variable temp_chum_lado_a --steps 48 --order 1 1 1 --seasonal 1 1 1 24
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import json
from sarimax.sarimax_visualize import SARIMAXVisualizer
from common.utils import find_input_for_variable


def run_command(cmd: list[str], description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed!")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    print(f"✅ {description} completed!\n")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Run full SARIMAX pipeline")
    parser.add_argument("--variable", required=True, 
                       help="Target variable (e.g., corriente_motor_a)")
    parser.add_argument("--input", default="data/raw/DESF/DATOS_DESF_CLEANED.csv",
                       help="Input CSV file")
    parser.add_argument("--steps", type=int, default=24,
                       help="Forecast horizon (hours)")
    parser.add_argument("--order", nargs=3, type=int, default=[1, 0, 1],
                       help="SARIMAX order (p d q)")
    parser.add_argument("--seasonal", nargs=4, type=int, default=[0, 0, 0, 0],
                       help="Seasonal order (P D Q s)")
    parser.add_argument("--resample", default="1h",
                       help="Resample rule (e.g., 1h, 30min)")
    parser.add_argument("--skip-prep", action="store_true",
                       help="Skip preprocessing (use existing processed data)")
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip training (use existing model)")
    
    args = parser.parse_args()

    # Auto-detect input CSV when default is used
    default_input = "data/raw/DESF/DATOS_DESF_CLEANED.csv"
    if args.input == default_input:
        selected = find_input_for_variable(args.variable, data_dir="data")
        if selected is None:
            print(f"❌ Target '{args.variable}' not found in any CSV under data/.")
            print("   Use --input to specify the CSV containing this variable.")
            sys.exit(1)
        selected = str(selected)
        if selected != default_input:
            print(f"🔍 Target '{args.variable}' not found in default input {default_input}.")
            print(f"   Using {selected} instead.")
            args.input = selected
    else:
        # Validate that explicit input contains the requested variable
        try:
            cols = pd.read_csv(args.input, nrows=0).columns
            cols = [c.strip().lower() for c in cols]
            if args.variable not in cols:
                print(f"❌ The specified input {args.input} does not contain variable '{args.variable}'.")
                print("   Use --input path/to/other.csv or omit --input to auto-detect.")
                sys.exit(1)
        except Exception as _e:
            print(f"❌ Could not read input file {args.input}: {_e}")
            sys.exit(1)
    
    # Setup paths
    processed_dir = Path("processed")
    models_dir = Path("models")
    forecasts_dir = Path("forecasts")
    reports_dir = Path("reports")
    
    processed_file = processed_dir / f"{args.variable}.parquet"
    meta_file = processed_dir / f"{args.variable}_meta.json"
    model_file = models_dir / f"{args.variable}_sarimax.pkl"
    forecast_csv = forecasts_dir / f"{args.variable}_forecast.csv"
    
    print("\n" + "="*70)
    print("🎯 SARIMAX FULL PIPELINE")
    print("="*70)
    print(f"Variable: {args.variable}")
    print(f"Forecast Steps: {args.steps} hours")
    print(f"SARIMAX Order: {args.order}")
    print(f"Seasonal Order: {args.seasonal}")
    print(f"Resample Rule: {args.resample}")
    print("="*70 + "\n")
    
    # Step 1: Preprocess data
    if not args.skip_prep:
        script_root = Path(__file__).resolve().parent
        prep_cmd = [
            sys.executable, str(script_root / "sarimax" / "sarimax_prep.py"),
            "--input", args.input,
            "--target", args.variable,
            "--resample", args.resample,
            "--outdir", "processed"
        ]
        run_command(prep_cmd, "Step 1: Data Preprocessing")
    else:
        print(f"⏭️  Skipping preprocessing (using {processed_file})\n")
    
    # Verify preprocessed data exists
    if not processed_file.exists():
        print(f"❌ Error: Processed file not found: {processed_file}")
        print("Run without --skip-prep to generate it.")
        sys.exit(1)
    
    # Step 2: Train SARIMAX model
    if not args.skip_train:
        train_cmd = [
            sys.executable, str(script_root / "sarimax" / "sarimax_train.py"),
            "--data", str(processed_file),
            "--order", *[str(x) for x in args.order],
            "--seasonal", *[str(x) for x in args.seasonal],
            "--outdir", "models"
        ]
        run_command(train_cmd, "Step 2: SARIMAX Model Training")
    else:
        print(f"⏭️  Skipping training (using {model_file})\n")
    
    # Verify model exists
    if not model_file.exists():
        print(f"❌ Error: Model file not found: {model_file}")
        print("Run without --skip-train to generate it.")
        sys.exit(1)
    
    # Step 3: Generate forecasts
    forecast_cmd = [
        sys.executable, str(script_root / "sarimax" / "sarimax_forecast.py"),
        "--model", str(model_file),
        "--data", str(processed_file),
        "--meta", str(meta_file),
        "--h", str(args.steps),
        "--outdir", "forecasts"
    ]
    run_command(forecast_cmd, "Step 3: Forecast Generation")
    
    # Verify forecast exists
    if not forecast_csv.exists():
        print(f"❌ Error: Forecast file not found: {forecast_csv}")
        sys.exit(1)
    
    # Step 4: Advanced visualization with sarimax_visualize.py
    print(f"\n{'='*70}")
    print("🎨 Step 4: Advanced Visualization")
    print(f"{'='*70}\n")
    
    # Load data
    train_df = pd.read_parquet(processed_file)
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    
    # Load forecast and convert column names
    forecast_raw = pd.read_csv(forecast_csv)
    forecast_df = pd.DataFrame({
        'timestamp': pd.to_datetime(forecast_raw['timestamp']),
        'y_pred': forecast_raw['yhat'],
        'lower_ci': forecast_raw['yhat_lo'],
        'upper_ci': forecast_raw['yhat_hi']
    })
    
    print(f"📊 Creating comprehensive visualizations...")
    print(f"   Training samples: {len(train_df):,}")
    print(f"   Forecast points: {len(forecast_df)}")
    print(f"   Variable: {meta.get('variable', args.variable)}")
    print(f"   Unit: {meta.get('unidad', 'N/A')}\n")
    
    # Create visualizer
    viz = SARIMAXVisualizer(train_df, meta, forecast_df)
    
    # Generate all visualizations
    reports_dir.mkdir(exist_ok=True)
    
    # 1. Forecast plot
    forecast_plot = reports_dir / f"{args.variable}_forecast.png"
    viz.plot_forecast(
        steps=args.steps,
        include_history=168,  # 1 week
        save_path=str(forecast_plot)
    )
    
    # 2. Dashboard
    dashboard_plot = reports_dir / f"{args.variable}_dashboard.png"
    viz.create_dashboard(save_path=str(dashboard_plot))
    
    # 3. Interactive dashboard (if plotly available)
    try:
        interactive_dash = reports_dir / f"{args.variable}_interactive.html"
        viz.create_interactive_dashboard(save_path=str(interactive_dash))
    except ImportError:
        print("⚠️  Plotly not available - skipping interactive dashboard")
    
    # 4. Seasonal decomposition
    try:
        decomp_plot = reports_dir / f"{args.variable}_decomposition.png"
        viz.plot_seasonal_decomposition(period=24, save_path=str(decomp_plot))
    except Exception as e:
        print(f"⚠️  Seasonal decomposition skipped: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("✅ PIPELINE COMPLETE!")
    print(f"{'='*70}\n")
    
    print("📁 Output Files:")
    print(f"   Preprocessed Data: {processed_file}")
    print(f"   Metadata: {meta_file}")
    print(f"   Trained Model: {model_file}")
    print(f"   Forecast CSV: {forecast_csv}")
    print(f"   Forecast Plot: {forecast_plot}")
    print(f"   Dashboard: {dashboard_plot}")
    if (reports_dir / f"{args.variable}_interactive.html").exists():
        print(f"   Interactive Dashboard: {reports_dir / f'{args.variable}_interactive.html'}")
    if (reports_dir / f"{args.variable}_decomposition.png").exists():
        print(f"   Decomposition: {reports_dir / f'{args.variable}_decomposition.png'}")
    
    print(f"\n💡 View interactive dashboard:")
    print(f"   start {reports_dir / f'{args.variable}_interactive.html'}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
