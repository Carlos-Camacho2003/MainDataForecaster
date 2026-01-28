#!/usr/bin/env python
"""
N-BEATS Forecasting Pipeline

A secure, unified pipeline for time series forecasting using N-BEATS exclusively.
All variables are aggregated by hourly maximum values for predictive maintenance.

Features:
- N-BEATS deep learning model for time series forecasting
- Hourly maximum aggregation for all variables
- Secure input validation and sanitization
- Support for DESFIBRADORA and PICADORA machines
- Comprehensive metadata and logging

Usage:
    # Basic usage
    python run_pipeline.py --variable corriente_motor_a --machine DESF
    
    # With custom parameters
    python run_pipeline.py --variable velocidad_chum_la_h --machine PICADORA --steps 48
    
    # Skip training (use existing model)
    python run_pipeline.py --variable corriente_motor_a --machine DESF --skip-train
"""

from __future__ import annotations
import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from common.utils import find_input_for_variable, validate_forecast_csv
except ImportError:
    # Fallback for direct execution - try relative import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "utils", 
            Path(__file__).parent / "common" / "utils.py"
        )
        utils_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils_module)
        find_input_for_variable = utils_module.find_input_for_variable
        validate_forecast_csv = utils_module.validate_forecast_csv
    except Exception:
        # Define minimal fallback functions
        def find_input_for_variable(variable, data_dir="data"):
            import pandas as pd
            data_path = Path(data_dir)
            for f in data_path.glob("*.csv"):
                try:
                    cols = pd.read_csv(f, nrows=0).columns
                    if variable.lower() in [c.lower() for c in cols]:
                        return f
                except Exception:
                    pass
            return None
        
        def validate_forecast_csv(csv_path):
            return Path(csv_path).exists()


def run_command(cmd: list[str], description: str) -> str:
    """
    Run a command and handle errors securely.
    
    Args:
        cmd: Command and arguments to run
        description: Human-readable description of the command
    
    Returns:
        Command stdout
    
    Raises:
        SystemExit: If command fails
    """
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")
    
    # Use subprocess with security options
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour timeout
        env=None  # Use parent environment
    )
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed!")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    print(f"✅ {description} completed!\n")
    return result.stdout


def check_pytorch_available() -> bool:
    """Check if PyTorch is available in the current environment."""
    result = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("\n❌ PyTorch is not installed in the current environment.")
        print("   Install PyTorch: pip install torch")
        print("   Or install all requirements: pip install -r requirements.txt")
        return False
    print(f"✅ PyTorch version: {result.stdout.strip()}")
    return True


def validate_input_file(input_path: str, variable: str) -> bool:
    """
    Validate that the input file exists and contains the target variable.
    
    Args:
        input_path: Path to input CSV file
        variable: Target variable name
    
    Returns:
        True if valid, False otherwise
    """
    import pandas as pd
    
    path = Path(input_path)
    if not path.exists():
        print(f"❌ Input file not found: {input_path}")
        return False
    
    try:
        cols = pd.read_csv(input_path, nrows=0).columns
        cols_lower = [c.strip().lower() for c in cols]
        
        if variable.lower() not in cols_lower:
            print(f"❌ Variable '{variable}' not found in {input_path}")
            print(f"   Available columns: {', '.join(cols_lower[:10])}...")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Could not read input file: {e}")
        return False


def run_nbeats_pipeline(args, machine_type: str) -> Dict[str, Any]:
    """
    Run the complete N-BEATS pipeline.
    
    Args:
        args: Parsed command line arguments
        machine_type: Machine type ('DESF' or 'PICADORA')
    
    Returns:
        Dictionary with pipeline results
    """
    script_root = Path(__file__).resolve().parent
    
    # Setup paths
    processed_dir = Path("processed") / machine_type
    models_dir = Path("models") / machine_type
    forecasts_dir = Path("forecasts") / machine_type
    
    processed_file = processed_dir / f"{args.variable}.parquet"
    meta_file = processed_dir / f"{args.variable}_meta.json"
    model_file = models_dir / f"{args.variable}_nbeats.pth"
    
    print("\n" + "="*70)
    print(f"🎯 N-BEATS FORECASTING PIPELINE")
    print("="*70)
    print(f"Machine: {machine_type}")
    print(f"Variable: {args.variable}")
    print(f"Forecast Steps: {args.steps} hours")
    print(f"Model Size: {args.model_size}")
    print(f"Aggregation: Maximum per hour")
    print(f"Lookback: {args.lookback} hours")
    print("="*70 + "\n")
    
    results = {
        "machine_type": machine_type,
        "variable": args.variable,
        "steps": args.steps,
        "success": False,
        "files": {}
    }
    
    # Step 1: Data Preprocessing
    if not args.skip_prep:
        prep_cmd = [
            sys.executable, str(script_root / "common" / "data_prep.py"),
            "--input", args.input,
            "--target", args.variable,
            "--resample", args.resample,
            "--outdir", "processed",
            "--machine", machine_type
        ]
        run_command(prep_cmd, "Step 1: Data Preprocessing (MAX aggregation)")
    else:
        print(f"⏭️  Skipping preprocessing (using {processed_file})\n")
    
    # Verify preprocessed data exists
    if not processed_file.exists():
        print(f"❌ Error: Processed file not found: {processed_file}")
        print("   Run without --skip-prep to generate it.")
        return results
    
    results["files"]["processed"] = str(processed_file)
    results["files"]["metadata"] = str(meta_file)
    
    # Step 2: Train N-BEATS Model
    if not args.skip_train:
        if not check_pytorch_available():
            return results
        
        train_cmd = [
            sys.executable, str(script_root / "nbeats" / "nbeats_train.py"),
            "--data", str(processed_file),
            "--lookback", str(args.lookback),
            "--horizon", str(args.steps),
            "--model-size", args.model_size,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--patience", str(args.patience),
            "--outdir", "models"
        ]
        run_command(train_cmd, "Step 2: N-BEATS Model Training")
    else:
        print(f"⏭️  Skipping training (using {model_file})\n")
    
    # Verify model exists
    if not model_file.exists():
        print(f"❌ Error: Model file not found: {model_file}")
        print("   Run without --skip-train to generate it.")
        return results
    
    results["files"]["model"] = str(model_file)
    
    # Step 3: Generate Forecast
    forecast_cmd = [
        sys.executable, str(script_root / "nbeats" / "nbeats_forecast.py"),
        "--model", str(model_file),
        "--data", str(processed_file),
        "--meta", str(meta_file),
        "--steps", str(args.steps),
        "--mc-samples", str(args.mc_samples),
        "--outdir", "forecasts"
    ]
    run_command(forecast_cmd, "Step 3: N-BEATS Forecast Generation")
    
    # Verify forecast outputs
    forecast_csv = forecasts_dir / f"{args.variable}_nbeats_forecast.csv"
    summary_json = forecasts_dir / f"{args.variable}_nbeats_summary.json"
    forecast_png = forecasts_dir / f"{args.variable}_nbeats_forecast.png"
    
    if forecast_csv.exists():
        results["files"]["forecast_csv"] = str(forecast_csv)
    if summary_json.exists():
        results["files"]["summary"] = str(summary_json)
    if forecast_png.exists():
        results["files"]["forecast_plot"] = str(forecast_png)
    
    results["success"] = True
    
    # Print summary
    print("\n" + "="*70)
    print("✅ N-BEATS PIPELINE COMPLETE!")
    print("="*70)
    print("\n📁 Output Files:")
    for key, path in results["files"].items():
        print(f"   {key}: {path}")
    
    # Load and display alert status
    if summary_json.exists():
        with open(summary_json) as f:
            summary = json.load(f)
        
        print(f"\n{'='*70}")
        print("🔔 ALERT STATUS")
        print(f"{'='*70}")
        
        if summary.get("trigger_red"):
            print("🔴 RED ALERT: Critical threshold likely to be exceeded!")
            if summary.get("eta_critico_hours"):
                print(f"   ETA to critical: {summary['eta_critico_hours']} hours")
        elif summary.get("trigger_amber"):
            print("🟡 AMBER ALERT: Alarm threshold likely to be exceeded")
            if summary.get("eta_alarma_hours"):
                print(f"   ETA to alarm: {summary['eta_alarma_hours']} hours")
        else:
            print("🟢 OK: Normal operation expected")
        
        if summary.get("max_p_exceed_alarma"):
            print(f"   Max P(alarm): {summary['max_p_exceed_alarma']:.2%}")
        if summary.get("max_p_exceed_critico"):
            print(f"   Max P(critical): {summary['max_p_exceed_critico']:.2%}")
    
    print(f"\n{'='*70}\n")
    
    return results


def main():
    """Main entry point for the N-BEATS pipeline."""
    parser = argparse.ArgumentParser(
        description="N-BEATS Forecasting Pipeline for Predictive Maintenance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic DESFIBRADORA forecast
    python run_pipeline.py --variable corriente_motor_a --machine DESF
    
    # PICADORA with custom horizon
    python run_pipeline.py --variable corriente_picadora --machine PICADORA --steps 48
    
    # Skip preprocessing and training (use existing model)
    python run_pipeline.py --variable corriente_motor_a --machine DESF --skip-prep --skip-train
    
    # Custom model configuration
    python run_pipeline.py --variable velocidad_chum_la_h --machine DESF \\
        --model-size large --epochs 200 --lookback 336
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--variable",
        required=True,
        help="Target variable to forecast (e.g., corriente_motor_a)"
    )
    parser.add_argument(
        "--machine",
        choices=["DESF", "PICADORA"],
        required=True,
        help="Machine type"
    )
    
    # Input/Output
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV file (auto-detected if not provided)"
    )
    parser.add_argument(
        "--resample",
        default="1h",
        help="Resample rule (default: 1h)"
    )
    
    # Forecast parameters
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="Forecast horizon in hours (default: 24)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=168,
        help="Historical window for N-BEATS (default: 168 = 1 week)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="medium",
        help="N-BEATS model size (default: medium)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)"
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=100,
        help="Monte Carlo samples for uncertainty (default: 100)"
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip data preprocessing"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip model training"
    )
    
    args = parser.parse_args()
    
    # Auto-select input file based on machine type
    if args.input is None:
        machine_files = {
            "DESF": "data/raw/DESF/DATOS_DESF_CLEANED.csv",
            "PICADORA": "data/raw/PICADORA/DATOS_PICADORA_CLEANED.csv"
        }
        args.input = machine_files.get(args.machine)
        
        if not args.input or not Path(args.input).exists():
            # Try to auto-detect
            result = find_input_for_variable(args.variable, data_dir="data")
            if result is None:
                print(f"❌ No input file found for {args.machine}")
                print(f"   Use --input to specify a custom CSV file.")
                sys.exit(1)
            args.input = str(result)
        
        print(f"Using default input for {args.machine}: {args.input}")
    
    # Validate input file
    if not validate_input_file(args.input, args.variable):
        sys.exit(1)
    
    # Run the pipeline
    results = run_nbeats_pipeline(args, args.machine)
    
    if results["success"]:
        print("🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("❌ Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
