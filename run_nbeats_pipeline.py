#!/usr/bin/env python
"""
N-BEATS Pipeline Runner

Runs the complete N-BEATS forecasting pipeline from scratch:
1. Train models for all variables (batch training)
2. Generate forecasts for all trained models

Prerequisites:
- Performance pipeline must be run first (DATOS_EPI_*_HOURLY.csv files must exist)
- Raw sensor data must exist (DATOS_DESF_CLEANED.csv, DATOS_PICADORA_CLEANED.csv)

Usage:
    # Train and forecast all machines
    python run_nbeats_pipeline.py
    
    # Train and forecast specific machine
    python run_nbeats_pipeline.py --machine PICADORA
    python run_nbeats_pipeline.py --machine DESF
    
    # Skip training (use existing models)
    python run_nbeats_pipeline.py --skip-train
    
    # Force retrain all models
    python run_nbeats_pipeline.py --force
    
    # Custom forecast horizon
    python run_nbeats_pipeline.py --horizon 5_days
    
    # Custom training parameters
    python run_nbeats_pipeline.py --epochs 150 --model-size large
    
    # Reproducible results with seed
    python run_nbeats_pipeline.py --seed 42
"""

from __future__ import annotations
import argparse
import random
import sys
import time
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Fix Windows console encoding
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')


def set_global_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  Random seed set to: {seed}")


def check_prerequisites() -> dict:
    """Check if required data files exist."""
    data_dir = Path("data")
    
    # Check for raw data folders or legacy files
    desf_raw = data_dir / "raw" / "DESF"
    picadora_raw = data_dir / "raw" / "PICADORA"
    
    desf_has_data = (desf_raw.exists() and any(desf_raw.glob("*.csv"))) or \
                    (data_dir / "DATOS_DESF_CLEANED.csv").exists()
                    
    picadora_has_data = (picadora_raw.exists() and any(picadora_raw.glob("*.csv"))) or \
                        (data_dir / "DATOS_PICADORA_CLEANED.csv").exists()
    
    requirements = {
        'DESF': {
            'sensor_ready': desf_has_data,
            'epi_data': data_dir / "epi" / "DESF" / "DATOS_EPI_DESF_HOURLY.csv"
        },
        'PICADORA': {
            'sensor_ready': picadora_has_data,
            'epi_data': data_dir / "epi" / "PICADORA" / "DATOS_EPI_PICADORA_HOURLY.csv"
        },
        'PLANT': {
            'sensor_ready': True,  # PLANT only has EPI
            'epi_data': data_dir / "epi" / "PLANT" / "DATOS_EPI_PLANT_HOURLY.csv"
        }
    }
    
    status = {}
    for machine, info in requirements.items():
        status[machine] = {
            'sensor_ready': info['sensor_ready'],
            'epi_ready': info['epi_data'].exists() if info['epi_data'] else False
        }
        status[machine]['ready'] = status[machine]['sensor_ready'] and status[machine]['epi_ready']
    
    return status


def print_status(status: dict):
    """Print prerequisite status."""
    print("\n" + "="*70)
    print("DATA PREREQUISITES CHECK")
    print("="*70)
    
    all_ready = True
    for machine, info in status.items():
        icon = "[OK]" if info['ready'] else "[X]"
        print(f"  {icon} {machine}:")
        sensor_status = "Ready" if info['sensor_ready'] else "Missing"
        epi_status = "Ready" if info['epi_ready'] else "Missing"
        print(f"      Sensor data: {sensor_status}")
        print(f"      EPI data:    {epi_status}")
        if not info['ready']:
            all_ready = False
    
    print("="*70)
    
    if not all_ready:
        print("\n[WARN] Some data files are missing!")
        print("   Run the performance pipeline first:")
        print("   python performance/run_performance_pipeline.py")
        print()
    
    return all_ready


def run_data_preparation(machine: str, aggregation: str = "mean") -> bool:
    """Run data preparation for a machine."""
    from nbeats.nbeats_train import prepare_all_data_for_machine
    
    print(f"\n{'='*70}")
    print(f"PREPARING DATA: {machine}")
    print(f"{'='*70}")
    
    try:
        prepare_all_data_for_machine(machine, aggregation=aggregation)
        return True
    except Exception as e:
        print(f"  ❌ Data preparation failed: {str(e)}")
        return False


def run_batch_training(
    machine: str,
    epochs: int = 100,
    model_size: str = "medium",
    force: bool = False,
    patience: int = 10,
    aggregation: str = "mean"
) -> bool:
    """Run batch training for a machine."""
    from nbeats.nbeats_train import batch_train_machine
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {machine}")
    print(f"{'='*70}")
    print(f"  Epochs: {epochs}")
    print(f"  Model Size: {model_size}")
    print(f"  Force Retrain: {force}")
    print(f"  Aggregation: {aggregation}")
    print(f"{'='*70}\n")
    
    try:
        results = batch_train_machine(
            machine=machine,
            include_epi=True,
            epochs=epochs,
            model_size=model_size,
            patience=patience,
            skip_existing=not force,
            aggregation=aggregation
        )
        
        n_success = len(results['successful'])
        n_failed = len(results['failed'])
        n_skipped = len(results['skipped'])
        
        print(f"\n  Training Summary for {machine}:")
        print(f"    Successful: {n_success}")
        print(f"    Failed:     {n_failed}")
        print(f"    Skipped:    {n_skipped}")
        
        return n_failed == 0
        
    except Exception as e:
        print(f"  ❌ Training failed: {str(e)}")
        return False


def run_batch_forecast(horizon: str = "2_days") -> bool:
    """Run batch forecast for all machines."""
    from nbeats.nbeats_forecast import batch_forecast_to_data
    
    print(f"\n{'='*70}")
    print(f"FORECASTING: All Machines")
    print(f"{'='*70}")
    print(f"  Horizon: {horizon}")
    print(f"{'='*70}\n")
    
    try:
        batch_forecast_to_data(horizon_preset=horizon)
        return True
    except Exception as e:
        print(f"  ❌ Forecasting failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="N-BEATS Pipeline: Train and Forecast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (train + forecast) for all machines
    python run_nbeats_pipeline.py
    
    # Specific machine only
    python run_nbeats_pipeline.py --machine PICADORA
    
    # Skip training, just forecast
    python run_nbeats_pipeline.py --skip-train --horizon 5_days
    
    # Force retrain all models
    python run_nbeats_pipeline.py --force --epochs 150
    
    # Large models with more epochs
    python run_nbeats_pipeline.py --model-size large --epochs 200
        """
    )
    
    # Machine selection
    parser.add_argument(
        "--machine",
        choices=["DESF", "PICADORA", "PLANT", "ALL"],
        default="ALL",
        help="Machine to process (default: ALL)"
    )
    
    # Training options
    parser.add_argument("--skip-train", action="store_true",
                       help="Skip training, use existing models")
    parser.add_argument("--force", action="store_true",
                       help="Force retrain all models")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Training epochs (default: 100)")
    parser.add_argument("--model-size", choices=["small", "medium", "large"],
                       default="medium", help="Model size (default: medium)")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience (default: 10)")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean",
                       help="Aggregation method for hourly data (default: mean)")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility (default: None = random)")
    
    # Forecast options
    parser.add_argument("--skip-forecast", action="store_true",
                       help="Skip forecasting, only train")
    parser.add_argument("--horizon", default="2_days",
                       choices=["2_days", "5_days", "15_days", "1_month"],
                       help="Forecast horizon (default: 2_days)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility if specified
    if args.seed is not None:
        set_global_seed(args.seed)
    
    # Print header
    print("\n" + "="*70)
    print("N-BEATS PIPELINE")
    print("="*70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Machine: {args.machine}")
    print(f"  Skip Training: {args.skip_train}")
    print(f"  Skip Forecast: {args.skip_forecast}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Aggregation: {args.agg}")
    print(f"  Seed: {args.seed if args.seed is not None else 'Random'}")
    print("="*70)
    
    start_time = time.time()
    
    # Check prerequisites
    status = check_prerequisites()
    
    # Determine which machines to process
    if args.machine == "ALL":
        machines = ["DESF", "PICADORA", "PLANT"]
    else:
        machines = [args.machine]
    
    # Filter to only ready machines
    machines_to_process = []
    for machine in machines:
        if status[machine]['ready']:
            machines_to_process.append(machine)
        else:
            print(f"[SKIP] {machine} - data not ready")
    
    if not machines_to_process:
        print("\n[ERROR] No machines ready to process!")
        print("   Run performance pipeline first:")
        print("   python performance/run_performance_pipeline.py")
        sys.exit(1)
    
    # Data Preparation Phase
    print("\n" + "="*70)
    print("PHASE 0: DATA PREPARATION")
    print("="*70)
    
    for machine in machines_to_process:
        # PLANT doesn't need preparation as it only has EPI data which is already prepared
        if machine == "PLANT":
            continue
            
        success = run_data_preparation(machine, aggregation=args.agg)
        if not success:
            print(f"[WARN] Data preparation failed for {machine}")

    # Training phase
    if not args.skip_train:
        print("\n" + "="*70)
        print("PHASE 1: TRAINING")
        print("="*70)
        
        for machine in machines_to_process:
            success = run_batch_training(
                machine=machine,
                epochs=args.epochs,
                model_size=args.model_size,
                force=args.force,
                patience=args.patience,
                aggregation=args.agg
            )
            if not success:
                print(f"[WARN] Some models failed for {machine}")
    else:
        print("\n[SKIP] Skipping training phase")
    
    # Forecasting phase
    if not args.skip_forecast:
        print("\n" + "="*70)
        print("PHASE 2: FORECASTING")
        print("="*70)
        
        success = run_batch_forecast(horizon=args.horizon)
        if not success:
            print("[WARN] Forecasting had errors")
    else:
        print("\n[SKIP] Skipping forecast phase")
    
    # Summary
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"  Elapsed Time: {elapsed/60:.1f} minutes")
    print(f"  Machines Processed: {', '.join(machines_to_process)}")
    
    if not args.skip_forecast:
        print(f"\n  Output Files (in data/forecasts/):")
        for machine in machines_to_process:
            print(f"    • FORECAST_{machine}_{args.horizon.upper()}.csv")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
