"""
Clean up and retrain all sensor models with improved MEAN aggregation.

The original models were trained on hourly MAX values, which are too volatile
for N-BEATS to learn meaningful patterns. This script:
1. Removes old processed parquet files
2. Removes old sensor models (keeps EPI models)
3. Retrains all sensor variables with MEAN aggregation

Run: python fix_sensor_models.py
"""

import os
import shutil
from pathlib import Path

def main():
    print("=" * 70)
    print("FIX SENSOR MODELS - Switch from MAX to MEAN aggregation")
    print("=" * 70)
    
    # Step 1: Clean up processed parquet files (sensor data only)
    print("\n[1/3] Cleaning up processed parquet files...")
    for machine in ["DESF", "PICADORA"]:
        processed_dir = Path(f"processed/{machine}")
        if processed_dir.exists():
            count = 0
            for f in processed_dir.glob("*.parquet"):
                # Keep EPI file if it exists
                if f.stem.lower() != "epi":
                    f.unlink()
                    count += 1
            print(f"  {machine}: Removed {count} parquet files")
    
    # Step 2: Remove sensor models (keep EPI)
    print("\n[2/3] Removing old sensor models...")
    for machine in ["DESF", "PICADORA"]:
        model_dir = Path(f"models/{machine}")
        if model_dir.exists():
            count = 0
            for f in model_dir.glob("*_nbeats.pth"):
                # Keep EPI model
                if f.stem.lower() != "epi_nbeats":
                    f.unlink()
                    count += 1
            print(f"  {machine}: Removed {count} model files")
    
    # Step 3: Retrain all sensor variables
    print("\n[3/3] Retraining sensor models with MEAN aggregation...")
    print("  This may take 30-60 minutes depending on your hardware.")
    print()
    
    # Import and run batch training
    from nbeats.nbeats_train import batch_train_all
    
    results = batch_train_all(
        machines=['DESF', 'PICADORA'],
        include_epi=False,  # EPI models are already good
        epochs=150,         # More epochs for better convergence
        model_size='medium',
        patience=15,        # More patience for complex patterns
        skip_existing=False # Force retrain everything
    )
    
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run forecasts: python -m nbeats.nbeats_forecast --batch --horizon 2_days")
    print("  2. Check model losses: python check_losses.py")
    print()

if __name__ == "__main__":
    main()
