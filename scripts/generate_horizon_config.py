import json
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_DIR = PROJECT_ROOT / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

def get_data_summary():
    """Get summary of available data across all sources."""
    summary = {}
    
    # Check processed files for each machine
    for machine in ["DESF", "PICADORA", "PLANT"]:
        machine_dir = PROCESSED_DIR / machine
        if machine_dir.exists():
            # Find the file with the most samples to represent the machine
            max_samples = 0
            freq_hours = 1.0
            
            files = list(machine_dir.glob("*.parquet"))
            for f in files:
                try:
                    # Skip intermediate clean files if you want only the final ones
                    # or just read them all and take the max
                    df = pd.read_parquet(f)
                    if len(df) > max_samples:
                        max_samples = len(df)
                        if 'timestamp' in df.columns and len(df) > 1:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                            freq_hours = df['timestamp'].diff().median().total_seconds() / 3600
                except Exception:
                    continue
            
            if max_samples > 0:
                summary[machine] = {
                    "n_samples": max_samples,
                    "freq_hours": freq_hours
                }
    return summary

def calculate_horizon_options(n_samples, freq_hours=1.0):
    """
    Calculate horizon options based on sample count.
    Returns a list of strings formatted for the UI.
    """
    # Standard intervals we want to offer
    intervals = [
        ("2_days", 48),
        ("5_days", 120),
        ("15_days", 360),
        ("1_month", 720)
    ]
    
    options = []
    
    for label, hours in intervals:
        steps = int(hours / freq_hours)
        # Calculate ratio of horizon to history
        ratio = steps / n_samples if n_samples > 0 else 1.0
        
        # Determine confidence based on ratio
        if ratio <= 0.10:
            precision = "~85-90%"
            conf = "ALTA confiabilidad"
        elif ratio <= 0.15:
            precision = "~75-85%"
            conf = "ALTA-MEDIA confiabilidad"
        elif ratio <= 0.25:
            precision = "~65-75%"
            conf = "MEDIA confiabilidad"
        elif ratio <= 0.40:
            precision = "~50-65%"
            conf = "BAJA confiabilidad"
        else:
            precision = "<50%"
            conf = "solo TENDENCIA"
            
        option_str = f"{label} ({hours}h, {precision} precisión, {conf})"
        options.append(option_str)
        
    return options

def main():
    print("Generating horizon configuration...")
    summary = get_data_summary()
    
    config = {}
    
    # Generate options for each machine
    for machine, data in summary.items():
        print(f"Processing {machine}: {data['n_samples']} samples")
        config[machine] = calculate_horizon_options(data['n_samples'], data['freq_hours'])
        
    # Generate a default based on the max data available
    if summary:
        max_samples = max(d['n_samples'] for d in summary.values())
        config["default"] = calculate_horizon_options(max_samples)
    else:
        # Fallback if no data found
        config["default"] = [
            "2_days (48h, ~85% precisión, ALTA confiabilidad)",
            "5_days (120h, ~72% precisión, MEDIA confiabilidad)",
            "15_days (360h, ~55% precisión, BAJA confiabilidad)",
            "1_month (720h, ~42% precisión, solo TENDENCIA)"
        ]
        
    output_path = MODELS_DIR / "horizon_config.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        
    print(f"Configuration saved to {output_path}")

if __name__ == "__main__":
    main()
