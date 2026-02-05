"""
Consolidate PLANT EPI Data

This script consolidates all individual plant_performance_*.csv files
in data/epi/PLANT/ into the main DATOS_EPI_PLANT_HOURLY.csv file.

This fixes the date alignment issue where the forecast was using outdated
data that ended in December instead of the latest January data.
"""

import pandas as pd
from pathlib import Path
from typing import Optional


def consolidate_plant_epi(
    base_dir: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Consolidate all plant_performance_*.csv files into DATOS_EPI_PLANT_HOURLY.csv.
    
    Args:
        base_dir: Project base directory. Defaults to detecting from script location.
        verbose: Print progress messages.
        
    Returns:
        The consolidated DataFrame.
    """
    if base_dir is None:
        # Detect project root from script location
        base_dir = Path(__file__).resolve().parent.parent
    else:
        base_dir = Path(base_dir)
    
    epi_plant_dir = base_dir / "data" / "epi" / "PLANT"
    output_file = epi_plant_dir / "DATOS_EPI_PLANT_HOURLY.csv"
    
    if not epi_plant_dir.exists():
        raise FileNotFoundError(f"Plant EPI directory not found: {epi_plant_dir}")
    
    if verbose:
        print(f"{'='*60}")
        print("CONSOLIDATING PLANT EPI DATA")
        print(f"{'='*60}")
        print(f"  Source directory: {epi_plant_dir}")
        print(f"  Output file: {output_file}")
    
    # Find all individual performance files
    # Pattern: plant_performance_*.csv (excludes DATOS_EPI_*.csv)
    performance_files = sorted(epi_plant_dir.glob("plant_performance*.csv"))
    
    if not performance_files:
        print("  No plant_performance_*.csv files found!")
        return pd.DataFrame()
    
    if verbose:
        print(f"  Found {len(performance_files)} individual performance files")
    
    # Load and concatenate all files
    all_dfs = []
    
    for f in performance_files:
        try:
            df = pd.read_csv(f)
            
            # Standardize column names
            # Expected: timestamp, y (or EPI variants)
            if 'timestamp' not in df.columns:
                # Try to find timestamp-like column
                ts_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
                if ts_cols:
                    df = df.rename(columns={ts_cols[0]: 'timestamp'})
            
            if 'y' not in df.columns:
                # Look for EPI or PPI columns
                value_cols = [c for c in df.columns if any(x in c.lower() for x in ['epi', 'ppi', 'performance', 'value'])]
                if value_cols:
                    df = df.rename(columns={value_cols[0]: 'y'})
            
            # Ensure we have the required columns
            if 'timestamp' not in df.columns or 'y' not in df.columns:
                if verbose:
                    print(f"    Skipping {f.name}: missing required columns (has: {list(df.columns)})")
                continue
            
            # Keep only timestamp and y
            df = df[['timestamp', 'y']].copy()
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            all_dfs.append(df)
            
            if verbose:
                print(f"    Loaded: {f.name} ({len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()})")
                
        except Exception as e:
            if verbose:
                print(f"    Error loading {f.name}: {e}")
    
    if not all_dfs:
        print("  No valid data loaded!")
        return pd.DataFrame()
    
    # Concatenate all
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (keep first occurrence, assumes later files have better data)
    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    # Resample to hourly to ensure consistent frequency
    combined_df = combined_df.set_index('timestamp')
    combined_df = combined_df.resample('1h').mean()
    combined_df = combined_df.dropna().reset_index()
    
    if verbose:
        print(f"\n  Combined data:")
        print(f"    Total rows: {len(combined_df)}")
        print(f"    Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    # Also load existing DATOS_EPI_PLANT_HOURLY.csv and merge
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], utc=True)
            
            if verbose:
                print(f"\n  Existing DATOS_EPI_PLANT_HOURLY.csv:")
                print(f"    Rows: {len(existing_df)}")
                print(f"    Date range: {existing_df['timestamp'].min()} to {existing_df['timestamp'].max()}")
            
            # Combine: existing + new, with new taking precedence
            all_data = pd.concat([existing_df, combined_df], ignore_index=True)
            all_data = all_data.drop_duplicates(subset=['timestamp'], keep='last')
            all_data = all_data.sort_values('timestamp').reset_index(drop=True)
            
            combined_df = all_data
            
            if verbose:
                print(f"\n  After merging with existing:")
                print(f"    Total rows: {len(combined_df)}")
                print(f"    Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
                
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not load existing file: {e}")
    
    # Resample again to ensure hourly frequency
    combined_df = combined_df.set_index('timestamp')
    combined_df = combined_df.resample('1h').mean()
    combined_df = combined_df.dropna().reset_index()
    
    # Save
    combined_df.to_csv(output_file, index=False)
    
    if verbose:
        print(f"\n  Saved consolidated file: {output_file}")
        print(f"    Final rows: {len(combined_df)}")
        print(f"    Final date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
        print(f"{'='*60}")
    
    return combined_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Consolidate PLANT EPI data files")
    parser.add_argument("--base-dir", help="Project base directory")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_args()
    
    consolidate_plant_epi(
        base_dir=args.base_dir,
        verbose=not args.quiet
    )
