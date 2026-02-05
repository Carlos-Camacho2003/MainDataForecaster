"""
N-BEATS Training Script

Trains N-BEATS model on preprocessed time series data.

Usage:
    python nbeats_train.py --data processed/corriente_motor_a.parquet --lookback 168 --horizon 24
    
    # With anomaly cleaning (recommended, enabled by default)
    python nbeats_train.py --batch --machine DESF
    
    # Without anomaly cleaning
    python nbeats_train.py --batch --machine DESF --no-clean-anomalies
"""

from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    # Running as package
    from nbeats.nbeats_model import create_nbeats_model
except Exception:
    # Fallback for direct script execution
    from nbeats_model import create_nbeats_model

# Import anomaly detection module
try:
    from anomaly import AnomalyDetector, AnomalyConfig, AnomalyMethod
    HAS_ANOMALY = True
except ImportError:
    HAS_ANOMALY = False


def set_seed(seed: int) -> None:
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clean_data_anomalies(
    df: pd.DataFrame,
    column: str = "y",
    config: Optional["AnomalyConfig"] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Detect and interpolate anomalies in the data before training.
    
    Using cleaned data for training helps the model learn genuine patterns
    rather than being distorted by sensor errors, spikes, or outliers.
    
    Args:
        df: DataFrame with time series data
        column: Column containing values
        config: Optional anomaly detection config
    
    Returns:
        Tuple of (cleaned DataFrame, number of anomalies found)
    """
    if not HAS_ANOMALY:
        return df, 0
    
    if config is None:
        # Use conservative settings for training - we want to catch
        # obvious spikes/errors but preserve genuine trends
        config = AnomalyConfig(
            method=AnomalyMethod.COMBINED,
            zscore_threshold=3.5,
            modified_zscore_threshold=4.0,
            iqr_multiplier=1.5,
            rolling_window=24,
            rolling_std_multiplier=3.5,
            min_anomaly_votes=2,
            interpolation_method=AnomalyConfig().interpolation_method,
            max_consecutive_anomalies=24  # Don't interpolate gaps > 24h
        )
    
    detector = AnomalyDetector(config)
    
    # Detect anomalies
    df_detected, result = detector.detect(df, column=column, return_details=True)
    
    if result.n_anomalies == 0:
        return df, 0
    
    # Interpolate anomalies
    df_clean = detector.interpolate_anomalies(df_detected, column=column)
    
    # Replace original column with cleaned version
    df_clean[column] = df_clean[f"{column}_clean"]
    
    return df_clean, result.n_anomalies


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series windowing."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "y",
        exog_cols: Optional[list] = None,
        lookback: int = 168,
        horizon: int = 24,
        normalize: bool = True
    ):
        """
        Args:
            data: DataFrame with time series data
            target_col: Name of target column
            exog_cols: List of exogenous variable columns (optional)
            lookback: Historical window size
            horizon: Forecast horizon
            normalize: Whether to normalize data
        """
        self.data = data.copy()
        self.target_col = target_col
        self.exog_cols = exog_cols or []
        self.lookback = lookback
        self.horizon = horizon
        
        # Extract target series
        self.y = self.data[target_col].values
        
        # Normalization
        if normalize:
            self.y_mean = np.nanmean(self.y)
            self.y_std = np.nanstd(self.y)
            self.y = (self.y - self.y_mean) / (self.y_std + 1e-8)
        else:
            self.y_mean = 0.0
            self.y_std = 1.0
        
        # Handle NaN values
        self.y = np.nan_to_num(self.y, nan=0.0)
        
        # Calculate number of valid windows
        self.n_samples = len(self.y) - lookback - horizon + 1
        
        if self.n_samples <= 0:
            raise ValueError(
                f"Insufficient data: need at least {lookback + horizon} samples, "
                f"got {len(self.y)}"
            )
    
    def __len__(self) -> int:
        return max(0, self.n_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single training sample.
        
        Returns:
            x: Historical window (lookback,)
            y: Future values (horizon,)
        """
        x = self.y[idx : idx + self.lookback]
        y = self.y[idx + self.lookback : idx + self.lookback + self.horizon]
        
        return torch.FloatTensor(x), torch.FloatTensor(y)
    
    def denormalize(self, y: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale."""
        return y * self.y_std + self.y_mean


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    desc: str = "Training"
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=desc, leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_nbeats(
    data_path: str,
    lookback: int = 168,
    horizon: int = 24,
    model_size: str = "medium",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    patience: int = 10,
    outdir: str = "models",
    device: Optional[str] = None,
    clean_anomalies: bool = True,
    anomaly_config: Optional["AnomalyConfig"] = None,
    seed: Optional[int] = None
) -> dict:
    """
    Train N-BEATS model.
    
    Args:
        data_path: Path to preprocessed parquet file
        lookback: Historical window size
        horizon: Forecast horizon
        model_size: Model size ('small', 'medium', 'large')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
        patience: Early stopping patience
        outdir: Output directory for model
        device: Device ('cpu', 'cuda', or None for auto)
        clean_anomalies: Whether to detect and interpolate anomalies before training
        anomaly_config: Optional custom anomaly detection configuration
        seed: Random seed for reproducibility (default: None)
    
    Returns:
        Training metadata
    """
    # Set random seed if specified
    if seed is not None:
        set_seed(seed)
        print(f"Using seed: {seed}")
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load data (support both CSV and Parquet)
    print(f"Loading data from {data_path}...")
    data_path_obj = Path(data_path)
    if data_path_obj.suffix.lower() == '.csv':
        df = pd.read_csv(data_path)
    else:
        df = pd.read_parquet(data_path)
    print(f"  Loaded {len(df)} samples")
    
    # Detect and interpolate anomalies before training
    n_anomalies_cleaned = 0
    if clean_anomalies and HAS_ANOMALY:
        df, n_anomalies_cleaned = clean_data_anomalies(df, column="y", config=anomaly_config)
        if n_anomalies_cleaned > 0:
            print(f"  Anomaly cleaning: {n_anomalies_cleaned} anomalies interpolated")
    elif clean_anomalies and not HAS_ANOMALY:
        print("  Warning: Anomaly module not available, training on raw data")
    
    # Create datasets
    print(f"Creating datasets (lookback={lookback}, horizon={horizon})...")
    full_dataset = TimeSeriesDataset(
        df,
        target_col="y",
        lookback=lookback,
        horizon=horizon,
        normalize=True
    )
    
    # Train/validation split
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    
    # Use provided seed or default to 42 for backward compatibility
    split_seed = seed if seed is not None else 42
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed)
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    # Create model
    print(f"Creating N-BEATS model (size={model_size})...")
    model = create_nbeats_model(
        input_size=lookback,
        horizon=horizon,
        model_size=model_size
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            desc=f"Epoch {epoch+1}/{epochs}"
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.6f}, "
            f"Val Loss: {val_loss:.6f}"
        )
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            # Extract machine type from data path
            data_path_obj = Path(data_path)
            machine_type = 'UNKNOWN'
            if 'DESF' in str(data_path_obj).upper():
                machine_type = 'DESF'
            elif 'PICADORA' in str(data_path_obj).upper():
                machine_type = 'PICADORA'
            elif 'PLANT' in str(data_path_obj).upper():
                machine_type = 'PLANT'
            
            outdir_path = Path(outdir) / machine_type
            outdir_path.mkdir(parents=True, exist_ok=True)
            
            target_name = data_path_obj.stem
            model_path = outdir_path / f"{target_name}_nbeats.pth"
            
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "y_mean": full_dataset.y_mean,
                "y_std": full_dataset.y_std,
                "lookback": lookback,
                "horizon": horizon,
                "model_size": model_size,
                "trained_on_clean_data": clean_anomalies and HAS_ANOMALY,
                "anomalies_cleaned": int(n_anomalies_cleaned)
            }, model_path)
            
            print(f"  [OK] Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "epochs_trained": epoch + 1,
        "training_time_sec": training_time,
        "model_params": n_params,
        "lookback": lookback,
        "horizon": horizon,
        "model_size": model_size,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    }
    
    history_path = outdir_path / f"{target_name}_nbeats_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    
    print("\nSaved:")
    print(f"  Model: {model_path}")
    print(f"  History: {history_path}")
    
    return history


def get_machine_variables(machine: str) -> dict:
    """
    Get the data file path and trainable variables for a machine.
    
    Args:
        machine: 'DESF', 'PICADORA', or 'PLANT'
    
    Returns:
        dict with 'data_file', 'variables', and 'epi_file' keys
    """
    configs = {
        'DESF': {
            'data_file': 'data/raw/DESF/DATOS_DESF_CLEANED.csv',
            'epi_file': 'data/epi/DESF/DATOS_EPI_DESF_HOURLY.csv',
            'variables': [
                'aceleracion_chum_la_h', 'velocidad_chum_la_h', 'envolvente_chum_la_h',
                'aceleracion_chum_la_a', 'velocidad_chum_la_a', 'envolvente_chum_la_a',
                'aceleracion_chum_lb_h', 'velocidad_chum_lb_h', 'envolvente_chum_lb_h',
                'desbalance_chum_la', 'desalineacion_rad_chum_la', 'soltura_chum_la',
                'rodamiento_chum_la', 'desalineacion_ang_chum_la',
                'desbalance_chum_lb', 'desalineacion_chum_lb', 'soltura_chum_lb',
                'rodamiento_chum_lb',
                'corriente_motor_a', 'corriente_motor_b',
                'temp_chum_la', 'temp_chum_lb'
            ]
        },
        'PICADORA': {
            'data_file': 'data/raw/PICADORA/DATOS_PICADORA_CLEANED.csv',
            'epi_file': 'data/epi/PICADORA/DATOS_EPI_PICADORA_HOURLY.csv',
            'variables': [
                'aceleracion_chum_la_h', 'velocidad_chum_la_h', 'envolvente_chum_la_h',
                'aceleracion_chum_la_a', 'velocidad_chum_la_a', 'envolvente_chum_la_a',
                'aceleracion_chum_ll_h', 'velocidad_chum_ll_h', 'envolvente_chum_ll_h',
                'corriente_motor',
                'temp_chum_la', 'temp_chum_ll',
                'desbalance_chum_la', 'desalineacion_rad_chum_la', 'soltura_chum_la',
                'rodamiento_chum_la', 'desalineacion_ang_chum_la',
                'desbalance_chum_ll', 'soltura_chum_ll', 'rodamiento_chum_ll'
            ]
        },
        'PLANT': {
            'data_file': 'data/epi/PLANT/DATOS_EPI_PLANT_HOURLY.csv',
            'epi_file': 'data/epi/PLANT/DATOS_EPI_PLANT_HOURLY.csv',
            'variables': ['EPI']
        }
    }
    return configs.get(machine.upper(), {'data_file': None, 'epi_file': None, 'variables': []})


def consolidate_machine_data(machine: str) -> Optional[str]:
    """
    Consolidate all raw CSV files for a machine into a single file.
    
    Args:
        machine: Machine name ('DESF' or 'PICADORA')
        
    Returns:
        Path to consolidated CSV file, or None if no files found
    """
    import pandas as pd
    from pathlib import Path
    
    machine = machine.upper()
    base_dir = Path("data/raw") / machine
    
    if not base_dir.exists():
        # Try fallback to old location if raw subfolders don't exist
        if machine == 'DESF' and Path("data/DATOS_DESF_CLEANED.csv").exists():
            return "data/DATOS_DESF_CLEANED.csv"
        if machine == 'PICADORA' and Path("data/DATOS_PICADORA_CLEANED.csv").exists():
            return "data/DATOS_PICADORA_CLEANED.csv"
        # Don't warn for PLANT as it has no sensor data
        if machine != 'PLANT':
            print(f"  [WARN] Data directory not found: {base_dir}")
        return None
        
    # Find all CSV files
    csv_files = sorted(list(base_dir.glob("*.csv")))
    
    if not csv_files:
        print(f"  [WARN] No CSV files found in {base_dir}")
        return None
        
    print(f"  Found {len(csv_files)} raw data files in {base_dir}")
    
    dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Standardize column names
            df.columns = [col.lower() if col.lower() == 'maquina' else col for col in df.columns]
            df.columns = ['Maquina' if col == 'maquina' else col for col in df.columns]
            dfs.append(df)
        except Exception as e:
            print(f"    [WARN] Error reading {file_path.name}: {e}")
            
    if not dfs:
        return None
        
    # Concatenate
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Handle timestamp
    if 'timestamp' in df_combined.columns:
        df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
        df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
        df_combined = df_combined.drop_duplicates()
    
    # Save to processed folder
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = processed_dir / f"{machine}_FULL.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"  Consolidated data saved to: {output_path} ({len(df_combined)} rows)")
    
    return str(output_path)


def consolidate_epi_data(machine: str) -> Optional[str]:
    """
    Consolidate downloaded EPI CSV files for a machine into a single file.
    
    Args:
        machine: Machine name ('DESF', 'PICADORA', 'PLANT')
        
    Returns:
        Path to consolidated CSV file, or None if no files found
    """
    import pandas as pd
    from pathlib import Path
    
    machine = machine.upper()
    base_dir = Path("data/epi") / machine
    
    if not base_dir.exists():
        if machine != 'PLANT': # PLANT might strictly rely on this, others might have raw data
             print(f"  [WARN] EPI directory not found: {base_dir}")
        return None
        
    # Find all CSV files matching the download pattern
    # Exclude the "final" consolidated file to avoid self-inclusion loop
    # Pattern: *_performance_*.csv (e.g., desfibradora_performance_20260202.csv)
    all_csvs = list(base_dir.glob("*.csv"))
    epi_files = []
    
    consolidated_name = f"DATOS_EPI_{machine}_HOURLY.csv"
    
    for f in all_csvs:
        # Filter out the target file itself and other non-performance files
        if f.name == consolidated_name:
            continue
        if "_performance_" in f.name:
            epi_files.append(f)
            
    if not epi_files:
        # If no partial files found, check if the main file exists (maybe manually placed)
        main_file = base_dir / consolidated_name
        if main_file.exists():
            return str(main_file)
        return None
        
    print(f"  Found {len(epi_files)} EPI data files in {base_dir}")
    
    dfs = []
    for file_path in sorted(epi_files):
        try:
            df = pd.read_csv(file_path)
            # Ensure required columns
            if 'timestamp' not in df.columns or 'y' not in df.columns:
                # Check for alternative names if needed, but spec says 'timestamp,y'
                # Maybe 'EPI' column?
                if 'EPI' in df.columns:
                     df = df.rename(columns={'EPI': 'y'})
                
            if 'timestamp' in df.columns and 'y' in df.columns:
                dfs.append(df[['timestamp', 'y']])
            else:
                print(f"    [WARN] Skipping {file_path.name}: missing columns")
        except Exception as e:
            print(f"    [WARN] Error reading {file_path.name}: {e}")
            
    if not dfs:
        return None
        
    # Concatenate
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Handle timestamp normalization (mixed timezones in downloads)
    # Convert to UTC
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'], utc=True)
    
    # Sort and Deduplicate
    # usage: 'last' keeps the most recent download/record for that timestamp
    df_combined = df_combined.sort_values('timestamp')
    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Save to standard EPI file location
    output_path = base_dir / consolidated_name
    df_combined.to_csv(output_path, index=False)
    print(f"  Consolidated EPI data saved to: {output_path} ({len(df_combined)} rows)")
    
    return str(output_path)


def prepare_all_data_for_machine(
    machine: str,
    aggregation: str = "mean",
    include_epi: bool = True
) -> list:
    """
    Prepare all data for a machine (consolidate and create parquet files).
    
    Args:
        machine: Machine type
        aggregation: Aggregation method
        include_epi: Whether to include EPI variable
        
    Returns:
        List of dictionaries with variable info
    """
    import pandas as pd
    from pathlib import Path
    
    machine = machine.upper()
    config = get_machine_variables(machine)
    
    print(f"\n{'='*70}")
    print(f"DATA PREPARATION: {machine}")
    print(f"{'='*70}")
    
    # Consolidate data first
    consolidated_file = consolidate_machine_data(machine)
    if consolidated_file:
        print(f"  Using consolidated data: {consolidated_file}")
        config['data_file'] = consolidated_file
    
    variables_to_process = []
    
    # Add sensor variables if data file exists
    if config['data_file'] and Path(config['data_file']).exists():
        # Load data to verify which columns exist
        try:
            df = pd.read_csv(config['data_file'])
            available_cols = [c.lower() for c in df.columns if c.lower() not in ['timestamp', 'timestamp_grid', 'maquina']]
            
            for var in config['variables']:
                if var.lower() in available_cols:
                    variables_to_process.append({
                        'name': var,
                        'source': 'sensor',
                        'data_file': config['data_file']
                    })
                else:
                    print(f"  [WARN] Variable '{var}' not found in data file, skipping")
        except Exception as e:
            print(f"  [ERROR] Could not read data file {config['data_file']}: {e}")
    
    # Consolidate EPI data if requested
    if include_epi:
        print(f"  [EPI] Consolidating downloaded files...")
        consolidated_epi = consolidate_epi_data(machine)
        if consolidated_epi:
            print(f"  [EPI] Using consolidated data: {consolidated_epi}")
            config['epi_file'] = consolidated_epi
    
    # Prepare EPI data first if this machine has EPI
    if include_epi and config['epi_file'] and Path(config['epi_file']).exists():
        print(f"  [EPI] Preparing EPI data...")
        epi_parquet = prepare_epi_data(machine, config['epi_file'], aggregation)
        if epi_parquet:
            print(f"  [EPI] Prepared: {epi_parquet}")
    
    # Add EPI if requested (use parquet if available)
    if include_epi and config['epi_file'] and Path(config['epi_file']).exists():
        epi_parquet_path = Path('processed') / machine / 'EPI.parquet'
        if epi_parquet_path.exists():
            variables_to_process.append({
                'name': 'EPI',
                'source': 'epi',
                'data_file': str(epi_parquet_path)
            })
        else:
            variables_to_process.append({
                'name': 'EPI',
                'source': 'epi',
                'data_file': config['epi_file']
            })
    
    total_vars = len(variables_to_process)
    print(f"  Variables to process: {total_vars}")
    
    processed_vars = []
    
    for idx, var_info in enumerate(variables_to_process, 1):
        var_name = var_info['name']
        data_file = var_info['data_file']
        source = var_info['source']
        
        print(f"  [{idx}/{total_vars}] Preparing {var_name.upper()}...")
        
        try:
            if source == 'sensor':
                # Prepare hourly aggregated data for this variable
                prepared_data = prepare_variable_data(data_file, var_name, machine, aggregation=aggregation)
                if prepared_data:
                    var_info['train_data_path'] = prepared_data
                    processed_vars.append(var_info)
                else:
                    print(f"      [FAIL] Could not prepare data for {var_name}")
            else:
                # EPI data is already prepared
                var_info['train_data_path'] = data_file
                processed_vars.append(var_info)
                
        except Exception as e:
            print(f"      [FAIL] Error preparing data for {var_name}: {str(e)}")
            
    return processed_vars


def batch_train_machine(
    machine: str,
    include_epi: bool = True,
    lookback: int = 168,
    horizon: int = 24,
    model_size: str = "medium",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
    patience: int = 10,
    outdir: str = "models",
    device: Optional[str] = None,
    skip_existing: bool = True,
    aggregation: str = "mean",
    clean_anomalies: bool = True,
    seed: Optional[int] = None
) -> dict:
    """
    Train N-BEATS models for all variables of a specific machine.
    
    Args:
        machine: Machine type ('DESF', 'PICADORA', 'PLANT')
        include_epi: Whether to also train EPI model
        lookback: Historical window size
        horizon: Forecast horizon
        model_size: Model size ('small', 'medium', 'large')
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
        patience: Early stopping patience
        outdir: Output directory for models
        device: Device ('cpu', 'cuda', or None for auto)
        skip_existing: Skip variables that already have trained models
        aggregation: Hourly aggregation method ('mean' or 'max')
        clean_anomalies: Whether to detect and interpolate anomalies before training
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Summary of training results
    """
    import pandas as pd
    from pathlib import Path
    
    machine = machine.upper()
    config = get_machine_variables(machine)
    
    print(f"\n{'='*70}")
    print(f"BATCH TRAINING: {machine}")
    print(f"{'='*70}")
    if clean_anomalies:
        print(f"  Anomaly cleaning: ENABLED")
    
    # Consolidate data first
    consolidated_file = consolidate_machine_data(machine)
    if consolidated_file:
        print(f"  Using consolidated data: {consolidated_file}")
        config['data_file'] = consolidated_file
        
    # Consolidate EPI data if requested
    if include_epi:
        print(f"  [EPI] Consolidating downloaded files...")
        consolidated_epi = consolidate_epi_data(machine)
        if consolidated_epi:
            print(f"  [EPI] Using consolidated data: {consolidated_epi}")
            config['epi_file'] = consolidated_epi
    
    results = {
        'machine': machine,
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    # Determine output directory for this machine
    machine_outdir = Path(outdir) / machine
    machine_outdir.mkdir(parents=True, exist_ok=True)
    
    # Get list of variables to train
    variables_to_train = []
    
    # Add sensor variables if data file exists
    if config['data_file'] and Path(config['data_file']).exists():
        # Load data to verify which columns exist
        df = pd.read_csv(config['data_file'])
        available_cols = [c.lower() for c in df.columns if c.lower() not in ['timestamp', 'timestamp_grid', 'maquina']]
        
        for var in config['variables']:
            if var.lower() in available_cols:
                variables_to_train.append({
                    'name': var,
                    'source': 'sensor',
                    'data_file': config['data_file']
                })
            else:
                print(f"  [WARN] Variable '{var}' not found in data file, skipping")
    
    # Prepare EPI data first if requested (creates parquet for unified workflow)
    if include_epi and config['epi_file'] and Path(config['epi_file']).exists():
        print(f"  [EPI] Preparing EPI data...")
        epi_parquet = prepare_epi_data(machine, config['epi_file'], aggregation)
        if epi_parquet:
            print(f"  [EPI] Prepared: {epi_parquet}")
    
    # Add EPI if requested (use parquet if available)
    if include_epi and config['epi_file'] and Path(config['epi_file']).exists():
        epi_parquet_path = Path('processed') / machine / 'EPI.parquet'
        if epi_parquet_path.exists():
            variables_to_train.append({
                'name': 'EPI',
                'source': 'epi',
                'data_file': str(epi_parquet_path)
            })
        else:
            variables_to_train.append({
                'name': 'EPI',
                'source': 'epi',
                'data_file': config['epi_file']
            })
    
    total_vars = len(variables_to_train)
    print(f"  Variables to train: {total_vars}")
    
    for idx, var_info in enumerate(variables_to_train, 1):
        var_name = var_info['name']
        data_file = var_info['data_file']
        source = var_info['source']
        
        print(f"\n  [{idx}/{total_vars}] {var_name.upper()}")
        print(f"      Source: {data_file}")
        
        # Check if model already exists
        model_path = machine_outdir / f"{var_name}_nbeats.pth"
        history_path = machine_outdir / f"{var_name}_nbeats_history.json"
        processed_path = Path('processed') / machine / f"{var_name}.parquet"
        
        # Prepare data FIRST (so forecast can use it even if we skip training)
        train_data_path = None
        try:
            # For sensor variables, we need to prepare the data first
            if source == 'sensor':
                # Prepare hourly aggregated data for this variable
                prepared_data = prepare_variable_data(data_file, var_name, machine, aggregation=aggregation)
                if prepared_data is None:
                    print(f"      [FAIL] Could not prepare data for {var_name}")
                    results['failed'].append({'name': var_name, 'error': 'Data preparation failed'})
                    continue
                train_data_path = prepared_data
            else:
                # EPI: use parquet if available, otherwise CSV
                epi_parquet_path = Path('processed') / machine / 'EPI.parquet'
                if epi_parquet_path.exists():
                    train_data_path = str(epi_parquet_path)
                else:
                    train_data_path = data_file
        except Exception as e:
            print(f"      [FAIL] Error preparing data for {var_name}: {str(e)}")
            results['failed'].append({'name': var_name, 'error': str(e)})
            continue

        if skip_existing and model_path.exists():
            print(f"      [SKIP] Model already exists: {model_path}")
            results['skipped'].append(var_name)
            continue
        
        # Clean up old files before retraining (force mode or new training)
        if model_path.exists():
            model_path.unlink()
            print(f"      [CLEAN] Removed old model")
        if history_path.exists():
            history_path.unlink()
        # processed_path is already updated by prepare_variable_data
        
        try:
            # Train the model
            history = train_nbeats(
                data_path=train_data_path,
                lookback=lookback,
                horizon=horizon,
                model_size=model_size,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                patience=patience,
                outdir=outdir,
                device=device,
                clean_anomalies=clean_anomalies,
                seed=seed
            )
            
            results['successful'].append({
                'name': var_name,
                'epochs_trained': history['epochs_trained'],
                'best_val_loss': history['best_val_loss']
            })
            
        except Exception as e:
            print(f"      [FAIL] Error training {var_name}: {str(e)}")
            results['failed'].append({'name': var_name, 'error': str(e)})
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"BATCH TRAINING COMPLETE: {machine}")
    print(f"{'='*70}")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed:     {len(results['failed'])}")
    print(f"  Skipped:    {len(results['skipped'])}")
    print(f"{'='*70}\n")
    
    return results


def prepare_variable_data(
    data_file: str,
    target_var: str,
    machine: str,
    aggregation: str = "mean"
) -> Optional[str]:
    """
    Prepare hourly aggregated data for a single variable.
    
    Args:
        data_file: Path to raw CSV file
        target_var: Target variable name
        machine: Machine type
        aggregation: Aggregation method ('mean' or 'max')
    
    Returns:
        Path to prepared data file, or None if failed
    """
    import pandas as pd
    from pathlib import Path
    
    try:
        # Load raw data
        df = pd.read_csv(data_file)
        
        # Find timestamp column
        ts_col = None
        for col in ['timestamp', 'timestamp_grid']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            print(f"        No timestamp column found")
            return None
        
        # Parse timestamp
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.dropna(subset=[ts_col])
        df = df.set_index(ts_col)
        
        # Find target column (case-insensitive)
        target_col = None
        for col in df.columns:
            if col.lower() == target_var.lower():
                target_col = col
                break
        
        if target_col is None:
            print(f"        Target variable '{target_var}' not found")
            return None
        
        # Extract and resample to hourly aggregation
        agg_method = aggregation.lower()
        if agg_method == 'max':
            series = df[target_col].resample('1h').max()
        else:  # default to mean
            series = df[target_col].resample('1h').mean()
        series = series.dropna()
        
        print(f"        Aggregation: {agg_method.upper()}")
        
        if len(series) < 200:
            print(f"        Insufficient data: only {len(series)} hourly samples")
            return None
        
        # Create training dataframe
        train_df = pd.DataFrame({
            'timestamp': series.index,
            'y': series.values
        })
        
        # Save to processed folder
        processed_dir = Path('processed') / machine
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = processed_dir / f"{target_var}.parquet"
        train_df.to_parquet(out_path, index=False)
        
        print(f"        Prepared {len(train_df)} samples -> {out_path}")
        return str(out_path)
        
    except Exception as e:
        print(f"        Error preparing data: {str(e)}")
        return None


def prepare_epi_data(
    machine: str,
    source_file: str,
    aggregation: str = "mean"
) -> Optional[str]:
    """
    Prepare hourly aggregated EPI data for training.
    
    Args:
        machine: Machine type ('PLANT', 'DESF', 'PICADORA')
        source_file: Path to source CSV with EPI data
        aggregation: Aggregation method ('mean' or 'max')
    
    Returns:
        Path to prepared EPI.parquet file, or None if failed
    """
    import pandas as pd
    from pathlib import Path
    
    try:
        df = pd.read_csv(source_file)
        
        # Find timestamp column
        ts_col = None
        for col in ['timestamp', 'timestamp_grid', 'Timestamp']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            print(f"        No timestamp column found in {source_file}")
            return None
        
        # Parse timestamp and set index
        df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
        df = df.dropna(subset=[ts_col])
        df = df.set_index(ts_col)
        
        # Find EPI column (case-insensitive, also check for 'y')
        epi_col = None
        for col in df.columns:
            if col.lower() in ['epi', 'y']:
                epi_col = col
                break
        
        if epi_col is None:
            print(f"        'EPI' or 'y' column not found in {source_file}")
            return None
        
        # Resample to hourly
        agg_method = aggregation.lower()
        if agg_method == 'max':
            series = df[epi_col].resample('1h').max()
        else:
            series = df[epi_col].resample('1h').mean()
        series = series.dropna()
        
        print(f"        Aggregation: {agg_method.upper()}")
        
        if len(series) < 200:
            print(f"        Insufficient data: only {len(series)} hourly samples")
            return None
        
        # Create training dataframe
        train_df = pd.DataFrame({
            'timestamp': series.index,
            'y': series.values
        })
        
        # Save to processed folder
        processed_dir = Path('processed') / machine.upper()
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = processed_dir / "EPI.parquet"
        train_df.to_parquet(out_path, index=False)
        
        print(f"        Prepared {len(train_df)} samples -> {out_path}")
        return str(out_path)
        
    except Exception as e:
        print(f"        Error preparing EPI data: {str(e)}")
        return None


def batch_train_all(
    machines: list = None,
    **kwargs
) -> dict:
    """
    Train models for all variables across multiple machines.
    
    Args:
        machines: List of machines to train (default: ['DESF', 'PICADORA', 'PLANT'])
        **kwargs: Additional arguments passed to batch_train_machine
    
    Returns:
        Combined results from all machines
    """
    if machines is None:
        machines = ['DESF', 'PICADORA', 'PLANT']
    
    all_results = {}
    
    for machine in machines:
        results = batch_train_machine(machine, **kwargs)
        all_results[machine] = results
    
    # Print grand summary
    print(f"\n{'='*70}")
    print("BATCH TRAINING SUMMARY - ALL MACHINES")
    print(f"{'='*70}")
    
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    for machine, results in all_results.items():
        n_success = len(results['successful'])
        n_failed = len(results['failed'])
        n_skipped = len(results['skipped'])
        total_success += n_success
        total_failed += n_failed
        total_skipped += n_skipped
        print(f"  {machine:10s}: {n_success} trained, {n_failed} failed, {n_skipped} skipped")
    
    print(f"  {'TOTAL':10s}: {total_success} trained, {total_failed} failed, {total_skipped} skipped")
    print(f"{'='*70}\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Train N-BEATS model")
    
    # Single variable training
    parser.add_argument("--data", help="Path to preprocessed data file (for single variable training)")
    
    # Batch training options
    parser.add_argument("--batch", action="store_true", help="Train all variables for specified machine(s)")
    parser.add_argument("--machine", choices=["DESF", "PICADORA", "PLANT", "ALL"], 
                       help="Machine to train (use with --batch)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                       help="Skip variables that already have trained models")
    parser.add_argument("--force", action="store_true",
                       help="Retrain even if model exists")
    parser.add_argument("--no-epi", action="store_true",
                       help="Skip EPI training (sensor variables only)")
    parser.add_argument("--agg", choices=["mean", "max"], default="mean",
                       help="Hourly aggregation method: 'mean' (smoother) or 'max' (peaks)")
    
    # Model hyperparameters
    parser.add_argument("--lookback", type=int, default=168, help="Lookback window (default: 168 hours)")
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon (default: 24 hours)")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--outdir", default="models", help="Output directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--no-clean-anomalies", action="store_true",
                       help="Disable anomaly detection and interpolation before training")
    
    args = parser.parse_args()
    
    # Determine if we should clean anomalies
    clean_anomalies = not args.no_clean_anomalies
    
    # Handle batch training
    if args.batch:
        if not args.machine:
            parser.error("--batch requires --machine (DESF, PICADORA, PLANT, or ALL)")
        
        common_kwargs = {
            'include_epi': not args.no_epi,
            'lookback': args.lookback,
            'horizon': args.horizon,
            'model_size': args.model_size,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'validation_split': args.val_split,
            'patience': args.patience,
            'outdir': args.outdir,
            'device': args.device,
            'skip_existing': not args.force,
            'aggregation': args.agg,
            'clean_anomalies': clean_anomalies
        }
        
        if args.machine.upper() == 'ALL':
            batch_train_all(**common_kwargs)
        else:
            batch_train_machine(args.machine, **common_kwargs)
    
    # Single variable training
    elif args.data:
        train_nbeats(
            data_path=args.data,
            lookback=args.lookback,
            horizon=args.horizon,
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_split=args.val_split,
            patience=args.patience,
            outdir=args.outdir,
            device=args.device,
            clean_anomalies=clean_anomalies
        )
    else:
        parser.error("Either --data (single variable) or --batch --machine (batch training) is required")


if __name__ == "__main__":
    main()
