"""
N-BEATS Forecast Visualization Tool

Provides plotting and dashboard capabilities for N-BEATS forecasts and training history.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
from datetime import datetime

try:
    import seaborn as sns
    sns.set(style="whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class NBeatsVisualizer:
    """Visualization toolkit for N-BEATS forecasts and training history."""

    def __init__(self, train_df: pd.DataFrame, meta: Optional[Dict] = None, forecast_df: Optional[pd.DataFrame] = None, history_path: Optional[str] = None):
        self.train_df = train_df.copy()
        if 'timestamp' in self.train_df.columns:
            self.train_df['timestamp'] = pd.to_datetime(self.train_df['timestamp'])
        self.meta = meta or {}
        self.forecast_df = forecast_df
        self.history_path = history_path
        self.variable = self.meta.get('variable', 'Variable')
        self.unit = self.meta.get('unidad', '')
        self.alarm_threshold = self.meta.get('alarma', None)
        self.critical_threshold = self.meta.get('critico', None)

        self._setup_style()

    def _setup_style(self):
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 10

    def plot_forecast(self, forecast_df: Optional[pd.DataFrame] = None, include_history: int = 168, save_path: Optional[str] = None) -> plt.Figure:
        """Plot N-BEATS forecast; accepts N-BEATS column names 'yhat','yhat_lo','yhat_hi'."""
        if forecast_df is not None:
            self.forecast_df = forecast_df
        if self.forecast_df is None:
            raise ValueError("No forecast data provided.")

        # Map possible column names
        df = self.forecast_df.copy()
        if 'y_pred' in df.columns and 'yhat' not in df.columns:
            df = df.rename(columns={'y_pred': 'yhat', 'lower_ci': 'yhat_lo', 'upper_ci': 'yhat_hi'})

        fig, ax = plt.subplots(figsize=(14, 6))
        hist = self.train_df.tail(include_history)
        ax.plot(hist['timestamp'], hist['y'], label='Historical', color='#2E86AB')

        ax.plot(df['timestamp'], df['yhat'], label='N-BEATS Forecast', color='#A23B72', linestyle='--')

        if 'yhat_lo' in df.columns:
            ax.fill_between(df['timestamp'], df['yhat_lo'], df['yhat_hi'], alpha=0.2, color='#A23B72', label='80% CI')

        if self.alarm_threshold is not None:
            ax.axhline(self.alarm_threshold, linestyle='--', color='orange', label=f"Alarma ({self.alarm_threshold} {self.unit})")
        if self.critical_threshold is not None:
            ax.axhline(self.critical_threshold, linestyle='--', color='red', label=f"Crítico ({self.critical_threshold} {self.unit})")

        ax.set_xlabel('Timestamp'); ax.set_ylabel(f'{self.variable} ({self.unit})')
        ax.set_title(f'N-BEATS Forecast: {self.variable}')
        ax.legend(); ax.grid(True)

        if save_path:
            plt.tight_layout(); plt.savefig(save_path, dpi=160)
            print(f"[OK] Forecast saved to: {save_path}")

        return fig

    def plot_training_history(self, history_path: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
        """Plot training/validation loss from history json (created by `nbeats_train.py`)."""
        path = history_path or self.history_path
        if path is None or not Path(path).exists():
            raise ValueError("History path not provided or not found")

        with open(path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        train_losses = history.get('train_losses', [])
        val_losses = history.get('val_losses', [])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label='Train Loss')
        ax.plot(np.arange(1, len(val_losses) + 1), val_losses, label='Val Loss')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('N-BEATS Training Loss')
        ax.legend(); ax.grid(True)

        if save_path:
            plt.tight_layout(); plt.savefig(save_path, dpi=160)
            print(f"[OK] Training history saved to: {save_path}")
        return fig

    def plot_actual_vs_predicted(self, test_df: pd.DataFrame, predictions: pd.Series, save_path: Optional[str] = None) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax1.plot(test_df['timestamp'], test_df['y'], label='Actual')
        ax1.plot(test_df['timestamp'], predictions, label='Predicted', linestyle='--')
        ax1.set_ylabel(f'{self.variable} ({self.unit})'); ax1.set_title('Actual vs Predicted'); ax1.legend(); ax1.grid(True)

        residuals = test_df['y'] - predictions
        ax2.plot(test_df['timestamp'], residuals, label='Residuals', color='orange')
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_xlabel('Timestamp'); ax2.set_ylabel('Residual'); ax2.grid(True)

        if save_path:
            plt.tight_layout(); plt.savefig(save_path, dpi=160)
            print(f"[OK] Actual vs Pred plot saved to: {save_path}")

        return fig

    def create_dashboard(self, forecast_df: Optional[pd.DataFrame] = None, save_path: Optional[str] = None):
        """Create dashboard with forecast and training history."""
        if forecast_df is not None:
            self.forecast_df = forecast_df
        fig = plt.figure(figsize=(18, 12))
        # Main plot (create inline to avoid unused var in static analysis)
        ax1 = fig.add_subplot(2, 2, 1)
        hist = self.train_df.tail(168)
        ax1.plot(hist['timestamp'], hist['y'], label='Historical', color='#2E86AB')
        if self.forecast_df is not None:
            df = self.forecast_df.copy()
            if 'yhat' not in df.columns and 'y_pred' in df.columns:
                df = df.rename(columns={'y_pred': 'yhat', 'lower_ci': 'yhat_lo', 'upper_ci': 'yhat_hi'})
            ax1.plot(df['timestamp'], df['yhat'], label='N-BEATS Forecast', color='#A23B72', linestyle='--')
            if 'yhat_lo' in df.columns:
                ax1.fill_between(df['timestamp'], df['yhat_lo'], df['yhat_hi'], alpha=0.2, color='#A23B72')
        plt.suptitle(f'N-BEATS Dashboard: {self.variable}')

        # Training history
        ax2 = fig.add_subplot(2, 2, 2)
        try:
            history = json.load(open(self.history_path)) if self.history_path else {}
            ax2.plot(history.get('train_losses', []), label='train')
            ax2.plot(history.get('val_losses', []), label='val')
            ax2.set_title('Training Loss'); ax2.legend()
        except Exception:
            ax2.text(0.5, 0.5, 'No history available', ha='center')

        # Distribution
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.hist(self.train_df['y'].dropna(), bins=40)
        ax3.set_title('Distribution')

        # Recent trend
        ax4 = fig.add_subplot(2, 2, 4)
        recent = self.train_df.tail(72)
        ax4.plot(recent['timestamp'], recent['y'])
        ax4.set_title('Recent Trend (72h)')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=160)
            print(f"[OK] Dashboard saved to: {save_path}")
        return fig

    def create_interactive_dashboard(self, save_path: Optional[str] = None):
        if not HAS_PLOTLY:
            print("⚠ Plotly not available")
            return None
        # Construct interactive plotly dashboard
        forecast = self.forecast_df
        hist = self.train_df.tail(168)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=hist['timestamp'], y=hist['y'], name='Historical'), row=1, col=1)
        if forecast is not None:
            fig.add_trace(go.Scatter(x=forecast['timestamp'], y=forecast['yhat'], name='Forecast'), row=1, col=1)
            if 'yhat_lo' in forecast.columns and 'yhat_hi' in forecast.columns:
                fig.add_trace(go.Scatter(x=forecast['timestamp'], y=forecast['yhat_hi'], name='Upper CI', line={'width':0}), row=1, col=1)
                fig.add_trace(go.Scatter(x=forecast['timestamp'], y=forecast['yhat_lo'], name='Lower CI', fill='tonexty', fillcolor='rgba(162,59,114,0.2)'), row=1, col=1)
        fig.update_layout(title_text=f'N-BEATS Interactive Dashboard: {self.variable}')
        if save_path:
            fig.write_html(save_path)
            print(f"[OK] Interactive HTML saved to: {save_path}")
        return fig

    def export_report(self, output_dir: str = 'reports', prefix: str = 'nbeats'):
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        if self.forecast_df is not None:
            self.plot_forecast(save_path=str(out / f'{prefix}_forecast_{ts}.png'))
        try:
            self.plot_training_history(save_path=str(out / f'{prefix}_training_{ts}.png'))
        except Exception:
            pass
        self.create_dashboard(save_path=str(out / f'{prefix}_dashboard_{ts}.png'))
        if HAS_PLOTLY:
            self.create_interactive_dashboard(save_path=str(out / f'{prefix}_interactive_{ts}.html'))
        summary = {
            'variable': self.variable,
            'samples': len(self.train_df),
            'generated_at': ts
        }
        with open(out / f'{prefix}_summary_{ts}.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print('[OK] N-BEATS report exported')


if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(description="N-BEATS Visualization tool")
    ap.add_argument("--train", required=True, help="Training parquet file")
    ap.add_argument("--meta", required=True, help="Metadata JSON file")
    ap.add_argument("--forecast", help="Forecast parquet file (optional)")
    ap.add_argument("--history", help="Training history JSON file (optional)")
    ap.add_argument("--output", default="reports", help="Output directory for reports")
    ap.add_argument("--type", choices=['forecast', 'dashboard', 'history', 'all'], default='all')
    args = ap.parse_args()

    train_df = pd.read_parquet(args.train)
    meta = json.load(open(args.meta))
    forecast_df = None
    if args.forecast:
        forecast_path = Path(args.forecast)
        if forecast_path.suffix.lower() == '.csv':
            forecast_df = pd.read_csv(forecast_path, parse_dates=['timestamp'])
        else:
            forecast_df = pd.read_parquet(forecast_path)

    viz = NBeatsVisualizer(train_df, meta, forecast_df, history_path=args.history)

    if args.type in ('forecast', 'all') and forecast_df is not None:
        viz.plot_forecast(forecast_df, save_path=f"{args.output}/nbeats_forecast.png")
    if args.type in ('history', 'all') and args.history:
        viz.plot_training_history(history_path=args.history, save_path=f"{args.output}/nbeats_training.png")
    if args.type in ('dashboard', 'all'):
        viz.create_dashboard(forecast_df, save_path=f"{args.output}/nbeats_dashboard.png")
    viz.export_report(output_dir=args.output, prefix='nbeats')
