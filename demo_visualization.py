"""
Demo: SARIMAX Visualization Examples

This script demonstrates how to use the sarimax_visualize module
to create various plots and dashboards for forecasting results.

Usage:
    python demo_visualization.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import visualization module
from sarimax.sarimax_visualize import SARIMAXVisualizer, quick_forecast_plot, quick_dashboard


def create_sample_forecast(train_df: pd.DataFrame, steps: int = 24) -> pd.DataFrame:
    """
    Create a sample forecast DataFrame for demonstration.
    
    In real usage, this would come from your SARIMAX model:
        results = model.fit()
        forecast = results.forecast(steps=24, exog=X_future)
    """
    last_timestamp = train_df['timestamp'].max()
    last_value = train_df['y'].iloc[-1]
    
    # Generate future timestamps
    future_timestamps = [
        last_timestamp + timedelta(hours=i+1) 
        for i in range(steps)
    ]
    
    # Simulate forecast (random walk with slight upward trend)
    np.random.seed(42)
    forecast_values = []
    current = last_value
    
    for i in range(steps):
        # Add trend + noise
        trend = 0.05 * i
        noise = np.random.normal(0, 2)
        current = current + trend + noise
        forecast_values.append(current)
    
    forecast_values = np.array(forecast_values)
    
    # Simulate confidence intervals (±10% of forecast)
    std_forecast = np.abs(forecast_values) * 0.1
    
    forecast_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'y_pred': forecast_values,
        'lower_ci': forecast_values - 1.96 * std_forecast,
        'upper_ci': forecast_values + 1.96 * std_forecast
    })
    
    return forecast_df


def demo_basic_forecast_plot():
    """Demo 1: Basic forecast plot."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Forecast Plot")
    print("="*70)
    
    # Load processed data
    train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
    with open('processed/corriente_motor_a_meta.json', 'r') as f:
        meta = json.load(f)
    
    # Create sample forecast
    forecast_df = create_sample_forecast(train_df, steps=24)
    
    # Create visualizer
    viz = SARIMAXVisualizer(train_df, meta, forecast_df)
    
    # Plot forecast
    fig = viz.plot_forecast(
        steps=24,
        include_history=168,  # 1 week of history
        confidence_intervals=True,
        show_thresholds=True,
        save_path='demo_outputs/forecast_basic.png'
    )
    
    print("✓ Basic forecast plot created: demo_outputs/forecast_basic.png")
    print(f"  - Historical: {len(train_df)} hours")
    print(f"  - Forecast: {len(forecast_df)} hours")
    print(f"  - Confidence intervals: ✓")
    print(f"  - Alarm thresholds: ✓")


def demo_diagnostics():
    """Demo 2: Model diagnostics (requires SARIMAX results)."""
    print("\n" + "="*70)
    print("DEMO 2: Model Diagnostics")
    print("="*70)
    
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        # Load data
        train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
        with open('processed/corriente_motor_a_meta.json', 'r') as f:
            meta = json.load(f)
        
        # Quick SARIMAX model (simple for demo)
        print("  Training quick SARIMAX model...")
        y = train_df['y'].values[:500]  # Use subset for speed
        
        model = SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False)
        print("  ✓ Model trained")
        
        # Create visualizer with results
        viz = SARIMAXVisualizer(train_df[:500], meta, results=results)
        
        # Plot diagnostics
        fig = viz.plot_diagnostics(save_path='demo_outputs/diagnostics.png')
        
        print("✓ Diagnostic plots created: demo_outputs/diagnostics.png")
        print("  - Residuals over time: ✓")
        print("  - Residual histogram: ✓")
        print("  - Q-Q plot: ✓")
        print("  - ACF/PACF: ✓")
        print("  - Statistics table: ✓")
        
    except ImportError:
        print("⚠ Skipping diagnostics demo (statsmodels not available)")


def demo_dashboard():
    """Demo 3: Comprehensive dashboard."""
    print("\n" + "="*70)
    print("DEMO 3: Comprehensive Dashboard")
    print("="*70)
    
    # Load data
    train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
    with open('processed/corriente_motor_a_meta.json', 'r') as f:
        meta = json.load(f)
    
    # Create forecast
    forecast_df = create_sample_forecast(train_df, steps=48)
    
    # Create dashboard
    viz = SARIMAXVisualizer(train_df, meta, forecast_df)
    fig = viz.create_dashboard(save_path='demo_outputs/dashboard.png')
    
    print("✓ Dashboard created: demo_outputs/dashboard.png")
    print("  Includes:")
    print("    - Main forecast plot")
    print("    - Value distribution")
    print("    - Hourly pattern boxplot")
    print("    - Summary statistics table")
    print("    - Recent trend (72h)")


def demo_actual_vs_predicted():
    """Demo 4: Actual vs. Predicted comparison."""
    print("\n" + "="*70)
    print("DEMO 4: Actual vs. Predicted Validation")
    print("="*70)
    
    # Load data
    train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
    with open('processed/corriente_motor_a_meta.json', 'r') as f:
        meta = json.load(f)
    
    # Split data (80/20)
    split_idx = int(len(train_df) * 0.8)
    train_set = train_df.iloc[:split_idx].copy()
    test_set = train_df.iloc[split_idx:].copy()
    
    # Simulate predictions (in reality, use model.predict)
    # Here we add small noise to actual values
    np.random.seed(42)
    predictions = test_set['y'] + np.random.normal(0, 2, len(test_set))
    
    # Create comparison plot
    viz = SARIMAXVisualizer(train_set, meta)
    fig = viz.plot_actual_vs_predicted(
        test_set,
        predictions,
        save_path='demo_outputs/actual_vs_predicted.png'
    )
    
    print("✓ Actual vs. Predicted plot created: demo_outputs/actual_vs_predicted.png")
    print(f"  - Test samples: {len(test_set)}")
    print("  - Metrics displayed: MAE, RMSE, R²")


def demo_seasonal_decomposition():
    """Demo 5: Seasonal decomposition."""
    print("\n" + "="*70)
    print("DEMO 5: Seasonal Decomposition")
    print("="*70)
    
    try:
        # Load data
        train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
        with open('processed/corriente_motor_a_meta.json', 'r') as f:
            meta = json.load(f)
        
        # Create visualizer
        viz = SARIMAXVisualizer(train_df, meta)
        
        # Plot decomposition
        fig = viz.plot_seasonal_decomposition(
            period=24,  # Daily seasonality
            save_path='demo_outputs/seasonal_decomposition.png'
        )
        
        print("✓ Seasonal decomposition created: demo_outputs/seasonal_decomposition.png")
        print("  - Period: 24 hours (daily pattern)")
        print("  - Components: Observed, Trend, Seasonal, Residual")
        
    except ImportError:
        print("⚠ Skipping seasonal decomposition (statsmodels not available)")


def demo_interactive_dashboard():
    """Demo 6: Interactive Plotly dashboard."""
    print("\n" + "="*70)
    print("DEMO 6: Interactive Dashboard (Plotly)")
    print("="*70)
    
    try:
        import plotly
        
        # Load data
        train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
        with open('processed/corriente_motor_a_meta.json', 'r') as f:
            meta = json.load(f)
        
        # Create forecast
        forecast_df = create_sample_forecast(train_df, steps=24)
        
        # Create interactive dashboard
        viz = SARIMAXVisualizer(train_df, meta, forecast_df)
        fig = viz.create_interactive_dashboard(
            save_path='demo_outputs/interactive_dashboard.html'
        )
        
        print("✓ Interactive dashboard created: demo_outputs/interactive_dashboard.html")
        print("  Open in browser for interactive features:")
        print("    - Zoom and pan")
        print("    - Hover tooltips")
        print("    - Toggle traces")
        print("    - Export to PNG")
        
    except ImportError:
        print("⚠ Skipping interactive dashboard (plotly not available)")
        print("  Install with: pip install plotly")


def demo_full_report():
    """Demo 7: Complete report export."""
    print("\n" + "="*70)
    print("DEMO 7: Full Report Export")
    print("="*70)
    
    # Load data
    train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
    with open('processed/corriente_motor_a_meta.json', 'r') as f:
        meta = json.load(f)
    
    # Create forecast
    forecast_df = create_sample_forecast(train_df, steps=24)
    
    # Create visualizer
    viz = SARIMAXVisualizer(train_df, meta, forecast_df)
    
    # Export complete report
    viz.export_report(
        forecast_df=forecast_df,
        output_dir='demo_outputs/full_report',
        prefix='corriente_motor_a'
    )
    
    print("✓ Full report exported to: demo_outputs/full_report/")


def demo_quick_utilities():
    """Demo 8: Quick utility functions."""
    print("\n" + "="*70)
    print("DEMO 8: Quick Utility Functions")
    print("="*70)
    
    # Load data
    train_df = pd.read_parquet('processed/corriente_motor_a.parquet')
    with open('processed/corriente_motor_a_meta.json', 'r') as f:
        meta = json.load(f)
    
    # Create forecast
    forecast_df = create_sample_forecast(train_df, steps=24)
    
    # Quick forecast plot
    print("  Creating quick forecast plot...")
    fig1 = quick_forecast_plot(
        train_df, forecast_df, meta,
        save_path='demo_outputs/quick_forecast.png'
    )
    print("  ✓ demo_outputs/quick_forecast.png")
    
    # Quick dashboard
    print("  Creating quick dashboard...")
    fig2 = quick_dashboard(
        train_df, forecast_df, meta,
        save_path='demo_outputs/quick_dashboard.png'
    )
    print("  ✓ demo_outputs/quick_dashboard.png")


def demo_multiple_variables():
    """Demo 9: Compare multiple variables."""
    print("\n" + "="*70)
    print("DEMO 9: Multiple Variables Comparison")
    print("="*70)
    
    import matplotlib.pyplot as plt
    
    # Load different variables
    variables = [
        'corriente_motor_a',
        'temp_chum_lado_a',
        'aceleracion_chum_la_h'
    ]
    
    fig, axes = plt.subplots(len(variables), 1, figsize=(14, 12), sharex=True)
    
    for idx, var in enumerate(variables):
        try:
            train_df = pd.read_parquet(f'processed/{var}.parquet')
            with open(f'processed/{var}_meta.json', 'r') as f:
                meta = json.load(f)
            
            # Plot recent data
            recent = train_df.tail(168)  # Last week
            axes[idx].plot(recent['timestamp'], recent['y'], 
                          linewidth=1.5, alpha=0.8, color='#2E86AB')
            
            # Add thresholds
            if meta.get('alarma'):
                axes[idx].axhline(y=meta['alarma'], color='#F18F01', 
                                 linestyle='--', alpha=0.7, linewidth=1.5)
            
            axes[idx].set_ylabel(f"{meta['variable']}\n({meta['unidad']})", 
                                fontweight='bold')
            axes[idx].set_title(f"{meta['variable']} - {meta['canal']}", 
                               fontweight='bold', fontsize=11)
            axes[idx].grid(True, alpha=0.3)
            
        except FileNotFoundError:
            print(f"  ⚠ Data not found for {var}, skipping...")
            continue
    
    axes[-1].set_xlabel('Timestamp', fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_outputs/multi_variable_comparison.png', dpi=300, bbox_inches='tight')
    
    print("✓ Multi-variable comparison created: demo_outputs/multi_variable_comparison.png")


def main():
    """Run all demos."""
    # Create output directory
    Path('demo_outputs').mkdir(exist_ok=True)
    Path('demo_outputs/full_report').mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("SARIMAX VISUALIZATION DEMOS")
    print("="*70)
    print("\nThis script demonstrates various visualization capabilities.")
    print("Make sure you have processed data in the 'processed/' directory.\n")
    
    # Check if data exists
    if not Path('processed/corriente_motor_a.parquet').exists():
        print("⚠ ERROR: No processed data found!")
        print("  Please run preprocessing first:")
        print("  python sarimax/sarimax_prep.py --input data/DATOS_DESF_CLEANED.csv --target corriente_motor_a")
        return
    
    # Run demos
    try:
        demo_basic_forecast_plot()
        demo_dashboard()
        demo_actual_vs_predicted()
        demo_quick_utilities()
        demo_seasonal_decomposition()
        demo_diagnostics()
        demo_interactive_dashboard()
        demo_full_report()
        demo_multiple_variables()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files in 'demo_outputs/' directory:")
        print("  - forecast_basic.png")
        print("  - dashboard.png")
        print("  - actual_vs_predicted.png")
        print("  - seasonal_decomposition.png")
        print("  - diagnostics.png")
        print("  - interactive_dashboard.html")
        print("  - multi_variable_comparison.png")
        print("  - full_report/ (complete report package)")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
