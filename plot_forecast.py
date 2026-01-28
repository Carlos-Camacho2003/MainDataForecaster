"""Compare MAX vs MEAN aggregation forecasts."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load MAX forecast data
hist_max = pd.read_parquet('processed/DESF/corriente_motor_a.parquet')
hist_max['timestamp'] = pd.to_datetime(hist_max['timestamp'])
fc_max = pd.read_csv('forecasts/DESF/corriente_motor_a_nbeats_forecast_2_days.csv')
fc_max['timestamp'] = pd.to_datetime(fc_max['timestamp'])

# Create plot
fig, ax = plt.subplots(figsize=(14, 6))

# Last 10 days of history
n_hours = 24 * 10
hist_plot = hist_max.tail(n_hours)

# Plot history
ax.plot(hist_plot['timestamp'], hist_plot['y'], 'b-', linewidth=1, label='Historical (MAX agg)', alpha=0.8)

# Plot forecast
ax.plot(fc_max['timestamp'], fc_max['yhat'], 'r-', linewidth=2, label='Forecast')
ax.fill_between(fc_max['timestamp'], fc_max['yhat_lo'], fc_max['yhat_hi'], alpha=0.3, color='red', label='80% CI')

# Add vertical line at forecast start
ax.axvline(fc_max['timestamp'].iloc[0], color='gray', linestyle='--', alpha=0.7)

ax.set_title('corriente_motor_a - MAX Aggregation Forecast (2 Days)', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Current (A)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Add stats
hist_mean = hist_plot['y'].mean()
hist_std = hist_plot['y'].std()
fc_mean = fc_max['yhat'].mean()
fc_min = fc_max['yhat'].min()
fc_max_val = fc_max['yhat'].max()

stats = f"Historical: mean={hist_mean:.0f}A, std={hist_std:.0f}A\nForecast: mean={fc_mean:.0f}A, range={fc_min:.0f}-{fc_max_val:.0f}A"
ax.annotate(stats, xy=(0.02, 0.05), xycoords='axes fraction', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('forecast_max_agg.png', dpi=150, bbox_inches='tight')
print('Saved: forecast_max_agg.png')
print()
print(f"Historical mean: {hist_mean:.1f} A")
print(f"Forecast mean: {fc_mean:.1f} A")
print(f"Forecast range: {fc_min:.1f} - {fc_max_val:.1f} A")
