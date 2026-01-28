"""Analyze training data to understand why sensor models perform poorly."""
import pandas as pd
import numpy as np
import os

print("=" * 80)
print("TRAINING DATA ANALYSIS")
print("=" * 80)

for machine in ["DESF", "PICADORA"]:
    data_dir = f"processed/{machine}"
    if not os.path.exists(data_dir):
        continue
    
    print(f"\n{machine}")
    print("-" * 60)
    
    results = []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.parquet'):
            df = pd.read_parquet(f'{data_dir}/{f}')
            y = df['y'].values
            
            var_name = f.replace('.parquet', '')
            n_samples = len(y)
            mean = np.mean(y)
            std = np.std(y)
            cv = std / (mean + 1e-8)  # Coefficient of variation
            
            # Check for spikes/outliers
            q1 = np.percentile(y, 25)
            q3 = np.percentile(y, 75)
            iqr = q3 - q1
            outliers = np.sum((y < q1 - 3*iqr) | (y > q3 + 3*iqr))
            outlier_pct = outliers / n_samples * 100
            
            # Check for sudden jumps (high diff variance)
            diff = np.diff(y)
            diff_std = np.std(diff)
            jump_ratio = diff_std / (std + 1e-8)
            
            results.append({
                'var': var_name,
                'n': n_samples,
                'mean': mean,
                'std': std,
                'cv': cv,
                'outliers': outliers,
                'outlier_pct': outlier_pct,
                'jump_ratio': jump_ratio
            })
    
    # Sort by jump ratio (higher = harder to predict)
    results.sort(key=lambda x: x['jump_ratio'], reverse=True)
    
    print(f"{'Variable':<35} {'N':>6} {'Mean':>10} {'Std':>10} {'CV':>6} {'Outliers':>8} {'JumpRatio':>10}")
    print("-" * 95)
    for r in results:
        print(f"{r['var']:<35} {r['n']:>6} {r['mean']:>10.2f} {r['std']:>10.2f} {r['cv']:>6.2f} {r['outliers']:>8} ({r['outlier_pct']:>4.1f}%) {r['jump_ratio']:>8.2f}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("- CV (Coefficient of Variation): Higher = more variable relative to mean")
print("- Jump Ratio: Higher = more erratic hour-to-hour changes")
print("- Outliers: Points beyond 3*IQR from quartiles")
print("- High jump ratios make time series hard to predict!")
print("=" * 80)
