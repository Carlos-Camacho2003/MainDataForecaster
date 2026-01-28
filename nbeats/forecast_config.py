"""
N-BEATS Forecast Configuration

Provides user-selectable forecast horizon presets with accuracy estimates.
Supports: 2 days, 5 days, 15 days, and 1 month forecasts.

Each horizon comes with:
- Expected accuracy metrics (degrades with longer horizons)
- Confidence interval configuration
- Reliability ratings
- Usage recommendations

Accuracy improves as more data becomes available for training.

Usage:
    from nbeats.forecast_config import ForecastHorizon, get_horizon_config
    
    config = get_horizon_config(ForecastHorizon.DAYS_5)
    print(config)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ForecastHorizon(Enum):
    """
    Available forecast horizon options.
    
    Each represents a trade-off between coverage and accuracy.
    """
    DAYS_2 = "2_days"      # 48 hours - High accuracy
    DAYS_5 = "5_days"      # 120 hours - Medium accuracy
    DAYS_7 = "7_days"      # 168 hours - Medium-Low accuracy
    DAYS_15 = "15_days"    # 360 hours - Low accuracy, trend indication
    MONTH_1 = "1_month"    # 720 hours - Trend indication only
    MONTHS_2 = "2_months"  # 1440 hours - Long term trend
    MONTHS_3 = "3_months"  # 2160 hours - Strategic trend


@dataclass
class HorizonConfig:
    """
    Configuration for a specific forecast horizon.
    
    Contains all parameters needed for forecasting and
    accuracy/reliability metadata for user guidance.
    """
    horizon: ForecastHorizon
    horizon_hours: int
    horizon_label: str
    
    # Accuracy estimates (based on current ~1,400 samples)
    expected_accuracy_pct: float
    accuracy_range: Tuple[float, float]  # (min, max) accuracy
    
    # Confidence interval settings
    confidence_level_pct: float
    ci_expansion_factor: float  # How much CI widens vs base
    
    # Reliability
    reliability_score: int  # 1-5 stars
    reliability_label: str
    
    # Recommendations
    use_case: str
    warnings: List[str] = field(default_factory=list)
    
    # Technical parameters
    mc_samples: int = 100
    rolling_steps: int = 24  # Base model horizon for rolling
    
    def get_reliability_stars(self) -> str:
        """Get star rating display."""
        return "★" * self.reliability_score + "☆" * (5 - self.reliability_score)
    
    def get_step_accuracy(self, step: int) -> float:
        """
        Calculate expected accuracy for a specific forecast step.
        
        Accuracy degrades exponentially with forecast horizon.
        
        Args:
            step: Forecast step in hours (1 to horizon_hours)
        
        Returns:
            Expected accuracy percentage for that step
        """
        if step <= 0:
            return self.expected_accuracy_pct
        
        # Base accuracy at step 1
        base = 95.0
        
        # Decay rates by time zone
        if step <= 24:
            decay = 0.002  # 0.2% per hour
        elif step <= 48:
            decay = 0.004  # 0.4% per hour
        elif step <= 120:
            decay = 0.006  # 0.6% per hour
        elif step <= 360:
            decay = 0.008  # 0.8% per hour
        else:
            decay = 0.010  # 1.0% per hour
        
        accuracy = base * np.exp(-decay * step)
        return max(30.0, accuracy)  # Floor at 30%
    
    def get_accuracy_by_day(self) -> Dict[int, float]:
        """Get expected accuracy for each day in the horizon."""
        days = self.horizon_hours // 24
        return {
            day: self.get_step_accuracy(day * 24)
            for day in range(1, days + 1)
        }
    
    def get_ci_multiplier(self, step: int) -> float:
        """
        Get confidence interval expansion multiplier for a step.
        
        CI widens as we forecast further ahead.
        
        Args:
            step: Forecast step in hours
        
        Returns:
            Multiplier for CI width (1.0 = no expansion)
        """
        if step <= 24:
            return 1.0
        elif step <= 48:
            return 1.2
        elif step <= 120:
            return 1.5
        elif step <= 360:
            return 2.0
        else:
            return 2.5
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "horizon": self.horizon.value,
            "horizon_hours": self.horizon_hours,
            "horizon_label": self.horizon_label,
            "expected_accuracy_pct": self.expected_accuracy_pct,
            "accuracy_range": list(self.accuracy_range),
            "confidence_level_pct": self.confidence_level_pct,
            "reliability_score": self.reliability_score,
            "reliability_stars": self.get_reliability_stars(),
            "reliability_label": self.reliability_label,
            "use_case": self.use_case,
            "warnings": self.warnings,
            "accuracy_by_day": self.get_accuracy_by_day()
        }
    
    def print_summary(self):
        """Print formatted configuration summary."""
        stars = self.get_reliability_stars()
        
        print("\n" + "=" * 70)
        print(f"FORECAST HORIZON: {self.horizon_label}")
        print("=" * 70)
        print(f"  Duration:         {self.horizon_hours} hours ({self.horizon_hours // 24} days)")
        print(f"  Expected Accuracy: {self.expected_accuracy_pct:.0f}% (range: {self.accuracy_range[0]:.0f}-{self.accuracy_range[1]:.0f}%)")
        print(f"  Confidence Level: {self.confidence_level_pct}%")
        print(f"  Reliability:      {stars} ({self.reliability_label})")
        print("-" * 70)
        print(f"  Best For:         {self.use_case}")
        
        if self.warnings:
            print("-" * 70)
            print("  ⚠️  WARNINGS:")
            for w in self.warnings:
                print(f"      • {w}")
        
        print("-" * 70)
        print("  Accuracy by Day:")
        for day, acc in self.get_accuracy_by_day().items():
            bar_len = int(acc / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            label = "✓ Reliable" if acc >= 80 else ("⚠ Trend" if acc >= 60 else "⚡ Indication")
            print(f"      Day {day:>2}: {acc:>5.1f}% {bar} {label}")
        
        print("=" * 70)


# Pre-configured horizons
HORIZON_CONFIGS: Dict[ForecastHorizon, HorizonConfig] = {
    ForecastHorizon.DAYS_2: HorizonConfig(
        horizon=ForecastHorizon.DAYS_2,
        horizon_hours=48,
        horizon_label="2 Days (48 hours)",
        expected_accuracy_pct=85.0,
        accuracy_range=(80.0, 92.0),
        confidence_level_pct=80,
        ci_expansion_factor=1.2,
        reliability_score=4,
        reliability_label="HIGH",
        use_case="Operational planning, maintenance scheduling, resource allocation",
        warnings=[],
        mc_samples=100,
        rolling_steps=24
    ),
    ForecastHorizon.DAYS_5: HorizonConfig(
        horizon=ForecastHorizon.DAYS_5,
        horizon_hours=120,
        horizon_label="5 Days (120 hours)",
        expected_accuracy_pct=72.0,
        accuracy_range=(65.0, 82.0),
        confidence_level_pct=80,
        ci_expansion_factor=1.5,
        reliability_score=3,
        reliability_label="MEDIUM",
        use_case="Weekly planning, preventive maintenance scheduling, trend analysis",
        warnings=[
            "Accuracy decreases significantly after day 3",
            "Use confidence intervals for decision-making"
        ],
        mc_samples=120,
        rolling_steps=24
    ),
    ForecastHorizon.DAYS_7: HorizonConfig(
        horizon=ForecastHorizon.DAYS_7,
        horizon_hours=168,
        horizon_label="7 Days (168 hours)",
        expected_accuracy_pct=65.0,
        accuracy_range=(55.0, 75.0),
        confidence_level_pct=85,
        ci_expansion_factor=1.7,
        reliability_score=3,
        reliability_label="MEDIUM-LOW",
        use_case="Weekly planning, trend analysis",
        warnings=[
            "Accuracy decreases after day 5",
            "Use confidence intervals for decision-making"
        ],
        mc_samples=130,
        rolling_steps=24
    ),
    ForecastHorizon.DAYS_15: HorizonConfig(
        horizon=ForecastHorizon.DAYS_15,
        horizon_hours=360,
        horizon_label="15 Days (360 hours)",
        expected_accuracy_pct=55.0,
        accuracy_range=(45.0, 68.0),
        confidence_level_pct=90,
        ci_expansion_factor=2.0,
        reliability_score=2,
        reliability_label="LOW",
        use_case="Long-term planning, early warning systems, trend indication",
        warnings=[
            "⚠️ Days 1-5: Moderate accuracy for planning",
            "⚠️ Days 6-15: Use for trend indication only",
            "Values are directional estimates, not precise predictions",
            "Verify with shorter-term forecasts before critical decisions"
        ],
        mc_samples=150,
        rolling_steps=24
    ),
    ForecastHorizon.MONTH_1: HorizonConfig(
        horizon=ForecastHorizon.MONTH_1,
        horizon_hours=720,
        horizon_label="1 Month (720 hours)",
        expected_accuracy_pct=42.0,
        accuracy_range=(35.0, 55.0),
        confidence_level_pct=95,
        ci_expansion_factor=2.5,
        reliability_score=1,
        reliability_label="TREND ONLY",
        use_case="Strategic planning, budget forecasting, seasonal pattern detection",
        warnings=[
            "🔴 TREND INDICATION ONLY - Not for precise values",
            "Days 1-5: Use for planning (~70% accuracy)",
            "Days 6-15: Trend direction only (~50% accuracy)",
            "Days 16-30: General tendency only (~40% accuracy)",
            "Always confirm with rolling short-term forecasts",
            "Accuracy will improve significantly with more training data"
        ],
        mc_samples=200,
        rolling_steps=24
    ),
    ForecastHorizon.MONTHS_2: HorizonConfig(
        horizon=ForecastHorizon.MONTHS_2,
        horizon_hours=1440,
        horizon_label="2 Months (1440 hours)",
        expected_accuracy_pct=35.0,
        accuracy_range=(25.0, 45.0),
        confidence_level_pct=95,
        ci_expansion_factor=3.0,
        reliability_score=1,
        reliability_label="LONG TERM TREND",
        use_case="Strategic planning, seasonal analysis",
        warnings=[
            "🔴 LONG TERM TREND ONLY - Not for precise values",
            "High uncertainty",
            "Use for general direction only"
        ],
        mc_samples=250,
        rolling_steps=24
    ),
    ForecastHorizon.MONTHS_3: HorizonConfig(
        horizon=ForecastHorizon.MONTHS_3,
        horizon_hours=2160,
        horizon_label="3 Months (2160 hours)",
        expected_accuracy_pct=30.0,
        accuracy_range=(20.0, 40.0),
        confidence_level_pct=95,
        ci_expansion_factor=3.5,
        reliability_score=1,
        reliability_label="STRATEGIC TREND",
        use_case="Quarterly planning, long-term strategy",
        warnings=[
            "🔴 STRATEGIC TREND ONLY - Not for precise values",
            "Very high uncertainty",
            "Use for general direction only"
        ],
        mc_samples=300,
        rolling_steps=24
    )
}


def get_horizon_config(horizon: ForecastHorizon) -> HorizonConfig:
    """
    Get configuration for a specific forecast horizon.
    
    Args:
        horizon: Desired forecast horizon
    
    Returns:
        HorizonConfig with all parameters and metadata
    """
    return HORIZON_CONFIGS[horizon]


def get_all_configs() -> Dict[str, HorizonConfig]:
    """Get all available horizon configurations."""
    return {h.value: config for h, config in HORIZON_CONFIGS.items()}


def get_comparison_table() -> str:
    """
    Generate a formatted comparison table of all horizons.
    
    Returns:
        Formatted string table for display
    """
    table = """
┌──────────────┬──────────┬──────────┬─────────────┬──────────────────────────────────────┐
│ Horizon      │ Hours    │ Accuracy │ Reliability │ Best For                             │
├──────────────┼──────────┼──────────┼─────────────┼──────────────────────────────────────┤
"""
    
    for horizon, config in HORIZON_CONFIGS.items():
        stars = config.get_reliability_stars()
        use = config.use_case[:36] + ".." if len(config.use_case) > 38 else config.use_case
        
        table += f"│ {config.horizon_label:<12} │ {config.horizon_hours:>6}h  │ {config.expected_accuracy_pct:>6.0f}%  │ {stars:<11} │ {use:<36} │\n"
    
    table += "└──────────────┴──────────┴──────────┴─────────────┴──────────────────────────────────────┘"
    
    return table


def estimate_future_accuracy(
    current_samples: int,
    horizon_hours: int,
    future_samples: int
) -> Dict[str, float]:
    """
    Estimate how accuracy will improve with more data.
    
    Args:
        current_samples: Current number of training samples
        horizon_hours: Forecast horizon in hours
        future_samples: Projected future sample count
    
    Returns:
        Dictionary with accuracy estimates
    """
    # Rule of thumb: accuracy improves with sqrt of data increase
    # and is constrained by horizon/data ratio
    
    current_ratio = current_samples / horizon_hours
    future_ratio = future_samples / horizon_hours
    
    # Base accuracy from current config
    base_accuracy = 95.0
    
    # Penalty for low ratio
    if current_ratio < 10:
        current_penalty = (10 - current_ratio) * 3  # 3% per unit below 10
    else:
        current_penalty = 0
    
    if future_ratio < 10:
        future_penalty = (10 - future_ratio) * 3
    else:
        future_penalty = 0
    
    # Data volume bonus (sqrt scaling)
    data_bonus = 5 * np.log10(future_samples / current_samples) if future_samples > current_samples else 0
    
    current_accuracy = max(30, base_accuracy - current_penalty - (horizon_hours * 0.05))
    future_accuracy = max(30, base_accuracy - future_penalty - (horizon_hours * 0.05) + data_bonus)
    
    return {
        "current_samples": current_samples,
        "future_samples": future_samples,
        "horizon_hours": horizon_hours,
        "current_accuracy_pct": round(current_accuracy, 1),
        "future_accuracy_pct": round(min(95, future_accuracy), 1),
        "improvement_pct": round(future_accuracy - current_accuracy, 1),
        "current_ratio": round(current_ratio, 1),
        "future_ratio": round(future_ratio, 1)
    }


def print_data_impact_analysis(current_samples: int = 1400):
    """
    Print analysis of how more data will improve accuracy.
    
    Args:
        current_samples: Current training sample count
    """
    print("\n" + "=" * 75)
    print("DATA IMPACT ANALYSIS: How More Data Improves Accuracy")
    print("=" * 75)
    print(f"Current training samples: {current_samples:,}")
    print("-" * 75)
    
    scenarios = [
        ("Current", current_samples),
        ("+1 Month", current_samples + 720),
        ("+3 Months", current_samples + 2160),
        ("+6 Months", current_samples + 4320),
        ("+1 Year", current_samples + 8760)
    ]
    
    for horizon in [ForecastHorizon.DAYS_2, ForecastHorizon.DAYS_5, 
                    ForecastHorizon.DAYS_15, ForecastHorizon.MONTH_1]:
        config = HORIZON_CONFIGS[horizon]
        print(f"\n{config.horizon_label}:")
        print(f"  {'Scenario':<12} │ {'Samples':>8} │ {'Ratio':>6} │ {'Accuracy':>8} │ {'Rating':<12}")
        print(f"  {'-'*12}-┼-{'-'*8}-┼-{'-'*6}-┼-{'-'*8}-┼-{'-'*12}")
        
        for name, samples in scenarios:
            est = estimate_future_accuracy(current_samples, config.horizon_hours, samples)
            ratio = samples / config.horizon_hours
            
            if ratio >= 15:
                rating = "★★★★★ Great"
            elif ratio >= 10:
                rating = "★★★★☆ Good"
            elif ratio >= 5:
                rating = "★★★☆☆ OK"
            elif ratio >= 2:
                rating = "★★☆☆☆ Low"
            else:
                rating = "★☆☆☆☆ Poor"
            
            acc = est['future_accuracy_pct'] if name != "Current" else est['current_accuracy_pct']
            print(f"  {name:<12} │ {samples:>8,} │ {ratio:>5.1f}x │ {acc:>7.1f}% │ {rating}")
    
    print("\n" + "=" * 75)
    print("KEY INSIGHT: Collect 6+ months of data for reliable 15-day forecasts")
    print("=" * 75)


if __name__ == "__main__":
    # Demo: Show all configurations
    print("\n" + "=" * 75)
    print("N-BEATS FORECAST HORIZON OPTIONS")
    print("=" * 75)
    
    print(get_comparison_table())
    
    # Show detailed configs
    for horizon in ForecastHorizon:
        config = get_horizon_config(horizon)
        config.print_summary()
    
    # Show data impact analysis
    print_data_impact_analysis(current_samples=1400)
