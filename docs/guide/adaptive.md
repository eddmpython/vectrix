---
title: Adaptive Intelligence
---

# Adaptive Intelligence

**These features are unique to Vectrix** — not available in statsforecast, Prophet, Darts, or any other forecasting library. Adaptive intelligence means your forecasts respond to changing conditions: detecting regime shifts, self-correcting as new data arrives, respecting business constraints, and profiling data DNA for intelligent model selection.

## Regime Detection

Real-world data rarely follows a single pattern. Markets alternate between bull and bear, retail demand shifts between seasons, and business metrics change after policy updates. Vectrix detects these **regimes** automatically using Hidden Markov Models

```python
from vectrix import RegimeDetector

detector = RegimeDetector(nRegimes=3)
result = detector.detect(data)

print(f"Current: {result.currentRegime}")
for label, stats in result.regimeStats.items():
    print(f"  {label}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

## Regime-Aware Forecasting

Traditional forecasters use one model for all data. `RegimeAwareForecaster` identifies the current regime and uses the model that performed best during similar past regimes

```python
from vectrix import RegimeAwareForecaster

raf = RegimeAwareForecaster()
result = raf.forecast(data, steps=30, period=7)
print(result.currentRegime)
print(result.modelPerRegime)
```

## Self-Healing Forecast

Forecasts degrade over time. Self-healing monitors errors as actual values arrive and automatically adjusts remaining predictions

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast(predictions, lower, upper, historical_data)
healer.observe(actual_values)

report = healer.getReport()
print(f"Health: {report.overallHealth} ({report.healthScore:.1f}/100)")
print(f"Improvement: {report.improvementPct:.1f}%")

updated_preds, updated_lower, updated_upper = healer.getUpdatedForecast()
```

## Constraint-Aware Forecasting

Statistical models don't know about your business rules. Predictions can go negative, exceed capacity, or show unrealistic year-over-year swings. Constraint-aware forecasting applies domain knowledge as post-processing rules

```python
from vectrix import ConstraintAwareForecaster, Constraint

caf = ConstraintAwareForecaster()
result = caf.apply(predictions, lower, upper, constraints=[
    Constraint('non_negative', {}),
    Constraint('range', {'min': 0, 'max': 5000}),
    Constraint('capacity', {'capacity': 10000}),
    Constraint('yoy_change', {'maxPct': 30, 'historicalData': past_year}),
])
```

### Constraint Types

| Type | Description |
|------|-------------|
| `non_negative` | No negative values |
| `range` | Min/max bounds |
| `capacity` | Capacity ceiling |
| `yoy_change` | Year-over-year change limit |
| `sum` | Total sum constraint |
| `monotone` | Increasing/decreasing only |
| `ratio` | Ratio between series |
| `custom` | Custom function |

## Forecast DNA

Every time series has a unique statistical signature. DNA profiling extracts 65+ features to create a deterministic fingerprint that drives intelligent model selection and difficulty estimation

```python
from vectrix import ForecastDNA

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"Fingerprint: {profile.fingerprint}")
print(f"Difficulty: {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"Recommended: {profile.recommendedModels}")
```

---

**Interactive tutorial:** `marimo run docs/tutorials/en/05_adaptive.py`
