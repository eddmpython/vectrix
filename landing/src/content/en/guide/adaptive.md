---
title: Adaptive Intelligence
---

# Adaptive Intelligence

Unique to Vectrix -- not available in any other forecasting library.

## Regime Detection

Detect regime changes (bull/bear, peak/off-season) using HMM

```python
from vectrix import RegimeDetector

detector = RegimeDetector(nRegimes=3)
result = detector.detect(data)

print(f"Current: {result.currentRegime}")
for label, stats in result.regimeStats.items():
    print(f"  {label}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

## Regime-Aware Forecasting

Automatically switches models per regime

```python
from vectrix import RegimeAwareForecaster

raf = RegimeAwareForecaster()
result = raf.forecast(data, steps=30, period=7)
print(result.currentRegime)
print(result.modelPerRegime)
```

## Self-Healing Forecast

Monitors errors in real-time and auto-corrects

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast(predictions, lower, upper, historical_data)
healer.observe(actual_values)

report = healer.getReport()
print(f"Health: {report.overallHealth} ({report.healthScore:.1f}/100)")
print(f"Improvement: {report.improvementPct:.1f}%")

updated = healer.getUpdatedForecast()
```

## Constraint-Aware Forecasting

Apply business constraints to predictions

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

Extract time series fingerprint for meta-learning

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
