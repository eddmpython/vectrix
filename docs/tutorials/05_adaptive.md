---
title: "Tutorial 05 — Adaptive Intelligence"
---

# Tutorial 05 — Adaptive Intelligence

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/05_adaptive.ipynb)

**These features are unique to Vectrix** — not available in statsforecast, Prophet, Darts, or any other forecasting library. Adaptive intelligence means your forecasts respond to changing conditions in real time: detecting regime shifts, self-correcting as new data arrives, respecting business constraints, and profiling data DNA for intelligent model selection.

## Regime Detection

Real-world data rarely follows a single pattern throughout its history. Markets alternate between bull and bear phases, retail demand shifts between peak and off-season, and business metrics change after product launches or policy changes. Vectrix detects these **regimes** automatically using Hidden Markov Models

```python
from vectrix import RegimeDetector
import numpy as np

np.random.seed(42)
regime1 = np.random.randn(50) * 2 + 100
regime2 = np.random.randn(50) * 5 + 130
regime3 = np.random.randn(50) * 2 + 110
data = np.concatenate([regime1, regime2, regime3])

detector = RegimeDetector(nRegimes=3)
result = detector.detect(data)

print(f"Current regime: {result.currentRegime}")
print(f"Number of regimes: {len(result.regimeStats)}")
for label, stats in result.regimeStats.items():
    print(f"  Regime {label}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
```

**Expected output:**

```
Current regime: 2
Number of regimes: 3
  Regime 0: mean=100.12, std=2.05
  Regime 1: mean=130.45, std=4.87
  Regime 2: mean=110.23, std=1.98
```

### RegimeDetector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nRegimes` | `2` | Number of regimes to detect |

### RegimeResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `currentRegime` | `int` | Index of the current (most recent) regime |
| `regimeStats` | `dict` | Per-regime statistics (mean, std, etc.) |
| `regimeSequence` | `np.ndarray` | Regime label for each observation |

## Regime-Aware Forecasting

Traditional forecasters use one model for all data — even if the data's behavior changed dramatically midway. `RegimeAwareForecaster` does something smarter: it identifies which regime the series is currently in and uses the model that performed best during similar past regimes

```python
from vectrix import RegimeAwareForecaster
import numpy as np

data = np.random.randn(200).cumsum() + 100

raf = RegimeAwareForecaster()
result = raf.forecast(data, steps=30, period=7)

print(f"Current regime: {result.currentRegime}")
print(f"Model per regime: {result.modelPerRegime}")
print(f"Predictions: {result.predictions[:5].round(2)}")
```

**Expected output:**

```
Current regime: 1
Model per regime: {0: 'AutoETS', 1: 'DOT', 2: 'Theta'}
Predictions: [102.34 103.12 103.89 104.67 105.44]
```

The forecaster detects which regime the series is currently in and uses the model that performed best during similar past regimes.

## Forecast DNA

Every time series has a unique statistical signature. **DNA profiling** extracts 65+ features — autocorrelation structure, Hurst exponent, entropy, volatility clustering, seasonal strength, and more — to create a deterministic fingerprint. This fingerprint drives intelligent model selection and difficulty estimation

```python
from vectrix import ForecastDNA
import numpy as np

dna = ForecastDNA()

t = np.arange(100)
data = 50 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.randn(100) * 3

profile = dna.analyze(data, period=12)

print(f"Fingerprint: {profile.fingerprint}")
print(f"Difficulty: {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"Category: {profile.category}")
print(f"Recommended models: {profile.recommendedModels[:5]}")
```

**Expected output:**

```
Fingerprint: b7e2f4a1
Difficulty: easy (28/100)
Category: seasonal
Recommended models: ['AutoETS', 'MSTL', 'Theta', 'DOT', 'AutoCES']
```

### DNA Profile Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `fingerprint` | `str` | Deterministic hash of the data's statistical properties |
| `difficulty` | `str` | `easy`, `medium`, `hard`, `very_hard` |
| `difficultyScore` | `float` | 0-100, higher means harder to forecast |
| `category` | `str` | `seasonal`, `trending`, `volatile`, `intermittent`, `stationary` |
| `recommendedModels` | `list[str]` | Models ranked by expected performance |

### Using DNA for Decision Making

```python
from vectrix import ForecastDNA, forecast

dna = ForecastDNA()
profile = dna.analyze(data, period=7)

if profile.difficultyScore > 70:
    print("Hard series -- consider using ensemble or longer history.")
elif profile.category == "intermittent":
    print("Sparse demand -- Croston variants recommended.")
else:
    result = forecast(data, steps=14)
    print(f"Forecast with {result.model}")
```

## Self-Healing Forecast

Forecasts degrade over time — the further out you predict, the less accurate the results. **Self-healing** solves this by monitoring errors as actual values arrive and automatically adjusting the remaining predictions. If your forecast was too optimistic for the first 3 days, the healer compensates for the remaining days

```python
from vectrix import SelfHealingForecast, forecast
import numpy as np

data = np.random.randn(100).cumsum() + 200
result = forecast(data, steps=14)

healer = SelfHealingForecast(
    result.predictions,
    result.lower,
    result.upper,
    data,
)
```

As actual values arrive, feed them to the healer

```python
actual_day1_to_5 = np.array([198.5, 201.2, 195.8, 203.4, 199.1])
healer.observe(actual_day1_to_5)

report = healer.getReport()
print(f"Health: {report.overallHealth} ({report.healthScore:.1f}/100)")
print(f"Observed: {report.totalObserved}")
print(f"Corrected: {report.totalCorrected}")
print(f"Improvement: {report.improvementPct:.1f}%")
```

**Expected output:**

```
Health: good (82.5/100)
Observed: 5
Corrected: 3
Improvement: 12.4%
```

Get the updated forecast with corrections applied

```python
updated_preds, updated_lower, updated_upper = healer.getUpdatedForecast()
print(f"Original remaining: {result.predictions[5:].round(1)}")
print(f"Updated remaining:  {updated_preds[5:].round(1)}")
```

### HealingReport Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `overallHealth` | `str` | `excellent`, `good`, `fair`, `poor` |
| `healthScore` | `float` | 0-100, higher is better |
| `totalObserved` | `int` | Number of actual values received |
| `totalCorrected` | `int` | Number of corrections applied |
| `improvementPct` | `float` | Error reduction percentage |

## Constraint-Aware Forecasting

Statistical models don't know about your business rules. Predictions can go negative (impossible for unit sales), exceed warehouse capacity, or show unrealistic year-over-year swings. **Constraint-aware forecasting** applies domain knowledge as post-processing rules — without modifying the underlying model

```python
from vectrix import ConstraintAwareForecaster, Constraint, forecast
import numpy as np

data = np.random.randn(100).cumsum() + 500
result = forecast(data, steps=14)

caf = ConstraintAwareForecaster()
constrained = caf.apply(
    result.predictions,
    result.lower,
    result.upper,
    constraints=[
        Constraint('non_negative', {}),
        Constraint('range', {'min': 100, 'max': 5000}),
        Constraint('capacity', {'capacity': 3000}),
    ]
)

print(f"Original: {result.predictions[:5].round(1)}")
print(f"Constrained: {constrained.predictions[:5].round(1)}")
```

### Year-over-Year Constraint

Limit how much the forecast can change compared to last year

```python
past_year = data[-365:]

constrained = caf.apply(
    result.predictions,
    result.lower,
    result.upper,
    constraints=[
        Constraint('non_negative', {}),
        Constraint('yoy_change', {'maxPct': 30, 'historicalData': past_year}),
    ]
)
```

### All Constraint Types

| Type | Parameters | Description |
|------|-----------|-------------|
| `non_negative` | `{}` | Ensures all predictions >= 0 |
| `range` | `{'min': N, 'max': M}` | Clips to [min, max] |
| `capacity` | `{'capacity': N}` | Caps at capacity ceiling |
| `yoy_change` | `{'maxPct': N, 'historicalData': arr}` | Limits year-over-year change |
| `sum_constraint` | `{'window': N, 'maxSum': M}` | Limits sum within rolling windows |
| `monotone` | `{'direction': 'increasing'}` | Forces monotonic increase or decrease |
| `ratio` | `{'minRatio': N, 'maxRatio': M}` | Limits consecutive value ratio |
| `custom` | `{'fn': callable}` | Applies a custom constraint function |

### Custom Constraint Example

```python
import numpy as np

def weekend_boost(predictions, lower, upper):
    boosted = predictions.copy()
    lo = lower.copy()
    hi = upper.copy()
    for i in range(len(boosted)):
        if i % 7 in [5, 6]:
            boosted[i] *= 1.2
            lo[i] *= 1.2
            hi[i] *= 1.2
    return boosted, lo, hi

constrained = caf.apply(
    result.predictions,
    result.lower,
    result.upper,
    constraints=[
        Constraint('custom', {'fn': weekend_boost}),
    ]
)
```

## Complete Example

A full adaptive forecasting workflow — profile the data, detect regimes, generate a forecast, and apply business constraints

```python
import numpy as np
from vectrix import (
    forecast, ForecastDNA, RegimeDetector,
    SelfHealingForecast, ConstraintAwareForecaster, Constraint
)

np.random.seed(42)
data = np.random.randn(200).cumsum() + 500

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"Difficulty: {profile.difficulty}")
print(f"Recommended: {profile.recommendedModels[:3]}")

detector = RegimeDetector(nRegimes=2)
regimes = detector.detect(data)
print(f"Current regime: {regimes.currentRegime}")

result = forecast(data, steps=14)
print(f"Model: {result.model}, MAPE: {result.mape:.2f}%")

caf = ConstraintAwareForecaster()
constrained = caf.apply(
    result.predictions, result.lower, result.upper,
    constraints=[
        Constraint('non_negative', {}),
        Constraint('range', {'min': 300, 'max': 800}),
    ]
)
print(f"Constrained predictions: {constrained.predictions[:5].round(1)}")
```

---
