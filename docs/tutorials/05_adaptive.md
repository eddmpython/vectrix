# Tutorial 05 — Adaptive Intelligence

**Forecasts that adapt to changing conditions.**

Vectrix includes an adaptive intelligence layer — regime detection, Forecast DNA profiling, self-healing forecasts, and business constraint enforcement.

## 1. Regime Detection

Real-world time series change behavior over time. The `RegimeDetector` identifies distinct regimes (periods with different statistical properties).

```python
import numpy as np
from vectrix import RegimeDetector

np.random.seed(42)
regime1 = np.random.normal(100, 5, 80)
regime2 = np.random.normal(150, 15, 60)
regime3 = np.random.normal(120, 8, 60)
y = np.concatenate([regime1, regime2, regime3])

detector = RegimeDetector()
result = detector.detect(y)

print(f"Regimes found: {len(result.regimeStats)}")
for i, stat in enumerate(result.regimeStats):
    print(f"  Regime {i}: mean={stat.mean:.1f}, std={stat.std:.1f}, "
          f"start={stat.startIdx}, end={stat.endIdx}")
```

```
Regimes found: 3
  Regime 0: mean=100.2, std=4.8, start=0, end=79
  Regime 1: mean=149.5, std=14.2, start=80, end=139
  Regime 2: mean=120.3, std=7.9, start=140, end=199
```

### Regime-Aware Forecasting

```python
from vectrix import RegimeAwareForecaster

forecaster = RegimeAwareForecaster()
predictions = forecaster.forecast(y, steps=20)
print(f"Forecast shape: {predictions.shape}")
print(f"Uses current regime (Regime 2) statistics for prediction")
```

## 2. Forecast DNA

DNA profiling gives each time series a unique fingerprint — useful for comparing series and selecting strategies.

```python
from vectrix import ForecastDNA

dna = ForecastDNA()

profile = dna.analyze(y, period=1)
print(f"Category:     {profile.category}")
print(f"Difficulty:   {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"Fingerprint:  {profile.fingerprint}")
print(f"Recommended:  {profile.recommendedModels[:3]}")
```

### DNA Features

The DNA profile extracts statistical features that characterize your data:

```python
for key, value in list(profile.features.items())[:5]:
    print(f"  {key}: {value:.4f}")
```

Key features include: `volatilityClustering`, `seasonalPeakPeriod`, `nonlinearAutocorr`, `demandDensity`, `hurstExponent`.

## 3. Self-Healing Forecasts

When incoming actual data deviates from predictions, the self-healing system auto-corrects:

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast()

original_forecast = np.array([100, 105, 110, 115, 120])
actuals_so_far = np.array([100, 112, 125])

report = healer.heal(
    originalForecast=original_forecast,
    actuals=actuals_so_far,
    historicalData=y
)

print(f"Health score: {report.healthScore:.0f}/100")
print(f"Corrections:  {report.totalCorrected}")
print(f"Corrected forecast: {report.correctedForecast}")
```

```
Health score: 72/100
Corrections:  2
Corrected forecast: [100.  112.  125.  128.5 133.2]
```

The healer:

1. Compares actuals vs. original forecast
2. Detects systematic bias (under/over-prediction)
3. Adjusts remaining predictions based on error pattern
4. Reports overall forecast health

## 4. Business Constraints

Enforce real-world business rules on forecasts:

```python
from vectrix import ConstraintAwareForecaster, Constraint

constraints = [
    Constraint(name="non-negative", minValue=0),
    Constraint(name="capacity", maxValue=500),
    Constraint(name="growth-limit", maxChangeRate=0.10),
]

forecaster = ConstraintAwareForecaster(constraints=constraints)
constrained = forecaster.apply(
    predictions=np.array([480, 520, -10, 450, 490]),
    historicalData=y
)

print(f"Original:    [480, 520, -10, 450, 490]")
print(f"Constrained: {constrained}")
```

```
Original:    [480, 520, -10, 450, 490]
Constrained: [480. 500.   0. 450. 490.]
```

### Common Constraints

| Constraint | Use Case |
|------------|----------|
| `minValue=0` | Revenue, demand (can't be negative) |
| `maxValue=N` | Capacity limits, inventory caps |
| `maxChangeRate=0.1` | Maximum 10% period-over-period change |

## 5. Putting It Together

A complete adaptive workflow:

```python
import numpy as np
from vectrix import forecast, analyze, ForecastDNA, RegimeDetector

data = [120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
        172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
        245, 268, 258, 280, 270, 295, 285, 310, 300, 325]

report = analyze(data)
print(f"1. DNA: {report.dna.difficulty} difficulty, {report.dna.category}")
print(f"   Changepoints: {len(report.changepoints)}")

result = forecast(data, steps=10)
print(f"2. Best model: {result.model} (MAPE={result.mape:.1f}%)")
print(f"   All models: {result.models}")

comparison = result.compare()
print(f"3. Model comparison:")
print(comparison.to_string(index=False))
```

## 6. API Reference

| Class | Purpose |
|-------|---------|
| `RegimeDetector` | Detect statistical regimes in time series |
| `RegimeAwareForecaster` | Forecast using current regime context |
| `ForecastDNA` | Profile time series characteristics |
| `SelfHealingForecast` | Auto-correct forecasts with incoming actuals |
| `ConstraintAwareForecaster` | Enforce business constraints on predictions |
| `Constraint` | Define min/max/rate constraints |

---

**Next:** [Tutorial 06 — Business Intelligence](06_business.md) — Anomaly detection, scenarios, backtesting
