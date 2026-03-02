# Tutorial 04 — 30+ Models

**Access every model directly and understand what Vectrix does under the hood.**

The `forecast()` function handles everything automatically. But sometimes you want to see all models, control the process, or understand why a specific model was chosen.

## 1. One-Line Model Comparison

The easiest way to compare all models:

```python
from vectrix import compare

df = compare([
    120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
    172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
], steps=5)

print(df)
```

```
                     model   mape   rmse    mae  smape  time_ms  selected
0  Dynamic Optimized Theta   6.14  19.69  14.85    inf      5.5      True
1          4Theta Ensemble   7.11  24.39  17.59    inf      2.0     False
2         AutoCES (Native)   9.00  27.88  22.09    inf     14.1     False
3         AutoETS (Native)  14.74  39.39  35.52    inf     28.6     False
```

## 2. The Vectrix Class

For full control, use the `Vectrix` class directly:

```python
import numpy as np
import pandas as pd
from vectrix import Vectrix

np.random.seed(42)
n = 150
t = np.arange(n, dtype=np.float64)
values = 500 + 2 * t + 30 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 10, n)

df = pd.DataFrame({
    "date": pd.date_range("2012-01-01", periods=n, freq="MS"),
    "revenue": values,
})

fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="revenue", steps=12)
```

### Result Structure

```python
print(f"Success: {result.success}")
print(f"Best model: {result.bestModelName}")
print(f"Models tested: {len(result.allModelResults)}")
print(f"Predictions: {result.predictions[:3]}...")
```

```
Success: True
Best model: 4Theta Ensemble
Models tested: 8
Predictions: [812.3  825.7  831.2]...
```

### All Model Results

```python
for modelId, m in result.allModelResults.items():
    if m.isValid:
        flat = " (FLAT)" if m.flatInfo and m.flatInfo.isFlat else ""
        print(f"  {m.modelName:<30} MAPE={m.mape:6.2f}%  time={m.trainingTime*1000:.0f}ms{flat}")
```

```
  4Theta Ensemble                MAPE=  2.73%  time=3ms
  Dynamic Optimized Theta        MAPE=  3.15%  time=6ms
  AutoCES (Native)               MAPE=  4.21%  time=12ms
  AutoETS (Native)               MAPE=  5.89%  time=35ms
  AutoARIMA (Native)             MAPE=  7.45%  time=18ms
  AutoMSTL                       MAPE=  8.12%  time=42ms
  DTSF                           MAPE= 11.34%  time=8ms
  ESN                            MAPE= 14.56%  time=15ms
```

## 3. Available Models

Vectrix includes these model families:

| Category | Models | Strengths |
|----------|--------|-----------|
| **Exponential Smoothing** | AutoETS, ETS-AAN, ETS-AAA | Trend + seasonality, widely applicable |
| **ARIMA** | AutoARIMA | Box-Jenkins methodology, flexible |
| **Theta** | Optimized Theta, 4Theta | M3 champion, simple yet powerful |
| **DOT** | Dynamic Optimized Theta | M4-level accuracy, auto-adaptive |
| **CES** | AutoCES | Complex Exponential Smoothing |
| **Decomposition** | MSTL, AutoMSTL | Multi-seasonal decomposition |
| **GARCH** | GARCH, EGARCH, GJR-GARCH | Volatility modeling |
| **Croston** | AutoCroston | Intermittent demand |
| **TBATS** | AutoTBATS | Multiple seasonalities |
| **Pattern Matching** | DTSF | Non-parametric, good for hourly data |
| **Neural** | ESN (Echo State) | Reservoir computing, ensemble diversity |
| **Baselines** | Naive, Seasonal Naive, Mean, RWD | Reference benchmarks |

## 4. Flat Prediction Defense

One of Vectrix's unique features: automatic detection and correction of flat (constant) predictions.

Some models produce flat lines when they fail to capture patterns. Vectrix detects this and either replaces the prediction or warns you.

```python
if result.flatInfo and result.flatInfo.isFlat:
    print(f"Flat prediction detected!")
    print(f"Correction: {result.flatInfo.message}")
```

### How It Works

1. **Detection** — Check if prediction variance is near zero relative to historical variance
2. **Risk Assessment** — Evaluate the severity (low / medium / high / critical)
3. **Correction** — Apply variance injection using historical patterns
4. **Fallback** — Switch to a different model if correction fails

## 5. Data Characteristics

The `Vectrix` class also analyzes your data before forecasting:

```python
c = result.characteristics
print(f"Period: {c.period}")
print(f"Frequency: {c.frequency}")
print(f"Trend: {c.hasTrend} ({c.trendDirection})")
print(f"Seasonality: {c.hasSeasonality}")
```

## 6. Ensemble Strategy

When multiple models perform well, Vectrix creates a variance-preserving ensemble:

```python
if result.bestModelId == "ensemble":
    print("Ensemble was selected!")
    print(f"Ensemble model: {result.bestModelName}")
```

The ensemble logic:

1. Top 3 models are refit on full data
2. MAPE-inverse weighted combination
3. Ensemble is chosen only if it preserves original data variance better than the single best model

## 7. Verbose Mode

See every step of the process:

```python
fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="revenue", steps=12)
```

This prints model training progress, timing, validation scores, and selection reasoning.

## 8. Result Object Reference (ForecastResult)

| Attribute | Type | Description |
|---|---|---|
| `.success` | `bool` | Whether forecasting succeeded |
| `.predictions` | `np.ndarray` | Final forecast values |
| `.dates` | `list` | Forecast date strings |
| `.lower95` | `np.ndarray` | 95% lower bound |
| `.upper95` | `np.ndarray` | 95% upper bound |
| `.bestModelId` | `str` | Selected model ID |
| `.bestModelName` | `str` | Selected model display name |
| `.allModelResults` | `dict` | All ModelResult objects keyed by ID |
| `.characteristics` | `DataCharacteristics` | Detected data properties |
| `.flatRisk` | `FlatRiskAssessment` | Flat prediction risk info |
| `.flatInfo` | `FlatPredictionInfo` | Flat detection/correction details |
| `.warnings` | `list` | Any warnings generated |

---

**Next:** [Tutorial 05 — Adaptive Intelligence](05_adaptive.md) — Regime detection, DNA, self-healing, constraints
