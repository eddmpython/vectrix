---
title: "Tutorial 04 — 30+ Models"
---

# Tutorial 04 — 30+ Models

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/04_models.ipynb)

**Vectrix ships with 30+ forecasting models** spanning exponential smoothing, ARIMA, decomposition, theta methods, intermittent demand, volatility, neural reservoirs, and pattern matching. Each model captures different aspects of time series dynamics — trend, seasonality, level shifts, nonlinear patterns — and Vectrix automatically selects the best one for your data.

This tutorial shows how to compare models, use the `Vectrix` class for full control, access individual engines directly, and understand the model selection process.

## Quick Model Comparison

The fastest way to see how all models perform on your data — one function call, ranked by accuracy

```python
from vectrix import compare

data = [120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215,
        125, 140, 155, 138, 160, 175, 168, 185, 200, 190, 210, 225]

ranking = compare(data, steps=6)
print(ranking)
```

**Expected output:**

```
              model   mape   rmse    mae  smape
0           AutoETS   3.21  10.45   8.12   3.15
1             Theta   3.89  12.31   9.67   3.78
2               DOT   4.12  13.02  10.23   4.01
3           AutoCES   4.34  13.87  10.89   4.22
4              MSTL   4.56  14.23  11.12   4.45
5         AutoARIMA   4.78  15.01  11.78   4.65
6            4Theta   4.92  15.34  12.01   4.81
...
```

Or access the comparison from a forecast result

```python
from vectrix import forecast

result = forecast(data, steps=6)
print(result.compare())
```

## The Vectrix Class

The Easy API (`forecast()`) is great for quick results. When you need full control — access to all model results, flat risk diagnostics, ensemble weights, and per-model metrics — use the `Vectrix` class directly

```python
import pandas as pd
from vectrix import Vectrix

df = pd.read_csv("sales.csv")
vx = Vectrix()
result = vx.forecast(
    df,
    dateCol="date",
    valueCol="sales",
    steps=14,
    trainRatio=0.8
)

print(f"Best model: {result.bestModelName}")
print(f"Predictions: {result.predictions}")
```

### Accessing All Model Results

```python
for modelId, mr in result.allModelResults.items():
    print(f"{mr.modelName}: MAPE={mr.mape:.2f}%, RMSE={mr.rmse:.2f}")

```

**Expected output:**

```
AutoETS: MAPE=3.21%, RMSE=10.45
AutoARIMA: MAPE=4.78%, RMSE=15.01
Theta: MAPE=3.89%, RMSE=12.31
DOT: MAPE=4.12%, RMSE=13.02
AutoCES: MAPE=4.34%, RMSE=13.87
MSTL: MAPE=4.56%, RMSE=14.23
...
```

## Available Models

All models below are evaluated automatically when you call `forecast()` or `Vectrix().forecast()`. Vectrix selects the best one based on validation performance — you never need to choose manually, but understanding the options helps interpret results

### Exponential Smoothing

| Model | Class | Best For |
|-------|-------|----------|
| AutoETS | `AutoETS` | Stable patterns, trend, seasonality |
| ETS (manual) | `ETSModel` | When you know the error/trend/seasonal type |

### ARIMA

| Model | Class | Best For |
|-------|-------|----------|
| AutoARIMA | `AutoARIMA` | Stationary and differenced series |

### Decomposition

| Model | Class | Best For |
|-------|-------|----------|
| MSTL | `MSTL` | Multiple seasonal periods |
| AutoMSTL | `AutoMSTL` | Multi-seasonal with auto-detection |

### Theta Methods

| Model | Class | Best For |
|-------|-------|----------|
| Theta | `OptimizedTheta` | General purpose, M3 winner method |
| DOT | `DynamicOptimizedTheta` | General purpose, strongest single model |
| 4Theta | `FourThetaModel` | Multi-theta line ensemble, M4 top-tier |

### Complex Exponential Smoothing

| Model | Class | Best For |
|-------|-------|----------|
| AutoCES | `AutoCES` | Nonlinear patterns, complex dynamics |

### Trigonometric

| Model | Class | Best For |
|-------|-------|----------|
| TBATS | `AutoTBATS` | Complex multi-seasonality |

### Intermittent Demand

| Model | Class | Best For |
|-------|-------|----------|
| Croston | `AutoCroston` | Sparse, intermittent demand |
| SBA | `AutoCroston(variant="sba")` | Bias-corrected Croston |
| TSB | `AutoCroston(variant="tsb")` | Teunter-Syntetos-Babai |

### Volatility

| Model | Class | Best For |
|-------|-------|----------|
| GARCH | `GARCH` | Financial volatility |
| EGARCH | `EGARCH` | Asymmetric volatility |
| GJR-GARCH | `GJRGARCH` | Leverage effects |

### Neural / Reservoir

| Model | Class | Best For |
|-------|-------|----------|
| ESN | `EchoStateNetwork` | Nonlinear dynamics, ensemble diversity |
| DTSF | `DynamicTimeScanForecaster` | Pattern-matching, hourly data |

### Baselines

| Model | Class | Best For |
|-------|-------|----------|
| Naive | `NaiveModel` | Benchmark (last value repeated) |
| Seasonal Naive | `SeasonalNaiveModel` | Benchmark (last season repeated) |
| Mean | `MeanModel` | Benchmark (historical mean) |
| Random Walk with Drift | `RandomWalkDrift` | Trending benchmarks |
| Window Average | `WindowAverage` | Recent-history benchmark |

## Direct Engine Access

For fine-grained control, use individual model engines directly. Each engine follows the same `fit()` → `predict()` interface and returns predictions with 95% confidence intervals

```python
from vectrix.engine.ets import AutoETS
import numpy as np

data = np.array([120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215])

ets = AutoETS(period=12)
ets.fit(data)
predictions, lower, upper = ets.predict(steps=6)

print(f"Predictions: {predictions}")
print(f"Lower 95%: {lower}")
print(f"Upper 95%: {upper}")
```

### Multiple Models Side-by-Side

```python
from vectrix.engine.ets import AutoETS
from vectrix.engine.theta import OptimizedTheta
from vectrix.engine.dot import DynamicOptimizedTheta
import numpy as np

data = np.array([100, 120, 130, 115, 140, 160, 150, 170, 195, 185, 200, 215,
                 110, 125, 135, 120, 145, 165, 155, 175, 200, 190, 210, 225])

models = {
    "AutoETS": AutoETS(period=12),
    "Theta": OptimizedTheta(period=12),
    "DOT": DynamicOptimizedTheta(period=12),
}

for name, model in models.items():
    model.fit(data)
    preds, _, _ = model.predict(steps=6)
    print(f"{name}: {preds.round(1)}")
```

## Flat Prediction Defense

A common failure mode in statistical forecasting is **flat (constant) predictions** — where the model outputs the same value for every future step. This typically happens with mean-reverting models on noisy data. Vectrix includes a unique 4-level defense system that detects and corrects this automatically

```python
from vectrix import Vectrix

vx = Vectrix()
result = vx.forecast(df, dateCol="date", valueCol="value", steps=14)

fr = result.flatRisk
print(f"Risk Level: {fr.riskLevel.name} ({fr.riskScore:.2f})")
print(f"Strategy: {fr.recommendedStrategy}")
```

### The 4 Levels

| Level | Component | What It Does |
|-------|-----------|-------------|
| 1 | FlatRiskDiagnostic | Pre-assessment: estimates risk before fitting |
| 2 | AdaptiveModelSelector | Selection: biases model selection away from flat-prone models |
| 3 | FlatPredictionDetector | Detection: checks if output predictions are flat |
| 4 | FlatPredictionCorrector | Correction: automatically fixes flat predictions |

## Data Characteristics Drive Selection

Vectrix doesn't pick models randomly. It uses **DNA-based meta-learning** — a system that extracts 65+ statistical features from your data and uses them to prioritize the most promising model candidates. The process

1. **Feature extraction** -- 65+ statistical features computed from your data
2. **DNA profiling** -- Features mapped to a difficulty score and category
3. **Model recommendation** -- Category-specific model ranking
4. **Validation** -- All models evaluated on a holdout set, best selected

```python
from vectrix import forecast

result = forecast(data, steps=6)
print(f"Selected: {result.model}")
print(f"Evaluated: {result.models}")
```

## Ensemble Strategy

When no single model clearly dominates — or when multiple models perform similarly — Vectrix combines them into a weighted ensemble. The ensemble typically outperforms any individual model because different models make different errors

```python
from vectrix import Vectrix

vx = Vectrix()
result = vx.forecast(df, dateCol="date", valueCol="value", steps=14)

if hasattr(result, 'ensembleWeights') and result.ensembleWeights:
    print("Ensemble weights:")
    for model, weight in result.ensembleWeights.items():
        print(f"  {model}: {weight:.3f}")
```

The ensemble uses inverse-error weighting: models with lower validation errors receive higher weights.

## Complete Example

```python
import numpy as np
from vectrix import forecast, compare

np.random.seed(42)
t = np.arange(100)
data = 50 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 12) + np.random.randn(100) * 5

ranking = compare(data, steps=12)
print("Top 5 models:")
print(ranking.head(5))

result = forecast(data, steps=12)
print(f"\nSelected model: {result.model}")
print(f"MAPE: {result.mape:.2f}%")
print(f"Next 12 steps: {result.predictions.round(1)}")
```

---
