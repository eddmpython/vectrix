---
title: Forecasting
---

# Forecasting

## Easy API

The simplest way to forecast:

```python
from vectrix import forecast

result = forecast(data, steps=30)
```

`forecast()` accepts lists, numpy arrays, pandas DataFrames, Series, dicts, and CSV file paths. It automatically selects the best model from 30+ candidates.

## Vectrix Class

For more control, use the `Vectrix` class directly:

```python
from vectrix import Vectrix

vx = Vectrix()
result = vx.forecast(
    df,
    dateCol="date",
    valueCol="sales",
    steps=14,
    period=7,
    trainRatio=0.8
)

print(result.bestModelName)
print(result.predictions)
```

### All Model Results

```python
for modelId, mr in result.allModelResults.items():
    print(f"{mr.modelName}: MAPE={mr.mape:.2f}%")
```

## Model Categories

| Category | Models | Best For |
|----------|--------|----------|
| **Exponential Smoothing** | AutoETS, ETS variants | Stable patterns |
| **ARIMA** | AutoARIMA | Stationary series |
| **Decomposition** | MSTL, AutoMSTL | Multi-seasonal |
| **Theta** | Theta, DOT | General purpose |
| **Trigonometric** | TBATS | Complex seasonality |
| **Complex** | AutoCES | Nonlinear patterns |
| **Intermittent** | Croston, SBA, TSB | Sparse demand |
| **Volatility** | GARCH, EGARCH, GJR | Financial data |
| **Baselines** | Naive, Seasonal, Mean, RWD | Benchmarks |

## Flat Defense System

Vectrix includes a unique 4-level system to prevent flat (constant) predictions:

1. **FlatRiskDiagnostic** -- Pre-assessment of flat prediction risk
2. **AdaptiveModelSelector** -- Risk-based model selection
3. **FlatPredictionDetector** -- Post-prediction flat detection
4. **FlatPredictionCorrector** -- Automatic correction of flat predictions

```python
result = vx.forecast(df, dateCol="date", valueCol="value", steps=14)
fr = result.flatRisk
print(f"Risk: {fr.riskLevel.name} ({fr.riskScore:.2f})")
print(f"Strategy: {fr.recommendedStrategy}")
```

## Direct Engine Access

Use individual models directly:

```python
from vectrix.engine import AutoETS, AutoARIMA, ThetaModel

ets = AutoETS(period=7)
ets.fit(data)
predictions, lower, upper = ets.predict(steps=30)
```

---

**Interactive tutorial:** `marimo run docs/tutorials/en/04_models.py`
