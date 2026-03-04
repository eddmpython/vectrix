---
title: Forecasting
---

# Forecasting

Vectrix provides two APIs for forecasting: the **Easy API** for quick, one-line usage, and the **Vectrix class** for full control over the forecasting pipeline.

## Easy API

The simplest way to forecast. One function call evaluates 30+ models, selects the best, and returns predictions with confidence intervals

```python
from vectrix import forecast

result = forecast(data, steps=30)
```

`forecast()` accepts lists, NumPy arrays, pandas DataFrames, Series, and CSV file paths. It automatically detects date/value columns, splits data for validation, and selects the model with the lowest validation error.

### Full Parameters

```python
result = forecast(
    data,
    date=None,           # date column name
    value=None,          # value column name
    steps=30,            # forecast horizon
    verbose=False,       # print progress
    models=None,         # list of model IDs to evaluate
    ensemble=None,       # 'mean', 'weighted', 'median', 'best'
    confidence=0.95      # CI level: 0.80, 0.90, 0.95, 0.99
)
```

### Model Selection

Restrict evaluation to specific models

```python
result = forecast(data, steps=14, models=['dot', 'auto_ets', 'auto_ces'])
```

### Ensemble Methods

Combine multiple models instead of selecting the single best

```python
result = forecast(data, steps=14, ensemble='weighted')
```

## Vectrix Class

For full access to all model results, flat risk diagnostics, ensemble weights, and per-model metrics, use the `Vectrix` class directly

```python
from vectrix import Vectrix

vx = Vectrix()
result = vx.forecast(
    df,
    dateCol="date",
    valueCol="sales",
    steps=14,
    trainRatio=0.8,
    models=None,             # list[str] | None
    ensembleMethod=None,     # str | None
    confidenceLevel=0.95     # float
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

Vectrix evaluates models across 10 categories. Each category captures different time series dynamics

| Category | Models | Best For |
|----------|--------|----------|
| **Exponential Smoothing** | AutoETS, ETS variants | Stable patterns with trend and seasonality |
| **ARIMA** | AutoARIMA | Stationary and differenced series |
| **Decomposition** | MSTL, AutoMSTL | Multiple seasonal periods (daily + weekly + yearly) |
| **Theta** | Theta, DOT, 4Theta | General purpose — DOT is often the strongest single model |
| **Complex ES** | AutoCES | Nonlinear and complex dynamics |
| **Trigonometric** | TBATS | Complex multi-seasonality with non-integer periods |
| **Intermittent** | Croston, SBA, TSB | Sparse demand with many zeros |
| **Volatility** | GARCH, EGARCH, GJR | Financial data with time-varying variance |
| **Neural/Reservoir** | ESN, DTSF | Nonlinear dynamics, pattern matching |
| **Baselines** | Naive, Seasonal, Mean, RWD | Benchmarks — if your model can't beat these, something is wrong |

## Flat Defense System

A common failure in statistical forecasting is flat (constant) predictions. Vectrix includes a unique 4-level defense system that detects and corrects this automatically

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

For fine-grained control, use individual model engines. Every engine follows the same `fit()` → `predict()` interface

```python
from vectrix.engine.ets import AutoETS
from vectrix.engine.arima import AutoARIMA
from vectrix.engine.theta import OptimizedTheta

ets = AutoETS(period=7)
ets.fit(data)
predictions, lower, upper = ets.predict(steps=30)
```
