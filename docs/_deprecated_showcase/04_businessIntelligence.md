# Showcase 04 — Business Intelligence

**End-to-end business forecasting: anomaly detection, scenarios, backtesting, and metrics.**

## Overview

A complete business forecasting workflow using Vectrix's `business` module:

1. **Anomaly Detection** — Find unusual data points before forecasting
2. **Forecasting** — Run the full 30+ model pipeline
3. **What-If Scenarios** — Explore growth, recession, and shock scenarios
4. **Backtesting** — Walk-forward validation of model accuracy
5. **Business Metrics** — Calculate MAPE, RMSE, MAE, bias, tracking signal

## Run Interactively

```bash
pip install vectrix pandas numpy marimo
marimo run docs/showcase/en/04_businessIntelligence.py
```

## Code

### Setup

```python
import numpy as np
import pandas as pd
from vectrix import forecast
from vectrix.business import AnomalyDetector, WhatIfAnalyzer, Backtester, BusinessMetrics
from vectrix.engine.ets import AutoETS
```

### 1. Anomaly Detection

```python
np.random.seed(42)
n = 150
t = np.arange(n, dtype=np.float64)
values = 200 + 1.2 * t + 30 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 10, n)
values[45] = 600
values[90] = 50
values[130] = 700

detector = AnomalyDetector()
result = detector.detect(values, threshold=2.0)

for idx in result.indices:
    print(f"  Index {idx}: value={values[idx]:.1f}, score={result.scores[idx]:+.2f}")
```

### 2. Forecast

```python
bizDf = pd.DataFrame({
    "date": pd.date_range("2013-01-01", periods=n, freq="MS"),
    "revenue": values,
})

fcResult = forecast(bizDf, date="date", value="revenue", steps=12)
print(f"Model: {fcResult.model}, MAPE: {fcResult.mape:.2f}%")
```

### 3. What-If Scenarios

```python
analyzer = WhatIfAnalyzer()
scenarios = [
    {"name": "base", "trend_change": 0},
    {"name": "growth_10pct", "trend_change": 0.10},
    {"name": "recession", "trend_change": -0.15, "level_shift": -0.05},
    {"name": "supply_shock", "shock_at": 3, "shock_magnitude": -0.25, "shock_duration": 3},
]

historical = np.array(bizDf["revenue"], dtype=np.float64)
results = analyzer.analyze(fcResult.predictions, historical, scenarios, period=12)
print(analyzer.compareSummary(results))
```

### 4. Backtesting

```python
bt = Backtester(nFolds=4, horizon=12, strategy="expanding", minTrainSize=60)
btResult = bt.run(values, modelFactory=AutoETS)
print(bt.summary(btResult))
```

### 5. Business Metrics

```python
actuals = np.array([320, 340, 310, 360, 345])
predicted = np.array([325, 335, 315, 355, 350])

metrics = BusinessMetrics()
result = metrics.calculate(actuals, predicted)
for key, value in result.items():
    print(f"  {key}: {value:.4f}")
```

!!! tip "Best Practice"
    Combine anomaly detection + backtesting + what-if analysis for robust business planning.
    Detect outliers first, validate model accuracy, then explore scenarios.

---

**Back to:** [Showcase Index](index.md)
