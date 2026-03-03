---
title: Business Intelligence
---

# Business Intelligence

End-to-end business workflow: detect anomalies, forecast the future, run what-if scenarios, backtest your models, and measure business impact.

## Anomaly Detection

Identify unusual patterns in your time series automatically.

```python
import numpy as np
from vectrix.business import AnomalyDetector

data = np.random.normal(100, 10, 365)
data[50] = 250
data[200] = 20

detector = AnomalyDetector()
result = detector.detect(data, method="auto")

print(f"Method used: {result.method}")
print(f"Anomalies found: {result.nAnomalies}")
print(f"Anomaly ratio: {result.anomalyRatio:.1%}")
print(f"Indices: {result.indices}")
```

Available detection methods

| Method | Description |
|--------|-------------|
| `auto` | Automatically selects the best method |
| `zscore` | Z-score threshold (default 3.0) |
| `iqr` | Interquartile range method |
| `rolling` | Rolling window deviation |

## Forecasting

Generate a baseline forecast to anchor your business analysis

```python
from vectrix import forecast

result = forecast(data, steps=30)
print(f"Model: {result.model}")
print(f"Next 30 days: mean = {result.predictions.mean():.1f}")
```

## What-If Scenarios

Test how changes in conditions affect the forecast.

```python
from vectrix.business import WhatIfAnalyzer

analyzer = WhatIfAnalyzer()

scenarios = [
    {"name": "Base Case", "adjustment": 1.0},
    {"name": "10% Growth", "adjustment": 1.10},
    {"name": "20% Decline", "adjustment": 0.80},
    {"name": "Shock Event", "adjustment": 0.50},
]

for scenario in scenarios:
    adjusted = result.predictions * scenario["adjustment"]
    print(f"{scenario['name']:20s}  mean={adjusted.mean():8.1f}")
```

Using WhatIfAnalyzer for more advanced scenario modeling

```python
analyzer = WhatIfAnalyzer()
whatIfResults = analyzer.analyze(
    result.predictions,
    data,
    [
        {"name": "Growth +10%", "trendChange": 0.10},
        {"name": "Decline -20%", "trendChange": -0.20},
    ],
)
for sr in whatIfResults:
    print(f"{sr.name}: mean={sr.predictions.mean():.1f}, impact={sr.impact:+.1%}")
```

## Backtesting

Validate forecast accuracy with rolling origin cross-validation.

```python
from vectrix.business import Backtester
from vectrix.engine.ets import AutoETS

backtester = Backtester(nFolds=5, horizon=12)
btResult = backtester.run(data, lambda: AutoETS())

print(f"Avg MAE:   {btResult.avgMAE:.2f}")
print(f"Avg RMSE:  {btResult.avgRMSE:.2f}")
print(f"Avg MAPE:  {btResult.avgMAPE:.2f}%")
print(f"Avg sMAPE: {btResult.avgSMAPE:.2f}%")
```

Backtesting creates multiple train/test splits by sliding the origin forward, giving a realistic estimate of out-of-sample performance.

## Business Metrics

Translate statistical accuracy into business-relevant KPIs.

```python
from vectrix.business import BusinessMetrics

metrics = BusinessMetrics()
result = metrics.calculate(actualSales, forecastedSales)

print(f"Bias: {result['bias']:+.2f}")
print(f"Bias %: {result['biasPercent']:+.2f}%")
print(f"WAPE: {result['wape']:.2f}%")
print(f"Accuracy: {result['forecastAccuracy']:.1f}%")
```

## Full Business Workflow

Putting it all together in a single pipeline

```python
import pandas as pd
from vectrix import forecast
from vectrix.business import AnomalyDetector, Backtester, BusinessMetrics
from vectrix.engine.ets import AutoETS

df = pd.read_csv("daily_sales.csv")
data = df["sales"].values

detector = AnomalyDetector()
anomalies = detector.detect(data, method="auto")
print(f"Step 1 - Anomalies: {anomalies.nAnomalies} detected")

result = forecast(data, steps=30)
print(f"Step 2 - Forecast: {result.model} selected")

backtester = Backtester(nFolds=5, horizon=30)
btResult = backtester.run(data, lambda: AutoETS())
print(f"Step 3 - Backtest: MAE={btResult.avgMAE:.2f}, MAPE={btResult.avgMAPE:.2f}%")

metrics = BusinessMetrics()
bizMetrics = metrics.calculate(data[-30:], result.predictions[:30])
print(f"Step 4 - Accuracy: {bizMetrics['forecastAccuracy']:.1f}%, Bias: {bizMetrics['bias']:+.2f}")
```

> **Note:** All business intelligence tools work with raw numpy arrays, pandas Series, or DataFrames. No special data format is required.
