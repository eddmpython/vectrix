---
title: Business Intelligence
---

# Business Intelligence

Beyond forecasting -- tools for decision-making.

## Anomaly Detection

```python
from vectrix.business import AnomalyDetector

detector = AnomalyDetector()
result = detector.detect(data, method="auto")

print(f"Method: {result.method}")
print(f"Anomalies: {result.nAnomalies}")
print(f"Ratio: {result.anomalyRatio:.1%}")
print(f"Indices: {result.indices}")
```

Methods: `auto`, `zscore`, `iqr`, `rolling`

## What-If Analysis

```python
from vectrix.business import WhatIfAnalyzer

analyzer = WhatIfAnalyzer()
results = analyzer.analyze(base_predictions, historical_data, [
    {"name": "Optimistic", "trendChange": 0.1},
    {"name": "Pessimistic", "trendChange": -0.15},
    {"name": "Shock", "shockAt": 10, "shockMagnitude": -0.3, "shockDuration": 5},
    {"name": "Level Shift", "levelShift": 0.05},
])

for sr in results:
    print(f"{sr.name}: mean={sr.predictions.mean():.2f}, impact={sr.impact:+.1%}")
```

## Backtesting

Walk-forward validation

```python
from vectrix.business import Backtester

bt = Backtester(nFolds=5, horizon=14, strategy='expanding')
result = bt.run(data, model_function)

print(f"Avg MAPE: {result.avgMAPE:.2f}%")
print(f"Avg RMSE: {result.avgRMSE:.2f}")
print(f"Best Fold: #{result.bestFold}")
print(f"Worst Fold: #{result.worstFold}")

for f in result.folds:
    print(f"  Fold {f.fold}: MAPE={f.mape:.2f}%")
```

Strategies: `expanding`, `sliding`

## Business Metrics

```python
from vectrix.business import BusinessMetrics

metrics = BusinessMetrics()
result = metrics.calculate(actual, predicted)

print(f"Bias: {result['bias']:+.2f}")
print(f"WAPE: {result['wape']:.2f}%")
print(f"MASE: {result['mase']:.2f}")
print(f"Accuracy: {result['forecastAccuracy']:.1f}%")
```

| Metric | Key | Description |
|--------|-----|-------------|
| Bias | `bias` | Positive = over-forecast |
| Bias % | `biasPercent` | Percentage bias |
| WAPE | `wape` | Weighted Absolute Percentage Error |
| MASE | `mase` | Below 1 means better than Naive |
| Accuracy | `forecastAccuracy` | Higher is better |
| Over-forecast | `overForecastRatio` | Predicted exceeds Actual ratio |
| Under-forecast | `underForecastRatio` | Predicted below Actual ratio |

---

**Interactive tutorial:** `marimo run docs/tutorials/en/06_business.py`
