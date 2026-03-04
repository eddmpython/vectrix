---
title: Business Intelligence
---

# Business Intelligence

**Forecasting is only the first step.** Real-world decision-making requires anomaly detection to clean your data, what-if scenarios for planning, backtesting to validate your approach, and business-specific accuracy metrics that go beyond MAPE. Vectrix's BI module provides all four.

## Anomaly Detection

Before forecasting, identify unusual observations that could distort model training:

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

Explore hypothetical scenarios against your baseline forecast — essential for budget planning, risk assessment, and stakeholder presentations:

```python
from vectrix.business import WhatIfAnalyzer

analyzer = WhatIfAnalyzer()
results = analyzer.analyze(base_predictions, historical_data, [
    {"name": "Optimistic", "trend_change": 0.1},
    {"name": "Pessimistic", "trend_change": -0.15},
    {"name": "Shock", "shock_at": 10, "shock_magnitude": -0.3, "shock_duration": 5},
    {"name": "Level Shift", "level_shift": 0.05},
])

for sr in results:
    print(f"{sr.name}: mean={sr.predictions.mean():.2f}, impact={sr.impact:+.1f}%")
```

## Backtesting

How do you know your forecasting approach works? **Walk-forward validation** simulates historical performance by repeatedly training and predicting:

```python
from vectrix.business import Backtester

from vectrix.engine.ets import AutoETS

bt = Backtester(nFolds=5, horizon=14, strategy='expanding')
result = bt.run(data, lambda: AutoETS())

print(f"Avg MAPE: {result.avgMAPE:.2f}%")
print(f"Avg RMSE: {result.avgRMSE:.2f}")
print(f"Best Fold: #{result.bestFold}")
print(f"Worst Fold: #{result.worstFold}")

for f in result.folds:
    print(f"  Fold {f.fold}: MAPE={f.mape:.2f}%")
```

`modelFactory` must be a callable that returns a model instance with `.fit(y)` and `.predict(steps)` methods.

Strategies: `expanding`, `sliding`

## Business Metrics

MAPE and RMSE tell you about statistical accuracy, but businesses need different answers: **systematic bias detection, volume-weighted error, and naive baseline comparison:**

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
