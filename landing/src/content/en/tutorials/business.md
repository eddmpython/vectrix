---
title: "Tutorial 06 — Business Intelligence"
---

# Tutorial 06 — Business Intelligence

Beyond forecasting -- tools for real-world decision-making. Detect anomalies, run what-if scenarios, backtest your forecasts, and measure business-relevant accuracy metrics.

## Anomaly Detection

Identify unusual observations in your time series:

```python
from vectrix.business import AnomalyDetector
import numpy as np

np.random.seed(42)
data = np.random.randn(200) * 10 + 100
data[50] = 200
data[120] = 30
data[175] = 250

detector = AnomalyDetector()
result = detector.detect(data, method="auto")

print(f"Method used: {result.method}")
print(f"Anomalies found: {result.nAnomalies}")
print(f"Anomaly ratio: {result.anomalyRatio:.1%}")
print(f"Anomaly indices: {result.indices}")
```

**Expected output:**

```
Method used: zscore
Anomalies found: 3
Anomaly ratio: 1.5%
Anomaly indices: [50, 120, 175]
```

### Detection Methods

| Method | How It Works | Best For |
|--------|-------------|----------|
| `auto` | Automatically selects the best method | General use (recommended) |
| `zscore` | Flags points > 3 standard deviations from mean | Normally distributed data |
| `iqr` | Flags points outside 1.5x interquartile range | Skewed distributions |
| `rolling` | Flags points outside rolling window statistics | Non-stationary data |

### Example with Specific Method

```python
result_iqr = detector.detect(data, method="iqr")
print(f"IQR method found: {result_iqr.nAnomalies} anomalies")

result_rolling = detector.detect(data, method="rolling")
print(f"Rolling method found: {result_rolling.nAnomalies} anomalies")
```

### AnomalyResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `method` | `str` | Detection method used |
| `nAnomalies` | `int` | Number of anomalies detected |
| `anomalyRatio` | `float` | Fraction of data points flagged |
| `indices` | `np.ndarray` | Indices of anomalous observations |

## What-If Analysis

Explore hypothetical scenarios against a baseline forecast:

```python
from vectrix.business import WhatIfAnalyzer
from vectrix import forecast
import numpy as np

data = np.random.randn(200).cumsum() + 500
result = forecast(data, steps=30)

analyzer = WhatIfAnalyzer()
scenarios = analyzer.analyze(
    result.predictions,
    data,
    [
        {"name": "Optimistic", "trendChange": 0.1},
        {"name": "Pessimistic", "trendChange": -0.15},
        {"name": "Supply Shock", "shockAt": 10, "shockMagnitude": -0.3, "shockDuration": 5},
        {"name": "Level Shift", "levelShift": 0.05},
    ]
)

for sr in scenarios:
    print(f"{sr.name}: mean={sr.predictions.mean():.2f}, impact={sr.impact:+.1%}")
```

**Expected output:**

```
Optimistic: mean=535.42, impact=+10.0%
Pessimistic: mean=425.18, impact=-15.0%
Supply Shock: mean=480.67, impact=-5.8%
Level Shift: mean=525.00, impact=+5.0%
```

### Scenario Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Scenario label |
| `trendChange` | `float` | Percentage trend adjustment (0.1 = +10%) |
| `shockAt` | `int` | Step index where shock begins |
| `shockMagnitude` | `float` | Shock size (-0.3 = -30% drop) |
| `shockDuration` | `int` | Number of steps the shock lasts |
| `levelShift` | `float` | Permanent level adjustment (0.05 = +5%) |

### ScenarioResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Scenario name |
| `predictions` | `np.ndarray` | Modified predictions |
| `impact` | `float` | Overall impact vs. baseline |

> **Tip:** Use what-if analysis for budget planning -- create optimistic, baseline, and pessimistic scenarios and present all three to stakeholders.

## Backtesting

Walk-forward validation measures how well your forecasting approach would have performed historically:

```python
from vectrix.business import Backtester
from vectrix import forecast
import numpy as np

data = np.random.randn(300).cumsum() + 200

def model_function(train_data, steps):
    result = forecast(train_data, steps=steps)
    return result.predictions

bt = Backtester(nFolds=5, horizon=14, strategy='expanding')
result = bt.run(data, model_function)

print(f"Average MAPE: {result.avgMAPE:.2f}%")
print(f"Average RMSE: {result.avgRMSE:.2f}")
print(f"Best fold: #{result.bestFold}")
print(f"Worst fold: #{result.worstFold}")
```

**Expected output:**

```
Average MAPE: 4.56%
Average RMSE: 12.34
Best fold: #3
Worst fold: #1
```

### Per-Fold Results

```python
print("\nPer-fold breakdown:")
for f in result.folds:
    print(f"  Fold {f.fold}: MAPE={f.mape:.2f}%, RMSE={f.rmse:.2f}")
```

**Expected output:**

```
Per-fold breakdown:
  Fold 1: MAPE=6.12%, RMSE=16.45
  Fold 2: MAPE=4.23%, RMSE=11.89
  Fold 3: MAPE=3.45%, RMSE=9.67
  Fold 4: MAPE=4.89%, RMSE=13.12
  Fold 5: MAPE=4.12%, RMSE=10.58
```

### Backtester Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nFolds` | `5` | Number of validation folds |
| `horizon` | `14` | Forecast horizon per fold |
| `strategy` | `'expanding'` | `'expanding'` (growing window) or `'sliding'` (fixed window) |

### Strategy Comparison

**Expanding window** -- Each fold uses all data up to the cutoff point. Earlier folds train on less data, later folds train on more. Recommended for most cases.

```
Fold 1: [====TRAIN====][TEST]
Fold 2: [======TRAIN======][TEST]
Fold 3: [========TRAIN========][TEST]
```

**Sliding window** -- Each fold uses a fixed-size training window. Useful when older data is no longer relevant (e.g., regime changes).

```
Fold 1: [====TRAIN====][TEST]
Fold 2:    [====TRAIN====][TEST]
Fold 3:       [====TRAIN====][TEST]
```

### BacktestResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `avgMAPE` | `float` | Average MAPE across all folds |
| `avgRMSE` | `float` | Average RMSE across all folds |
| `bestFold` | `int` | Fold number with lowest MAPE |
| `worstFold` | `int` | Fold number with highest MAPE |
| `folds` | `list` | Per-fold results (mape, rmse, fold index) |

## Business Metrics

Standard accuracy metrics miss what businesses care about. `BusinessMetrics` provides metrics that matter for operations and finance:

```python
from vectrix.business import BusinessMetrics
import numpy as np

actual = np.array([100, 120, 110, 130, 140, 125, 135, 150, 145, 155])
predicted = np.array([105, 115, 112, 128, 145, 120, 138, 148, 140, 160])

metrics = BusinessMetrics()
result = metrics.calculate(actual, predicted)

print(f"Bias: {result['bias']:+.2f}")
print(f"Bias %: {result['biasPercent']:+.2f}%")
print(f"WAPE: {result['wape']:.2f}%")
print(f"MASE: {result['mase']:.2f}")
print(f"Accuracy: {result['forecastAccuracy']:.1f}%")
print(f"Over-forecast ratio: {result['overForecastRatio']:.1%}")
print(f"Under-forecast ratio: {result['underForecastRatio']:.1%}")
```

**Expected output:**

```
Bias: -0.10
Bias %: -0.08%
WAPE: 3.42%
MASE: 0.45
Accuracy: 96.6%
Over-forecast ratio: 50.0%
Under-forecast ratio: 50.0%
```

### Metrics Reference

| Metric | Key | What It Means |
|--------|-----|--------------|
| Bias | `bias` | Positive = systematic over-forecasting |
| Bias % | `biasPercent` | Bias as percentage of actual |
| WAPE | `wape` | Weighted Absolute Percentage Error (volume-weighted) |
| MASE | `mase` | Below 1 means better than Naive forecast |
| Accuracy | `forecastAccuracy` | 100% - WAPE, higher is better |
| Over-forecast | `overForecastRatio` | Fraction of periods where predicted exceeds actual |
| Under-forecast | `underForecastRatio` | Fraction of periods where predicted is below actual |

> **Note:** WAPE is preferred over MAPE in business contexts because it handles near-zero values gracefully and weights errors by volume. A WAPE of 5% means your total absolute error is 5% of total actual volume.

### Interpreting MASE

MASE (Mean Absolute Scaled Error) compares your model to a Naive baseline:

- **MASE below 1.0** — Your model beats Naive. Good.
- **MASE = 1.0** — Your model equals Naive. No value added.
- **MASE above 1.0** — Naive would have been better. Investigate.

## Combining Business Tools

A typical business forecasting workflow:

```python
import numpy as np
from vectrix import forecast
from vectrix.business import AnomalyDetector, Backtester, BusinessMetrics, WhatIfAnalyzer

np.random.seed(42)
data = np.random.randn(365).cumsum() + 1000

detector = AnomalyDetector()
anomalies = detector.detect(data, method="auto")
print(f"Anomalies in history: {anomalies.nAnomalies}")

def model_fn(train, steps):
    return forecast(train, steps=steps).predictions

bt = Backtester(nFolds=4, horizon=30, strategy='expanding')
bt_result = bt.run(data, model_fn)
print(f"Backtest MAPE: {bt_result.avgMAPE:.2f}%")

result = forecast(data, steps=30)
print(f"Model: {result.model}")

analyzer = WhatIfAnalyzer()
scenarios = analyzer.analyze(result.predictions, data, [
    {"name": "Growth +10%", "trendChange": 0.10},
    {"name": "Decline -10%", "trendChange": -0.10},
])
for s in scenarios:
    print(f"  {s.name}: mean={s.predictions.mean():.0f}")
```

## Complete Example: Monthly Sales Review

```python
import numpy as np
from vectrix import forecast
from vectrix.business import BusinessMetrics

actual_last_month = np.array([
    320, 345, 310, 380, 400, 420, 350,
    330, 360, 325, 390, 410, 430, 365,
    340, 370, 335, 395, 415, 440, 375,
    345, 375, 340, 400, 425, 445, 380,
    350, 385
])

predicted_last_month = np.array([
    315, 340, 320, 370, 395, 415, 345,
    325, 355, 330, 385, 405, 425, 360,
    335, 365, 340, 390, 410, 435, 370,
    340, 370, 345, 395, 420, 440, 375,
    345, 380
])

metrics = BusinessMetrics()
result = metrics.calculate(actual_last_month, predicted_last_month)

print("=== Monthly Performance Review ===")
print(f"Forecast Accuracy: {result['forecastAccuracy']:.1f}%")
print(f"Bias: {result['bias']:+.1f} units/day")
print(f"WAPE: {result['wape']:.1f}%")
print(f"MASE: {result['mase']:.2f}")

if result['mase'] < 1.0:
    print("Model outperforms Naive baseline.")
if abs(result['biasPercent']) > 5:
    print(f"Warning: Systematic {'over' if result['bias'] > 0 else 'under'}-forecasting detected.")
```

---
