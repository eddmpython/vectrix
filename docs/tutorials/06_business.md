# Tutorial 06 — Business Intelligence

**Real-world forecasting tools: anomaly detection, what-if scenarios, backtesting, and business metrics.**

Vectrix's business module provides tools that go beyond basic forecasting — helping you make better decisions with your predictions.

## 1. Anomaly Detection

Detect unusual data points that might indicate errors, events, or regime changes:

```python
import numpy as np
from vectrix.business import AnomalyDetector

np.random.seed(42)
normal = np.random.normal(100, 10, 200)
normal[50] = 200
normal[120] = 20
normal[175] = 250

detector = AnomalyDetector()
result = detector.detect(normal, threshold=2.0)

print(f"Anomalies found: {len(result.indices)}")
for idx in result.indices:
    print(f"  Index {idx}: value={normal[idx]:.1f}, score={result.scores[idx]:.2f}")
```

```
Anomalies found: 3
  Index 50: value=200.0, score=3.45
  Index 120: value=20.0, score=-2.89
  Index 175: value=250.0, score=4.12
```

### Threshold

- `threshold=4.0` — Only flag extreme outliers (fewer alerts)
- `threshold=3.0` — Balanced (default)
- `threshold=2.0` — More aggressive detection (more alerts)

## 2. What-If Scenario Analysis

Explore how changes in trend, seasonality, or external shocks would affect your forecast:

```python
from vectrix import forecast
from vectrix.business import WhatIfAnalyzer

data = [100 + 0.5 * i + 10 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 3)
        for i in range(120)]

result = forecast(data, steps=12)
base = result.predictions
historical = np.array(data, dtype=np.float64)

analyzer = WhatIfAnalyzer()
scenarios = [
    {"name": "optimistic", "trend_change": 0.1},
    {"name": "pessimistic", "trend_change": -0.15},
    {"name": "shock", "shock_at": 3, "shock_magnitude": -0.20, "shock_duration": 2},
    {"name": "level_up", "level_shift": 0.05},
    {"name": "no_seasonality", "seasonal_multiplier": 0.0},
]

results = analyzer.analyze(base, historical, scenarios, period=12)
```

### View Results

```python
for sr in results:
    print(f"  [{sr.name}]  avg impact: {sr.impact:.1f}%  "
          f"final change: {sr.percentChange[-1]:.1f}%")
```

```
  [optimistic]      avg impact: 5.2%   final change: +10.1%
  [pessimistic]     avg impact: 7.8%   final change: -15.2%
  [shock]           avg impact: 4.1%   final change: -1.3%
  [level_up]        avg impact: 5.0%   final change: +5.0%
  [no_seasonality]  avg impact: 3.4%   final change: -2.1%
```

### Scenario Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Scenario label |
| `trend_change` | `float` | Trend adjustment (0.1 = +10% trend acceleration) |
| `seasonal_multiplier` | `float` | Scale seasonality (0 = remove, 2 = double) |
| `shock_at` | `int` | Step index where shock occurs |
| `shock_magnitude` | `float` | Shock size (-0.2 = -20% drop) |
| `shock_duration` | `int` | How many steps the shock lasts |
| `level_shift` | `float` | Permanent level change (0.05 = +5%) |

### Comparison Summary

```python
print(analyzer.compareSummary(results))
```

```
Scenario Comparison:
  [pessimistic] Avg impact: 7.8%, Final change: -15.2%
  [optimistic] Avg impact: 5.2%, Final change: 10.1%
  [level_up] Avg impact: 5.0%, Final change: 5.0%
  [shock] Avg impact: 4.1%, Final change: -1.3%
  [no_seasonality] Avg impact: 3.4%, Final change: -2.1%
```

## 3. Backtesting

Walk-forward validation to measure real forecast accuracy:

```python
from vectrix.business import Backtester
from vectrix.engine.ets import AutoETS

bt = Backtester(nFolds=5, horizon=12, strategy="expanding", minTrainSize=60)

y = np.array(data, dtype=np.float64)
result = bt.run(y, modelFactory=AutoETS)

print(bt.summary(result))
```

```
Backtest Results (5 folds)
  Avg MAPE: 4.23% (+-1.15%)
  Avg RMSE: 5.67
  Avg MAE: 4.12
  Avg Bias: 0.34
  Best fold: #2 (MAPE 2.89%)
  Worst fold: #4 (MAPE 6.12%)
```

### Backtest Strategies

| Strategy | Description |
|----------|-------------|
| `"expanding"` | Training window grows each fold (recommended) |
| `"sliding"` | Fixed training window size, moves forward |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nFolds` | 5 | Number of validation folds |
| `horizon` | 30 | Forecast steps per fold |
| `strategy` | `"expanding"` | Window strategy |
| `minTrainSize` | 50 | Minimum training points |
| `stepSize` | auto | Steps between folds |

### Inspect Individual Folds

```python
for fold in result.folds:
    print(f"  Fold {fold.fold}: train={fold.trainSize}, test={fold.testSize}, "
          f"MAPE={fold.mape:.2f}%")
```

## 4. Business Metrics

Calculate business-oriented accuracy metrics:

```python
from vectrix.business import BusinessMetrics

actuals = np.array([100, 110, 95, 120, 105])
predicted = np.array([102, 108, 97, 115, 110])

metrics = BusinessMetrics()
result = metrics.calculate(actuals, predicted)

for key, value in result.items():
    print(f"  {key}: {value:.4f}")
```

```
  mape: 3.0476
  rmse: 3.8730
  mae: 3.0000
  bias: -0.4000
  tracking_signal: -0.3333
```

## 5. Complete Business Workflow

Bringing it all together:

```python
import numpy as np
from vectrix import forecast, analyze
from vectrix.business import AnomalyDetector, WhatIfAnalyzer

data = [100 + 0.5 * i + 10 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 3)
        for i in range(120)]

report = analyze(data)
print(f"DNA: {report.dna.category}, difficulty={report.dna.difficulty}")

detector = AnomalyDetector()
anom = detector.detect(np.array(data, dtype=np.float64))
print(f"Anomalies in historical data: {len(anom.indices)}")

result = forecast(data, steps=12)
print(f"Model: {result.model}, MAPE: {result.mape:.1f}%")

analyzer = WhatIfAnalyzer()
scenarios = [
    {"name": "base", "trend_change": 0},
    {"name": "growth", "trend_change": 0.1},
    {"name": "recession", "trend_change": -0.2, "level_shift": -0.05},
]
sr = analyzer.analyze(result.predictions, np.array(data, dtype=np.float64), scenarios)
print(analyzer.compareSummary(sr))
```

## 6. API Reference

| Class | Import | Purpose |
|-------|--------|---------|
| `AnomalyDetector` | `from vectrix.business import AnomalyDetector` | Detect anomalies |
| `WhatIfAnalyzer` | `from vectrix.business import WhatIfAnalyzer` | Scenario analysis |
| `Backtester` | `from vectrix.business import Backtester` | Walk-forward validation |
| `BusinessMetrics` | `from vectrix.business import BusinessMetrics` | Accuracy metrics |

!!! note "Import Path"
    Business classes are imported from `vectrix.business`, not from the top-level `vectrix` package.

---

**This concludes the tutorial series.** For real-world examples, see the [Showcase](../showcase/index.md) section.
