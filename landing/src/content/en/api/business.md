---
title: Business API
---

# Business API

Anomaly detection, what-if analysis, backtesting, and business metrics.

## AnomalyDetector

`AnomalyDetector()`

### Methods

- `detect(data, method="auto", threshold=3.0)` → `AnomalyResult`

### AnomalyResult

| Attribute | Type | Description |
|---|---|---|
| `.indices` | `np.ndarray` | Anomaly indices |
| `.scores` | `np.ndarray` | Anomaly scores |
| `.method` | `str` | Method used |
| `.nAnomalies` | `int` | Count |
| `.anomalyRatio` | `float` | Ratio |

Methods: `auto`, `zscore`, `iqr`, `rolling`

## WhatIfAnalyzer

`WhatIfAnalyzer()`

### Methods

- `analyze(basePredictions, historicalData, scenarios, period=7)` → list of `ScenarioResult`
- `compareSummary(results)` → `str`

### Scenario Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Scenario label |
| `trend_change` | `float` | Trend adjustment |
| `seasonal_multiplier` | `float` | Scale seasonality |
| `shock_at` | `int` | Shock step index |
| `shock_magnitude` | `float` | Shock size |
| `shock_duration` | `int` | Shock length |
| `level_shift` | `float` | Permanent level change |

## Backtester

`Backtester(nFolds=5, horizon=30, strategy='expanding', minTrainSize=50)`

### Methods

- `run(y, modelFactory)` → `BacktestResult`
  - `y`: Full time series (ndarray)
  - `modelFactory`: Zero-argument callable that returns a model with `.fit(y)` and `.predict(steps)` methods

### BacktestResult

| Attribute | Type | Description |
|---|---|---|
| `.avgMAPE` | `float` | Average MAPE |
| `.avgRMSE` | `float` | Average RMSE |
| `.folds` | `list` | Per-fold results |
| `.bestFold` | `int` | Best fold number |
| `.worstFold` | `int` | Worst fold number |

## BusinessMetrics

`BusinessMetrics()`

### Methods

- `calculate(actual, predicted)` → `dict`

Returns: `bias`, `biasPercent`, `trackingSignal`, `wape`, `mase`, `overForecastRatio`, `underForecastRatio`, `fillRateImpact`, `forecastAccuracy`
