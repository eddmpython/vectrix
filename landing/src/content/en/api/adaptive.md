---
title: Adaptive API
---

# Adaptive API

Regime detection, self-healing forecasts, business constraints, and Forecast DNA.

## RegimeDetector

`RegimeDetector(nRegimes=2)`

Detect statistical regimes using Hidden Markov Models.

### Methods

- `detect(data)` → `RegimeResult`

### RegimeResult

| Attribute | Type | Description |
|---|---|---|
| `.currentRegime` | `int` | Current regime index |
| `.regimeStats` | `list` | Per-regime statistics |
| `.labels` | `np.ndarray` | Regime label per observation |

## RegimeAwareForecaster

`RegimeAwareForecaster()`

Forecast using the current regime context.

### Methods

- `forecast(data, steps, period=None)` → predictions

## ForecastDNA

`ForecastDNA()`

Profile time series characteristics for meta-learning.

### Methods

- `analyze(data, period=None)` → `DNAProfile`

### DNAProfile

| Attribute | Type | Description |
|---|---|---|
| `.category` | `str` | Series type |
| `.difficulty` | `str` | 'easy', 'medium', 'hard' |
| `.difficultyScore` | `float` | 0–100 score |
| `.fingerprint` | `str` | Unique fingerprint code |
| `.recommendedModels` | `list` | Recommended model IDs |
| `.features` | `dict` | Extracted features |

## SelfHealingForecast

Auto-correct forecasts with incoming actuals.

### Methods

- `heal(originalForecast, actuals, historicalData)` → `HealingReport`

### HealingReport

| Attribute | Type | Description |
|---|---|---|
| `.healthScore` | `float` | 0–100 health score |
| `.overallHealth` | `str` | Health status label |
| `.totalCorrected` | `int` | Number of corrections |
| `.correctedForecast` | `np.ndarray` | Updated predictions |

## ConstraintAwareForecaster

`ConstraintAwareForecaster()`

Enforce business constraints on predictions.

### Methods

- `apply(predictions, lower, upper, constraints)` → constrained predictions

## Constraint

`Constraint(type, params)`

| Type | Parameters |
|------|-----------|
| `non_negative` | `{}` |
| `range` | `{'min': 0, 'max': 5000}` |
| `capacity` | `{'capacity': 10000}` |
| `yoy_change` | `{'maxPct': 30, 'historicalData': array}` |
| `sum` | `{'total': 1000}` |
| `monotone` | `{'direction': 'increasing'}` |
| `custom` | `{'func': callable}` |
