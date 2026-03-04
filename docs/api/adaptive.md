---
title: Adaptive API
---

# Adaptive API

Regime detection, self-healing forecasts, business constraints, and Forecast DNA.

## RegimeDetector

`RegimeDetector(nRegimes=2)`

Detect statistical regimes using Hidden Markov Models.

### Methods

- `detect(y)` → `RegimeResult`

### RegimeResult

| Attribute | Type | Description |
|---|---|---|
| `.currentRegime` | `str` | Current regime label |
| `.regimeStats` | `dict` | Per-regime statistics (mean, std, etc.) |
| `.states` | `np.ndarray` | Regime index per observation |
| `.labels` | `list[str]` | Regime label per observation |
| `.transitionMatrix` | `np.ndarray` | K x K transition probability matrix |
| `.nRegimes` | `int` | Number of detected regimes |

## RegimeAwareForecaster

`RegimeAwareForecaster()`

Forecast using the current regime context.

### Methods

- `forecast(y, steps, period=7)` → `RegimeForecastResult`

## ForecastDNA

`ForecastDNA()`

Profile time series characteristics for meta-learning.

### Methods

- `analyze(y, period=1)` → `DNAProfile`

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

`SelfHealingForecast(predictions, lower95, upper95, historicalData, period=7, healingMode='adaptive')`

Auto-correct forecasts as actual data arrives.

### Methods

- `observe(actuals)` → `HealingStatus` — Feed actual values as they arrive
- `getUpdatedForecast()` → `Tuple[np.ndarray, np.ndarray, np.ndarray]` — (predictions, lower95, upper95)
- `getReport()` → `HealingReport` — Full healing process report

### HealingReport

| Attribute | Type | Description |
|---|---|---|
| `.overallHealth` | `str` | `excellent`, `good`, `fair`, `poor` |
| `.healthScore` | `float` | 0–100 health score |
| `.totalObserved` | `int` | Number of actual values received |
| `.totalCorrected` | `int` | Number of corrections applied |
| `.improvementPct` | `float` | Error reduction percentage |

## ConstraintAwareForecaster

`ConstraintAwareForecaster()`

Enforce business constraints on predictions.

### Methods

- `apply(predictions, lower95, upper95, constraints, smoothing=True)` → `ConstraintResult`

## Constraint

`Constraint(type, params)`

| Type | Parameters |
|------|-----------|
| `non_negative` | `{}` |
| `range` | `{'min': 0, 'max': 5000}` |
| `capacity` | `{'capacity': 10000}` |
| `yoy_change` | `{'maxPct': 30, 'historicalData': array}` |
| `sum_constraint` | `{'window': 7, 'maxSum': 10000}` |
| `monotone` | `{'direction': 'increasing'}` |
| `custom` | `{'func': callable}` |
