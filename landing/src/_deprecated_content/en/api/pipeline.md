---
title: Pipeline API
---

# Pipeline API

## ForecastPipeline

`ForecastPipeline(steps)`

Sequential pipeline that chains transformers with a forecaster.

### Methods

- `fit(y)` → self
- `predict(steps)` → (predictions, lower, upper)
- `transform(y)` → transformed data
- `inverseTransform(y)` → original scale data
- `listSteps()` → list of step names
- `getStep(name)` → transformer/forecaster instance
- `getParams()` → dict of all parameters

## Transformers

All transformers implement `fitTransform(y)` and `inverseTransform(y)`.

### Scaler

`Scaler(method='zscore')`

Methods: `'zscore'` (mean=0, std=1), `'minmax'` (0-1 range)

### LogTransformer

`LogTransformer()`

`log(1 + y)` with automatic shift for negative values.

### BoxCoxTransformer

`BoxCoxTransformer(lmbda=None)`

Optimal Box-Cox lambda via MLE. Pass `lmbda` to fix.

### Differencer

`Differencer(d=1)`

d-th order differencing.

### Deseasonalizer

`Deseasonalizer(period=7)`

Remove seasonal component by period averaging.

### Detrend

`Detrend()`

Remove linear trend.

### OutlierClipper

`OutlierClipper(factor=3.0)`

IQR-based outlier clipping. No inverse transform.

### MissingValueImputer

`MissingValueImputer(method='linear')`

Methods: `'linear'`, `'mean'`, `'ffill'`. No inverse transform.
