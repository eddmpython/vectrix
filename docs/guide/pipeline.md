# Pipeline

The `ForecastPipeline` chains preprocessing transformers with a forecasting model. Transformations are automatically inverted when generating predictions, so your forecasts come back in the original data scale.

## Basic Pipeline

```python
from vectrix.pipeline import ForecastPipeline, Scaler, OutlierClipper

pipe = ForecastPipeline([
    ('clip', OutlierClipper()),
    ('scale', Scaler()),
    ('forecast', MyForecaster()),
])

pipe.fit(y)
predictions, lower, upper = pipe.predict(12)
```

The pipeline:

1. **fit** — sequentially `fitTransform` each transformer, then `fit` the forecaster on transformed data
2. **predict** — get predictions from the forecaster, then `inverseTransform` through all transformers in reverse order

## Built-in Transformers

| Transformer | What it does | Inverse? |
|:--|:--|:--:|
| `Scaler(method='zscore')` | Z-score standardization or MinMax normalization | :white_check_mark: |
| `LogTransformer()` | `log(1 + y)` with automatic shift for negative values | :white_check_mark: |
| `BoxCoxTransformer()` | Optimal Box-Cox lambda estimation via MLE | :white_check_mark: |
| `Differencer(d=1)` | d-th order differencing for stationarity | :white_check_mark: |
| `Deseasonalizer(period=7)` | Remove seasonal component by period averaging | :white_check_mark: |
| `Detrend()` | Remove linear trend | :white_check_mark: |
| `OutlierClipper(factor=3.0)` | IQR-based outlier clipping | :x: |
| `MissingValueImputer(method='linear')` | Fill NaN via linear interpolation, mean, or forward fill | :x: |

## Multi-step Preprocessing

Chain multiple transformers for complex data preparation:

```python
from vectrix.pipeline import (
    ForecastPipeline, MissingValueImputer, OutlierClipper,
    LogTransformer, Deseasonalizer, Scaler
)

pipe = ForecastPipeline([
    ('impute', MissingValueImputer(method='linear')),
    ('clip', OutlierClipper(factor=3.0)),
    ('log', LogTransformer()),
    ('deseason', Deseasonalizer(period=7)),
    ('scale', Scaler(method='zscore')),
    ('forecast', MyForecaster()),
])
```

## Transform Without Forecasting

Use the pipeline as a pure preprocessing tool:

```python
pipe.fit(train_data)

transformed = pipe.transform(test_data)
original = pipe.inverseTransform(transformed)
```

## Inspecting the Pipeline

```python
pipe.listSteps()
# ['impute', 'clip', 'log', 'deseason', 'scale', 'forecast']

scaler = pipe.getStep('scale')
print(scaler._mean, scaler._std)

pipe.getParams()
# {'clip__factor': 3.0, 'scale__method': 'zscore', ...}
```

## Scaler Options

=== "Z-score (default)"

    ```python
    Scaler(method='zscore')
    ```
    Centers to mean=0, std=1. Best for general-purpose standardization.

=== "MinMax"

    ```python
    Scaler(method='minmax')
    ```
    Scales to [0, 1] range. Best when bounded output is needed.

## Box-Cox Transform

Automatically finds the optimal lambda to normalize your data distribution:

```python
from vectrix.pipeline import BoxCoxTransformer

bc = BoxCoxTransformer()       # auto-estimate lambda
bc = BoxCoxTransformer(lmbda=0.5)  # fixed lambda (square root)
```

## Missing Value Strategies

```python
MissingValueImputer(method='linear')  # Linear interpolation (default)
MissingValueImputer(method='mean')    # Replace with series mean
MissingValueImputer(method='ffill')   # Forward fill
```

---

**API Reference:** [Pipeline API](../api/pipeline.md)
