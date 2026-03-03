---
title: Foundation Models
---

# Foundation Models

Vectrix provides optional wrappers for state-of-the-art pretrained forecasting models. These models perform **zero-shot forecasting** — no training required on your specific data.

> **Note:** Foundation model wrappers require additional packages: `pip install "vectrix[foundation]"`

## Amazon Chronos-2

[Chronos](https://github.com/amazon-science/chronos-forecasting) is a family of pretrained probabilistic time series models by Amazon. They tokenize time series into bins and use transformer architectures for forecasting.

```python
from vectrix import ChronosForecaster

model = ChronosForecaster(
    modelId="amazon/chronos-bolt-small",
    device="cpu",
)

model.fit(y)  # stores context only — no training
predictions, lower, upper = model.predict(steps=12)
```

### Available Models

| Model | Parameters | Speed | Accuracy |
|:--|:--|:--|:--|
| `amazon/chronos-bolt-tiny` | 8M | Fastest | Good |
| `amazon/chronos-bolt-small` | 48M | Fast | Better |
| `amazon/chronos-bolt-base` | 205M | Medium | Best |

### Quantile Forecasts

```python
import numpy as np

quantiles = model.predictQuantiles(
    steps=12,
    quantileLevels=[0.1, 0.5, 0.9],
)
# shape: (3, 12) — one row per quantile level
```

### Batch Prediction

Forecast multiple series at once

```python
series = [y1, y2, y3]
results = model.predictBatch(series, steps=12)
# returns list of (predictions, lower, upper) tuples
```

## Google TimesFM 2.5

[TimesFM](https://github.com/google-research/timesfm) is Google's foundation model for time series forecasting, supporting up to 2048 context length.

```python
from vectrix import TimesFMForecaster

model = TimesFMForecaster(
    modelId="google/timesfm-2.5-200m-pytorch",
)

model.fit(y)
predictions, lower, upper = model.predict(steps=12)
```

### With Covariates

TimesFM supports exogenous variables (covariates)

```python
predictions, lower, upper = model.predictWithCovariates(
    steps=12,
    dynamicNumerical=future_numerical_features,  # (steps, n_features)
    dynamicCategorical=future_categorical_features,
)
```

## Checking Availability

```python
from vectrix import CHRONOS_AVAILABLE, TIMESFM_AVAILABLE

if CHRONOS_AVAILABLE:
    model = ChronosForecaster()
else:
    print("Install: pip install 'vectrix[foundation]'")
```

## When to Use Foundation Models

| Scenario | Recommended |
|:--|:--|
| Sufficient historical data (100+ points) | Statistical models (default Vectrix) |
| Cold start / very short series | Foundation models |
| Need explainability | Statistical models |
| Many heterogeneous series | Foundation models |
| Production latency constraints | Statistical models |
