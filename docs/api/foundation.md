---
title: Foundation Models API
---

# Foundation Models API

## ChronosForecaster

`ChronosForecaster(modelId="amazon/chronos-bolt-small", device="cpu")`

Amazon Chronos-2 wrapper.

### Methods

- `fit(y)` → self (stores context, no training)
- `predict(steps)` → (predictions, lower, upper)
- `predictQuantiles(steps, quantileLevels)` → ndarray
- `predictBatch(series_list, steps)` → list of tuples

## TimesFMForecaster

`TimesFMForecaster(modelId="google/timesfm-2.5-200m-pytorch")`

Google TimesFM 2.5 wrapper.

### Methods

- `fit(y)` → self
- `predict(steps)` → (predictions, lower, upper)
- `predictWithCovariates(steps, dynamicNumerical, dynamicCategorical)` → tuple

## NeuralForecaster

`NeuralForecaster(architecture="nbeats", **kwargs)`

NeuralForecast wrapper.

### Convenience Classes

- `NBEATSForecaster(**kwargs)`
- `NHITSForecaster(**kwargs)`
- `TFTForecaster(**kwargs)`

## Availability Flags

```python
from vectrix import CHRONOS_AVAILABLE, TIMESFM_AVAILABLE, NEURALFORECAST_AVAILABLE
```

Boolean flags indicating whether optional dependencies are installed.
