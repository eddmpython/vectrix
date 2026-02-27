# Multivariate Forecasting

Vectrix supports multivariate time series forecasting through Vector AutoRegression (VAR) and Vector Error Correction Models (VECM).

## VAR Model

VAR models capture the linear interdependencies between multiple time series. Each variable is modeled as a linear combination of its own lagged values and the lagged values of all other variables.

```python
from vectrix.engine.var import VARModel
import numpy as np

# Y shape: (T, k) — T time steps, k variables
Y = np.column_stack([sales, advertising, inventory])

model = VARModel(maxLag=5, criterion="aic")
model.fit(Y)

predictions, lower, upper = model.predict(steps=12)
# predictions shape: (12, 3) — 12 steps, 3 variables
```

### Automatic Lag Selection

The model tests all lag orders from 1 to `maxLag` and selects the best one based on information criteria:

```python
model = VARModel(maxLag=8, criterion="aic")   # Akaike IC (default)
model = VARModel(maxLag=8, criterion="bic")   # Bayesian IC (sparser)

model.fit(Y)
print(f"Selected lag order: {model.order}")
```

### Granger Causality

Test whether one variable helps predict another:

```python
model = VARModel()
result = model.grangerCausality(Y, cause=0, effect=1, maxLag=5)

print(f"F-statistic: {result['fStat']:.3f}")
print(f"p-value: {result['pValue']:.4f}")

if result['pValue'] < 0.05:
    print("Variable 0 Granger-causes Variable 1")
```

## VECM Model

Vector Error Correction Models are used when variables are **cointegrated** — they share a long-run equilibrium relationship despite being individually non-stationary.

```python
from vectrix.engine.var import VECMModel

# Cointegrated series example: prices of related assets
prices = np.column_stack([stock_a, stock_b])

model = VECMModel(maxLag=4)
model.fit(prices)

predictions, lower, upper = model.predict(steps=10)
```

### When to Use VECM vs VAR

| Data characteristic | Use |
|:--|:--|
| Stationary series | VAR |
| Non-stationary but cointegrated | VECM |
| Non-stationary, no cointegration | VAR on differenced data |
| Unknown | Try both, compare results |

### Cointegration Rank

VECM automatically estimates the cointegration rank (number of independent cointegrating relationships):

```python
model = VECMModel(maxLag=4, rank=None)  # auto-detect
model = VECMModel(maxLag=4, rank=1)     # force rank=1

model.fit(Y)
print(f"Cointegration rank: {model._rank}")
```

## Complete Example

```python
import numpy as np
from vectrix.engine.var import VARModel

rng = np.random.default_rng(42)
T = 200

# Generate two related series
Y = np.zeros((T, 2))
for t in range(1, T):
    Y[t, 0] = 0.5 * Y[t-1, 0] + 0.2 * Y[t-1, 1] + rng.normal(0, 0.1)
    Y[t, 1] = 0.3 * Y[t-1, 0] + 0.4 * Y[t-1, 1] + rng.normal(0, 0.1)

# Fit and predict
model = VARModel(maxLag=5, criterion="bic")
model.fit(Y)

print(f"Selected order: {model.order}")

pred, lo, hi = model.predict(steps=10)
print(f"Predictions shape: {pred.shape}")  # (10, 2)

# Granger causality
gc = model.grangerCausality(Y, cause=0, effect=1, maxLag=3)
print(f"Does series 0 cause series 1? p={gc['pValue']:.4f}")
```

---

**API Reference:** [VAR/VECM API](../api/multivariate.md)
