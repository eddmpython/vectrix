---
title: Regression
---

# Regression

## R-style Formula

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ ads + price + promo")
print(model.summary())
```

## Methods

| Method | Description |
|--------|-------------|
| `ols` | Ordinary Least Squares (default) |
| `ridge` | L2 regularization |
| `lasso` | L1 regularization |
| `huber` | Robust regression |
| `quantile` | Quantile regression |

```python
model = regress(data=df, formula="sales ~ ads + price", method="ridge", alpha=1.0)
```

## Full Parameters

```python
model = regress(
    y=None,              # ndarray (direct mode)
    X=None,              # ndarray (direct mode)
    data=None,           # DataFrame (formula mode)
    formula=None,        # "y ~ x1 + x2"
    method='ols',        # 'ols', 'ridge', 'lasso', 'huber', 'quantile'
    summary=True,        # auto-print summary
    alpha=None,          # regularization strength
    diagnostics=False    # auto-run diagnostics
)
```

## Results

camelCase is the primary naming. snake_case aliases are available for backward compatibility.

```python
print(model.rSquared)         # R² (primary)
print(model.adjRSquared)      # Adjusted R² (primary)
print(model.fStat)            # F-statistic (primary)
print(model.durbinWatson)     # Durbin-Watson (primary)
print(model.coefficients)     # Coefficient array
print(model.pvalues)          # P-values array

# snake_case aliases also work
print(model.r_squared)
print(model.adj_r_squared)
print(model.f_stat)
print(model.durbin_watson)
```

## Diagnostics

```python
print(model.diagnose())
```

Returns a text report with

- **VIF**: Multicollinearity check (>10 is problematic)
- **Breusch-Pagan**: Heteroscedasticity test
- **Jarque-Bera**: Residual normality test
- **Durbin-Watson**: Autocorrelation test

## Prediction

```python
import pandas as pd

new_data = pd.DataFrame({
    "ads": [50, 75, 90],
    "price": [20, 15, 10],
    "promo": [0, 1, 1],
})
predictions = model.predict(new_data)
```

## Formula Syntax

```python
regress(data=df, formula="y ~ x1 + x2")       # Specific variables
regress(data=df, formula="y ~ .")              # All variables
regress(data=df, formula="y ~ x1 * x2")       # Interaction terms
regress(data=df, formula="y ~ x + I(x**2)")   # Polynomial
```

## Direct Array Input

```python
model = regress(y=y_array, X=X_array)
```
