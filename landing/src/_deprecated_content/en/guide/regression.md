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
model = regress(data=df, formula="sales ~ ads + price", method="ridge")
```

## Results

```python
print(model.r_squared)        # R-squared
print(model.adj_r_squared)    # Adjusted R-squared
print(model.f_stat)           # F-statistic
print(model.coefficients)     # Coefficient array
print(model.pvalues)          # P-values array
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
predictions = model.predict(new_data)  # Returns DataFrame
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

---

**Interactive tutorial:** `marimo run docs/tutorials/en/03_regression.py`
