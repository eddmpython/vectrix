# Tutorial 03 — Regression

**statsmodels-level regression in one line.**

Vectrix's `regress()` function supports R-style formulas, multiple methods (OLS, Ridge, Lasso, Huber, Quantile), full diagnostics, and prediction intervals — all without importing statsmodels.

## 1. Direct Input Mode

Pass y (dependent) and X (independent) arrays directly:

```python
import numpy as np
from vectrix import regress

np.random.seed(42)
n = 100
x1 = np.random.uniform(10, 50, n)
x2 = np.random.uniform(1, 10, n)
noise = np.random.normal(0, 5, n)
y = 20 + 3.5 * x1 - 2.0 * x2 + noise

X = np.column_stack([x1, x2])

result = regress(y, X)
```

This automatically prints a summary table showing coefficients, standard errors, t-values, and p-values.

### Key Results

```python
print(f"R²:          {result.r_squared:.4f}")
print(f"Adjusted R²: {result.adj_r_squared:.4f}")
print(f"F-statistic: {result.f_stat:.2f}")
print(f"Coefficients: {result.coefficients}")
print(f"P-values:     {result.pvalues}")
```

```
R²:          0.9812
Adjusted R²: 0.9808
F-statistic: 2531.45
Coefficients: [19.23  3.51 -2.13]
P-values:     [0.000 0.000 0.001]
```

## 2. Formula Mode

Use R-style formula strings with a DataFrame — more readable and powerful:

```python
import pandas as pd
from vectrix import regress

df = pd.DataFrame({"sales": y, "ads": x1, "price": x2})

result = regress(data=df, formula="sales ~ ads + price")
```

### Formula Syntax

| Syntax | Example | Meaning |
|--------|---------|---------|
| Basic | `"y ~ x1 + x2"` | Linear regression |
| All columns | `"y ~ ."` | Use all other numeric columns |
| Interaction | `"y ~ x1 * x2"` | x1 + x2 + x1:x2 |
| Cross term only | `"y ~ x1 : x2"` | Only the product x1·x2 |
| Polynomial | `"y ~ x + I(x**2)"` | Add squared term |
| Mixed | `"sales ~ ads + I(ads**2) + price"` | Linear + quadratic |

### Polynomial Example

```python
np.random.seed(42)
x = np.random.uniform(0, 10, 80)
y = 5 + 2 * x - 0.3 * x**2 + np.random.normal(0, 2, 80)

df = pd.DataFrame({"y": y, "x": x})
result = regress(data=df, formula="y ~ x + I(x**2)")
```

## 3. Diagnostics

Run VIF, normality, homoscedasticity, autocorrelation, and influence analysis — all at once:

```python
print(result.diagnose())
```

```
============================================
     Regression Diagnostics Report
============================================

  [Multicollinearity - VIF]
    ads:   1.02  (OK)
    price: 1.02  (OK)

  [Normality of Residuals]
    Shapiro-Wilk: W=0.993, p=0.891
    → Residuals appear normally distributed

  [Homoscedasticity]
    Breusch-Pagan: stat=2.14, p=0.343
    → No evidence of heteroscedasticity

  [Autocorrelation]
    Durbin-Watson: 2.03
    → No significant autocorrelation

  [Influential Points]
    High leverage: 2 points
    High Cook's D: 0 points
============================================
```

## 4. Prediction

Predict on new data with confidence or prediction intervals:

```python
X_new = np.array([[30, 5], [40, 3], [25, 8]])

pred_df = result.predict(X_new, interval="prediction", alpha=0.05)
print(pred_df)
```

```
   prediction      lower      upper
0      115.23      104.89     125.57
1      149.87      139.45     160.29
2       91.45       81.12     101.78
```

### Interval Types

| Type | Meaning |
|------|---------|
| `'prediction'` | Interval for a new individual observation (wider) |
| `'confidence'` | Interval for the mean response (narrower) |
| `'none'` | No interval, prediction column only |

## 5. Regression Methods

```python
result_ols    = regress(y, X, method="ols")      # Default
result_ridge  = regress(y, X, method="ridge")    # L2 regularization
result_lasso  = regress(y, X, method="lasso")    # L1 regularization
result_huber  = regress(y, X, method="huber")    # Robust to outliers
result_quant  = regress(y, X, method="quantile") # Median regression
```

## 6. Suppress Auto-Print

By default, `regress()` prints the summary automatically. To suppress it:

```python
result = regress(y, X, summary=False)
```

## 7. Result Object Reference

| Attribute / Method | Type | Description |
|---|---|---|
| `.coefficients` | `np.ndarray` | Regression coefficients (including intercept) |
| `.pvalues` | `np.ndarray` | P-values for each coefficient |
| `.r_squared` | `float` | R² (coefficient of determination) |
| `.adj_r_squared` | `float` | Adjusted R² |
| `.f_stat` | `float` | F-statistic |
| `.summary()` | `str` | Full regression table |
| `.diagnose()` | `str` | VIF + normality + homoscedasticity + autocorrelation + influence |
| `.predict(X, interval, alpha)` | `DataFrame` | prediction, lower, upper columns |

---

**Next:** [Tutorial 04 — 30+ Models](04_models.md) — Direct model access and comparison
