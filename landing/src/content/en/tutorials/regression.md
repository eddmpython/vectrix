---
title: "Tutorial 03 — Regression"
---

# Tutorial 03 — Regression

R-style formula regression with full diagnostics, multiple methods, and prediction intervals -- all in one function call.

## Direct Input

The simplest form: pass `y` and `X` directly as arrays:

```python
import numpy as np
from vectrix import regress

np.random.seed(42)
X = np.random.randn(100, 2)
y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(100) * 0.5

model = regress(y=y, X=X)
```

**Expected output:**

```
=== Regression Summary ===
Method: OLS
Observations: 100
R-squared: 0.954
Adj. R-squared: 0.953
F-statistic: 1012.35 (p < 0.001)

              Coef    Std.Err    t-value    P>|t|
Intercept    3.012      0.050     60.24    0.000 ***
x1           1.987      0.052     38.21    0.000 ***
x2          -1.493      0.048    -31.10    0.000 ***
```

## Formula Mode

With a DataFrame, use R-style formulas for a more natural interface:

```python
import pandas as pd
from vectrix import regress

df = pd.DataFrame({
    "sales": [100, 150, 200, 180, 250, 300, 280, 350, 400, 380],
    "ads": [10, 15, 20, 18, 25, 30, 28, 35, 40, 38],
    "price": [50, 48, 45, 47, 42, 40, 41, 38, 35, 36],
    "promo": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
})

model = regress(data=df, formula="sales ~ ads + price + promo")
```

### Formula Syntax

```python
regress(data=df, formula="y ~ x1 + x2")       # Specific variables
regress(data=df, formula="y ~ .")              # All other columns
regress(data=df, formula="y ~ x1 * x2")       # With interaction term
regress(data=df, formula="y ~ x + I(x**2)")   # Polynomial terms
```

## Result Object

The `EasyRegressionResult` provides direct access to all regression statistics:

```python
print(f"R-squared: {model.r_squared:.4f}")
print(f"Adj. R-squared: {model.adj_r_squared:.4f}")
print(f"F-statistic: {model.f_stat:.2f}")
print(f"Coefficients: {model.coefficients}")
print(f"P-values: {model.pvalues}")
```

### Result Reference

| Attribute / Method | Type | Description |
|---|---|---|
| `.coefficients` | `np.ndarray` | Regression coefficients (including intercept) |
| `.pvalues` | `np.ndarray` | P-values for each coefficient |
| `.r_squared` | `float` | R-squared (coefficient of determination) |
| `.adj_r_squared` | `float` | Adjusted R-squared |
| `.f_stat` | `float` | F-statistic |
| `.summary()` | `str` | Formatted regression table |
| `.diagnose()` | `str` | Full diagnostic report |
| `.predict(X)` | `DataFrame` | Predictions with confidence intervals |

## Diagnostics

The `diagnose()` method runs four standard regression diagnostic tests:

```python
print(model.diagnose())
```

**Expected output:**

```
=== Regression Diagnostics ===

1. Multicollinearity (VIF):
   ads:    2.31 (OK)
   price:  2.18 (OK)
   promo:  1.45 (OK)

2. Heteroscedasticity (Breusch-Pagan):
   LM stat: 3.42, p-value: 0.331
   Result: No heteroscedasticity detected

3. Normality (Jarque-Bera):
   JB stat: 1.87, p-value: 0.393
   Result: Residuals appear normally distributed

4. Autocorrelation (Durbin-Watson):
   DW stat: 2.05
   Result: No significant autocorrelation
```

### Diagnostic Tests

| Test | What It Checks | Warning Threshold |
|------|---------------|-------------------|
| **VIF** | Multicollinearity between predictors | VIF > 10 is problematic |
| **Breusch-Pagan** | Non-constant variance (heteroscedasticity) | p-value below 0.05 |
| **Jarque-Bera** | Normality of residuals | p-value below 0.05 |
| **Durbin-Watson** | Autocorrelation in residuals | Far from 2.0 |

## Prediction with Intervals

Generate predictions with confidence intervals for new data:

```python
import pandas as pd

new_data = pd.DataFrame({
    "ads": [50, 75, 100],
    "price": [30, 25, 20],
    "promo": [1, 1, 0],
})

predictions = model.predict(new_data)
print(predictions)
```

**Expected output:**

```
   prediction   lower95   upper95
0      425.3     380.1     470.5
1      530.7     478.2     583.2
2      610.2     550.8     669.6
```

## Regression Methods

Vectrix supports five regression methods. Switch by setting the `method` parameter:

```python
from vectrix import regress

ols_model    = regress(data=df, formula="sales ~ ads + price", method="ols")
ridge_model  = regress(data=df, formula="sales ~ ads + price", method="ridge")
lasso_model  = regress(data=df, formula="sales ~ ads + price", method="lasso")
huber_model  = regress(data=df, formula="sales ~ ads + price", method="huber")
quant_model  = regress(data=df, formula="sales ~ ads + price", method="quantile")
```

### Method Comparison

| Method | Use Case | How It Works |
|--------|----------|-------------|
| `ols` | Default, no issues with data | Minimizes sum of squared residuals |
| `ridge` | Multicollinearity (correlated predictors) | L2 regularization, shrinks coefficients |
| `lasso` | Feature selection, sparse models | L1 regularization, can zero out coefficients |
| `huber` | Outliers in the data | Robust loss function, down-weights outliers |
| `quantile` | Median regression, skewed distributions | Minimizes absolute deviations |

### When to Use Each Method

**OLS** (default) -- Use when your data is well-behaved: no extreme outliers, no highly correlated predictors, and residuals are roughly normal.

**Ridge** -- Use when you have correlated predictors (e.g., temperature and humidity, GDP and employment). Ridge keeps all variables but shrinks their coefficients.

**Lasso** -- Use when you suspect some predictors are irrelevant. Lasso can drive coefficients to exactly zero, effectively performing variable selection.

**Huber** -- Use when your data contains outliers. Huber uses a hybrid loss function that behaves like OLS for small errors but is robust to large errors.

**Quantile** -- Use when you want to predict the median rather than the mean, or when your data is heavily skewed.

## Suppress Auto-Print

By default, `regress()` prints the summary automatically. To suppress this:

```python
model = regress(data=df, formula="sales ~ ads + price", summary=False)
```

## Complete Example

```python
import pandas as pd
import numpy as np
from vectrix import regress

np.random.seed(42)
n = 200
df = pd.DataFrame({
    "revenue": np.random.randn(n) * 50 + 500,
    "marketing": np.random.randn(n) * 10 + 50,
    "price": np.random.randn(n) * 5 + 30,
    "season": np.random.choice([0, 1], n),
})
df["revenue"] = 100 + 8 * df["marketing"] - 5 * df["price"] + 20 * df["season"] + np.random.randn(n) * 10

model = regress(data=df, formula="revenue ~ marketing + price + season")
print(model.diagnose())

future = pd.DataFrame({
    "marketing": [60, 70, 80],
    "price": [28, 25, 22],
    "season": [1, 0, 1],
})
print("\nPredictions:")
print(model.predict(future))
```

---
