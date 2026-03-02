---
name: vectrix-regress
description: Run Vectrix regression analysis with R-style formulas. Use when the user asks to do regression, OLS, predict Y from X, or analyze variable relationships.
allowed-tools: Bash(uv *), Bash(python *), Read, Glob, Grep
---

# Vectrix Regression

When the user wants regression analysis:

## R-Style Formula

```python
from vectrix import regress

reg = regress(data=df, formula="revenue ~ ads + price + season")
print(reg.summary())
print(f"R²: {reg.r_squared:.4f}")
print(reg.diagnose())  # Durbin-Watson, Breusch-Pagan, VIF, Jarque-Bera
```

## Array Input

```python
from vectrix import regress
reg = regress(y=y_array, X=X_array, method='ols')
```

## Methods

- `method='ols'` — Ordinary Least Squares (default)
- `method='ridge'` — Ridge regression
- `method='lasso'` — Lasso regression
- `method='huber'` — Robust to outliers
- `method='quantile'` — Quantile regression (median by default)

## Result Object (EasyRegressionResult)

### Attributes
- `reg.coefficients` — pd.Series with coefficient names and values
- `reg.pvalues` — pd.Series of p-values
- `reg.r_squared` — float
- `reg.adj_r_squared` — float
- `reg.f_stat` — float

### Methods
- `reg.summary()` — full regression table
- `reg.diagnose()` — diagnostic tests (DW, BP, VIF, JB)
- `reg.predict(X, interval='prediction', alpha=0.05)` — predictions with intervals

## Advanced Regression

### Variable Selection
```python
from vectrix.regression import StepwiseSelector, BestSubsetSelector
selector = StepwiseSelector(direction='both')
result = selector.select(X, y)
```

### Time Series Regression
```python
from vectrix.regression import NeweyWestOLS, CochraneOrcutt, GrangerCausality
nw = NeweyWestOLS()
result = nw.fit(X, y)  # HAC robust standard errors

gc = GrangerCausality()
result = gc.test(y1, y2, maxLag=4)  # Granger causality test
```

## Common Gotchas

- Result uses snake_case: `r_squared`, `adj_r_squared`, `f_stat`
- `regress()` with formula requires `data=` DataFrame parameter
- `diagnose()` returns a string report, not a dict
