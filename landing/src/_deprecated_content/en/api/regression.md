---
title: Regression API
---

# Regression API

R-style formula regression with full diagnostics.

## LinearRegressor

`LinearRegressor()`

OLS linear regression.

### Methods

- `fit(y, X)` → self
- `predict(X, interval='none', alpha=0.05)` → DataFrame
- `summary()` → str
- `diagnose()` → str

## RidgeRegressor

`RidgeRegressor(alpha=1.0)`

L2 regularization.

## LassoRegressor

`LassoRegressor(alpha=1.0)`

L1 regularization (sparse solutions).

## HuberRegressor

`HuberRegressor(delta=1.345)`

Robust regression — resistant to outliers.

## QuantileRegressor

`QuantileRegressor(quantile=0.5)`

Median regression (quantile=0.5) or arbitrary quantiles.

## RegressionDiagnostics

`RegressionDiagnostics()`

### Methods

- `vif(X)` → Variance Inflation Factors
- `shapiroWilk(residuals)` → Normality test
- `breuschPagan(residuals, X)` → Homoscedasticity test
- `durbinWatson(residuals)` → Autocorrelation test
- `cooksDistance(y, X)` → Influence analysis
- `fullReport(y, X, residuals)` → Complete diagnostics string
