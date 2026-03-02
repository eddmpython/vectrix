---
title: Easy API
---

# Easy API

The simplest way to use Vectrix. One function call for each task.

## Functions

### `forecast(data, steps=10, **kwargs)`

One-call forecasting with automatic model selection.

**Parameters:**
- `data` — Input data (list, ndarray, Series, DataFrame, dict, or CSV path)
- `steps` — Number of forecast steps (default: 10)
- `date` — Date column name (optional, auto-detected)
- `value` — Value column name (optional, auto-detected)
- `period` — Seasonal period (optional, auto-detected)

**Returns:** `EasyForecastResult`

### `analyze(data, **kwargs)`

Time series DNA profiling, changepoint detection, anomaly identification.

**Parameters:**
- `data` — Input data (same formats as forecast)
- `date` — Date column name (optional)
- `value` — Value column name (optional)

**Returns:** `EasyAnalysisResult`

### `regress(y=None, X=None, data=None, formula=None, method="ols", **kwargs)`

R-style formula regression with full diagnostics.

**Parameters:**
- `y` — Dependent variable (ndarray)
- `X` — Independent variables (ndarray)
- `data` — DataFrame (use with formula)
- `formula` — R-style formula string (e.g. `"y ~ x1 + x2"`)
- `method` — `"ols"`, `"ridge"`, `"lasso"`, `"huber"`, `"quantile"`
- `summary` — Print summary automatically (default: True)

**Returns:** `EasyRegressionResult`

### `quick_report(data, steps=10, **kwargs)`

Combined analysis + forecast report generation.

**Returns:** `str` — Formatted report

## Result Classes

### EasyForecastResult

| Attribute | Type | Description |
|---|---|---|
| `.predictions` | `np.ndarray` | Forecast values |
| `.dates` | `list` | Forecast date strings |
| `.lower` | `np.ndarray` | 95% lower bound |
| `.upper` | `np.ndarray` | 95% upper bound |
| `.model` | `str` | Selected model name |
| `.mape` | `float` | Validation MAPE (%) |
| `.rmse` | `float` | Validation RMSE |
| `.mae` | `float` | Validation MAE |
| `.smape` | `float` | Validation sMAPE |
| `.models` | `list` | All evaluated model names |
| `.compare()` | `DataFrame` | All models ranked by MAPE |
| `.all_forecasts()` | `DataFrame` | Every model's predictions |
| `.summary()` | `str` | Formatted text summary |
| `.to_dataframe()` | `DataFrame` | date, prediction, lower95, upper95 |
| `.to_csv(path)` | `self` | Save to CSV |
| `.to_json(path)` | `str` | Save to JSON |
| `.describe()` | `DataFrame` | Pandas-style statistics |

### EasyAnalysisResult

| Attribute | Type | Description |
|---|---|---|
| `.dna` | `DNAProfile` | DNA profile object |
| `.changepoints` | `np.ndarray` | Changepoint indices |
| `.anomalies` | `np.ndarray` | Anomaly indices |
| `.features` | `dict` | Extracted features |
| `.characteristics` | `DataCharacteristics` | Data properties |
| `.summary()` | `str` | Formatted report |

### EasyRegressionResult

| Attribute | Type | Description |
|---|---|---|
| `.coefficients` | `np.ndarray` | Regression coefficients |
| `.pvalues` | `np.ndarray` | P-values |
| `.r_squared` | `float` | R² |
| `.adj_r_squared` | `float` | Adjusted R² |
| `.f_stat` | `float` | F-statistic |
| `.summary()` | `str` | Regression table |
| `.diagnose()` | `str` | Full diagnostics |
| `.predict(X)` | `DataFrame` | Predictions with intervals |
