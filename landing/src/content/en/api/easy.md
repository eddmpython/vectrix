---
title: Easy API
---

# Easy API

The simplest way to use Vectrix. One function call for each task.

## Functions

### `forecast(data, date=None, value=None, steps=30, frequency='auto', verbose=False)`

One-call forecasting with automatic model selection.

**Parameters:**
- `data` — Input data (list, ndarray, Series, DataFrame, dict, or CSV path)
- `date` — Date column name (optional, auto-detected)
- `value` — Value column name (optional, auto-detected)
- `steps` — Number of forecast steps (default: 30)
- `frequency` — Frequency hint (default: `'auto'`, auto-detected)
- `verbose` — Print progress (default: False)

**Returns:** `EasyForecastResult`

### `analyze(data, date=None, value=None, period=None)`

Time series DNA profiling, changepoint detection, anomaly identification.

**Parameters:**
- `data` — Input data (same formats as forecast)
- `date` — Date column name (optional)
- `value` — Value column name (optional)
- `period` — Seasonal period (optional, auto-detected)

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

### `compare(data, date=None, value=None, steps=30, verbose=False)`

Compare all models on the given data and return a ranked DataFrame.

**Returns:** `DataFrame` — Models ranked by accuracy (sMAPE, MAPE, RMSE, MAE)

### `quick_report(data, date=None, value=None, steps=30)`

Combined analysis + forecast report generation.

**Returns:** `Dict[str, Any]` — Report dictionary with `'summary'`, `'forecast'`, `'analysis'` keys

### `listSamples()`

List available built-in sample datasets.

**Returns:** `DataFrame` — Dataset names and descriptions

### `loadSample(name)`

Load a built-in sample dataset.

**Parameters:**
- `name` — Dataset name (e.g. `"airline"`, `"retail"`, `"stock"`)

**Returns:** `DataFrame`

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
