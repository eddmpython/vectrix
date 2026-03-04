---
title: Easy API
---

# Easy API

The simplest way to use Vectrix. One function call for each task.

## Functions

### `forecast()`

```python
forecast(
    data,                    # str | DataFrame | ndarray | list | Series | dict
    date=None,               # str — date column name
    value=None,              # str — value column name
    steps=30,                # int — forecast horizon
    verbose=False,           # bool
    models=None,             # list[str] | None — model IDs to evaluate
    ensemble=None,           # str | None — 'mean', 'weighted', 'median', 'best'
    confidence=0.95          # float — 0.80, 0.90, 0.95, 0.99
) -> EasyForecastResult
```

**Available model IDs:** `'dot'`, `'auto_ets'`, `'auto_arima'`, `'auto_ces'`, `'four_theta'`, `'auto_mstl'`, `'tbats'`, `'theta'`, `'dtsf'`, `'esn'`, `'garch'`, `'croston'`, `'ets_aan'`, `'ets_aaa'`, `'naive'`, `'mean'`, `'rwd'`, `'window_avg'`, `'egarch'`, `'gjr_garch'`, `'seasonal_naive'`, `'mstl'`

### `analyze()`

```python
analyze(
    data,                    # str | DataFrame | ndarray | list | Series | dict
    date=None,               # str
    value=None,              # str
    period=None,             # int | None — seasonal period (auto if None)
    features=True,           # bool
    changepoints=True,       # bool
    anomalies=True,          # bool
    anomalyThreshold=3.0     # float — z-score threshold
) -> EasyAnalysisResult
```

### `regress()`

```python
regress(
    y=None,                  # ndarray | Series | None (direct mode)
    X=None,                  # ndarray | DataFrame | None (direct mode)
    data=None,               # DataFrame | None (formula mode)
    formula=None,            # str | None — "y ~ x1 + x2"
    method='ols',            # str — 'ols', 'ridge', 'lasso', 'huber', 'quantile'
    summary=True,            # bool — auto-print summary
    alpha=None,              # float | None — regularization strength
    diagnostics=False        # bool — auto-run diagnostics
) -> EasyRegressionResult
```

### `compare()`

```python
compare(
    data,                    # str | DataFrame | ndarray | list | Series | dict
    date=None,               # str
    value=None,              # str
    steps=30,                # int
    verbose=False,           # bool
    models=None              # list[str] | None
) -> pd.DataFrame           # Returns DataFrame directly, NOT a Result object
```

**Returned DataFrame columns:** `model`, `mape`, `rmse`, `mae`, `smape`, `time_ms`, `selected`

### `quickReport()`

```python
quickReport(
    data, date=None, value=None, steps=30
) -> dict                   # Returns dict, NOT a Result object
```

**Returned dict keys:** `'forecast'` (EasyForecastResult), `'analysis'` (EasyAnalysisResult), `'summary'` (str)

**Alias:** `quick_report` = `quickReport` (backward compatibility)

### `loadSample()`

```python
loadSample(name: str) -> pd.DataFrame
```

Load a built-in sample dataset.

**Available samples:** `'airline'`, `'retail'`, `'stock'`, `'temperature'`, `'energy'`, `'web'`, `'intermittent'`

| Sample | date col | value col |
|--------|----------|-----------|
| airline | date | passengers |
| retail | date | sales |
| stock | date | close |
| temperature | date | temperature |
| energy | date | consumption_kwh |
| web | date | pageviews |
| intermittent | date | demand |

### `listSamples()`

```python
listSamples() -> pd.DataFrame
```

List available built-in sample datasets.

## Result Classes

### EasyForecastResult

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `.predictions` | `np.ndarray` | Forecast values |
| `.dates` | `list[str]` | Forecast dates |
| `.lower` | `np.ndarray` | Lower CI |
| `.upper` | `np.ndarray` | Upper CI |
| `.model` | `str` | Best model name |
| `.mape` | `float` | MAPE % |
| `.rmse` | `float` | RMSE |
| `.mae` | `float` | MAE |
| `.smape` | `float` | sMAPE |
| `.models` | `list[str]` | All valid model names (sorted by MAPE) |

**Methods:**

| Method | Alias | Returns | Description |
|--------|-------|---------|-------------|
| `summary()` | — | `str` | Text summary |
| `toDataframe()` | `to_dataframe()` | `DataFrame` | date, prediction, lower95, upper95 |
| `compare()` | — | `DataFrame` | All models ranked by MAPE |
| `allForecasts()` | `all_forecasts()` | `DataFrame` | date + one col per model |
| `describe()` | — | `DataFrame` | `.describe()` style stats |
| `toCsv(path)` | `to_csv(path)` | `self` | Save to CSV |
| `toJson(path=None)` | `to_json(path=None)` | `str` | JSON string or save to file |
| `save(path)` | — | `self` | Alias for `toJson(path)` |
| `plot()` | — | `Figure` | matplotlib plot (optional dep) |

### EasyAnalysisResult

| Attribute | Type | Description |
|---|---|---|
| `.dna` | `DNAProfile` | DNA profile object |
| `.changepoints` | `np.ndarray` | Changepoint **int indices** (NOT dicts) |
| `.anomalies` | `np.ndarray` | Anomaly **int indices** (NOT dicts) |
| `.features` | `dict` | Statistical features dict |
| `.characteristics` | `DataCharacteristics` | Data characteristics |
| `.summary()` | `str` | Formatted report |

!!! warning "anomalies/changepoints are int arrays"
    ```python
    # CORRECT
    for idx in analysis.anomalies:
        print(f"Anomaly at index {idx}")

    # WRONG — will crash
    for a in analysis.anomalies:
        print(a['index'], a['value'])  # TypeError!
    ```

### EasyRegressionResult

**Attributes (camelCase primary, snake_case aliases):**

| Primary | Alias | Type | Description |
|---------|-------|------|-------------|
| `coefficients` | — | `np.ndarray` | Including intercept |
| `pvalues` | — | `np.ndarray` | P-values |
| `rSquared` | `r_squared` | `float` | R² |
| `adjRSquared` | `adj_r_squared` | `float` | Adjusted R² |
| `fStat` | `f_stat` | `float` | F-statistic |
| `durbinWatson` | `durbin_watson` | `float` | Durbin-Watson statistic |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `summary()` | `str` | Regression summary table |
| `diagnose()` | `str` | Full diagnostics report |
| `predict(X, interval, alpha)` | `DataFrame` | Predictions with intervals |
