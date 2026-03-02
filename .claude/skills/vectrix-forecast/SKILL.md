---
name: vectrix-forecast
description: Run Vectrix time series forecasting. Use when the user asks to forecast, predict future values, or do time series prediction with Vectrix.
allowed-tools: Bash(uv *), Bash(python *), Read, Glob, Grep
---

# Vectrix Forecasting

When the user wants to forecast time series data using Vectrix:

## Quick Start

```python
from vectrix import forecast

result = forecast(df, date="date", value="sales", steps=12)
print(result.summary())
print(f"Model: {result.model}")
print(f"MAPE: {result.mape:.2f}%")
```

## Input Flexibility

`forecast()` accepts multiple data types:
- `pd.DataFrame` with date and value columns
- `str` — path to CSV file
- `np.ndarray`, `list`, `tuple` — raw numeric values
- `pd.Series` with DatetimeIndex
- `dict` with column data

## Key Parameters

- `data` — the input data (required)
- `date=None` — date column name (auto-detected)
- `value=None` — value column name (auto-detected)
- `steps=30` — forecast horizon
- `frequency='auto'` — 'D', 'W', 'M', 'Q', 'Y', 'H', or 'auto'

## Result Object (EasyForecastResult)

### Attributes
- `result.predictions` — np.ndarray of forecasted values
- `result.dates` — pd.DatetimeIndex
- `result.lower` / `result.upper` — 95% confidence interval bounds
- `result.model` — best model name (e.g. "AutoETS(A,Ad,A)")
- `result.mape`, `.rmse`, `.mae`, `.smape` — accuracy metrics
- `result.models` — list of all evaluated models

### Methods
- `result.summary()` — human-readable summary
- `result.to_dataframe()` — DataFrame with date, forecast, lower, upper
- `result.compare()` — all models comparison table (sMAPE, MAPE, RMSE, MAE)
- `result.all_forecasts()` — DataFrame of all model forecasts
- `result.plot()` — matplotlib visualization
- `result.to_csv(path)` — save to CSV
- `result.to_json()` — JSON output

## Model Comparison

```python
from vectrix import compare

comparison = compare(df, date="date", value="sales", steps=12)
print(comparison)  # DataFrame sorted by sMAPE
```

## Sample Data

```python
from vectrix import loadSample, listSamples
listSamples()  # see all 7 datasets
df = loadSample("airline")  # 144 monthly observations
```

## Common Gotchas

- Column auto-detection: "date", "ds", "timestamp" for dates; "value", "y", "sales" for values
- Use `date=` and `value=` explicitly if column names are non-standard
- All models use `fit(y) → predict(steps) → (pred, lower, upper)` interface
- DOT and AutoCES are the strongest general-purpose models
