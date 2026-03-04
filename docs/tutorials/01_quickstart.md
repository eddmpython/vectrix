---
title: "Tutorial 01 — Quickstart"
---

# Tutorial 01 — Quickstart

**Build your first forecast in under a minute.** No configuration, no boilerplate, no model selection — just pass your data and get predictions with confidence intervals.

Vectrix is a zero-config time series forecasting library. It evaluates 30+ statistical models (ETS, ARIMA, Theta, CES, DOT, and more), validates each on a holdout set, and returns the best one — all in a single function call.

## Installation

```bash
pip install vectrix
```

## Forecast from a List

The simplest possible forecast. Pass a Python list and the number of future steps to predict

```python
from vectrix import forecast

data = [120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215]
result = forecast(data, steps=5)

print(result.model)        # Best model selected automatically
print(result.predictions)  # Forecast values as numpy array
print(result.mape)         # Validation MAPE (%)
```

**Expected output:**

```
AutoETS
[221.3  228.7  235.1  241.6  248.0]
4.23
```

What happened behind the scenes
1. Vectrix inferred dates from the data length (daily frequency by default)
2. Split data into training (80%) and validation (20%) sets
3. Fitted 30+ models on the training portion
4. Ranked models by validation MAPE and selected the winner
5. Re-fitted the winner on the full dataset and generated predictions with 95% confidence intervals

## Forecast from a DataFrame

Real-world data usually comes as a DataFrame with a date column and a value column. Vectrix auto-detects both, or you can specify them explicitly

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=30)

print(result.summary())
```

**Expected output:**

```
=== Vectrix Forecast Summary ===
Model: AutoETS
Forecast Steps: 30
MAPE: 3.85%
RMSE: 42.17
MAE: 35.62

Predictions:
  2026-03-03: 1,245.3 [1,102.1 - 1,388.5]
  2026-03-04: 1,251.8 [1,095.4 - 1,408.2]
  ...
```

## Forecast from a CSV File

Skip the pandas step entirely. Pass a file path and Vectrix reads it, detects columns, and forecasts — all in one call

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result.model)
print(result.predictions)
```

## Full Text Summary

The `.summary()` method returns a human-readable report with the selected model, accuracy metrics, and predictions with confidence intervals

```python
result = forecast([100, 120, 130, 115, 140, 160, 150, 170], steps=5)
print(result.summary())
```

## Compare All Models

Every forecast evaluates 30+ models internally. You can see the full ranking to understand how close the competition was

```python
result = forecast(data, steps=5)

comparison = result.compare()
print(comparison)
```

**Expected output:**

```
          model   mape   rmse    mae  smape
0       AutoETS   4.23  12.41   9.87   4.15
1    AutoARIMA   4.87  14.23  11.42   4.72
2        Theta   5.12  15.67  12.31   5.01
3          DOT   5.34  16.12  12.89   5.22
4         MSTL   5.89  17.45  13.67   5.74
...
```

The standalone `compare()` function provides an even quicker way to rank models without storing the full forecast result

```python
from vectrix import compare

ranking = compare(data, steps=5)
print(ranking)
```

## Get All Model Forecasts

Sometimes you want predictions from every model, not just the winner. This is useful for ensemble building or understanding model disagreement

```python
all_preds = result.all_forecasts()
print(all_preds)
```

**Expected output:**

```
   step  AutoETS  AutoARIMA  Theta    DOT   MSTL  ...
0     1   221.3     219.8   222.1  220.5  218.9  ...
1     2   228.7     226.4   229.3  227.8  225.1  ...
2     3   235.1     233.2   236.0  234.6  231.8  ...
...
```

## Export Results

Convert results to DataFrame, CSV, or JSON for downstream use in dashboards, reports, or other systems

```python
df_result = result.to_dataframe()
print(df_result)
```

**Expected output:**

```
         date  prediction   lower95   upper95
0  2026-03-03      221.3     198.2     244.4
1  2026-03-04      228.7     202.1     255.3
2  2026-03-05      235.1     205.8     264.4
...
```

```python
result.to_csv("forecast_output.csv")

json_str = result.to_json()
print(json_str[:100])
```

## Result Object Reference

`EasyForecastResult` provides these attributes and methods

| Attribute / Method | Type | Description |
|---|---|---|
| `.predictions` | `np.ndarray` | Forecast values |
| `.dates` | `list` | Forecast date strings |
| `.lower` | `np.ndarray` | 95% lower confidence bound |
| `.upper` | `np.ndarray` | 95% upper confidence bound |
| `.model` | `str` | Selected model name |
| `.mape` | `float` | Validation MAPE (%) |
| `.rmse` | `float` | Validation RMSE |
| `.mae` | `float` | Validation MAE |
| `.smape` | `float` | Validation sMAPE |
| `.models` | `list` | All evaluated model names |
| `.compare()` | `DataFrame` | All models ranked by MAPE |
| `.all_forecasts()` | `DataFrame` | Every model's predictions |
| `.summary()` | `str` | Formatted text summary |
| `.describe()` | `DataFrame` | Pandas-style statistics |
| `.to_dataframe()` | `DataFrame` | date, prediction, lower95, upper95 |
| `.to_csv(path)` | `self` | Export to CSV file |
| `.to_json(path)` | `str` | Export to JSON string |

## Supported Input Formats

Vectrix accepts five input formats out of the box. No manual conversion or preprocessing needed

```python
import numpy as np
import pandas as pd
from vectrix import forecast

forecast([1, 2, 3, 4, 5], steps=3)                    # Python list
forecast(np.array([1, 2, 3, 4, 5]), steps=3)           # NumPy array
forecast(pd.Series([1, 2, 3, 4, 5]), steps=3)          # Pandas Series
forecast(df, date="date", value="sales", steps=3)      # DataFrame
forecast("data.csv", steps=3)                           # CSV file path
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data` | (required) | Input data in any supported format |
| `steps` | `10` | Number of forecast steps |
| `date` | auto | Date column name (DataFrame/CSV only) |
| `value` | auto | Value column name (DataFrame/CSV only) |
| `frequency` | `'auto'` | Frequency hint (auto-detected if omitted) |

## Complete Example

A full end-to-end workflow: forecast, inspect metrics, export results, and compare all models

```python
from vectrix import forecast

monthly_sales = [
    450, 470, 520, 540, 580, 620, 590, 610, 650, 680, 710, 750,
    460, 490, 530, 560, 600, 640, 610, 630, 670, 700, 730, 770,
]

result = forecast(monthly_sales, steps=6)

print(f"Model: {result.model}")
print(f"MAPE: {result.mape:.2f}%")
print(f"Next 6 months: {result.predictions}")
print(f"Lower bound: {result.lower}")
print(f"Upper bound: {result.upper}")

result.to_csv("forecast.csv")
print()
print("All models evaluated:")
print(result.compare().head(10))
```

---
