# Tutorial 01 — Quickstart

**Get your first forecast in under 3 minutes.**

Vectrix is a zero-config time series forecasting library.
Give it data — a list, NumPy array, pandas DataFrame, or CSV file — and it automatically selects the best model from 30+ candidates.

## Installation

```bash
pip install vectrix
```

## 1. Forecast from a List

The simplest way to start. No dates, no column names, no model selection needed.

```python
from vectrix import forecast

sales = [
    120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
    172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
    245, 268, 258, 280, 270, 295, 285, 310, 300, 325,
]

result = forecast(sales, steps=10)
```

That's it. Vectrix automatically:

- Generates dates (ending today)
- Tries multiple models (ETS, ARIMA, Theta, CES, DOT, …)
- Picks the best one by validation MAPE
- Returns predictions with 95% confidence intervals

### Inspect the Result

```python
print(result.model)         # e.g. 'Dynamic Optimized Theta'
print(result.predictions)   # array of 10 forecast values
print(result.mape)          # validation MAPE (%)
```

Expected output:

```
Dynamic Optimized Theta
[331.2  337.8  344.5  ...]
5.14
```

## 2. Forecast from a DataFrame

When you have a pandas DataFrame with a date column, Vectrix auto-detects both the date and value columns.

```python
import numpy as np
import pandas as pd
from vectrix import forecast

np.random.seed(42)
n = 120
t = np.arange(n, dtype=np.float64)
trend = 100 + 0.5 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 5, n)

df = pd.DataFrame({
    "date": pd.date_range("2015-01-01", periods=n, freq="MS"),
    "sales": trend + seasonal + noise,
})

result = forecast(df, steps=12)
```

If column names aren't obvious, specify them explicitly:

```python
result = forecast(df, date="date", value="sales", steps=12)
```

### View the Summary

```python
print(result.summary())
```

```
==================================================
        Vectrix Forecast Summary
==================================================
  Model: 4Theta Ensemble
  Horizon: 12 steps
  Start: 2025-01-01
  End: 2025-12-01
  Mean: 163.42
  Min: 139.68
  Max: 185.88

  [Model Comparison]
    4Theta Ensemble: MAPE=3.21%
    Dynamic Optimized Theta: MAPE=3.58%
    AutoCES (Native): MAPE=4.12%
    AutoETS (Native): MAPE=5.67%
    AutoARIMA (Native): MAPE=8.94%
==================================================
```

## 3. Export Results

### As a DataFrame

```python
pred_df = result.to_dataframe()
print(pred_df.head())
```

```
         date  prediction     lower95     upper95
0  2025-01-01      159.69      140.12      179.26
1  2025-02-01      169.33      145.87      192.79
2  2025-03-01      176.37      149.23      203.52
3  2025-04-01      185.88      155.31      216.45
4  2025-05-01      181.74      147.82      215.66
```

### As CSV / JSON

```python
result.to_csv("forecast.csv")

json_str = result.to_json()
result.to_json("forecast.json")
```

## 4. Compare All Models

See how every model performed:

```python
print(result.compare())
```

```
                     model   mape   rmse    mae  smape  time_ms  selected
0          4Theta Ensemble   3.21  12.45   9.87   3.15      2.1      True
1  Dynamic Optimized Theta   3.58  14.23  11.02   3.49      5.5     False
2         AutoCES (Native)   4.12  16.78  13.45   4.03      9.3     False
3         AutoETS (Native)   5.67  21.34  17.89   5.52     28.6     False
4       AutoARIMA (Native)   8.94  32.56  26.78   8.67     15.2     False
```

### Get Every Model's Predictions

```python
all_df = result.all_forecasts()
print(all_df.head())
```

```
         date  4Theta Ensemble  Dynamic Optimized Theta  AutoCES  AutoETS  AutoARIMA
0  2025-01-01           159.69                   157.74   154.04   156.83     153.26
1  2025-02-01           169.33                   166.34   162.24   163.97     153.26
2  2025-03-01           176.37                   172.87   168.37   168.81     153.26
...
```

## 5. Interactive Slider (Horizon)

Adjust the forecast horizon and see how results change:

```python
for steps in [7, 14, 30]:
    r = forecast(df, steps=steps)
    print(f"steps={steps:>2}  model={r.model:<30}  mean={r.predictions.mean():.1f}")
```

```
steps= 7  model=Dynamic Optimized Theta    mean=168.3
steps=14  model=4Theta Ensemble             mean=165.7
steps=30  model=4Theta Ensemble             mean=158.2
```

## 6. Result Object Reference

| Attribute / Method | Type | Description |
|---|---|---|
| `.predictions` | `np.ndarray` | Forecast values |
| `.dates` | `list` | Forecast date strings |
| `.lower` | `np.ndarray` | 95% lower bound |
| `.upper` | `np.ndarray` | 95% upper bound |
| `.model` | `str` | Selected model name |
| `.mape` | `float` | Validation MAPE (%) |
| `.rmse` | `float` | Validation RMSE |
| `.models` | `list` | All evaluated model names (ranked) |
| `.to_dataframe()` | `DataFrame` | date, prediction, lower95, upper95 |
| `.compare()` | `DataFrame` | All models ranked by MAPE |
| `.all_forecasts()` | `DataFrame` | Every model's predictions |
| `.summary()` | `str` | Formatted text summary |
| `.to_csv(path)` | `self` | Save to CSV |
| `.to_json(path)` | `str` | Save to JSON |
| `.describe()` | `DataFrame` | Pandas-style statistics |

## 7. Supported Input Formats

```python
forecast([1, 2, 3, 4, 5])                    # list
forecast(np.array([1, 2, 3, 4, 5]))          # NumPy array
forecast(pd.Series([1, 2, 3, 4, 5]))         # pandas Series
forecast({"value": [1, 2, 3, 4, 5]})         # dict
forecast(df, date="date", value="sales")      # DataFrame
forecast("data.csv")                          # CSV file path
```

---

**Next:** [Tutorial 02 — Analysis & DNA](02_analyze.md) — Automatic time series profiling
