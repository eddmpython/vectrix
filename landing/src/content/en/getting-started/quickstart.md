---
title: Quickstart
---

# Quickstart

## Forecast from a List

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170], steps=5)
print(result.model)          # Selected model name
print(result.predictions)    # Forecast values
print(result.summary())      # Full text summary
```

## Forecast from a DataFrame

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

## Forecast from a CSV File

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
```

## Working with Results

`EasyForecastResult` provides:

| Attribute / Method | Description |
|-------------------|-------------|
| `.predictions` | Forecast values (numpy array) |
| `.dates` | Forecast dates |
| `.lower` | 95% lower bound |
| `.upper` | 95% upper bound |
| `.model` | Selected model name |
| `.summary()` | Text summary |
| `.to_dataframe()` | Convert to DataFrame |
| `.to_csv(path)` | Export to CSV |
| `.to_json()` | Export to JSON string |
| `.plot()` | Matplotlib visualization |

## Supported Input Formats

```python
forecast([1, 2, 3, 4, 5])                    # list
forecast(np.array([1, 2, 3, 4, 5]))          # numpy array
forecast(pd.Series([1, 2, 3, 4, 5]))         # pandas Series
forecast({"value": [1, 2, 3, 4, 5]})         # dict
forecast(df, date="date", value="sales")      # DataFrame
forecast("data.csv")                           # CSV file path
```

## Quick Analysis

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
print(f"Difficulty: {report.dna.difficulty}")
print(f"Category: {report.dna.category}")
print(report.summary())
```

## Quick Regression

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ ads + price")
print(model.summary())
print(model.diagnose())
```

---

**Interactive:** Run `marimo run docs/tutorials/en/01_quickstart.py` for the interactive version.
