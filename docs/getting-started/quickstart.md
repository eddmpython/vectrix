---
title: Quickstart
---

# Quickstart

Get your first forecast in 3 lines of Python. No configuration, no model selection, no parameter tuning — Vectrix handles everything automatically.

## Forecast from a List

The simplest possible usage. Pass any numeric sequence and the number of steps to predict

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170], steps=5)
print(result.model)          # Selected model name
print(result.predictions)    # Forecast values
print(result.summary())      # Full text summary
```

Behind the scenes, Vectrix evaluates 30+ model candidates, validates each on a holdout set, and returns the winner with 95% confidence intervals.

## Forecast from a DataFrame

For real-world data with timestamps, pass a pandas DataFrame. Vectrix auto-detects date and value columns

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

## Forecast from a CSV File

Skip the pandas step entirely — pass a file path and Vectrix reads it for you

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
```

## Working with Results

Every forecast returns an `EasyForecastResult` object with predictions, confidence intervals, metrics, and export methods

| Attribute / Method | Description |
|-------------------|-------------|
| `.predictions` | Forecast values (numpy array) |
| `.dates` | Forecast dates |
| `.lower` | 95% lower bound |
| `.upper` | 95% upper bound |
| `.model` | Selected model name |
| `.mape` | Validation MAPE (%) |
| `.rmse` | Validation RMSE |
| `.summary()` | Formatted text report |
| `.compare()` | All models ranked by accuracy |
| `.all_forecasts()` | Every model's predictions side-by-side |
| `.to_dataframe()` | Convert to DataFrame |
| `.to_csv(path)` | Export to CSV |
| `.to_json()` | Export to JSON string |
| `.plot()` | Matplotlib visualization |

## Supported Input Formats

Vectrix accepts five input formats — no manual conversion needed

```python
forecast([1, 2, 3, 4, 5])                    # list
forecast(np.array([1, 2, 3, 4, 5]))          # numpy array
forecast(pd.Series([1, 2, 3, 4, 5]))         # pandas Series
forecast(df, date="date", value="sales")      # DataFrame
forecast("data.csv")                           # CSV file path
```

## Quick Analysis

Profile your data before forecasting — understand its difficulty, seasonality, and recommended models

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
print(f"Difficulty: {report.dna.difficulty}")
print(f"Category: {report.dna.category}")
print(report.summary())
```

## Quick Regression

R-style regression with automatic diagnostics

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ ads + price")
print(model.summary())
print(model.diagnose())
```

## What's Next?

- **[Tutorial 01 — Quickstart](/docs/tutorials/quickstart)** — Detailed walkthrough with expected outputs
- **[Tutorial 02 — Analysis & DNA](/docs/tutorials/analyze)** — Understand your data's DNA fingerprint
- **[Tutorial 04 — 30+ Models](/docs/tutorials/models)** — Deep dive into every model Vectrix offers

---

**Interactive:** Run `marimo run docs/tutorials/en/01_quickstart.py` for the interactive version.
