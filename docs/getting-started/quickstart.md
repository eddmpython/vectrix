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
result.toCsv("forecast.csv")
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
| `.lower` | Lower CI bound |
| `.upper` | Upper CI bound |
| `.model` | Selected model name |
| `.mape` | Validation MAPE (%) |
| `.rmse` | Validation RMSE |
| `.smape` | Validation sMAPE |
| `.summary()` | Formatted text report |
| `.compare()` | All models ranked by accuracy |
| `.toDataframe()` | Convert to DataFrame |
| `.allForecasts()` | Every model's predictions side-by-side |
| `.toCsv(path)` | Export to CSV |
| `.toJson()` | Export to JSON string |
| `.plot()` | Matplotlib visualization |

snake_case aliases (`to_dataframe()`, `all_forecasts()`, `to_csv()`, `to_json()`) are also available.

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

- **[Forecasting Guide](../guide/forecasting.md)** — Full parameter reference and model categories
- **[Analysis & DNA](../guide/analysis.md)** — Understand your data's DNA fingerprint
- **[API Reference](../api/easy.md)** — Complete Easy API specification
