---
title: Korean Economy Forecasting
---

# Korean Economy Forecasting

Forecasting Korean economic indicators using publicly available FRED data. This showcase demonstrates how Vectrix handles real-world macroeconomic time series with a single function call.

## Data Sources

All data is fetched directly from the Federal Reserve Economic Data (FRED) API:

| Indicator | FRED Series | Frequency | Description |
|-----------|:-----------:|:---------:|-------------|
| USD/KRW Exchange Rate | `EXKOUS` | Monthly | Korean Won per US Dollar |
| KOSPI Index | `KOSPI` | Daily | Korea Composite Stock Price Index |
| Consumer Price Index | `KORCPIALLMINMEI` | Monthly | CPI for All Items in Korea |

## USD/KRW Exchange Rate Forecast

```python
import pandas as pd
from vectrix import forecast

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
df = pd.read_csv(url)
df.columns = ["date", "value"]
df["date"] = pd.to_datetime(df["date"])
df = df.dropna()

result = forecast(df, date="date", value="value", steps=12)
print(result.model)        # Auto-selected model
print(result.predictions)  # 12-month forecast
print(result.summary())    # Full summary with confidence intervals
```

### Inspecting the Result

```python
print(f"Selected model: {result.model}")
print(f"Mean prediction: {result.predictions.mean():,.1f} KRW/USD")
print(f"95% CI: {result.lower.min():,.1f} ~ {result.upper.max():,.1f} KRW")

result.to_dataframe()
```

## KOSPI Index Forecast

```python
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=KOSPI"
kospiDf = pd.read_csv(url)
kospiDf.columns = ["date", "value"]
kospiDf["date"] = pd.to_datetime(kospiDf["date"])
kospiDf = kospiDf.dropna()

result = forecast(kospiDf, date="date", value="value", steps=30)
print(result.summary())
```

## CPI Forecast

```python
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=KORCPIALLMINMEI"
cpiDf = pd.read_csv(url)
cpiDf.columns = ["date", "value"]
cpiDf["date"] = pd.to_datetime(cpiDf["date"])
cpiDf = cpiDf.dropna()

result = forecast(cpiDf, date="date", value="value", steps=12)
print(result.summary())
```

## Comparing Models

Use `compare()` to see how different models perform on the same data:

```python
result = forecast(df, date="date", value="value", steps=12)

comparison = result.compare()
print(comparison)
```

This returns a DataFrame with sMAPE, MAPE, RMSE, and MAE for every valid model, sorted by accuracy.

## Export Results

```python
result.to_dataframe().to_csv("krw_forecast.csv", index=False)

result.to_json("krw_forecast.json")
```

> **Disclaimer:** This showcase is for educational and demonstration purposes only. Forecasts of financial and economic indicators should not be used for investment decisions. Past performance does not guarantee future results.
