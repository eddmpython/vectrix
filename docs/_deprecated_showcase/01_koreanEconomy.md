# Showcase 01 — Korean Economy Forecasting

**Forecast real Korean economic indicators using publicly available data.**

## Overview

This showcase demonstrates Vectrix forecasting on real Korean economic data from FRED (Federal Reserve Economic Data). No API key required — all data is fetched automatically.

### What You'll See

- **USD/KRW Exchange Rate** — Monthly data from 1981, 12-month forecast
- **KOSPI Stock Index** — Monthly stock market index, 12-month forecast
- **Consumer Price Index (CPI)** — Inflation tracking, 12-month forecast
- **Multi-indicator DNA Analysis** — Compare difficulty and characteristics across indicators

## Run Interactively

```bash
pip install vectrix pandas marimo
marimo run docs/showcase/en/01_koreanEconomy.py
```

## Code

```python
import pandas as pd
from vectrix import forecast

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
df = pd.read_csv(url)
df.columns = ["date", "value"]
df["date"] = pd.to_datetime(df["date"])
df = df.dropna()

result = forecast(df, date="date", value="value", steps=12)
print(f"Model: {result.model}")
print(f"Mean prediction: {result.predictions.mean():,.1f} KRW/USD")
print(f"95% CI: {result.lower.min():,.1f} ~ {result.upper.max():,.1f}")
```

### Expected Output

```
Model: AutoETS
Mean prediction: 1,380.5 KRW/USD
95% CI: 1,250.3 ~ 1,510.8
```

The selected model and predictions will vary based on the latest data available from FRED.

## Forecast Summary

```python
print(result.summary())
```

The summary includes the selected model, accuracy metrics, and prediction intervals for each forecast step.

## Data Source

| Source | Series | URL |
|--------|--------|-----|
| FRED | EXKOUS (KRW/USD) | `fred.stlouisfed.org/series/EXKOUS` |
| FRED | KOSPI | `fred.stlouisfed.org/series/SPASTT01KRM661N` |
| FRED | CPI Korea | `fred.stlouisfed.org/series/KORCPIALLMINMEI` |

!!! note "Disclaimer"
    This analysis is for educational purposes only. Do not use for actual investment or business decisions.

---

**Next:** [Showcase 02 — Korean Regression](02_koreanRegression.md)
