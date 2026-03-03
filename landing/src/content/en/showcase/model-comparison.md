---
title: Model Comparison
---

# Model Comparison

Compare 30+ forecasting models side-by-side with a single function call. Vectrix automatically evaluates every compatible model and ranks them by accuracy.

## Quick One-Liner

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170, 180, 195], steps=5)
print(result.compare())
```

This prints a comparison table of all valid models with sMAPE, MAPE, RMSE, and MAE metrics, sorted by accuracy.

## DNA Analysis First

Before comparing models, understand your data with DNA profiling

```python
from vectrix import analyze

report = analyze([100, 120, 130, 115, 140, 160, 150, 170, 180, 195])

print(f"Trend: {report.characteristics.hasTrend} ({report.characteristics.trendDirection})")
print(f"Seasonality: {report.characteristics.hasSeasonality}")
print(f"Volatility: {report.characteristics.volatility}")
print(f"Difficulty: {report.dna.difficulty}")
print(f"Recommended models: {report.dna.recommendedModels}")
```

DNA profiling extracts 65+ time series features and recommends the best-fit models based on a meta-learning system.

## Forecast and Compare

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=12)

print(f"Best model: {result.model}")
print(f"sMAPE: {result.smape:.2f}")
print(f"MAPE:  {result.mape:.2f}")
print(f"RMSE:  {result.rmse:.2f}")

comparison = result.compare()
print(comparison)
```

The `compare()` method returns a DataFrame

```
              sMAPE     MAPE     RMSE      MAE
DOT           3.214    3.187   12.453    9.876
AutoCES       3.456    3.421   13.102   10.234
AutoETS       3.789    3.752   14.567   11.345
FourTheta     3.891    3.844   14.892   11.678
...
```

## All Model Forecasts

Access the raw forecast values from every model

```python
allForecasts = result.all_forecasts()
print(allForecasts)
```

This returns a DataFrame where each column is a model and each row is a forecast step

```
   step        DOT    AutoCES    AutoETS  FourTheta  ...
0     1    152.340    153.120    151.890    154.230  ...
1     2    155.670    156.440    154.230    157.560  ...
2     3    158.120    159.890    157.560    160.120  ...
...
```

## Using a Specific Model

If DNA or comparison suggests a particular model

```python
from vectrix.engine.dot import DynamicOptimizedTheta

model = DynamicOptimizedTheta(period=12)
model.fit(data)
pred, lower, upper = model.predict(steps=12)
```

## Available Models

Vectrix includes 30+ models across several categories

**Statistical**
- AutoETS (30 state-space combinations)
- AutoARIMA (stepwise order selection)
- Theta / Dynamic Optimized Theta (DOT)
- AutoCES (Complex Exponential Smoothing)
- FourTheta (Adaptive Theta Ensemble)
- AutoTBATS (Trigonometric multi-seasonal)
- AutoMSTL (Multi-seasonal STL decomposition)

**Volatility**
- GARCH / EGARCH / GJR-GARCH

**Intermittent Demand**
- Croston Classic / SBA / TSB / AutoCroston

**Novel Methods**
- DTSF (Dynamic Time Scan Forecaster)
- ESN (Echo State Network)
- Lotka-Volterra Ensemble
- Phase Transition Forecaster
- Hawkes Intermittent Demand

**Baseline**
- Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

**Optional (requires extra dependencies)**
- Chronos (Amazon zero-shot)
- TimesFM (Google zero-shot)
- NBEATS / NHITS / TFT (deep learning)

## Full Workflow Example

```python
import pandas as pd
from vectrix import forecast, analyze

df = pd.read_csv("monthly_sales.csv")

report = analyze(df, date="date", value="sales")
print(f"DNA difficulty: {report.dna.difficulty}")
print(f"Recommended: {report.dna.recommendedModels}")

result = forecast(df, date="date", value="sales", steps=12)

print(f"\nBest model: {result.model}")
print(f"\nAll model comparison:")
print(result.compare())

print(f"\nAll forecasts:")
print(result.all_forecasts())

result.to_dataframe().to_csv("forecast_output.csv", index=False)
```

> **Tip:** The `compare()` output is a standard pandas DataFrame. You can sort, filter, or export it like any other DataFrame.
