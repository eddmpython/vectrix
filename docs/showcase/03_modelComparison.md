# Showcase 03 — Model Comparison & Adaptive Intelligence

**Compare 30+ forecasting models side-by-side and explore adaptive features.**

## Overview

Vectrix automatically selects the best model from 30+ candidates, but you can inspect all results:

- **DNA Analysis** — Understand your data's difficulty, category, and fingerprint
- **Model Ranking** — See how every model performed (MAPE, RMSE, MAE, sMAPE)
- **All Forecasts** — Get every model's predictions in a single DataFrame
- **Quick Compare** — One-liner comparison with `compare()`

## Run Interactively

```bash
pip install vectrix pandas numpy marimo
marimo run docs/showcase/en/03_modelComparison.py
```

## Code

### Generate Data

```python
import numpy as np
import pandas as pd
from vectrix import forecast, analyze, compare

np.random.seed(42)
n = 120
t = np.arange(n, dtype=np.float64)
trend = 100 + 0.8 * t
seasonal = 25 * np.sin(2 * np.pi * t / 12) + 10 * np.cos(2 * np.pi * t / 6)
noise = np.random.normal(0, 8, n)

salesDf = pd.DataFrame({
    "date": pd.date_range("2015-01-01", periods=n, freq="MS"),
    "revenue": trend + seasonal + noise,
})
```

### DNA Analysis

```python
report = analyze(salesDf, date="date", value="revenue")
print(f"Category: {report.dna.category}")
print(f"Difficulty: {report.dna.difficulty} ({report.dna.difficultyScore:.0f}/100)")
print(f"Recommended: {report.dna.recommendedModels[:5]}")
```

### Forecast & Compare

```python
result = forecast(salesDf, date="date", value="revenue", steps=12)

print(f"Best model: {result.model}")
print(f"MAPE: {result.mape:.2f}%")

ranking = result.compare()
print(ranking)
```

### All Model Forecasts

```python
allForecasts = result.all_forecasts()
print(allForecasts)
```

This returns a DataFrame with `date` column and one column per model — useful for building custom ensembles.

### Quick One-Liner

```python
comparison = compare(salesDf, date="date", value="revenue", steps=12)
print(comparison)
```

The `compare()` function runs the full pipeline and returns the model comparison DataFrame directly.

!!! tip "Model Selection"
    Models are ranked by **out-of-sample** accuracy (cross-validation), not in-sample fit.
    Rankings may vary between runs due to different CV splits.

---

**Next:** [Showcase 04 — Business Intelligence](04_businessIntelligence.md)
