# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vectrix",
#     "pandas",
#     "numpy",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from vectrix import forecast, analyze, compare

    return analyze, compare, forecast, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        """
# Model Comparison & Adaptive Intelligence

Compare **30+ forecasting models** side-by-side on real data.
Vectrix automatically selects the best model, but you can inspect all candidates.
"""
    )


@app.cell
def _(mo):
    mo.md(
        """
## 1. Generate Realistic Data

Monthly sales data with trend, seasonality, and noise — a common pattern in business forecasting.
"""
    )


@app.cell
def _(np, pd):
    np.random.seed(42)
    _n = 120
    _t = np.arange(_n, dtype=np.float64)
    _trend = 100 + 0.8 * _t
    _seasonal = 25 * np.sin(2 * np.pi * _t / 12) + 10 * np.cos(2 * np.pi * _t / 6)
    _noise = np.random.normal(0, 8, _n)

    salesDf = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=_n, freq="MS"),
        "revenue": _trend + _seasonal + _noise,
    })
    salesDf
    return (salesDf,)


@app.cell
def _(mo):
    mo.md(
        """
## 2. DNA Analysis

Before forecasting, understand what kind of data you're working with.
The DNA profiler extracts a unique fingerprint: difficulty, category, and recommended models.
"""
    )


@app.cell
def _(analyze, mo, salesDf):
    report = analyze(salesDf, date="date", value="revenue")
    mo.md(
        f"""
| Property | Value |
|----------|-------|
| Category | {report.dna.category} |
| Difficulty | {report.dna.difficulty} ({report.dna.difficultyScore:.0f}/100) |
| Fingerprint | `{report.dna.fingerprint}` |
| Trend | {report.trend} |
| Seasonality | period = {report.seasonalPeriod} |
| Changepoints | {len(report.changepoints)} detected |
| Recommended | {', '.join(report.dna.recommendedModels[:5])} |
"""
    )
    return (report,)


@app.cell
def _(mo):
    mo.md(
        """
## 3. Forecast & Compare All Models

`forecast()` runs 30+ models and selects the best one.
Use `.compare()` to see how every model performed.
"""
    )


@app.cell
def _(forecast, salesDf):
    result = forecast(salesDf, date="date", value="revenue", steps=12)
    return (result,)


@app.cell
def _(mo, result):
    mo.md(
        f"""
### Best Model: `{result.model}`

| Metric | Value |
|--------|-------|
| MAPE | {result.mape:.2f}% |
| RMSE | {result.rmse:.2f} |
| MAE | {result.mae:.2f} |
| sMAPE | {result.smape:.2f}% |
"""
    )


@app.cell
def _(mo):
    mo.md("### All Model Rankings")


@app.cell
def _(result):
    comparisonDf = result.compare()
    comparisonDf


@app.cell
def _(mo):
    mo.md(
        """
## 4. All Model Forecasts

Every model's future predictions in a single DataFrame.
Useful for building custom ensembles or analyzing disagreement between models.
"""
    )


@app.cell
def _(result):
    allForecasts = result.all_forecasts()
    allForecasts


@app.cell
def _(mo, result):
    mo.md(
        f"""
## 5. Forecast Summary

```
{result.summary()}
```
"""
    )


@app.cell
def _(result):
    result.to_dataframe()


@app.cell
def _(mo):
    mo.md(
        """
## 6. Side-by-Side Comparison

Use the top-level `compare()` function for a quick one-liner comparison.
"""
    )


@app.cell
def _(compare, salesDf):
    quickCompare = compare(salesDf, date="date", value="revenue", steps=12)
    quickCompare


@app.cell
def _(mo):
    mo.md(
        """
> **Note**: Model rankings may vary between runs due to cross-validation splits.
> The best model is selected based on out-of-sample accuracy, not in-sample fit.
"""
    )


if __name__ == "__main__":
    app.run()
