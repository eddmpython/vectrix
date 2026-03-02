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
    from vectrix import forecast, analyze
    from vectrix.business import AnomalyDetector, WhatIfAnalyzer, Backtester, BusinessMetrics
    from vectrix.engine.ets import AutoETS

    return (
        AnomalyDetector,
        AutoETS,
        Backtester,
        BusinessMetrics,
        WhatIfAnalyzer,
        analyze,
        forecast,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        """
# Business Intelligence Showcase

End-to-end business forecasting workflow: **anomaly detection**, **what-if scenarios**,
**backtesting**, and **business metrics** — all with Vectrix.
"""
    )


@app.cell
def _(mo):
    mo.md("## 1. Prepare Business Data")


@app.cell
def _(np, pd):
    np.random.seed(42)
    _n = 150
    _t = np.arange(_n, dtype=np.float64)
    _trend = 200 + 1.2 * _t
    _seasonal = 30 * np.sin(2 * np.pi * _t / 12)
    _noise = np.random.normal(0, 10, _n)
    _values = _trend + _seasonal + _noise

    _values[45] = 600
    _values[90] = 50
    _values[130] = 700

    bizDf = pd.DataFrame({
        "date": pd.date_range("2013-01-01", periods=_n, freq="MS"),
        "revenue": _values,
    })
    bizDf
    return (bizDf,)


@app.cell
def _(mo):
    mo.md(
        """
## 2. Anomaly Detection

Detect unusual data points before forecasting.
Anomalies can distort model selection and prediction accuracy.
"""
    )


@app.cell
def _(AnomalyDetector, bizDf, mo, np):
    detector = AnomalyDetector()
    _y = np.array(bizDf["revenue"], dtype=np.float64)
    anomResult = detector.detect(_y, sensitivity=0.95)

    _rows = []
    for _idx in anomResult.indices:
        _rows.append(
            f"| {_idx} | {bizDf['date'].iloc[_idx].strftime('%Y-%m')} "
            f"| {_y[_idx]:,.1f} | {anomResult.scores[_idx]:+.2f} |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
**{len(anomResult.indices)} anomalies detected**

| Index | Date | Value | Z-Score |
|-------|------|-------|---------|
{_table}
"""
    )
    return (anomResult,)


@app.cell
def _(mo):
    mo.md(
        """
## 3. Forecast

Run the full forecasting pipeline on the data (including anomalies — Vectrix handles them internally).
"""
    )


@app.cell
def _(bizDf, forecast):
    fcResult = forecast(bizDf, date="date", value="revenue", steps=12)
    return (fcResult,)


@app.cell
def _(fcResult, mo):
    mo.md(
        f"""
### Forecast Result

| Item | Value |
|------|-------|
| Model | `{fcResult.model}` |
| MAPE | {fcResult.mape:.2f}% |
| 12-month mean | {fcResult.predictions.mean():,.1f} |
| 95% CI width | {(fcResult.upper - fcResult.lower).mean():,.1f} |
"""
    )


@app.cell
def _(fcResult):
    fcResult.to_dataframe()


@app.cell
def _(mo):
    mo.md(
        """
## 4. What-If Scenario Analysis

Explore how different business conditions would change the forecast.
"""
    )


@app.cell
def _(WhatIfAnalyzer, bizDf, fcResult, mo, np):
    analyzer = WhatIfAnalyzer()
    _scenarios = [
        {"name": "base", "trend_change": 0},
        {"name": "growth_10pct", "trend_change": 0.10},
        {"name": "recession", "trend_change": -0.15, "level_shift": -0.05},
        {"name": "supply_shock", "shock_at": 3, "shock_magnitude": -0.25, "shock_duration": 3},
        {"name": "expansion", "level_shift": 0.10, "seasonal_multiplier": 1.3},
    ]
    _historical = np.array(bizDf["revenue"], dtype=np.float64)
    scenarioResults = analyzer.analyze(fcResult.predictions, _historical, _scenarios, period=12)

    _rows = []
    for _sr in scenarioResults:
        _rows.append(
            f"| {_sr.name} | {_sr.impact:.1f}% | {_sr.percentChange[-1]:+.1f}% |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
| Scenario | Avg Impact | Final Change |
|----------|-----------|--------------|
{_table}
"""
    )
    return (scenarioResults,)


@app.cell
def _(WhatIfAnalyzer, mo, scenarioResults):
    mo.md(f"```\n{WhatIfAnalyzer().compareSummary(scenarioResults)}\n```")


@app.cell
def _(mo):
    mo.md(
        """
## 5. Backtesting

Walk-forward validation: how accurate would the model have been historically?
"""
    )


@app.cell
def _(AutoETS, Backtester, bizDf, mo, np):
    bt = Backtester(nFolds=4, horizon=12, strategy="expanding", minTrainSize=60)
    _y = np.array(bizDf["revenue"], dtype=np.float64)
    btResult = bt.run(_y, modelFactory=AutoETS)
    mo.md(f"```\n{bt.summary(btResult)}\n```")
    return (btResult,)


@app.cell
def _(btResult, mo):
    _rows = []
    for _fold in btResult.folds:
        _rows.append(
            f"| {_fold.fold} | {_fold.trainSize} | {_fold.testSize} | {_fold.mape:.2f}% |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
### Fold Details

| Fold | Train | Test | MAPE |
|------|-------|------|------|
{_table}
"""
    )


@app.cell
def _(mo):
    mo.md(
        """
## 6. Business Metrics

Calculate standard business accuracy metrics for any actual vs. predicted comparison.
"""
    )


@app.cell
def _(BusinessMetrics, mo, np):
    _actuals = np.array([320, 340, 310, 360, 345, 370, 355, 380, 365, 390, 375, 400])
    _predicted = np.array([325, 335, 315, 355, 350, 365, 360, 375, 370, 385, 380, 395])

    metrics = BusinessMetrics()
    metricResult = metrics.calculate(_actuals, _predicted)

    _rows = []
    for _k, _v in metricResult.items():
        _rows.append(f"| {_k} | {_v:.4f} |")
    _table = "\n".join(_rows)

    mo.md(
        f"""
| Metric | Value |
|--------|-------|
{_table}
"""
    )


@app.cell
def _(mo):
    mo.md(
        """
> **Tip**: Combine anomaly detection + backtesting + what-if analysis for robust business planning.
> Detect outliers first, validate model accuracy, then explore scenarios.
"""
    )


if __name__ == "__main__":
    app.run()
