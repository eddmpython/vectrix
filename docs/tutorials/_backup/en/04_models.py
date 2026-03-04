# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # 30+ Model Comparison

        **Vectrix automatically runs 30+ models and selects the best one.**

        In this tutorial, you'll use the `Vectrix` class directly to:
        - Compare performance across all models
        - See how the Flat Defense system works
        - Understand ensemble vs individual model differences
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import Vectrix
    return np, pd, Vectrix


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. Data Preparation & Forecasting
        """
    )
    return


@app.cell
def trainRatioSlider(mo):
    ratioControl = mo.ui.slider(
        start=0.6, stop=0.9, step=0.05, value=0.8,
        label="Train Ratio"
    )
    return ratioControl,


@app.cell
def showRatioSlider(ratioControl):
    ratioControl
    return


@app.cell
def createData(np, pd):
    np.random.seed(42)
    _n = 200
    _t = np.arange(_n, dtype=np.float64)
    values = 100 + 0.3 * _t + 15 * np.sin(2 * np.pi * _t / 7) + np.random.normal(0, 3, _n)

    tsDf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=_n, freq="D"),
        "value": values,
    })
    return tsDf,


@app.cell
def runForecast(Vectrix, tsDf, ratioControl):
    vx = Vectrix(verbose=False)
    fcResult = vx.forecast(
        tsDf,
        dateCol="date",
        valueCol="value",
        steps=14,
        trainRatio=ratioControl.value
    )
    return vx, fcResult


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. Model Performance Comparison
        """
    )
    return


@app.cell
def modelComparison(mo, fcResult, pd):
    rows = []
    if fcResult.allModelResults:
        for modelId, mr in fcResult.allModelResults.items():
            isFlat = ""
            if mr.flatInfo and mr.flatInfo.isFlat:
                isFlat = mr.flatInfo.flatType
            rows.append({
                "Model": mr.modelName,
                "MAPE": round(mr.mape, 2),
                "RMSE": round(mr.rmse, 2),
                "MAE": round(mr.mae, 2),
                "Flat": isFlat,
                "Time(s)": round(mr.trainingTime, 3),
            })

    compDf = pd.DataFrame(rows).sort_values("MAPE")
    mo.md(f"### All Model Results ({len(rows)} models)")
    return compDf,


@app.cell
def showComparison(mo, compDf):
    return mo.ui.table(compDf)


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. Best Model & Data Characteristics
        """
    )
    return


@app.cell
def showBestModel(mo, fcResult):
    _c = fcResult.characteristics
    _fr = fcResult.flatRisk

    mo.md(
        f"""
        | Item | Value |
        |------|-------|
        | **Best Model** | `{fcResult.bestModelName}` |
        | Success | {fcResult.success} |
        | Data Length | {_c.length} |
        | Detected Period | {_c.period} |
        | Trend | {_c.trendDirection} (strength {_c.trendStrength:.2f}) |
        | Seasonality Strength | {_c.seasonalStrength:.2f} |
        | Flat Risk | {_fr.riskLevel.name} (score {_fr.riskScore:.2f}) |
        """
    )
    return


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. Flat Defense System

        Vectrix defends against flat predictions in 4 stages:

        1. **FlatRiskDiagnostic** — Pre-diagnosis risk assessment
        2. **AdaptiveModelSelector** — Risk-based model selection
        3. **FlatPredictionDetector** — Post-detection of flat forecasts
        4. **FlatPredictionCorrector** — Intelligent correction

        This is unique to Vectrix — no other library has this.
        """
    )
    return


@app.cell
def showFlatDefense(mo, fcResult):
    _fr = fcResult.flatRisk

    _factorStr = ""
    for _factor, _active in _fr.riskFactors.items():
        if _active:
            _factorStr += f"- {_factor}\n"
    if not _factorStr:
        _factorStr = "- None"

    _warningStr = ""
    for _w in (_fr.warnings or []):
        _warningStr += f"- {_w}\n"
    if not _warningStr:
        _warningStr = "- None"

    mo.md(
        f"""
        ### Flat Risk Analysis

        **Risk Level:** {_fr.riskLevel.name} (score: {_fr.riskScore:.2f})

        **Risk Factors:**
        {_factorStr}

        **Recommended Strategy:** {_fr.recommendedStrategy}

        **Warnings:**
        {_warningStr}
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. Model Categories

        | Category | Models | Best For |
        |----------|--------|----------|
        | **Exponential Smoothing** | AutoETS, ETS(A,A,A), ETS(A,A,N) | Stable patterns |
        | **ARIMA** | AutoARIMA | Stationary series |
        | **Decomposition** | MSTL, AutoMSTL | Multi-seasonality |
        | **Theta** | Theta, DOT | General purpose |
        | **Trigonometric** | TBATS | Complex seasonality |
        | **Complex** | AutoCES | Non-linear patterns |
        | **Intermittent** | Croston, SBA, TSB | Sparse demand |
        | **Volatility** | GARCH, EGARCH, GJR | Financial data |
        | **Baseline** | Naive, Seasonal Naive, Mean, RWD | Benchmarks |

        **Next tutorial:** `05_adaptive.py` — Adaptive Forecasting
        """
    )
    return


if __name__ == "__main__":
    app.run()
