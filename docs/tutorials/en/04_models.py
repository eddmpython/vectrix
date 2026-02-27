# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
#     "matplotlib",
# ]
# ///

import marimo

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
    n = 200
    t = np.arange(n, dtype=np.float64)
    values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

    tsDf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
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
    c = fcResult.characteristics
    fr = fcResult.flatRisk

    mo.md(
        f"""
        | Item | Value |
        |------|-------|
        | **Best Model** | `{fcResult.bestModelName}` |
        | Success | {fcResult.success} |
        | Data Length | {c.length} |
        | Detected Period | {c.period} |
        | Trend | {c.trendDirection} (strength {c.trendStrength:.2f}) |
        | Seasonality Strength | {c.seasonalStrength:.2f} |
        | Flat Risk | {fr.riskLevel.name} (score {fr.riskScore:.2f}) |
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
    fr = fcResult.flatRisk

    factorStr = ""
    for factor, active in fr.riskFactors.items():
        if active:
            factorStr += f"- {factor}\n"
    if not factorStr:
        factorStr = "- None"

    warningStr = ""
    for w in (fr.warnings or []):
        warningStr += f"- {w}\n"
    if not warningStr:
        warningStr = "- None"

    mo.md(
        f"""
        ### Flat Risk Analysis

        **Risk Level:** {fr.riskLevel.name} (score: {fr.riskScore:.2f})

        **Risk Factors:**
        {factorStr}

        **Recommended Strategy:** {fr.recommendedStrategy}

        **Warnings:**
        {warningStr}
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
