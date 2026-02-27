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
        # Time Series DNA Analysis

        **Automatically analyze data characteristics and get optimal forecasting strategy recommendations.**

        Vectrix's `analyze()` function extracts the "DNA" of your time series:
        - Difficulty score (0-100)
        - Category classification (trending, seasonal, volatile, etc.)
        - Automatic changepoint & anomaly detection
        - Optimal model recommendations
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import analyze, quick_report
    return np, pd, analyze, quick_report


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. Select Sample Data

        Choose different time series patterns to compare DNA analysis results.
        """
    )
    return


@app.cell
def dataSelector(mo):
    dataChoice = mo.ui.dropdown(
        options={
            "Trend + Seasonality": "trendSeasonal",
            "Pure Trend": "pureTrend",
            "High Volatility": "volatile",
            "Intermittent Demand": "intermittent",
            "Multi-Seasonality": "multiSeasonal",
        },
        value="trendSeasonal",
        label="Data Pattern"
    )
    return dataChoice,


@app.cell
def showSelector(dataChoice):
    dataChoice
    return


@app.cell
def generateData(np, pd, dataChoice):
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=np.float64)

    choice = dataChoice.value
    if choice == "trendSeasonal":
        values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)
        desc = "Linear trend + weekly seasonality + noise"
    elif choice == "pureTrend":
        values = 50 + 0.8 * t + np.random.normal(0, 2, n)
        desc = "Strong upward trend + weak noise"
    elif choice == "volatile":
        returns = np.zeros(n)
        sigma2 = np.ones(n)
        for i in range(1, n):
            sigma2[i] = 0.05 + 0.1 * returns[i - 1] ** 2 + 0.85 * sigma2[i - 1]
            returns[i] = np.random.normal(0, np.sqrt(sigma2[i]))
        values = 100 + np.cumsum(returns)
        desc = "GARCH-style volatility clustering"
    elif choice == "intermittent":
        values = np.zeros(n)
        for i in range(n):
            if np.random.random() < 0.3:
                values[i] = np.random.exponential(50)
        desc = "70% zeros, intermittent demand pattern"
    else:
        values = (100 + 0.2 * t
                  + 10 * np.sin(2 * np.pi * t / 7)
                  + 20 * np.sin(2 * np.pi * t / 30)
                  + np.random.normal(0, 3, n))
        desc = "Weekly (7) + monthly (30) dual seasonality"

    sampleDf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "value": values,
    })
    return sampleDf, desc


@app.cell
def showDataDesc(mo, desc):
    mo.md(f"**Selected pattern:** {desc}")
    return


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. Run DNA Analysis
        """
    )
    return


@app.cell
def runAnalysis(analyze, sampleDf):
    report = analyze(sampleDf)
    return report,


@app.cell
def showDna(mo, report):
    dna = report.dna
    mo.md(
        f"""
        ### DNA Profile

        | Item | Result |
        |------|--------|
        | Difficulty | **{dna.difficulty}** ({dna.difficultyScore:.0f}/100) |
        | Category | **{dna.category}** |
        | Fingerprint | `{dna.fingerprint}` |
        | Recommended Models | {', '.join(f'`{m}`' for m in dna.recommendedModels[:3])} |
        """
    )
    return


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. Data Characteristics
        """
    )
    return


@app.cell
def showCharacteristics(mo, report):
    c = report.characteristics
    mo.md(
        f"""
        | Characteristic | Value |
        |----------------|-------|
        | Length | {c.length} |
        | Frequency | {c.frequency} |
        | Period | {c.period} |
        | Trend | {'Yes' if c.hasTrend else 'No'} ({c.trendDirection}, strength {c.trendStrength:.2f}) |
        | Seasonality | {'Yes' if c.hasSeasonality else 'No'} (strength {c.seasonalStrength:.2f}) |
        | Volatility | {c.volatilityLevel} ({c.volatility:.4f}) |
        | Predictability | {c.predictabilityScore:.0f}/100 |
        | Outliers | {c.outlierCount} ({c.outlierRatio:.1%}) |
        """
    )
    return


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. Changepoints & Anomalies
        """
    )
    return


@app.cell
def showDetection(mo, report):
    nCp = len(report.changepoints) if report.changepoints is not None else 0
    nAn = len(report.anomalies) if report.anomalies is not None else 0

    cpStr = str(list(report.changepoints[:5])) if nCp > 0 else "None"
    anStr = str(list(report.anomalies[:5])) if nAn > 0 else "None"

    mo.md(
        f"""
        | Detection | Count | Locations (up to 5) |
        |-----------|-------|---------------------|
        | Changepoints | {nCp} | {cpStr} |
        | Anomalies | {nAn} | {anStr} |
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. Quick Report

        `quick_report()` runs analysis + forecast in one call.
        """
    )
    return


@app.cell
def runQuickReport(quick_report, sampleDf):
    fullReport = quick_report(sampleDf, steps=14)
    return fullReport,


@app.cell
def showQuickReport(mo, fullReport):
    mo.md(
        f"""
        ```
        {fullReport['summary']}
        ```

        **Next tutorial:** `03_regression.py` — R-style Regression
        """
    )
    return


if __name__ == "__main__":
    app.run()
