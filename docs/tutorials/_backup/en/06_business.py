# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # Business Intelligence

        **Beyond forecasting — tools for decision-making.**

        - **AnomalyDetector**: Automatic outlier detection
        - **WhatIfAnalyzer**: "What if?" scenario simulation
        - **Backtester**: Forecast model validation
        - **BusinessMetrics**: MAPE, RMSE, MAE calculation
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix.business import (
        AnomalyDetector,
        WhatIfAnalyzer,
        Backtester,
        BusinessMetrics,
    )
    return np, pd, AnomalyDetector, WhatIfAnalyzer, Backtester, BusinessMetrics


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. Anomaly Detection

        Detect outliers using Z-score, IQR, seasonal residuals,
        or rolling window methods.
        """
    )
    return


@app.cell
def methodSelector(mo):
    anomalyMethod = mo.ui.dropdown(
        options={
            "Auto": "auto",
            "Z-score": "zscore",
            "IQR": "iqr",
            "Rolling Window": "rolling",
        },
        value="auto",
        label="Detection Method"
    )
    return anomalyMethod,


@app.cell
def showMethodSelector(anomalyMethod):
    anomalyMethod
    return


@app.cell
def createAnomalyData(np):
    np.random.seed(42)
    _n = 200
    _t = np.arange(_n, dtype=np.float64)
    normalData = 100 + 0.2 * _t + 10 * np.sin(2 * np.pi * _t / 7) + np.random.normal(0, 3, _n)

    normalData[45] = 250
    normalData[120] = 20
    normalData[175] = 280
    return normalData,


@app.cell
def runAnomaly(AnomalyDetector, normalData, anomalyMethod):
    detector = AnomalyDetector()
    anomalyResult = detector.detect(normalData, method=anomalyMethod.value)
    return anomalyResult,


@app.cell
def showAnomaly(mo, anomalyResult):
    mo.md(
        f"""
        ### Anomaly Detection Results

        | Item | Value |
        |------|-------|
        | Method | `{anomalyResult.method}` |
        | Anomalies Found | {anomalyResult.nAnomalies} |
        | Anomaly Ratio | {anomalyResult.anomalyRatio:.1%} |
        | Threshold | {anomalyResult.threshold:.2f} |
        | Locations | {list(anomalyResult.indices[:10])} |
        """
    )
    return


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. What-If Scenario Analysis

        Simulate scenarios like "What if trend increases by 10%?"
        or "What if a shock hits on day 30?"
        """
    )
    return


@app.cell
def createScenarioData(np):
    np.random.seed(42)
    basePred = np.linspace(100, 130, 30)
    histData = np.random.normal(100, 10, 100)
    return basePred, histData


@app.cell
def runScenarios(WhatIfAnalyzer, basePred, histData):
    analyzer = WhatIfAnalyzer()
    scenarios = [
        {"name": "Optimistic", "trend_change": 0.1},
        {"name": "Pessimistic", "trend_change": -0.15},
        {"name": "Shock Event", "shock_at": 10, "shock_magnitude": -0.3, "shock_duration": 5},
        {"name": "Level Shift", "level_shift": 0.05},
    ]
    scenarioResults = analyzer.analyze(basePred, histData, scenarios)
    return scenarioResults,


@app.cell
def showScenarios(mo, scenarioResults, pd):
    rows = []
    for sr in scenarioResults:
        rows.append({
            "Scenario": sr.name,
            "Mean Forecast": round(sr.predictions.mean(), 2),
            "vs Baseline": f"{sr.impact:+.1f}%",
        })
    scenDf = pd.DataFrame(rows)
    mo.md("### Scenario Comparison")
    return scenDf,


@app.cell
def showScenarioTable(mo, scenDf):
    return mo.ui.table(scenDf)


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. Backtesting

        Validate model performance with walk-forward validation.
        Supports expanding and sliding window strategies.
        """
    )
    return


@app.cell
def runBacktest(Backtester, np):
    np.random.seed(42)
    _n = 300
    _t = np.arange(_n, dtype=np.float64)
    btData = 100 + 0.3 * _t + 10 * np.sin(2 * np.pi * _t / 7) + np.random.normal(0, 3, _n)

    class _NaiveModel:
        def fit(self, train):
            self._last = train[-1]
        def predict(self, steps):
            pred = np.full(steps, self._last)
            return pred, pred - 10, pred + 10

    bt = Backtester(nFolds=5, horizon=14, strategy='expanding')
    btResult = bt.run(btData, _NaiveModel)
    return btResult,


@app.cell
def showBacktest(mo, btResult, pd):
    mo.md(
        f"""
        ### Backtest Results ({btResult.nFolds} folds)

        | Metric | Mean | Std |
        |--------|------|-----|
        | MAPE | {btResult.avgMAPE:.2f}% | {btResult.mapeStd:.2f}% |
        | RMSE | {btResult.avgRMSE:.2f} | - |
        | MAE | {btResult.avgMAE:.2f} | - |
        | sMAPE | {btResult.avgSMAPE:.2f}% | - |
        | Bias | {btResult.avgBias:+.2f} | - |
        | Best Fold | #{btResult.bestFold} |
        | Worst Fold | #{btResult.worstFold} |
        """
    )
    return


@app.cell
def showFoldDetails(mo, btResult, pd):
    foldRows = []
    for f in btResult.folds:
        foldRows.append({
            "Fold": f.fold,
            "Train Size": f.trainSize,
            "Test Size": f.testSize,
            "MAPE": round(f.mape, 2),
            "RMSE": round(f.rmse, 2),
        })
    foldDf = pd.DataFrame(foldRows)
    mo.md("### Fold Details")
    return foldDf,


@app.cell
def showFoldTable(mo, foldDf):
    return mo.ui.table(foldDf)


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. Business Metrics

        Calculate various metrics comparing actuals vs predictions.
        """
    )
    return


@app.cell
def runMetrics(BusinessMetrics, np):
    np.random.seed(42)
    actual = np.array([100, 120, 130, 115, 140, 160, 150, 170])
    predicted = np.array([105, 118, 135, 110, 145, 155, 148, 175])

    metrics = BusinessMetrics()
    metricsResult = metrics.calculate(actual, predicted)
    return actual, predicted, metricsResult


@app.cell
def showMetrics(mo, metricsResult):
    mo.md(
        f"""
        ### Metrics Results

        | Metric | Value | Interpretation |
        |--------|-------|----------------|
        | **Bias** | {metricsResult.get('bias', 0):+.2f} | Positive=over-forecast |
        | **Bias %** | {metricsResult.get('biasPercent', 0):+.2f}% | Percentage bias |
        | **WAPE** | {metricsResult.get('wape', 0):.2f}% | Weighted Absolute Percentage Error |
        | **MASE** | {metricsResult.get('mase', 0):.2f} | <1 means better than Naive |
        | **Forecast Accuracy** | {metricsResult.get('forecastAccuracy', 0):.1f}% | Higher is better |
        | **Over-forecast** | {metricsResult.get('overForecastRatio', 0):.1%} | Predicted > Actual |
        | **Under-forecast** | {metricsResult.get('underForecastRatio', 0):.1%} | Predicted < Actual |

        ---

        You've completed the tutorial series.

        For more details, visit [GitHub](https://github.com/eddmpython/vectrix).
        """
    )
    return


if __name__ == "__main__":
    app.run()
