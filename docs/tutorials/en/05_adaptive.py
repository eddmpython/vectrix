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
        # Adaptive Forecasting

        **Unique to Vectrix — not available in any other library.**

        - **RegimeDetector**: HMM-based regime change detection
        - **ForecastDNA**: Time series DNA fingerprinting + difficulty scoring
        - **SelfHealingForecast**: Automatic error detection & correction
        - **ConstraintAwareForecaster**: Business constraint enforcement
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import (
        RegimeDetector,
        RegimeAwareForecaster,
        ForecastDNA,
        SelfHealingForecast,
        ConstraintAwareForecaster,
        Constraint,
    )
    return (np, pd, RegimeDetector, RegimeAwareForecaster,
            ForecastDNA, SelfHealingForecast,
            ConstraintAwareForecaster, Constraint)


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. Regime Detection

        Automatically detect "regimes" in time series data.
        Identifies bull/bear/sideways in financial data, peak/off-season in demand data.
        """
    )
    return


@app.cell
def regimeSlider(mo):
    nRegimesControl = mo.ui.slider(
        start=2, stop=5, step=1, value=3,
        label="Number of Regimes"
    )
    return nRegimesControl,


@app.cell
def showRegimeSlider(nRegimesControl):
    nRegimesControl
    return


@app.cell
def createRegimeData(np):
    np.random.seed(42)
    regime1 = np.random.normal(100, 5, 80)
    regime2 = np.random.normal(150, 15, 60)
    regime3 = np.random.normal(80, 3, 60)
    regimeData = np.concatenate([regime1, regime2, regime3])
    return regimeData,


@app.cell
def runRegimeDetection(RegimeDetector, regimeData, nRegimesControl):
    detector = RegimeDetector(nRegimes=nRegimesControl.value)
    regimeResult = detector.detect(regimeData)
    return regimeResult,


@app.cell
def showRegimeResult(mo, regimeResult, nRegimesControl):
    nRegimes = len(regimeResult.regimeStats)
    nTransitions = max(0, len(regimeResult.regimeHistory) - 1)
    mo.md(
        f"""
        ### Regime Detection Results ({nRegimesControl.value} regimes)

        | Item | Value |
        |------|-------|
        | Current Regime | {regimeResult.currentRegime} |
        | Detected Regimes | {nRegimes} |
        | Transitions | {nTransitions} |

        **Per-Regime Statistics:**
        """
    )
    return


@app.cell
def showRegimeStats(mo, regimeResult, pd):
    rows = []
    for label, stats in regimeResult.regimeStats.items():
        rows.append({
            "Regime": label,
            "Mean": round(stats.get('mean', 0), 2),
            "Std": round(stats.get('std', 0), 2),
        })
    statsDf = pd.DataFrame(rows)
    return mo.ui.table(statsDf)


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. Regime-Aware Forecasting

        Automatically switches to optimal models per regime.
        """
    )
    return


@app.cell
def runRegimeForecast(RegimeAwareForecaster, regimeData):
    raf = RegimeAwareForecaster()
    rafResult = raf.forecast(regimeData, steps=30, period=1)
    return rafResult,


@app.cell
def showRegimeForecast(mo, rafResult):
    mo.md(
        f"""
        ### Regime-Aware Forecast

        | Item | Value |
        |------|-------|
        | Current Regime | {rafResult.currentRegime} |
        | Model per Regime | {rafResult.modelPerRegime} |
        | Forecast Horizon | {len(rafResult.predictions)} periods |
        | Mean Prediction | {rafResult.predictions.mean():.2f} |
        """
    )
    return


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. Forecast DNA

        Extract the "DNA" of a time series to generate a fingerprint,
        assess difficulty, and recommend optimal models.
        """
    )
    return


@app.cell
def runDna(ForecastDNA, np):
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=np.float64)
    seasonalData = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

    dna = ForecastDNA()
    profile = dna.analyze(seasonalData, period=7)
    return profile,


@app.cell
def showDnaProfile(mo, profile):
    mo.md(
        f"""
        ### DNA Profile

        | Item | Value |
        |------|-------|
        | Fingerprint | `{profile.fingerprint}` |
        | Difficulty | **{profile.difficulty}** ({profile.difficultyScore:.0f}/100) |
        | Category | **{profile.category}** |
        | Recommended Models | {', '.join(f'`{m}`' for m in profile.recommendedModels[:5])} |

        The DNA fingerprint is deterministic — identical data always produces
        the same fingerprint. Similar patterns yield similar fingerprints.
        """
    )
    return


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. Self-Healing Forecast

        Monitors forecast errors in real-time and auto-corrects predictions.
        Continuously improves forecast quality in production environments.
        """
    )
    return


@app.cell
def runSelfHealing(SelfHealingForecast, np):
    np.random.seed(42)
    n = 50
    originalPred = np.full(n, 100.0)
    lower = originalPred - 10
    upper = originalPred + 10
    historicalData = np.random.normal(100, 5, 100)

    healer = SelfHealingForecast(originalPred, lower, upper, historicalData)

    actualValues = np.array([105, 110, 115, 108, 120])
    healer.observe(actualValues)
    healingReport = healer.getReport()
    updatedForecast = healer.getUpdatedForecast()
    return healer, healingReport, updatedForecast, originalPred


@app.cell
def showHealing(mo, healingReport, originalPred, updatedForecast):
    mo.md(
        f"""
        ### Self-Healing Results

        | Item | Value |
        |------|-------|
        | Health Status | {healingReport.overallHealth} |
        | Health Score | {healingReport.healthScore:.1f}/100 |
        | Observations | {healingReport.totalObserved} |
        | Corrections | {healingReport.totalCorrected} |
        | Original MAPE | {healingReport.originalMape:.2f}% |
        | Healed MAPE | {healingReport.healedMape:.2f}% |
        | Improvement | {healingReport.improvementPct:.1f}% |

        When actuals consistently exceed forecasts, the self-healing system
        adjusts remaining predictions upward.
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. Constraint-Aware Forecasting

        Apply business constraints to predictions:
        - Non-negative (inventory, revenue)
        - Range limits (capacity, budget)
        - Rate-of-change limits (prevent extreme swings)
        """
    )
    return


@app.cell
def runConstraints(ConstraintAwareForecaster, Constraint, np):
    np.random.seed(42)
    rawPred = np.array([150, -20, 300, 50, 6000, 80, 120, 250, -10, 400])
    lower = rawPred - 30
    upper = rawPred + 30

    caf = ConstraintAwareForecaster()
    constrainedResult = caf.apply(rawPred, lower, upper, constraints=[
        Constraint('non_negative', {}),
        Constraint('range', {'min': 0, 'max': 500}),
    ])
    return rawPred, constrainedResult


@app.cell
def showConstraints(mo, rawPred, constrainedResult, pd):
    compDf = pd.DataFrame({
        "Original": rawPred,
        "Constrained": constrainedResult.predictions,
        "Changed": rawPred != constrainedResult.predictions,
    })

    mo.md(
        f"""
        ### Constraint Results

        Constraints applied: `non_negative` + `range(0, 500)`

        ```
        {compDf.to_string(index=False)}
        ```

        -20 → 0 (non-negative), 6000 → 500 (range capped)

        **Next tutorial:** `06_business.py` — Business Intelligence
        """
    )
    return


if __name__ == "__main__":
    app.run()
