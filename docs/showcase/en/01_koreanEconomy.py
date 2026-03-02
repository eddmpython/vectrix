# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
#     "matplotlib",
#     "pandas",
# ]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # Korean Economy Forecasting Showcase

        **Real-world time series forecasting with publicly available Korean economic data**

        This notebook demonstrates Vectrix's forecasting capabilities using real macroeconomic
        indicators from South Korea, sourced via [FRED](https://fred.stlouisfed.org/) (Federal
        Reserve Economic Data).

        We will forecast:
        1. USD/KRW Exchange Rate
        2. KOSPI Stock Index
        3. Consumer Price Index (CPI)
        4. Multi-indicator DNA comparison

        > **Disclaimer:** This analysis is for educational purposes only.
        > Do not use for actual investment decisions.
        """
    )
    return


@app.cell
def imports():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from vectrix import forecast, analyze
    return pd, np, plt, forecast, analyze


@app.cell
def scenario1Header(mo):
    mo.md(
        """
        ---
        ## Scenario 1: USD/KRW Exchange Rate Forecast

        The USD/KRW exchange rate is one of the most closely watched indicators in the Korean
        economy. We load monthly data from FRED starting from 1981, and forecast 12 months ahead.

        **Data source:** [FRED EXKOUS](https://fred.stlouisfed.org/series/EXKOUS)
        — South Korea / U.S. Foreign Exchange Rate (Korean Won per U.S. Dollar)
        """
    )
    return


@app.cell
def loadExchangeRate(pd):
    exchangeUrl = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
    exchangeDf = pd.read_csv(exchangeUrl, parse_dates=["DATE"])
    exchangeDf.columns = ["date", "value"]
    exchangeDf = exchangeDf.dropna()
    return exchangeDf,


@app.cell
def showExchangeData(mo, exchangeDf):
    mo.md(
        f"""
        ### Exchange Rate Data Overview

        | Property | Value |
        |----------|-------|
        | Total observations | {len(exchangeDf)} |
        | Date range | {exchangeDf['date'].min().strftime('%Y-%m')} to {exchangeDf['date'].max().strftime('%Y-%m')} |
        | Current rate | {exchangeDf['value'].iloc[-1]:.2f} KRW/USD |
        | Historical min | {exchangeDf['value'].min():.2f} |
        | Historical max | {exchangeDf['value'].max():.2f} |
        """
    )
    return


@app.cell
def forecastExchangeRate(forecast, exchangeDf):
    exchangeResult = forecast(exchangeDf, date="date", value="value", steps=12)
    return exchangeResult,


@app.cell
def showExchangeSummary(mo, exchangeResult):
    mo.md(
        f"""
        ### Exchange Rate Forecast Results

        | Item | Value |
        |------|-------|
        | Selected Model | `{exchangeResult.model}` |
        | Forecast Horizon | {len(exchangeResult.predictions)} months |
        | Predicted Mean | {exchangeResult.predictions.mean():.2f} KRW/USD |
        | Predicted Range | {exchangeResult.predictions.min():.2f} ~ {exchangeResult.predictions.max():.2f} |
        | 95% CI Lower | {exchangeResult.lower.min():.2f} |
        | 95% CI Upper | {exchangeResult.upper.max():.2f} |
        """
    )
    return


@app.cell
def showExchangeFullSummary(mo, exchangeResult):
    mo.md(f"```\n{exchangeResult.summary()}\n```")
    return


@app.cell
def plotExchangeForecast(plt, exchangeDf, exchangeResult, np):
    fig1, ax1 = plt.subplots(figsize=(12, 5))

    recentN = 60
    recentDf = exchangeDf.tail(recentN)
    ax1.plot(recentDf["date"], recentDf["value"], color="#2196F3", linewidth=1.5, label="Historical")

    forecastDates = pd.to_datetime(exchangeResult.dates)
    ax1.plot(forecastDates, exchangeResult.predictions, color="#FF5722", linewidth=2, label="Forecast")
    ax1.fill_between(
        forecastDates,
        exchangeResult.lower,
        exchangeResult.upper,
        alpha=0.2,
        color="#FF5722",
        label="95% CI",
    )

    ax1.set_title("USD/KRW Exchange Rate — 12-Month Forecast", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("KRW per USD")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    return fig1,


@app.cell
def displayExchangePlot(mo, fig1):
    mo.center(mo.as_html(fig1))
    return


@app.cell
def scenario2Header(mo):
    mo.md(
        """
        ---
        ## Scenario 2: KOSPI Stock Index Forecast

        The KOSPI (Korea Composite Stock Price Index) is the benchmark stock market index
        for South Korea. We forecast 12 months ahead using monthly data.

        **Data source:** [FRED SPASTT01KRM661N](https://fred.stlouisfed.org/series/SPASTT01KRM661N)
        — Share Prices: All Shares for Korea
        """
    )
    return


@app.cell
def loadKospi(pd):
    kospiUrl = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SPASTT01KRM661N"
    kospiDf = pd.read_csv(kospiUrl, parse_dates=["DATE"])
    kospiDf.columns = ["date", "value"]
    kospiDf = kospiDf.dropna()
    return kospiDf,


@app.cell
def showKospiData(mo, kospiDf):
    mo.md(
        f"""
        ### KOSPI Data Overview

        | Property | Value |
        |----------|-------|
        | Total observations | {len(kospiDf)} |
        | Date range | {kospiDf['date'].min().strftime('%Y-%m')} to {kospiDf['date'].max().strftime('%Y-%m')} |
        | Latest value | {kospiDf['value'].iloc[-1]:.2f} |
        | Historical min | {kospiDf['value'].min():.2f} |
        | Historical max | {kospiDf['value'].max():.2f} |
        """
    )
    return


@app.cell
def forecastKospi(forecast, kospiDf):
    kospiResult = forecast(kospiDf, date="date", value="value", steps=12)
    return kospiResult,


@app.cell
def showKospiSummary(mo, kospiResult):
    mo.md(
        f"""
        ### KOSPI Forecast Results

        | Item | Value |
        |------|-------|
        | Selected Model | `{kospiResult.model}` |
        | Forecast Horizon | {len(kospiResult.predictions)} months |
        | Predicted Mean | {kospiResult.predictions.mean():.2f} |
        | Predicted Range | {kospiResult.predictions.min():.2f} ~ {kospiResult.predictions.max():.2f} |
        | 95% CI Lower | {kospiResult.lower.min():.2f} |
        | 95% CI Upper | {kospiResult.upper.max():.2f} |
        """
    )
    return


@app.cell
def showKospiFullSummary(mo, kospiResult):
    mo.md(f"```\n{kospiResult.summary()}\n```")
    return


@app.cell
def plotKospiForecast(plt, kospiDf, kospiResult):
    fig2, ax2 = plt.subplots(figsize=(12, 5))

    recentN = 60
    recentKospi = kospiDf.tail(recentN)
    ax2.plot(recentKospi["date"], recentKospi["value"], color="#4CAF50", linewidth=1.5, label="Historical")

    forecastDates = pd.to_datetime(kospiResult.dates)
    ax2.plot(forecastDates, kospiResult.predictions, color="#E91E63", linewidth=2, label="Forecast")
    ax2.fill_between(
        forecastDates,
        kospiResult.lower,
        kospiResult.upper,
        alpha=0.2,
        color="#E91E63",
        label="95% CI",
    )

    ax2.set_title("KOSPI Index — 12-Month Forecast", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Index")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    return fig2,


@app.cell
def displayKospiPlot(mo, fig2):
    mo.center(mo.as_html(fig2))
    return


@app.cell
def scenario3Header(mo):
    mo.md(
        """
        ---
        ## Scenario 3: Korean CPI Forecast

        The Consumer Price Index measures inflation in South Korea.
        CPI data tends to exhibit a strong upward trend with seasonal fluctuations,
        making it an interesting test case for forecasting models.

        **Data source:** [FRED KORCPIALLMINMEI](https://fred.stlouisfed.org/series/KORCPIALLMINMEI)
        — Consumer Price Index: All Items for Korea
        """
    )
    return


@app.cell
def loadCpi(pd):
    cpiUrl = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=KORCPIALLMINMEI"
    cpiDf = pd.read_csv(cpiUrl, parse_dates=["DATE"])
    cpiDf.columns = ["date", "value"]
    cpiDf = cpiDf.dropna()
    return cpiDf,


@app.cell
def showCpiData(mo, cpiDf):
    mo.md(
        f"""
        ### CPI Data Overview

        | Property | Value |
        |----------|-------|
        | Total observations | {len(cpiDf)} |
        | Date range | {cpiDf['date'].min().strftime('%Y-%m')} to {cpiDf['date'].max().strftime('%Y-%m')} |
        | Latest value | {cpiDf['value'].iloc[-1]:.2f} |
        | Historical min | {cpiDf['value'].min():.2f} |
        | Historical max | {cpiDf['value'].max():.2f} |
        """
    )
    return


@app.cell
def forecastCpi(forecast, cpiDf):
    cpiResult = forecast(cpiDf, date="date", value="value", steps=12)
    return cpiResult,


@app.cell
def showCpiSummary(mo, cpiResult):
    mo.md(
        f"""
        ### CPI Forecast Results

        | Item | Value |
        |------|-------|
        | Selected Model | `{cpiResult.model}` |
        | Forecast Horizon | {len(cpiResult.predictions)} months |
        | Predicted Mean | {cpiResult.predictions.mean():.2f} |
        | Predicted Range | {cpiResult.predictions.min():.2f} ~ {cpiResult.predictions.max():.2f} |
        | 95% CI Lower | {cpiResult.lower.min():.2f} |
        | 95% CI Upper | {cpiResult.upper.max():.2f} |
        """
    )
    return


@app.cell
def showCpiFullSummary(mo, cpiResult):
    mo.md(f"```\n{cpiResult.summary()}\n```")
    return


@app.cell
def plotCpiForecast(plt, cpiDf, cpiResult):
    fig3, ax3 = plt.subplots(figsize=(12, 5))

    recentN = 60
    recentCpi = cpiDf.tail(recentN)
    ax3.plot(recentCpi["date"], recentCpi["value"], color="#9C27B0", linewidth=1.5, label="Historical")

    forecastDates = pd.to_datetime(cpiResult.dates)
    ax3.plot(forecastDates, cpiResult.predictions, color="#FF9800", linewidth=2, label="Forecast")
    ax3.fill_between(
        forecastDates,
        cpiResult.lower,
        cpiResult.upper,
        alpha=0.2,
        color="#FF9800",
        label="95% CI",
    )

    ax3.set_title("Korean CPI — 12-Month Forecast", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("CPI Index")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    return fig3,


@app.cell
def displayCpiPlot(mo, fig3):
    mo.center(mo.as_html(fig3))
    return


@app.cell
def scenario4Header(mo):
    mo.md(
        """
        ---
        ## Scenario 4: Multi-Indicator DNA Comparison

        Vectrix's **DNA Analysis** extracts a unique fingerprint from each time series,
        revealing its inherent characteristics: difficulty, category, recommended models, and more.

        We compare four key Korean economic indicators side by side:

        | Indicator | FRED Code | Description |
        |-----------|-----------|-------------|
        | Exchange Rate | EXKOUS | KRW per USD |
        | Long-term Interest Rate | IRLTLT01KRM156N | 10-year government bond yield |
        | Unemployment Rate | LRUNTTTTKRM156S | Harmonized unemployment rate |
        | Stock Index | SPASTT01KRM661N | KOSPI share prices |
        """
    )
    return


@app.cell
def loadAllIndicators(pd):
    indicatorConfig = {
        "Exchange Rate (KRW/USD)": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS",
        "Long-term Interest Rate": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01KRM156N",
        "Unemployment Rate": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=LRUNTTTTKRM156S",
        "Stock Index (KOSPI)": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SPASTT01KRM661N",
    }

    indicatorData = {}
    for label, url in indicatorConfig.items():
        rawDf = pd.read_csv(url, parse_dates=["DATE"])
        rawDf.columns = ["date", "value"]
        rawDf = rawDf.dropna()
        indicatorData[label] = rawDf

    return indicatorData,


@app.cell
def analyzeAllIndicators(analyze, indicatorData):
    dnaResults = {}
    for label, df in indicatorData.items():
        dnaResults[label] = analyze(df, date="date", value="value")
    return dnaResults,


@app.cell
def showDnaComparison(mo, dnaResults, indicatorData):
    rows = []
    for label in indicatorData.keys():
        result = dnaResults[label]
        dna = result.dna
        nObs = len(indicatorData[label])
        recommendedStr = ", ".join(dna.recommendedModels[:3]) if dna.recommendedModels else "N/A"
        rows.append(
            f"| {label} | {nObs} | {dna.category} | {dna.difficulty} | "
            f"{dna.difficultyScore:.1f} | `{dna.fingerprint}` | {recommendedStr} |"
        )

    tableBody = "\n        ".join(rows)

    mo.md(
        f"""
        ### DNA Comparison Table

        | Indicator | Obs | Category | Difficulty | Score | Fingerprint | Top 3 Models |
        |-----------|-----|----------|------------|-------|-------------|--------------|
        {tableBody}

        **Key observations:**
        - Each time series has a unique DNA fingerprint based on 65+ statistical features
        - The difficulty score (0-100) predicts how hard the series is to forecast
        - Vectrix automatically recommends the best models based on these DNA characteristics
        """
    )
    return


@app.cell
def showIndividualDna(mo, dnaResults):
    summaries = []
    for label, result in dnaResults.items():
        summaries.append(f"### {label}\n```\n{result.summary()}\n```")

    fullText = "\n\n".join(summaries)
    mo.md(fullText)
    return


@app.cell
def plotDnaRadar(plt, np, dnaResults, indicatorData):
    fig4, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    labels = list(indicatorData.keys())

    for idx, label in enumerate(labels):
        ax = axes[idx]
        result = dnaResults[label]
        c = result.characteristics

        metricNames = ["Length", "Period", "Trend\nStrength", "Seasonal\nStrength"]
        metricValues = [
            min(c.length / 100, 5),
            min(c.period / 12, 3) if c.period else 0,
            c.trendStrength if c.trendStrength else 0,
            c.seasonalStrength if c.seasonalStrength else 0,
        ]

        xPos = np.arange(len(metricNames))
        ax.bar(xPos, metricValues, color=colors[idx], alpha=0.7, width=0.6)
        ax.set_xticks(xPos)
        ax.set_xticklabels(metricNames, fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(metricValues) * 1.3 if max(metricValues) > 0 else 1)
        ax.grid(True, alpha=0.3, axis="y")

    fig4.suptitle("Data Characteristics Comparison", fontsize=14, fontweight="bold", y=1.02)
    fig4.tight_layout()
    return fig4,


@app.cell
def displayRadarPlot(mo, fig4):
    mo.center(mo.as_html(fig4))
    return


@app.cell
def showAnomalyComparison(mo, dnaResults, indicatorData):
    anomalyRows = []
    for label in indicatorData.keys():
        result = dnaResults[label]
        nChangepoints = len(result.changepoints) if result.changepoints is not None else 0
        nAnomalies = len(result.anomalies) if result.anomalies is not None else 0
        anomalyRows.append(f"| {label} | {nChangepoints} | {nAnomalies} |")

    anomalyTable = "\n        ".join(anomalyRows)

    mo.md(
        f"""
        ### Structural Analysis

        | Indicator | Changepoints | Anomalies |
        |-----------|-------------|-----------|
        {anomalyTable}

        Changepoints indicate structural breaks in the time series (e.g., policy shifts,
        economic crises). Anomalies are individual data points that deviate significantly
        from expected patterns.
        """
    )
    return


@app.cell
def footer(mo):
    mo.md(
        """
        ---
        ## Summary

        In this showcase, we demonstrated Vectrix's capabilities on four real Korean
        economic indicators:

        - **Zero-config forecasting**: A single `forecast()` call automatically selects
          the best model from 30+ candidates
        - **DNA analysis**: The `analyze()` function extracts unique characteristics from
          each time series, including difficulty, category, and optimal model recommendations
        - **Confidence intervals**: All forecasts include 95% prediction intervals
        - **Structural detection**: Changepoints and anomalies are automatically identified

        All data was loaded directly from FRED with no preprocessing required.

        ---

        > **Disclaimer:** This analysis is for educational purposes only.
        > The forecasts shown are statistical projections based on historical patterns.
        > Do not use for actual investment decisions.

        **Vectrix** — Zero-config time series forecasting for Python

        ```
        pip install vectrix
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
