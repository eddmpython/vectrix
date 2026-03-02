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
        # Korean Real-World Regression Showcase

        **Demonstrating Vectrix regression analysis on authentic Korean datasets.**

        This showcase walks through three scenarios using publicly available data:

        1. **Seoul Bike Sharing** — Demand regression with weather features
        2. **Korean Macro Economics** — Exchange rate determinants from FRED
        3. **From Regression to Forecasting** — Same data, two perspectives

        > *This analysis is for educational purposes only.*
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from vectrix import regress, forecast
    return np, pd, plt, regress, forecast


@app.cell
def scenario1Header(mo):
    mo.md(
        """
        ---
        ## Scenario 1: Seoul Bike Sharing Demand Regression

        The [Seoul Bike Sharing dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
        contains hourly rental counts along with weather conditions, holidays,
        and seasonal information from Seoul, South Korea (2017-2018).

        We'll model **rented bike count** as a function of temperature, humidity,
        wind speed, rainfall, hour of day, and solar radiation.
        """
    )
    return


@app.cell
def loadBikeData(pd):
    bikeUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
    bikeDf = pd.read_csv(bikeUrl, encoding="unicode_escape")

    bikeDf = bikeDf.rename(columns={
        "Rented Bike Count": "rentedBikeCount",
        "Temperature(°C)": "temperature",
        "Humidity(%)": "humidity",
        "Wind speed (m/s)": "windSpeed",
        "Visibility (10m)": "visibility",
        "Dew point temperature(°C)": "dewPoint",
        "Solar Radiation (MJ/m2)": "solarRadiation",
        "Rainfall(mm)": "rainfall",
        "Snowfall (cm)": "snowfall",
        "Seasons": "seasons",
        "Holiday": "holiday",
        "Functioning Day": "functioningDay",
    })
    return bikeDf,


@app.cell
def previewBikeData(mo, bikeDf):
    mo.md(
        f"""
        ### Dataset Overview

        - **Rows**: {len(bikeDf):,}
        - **Columns**: {len(bikeDf.columns)}
        - **Time span**: Hourly observations from Seoul, South Korea
        """
    )
    return


@app.cell
def displayBikeTable(mo, bikeDf):
    return mo.ui.table(bikeDf.head(10))


@app.cell
def bikeRegressionHeader(mo):
    mo.md(
        """
        ### Running the Regression

        We use the R-style formula syntax to model bike demand:

        ```python
        result = regress(
            data=bikeDf,
            formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + Hour + solarRadiation"
        )
        ```
        """
    )
    return


@app.cell
def runBikeRegression(regress, bikeDf):
    bikeResult = regress(
        data=bikeDf,
        formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + Hour + solarRadiation",
        summary=False,
    )
    return bikeResult,


@app.cell
def showBikeMetrics(mo, bikeResult):
    mo.md(
        f"""
        ### Model Fit

        | Metric | Value |
        |--------|-------|
        | R-squared | {bikeResult.r_squared:.4f} |
        | Adjusted R-squared | {bikeResult.adj_r_squared:.4f} |
        | F-statistic | {bikeResult.f_stat:.2f} |

        An R-squared of **{bikeResult.r_squared:.3f}** means that weather and time features
        explain roughly {bikeResult.r_squared * 100:.1f}% of the variation in hourly bike rentals.
        """
    )
    return


@app.cell
def showBikeSummary(mo, bikeResult):
    mo.md(
        f"""
        ### Full Summary

        ```
{bikeResult.summary()}
        ```
        """
    )
    return


@app.cell
def showBikeCoefficients(mo, bikeResult):
    mo.md(
        f"""
        ### Interpreting Coefficients

        The regression coefficients tell us:

        - **Temperature**: Each 1 degree C increase is associated with a change in bike rentals
        - **Humidity**: Higher humidity tends to reduce demand
        - **Rainfall**: Rain significantly decreases cycling activity
        - **Hour**: Later hours capture diurnal patterns in commuting
        - **Solar Radiation**: Sunny conditions encourage cycling

        Use `.diagnose()` for full diagnostic checks (VIF, heteroscedasticity, normality):

        ```
{bikeResult.diagnose()}
        ```
        """
    )
    return


@app.cell
def scenario2Header(mo):
    mo.md(
        """
        ---
        ## Scenario 2: Korean Macro Regression — Exchange Rate Determinants

        What drives the Korean Won / US Dollar exchange rate?
        We pull five macroeconomic indicators from [FRED](https://fred.stlouisfed.org/)
        (Federal Reserve Economic Data):

        | Variable | FRED Series | Description |
        |----------|-------------|-------------|
        | Exchange Rate | EXKOUS | KRW per USD |
        | Bond Yield | IRLTLT01KRM156N | Long-term government bond yield |
        | Unemployment | LRUNTTTTKRM156S | Harmonized unemployment rate |
        | Stock Index | SPASTT01KRM661N | Share price index |
        | CPI | KORCPIALLMINMEI | Consumer price index |
        """
    )
    return


@app.cell
def loadMacroData(pd):
    seriesIds = [
        "EXKOUS",
        "IRLTLT01KRM156N",
        "LRUNTTTTKRM156S",
        "SPASTT01KRM661N",
        "KORCPIALLMINMEI",
    ]

    frames = []
    for sid in seriesIds:
        fredUrl = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        tempDf = pd.read_csv(fredUrl, parse_dates=["DATE"], na_values=".")
        tempDf.columns = ["date", sid]
        tempDf = tempDf.set_index("date")
        frames.append(tempDf)

    merged = pd.concat(frames, axis=1).dropna()
    merged = merged.rename(columns={
        "EXKOUS": "exchangeRate",
        "IRLTLT01KRM156N": "bondYield",
        "LRUNTTTTKRM156S": "unemployment",
        "SPASTT01KRM661N": "stockIndex",
        "KORCPIALLMINMEI": "cpi",
    })
    merged = merged.reset_index()
    return merged,


@app.cell
def previewMacroData(mo, merged):
    mo.md(
        f"""
        ### Merged Macro Dataset

        - **Observations**: {len(merged):,}
        - **Date range**: {merged['date'].min().strftime('%Y-%m')} to {merged['date'].max().strftime('%Y-%m')}
        - **Variables**: exchangeRate, bondYield, unemployment, stockIndex, cpi
        """
    )
    return


@app.cell
def displayMacroTable(mo, merged):
    return mo.ui.table(merged.head(10))


@app.cell
def macroRegressionHeader(mo):
    mo.md(
        """
        ### Exchange Rate Regression

        ```python
        result = regress(
            data=merged,
            formula="exchangeRate ~ bondYield + unemployment + stockIndex + cpi"
        )
        ```
        """
    )
    return


@app.cell
def runMacroRegression(regress, merged):
    macroResult = regress(
        data=merged,
        formula="exchangeRate ~ bondYield + unemployment + stockIndex + cpi",
        summary=False,
    )
    return macroResult,


@app.cell
def showMacroMetrics(mo, macroResult):
    mo.md(
        f"""
        ### Model Fit

        | Metric | Value |
        |--------|-------|
        | R-squared | {macroResult.r_squared:.4f} |
        | Adjusted R-squared | {macroResult.adj_r_squared:.4f} |
        | F-statistic | {macroResult.f_stat:.2f} |
        """
    )
    return


@app.cell
def showMacroSummary(mo, macroResult):
    mo.md(
        f"""
        ### Full Summary

        ```
{macroResult.summary()}
        ```
        """
    )
    return


@app.cell
def showMacroDiagnostics(mo, macroResult):
    mo.md(
        f"""
        ### Diagnostics

        Macro time series regressions often violate OLS assumptions.
        Check for autocorrelation (Durbin-Watson) and multicollinearity (VIF):

        ```
{macroResult.diagnose()}
        ```

        **Note**: Significant autocorrelation is expected with monthly macro data.
        Consider time series regression methods for production use.
        """
    )
    return


@app.cell
def scenario3Header(mo):
    mo.md(
        """
        ---
        ## Scenario 3: From Regression to Time Series Forecasting

        The Seoul Bike dataset also works as a **time series**.
        By aggregating hourly data into daily totals, we can forecast
        future bike demand using `forecast()`.

        This demonstrates how the same dataset can serve both
        **cross-sectional regression** (Scenario 1) and **time series forecasting**.
        """
    )
    return


@app.cell
def aggregateDailyBikes(pd, bikeDf):
    bikeDf["dateStr"] = bikeDf["Date"]
    bikeDf["dateParsed"] = pd.to_datetime(bikeDf["dateStr"], format="%d/%m/%Y")

    dailyBikes = (
        bikeDf
        .groupby("dateParsed")["rentedBikeCount"]
        .sum()
        .reset_index()
    )
    dailyBikes.columns = ["date", "totalRentals"]
    dailyBikes = dailyBikes.sort_values("date").reset_index(drop=True)
    return dailyBikes,


@app.cell
def previewDailyBikes(mo, dailyBikes):
    mo.md(
        f"""
        ### Daily Aggregated Bike Rentals

        - **Days**: {len(dailyBikes)}
        - **Date range**: {dailyBikes['date'].min().strftime('%Y-%m-%d')} to {dailyBikes['date'].max().strftime('%Y-%m-%d')}
        - **Mean daily rentals**: {dailyBikes['totalRentals'].mean():,.0f}
        - **Max daily rentals**: {dailyBikes['totalRentals'].max():,}
        """
    )
    return


@app.cell
def displayDailyTable(mo, dailyBikes):
    return mo.ui.table(dailyBikes.head(10))


@app.cell
def runBikeForecast(forecast, dailyBikes):
    bikeForecastResult = forecast(
        dailyBikes,
        date="date",
        value="totalRentals",
        steps=30,
    )
    return bikeForecastResult,


@app.cell
def showForecastResult(mo, bikeForecastResult):
    mo.md(
        f"""
        ### 30-Day Bike Demand Forecast

        | Item | Value |
        |------|-------|
        | Selected Model | `{bikeForecastResult.model}` |
        | Forecast Horizon | {len(bikeForecastResult.predictions)} days |
        | Predicted Range | {bikeForecastResult.predictions.min():,.0f} ~ {bikeForecastResult.predictions.max():,.0f} |
        | Mean Prediction | {bikeForecastResult.predictions.mean():,.0f} |
        """
    )
    return


@app.cell
def showForecastSummary(mo, bikeForecastResult):
    mo.md(
        f"""
        ### Forecast Summary

        ```
{bikeForecastResult.summary()}
        ```
        """
    )
    return


@app.cell
def showForecastTable(mo, bikeForecastResult):
    forecastDf = bikeForecastResult.to_dataframe()
    mo.md("### Forecast Table")
    return forecastDf,


@app.cell
def displayForecastTable(mo, forecastDf):
    return mo.ui.table(forecastDf)


@app.cell
def plotForecast(plt, dailyBikes, bikeForecastResult):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        dailyBikes["date"].values[-90:],
        dailyBikes["totalRentals"].values[-90:],
        color="#2563EB",
        linewidth=1.5,
        label="Actual (last 90 days)",
    )

    ax.plot(
        bikeForecastResult.dates,
        bikeForecastResult.predictions,
        color="#DC2626",
        linewidth=2,
        label="Forecast (30 days)",
    )

    ax.fill_between(
        bikeForecastResult.dates,
        bikeForecastResult.lower,
        bikeForecastResult.upper,
        color="#DC2626",
        alpha=0.15,
        label="95% Prediction Interval",
    )

    ax.set_title("Seoul Bike Sharing: 30-Day Demand Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Daily Rentals")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig,


@app.cell
def showPlot(mo, fig):
    mo.center(mo.as_html(fig))
    return


@app.cell
def wrapUp(mo):
    mo.md(
        """
        ---
        ## Key Takeaways

        | Scenario | Technique | Key Insight |
        |----------|-----------|-------------|
        | Bike Sharing | Multi-variable regression | Weather and time explain significant demand variation |
        | Macro Economics | Exchange rate regression | Bond yields, CPI, and employment are key FX drivers |
        | Time Series | Daily forecasting | Same data supports both regression and forecasting |

        ### What We Demonstrated

        - **`regress()`** handles real-world messy data with a single formula
        - **`.summary()`** and **`.diagnose()`** provide publication-quality output
        - **`forecast()`** transitions seamlessly from cross-sectional to temporal analysis
        - All with **zero configuration** — just data and a formula

        ---

        *This analysis is for educational purposes only.
        Data sourced from UCI Machine Learning Repository and FRED.*

        **Next showcase**: Explore more at [vectrix on PyPI](https://pypi.org/project/vectrix/)
        """
    )
    return


if __name__ == "__main__":
    app.run()
