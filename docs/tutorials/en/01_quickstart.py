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
        # Vectrix Quickstart

        **Make your first forecast in 3 minutes.**

        Vectrix is a zero-config time series forecasting library.
        It automatically compares 30+ models and selects the best one.

        ```
        pip install vectrix
        ```
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import forecast
    return np, pd, forecast


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. Forecast from a List

        All you need is a list of numbers.
        Dates, column names, model selection — all automatic.
        """
    )
    return


@app.cell
def forecastFromList(forecast):
    salesData = [
        120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
        172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
        245, 268, 258, 280, 270, 295, 285, 310, 300, 325,
        180, 200, 215, 195, 220, 235, 210, 245, 230, 260,
    ]

    result = forecast(salesData, steps=10)
    return result,


@app.cell
def showResult(mo, result):
    mo.md(
        f"""
        ### Results

        | Item | Value |
        |------|-------|
        | Selected Model | `{result.model}` |
        | Forecast Horizon | {len(result.predictions)} periods |
        | Prediction Range | {result.predictions.min():.1f} ~ {result.predictions.max():.1f} |
        """
    )
    return


@app.cell
def showDataframe(mo, result):
    df = result.to_dataframe()
    mo.md("### Forecast Table")
    return df,


@app.cell
def displayTable(mo, df):
    return mo.ui.table(df)


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. Forecast from a DataFrame

        With a pandas DataFrame, Vectrix auto-detects date and value columns.
        """
    )
    return


@app.cell
def forecastFromDf(np, pd, forecast):
    np.random.seed(42)
    n = 120
    t = np.arange(n, dtype=np.float64)
    trend = 100 + 0.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)

    monthlyDf = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=n, freq="MS"),
        "sales": trend + seasonal + noise,
    })

    dfResult = forecast(monthlyDf, steps=12)
    return monthlyDf, dfResult


@app.cell
def showDfResult(mo, dfResult):
    mo.md(
        f"""
        ### DataFrame Forecast Result

        - Model: `{dfResult.model}`
        - 12-month forecast generated

        Use `.summary()` for the full report:
        """
    )
    return


@app.cell
def printSummary(mo, dfResult):
    mo.md(f"```\n{dfResult.summary()}\n```")
    return


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. Adjust Forecast Horizon

        Drag the slider to change the forecast horizon (steps).
        """
    )
    return


@app.cell
def stepsSlider(mo):
    stepsControl = mo.ui.slider(
        start=5, stop=60, step=5, value=15,
        label="Forecast Horizon (steps)"
    )
    return stepsControl,


@app.cell
def showSlider(stepsControl):
    stepsControl
    return


@app.cell
def interactiveForecast(np, pd, forecast, stepsControl):
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=np.float64)
    values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

    interDf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "value": values,
    })

    interResult = forecast(interDf, steps=stepsControl.value)
    return interDf, interResult


@app.cell
def showInteractive(mo, interResult, stepsControl):
    predDf = interResult.to_dataframe()
    mo.md(
        f"""
        **{stepsControl.value}-day forecast** | Model: `{interResult.model}`

        Mean prediction: {interResult.predictions.mean():.1f}
        """
    )
    return predDf,


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. Working with Results

        `EasyForecastResult` methods:
        """
    )
    return


@app.cell
def resultMethods(mo, interResult):
    forecastDf = interResult.to_dataframe()
    jsonStr = interResult.to_json()

    mo.md(
        f"""
        | Method | Description | Example |
        |--------|-------------|---------|
        | `.predictions` | Forecast array | `[{interResult.predictions[0]:.1f}, {interResult.predictions[1]:.1f}, ...]` |
        | `.dates` | Forecast dates | `[{interResult.dates[0]}, ...]` |
        | `.lower` | 95% lower bound | `[{interResult.lower[0]:.1f}, ...]` |
        | `.upper` | 95% upper bound | `[{interResult.upper[0]:.1f}, ...]` |
        | `.model` | Selected model | `{interResult.model}` |
        | `.to_dataframe()` | Convert to DataFrame | {len(forecastDf)} rows |
        | `.to_json()` | Convert to JSON | {len(jsonStr)} chars |
        | `.summary()` | Text summary | See above |
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. Supported Input Formats

        Vectrix accepts almost any data format:

        ```python
        forecast([1, 2, 3, 4, 5])                    # list
        forecast(np.array([1, 2, 3, 4, 5]))           # numpy array
        forecast(pd.Series([1, 2, 3, 4, 5]))          # pandas Series
        forecast({"value": [1, 2, 3, 4, 5]})          # dict
        forecast(df, date="date", value="sales")       # DataFrame
        forecast("data.csv")                           # CSV file path
        ```

        **Next tutorial:** `02_analyze.py` — Time Series DNA Analysis
        """
    )
    return


if __name__ == "__main__":
    app.run()
