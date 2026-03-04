# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vectrix",
#     "pandas",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from vectrix import forecast

    return forecast, mo, pd


@app.cell
def _(mo):
    mo.md(
        """
# KRW/USD Exchange Rate Forecast

Fetching **Korean Won / US Dollar exchange rate (EXKOUS)** monthly data
from FRED and forecasting the next 12 months with Vectrix.
"""
    )


@app.cell
def _(pd):
    _url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
    df = pd.read_csv(_url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    df


@app.cell
def _(df, mo):
    mo.md(
        f"""
| Item | Value |
|------|-------|
| Period | {df['date'].iloc[0].strftime('%Y-%m')} ~ {df['date'].iloc[-1].strftime('%Y-%m')} |
| Observations | {len(df):,} |
| Latest rate | {df['value'].iloc[-1]:,.1f} KRW/USD |
"""
    )


@app.cell
def _(df, forecast):
    result = forecast(df, date="date", value="value", steps=12)
    result


@app.cell
def _(mo, result):
    mo.md(
        f"""
## Forecast Result

| Item | Value |
|------|-------|
| Selected model | `{result.model}` |
| Mean prediction | {result.predictions.mean():,.1f} KRW |
| 95% CI | {result.lower.min():,.1f} ~ {result.upper.max():,.1f} KRW |
"""
    )


@app.cell
def _(mo, result):
    mo.md(f"```\n{result.summary()}\n```")


@app.cell
def _(result):
    result.to_dataframe()


@app.cell
def _(mo):
    mo.md(
        """
> **Disclaimer**: For educational purposes only. Do not use for investment decisions.
"""
    )


if __name__ == "__main__":
    app.run()
