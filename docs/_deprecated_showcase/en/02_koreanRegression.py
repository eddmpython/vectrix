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
    from vectrix import regress

    return mo, pd, regress


@app.cell
def _(mo):
    mo.md(
        """
# Seoul Bike Sharing Regression

Analyzing how weather conditions affect bike rental demand
using UCI's **Seoul Bike Sharing Demand** dataset.
"""
    )


@app.cell
def _(pd):
    _url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
    bikeDf = pd.read_csv(_url, encoding="unicode_escape")

    _colMap = {
        "Rented Bike Count": "rentedBikeCount",
        "Hour": "hour",
        "Humidity(%)": "humidity",
        "Wind speed (m/s)": "windSpeed",
        "Rainfall(mm)": "rainfall",
        "Solar Radiation (MJ/m2)": "solarRadiation",
    }

    _tempCols = [c for c in bikeDf.columns if "Temperature" in c and "Dew" not in c]
    if _tempCols:
        _colMap[_tempCols[0]] = "temperature"

    bikeDf = bikeDf.rename(columns=_colMap)
    bikeDf


@app.cell
def _(bikeDf, mo):
    mo.md(
        f"""
| Item | Value |
|------|-------|
| Observations | {len(bikeDf):,} (hourly) |
| Period | {bikeDf['Date'].iloc[0]} ~ {bikeDf['Date'].iloc[-1]} |
"""
    )


@app.cell
def _(bikeDf, regress):
    result = regress(
        data=bikeDf,
        formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + hour + solarRadiation",
        summary=False,
    )
    result


@app.cell
def _(mo, result):
    mo.md(
        f"""
## Results

| Metric | Value |
|--------|-------|
| R-squared | {result.r_squared:.4f} |
| Adj R-squared | {result.adj_r_squared:.4f} |
| F-statistic | {result.f_stat:.2f} |

```
{result.summary()}
```
"""
    )


@app.cell
def _(mo, pd, result):
    _labels = ["Intercept", "Temperature", "Humidity", "Wind Speed", "Rainfall", "Hour", "Solar Radiation"]
    coefDf = pd.DataFrame(
        {
            "Variable": _labels,
            "Coefficient": result.coefficients,
            "p-value": [f"{p:.4f}" for p in result.pvalues],
        }
    )
    mo.ui.table(coefDf)


@app.cell
def _(mo, result):
    mo.md(
        f"""
## Interpretation

- **Temperature** +1C → {result.coefficients[1]:+.1f} rentals
- **Humidity** +1% → {result.coefficients[2]:+.1f} rentals
- **Rainfall** +1mm → {result.coefficients[4]:+.1f} rentals

```
{result.diagnose()}
```
"""
    )


@app.cell
def _(mo):
    mo.md(
        """
> **Disclaimer**: For educational purposes only.
> Data source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
"""
    )


if __name__ == "__main__":
    app.run()
