# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
#     "numpy",
#     "pandas",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # R-style Regression

        **Run regression with a single formula, just like R.**

        ```python
        from vectrix import regress
        result = regress(data=df, formula="sales ~ ads + price")
        ```

        Supports OLS, Ridge, Lasso, Huber, and Quantile regression
        with VIF, heteroscedasticity, normality, and autocorrelation diagnostics.
        """
    )
    return


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from vectrix import regress
    return mo, np, pd, regress


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. Basic Regression

        Let's build a sales prediction model with marketing data.
        """
    )
    return


@app.cell
def createData(np, pd):
    np.random.seed(42)
    _n = 200

    ads = np.random.uniform(10, 100, _n)
    price = np.random.uniform(5, 50, _n)
    promo = np.random.binomial(1, 0.3, _n).astype(float)
    sales = 50 + 2.5 * ads - 1.8 * price + 30 * promo + np.random.normal(0, 15, _n)

    marketDf = pd.DataFrame({
        "sales": sales,
        "ads": ads,
        "price": price,
        "promo": promo,
    })
    return marketDf,


@app.cell
def showData(mo, marketDf):
    mo.md("### Data Preview")
    return


@app.cell
def displayData(mo, marketDf):
    return mo.ui.table(marketDf.head(10))


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. Formula-based Regression

        R-style formula: `"sales ~ ads + price + promo"`
        """
    )
    return


@app.cell
def runRegression(regress, marketDf):
    result = regress(data=marketDf, formula="sales ~ ads + price + promo", summary=False)
    return result,


@app.cell
def showSummary(mo, result):
    mo.md(f"```\n{result.summary()}\n```")
    return


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. Choose Regression Method

        Select a method from the dropdown.
        """
    )
    return


@app.cell
def methodSelector(mo):
    methodChoice = mo.ui.dropdown(
        options={
            "OLS (Ordinary Least Squares)": "ols",
            "Ridge (L2 Regularization)": "ridge",
            "Lasso (L1 Regularization)": "lasso",
            "Huber (Robust)": "huber",
            "Quantile Regression": "quantile",
        },
        value="ols",
        label="Regression Method"
    )
    return methodChoice,


@app.cell
def showMethodSelector(methodChoice):
    methodChoice
    return


@app.cell
def runMethodRegression(regress, marketDf, methodChoice):
    methodResult = regress(
        data=marketDf,
        formula="sales ~ ads + price + promo",
        method=methodChoice.value,
        summary=False
    )
    return methodResult,


@app.cell
def showMethodResult(mo, methodResult, methodChoice):
    mo.md(
        f"""
        ### {methodChoice.value.upper()} Results

        | Metric | Value |
        |--------|-------|
        | R² | {methodResult.r_squared:.4f} |
        | Adjusted R² | {methodResult.adj_r_squared:.4f} |
        | F-statistic | {methodResult.f_stat:.2f} |

        **Coefficients:**
        ```
        {methodResult.summary()}
        ```
        """
    )
    return


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. Regression Diagnostics

        `.diagnose()` validates statistical assumptions:

        - **VIF**: Multicollinearity (>10 is problematic)
        - **Breusch-Pagan**: Heteroscedasticity test
        - **Jarque-Bera**: Residual normality test
        - **Durbin-Watson**: Autocorrelation test
        """
    )
    return


@app.cell
def runDiagnose(mo, result):
    diagStr = result.diagnose()
    mo.md(f"```\n{diagStr}\n```")
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. Prediction

        Predict on new data with confidence intervals:
        """
    )
    return


@app.cell
def runPredict(mo, result, np, pd):
    newData = pd.DataFrame({
        "ads": [50, 75, 90],
        "price": [20, 15, 10],
        "promo": [0, 1, 1],
    })

    predictions = result.predict(newData)
    mo.md("### Predictions")
    return predictions,


@app.cell
def showPredictions(mo, predictions):
    return mo.ui.table(predictions)


@app.cell
def section6(mo):
    mo.md(
        """
        ---
        ## 6. More Features

        ```python
        # Use all variables
        regress(data=df, formula="y ~ .")

        # Interaction terms
        regress(data=df, formula="y ~ x1 * x2")

        # Polynomial terms
        regress(data=df, formula="y ~ x + I(x**2)")

        # Direct array input
        regress(y=y_array, X=X_array)
        ```

        **Next tutorial:** `04_models.py` — 30+ Model Comparison
        """
    )
    return


if __name__ == "__main__":
    app.run()
