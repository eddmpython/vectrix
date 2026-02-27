# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
# ]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # R-style 회귀분석

        **R처럼 수식 한 줄로 회귀분석을 수행합니다.**

        ```python
        from vectrix import regress
        result = regress(data=df, formula="sales ~ ads + price")
        ```

        OLS, Ridge, Lasso, Huber, Quantile 회귀를 지원하고,
        VIF, 이분산성, 정규성, 자기상관 진단까지 한 번에 제공합니다.
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import regress
    return np, pd, regress


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. 기본 회귀분석

        마케팅 데이터로 매출 예측 모델을 만들어봅시다.
        """
    )
    return


@app.cell
def createData(np, pd):
    np.random.seed(42)
    n = 200

    ads = np.random.uniform(10, 100, n)
    price = np.random.uniform(5, 50, n)
    promo = np.random.binomial(1, 0.3, n).astype(float)
    sales = 50 + 2.5 * ads - 1.8 * price + 30 * promo + np.random.normal(0, 15, n)

    marketDf = pd.DataFrame({
        "sales": sales,
        "ads": ads,
        "price": price,
        "promo": promo,
    })
    return marketDf,


@app.cell
def showData(mo, marketDf):
    mo.md("### 데이터 미리보기")
    return


@app.cell
def displayData(mo, marketDf):
    return mo.ui.table(marketDf.head(10))


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. 수식으로 회귀분석

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
        ## 3. 회귀분석 방법 선택

        드롭다운에서 방법을 바꿔보세요.
        """
    )
    return


@app.cell
def methodSelector(mo):
    methodChoice = mo.ui.dropdown(
        options={
            "OLS (최소자승법)": "ols",
            "Ridge (L2 정규화)": "ridge",
            "Lasso (L1 정규화)": "lasso",
            "Huber (로버스트)": "huber",
            "Quantile (분위 회귀)": "quantile",
        },
        value="ols",
        label="회귀 방법"
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
        ### {methodChoice.value.upper()} 결과

        | 지표 | 값 |
        |------|-----|
        | R² | {methodResult.r_squared:.4f} |
        | Adjusted R² | {methodResult.adj_r_squared:.4f} |
        | F-statistic | {methodResult.f_stat:.2f} |

        **계수:**
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
        ## 4. 회귀 진단

        `.diagnose()`로 모델의 통계적 가정을 검증합니다.

        - **VIF**: 다중공선성 (10 이상이면 문제)
        - **Breusch-Pagan**: 이분산성 검정
        - **Jarque-Bera**: 잔차 정규성 검정
        - **Durbin-Watson**: 자기상관 검정
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
        ## 5. 예측

        새 데이터에 대한 예측 + 신뢰구간:
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
    mo.md("### 예측 결과")
    return predictions,


@app.cell
def showPredictions(mo, predictions):
    return mo.ui.table(predictions)


@app.cell
def section6(mo):
    mo.md(
        """
        ---
        ## 6. 편의 기능

        ```python
        # 전체 변수 사용
        regress(data=df, formula="y ~ .")

        # 교호작용
        regress(data=df, formula="y ~ x1 * x2")

        # 다항식
        regress(data=df, formula="y ~ x + I(x**2)")

        # 직접 배열 입력
        regress(y=y_array, X=X_array)
        ```

        **다음 튜토리얼:** `04_models.py` — 30+ 모델 비교
        """
    )
    return


if __name__ == "__main__":
    app.run()
