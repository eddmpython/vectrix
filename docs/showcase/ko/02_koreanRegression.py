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
    mo.md("""
    # 서울 자전거 대여량 회귀분석

    UCI의 **Seoul Bike Sharing Demand** 데이터로
    기상 조건이 자전거 대여량에 미치는 영향을 분석합니다.
    """)
    return


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
    return (bikeDf,)


@app.cell
def _(bikeDf, mo):
    mo.md(f"""
    | 항목 | 값 |
    |------|-----|
    | 관측 수 | {len(bikeDf):,}개 (시간별) |
    | 기간 | {bikeDf['Date'].iloc[0]} ~ {bikeDf['Date'].iloc[-1]} |
    """)
    return


@app.cell
def _(bikeDf, regress):
    result = regress(
        data=bikeDf,
        formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + hour + solarRadiation",
        summary=False,
    )
    result
    return (result,)


@app.cell
def _(mo, result):
    mo.md(f"""
    ## 결과 요약

    | 지표 | 값 |
    |------|-----|
    | R-squared | {result.r_squared:.4f} |
    | Adj R-squared | {result.adj_r_squared:.4f} |
    | F-statistic | {result.f_stat:.2f} |

    ```
    {result.summary()}
    ```
    """)
    return


@app.cell
def _(mo, pd, result):
    _labels = ["절편", "기온", "습도", "풍속", "강수량", "시간대", "태양복사량"]
    coefDf = pd.DataFrame(
        {
            "변수": _labels,
            "계수": result.coefficients,
            "p-value": [f"{p:.4f}" for p in result.pvalues],
        }
    )
    mo.ui.table(coefDf)
    return


@app.cell
def _(mo, result):
    mo.md(f"""
    ## 해석

    - **기온** 1도 상승 → 대여량 {result.coefficients[1]:+.1f}건
    - **습도** 1% 상승 → 대여량 {result.coefficients[2]:+.1f}건
    - **강수량** 1mm 증가 → 대여량 {result.coefficients[4]:+.1f}건

    ```
    {result.diagnose()}
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    > **면책 조항**: 교육 목적이며, 실제 사업 결정에 사용하지 마세요.
    > 데이터 출처: [UCI ML Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)
    """)
    return


if __name__ == "__main__":
    app.run()
