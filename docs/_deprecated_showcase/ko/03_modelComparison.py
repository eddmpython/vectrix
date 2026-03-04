# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vectrix",
#     "pandas",
#     "numpy",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from vectrix import forecast, analyze, compare

    return analyze, compare, forecast, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        """
# 모델 비교 & 적응형 지능

**30+ 예측 모델**을 실제 데이터에서 나란히 비교합니다.
Vectrix는 자동으로 최적 모델을 선택하지만, 모든 후보를 직접 살펴볼 수 있습니다.
"""
    )


@app.cell
def _(mo):
    mo.md(
        """
## 1. 현실적인 데이터 생성

추세, 계절성, 노이즈가 있는 월간 매출 데이터 — 비즈니스 예측에서 흔한 패턴입니다.
"""
    )


@app.cell
def _(np, pd):
    np.random.seed(42)
    _n = 120
    _t = np.arange(_n, dtype=np.float64)
    _trend = 100 + 0.8 * _t
    _seasonal = 25 * np.sin(2 * np.pi * _t / 12) + 10 * np.cos(2 * np.pi * _t / 6)
    _noise = np.random.normal(0, 8, _n)

    salesDf = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=_n, freq="MS"),
        "revenue": _trend + _seasonal + _noise,
    })
    salesDf
    return (salesDf,)


@app.cell
def _(mo):
    mo.md(
        """
## 2. DNA 분석

예측 전에 데이터의 특성을 파악합니다.
DNA 프로파일러가 고유 핑거프린트를 추출합니다: 난이도, 카테고리, 추천 모델.
"""
    )


@app.cell
def _(analyze, mo, salesDf):
    report = analyze(salesDf, date="date", value="revenue")
    mo.md(
        f"""
| 속성 | 값 |
|------|-----|
| 카테고리 | {report.dna.category} |
| 난이도 | {report.dna.difficulty} ({report.dna.difficultyScore:.0f}/100) |
| 핑거프린트 | `{report.dna.fingerprint}` |
| 추세 | {report.trend} |
| 계절성 | 주기 = {report.seasonalPeriod} |
| 변화점 | {len(report.changepoints)}개 감지 |
| 추천 모델 | {', '.join(report.dna.recommendedModels[:5])} |
"""
    )
    return (report,)


@app.cell
def _(mo):
    mo.md(
        """
## 3. 예측 & 전체 모델 비교

`forecast()`는 30+ 모델을 실행하고 최적의 모델을 선택합니다.
`.compare()`로 모든 모델의 성능을 확인하세요.
"""
    )


@app.cell
def _(forecast, salesDf):
    result = forecast(salesDf, date="date", value="revenue", steps=12)
    return (result,)


@app.cell
def _(mo, result):
    mo.md(
        f"""
### 최적 모델: `{result.model}`

| 지표 | 값 |
|------|-----|
| MAPE | {result.mape:.2f}% |
| RMSE | {result.rmse:.2f} |
| MAE | {result.mae:.2f} |
| sMAPE | {result.smape:.2f}% |
"""
    )


@app.cell
def _(mo):
    mo.md("### 전체 모델 순위")


@app.cell
def _(result):
    comparisonDf = result.compare()
    comparisonDf


@app.cell
def _(mo):
    mo.md(
        """
## 4. 전체 모델 예측값

모든 모델의 미래 예측값을 하나의 DataFrame으로 확인합니다.
커스텀 앙상블을 구축하거나 모델 간 의견 차이를 분석하는 데 유용합니다.
"""
    )


@app.cell
def _(result):
    allForecasts = result.all_forecasts()
    allForecasts


@app.cell
def _(mo, result):
    mo.md(
        f"""
## 5. 예측 요약

```
{result.summary()}
```
"""
    )


@app.cell
def _(result):
    result.to_dataframe()


@app.cell
def _(mo):
    mo.md(
        """
## 6. 한 줄 비교

최상위 `compare()` 함수로 빠르게 모든 모델을 비교할 수 있습니다.
"""
    )


@app.cell
def _(compare, salesDf):
    quickCompare = compare(salesDf, date="date", value="revenue", steps=12)
    quickCompare


@app.cell
def _(mo):
    mo.md(
        """
> **참고**: 교차 검증 분할에 따라 모델 순위가 달라질 수 있습니다.
> 최적 모델은 in-sample 적합이 아닌 out-of-sample 정확도 기준으로 선택됩니다.
"""
    )


if __name__ == "__main__":
    app.run()
