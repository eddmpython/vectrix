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
    from vectrix import forecast, analyze
    from vectrix.business import AnomalyDetector, WhatIfAnalyzer, Backtester, BusinessMetrics
    from vectrix.engine.ets import AutoETS

    return (
        AnomalyDetector,
        AutoETS,
        Backtester,
        BusinessMetrics,
        WhatIfAnalyzer,
        forecast,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md("""
    # 비즈니스 인텔리전스 쇼케이스

    End-to-end 비즈니스 예측 워크플로우: **이상치 탐지**, **What-If 시나리오**,
    **백테스팅**, **비즈니스 지표** — 모두 Vectrix로 수행합니다.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. 비즈니스 데이터 준비
    """)
    return


@app.cell
def _(np, pd):
    np.random.seed(42)
    _n = 150
    _t = np.arange(_n, dtype=np.float64)
    _trend = 200 + 1.2 * _t
    _seasonal = 30 * np.sin(2 * np.pi * _t / 12)
    _noise = np.random.normal(0, 10, _n)
    _values = _trend + _seasonal + _noise

    _values[45] = 600
    _values[90] = 50
    _values[130] = 700

    bizDf = pd.DataFrame({
        "date": pd.date_range("2013-01-01", periods=_n, freq="MS"),
        "revenue": _values,
    })
    bizDf
    return (bizDf,)


@app.cell
def _(mo):
    mo.md("""
    ## 2. 이상치 탐지

    예측 전에 비정상적인 데이터 포인트를 감지합니다.
    이상치는 모델 선택과 예측 정확도를 왜곡할 수 있습니다.
    """)
    return


@app.cell
def _(AnomalyDetector, bizDf, mo, np):
    detector = AnomalyDetector()
    _y = np.array(bizDf["revenue"], dtype=np.float64)
    anomResult = detector.detect(_y, sensitivity=0.95)

    _rows = []
    for _idx in anomResult.indices:
        _rows.append(
            f"| {_idx} | {bizDf['date'].iloc[_idx].strftime('%Y-%m')} "
            f"| {_y[_idx]:,.1f} | {anomResult.scores[_idx]:+.2f} |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
    **{len(anomResult.indices)}개 이상치 감지**

    | 인덱스 | 날짜 | 값 | Z-Score |
    |--------|------|-----|---------|
    {_table}
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. 예측

    전체 예측 파이프라인을 실행합니다 (이상치 포함 — Vectrix가 내부적으로 처리).
    """)
    return


@app.cell
def _(bizDf, forecast):
    fcResult = forecast(bizDf, date="date", value="revenue", steps=12)
    return (fcResult,)


@app.cell
def _(fcResult, mo):
    mo.md(f"""
    ### 예측 결과

    | 항목 | 값 |
    |------|-----|
    | 모델 | `{fcResult.model}` |
    | MAPE | {fcResult.mape:.2f}% |
    | 12개월 평균 | {fcResult.predictions.mean():,.1f} |
    | 95% CI 폭 | {(fcResult.upper - fcResult.lower).mean():,.1f} |
    """)
    return


@app.cell
def _(fcResult):
    fcResult.to_dataframe()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. What-If 시나리오 분석

    다양한 비즈니스 상황이 예측에 어떤 영향을 미치는지 탐색합니다.
    """)
    return


@app.cell
def _(WhatIfAnalyzer, bizDf, fcResult, mo, np):
    analyzer = WhatIfAnalyzer()
    _scenarios = [
        {"name": "base", "trend_change": 0},
        {"name": "growth_10pct", "trend_change": 0.10},
        {"name": "recession", "trend_change": -0.15, "level_shift": -0.05},
        {"name": "supply_shock", "shock_at": 3, "shock_magnitude": -0.25, "shock_duration": 3},
        {"name": "expansion", "level_shift": 0.10, "seasonal_multiplier": 1.3},
    ]
    _historical = np.array(bizDf["revenue"], dtype=np.float64)
    scenarioResults = analyzer.analyze(fcResult.predictions, _historical, _scenarios, period=12)

    _rows = []
    for _sr in scenarioResults:
        _rows.append(
            f"| {_sr.name} | {_sr.impact:.1f}% | {_sr.percentChange[-1]:+.1f}% |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
    | 시나리오 | 평균 영향 | 최종 변화 |
    |----------|----------|----------|
    {_table}
    """
    )
    return (scenarioResults,)


@app.cell
def _(WhatIfAnalyzer, mo, scenarioResults):
    mo.md(f"""
    ```\n{WhatIfAnalyzer().compareSummary(scenarioResults)}\n```
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. 백테스팅

    Walk-forward 검증: 모델이 과거에 얼마나 정확했을까?
    """)
    return


@app.cell
def _(AutoETS, Backtester, bizDf, mo, np):
    bt = Backtester(nFolds=4, horizon=12, strategy="expanding", minTrainSize=60)
    _y = np.array(bizDf["revenue"], dtype=np.float64)
    btResult = bt.run(_y, modelFactory=AutoETS)
    mo.md(f"```\n{bt.summary(btResult)}\n```")
    return (btResult,)


@app.cell
def _(btResult, mo):
    _rows = []
    for _fold in btResult.folds:
        _rows.append(
            f"| {_fold.fold} | {_fold.trainSize} | {_fold.testSize} | {_fold.mape:.2f}% |"
        )
    _table = "\n".join(_rows)

    mo.md(
        f"""
    ### Fold 상세

    | Fold | 학습 | 테스트 | MAPE |
    |------|------|--------|------|
    {_table}
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. 비즈니스 지표

    실제값 vs 예측값 비교를 위한 표준 비즈니스 정확도 지표를 계산합니다.
    """)
    return


@app.cell
def _(BusinessMetrics, mo, np):
    _actuals = np.array([320, 340, 310, 360, 345, 370, 355, 380, 365, 390, 375, 400])
    _predicted = np.array([325, 335, 315, 355, 350, 365, 360, 375, 370, 385, 380, 395])

    metrics = BusinessMetrics()
    metricResult = metrics.calculate(_actuals, _predicted)

    _rows = []
    for _k, _v in metricResult.items():
        _rows.append(f"| {_k} | {_v:.4f} |")
    _table = "\n".join(_rows)

    mo.md(
        f"""
    | 지표 | 값 |
    |------|-----|
    {_table}
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    > **팁**: 이상치 탐지 + 백테스팅 + What-If 분석을 결합하면 견고한 비즈니스 계획을 세울 수 있습니다.
    > 이상치를 먼저 감지하고, 모델 정확도를 검증한 후, 시나리오를 탐색하세요.
    """)
    return


if __name__ == "__main__":
    app.run()
