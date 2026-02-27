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
        # 비즈니스 인텔리전스

        **예측을 넘어 — 의사결정을 위한 도구들**

        - **AnomalyDetector**: 이상치 자동 탐지
        - **WhatIfAnalyzer**: "만약 ~라면?" 시나리오 분석
        - **Backtester**: 예측 모델 신뢰도 검증
        - **BusinessMetrics**: MAPE, RMSE, MAE 계산
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix.business import (
        AnomalyDetector,
        WhatIfAnalyzer,
        Backtester,
        BusinessMetrics,
    )
    return np, pd, AnomalyDetector, WhatIfAnalyzer, Backtester, BusinessMetrics


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. 이상치 탐지

        Z-score, IQR, 계절 잔차, 이동 윈도우 등
        여러 방법으로 이상치를 자동 탐지합니다.
        """
    )
    return


@app.cell
def methodSelector(mo):
    anomalyMethod = mo.ui.dropdown(
        options={
            "자동 선택": "auto",
            "Z-score": "zscore",
            "IQR (사분위범위)": "iqr",
            "이동 윈도우": "rolling",
        },
        value="auto",
        label="탐지 방법"
    )
    return anomalyMethod,


@app.cell
def showMethodSelector(anomalyMethod):
    anomalyMethod
    return


@app.cell
def createAnomalyData(np):
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=np.float64)
    normalData = 100 + 0.2 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

    normalData[45] = 250
    normalData[120] = 20
    normalData[175] = 280
    return normalData,


@app.cell
def runAnomaly(AnomalyDetector, normalData, anomalyMethod):
    detector = AnomalyDetector()
    anomalyResult = detector.detect(normalData, method=anomalyMethod.value)
    return anomalyResult,


@app.cell
def showAnomaly(mo, anomalyResult):
    mo.md(
        f"""
        ### 이상치 탐지 결과

        | 항목 | 값 |
        |------|-----|
        | 방법 | `{anomalyResult.method}` |
        | 감지된 이상치 | {anomalyResult.nAnomalies}개 |
        | 이상치 비율 | {anomalyResult.anomalyRatio:.1%} |
        | 임계값 | {anomalyResult.threshold:.2f} |
        | 이상치 위치 | {list(anomalyResult.indices[:10])} |
        """
    )
    return


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. What-If 시나리오 분석

        "만약 추세가 10% 증가하면?", "30일차에 충격이 발생하면?"
        같은 시나리오를 시뮬레이션합니다.
        """
    )
    return


@app.cell
def createScenarioData(np):
    np.random.seed(42)
    basePred = np.linspace(100, 130, 30)
    histData = np.random.normal(100, 10, 100)
    return basePred, histData


@app.cell
def runScenarios(WhatIfAnalyzer, basePred, histData):
    analyzer = WhatIfAnalyzer()
    scenarios = [
        {"name": "낙관적", "trendChange": 0.1},
        {"name": "비관적", "trendChange": -0.15},
        {"name": "충격 발생", "shockAt": 10, "shockMagnitude": -0.3, "shockDuration": 5},
        {"name": "수준 상승", "levelShift": 0.05},
    ]
    scenarioResults = analyzer.analyze(basePred, histData, scenarios)
    return scenarioResults,


@app.cell
def showScenarios(mo, scenarioResults, pd):
    rows = []
    for sr in scenarioResults:
        rows.append({
            "시나리오": sr.name,
            "평균 예측": round(sr.predictions.mean(), 2),
            "기준 대비 변화": f"{sr.impact:+.1%}",
        })
    scenDf = pd.DataFrame(rows)
    mo.md("### 시나리오 비교")
    return scenDf,


@app.cell
def showScenarioTable(mo, scenDf):
    return mo.ui.table(scenDf)


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. 백테스트

        Walk-forward validation으로 모델의 실전 성능을 검증합니다.
        확장 윈도우(expanding) 또는 슬라이딩 윈도우 전략을 지원합니다.
        """
    )
    return


@app.cell
def runBacktest(Backtester, np):
    np.random.seed(42)
    n = 300
    t = np.arange(n, dtype=np.float64)
    btData = 100 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

    def naiveModel(train, horizon):
        return np.full(horizon, train[-1])

    bt = Backtester(nFolds=5, horizon=14, strategy='expanding')
    btResult = bt.run(btData, naiveModel)
    return btResult,


@app.cell
def showBacktest(mo, btResult, pd):
    mo.md(
        f"""
        ### 백테스트 결과 ({btResult.nFolds} folds)

        | 지표 | 평균 | 표준편차 |
        |------|------|----------|
        | MAPE | {btResult.avgMAPE:.2f}% | {btResult.mapeStd:.2f}% |
        | RMSE | {btResult.avgRMSE:.2f} | - |
        | MAE | {btResult.avgMAE:.2f} | - |
        | sMAPE | {btResult.avgSMAPE:.2f}% | - |
        | Bias | {btResult.avgBias:+.2f} | - |
        | 최고 Fold | #{btResult.bestFold} |
        | 최악 Fold | #{btResult.worstFold} |
        """
    )
    return


@app.cell
def showFoldDetails(mo, btResult, pd):
    foldRows = []
    for f in btResult.folds:
        foldRows.append({
            "Fold": f.fold,
            "학습 크기": f.trainSize,
            "테스트 크기": f.testSize,
            "MAPE": round(f.mape, 2),
            "RMSE": round(f.rmse, 2),
        })
    foldDf = pd.DataFrame(foldRows)
    mo.md("### Fold별 상세")
    return foldDf,


@app.cell
def showFoldTable(mo, foldDf):
    return mo.ui.table(foldDf)


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. 비즈니스 지표

        실측과 예측을 비교하는 다양한 지표를 계산합니다.
        """
    )
    return


@app.cell
def runMetrics(BusinessMetrics, np):
    np.random.seed(42)
    actual = np.array([100, 120, 130, 115, 140, 160, 150, 170])
    predicted = np.array([105, 118, 135, 110, 145, 155, 148, 175])

    metrics = BusinessMetrics()
    metricsResult = metrics.calculate(actual, predicted)
    return actual, predicted, metricsResult


@app.cell
def showMetrics(mo, metricsResult):
    mo.md(
        f"""
        ### 지표 계산 결과

        | 지표 | 값 | 해석 |
        |------|-----|------|
        | **Bias** | {metricsResult.get('bias', 0):+.2f} | 양수=과대예측, 음수=과소예측 |
        | **Bias %** | {metricsResult.get('biasPercent', 0):+.2f}% | 백분율 편향 |
        | **WAPE** | {metricsResult.get('wape', 0):.2f}% | 가중 절대 백분율 오차 |
        | **MASE** | {metricsResult.get('mase', 0):.2f} | 1 미만이면 Naive보다 우수 |
        | **예측 정확도** | {metricsResult.get('forecastAccuracy', 0):.1f}% | 높을수록 좋음 |

        ---

        이 튜토리얼 시리즈를 완료했습니다.

        더 자세한 내용은 [GitHub](https://github.com/eddmpython/vectrix)을 참조하세요.
        """
    )
    return


if __name__ == "__main__":
    app.run()
