# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # 적응형 예측

        **Vectrix만의 차별화 기능 — 기존 라이브러리에는 없습니다.**

        - **RegimeDetector**: HMM 기반 레짐(국면) 전환 감지
        - **ForecastDNA**: 시계열 DNA 핑거프린팅 + 난이도 점수
        - **SelfHealingForecast**: 예측 오차 자동 감지 & 교정
        - **ConstraintAwareForecaster**: 비즈니스 제약 조건 적용
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import (
        RegimeDetector,
        RegimeAwareForecaster,
        ForecastDNA,
        SelfHealingForecast,
        ConstraintAwareForecaster,
        Constraint,
    )
    return (np, pd, RegimeDetector, RegimeAwareForecaster,
            ForecastDNA, SelfHealingForecast,
            ConstraintAwareForecaster, Constraint)


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. 레짐(국면) 감지

        시계열의 "레짐"을 자동으로 감지합니다.
        금융 데이터의 상승/하락/횡보, 수요 데이터의 성수기/비수기 등을 식별합니다.
        """
    )
    return


@app.cell
def regimeSlider(mo):
    nRegimesControl = mo.ui.slider(
        start=2, stop=5, step=1, value=3,
        label="레짐 수"
    )
    return nRegimesControl,


@app.cell
def showRegimeSlider(nRegimesControl):
    nRegimesControl
    return


@app.cell
def createRegimeData(np):
    np.random.seed(42)
    regime1 = np.random.normal(100, 5, 80)
    regime2 = np.random.normal(150, 15, 60)
    regime3 = np.random.normal(80, 3, 60)
    regimeData = np.concatenate([regime1, regime2, regime3])
    return regimeData,


@app.cell
def runRegimeDetection(RegimeDetector, regimeData, nRegimesControl):
    detector = RegimeDetector(nRegimes=nRegimesControl.value)
    regimeResult = detector.detect(regimeData)
    return regimeResult,


@app.cell
def showRegimeResult(mo, regimeResult, nRegimesControl):
    nRegimes = len(regimeResult.regimeStats)
    nTransitions = max(0, len(regimeResult.regimeHistory) - 1)
    mo.md(
        f"""
        ### 레짐 감지 결과 ({nRegimesControl.value}개 레짐)

        | 항목 | 값 |
        |------|-----|
        | 현재 레짐 | {regimeResult.currentRegime} |
        | 감지된 레짐 수 | {nRegimes} |
        | 전환 횟수 | {nTransitions} |

        **레짐별 통계:**
        """
    )
    return


@app.cell
def showRegimeStats(mo, regimeResult, pd):
    rows = []
    for label, stats in regimeResult.regimeStats.items():
        rows.append({
            "레짐": label,
            "평균": round(stats.get('mean', 0), 2),
            "표준편차": round(stats.get('std', 0), 2),
        })
    statsDf = pd.DataFrame(rows)
    return mo.ui.table(statsDf)


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. 레짐 인식 예측

        레짐별로 최적의 모델을 자동 전환하여 예측합니다.
        """
    )
    return


@app.cell
def runRegimeForecast(RegimeAwareForecaster, regimeData):
    raf = RegimeAwareForecaster()
    rafResult = raf.forecast(regimeData, steps=30, period=1)
    return rafResult,


@app.cell
def showRegimeForecast(mo, rafResult):
    mo.md(
        f"""
        ### 레짐 인식 예측 결과

        | 항목 | 값 |
        |------|-----|
        | 현재 레짐 | {rafResult.currentRegime} |
        | 레짐별 모델 | {rafResult.modelPerRegime} |
        | 예측 기간 | {len(rafResult.predictions)}일 |
        | 예측 평균 | {rafResult.predictions.mean():.2f} |
        """
    )
    return


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. Forecast DNA

        시계열의 "DNA"를 추출하여 핑거프린트를 생성하고,
        난이도를 평가하고, 최적 모델을 추천합니다.
        """
    )
    return


@app.cell
def runDna(ForecastDNA, np):
    np.random.seed(42)
    _n = 200
    _t = np.arange(_n, dtype=np.float64)
    seasonalData = 100 + 0.3 * _t + 15 * np.sin(2 * np.pi * _t / 7) + np.random.normal(0, 3, _n)

    dna = ForecastDNA()
    profile = dna.analyze(seasonalData, period=7)
    return profile,


@app.cell
def showDnaProfile(mo, profile):
    mo.md(
        f"""
        ### DNA 프로필

        | 항목 | 값 |
        |------|-----|
        | 핑거프린트 | `{profile.fingerprint}` |
        | 난이도 | **{profile.difficulty}** ({profile.difficultyScore:.0f}/100) |
        | 카테고리 | **{profile.category}** |
        | 추천 모델 | {', '.join(f'`{m}`' for m in profile.recommendedModels[:5])} |

        DNA 핑거프린트는 동일한 데이터에 대해 항상 같은 값을 생성합니다.
        유사한 패턴의 시계열은 유사한 핑거프린트를 가집니다.
        """
    )
    return


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. 자가 치유 예측 (Self-Healing)

        실시간으로 예측 오차를 모니터링하고, 자동으로 교정합니다.
        프로덕션 환경에서 예측 품질을 지속적으로 개선합니다.
        """
    )
    return


@app.cell
def runSelfHealing(SelfHealingForecast, np):
    np.random.seed(42)
    _n = 50
    originalPred = np.full(_n, 100.0)
    _lower = originalPred - 10
    _upper = originalPred + 10
    _historicalData = np.random.normal(100, 5, 100)

    healer = SelfHealingForecast(originalPred, _lower, _upper, _historicalData)

    actualValues = np.array([105, 110, 115, 108, 120])
    healer.observe(actualValues)
    healingReport = healer.getReport()
    updatedForecast = healer.getUpdatedForecast()
    return healer, healingReport, updatedForecast, originalPred


@app.cell
def showHealing(mo, healingReport, originalPred, updatedForecast):
    mo.md(
        f"""
        ### 자가 치유 결과

        | 항목 | 값 |
        |------|-----|
        | 건강 상태 | {healingReport.overallHealth} |
        | 건강 점수 | {healingReport.healthScore:.1f}/100 |
        | 관측 횟수 | {healingReport.totalObserved} |
        | 교정 횟수 | {healingReport.totalCorrected} |
        | 원래 MAPE | {healingReport.originalMape:.2f}% |
        | 치유 후 MAPE | {healingReport.healedMape:.2f}% |
        | 개선율 | {healingReport.improvementPct:.1f}% |

        실제 값이 예측보다 높게 나오면, 자가 치유 시스템이
        나머지 예측을 상향 조정합니다.
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. 제약 조건 인식 예측

        비즈니스 제약 조건을 예측에 적용합니다:
        - 음수 불가 (재고, 매출)
        - 범위 제한 (용량, 예산)
        - 변화율 제한 (급격한 변동 방지)
        """
    )
    return


@app.cell
def runConstraints(ConstraintAwareForecaster, Constraint, np):
    np.random.seed(42)
    rawPred = np.array([150, -20, 300, 50, 6000, 80, 120, 250, -10, 400])
    _lower = rawPred - 30
    _upper = rawPred + 30

    caf = ConstraintAwareForecaster()
    constrainedResult = caf.apply(rawPred, _lower, _upper, constraints=[
        Constraint('non_negative', {}),
        Constraint('range', {'min': 0, 'max': 500}),
    ])
    return rawPred, constrainedResult


@app.cell
def showConstraints(mo, rawPred, constrainedResult, pd):
    compDf = pd.DataFrame({
        "원래 예측": rawPred,
        "제약 적용 후": constrainedResult.predictions,
        "변경됨": rawPred != constrainedResult.predictions,
    })

    mo.md(
        f"""
        ### 제약 조건 적용 결과

        적용된 제약: `non_negative` + `range(0, 500)`
        """
    )
    return compDf,


@app.cell
def showConstraintTable(mo, compDf):
    mo.md(
        f"""
        ```
        {compDf.to_string(index=False)}
        ```

        -20 → 0 (음수 제거), 6000 → 500 (범위 초과 제한)

        **다음 튜토리얼:** `06_business.py` — 비즈니스 인텔리전스
        """
    )
    return


if __name__ == "__main__":
    app.run()
