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
        # 30+ 모델 비교

        **Vectrix는 30개 이상의 모델을 자동으로 실행하고, 최적의 모델을 선택합니다.**

        이 튜토리얼에서는 `Vectrix` 클래스를 직접 사용하여:
        - 모든 모델의 성능을 비교하고
        - Flat Defense 시스템이 어떻게 작동하는지
        - 앙상블 vs 개별 모델 차이를 확인합니다.
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import Vectrix
    return np, pd, Vectrix


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. 데이터 준비 & 예측 실행
        """
    )
    return


@app.cell
def trainRatioSlider(mo):
    ratioControl = mo.ui.slider(
        start=0.6, stop=0.9, step=0.05, value=0.8,
        label="학습 비율 (trainRatio)"
    )
    return ratioControl,


@app.cell
def showRatioSlider(ratioControl):
    ratioControl
    return


@app.cell
def createData(np, pd):
    np.random.seed(42)
    _n = 200
    _t = np.arange(_n, dtype=np.float64)
    values = 100 + 0.3 * _t + 15 * np.sin(2 * np.pi * _t / 7) + np.random.normal(0, 3, _n)

    tsDf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=_n, freq="D"),
        "value": values,
    })
    return tsDf,


@app.cell
def runForecast(Vectrix, tsDf, ratioControl):
    vx = Vectrix(verbose=False)
    fcResult = vx.forecast(
        tsDf,
        dateCol="date",
        valueCol="value",
        steps=14,
        trainRatio=ratioControl.value
    )
    return vx, fcResult


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. 모델 성능 비교
        """
    )
    return


@app.cell
def modelComparison(mo, fcResult, pd):
    rows = []
    if fcResult.allModelResults:
        for modelId, mr in fcResult.allModelResults.items():
            isFlat = ""
            if mr.flatInfo and mr.flatInfo.isFlat:
                isFlat = mr.flatInfo.flatType
            rows.append({
                "모델": mr.modelName,
                "MAPE": round(mr.mape, 2),
                "RMSE": round(mr.rmse, 2),
                "MAE": round(mr.mae, 2),
                "Flat": isFlat,
                "학습시간(s)": round(mr.trainingTime, 3),
            })

    compDf = pd.DataFrame(rows).sort_values("MAPE")
    mo.md(f"### 전체 모델 결과 ({len(rows)}개)")
    return compDf,


@app.cell
def showComparison(mo, compDf):
    return mo.ui.table(compDf)


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. 최적 모델 & 데이터 특성
        """
    )
    return


@app.cell
def showBestModel(mo, fcResult):
    _c = fcResult.characteristics
    _fr = fcResult.flatRisk

    mo.md(
        f"""
        | 항목 | 값 |
        |------|-----|
        | **최적 모델** | `{fcResult.bestModelName}` |
        | 예측 성공 | {fcResult.success} |
        | 데이터 길이 | {_c.length} |
        | 감지된 주기 | {_c.period} |
        | 추세 | {_c.trendDirection} (강도 {_c.trendStrength:.2f}) |
        | 계절성 강도 | {_c.seasonalStrength:.2f} |
        | Flat Risk | {_fr.riskLevel.name} (점수 {_fr.riskScore:.2f}) |
        """
    )
    return


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. Flat Defense 시스템

        Vectrix는 4단계로 평탄한 예측(flat prediction)을 방어합니다:

        1. **FlatRiskDiagnostic** — 사전 위험도 평가
        2. **AdaptiveModelSelector** — 위험도 기반 모델 선택
        3. **FlatPredictionDetector** — 사후 평탄 감지
        4. **FlatPredictionCorrector** — 평탄 예측 교정

        다른 예측 라이브러리에는 없는 Vectrix만의 기능입니다.
        """
    )
    return


@app.cell
def showFlatDefense(mo, fcResult):
    _fr = fcResult.flatRisk

    _factorStr = ""
    for _factor, _active in _fr.riskFactors.items():
        if _active:
            _factorStr += f"- {_factor}\n"
    if not _factorStr:
        _factorStr = "- 없음"

    _warningStr = ""
    for _w in (_fr.warnings or []):
        _warningStr += f"- {_w}\n"
    if not _warningStr:
        _warningStr = "- 없음"

    mo.md(
        f"""
        ### Flat Risk 분석

        **위험 수준:** {_fr.riskLevel.name} (점수: {_fr.riskScore:.2f})

        **위험 요인:**
        {_factorStr}

        **추천 전략:** {_fr.recommendedStrategy}

        **경고:**
        {_warningStr}
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. 모델 카테고리

        | 카테고리 | 모델 | 최적 대상 |
        |----------|------|-----------|
        | **지수평활** | AutoETS, ETS(A,A,A), ETS(A,A,N) | 안정적 패턴 |
        | **ARIMA** | AutoARIMA | 정상 시계열 |
        | **분해** | MSTL, AutoMSTL | 다중 계절성 |
        | **Theta** | Theta, DOT | 범용 |
        | **삼각함수** | TBATS | 복잡한 계절성 |
        | **복소수** | AutoCES | 비선형 패턴 |
        | **간헐적** | Croston, SBA, TSB | 희소 수요 |
        | **변동성** | GARCH, EGARCH, GJR | 금융 데이터 |
        | **기준선** | Naive, Seasonal Naive, Mean, RWD | 벤치마크 |

        **다음 튜토리얼:** `05_adaptive.py` — 적응형 예측
        """
    )
    return


if __name__ == "__main__":
    app.run()
