# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
#     "matplotlib",
# ]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # 시계열 DNA 분석

        **데이터의 특성을 자동으로 분석하고, 최적의 예측 전략을 추천합니다.**

        Vectrix의 `analyze()` 함수는 시계열의 "DNA"를 추출합니다:
        - 난이도 점수 (0-100)
        - 카테고리 분류 (추세형, 계절형, 변동형 등)
        - 변화점/이상치 자동 탐지
        - 최적 모델 추천
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import analyze, quick_report
    return np, pd, analyze, quick_report


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. 샘플 데이터 선택

        다양한 패턴의 시계열을 선택해서 DNA 분석 결과를 비교해보세요.
        """
    )
    return


@app.cell
def dataSelector(mo):
    dataChoice = mo.ui.dropdown(
        options={
            "추세 + 계절성": "trendSeasonal",
            "순수 추세": "pureTrend",
            "높은 변동성": "volatile",
            "간헐적 수요": "intermittent",
            "다중 계절성": "multiSeasonal",
        },
        value="trendSeasonal",
        label="데이터 패턴"
    )
    return dataChoice,


@app.cell
def showSelector(dataChoice):
    dataChoice
    return


@app.cell
def generateData(np, pd, dataChoice):
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=np.float64)

    choice = dataChoice.value
    if choice == "trendSeasonal":
        values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)
        desc = "선형 추세 + 주간 계절성 + 노이즈"
    elif choice == "pureTrend":
        values = 50 + 0.8 * t + np.random.normal(0, 2, n)
        desc = "강한 상승 추세 + 약한 노이즈"
    elif choice == "volatile":
        returns = np.zeros(n)
        sigma2 = np.ones(n)
        for i in range(1, n):
            sigma2[i] = 0.05 + 0.1 * returns[i - 1] ** 2 + 0.85 * sigma2[i - 1]
            returns[i] = np.random.normal(0, np.sqrt(sigma2[i]))
        values = 100 + np.cumsum(returns)
        desc = "GARCH 스타일 변동성 클러스터링"
    elif choice == "intermittent":
        values = np.zeros(n)
        for i in range(n):
            if np.random.random() < 0.3:
                values[i] = np.random.exponential(50)
        desc = "70% 확률로 0, 간헐적 수요 패턴"
    else:
        values = (100 + 0.2 * t
                  + 10 * np.sin(2 * np.pi * t / 7)
                  + 20 * np.sin(2 * np.pi * t / 30)
                  + np.random.normal(0, 3, n))
        desc = "주간(7) + 월간(30) 이중 계절성"

    sampleDf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "value": values,
    })
    return sampleDf, desc


@app.cell
def showDataDesc(mo, desc):
    mo.md(f"**선택된 패턴:** {desc}")
    return


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. DNA 분석 실행
        """
    )
    return


@app.cell
def runAnalysis(analyze, sampleDf):
    report = analyze(sampleDf)
    return report,


@app.cell
def showDna(mo, report):
    dna = report.dna
    mo.md(
        f"""
        ### DNA 프로필

        | 항목 | 결과 |
        |------|------|
        | 난이도 | **{dna.difficulty}** ({dna.difficultyScore:.0f}/100) |
        | 카테고리 | **{dna.category}** |
        | 핑거프린트 | `{dna.fingerprint}` |
        | 추천 모델 | {', '.join(f'`{m}`' for m in dna.recommendedModels[:3])} |
        """
    )
    return


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. 데이터 특성
        """
    )
    return


@app.cell
def showCharacteristics(mo, report):
    c = report.characteristics
    mo.md(
        f"""
        | 특성 | 값 |
        |------|-----|
        | 데이터 길이 | {c.length} |
        | 빈도 | {c.frequency} |
        | 주기 | {c.period} |
        | 추세 | {'있음' if c.hasTrend else '없음'} ({c.trendDirection}, 강도 {c.trendStrength:.2f}) |
        | 계절성 | {'있음' if c.hasSeasonality else '없음'} (강도 {c.seasonalStrength:.2f}) |
        | 변동성 | {c.volatilityLevel} ({c.volatility:.4f}) |
        | 예측 가능성 | {c.predictabilityScore:.0f}/100 |
        | 이상치 | {c.outlierCount}개 ({c.outlierRatio:.1%}) |
        """
    )
    return


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. 변화점 & 이상치 탐지
        """
    )
    return


@app.cell
def showDetection(mo, report):
    nCp = len(report.changepoints) if report.changepoints is not None else 0
    nAn = len(report.anomalies) if report.anomalies is not None else 0

    cpStr = str(list(report.changepoints[:5])) if nCp > 0 else "없음"
    anStr = str(list(report.anomalies[:5])) if nAn > 0 else "없음"

    mo.md(
        f"""
        | 탐지 | 개수 | 위치 (최대 5개) |
        |------|------|-----------------|
        | 변화점 | {nCp}개 | {cpStr} |
        | 이상치 | {nAn}개 | {anStr} |
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. 통합 리포트 (quick_report)

        `quick_report()`는 분석 + 예측을 한 번에 수행합니다.
        """
    )
    return


@app.cell
def runQuickReport(quick_report, sampleDf):
    fullReport = quick_report(sampleDf, steps=14)
    return fullReport,


@app.cell
def showQuickReport(mo, fullReport):
    mo.md(
        f"""
        ```
        {fullReport['summary']}
        ```

        **다음 튜토리얼:** `03_regression.py` — R-style 회귀분석
        """
    )
    return


if __name__ == "__main__":
    app.run()
