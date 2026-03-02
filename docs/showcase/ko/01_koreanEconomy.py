# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "vectrix",
#     "matplotlib",
#     "pandas",
# ]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def header(mo):
    mo.md(
        """
        # 한국 경제 데이터 예측 쇼케이스

        **실제 한국 경제 지표를 Vectrix로 분석하고 예측합니다.**

        이 노트북에서는 FRED(미국 연방준비은행 경제 데이터)에서
        한국 관련 경제 시계열을 실시간으로 가져와서
        Vectrix의 자동 예측 엔진을 시연합니다.

        - 원/달러 환율
        - KOSPI 주가지수
        - 소비자물가지수 (CPI)
        - 4개 지표 비교 DNA 분석
        """
    )
    return


@app.cell
def imports():
    import pandas as pd
    from vectrix import forecast, analyze
    return pd, forecast, analyze


@app.cell
def scenario1Header(mo):
    mo.md(
        """
        ---
        ## 1. 원/달러 환율 예측

        FRED에서 제공하는 **한국 원/달러 환율(EXKOUS)** 월간 데이터입니다.
        1981년부터 현재까지의 환율 데이터를 가져와 향후 12개월을 예측합니다.

        Vectrix는 30개 이상의 모델을 자동으로 비교하여
        이 데이터에 가장 적합한 모델을 선택합니다.
        """
    )
    return


@app.cell
def loadExchangeRate(pd):
    exchangeUrl = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
    exchangeDf = pd.read_csv(exchangeUrl, parse_dates=["DATE"])
    exchangeDf.columns = ["date", "value"]
    exchangeDf = exchangeDf.dropna()
    return exchangeDf,


@app.cell
def showExchangeData(mo, exchangeDf):
    mo.md(
        f"""
        ### 데이터 개요

        | 항목 | 값 |
        |------|-----|
        | 시작일 | {exchangeDf['date'].iloc[0].strftime('%Y-%m-%d')} |
        | 종료일 | {exchangeDf['date'].iloc[-1].strftime('%Y-%m-%d')} |
        | 데이터 수 | {len(exchangeDf):,}개 |
        | 최근 환율 | {exchangeDf['value'].iloc[-1]:,.1f} 원/달러 |
        | 최고 환율 | {exchangeDf['value'].max():,.1f} 원/달러 |
        | 최저 환율 | {exchangeDf['value'].min():,.1f} 원/달러 |
        """
    )
    return


@app.cell
def forecastExchangeRate(forecast, exchangeDf):
    exchangeResult = forecast(exchangeDf, date="date", value="value", steps=12)
    return exchangeResult,


@app.cell
def showExchangeResult(mo, exchangeResult):
    mo.md(
        f"""
        ### 환율 예측 결과

        | 항목 | 값 |
        |------|-----|
        | 선택된 모델 | `{exchangeResult.model}` |
        | 예측 기간 | {len(exchangeResult.predictions)}개월 |
        | 예측 평균 | {exchangeResult.predictions.mean():,.1f} 원 |
        | 예측 범위 | {exchangeResult.predictions.min():,.1f} ~ {exchangeResult.predictions.max():,.1f} 원 |
        | 95% 신뢰구간 하한 | {exchangeResult.lower.min():,.1f} 원 |
        | 95% 신뢰구간 상한 | {exchangeResult.upper.max():,.1f} 원 |
        """
    )
    return


@app.cell
def showExchangeSummary(mo, exchangeResult):
    mo.md(f"```\n{exchangeResult.summary()}\n```")
    return


@app.cell
def showExchangeTable(mo, exchangeResult):
    exchangeForecastDf = exchangeResult.to_dataframe()
    mo.md("### 환율 예측 상세 테이블")
    return exchangeForecastDf,


@app.cell
def displayExchangeTable(mo, exchangeForecastDf):
    return mo.ui.table(exchangeForecastDf)


@app.cell
def scenario2Header(mo):
    mo.md(
        """
        ---
        ## 2. KOSPI 주가지수 예측

        **KOSPI 종합주가지수(SPASTT01KRM661N)** 월간 데이터입니다.
        한국 주식시장의 대표 지수를 Vectrix로 예측합니다.

        주가지수는 변동성이 높고 비선형 패턴이 강하여
        예측 난이도가 높은 데이터 유형입니다.
        """
    )
    return


@app.cell
def loadKospi(pd):
    kospiUrl = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SPASTT01KRM661N"
    kospiDf = pd.read_csv(kospiUrl, parse_dates=["DATE"])
    kospiDf.columns = ["date", "value"]
    kospiDf = kospiDf.dropna()
    return kospiDf,


@app.cell
def showKospiData(mo, kospiDf):
    mo.md(
        f"""
        ### 데이터 개요

        | 항목 | 값 |
        |------|-----|
        | 시작일 | {kospiDf['date'].iloc[0].strftime('%Y-%m-%d')} |
        | 종료일 | {kospiDf['date'].iloc[-1].strftime('%Y-%m-%d')} |
        | 데이터 수 | {len(kospiDf):,}개 |
        | 최근 지수 | {kospiDf['value'].iloc[-1]:,.2f} |
        | 최고 지수 | {kospiDf['value'].max():,.2f} |
        | 최저 지수 | {kospiDf['value'].min():,.2f} |
        """
    )
    return


@app.cell
def forecastKospi(forecast, kospiDf):
    kospiResult = forecast(kospiDf, date="date", value="value", steps=12)
    return kospiResult,


@app.cell
def showKospiResult(mo, kospiResult):
    mo.md(
        f"""
        ### KOSPI 예측 결과

        | 항목 | 값 |
        |------|-----|
        | 선택된 모델 | `{kospiResult.model}` |
        | 예측 기간 | {len(kospiResult.predictions)}개월 |
        | 예측 평균 | {kospiResult.predictions.mean():,.2f} |
        | 예측 범위 | {kospiResult.predictions.min():,.2f} ~ {kospiResult.predictions.max():,.2f} |
        | 95% 신뢰구간 하한 | {kospiResult.lower.min():,.2f} |
        | 95% 신뢰구간 상한 | {kospiResult.upper.max():,.2f} |
        """
    )
    return


@app.cell
def showKospiSummary(mo, kospiResult):
    mo.md(f"```\n{kospiResult.summary()}\n```")
    return


@app.cell
def scenario3Header(mo):
    mo.md(
        """
        ---
        ## 3. 한국 소비자물가지수 (CPI) 예측

        **한국 소비자물가지수(KORCPIALLMINMEI)** 월간 데이터입니다.
        CPI는 인플레이션의 핵심 지표로, 강한 추세와
        안정적인 패턴을 보이는 시계열입니다.

        이런 유형의 데이터에서 Vectrix는 특히 높은 정확도를 보입니다.
        """
    )
    return


@app.cell
def loadCpi(pd):
    cpiUrl = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=KORCPIALLMINMEI"
    cpiDf = pd.read_csv(cpiUrl, parse_dates=["DATE"])
    cpiDf.columns = ["date", "value"]
    cpiDf = cpiDf.dropna()
    return cpiDf,


@app.cell
def showCpiData(mo, cpiDf):
    mo.md(
        f"""
        ### 데이터 개요

        | 항목 | 값 |
        |------|-----|
        | 시작일 | {cpiDf['date'].iloc[0].strftime('%Y-%m-%d')} |
        | 종료일 | {cpiDf['date'].iloc[-1].strftime('%Y-%m-%d')} |
        | 데이터 수 | {len(cpiDf):,}개 |
        | 최근 CPI | {cpiDf['value'].iloc[-1]:.2f} |
        | 최고 CPI | {cpiDf['value'].max():.2f} |
        | 최저 CPI | {cpiDf['value'].min():.2f} |
        """
    )
    return


@app.cell
def forecastCpi(forecast, cpiDf):
    cpiResult = forecast(cpiDf, date="date", value="value", steps=12)
    return cpiResult,


@app.cell
def showCpiResult(mo, cpiResult):
    mo.md(
        f"""
        ### CPI 예측 결과

        | 항목 | 값 |
        |------|-----|
        | 선택된 모델 | `{cpiResult.model}` |
        | 예측 기간 | {len(cpiResult.predictions)}개월 |
        | 예측 평균 | {cpiResult.predictions.mean():.2f} |
        | 예측 범위 | {cpiResult.predictions.min():.2f} ~ {cpiResult.predictions.max():.2f} |
        | 95% 신뢰구간 하한 | {cpiResult.lower.min():.2f} |
        | 95% 신뢰구간 상한 | {cpiResult.upper.max():.2f} |
        """
    )
    return


@app.cell
def showCpiSummary(mo, cpiResult):
    mo.md(f"```\n{cpiResult.summary()}\n```")
    return


@app.cell
def scenario4Header(mo):
    mo.md(
        """
        ---
        ## 4. Multi-indicator 비교 분석

        4개 한국 경제 지표의 **시계열 DNA**를 비교합니다.

        | 코드 | 지표 |
        |------|------|
        | EXKOUS | 원/달러 환율 |
        | IRLTLT01KRM156N | 장기 금리 |
        | LRUNTTTTKRM156S | 실업률 |
        | SPASTT01KRM661N | KOSPI 지수 |

        각 시계열의 난이도, 카테고리, 특성을 DNA 분석으로 비교하면
        어떤 데이터가 예측하기 쉽고, 어떤 데이터가 어려운지 파악할 수 있습니다.
        """
    )
    return


@app.cell
def loadMultiIndicators(pd):
    seriesIds = {
        "EXKOUS": "원/달러 환율",
        "IRLTLT01KRM156N": "장기 금리",
        "LRUNTTTTKRM156S": "실업률",
        "SPASTT01KRM661N": "KOSPI 지수",
    }

    multiData = {}
    for seriesId, label in seriesIds.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={seriesId}"
        rawDf = pd.read_csv(url, parse_dates=["DATE"])
        rawDf.columns = ["date", "value"]
        rawDf = rawDf.dropna()
        multiData[label] = rawDf

    return multiData, seriesIds


@app.cell
def analyzeMultiIndicators(analyze, multiData):
    dnaResults = {}
    for label, df in multiData.items():
        report = analyze(df, date="date", value="value")
        dnaResults[label] = report

    return dnaResults,


@app.cell
def showDnaComparison(mo, dnaResults):
    rows = []
    for label, report in dnaResults.items():
        dna = report.dna
        chars = report.characteristics
        rows.append(
            f"| {label} | {dna.difficulty} | {dna.difficultyScore:.0f}/100 "
            f"| {dna.category} | {chars.trendStrength:.2f} "
            f"| {chars.seasonalStrength:.2f} | {chars.volatilityLevel} "
            f"| {', '.join(f'`{m}`' for m in dna.recommendedModels[:2])} |"
        )

    tableBody = "\n".join(rows)

    mo.md(
        f"""
        ### DNA 비교 분석 결과

        | 지표 | 난이도 | 점수 | 카테고리 | 추세 강도 | 계절성 강도 | 변동성 | 추천 모델 (상위 2개) |
        |------|--------|------|----------|-----------|-------------|--------|----------------------|
        {tableBody}
        """
    )
    return


@app.cell
def showDnaFingerprints(mo, dnaResults):
    fpRows = []
    for label, report in dnaResults.items():
        dna = report.dna
        chars = report.characteristics
        fpRows.append(
            f"| {label} | `{dna.fingerprint}` "
            f"| {chars.length} | {chars.frequency} "
            f"| {chars.predictabilityScore:.0f}/100 |"
        )

    fpTable = "\n".join(fpRows)

    mo.md(
        f"""
        ### DNA 핑거프린트 비교

        | 지표 | 핑거프린트 | 데이터 수 | 빈도 | 예측 가능성 |
        |------|------------|-----------|------|-------------|
        {fpTable}

        핑거프린트는 시계열의 고유한 특성을 압축한 코드입니다.
        유사한 핑거프린트를 가진 시계열은 비슷한 예측 전략이 효과적입니다.
        """
    )
    return


@app.cell
def showChangepoints(mo, dnaResults):
    cpRows = []
    for label, report in dnaResults.items():
        nCp = len(report.changepoints) if report.changepoints is not None else 0
        nAn = len(report.anomalies) if report.anomalies is not None else 0
        cpRows.append(f"| {label} | {nCp}개 | {nAn}개 |")

    cpTable = "\n".join(cpRows)

    mo.md(
        f"""
        ### 변화점 & 이상치 탐지

        | 지표 | 변화점 | 이상치 |
        |------|--------|--------|
        {cpTable}

        변화점이 많을수록 구조적 변화가 잦은 시계열이며,
        이상치가 많을수록 예측 불확실성이 높아집니다.
        """
    )
    return


@app.cell
def conclusionHeader(mo):
    mo.md(
        """
        ---
        ## 분석 요약

        이 쇼케이스에서 확인할 수 있는 핵심 포인트:

        1. **Vectrix는 한 줄의 코드로** 실제 경제 데이터를 예측합니다
        2. **DNA 분석**으로 각 시계열의 본질적 특성을 파악할 수 있습니다
        3. **모델 자동 선택**이 데이터 특성에 맞는 최적 모델을 찾아냅니다
        4. **95% 신뢰구간**이 예측의 불확실성을 정량화합니다
        5. 경제 지표마다 **예측 난이도와 최적 전략이 다릅니다**

        ```python
        from vectrix import forecast, analyze

        result = forecast(df, date="date", value="value", steps=12)
        report = analyze(df, date="date", value="value")
        ```
        """
    )
    return


@app.cell
def disclaimer(mo):
    mo.md(
        """
        ---
        > **면책 조항**: 이 분석은 교육 목적이며, 실제 투자 결정에 사용하지 마세요.
        > 과거 데이터 기반 통계적 예측은 미래 수익을 보장하지 않습니다.
        > 경제 예측에는 정치, 지정학, 정책 변화 등 모델에 포함되지 않는
        > 다양한 외부 요인이 영향을 미칩니다.
        """
    )
    return


if __name__ == "__main__":
    app.run()
