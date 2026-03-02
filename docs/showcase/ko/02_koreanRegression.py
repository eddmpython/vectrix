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
def imports():
    import pandas as pd
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import marimo as mo
    from vectrix import regress, forecast

    matplotlib.rcParams["font.family"] = "Malgun Gothic"
    matplotlib.rcParams["axes.unicode_minus"] = False

    return forecast, mo, np, pd, plt, regress


@app.cell
def header(mo):
    return mo.md(
        """
        # 한국 실제 데이터 회귀분석 쇼케이스

        **Vectrix**의 회귀분석 기능을 한국 관련 실제 데이터로 시연합니다.

        | 시나리오 | 데이터 | 분석 |
        |---------|--------|------|
        | 1 | 서울 자전거 대여량 | 다중 회귀분석 (기온, 습도, 강수량 등) |
        | 2 | 한국 거시경제 지표 | 환율 결정요인 분석 (FRED 데이터) |
        | 3 | 자전거 일별 집계 | 시계열 예측 (같은 데이터, 다른 관점) |

        > **이 분석은 교육 목적이며, 실제 투자/사업 결정에 사용하지 마세요.**
        """
    )


@app.cell
def scenario1Title(mo):
    return mo.md(
        """
        ---
        ## 시나리오 1: 서울 자전거 대여량 회귀분석

        UCI Machine Learning Repository의 **Seoul Bike Sharing Demand** 데이터를 사용합니다.
        2017~2018년 서울시 공공자전거(따릉이) 시간별 대여 데이터로,
        기온, 습도, 풍속, 강수량 등 기상 조건이 자전거 대여량에 미치는 영향을 분석합니다.
        """
    )


@app.cell
def loadBikeData(pd):
    bikeUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
    bikeDf = pd.read_csv(bikeUrl, encoding="unicode_escape")

    columnMap = {
        "Rented Bike Count": "rentedBikeCount",
        "Hour": "hour",
        "Humidity(%)": "humidity",
        "Wind speed (m/s)": "windSpeed",
        "Visibility (10m)": "visibility",
        "Solar Radiation (MJ/m2)": "solarRadiation",
        "Rainfall(mm)": "rainfall",
        "Snowfall (cm)": "snowfall",
        "Seasons": "seasons",
        "Holiday": "holiday",
        "Functioning Day": "functioningDay",
    }

    tempCols = [c for c in bikeDf.columns if "Temperature" in c and "Dew" not in c]
    if tempCols:
        columnMap[tempCols[0]] = "temperature"

    dewCols = [c for c in bikeDf.columns if "Dew" in c]
    if dewCols:
        columnMap[dewCols[0]] = "dewPoint"

    bikeDf = bikeDf.rename(columns=columnMap)

    return (bikeDf,)


@app.cell
def showBikeData(mo, bikeDf):
    return mo.md(
        f"""
        ### 데이터 미리보기

        - **행 수**: {len(bikeDf):,}개 (시간별 관측)
        - **기간**: {bikeDf['Date'].iloc[0]} ~ {bikeDf['Date'].iloc[-1]}
        - **변수**: 기온, 습도, 풍속, 가시거리, 이슬점, 태양복사, 강수량, 적설량, 계절, 공휴일

        {mo.as_html(mo.ui.table(bikeDf.head(10)))}
        """
    )


@app.cell
def bikeRegressionTitle(mo):
    return mo.md(
        """
        ### 회귀분석 실행

        **종속변수**: 자전거 대여량 (`rentedBikeCount`)

        **독립변수**: 기온, 습도, 풍속, 강수량, 시간대, 태양복사량
        """
    )


@app.cell
def bikeRegression(bikeDf, regress):
    bikeResult = regress(
        data=bikeDf,
        formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + hour + solarRadiation",
        summary=False,
    )
    return (bikeResult,)


@app.cell
def bikeResultSummary(mo, bikeResult):
    return mo.md(
        f"""
        ### 회귀분석 결과 요약

        ```
{bikeResult.summary()}
        ```

        | 지표 | 값 |
        |------|-----|
        | **R-squared** | {bikeResult.r_squared:.4f} |
        | **Adjusted R-squared** | {bikeResult.adj_r_squared:.4f} |
        | **F-statistic** | {bikeResult.f_stat:.2f} |
        | **계수 수** | {len(bikeResult.coefficients)} (절편 포함) |
        """
    )


@app.cell
def bikeCoefficients(mo, bikeResult, pd):
    featureLabels = ["절편", "기온", "습도", "풍속", "강수량", "시간대", "태양복사량"]
    coefDf = pd.DataFrame(
        {
            "변수": featureLabels,
            "계수": bikeResult.coefficients,
            "p-value": bikeResult.pvalues,
            "유의성": [
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                for p in bikeResult.pvalues
            ],
        }
    )

    return mo.md(
        f"""
        ### 회귀계수 해석

        {mo.as_html(mo.ui.table(coefDf))}

        **해석 가이드**:
        - **기온**: 1도 상승 시 대여량 약 {bikeResult.coefficients[1]:.1f}건 변화
        - **습도**: 1% 상승 시 대여량 약 {bikeResult.coefficients[2]:.1f}건 변화
        - **시간대**: 1시간 증가 시 대여량 약 {bikeResult.coefficients[5]:.1f}건 변화
        - **강수량**: 비가 올수록 대여량 감소 (계수 = {bikeResult.coefficients[4]:.1f})
        - `***` p < 0.001, `**` p < 0.01, `*` p < 0.05, `n.s.` 유의하지 않음
        """
    )


@app.cell
def bikeDiagnostics(mo, bikeResult):
    return mo.md(
        f"""
        ### 회귀 진단

        ```
{bikeResult.diagnose()}
        ```
        """
    )


@app.cell
def bikePlot(mo, bikeDf, bikeResult, plt, np):
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    predicted1 = bikeResult._result.fittedValues
    actual1 = bikeDf["rentedBikeCount"].values[: len(predicted1)]

    axes1[0].scatter(actual1, predicted1, alpha=0.1, s=5, color="#3498db")
    maxVal1 = max(actual1.max(), predicted1.max())
    axes1[0].plot([0, maxVal1], [0, maxVal1], "r--", linewidth=1.5, label="y = x")
    axes1[0].set_xlabel("실제 대여량")
    axes1[0].set_ylabel("예측 대여량")
    axes1[0].set_title("실제값 vs 예측값")
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)

    residuals1 = bikeResult._result.residuals
    axes1[1].hist(residuals1, bins=60, color="#2ecc71", edgecolor="white", alpha=0.8)
    axes1[1].axvline(x=0, color="red", linestyle="--", linewidth=1.5)
    axes1[1].set_xlabel("잔차")
    axes1[1].set_ylabel("빈도")
    axes1[1].set_title("잔차 분포")
    axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()

    return mo.md(
        f"""
        ### 실제값 vs 예측값 시각화

        {mo.as_html(fig1)}
        """
    )


@app.cell
def scenario2Title(mo):
    return mo.md(
        """
        ---
        ## 시나리오 2: 한국 거시경제 회귀 — 환율 결정요인 분석

        **FRED**(Federal Reserve Economic Data)에서 한국 관련 거시경제 지표를 가져와
        원/달러 환율의 결정요인을 분석합니다.

        | 변수 | FRED 코드 | 설명 |
        |------|----------|------|
        | 환율 (종속변수) | EXKOUS | 원/달러 환율 |
        | 장기채권금리 | IRLTLT01KRM156N | 한국 장기 국채 수익률 |
        | 실업률 | LRUNTTTTKRM156S | 한국 실업률 |
        | 주가지수 | SPASTT01KRM661N | 한국 주가지수 |
        | 소비자물가지수 | KORCPIALLMINMEI | 한국 CPI |
        """
    )


@app.cell
def loadMacroData(pd):
    seriesIds = [
        "EXKOUS",
        "IRLTLT01KRM156N",
        "LRUNTTTTKRM156S",
        "SPASTT01KRM661N",
        "KORCPIALLMINMEI",
    ]
    frames = []
    for sid in seriesIds:
        fredUrl = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
        tempDf = pd.read_csv(fredUrl, parse_dates=["DATE"], na_values=".")
        tempDf.columns = ["date", sid]
        tempDf = tempDf.set_index("date")
        frames.append(tempDf)

    macroDf = pd.concat(frames, axis=1).dropna()
    macroDf = macroDf.rename(
        columns={
            "EXKOUS": "exchangeRate",
            "IRLTLT01KRM156N": "bondYield",
            "LRUNTTTTKRM156S": "unemployment",
            "SPASTT01KRM661N": "stockIndex",
            "KORCPIALLMINMEI": "cpi",
        }
    )
    macroDf = macroDf.reset_index()

    return (macroDf,)


@app.cell
def showMacroData(mo, macroDf):
    return mo.md(
        f"""
        ### 거시경제 데이터 미리보기

        - **관측 수**: {len(macroDf):,}개 (월별)
        - **기간**: {macroDf['date'].iloc[0].strftime('%Y-%m')} ~ {macroDf['date'].iloc[-1].strftime('%Y-%m')}

        {mo.as_html(mo.ui.table(macroDf.tail(10)))}
        """
    )


@app.cell
def macroRegression(macroDf, regress):
    macroResult = regress(
        data=macroDf,
        formula="exchangeRate ~ bondYield + unemployment + stockIndex + cpi",
        summary=False,
    )
    return (macroResult,)


@app.cell
def macroResultSummary(mo, macroResult):
    return mo.md(
        f"""
        ### 환율 회귀분석 결과

        ```
{macroResult.summary()}
        ```

        | 지표 | 값 |
        |------|-----|
        | **R-squared** | {macroResult.r_squared:.4f} |
        | **Adjusted R-squared** | {macroResult.adj_r_squared:.4f} |
        | **F-statistic** | {macroResult.f_stat:.2f} |
        """
    )


@app.cell
def macroCoefficients(mo, macroResult, pd):
    macroLabels = ["절편", "채권금리", "실업률", "주가지수", "소비자물가지수"]
    macroCoefDf = pd.DataFrame(
        {
            "변수": macroLabels,
            "계수": macroResult.coefficients,
            "p-value": macroResult.pvalues,
            "유의성": [
                "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                for p in macroResult.pvalues
            ],
        }
    )

    return mo.md(
        f"""
        ### 환율 결정요인 계수 해석

        {mo.as_html(mo.ui.table(macroCoefDf))}

        **해석**:
        - **채권금리** 상승 시 환율 변화: {macroResult.coefficients[1]:+.2f}원
        - **실업률** 1%p 상승 시 환율 변화: {macroResult.coefficients[2]:+.2f}원
        - **주가지수** 1포인트 상승 시 환율 변화: {macroResult.coefficients[3]:+.4f}원
        - **CPI** 1포인트 상승 시 환율 변화: {macroResult.coefficients[4]:+.2f}원

        > 거시경제 변수 간에는 다중공선성이 존재할 수 있으므로 해석에 주의가 필요합니다.
        """
    )


@app.cell
def macroPlot(mo, macroDf, macroResult, plt):
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    dates2 = macroDf["date"]
    actual2 = macroDf["exchangeRate"].values
    predicted2 = macroResult._result.fittedValues[: len(actual2)]

    axes2[0, 0].plot(dates2, actual2, color="#2c3e50", linewidth=1, label="실제 환율")
    axes2[0, 0].plot(
        dates2, predicted2, color="#e74c3c", linewidth=1, alpha=0.8, label="회귀 적합값"
    )
    axes2[0, 0].set_title("원/달러 환율: 실제 vs 적합")
    axes2[0, 0].legend()
    axes2[0, 0].grid(True, alpha=0.3)

    axes2[0, 1].scatter(
        macroDf["bondYield"], actual2, alpha=0.4, s=10, color="#3498db"
    )
    axes2[0, 1].set_xlabel("장기채권금리 (%)")
    axes2[0, 1].set_ylabel("환율 (원)")
    axes2[0, 1].set_title("채권금리 vs 환율")
    axes2[0, 1].grid(True, alpha=0.3)

    axes2[1, 0].scatter(macroDf["cpi"], actual2, alpha=0.4, s=10, color="#e67e22")
    axes2[1, 0].set_xlabel("소비자물가지수")
    axes2[1, 0].set_ylabel("환율 (원)")
    axes2[1, 0].set_title("CPI vs 환율")
    axes2[1, 0].grid(True, alpha=0.3)

    residuals2 = macroResult._result.residuals[: len(actual2)]
    axes2[1, 1].plot(dates2, residuals2, color="#27ae60", linewidth=0.8)
    axes2[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=1)
    axes2[1, 1].set_title("잔차 시계열")
    axes2[1, 1].set_ylabel("잔차")
    axes2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    return mo.md(
        f"""
        ### 환율 시계열 시각화

        {mo.as_html(fig2)}
        """
    )


@app.cell
def scenario3Title(mo):
    return mo.md(
        """
        ---
        ## 시나리오 3: 같은 데이터, 다른 관점 — 시계열 예측

        서울 자전거 대여 데이터를 **일별 합계**로 집계한 후,
        Vectrix의 `forecast()` 함수로 향후 30일을 예측합니다.

        **회귀분석**은 "무엇이 대여량에 영향을 미치는가?"를 분석하고,
        **시계열 예측**은 "앞으로 대여량이 어떻게 될까?"를 예측합니다.
        같은 데이터에서 서로 다른 인사이트를 얻을 수 있습니다.
        """
    )


@app.cell
def prepareDailyData(bikeDf, pd):
    dailyDf = bikeDf.copy()
    dailyDf["date"] = pd.to_datetime(dailyDf["Date"], format="%d/%m/%Y")
    dailyAgg = dailyDf.groupby("date")["rentedBikeCount"].sum().reset_index()
    dailyAgg.columns = ["date", "dailyTotal"]

    return (dailyAgg,)


@app.cell
def showDailyData(mo, dailyAgg):
    return mo.md(
        f"""
        ### 일별 자전거 대여 현황

        - **기간**: {dailyAgg['date'].iloc[0].strftime('%Y-%m-%d')} ~ {dailyAgg['date'].iloc[-1].strftime('%Y-%m-%d')}
        - **총 일수**: {len(dailyAgg)}일
        - **일평균 대여량**: {dailyAgg['dailyTotal'].mean():,.0f}건
        - **최대 대여량**: {dailyAgg['dailyTotal'].max():,.0f}건 ({dailyAgg.loc[dailyAgg['dailyTotal'].idxmax(), 'date'].strftime('%Y-%m-%d')})
        - **최소 대여량**: {dailyAgg['dailyTotal'].min():,.0f}건 ({dailyAgg.loc[dailyAgg['dailyTotal'].idxmin(), 'date'].strftime('%Y-%m-%d')})
        """
    )


@app.cell
def runForecast(dailyAgg, forecast):
    forecastResult = forecast(
        dailyAgg,
        date="date",
        value="dailyTotal",
        steps=30,
    )

    return (forecastResult,)


@app.cell
def forecastSummary(mo, forecastResult):
    return mo.md(
        f"""
        ### 예측 결과 요약

        ```
{forecastResult.summary()}
        ```

        | 항목 | 값 |
        |------|-----|
        | **선택 모델** | {forecastResult.model} |
        | **예측 기간** | {forecastResult.dates[0]} ~ {forecastResult.dates[-1]} |
        | **예측 평균** | {forecastResult.predictions.mean():,.0f}건/일 |
        """
    )


@app.cell
def forecastPlot(mo, forecastResult, dailyAgg, plt, pd):
    fig3, ax3 = plt.subplots(figsize=(14, 6))

    historicalDates = dailyAgg["date"].values
    historicalValues = dailyAgg["dailyTotal"].values

    ax3.plot(
        historicalDates[-90:],
        historicalValues[-90:],
        color="#2c3e50",
        linewidth=1.5,
        label="실제 데이터 (최근 90일)",
    )

    forecastDates = pd.to_datetime(forecastResult.dates)
    ax3.plot(
        forecastDates,
        forecastResult.predictions,
        color="#e74c3c",
        linewidth=2,
        label=f"예측 ({forecastResult.model})",
    )

    ax3.fill_between(
        forecastDates,
        forecastResult.lower,
        forecastResult.upper,
        color="#e74c3c",
        alpha=0.15,
        label="95% 신뢰구간",
    )

    ax3.set_title(
        "서울 자전거 대여량 — 시계열 예측 (30일)", fontsize=14, fontweight="bold"
    )
    ax3.set_xlabel("날짜")
    ax3.set_ylabel("일별 대여량")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    return mo.md(
        f"""
        ### 시계열 예측 시각화

        {mo.as_html(fig3)}
        """
    )


@app.cell
def comparisonInsight(mo, bikeResult, forecastResult):
    return mo.md(
        f"""
        ---
        ## 회귀분석 vs 시계열 예측 비교

        | 구분 | 회귀분석 | 시계열 예측 |
        |------|---------|-----------|
        | **핵심 질문** | 어떤 변수가 대여량에 영향을 미치는가? | 내일/다음 주 대여량은? |
        | **R-squared** | {bikeResult.r_squared:.4f} | — |
        | **예측 모델** | OLS (6개 독립변수) | {forecastResult.model} |
        | **활용** | 정책 분석, 원인 규명 | 수요 예측, 자원 배분 |
        | **한계** | 미래 독립변수 필요 | 외부 요인 미반영 |

        **결론**: 두 방법은 상호보완적입니다.
        - **왜** 대여량이 변하는지 알려면 **회귀분석**
        - **얼마나** 빌려질지 알려면 **시계열 예측**
        """
    )


@app.cell
def footer(mo):
    return mo.md(
        """
        ---

        > **이 분석은 교육 목적이며, 실제 투자/사업 결정에 사용하지 마세요.**

        - 데이터 출처: [UCI ML Repository — Seoul Bike Sharing](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand), [FRED](https://fred.stlouisfed.org/)
        - 분석 도구: [Vectrix](https://github.com/eddmpython/vectrix) — 순수 NumPy/SciPy 기반 시계열 예측 + 회귀분석 라이브러리
        - 인터랙티브 노트북: [marimo](https://marimo.io/)
        """
    )


if __name__ == "__main__":
    app.run()
