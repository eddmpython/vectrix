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
        # Vectrix Quickstart

        **3분 만에 첫 예측을 만들어보세요.**

        Vectrix는 설정 없이 바로 사용할 수 있는 시계열 예측 라이브러리입니다.
        30+ 모델을 자동으로 비교하고, 최적의 모델을 선택합니다.

        ```
        pip install vectrix
        ```
        """
    )
    return


@app.cell
def imports():
    import numpy as np
    import pandas as pd
    from vectrix import forecast
    return np, pd, forecast


@app.cell
def section1(mo):
    mo.md(
        """
        ---
        ## 1. 리스트로 바로 예측

        데이터가 숫자 리스트만 있어도 됩니다.
        날짜, 컬럼명, 모델 선택 — 전부 자동입니다.
        """
    )
    return


@app.cell
def forecastFromList(forecast):
    salesData = [
        120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
        172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
        245, 268, 258, 280, 270, 295, 285, 310, 300, 325,
        180, 200, 215, 195, 220, 235, 210, 245, 230, 260,
    ]

    result = forecast(salesData, steps=10)
    return result,


@app.cell
def showResult(mo, result):
    mo.md(
        f"""
        ### 결과 확인

        | 항목 | 값 |
        |------|-----|
        | 선택된 모델 | `{result.model}` |
        | 예측 기간 | {len(result.predictions)}일 |
        | 예측 범위 | {result.predictions.min():.1f} ~ {result.predictions.max():.1f} |
        """
    )
    return


@app.cell
def showDataframe(mo, result):
    df = result.to_dataframe()
    mo.md("### 예측 결과 테이블")
    return df,


@app.cell
def displayTable(mo, df):
    return mo.ui.table(df)


@app.cell
def section2(mo):
    mo.md(
        """
        ---
        ## 2. DataFrame에서 예측

        pandas DataFrame이 있으면 날짜와 값 컬럼을 자동으로 감지합니다.
        """
    )
    return


@app.cell
def forecastFromDf(np, pd, forecast):
    np.random.seed(42)
    n = 120
    t = np.arange(n, dtype=np.float64)
    trend = 100 + 0.5 * t
    seasonal = 20 * np.sin(2 * np.pi * t / 12)
    noise = np.random.normal(0, 5, n)

    monthlyDf = pd.DataFrame({
        "date": pd.date_range("2015-01-01", periods=n, freq="MS"),
        "sales": trend + seasonal + noise,
    })

    dfResult = forecast(monthlyDf, steps=12)
    return monthlyDf, dfResult


@app.cell
def showDfResult(mo, dfResult):
    mo.md(
        f"""
        ### DataFrame 예측 결과

        - 모델: `{dfResult.model}`
        - 12개월 예측 생성 완료

        `.summary()` 로 전체 요약을 확인할 수 있습니다:
        """
    )
    return


@app.cell
def printSummary(mo, dfResult):
    mo.md(f"```\n{dfResult.summary()}\n```")
    return


@app.cell
def section3(mo):
    mo.md(
        """
        ---
        ## 3. 예측 기간 조절

        슬라이더로 예측 기간(steps)을 바꿔보세요.
        """
    )
    return


@app.cell
def stepsSlider(mo):
    stepsControl = mo.ui.slider(
        start=5, stop=60, step=5, value=15,
        label="예측 기간 (steps)"
    )
    return stepsControl,


@app.cell
def showSlider(stepsControl):
    stepsControl
    return


@app.cell
def interactiveForecast(np, pd, forecast, stepsControl):
    np.random.seed(42)
    n = 200
    t = np.arange(n, dtype=np.float64)
    values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

    interDf = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "value": values,
    })

    interResult = forecast(interDf, steps=stepsControl.value)
    return interDf, interResult


@app.cell
def showInteractive(mo, interResult, stepsControl):
    predDf = interResult.to_dataframe()
    mo.md(
        f"""
        **{stepsControl.value}일 예측** | 모델: `{interResult.model}`

        평균 예측값: {interResult.predictions.mean():.1f}
        """
    )
    return predDf,


@app.cell
def section4(mo):
    mo.md(
        """
        ---
        ## 4. 결과 활용

        `EasyForecastResult` 객체의 다양한 메서드:
        """
    )
    return


@app.cell
def resultMethods(mo, interResult):
    forecastDf = interResult.to_dataframe()
    jsonStr = interResult.to_json()

    mo.md(
        f"""
        | 메서드 | 설명 | 예시 |
        |--------|------|------|
        | `.predictions` | 예측값 배열 | `[{interResult.predictions[0]:.1f}, {interResult.predictions[1]:.1f}, ...]` |
        | `.dates` | 예측 날짜 | `[{interResult.dates[0]}, ...]` |
        | `.lower` | 95% 하한 | `[{interResult.lower[0]:.1f}, ...]` |
        | `.upper` | 95% 상한 | `[{interResult.upper[0]:.1f}, ...]` |
        | `.model` | 선택된 모델 | `{interResult.model}` |
        | `.to_dataframe()` | DataFrame 변환 | {len(forecastDf)} rows |
        | `.to_json()` | JSON 변환 | {len(jsonStr)} chars |
        | `.summary()` | 텍스트 요약 | 위 참조 |
        """
    )
    return


@app.cell
def section5(mo):
    mo.md(
        """
        ---
        ## 5. 다양한 입력 형식

        Vectrix는 거의 모든 형태의 데이터를 받습니다:

        ```python
        forecast([1, 2, 3, 4, 5])                    # 리스트
        forecast(np.array([1, 2, 3, 4, 5]))           # numpy 배열
        forecast(pd.Series([1, 2, 3, 4, 5]))          # pandas Series
        forecast({"value": [1, 2, 3, 4, 5]})          # dict
        forecast(df, date="날짜", value="매출")         # DataFrame
        forecast("data.csv")                           # CSV 파일 경로
        ```

        **다음 튜토리얼:** `02_analyze.py` — 시계열 DNA 분석
        """
    )
    return


if __name__ == "__main__":
    app.run()
