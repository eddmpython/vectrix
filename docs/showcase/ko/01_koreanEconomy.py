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
    from vectrix import forecast

    return forecast, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # 원/달러 환율 예측

    FRED에서 **한국 원/달러 환율(EXKOUS)** 월간 데이터를 가져와
    Vectrix로 향후 12개월을 예측합니다.
    """)
    return


@app.cell
def _(pd):
    _url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
    df = pd.read_csv(_url)
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    df
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    | 항목 | 값 |
    |------|-----|
    | 기간 | {df['date'].iloc[0].strftime('%Y-%m')} ~ {df['date'].iloc[-1].strftime('%Y-%m')} |
    | 데이터 수 | {len(df):,}개 |
    | 최근 환율 | {df['value'].iloc[-1]:,.1f} 원/달러 |
    """)
    return


@app.cell
def _(df, forecast):
    result = forecast(df, date="date", value="value", steps=12)
    result
    return (result,)


@app.cell
def _(mo, result):
    mo.md(f"""
    ## 예측 결과

    | 항목 | 값 |
    |------|-----|
    | 선택된 모델 | `{result.model}` |
    | 예측 평균 | {result.predictions.mean():,.1f} 원 |
    | 95% 신뢰구간 | {result.lower.min():,.1f} ~ {result.upper.max():,.1f} 원 |
    """)
    return


@app.cell
def _(mo, result):
    mo.md(f"""
    ```\n{result.summary()}\n```
    """)
    return


@app.cell
def _(result):
    result.to_dataframe()
    return


@app.cell
def _(mo):
    mo.md("""
    > **면책 조항**: 교육 목적이며, 투자 결정에 사용하지 마세요.
    """)
    return


if __name__ == "__main__":
    app.run()
