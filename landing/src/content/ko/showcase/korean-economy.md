---
title: "한국 경제 예측"
---

# 한국 경제 예측

공개 FRED 데이터를 활용한 한국 경제 지표 예측 쇼케이스입니다. Vectrix가 실제 거시경제 시계열을 단일 함수 호출로 처리하는 방법을 보여줍니다.

## 데이터 출처

모든 데이터는 Federal Reserve Economic Data (FRED) API에서 직접 가져옵니다

| 지표 | FRED 코드 | 빈도 | 설명 |
|------|:---------:|:----:|------|
| 원/달러 환율 | `EXKOUS` | 월간 | 미 달러 대비 원화 환율 |
| KOSPI 지수 | `KOSPI` | 일간 | 한국 종합주가지수 |
| 소비자물가지수 | `KORCPIALLMINMEI` | 월간 | 한국 전체 항목 CPI |

## 원/달러 환율 예측

```python
import pandas as pd
from vectrix import forecast

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
df = pd.read_csv(url)
df.columns = ["date", "value"]
df["date"] = pd.to_datetime(df["date"])
df = df.dropna()

result = forecast(df, date="date", value="value", steps=12)
print(result.model)
print(result.predictions)
print(result.summary())
```

### 결과 상세 확인

```python
print(f"선택된 모델: {result.model}")
print(f"평균 예측값: {result.predictions.mean():,.1f} KRW/USD")
print(f"95% 신뢰구간: {result.lower.min():,.1f} ~ {result.upper.max():,.1f} KRW")

result.to_dataframe()
```

## KOSPI 지수 예측

```python
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=KOSPI"
kospiDf = pd.read_csv(url)
kospiDf.columns = ["date", "value"]
kospiDf["date"] = pd.to_datetime(kospiDf["date"])
kospiDf = kospiDf.dropna()

result = forecast(kospiDf, date="date", value="value", steps=30)
print(result.summary())
```

## CPI 예측

```python
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=KORCPIALLMINMEI"
cpiDf = pd.read_csv(url)
cpiDf.columns = ["date", "value"]
cpiDf["date"] = pd.to_datetime(cpiDf["date"])
cpiDf = cpiDf.dropna()

result = forecast(cpiDf, date="date", value="value", steps=12)
print(result.summary())
```

## 모델 비교

`compare()`로 동일 데이터에 대한 여러 모델의 성능을 비교할 수 있습니다

```python
result = forecast(df, date="date", value="value", steps=12)

comparison = result.compare()
print(comparison)
```

모든 유효 모델의 sMAPE, MAPE, RMSE, MAE를 정확도 순으로 정렬한 DataFrame을 반환합니다.

## 결과 내보내기

```python
result.to_dataframe().to_csv("krw_forecast.csv", index=False)

result.to_json("krw_forecast.json")
```

> **참고:** 이 쇼케이스는 교육 및 시연 목적으로만 제공됩니다. 금융 및 경제 지표 예측은 투자 결정에 사용해서는 안 됩니다. 과거 성과가 미래 결과를 보장하지 않습니다.
