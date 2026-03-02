# 쇼케이스 01 — 한국 경제 데이터 예측

**공개 데이터로 실제 한국 경제 지표를 예측합니다.**

## 개요

FRED(미국 연방준비은행 경제 데이터)에서 제공하는 한국 경제 데이터로 Vectrix 예측을 시연합니다. API 키 불필요 — 모든 데이터가 자동으로 수집됩니다.

### 주요 내용

- **원/달러 환율** — 1981년부터 월간 데이터, 12개월 예측
- **KOSPI 주가지수** — 월간 주가지수, 12개월 예측
- **소비자물가지수 (CPI)** — 인플레이션 추적, 12개월 예측
- **다중 지표 DNA 분석** — 지표별 난이도와 특성 비교

## 인터랙티브 실행

```bash
pip install vectrix pandas marimo
marimo run docs/showcase/ko/01_koreanEconomy.py
```

## 코드

```python
import pandas as pd
from vectrix import forecast

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=EXKOUS"
df = pd.read_csv(url)
df.columns = ["date", "value"]
df["date"] = pd.to_datetime(df["date"])
df = df.dropna()

result = forecast(df, date="date", value="value", steps=12)
print(f"모델: {result.model}")
print(f"평균 예측: {result.predictions.mean():,.1f} 원/달러")
print(f"95% 구간: {result.lower.min():,.1f} ~ {result.upper.max():,.1f}")
```

### 예상 출력

```
모델: AutoETS
평균 예측: 1,380.5 원/달러
95% 구간: 1,250.3 ~ 1,510.8
```

선택된 모델과 예측값은 FRED에서 제공하는 최신 데이터에 따라 달라집니다.

## 예측 요약

```python
print(result.summary())
```

요약에는 선택된 모델, 정확도 지표, 각 예측 스텝의 예측 구간이 포함됩니다.

## 데이터 출처

| 출처 | 시리즈 | URL |
|------|--------|-----|
| FRED | EXKOUS (원/달러) | `fred.stlouisfed.org/series/EXKOUS` |
| FRED | KOSPI | `fred.stlouisfed.org/series/SPASTT01KRM661N` |
| FRED | CPI 한국 | `fred.stlouisfed.org/series/KORCPIALLMINMEI` |

!!! note "주의사항"
    이 분석은 교육 목적이며, 실제 투자나 사업 결정에 사용하지 마세요.

---

**다음:** [쇼케이스 02 — 한국 데이터 회귀분석](02_koreanRegression.md)
