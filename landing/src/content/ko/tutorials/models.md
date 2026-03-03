---
title: "튜토리얼 04 — 30+ 모델"
---

# 튜토리얼 04 — 30+ 모델

Vectrix는 30개 이상의 예측 모델을 내장하고 있습니다. 자동 선택, 수동 지정, 비교 분석까지 모든 워크플로우를 지원합니다.

## 자동 모델 선택

`forecast()` 함수는 모든 호환 모델을 평가하고 최적 모델을 자동 선택합니다

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170, 180, 195], steps=5)
print(f"선택된 모델: {result.model}")
print(f"MAPE: {result.mape:.2f}%")
```

## 모델 비교

`compare()` 메서드로 모든 모델의 성능을 한눈에 볼 수 있습니다

```python
from vectrix import forecast

result = forecast(data, steps=12)
comparison = result.compare()
print(comparison)
```

**예상 출력:**

```
              sMAPE     MAPE     RMSE      MAE
DOT           3.214    3.187   12.453    9.876
AutoCES       3.456    3.421   13.102   10.234
AutoETS       3.789    3.752   14.567   11.345
FourTheta     3.891    3.844   14.892   11.678
...
```

`compare()` 함수를 사용하면 더 간결하게 비교할 수 있습니다

```python
from vectrix import compare

ranking = compare(data, steps=12)
print(ranking)
```

## 모든 모델의 예측값

각 모델이 생성한 예측값을 모두 확인할 수 있습니다

```python
allForecasts = result.all_forecasts()
print(allForecasts)
```

**예상 출력:**

```
   step        DOT    AutoCES    AutoETS  FourTheta  ...
0     1    152.340    153.120    151.890    154.230  ...
1     2    155.670    156.440    154.230    157.560  ...
2     3    158.120    159.890    157.560    160.120  ...
...
```

## Vectrix 클래스 직접 사용

더 세밀한 제어가 필요하면 `Vectrix` 클래스를 직접 사용합니다

```python
from vectrix import Vectrix

vx = Vectrix()
result = vx.forecast(
    df,
    dateCol="date",
    valueCol="sales",
    steps=14,
    period=7,
    trainRatio=0.8
)

print(result.bestModelName)
print(result.predictions)
```

### 개별 모델 결과 확인

```python
for modelId, mr in result.allModelResults.items():
    print(f"{mr.modelName}: MAPE={mr.metrics.mape:.2f}%")
```

## 모델 카테고리

| 카테고리 | 모델 | 적합 대상 |
|---------|------|----------|
| **지수 평활** | AutoETS, ETS 변형 | 안정적인 패턴 |
| **ARIMA** | AutoARIMA | 정상 시계열 |
| **분해** | MSTL, AutoMSTL | 다중 계절성 |
| **Theta** | Theta, DOT | 범용 |
| **삼각함수** | TBATS | 복잡한 계절성 |
| **복소수** | AutoCES | 비선형 패턴 |
| **간헐적 수요** | Croston, SBA, TSB | 희소 수요 |
| **변동성** | GARCH, EGARCH, GJR | 금융 데이터 |
| **신규** | DTSF, ESN, FourTheta | 패턴 매칭, 비선형, 앙상블 |
| **기준선** | Naive, Seasonal, Mean, RWD | 벤치마크 |

## 엔진 직접 접근

개별 모델을 직접 사용할 수 있습니다

```python
from vectrix.engine import AutoETS, AutoARIMA, DynamicOptimizedTheta

ets = AutoETS(period=7)
ets.fit(data)
predictions, lower, upper = ets.predict(steps=30)
print(f"AutoETS 예측: {predictions[:5]}")
```

```python
dot = DynamicOptimizedTheta(period=12)
dot.fit(data)
predictions, lower, upper = dot.predict(steps=12)
print(f"DOT 예측: {predictions[:5]}")
```

## Flat Defense 시스템

Vectrix는 평탄(상수) 예측을 방지하는 고유한 4단계 시스템을 내장하고 있습니다

1. **FlatRiskDiagnostic** -- 평탄 예측 위험 사전 평가
2. **AdaptiveModelSelector** -- 위험 기반 모델 선택
3. **FlatPredictionDetector** -- 예측 후 평탄 감지
4. **FlatPredictionCorrector** -- 평탄 예측 자동 교정

```python
result = vx.forecast(df, dateCol="date", valueCol="value", steps=14)
fr = result.flatRisk
print(f"위험: {fr.riskLevel.name} ({fr.riskScore:.2f})")
print(f"전략: {fr.recommendedStrategy}")
```

## DNA 기반 모델 추천

분석 결과를 활용하여 최적 모델을 선택할 수 있습니다

```python
from vectrix import analyze, forecast

report = analyze(df, date="date", value="sales")
print(f"DNA 난이도: {report.dna.difficulty}")
print(f"추천 모델: {report.dna.recommendedModels[:3]}")

result = forecast(df, date="date", value="sales", steps=12)
print(f"\n최적 모델: {result.model}")
print(f"\n전체 모델 비교:")
print(result.compare())
```

## 완전한 워크플로우

```python
import pandas as pd
from vectrix import forecast, analyze

df = pd.read_csv("monthly_sales.csv")

report = analyze(df, date="date", value="sales")
print(f"카테고리: {report.dna.category}")
print(f"난이도: {report.dna.difficulty}")
print(f"추천: {report.dna.recommendedModels[:3]}")

result = forecast(df, date="date", value="sales", steps=12)
print(f"\n최적 모델: {result.model}")
print(result.compare())
print(result.all_forecasts())

result.to_dataframe().to_csv("forecast_output.csv", index=False)
```

> **참고:** `compare()` 출력은 표준 pandas DataFrame입니다. 정렬, 필터링, 내보내기 등 모든 DataFrame 연산을 사용할 수 있습니다.

---
