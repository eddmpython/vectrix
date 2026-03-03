---
title: 예측
---

# 예측

## Easy API

가장 간단한 예측 방법

```python
from vectrix import forecast

result = forecast(data, steps=30)
```

`forecast()`는 리스트, numpy 배열, pandas DataFrame, Series, dict, CSV 파일 경로를 받습니다. 30+ 후보 모델 중 최적의 모델을 자동으로 선택합니다.

## Vectrix 클래스

더 세밀한 제어가 필요할 때 `Vectrix` 클래스를 직접 사용합니다

```python
from vectrix import Vectrix

vx = Vectrix()
result = vx.forecast(
    df,
    dateCol="date",
    valueCol="sales",
    steps=14,
    trainRatio=0.8
)

print(result.bestModelName)
print(result.predictions)
```

### 전체 모델 결과

```python
for modelId, mr in result.allModelResults.items():
    print(f"{mr.modelName}: MAPE={mr.mape:.2f}%")
```

## 모델 카테고리

| 카테고리 | 모델 | 최적 대상 |
|----------|------|-----------|
| **지수평활** | AutoETS, ETS 변형 | 안정적 패턴 |
| **ARIMA** | AutoARIMA | 정상 시계열 |
| **분해** | MSTL, AutoMSTL | 다중 계절성 |
| **Theta** | Theta, DOT | 범용 |
| **삼각함수** | TBATS | 복잡한 계절성 |
| **복소수** | AutoCES | 비선형 패턴 |
| **간헐적** | Croston, SBA, TSB | 희소 수요 |
| **변동성** | GARCH, EGARCH, GJR | 금융 데이터 |
| **기준선** | Naive, Seasonal, Mean, RWD | 벤치마크 |

## Flat Defense 시스템

Vectrix만의 4단계 평탄 예측 방어 시스템

1. **FlatRiskDiagnostic** -- 평탄 예측 위험도 사전 평가
2. **AdaptiveModelSelector** -- 위험도 기반 모델 선택
3. **FlatPredictionDetector** -- 사후 평탄 감지
4. **FlatPredictionCorrector** -- 평탄 예측 자동 교정

```python
result = vx.forecast(df, dateCol="date", valueCol="value", steps=14)
fr = result.flatRisk
print(f"위험도: {fr.riskLevel.name} ({fr.riskScore:.2f})")
print(f"전략: {fr.recommendedStrategy}")
```

## 엔진 직접 접근

개별 모델을 직접 사용

```python
from vectrix.engine import AutoETS, AutoARIMA, ThetaModel

ets = AutoETS(period=7)
ets.fit(data)
predictions, lower, upper = ets.predict(steps=30)
```

---

**인터랙티브 튜토리얼:** `marimo run docs/tutorials/ko/04_models.py`
