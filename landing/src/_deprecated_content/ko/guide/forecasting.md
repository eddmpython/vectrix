---
title: 예측
---

# 예측

Vectrix는 두 가지 예측 API를 제공합니다: 한 줄로 사용하는 **Easy API**와 전체 파이프라인을 세밀하게 제어하는 **Vectrix 클래스**.

## Easy API

가장 간단한 예측 방법. 한 번의 함수 호출로 30개 이상의 모델을 평가하고, 최적 모델을 선택하고, 95% 신뢰구간과 함께 예측값을 반환합니다:

```python
from vectrix import forecast

result = forecast(data, steps=30)
```

`forecast()`는 리스트, NumPy 배열, pandas DataFrame, Series, CSV 파일 경로를 받습니다. 날짜/값 컬럼을 자동 감지하고, 데이터를 분할하여 검증하고, 검증 오차가 가장 낮은 모델을 선택합니다.

## Vectrix 클래스

모든 모델 결과, 평탄 위험 진단, 앙상블 가중치, 개별 모델 메트릭에 완전히 접근하려면 `Vectrix` 클래스를 직접 사용합니다:

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

Vectrix는 10개 카테고리의 모델을 평가합니다. 각 카테고리는 시계열의 서로 다른 동적 특성을 포착합니다:

| 카테고리 | 모델 | 최적 대상 |
|----------|------|-----------|
| **지수평활** | AutoETS, ETS 변형 | 추세와 계절성이 있는 안정적 패턴 |
| **ARIMA** | AutoARIMA | 정상 및 차분 시계열 |
| **분해** | MSTL, AutoMSTL | 다중 계절 주기 (일간 + 주간 + 연간) |
| **Theta** | Theta, DOT, 4Theta | 범용 — DOT가 단일 모델 중 가장 강력 |
| **복소수 ES** | AutoCES | 비선형 및 복잡한 동태 |
| **삼각함수** | TBATS | 비정수 주기의 복잡한 다중 계절성 |
| **간헐적** | Croston, SBA, TSB | 0이 많은 희소 수요 데이터 |
| **변동성** | GARCH, EGARCH, GJR | 시변 분산이 있는 금융 데이터 |
| **신경망/저장소** | ESN, DTSF | 비선형 동태, 패턴 매칭 |
| **기준선** | Naive, Seasonal, Mean, RWD | 벤치마크 — 이것도 못 이기면 문제 |

## Flat Defense 시스템

통계 예측의 흔한 실패 모드는 평탄(상수) 예측입니다. Vectrix는 이를 자동으로 감지하고 교정하는 고유한 4단계 방어 시스템을 포함합니다:

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

세밀한 제어가 필요할 때 개별 모델 엔진을 직접 사용합니다. 모든 엔진은 동일한 `fit()` → `predict()` 인터페이스를 따릅니다:

```python
from vectrix.engine.ets import AutoETS
from vectrix.engine.arima import AutoARIMA
from vectrix.engine.theta import OptimizedTheta

ets = AutoETS(period=7)
ets.fit(data)
predictions, lower, upper = ets.predict(steps=30)
```

---

**인터랙티브 튜토리얼:** `marimo run docs/tutorials/ko/04_models.py`
