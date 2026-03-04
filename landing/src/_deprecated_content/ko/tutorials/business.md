---
title: "튜토리얼 06 — 비즈니스 인텔리전스"
---

# 튜토리얼 06 — 비즈니스 인텔리전스

**예측은 첫 번째 단계일 뿐입니다.** 실제 의사결정에는 예측값 이상의 것이 필요합니다 — 과거 데이터의 이상값은 무엇이었는지, 조건이 바뀌면 예측이 어떻게 달라지는지, 모델이 과거에 얼마나 정확했는지, 통계적 오차가 비즈니스 관점에서 어떤 의미인지. Vectrix의 비즈니스 인텔리전스 도구는 이 모든 질문에 답합니다.

## 이상치 감지

예측하기 전에, 과거 데이터의 이상값을 식별하고 이해하세요. 이상치는 예측 모델의 학습을 왜곡하고, 결과의 신뢰성을 떨어뜨릴 수 있습니다. Vectrix는 Z-score, IQR, 이동 창 방법 등 다양한 감지 알고리즘을 제공하며, 데이터에 따라 최적 방법을 자동으로 선택합니다.

```python
import numpy as np
from vectrix.business import AnomalyDetector

data = np.random.normal(100, 10, 365)
data[50] = 250
data[200] = 20

detector = AnomalyDetector()
result = detector.detect(data, method="auto", threshold=3.0, period=1)

print(f"감지 방법: {result.method}")
print(f"이상치 수: {result.nAnomalies}")
print(f"이상치 비율: {result.anomalyRatio:.1%}")
print(f"이상치 위치: {result.indices}")
```

**예상 출력:**

```
감지 방법: zscore
이상치 수: 2
이상치 비율: 0.5%
이상치 위치: [50, 200]
```

### 감지 방법

| 방법 | 설명 |
|------|------|
| `auto` | 최적 방법을 자동 선택 |
| `zscore` | Z-score 임계값 (기본 3.0) |
| `iqr` | 사분위수 범위 방법 |
| `rolling` | 이동 창 편차 |

## 시나리오 분석 (What-If)

"광고비를 10% 늘리면?", "경기 침체가 오면?", "경쟁사가 가격을 인하하면?" — 시나리오 분석은 이러한 가정을 예측에 반영하여 조건별 미래를 시뮬레이션합니다. 예산 계획, 리스크 평가, 이해관계자 보고에 필수적인 도구입니다.

```python
from vectrix import forecast
from vectrix.business import WhatIfAnalyzer

result = forecast(data, steps=30)

analyzer = WhatIfAnalyzer()
scenarios = analyzer.analyze(result.predictions, data, [
    {"name": "낙관적", "trend_change": 0.1},
    {"name": "비관적", "trend_change": -0.15},
    {"name": "충격", "shock_at": 10, "shock_magnitude": -0.3, "shock_duration": 5},
    {"name": "수준 이동", "level_shift": 0.05},
])

for sr in scenarios:
    print(f"{sr.name:12s}: 평균={sr.predictions.mean():.2f}, 영향={sr.impact:+.1%}")
```

**예상 출력:**

```
낙관적      : 평균=115.32, 영향=+10.0%
비관적      : 평균=89.18, 영향=-15.0%
충격        : 평균=96.45, 영향=-8.1%
수준 이동   : 평균=110.05, 영향=+5.0%
```

## 백테스팅

모델이 미래에 얼마나 정확할지 알 수 있는 가장 신뢰할 수 있는 방법은 과거에 얼마나 정확했는지 측정하는 것입니다. 백테스팅은 워크포워드(walk-forward) 교차 검증을 수행합니다: 데이터를 시간 순서대로 여러 구간(fold)으로 나누고, 각 구간에서 과거 데이터만으로 모델을 학습한 뒤 직후 미래를 예측하여 실제 값과 비교합니다. 이 과정을 반복하면 모델의 일관된 성능을 평가할 수 있습니다.

```python
from vectrix.business import Backtester

from vectrix.engine.ets import AutoETS

bt = Backtester(nFolds=5)
result = bt.run(data, lambda: AutoETS())

print(f"평균 MAPE: {result.avgMAPE:.2f}%")
print(f"평균 RMSE: {result.avgRMSE:.2f}")

for f in result.folds:
    print(f"  폴드: MAPE={f.mape:.2f}%")
```

**예상 출력:**

```
평균 MAPE: 5.23%
평균 RMSE: 12.45

  폴드: MAPE=7.12%
  폴드: MAPE=5.45%
  폴드: MAPE=3.89%
  폴드: MAPE=4.67%
  폴드: MAPE=5.02%
```

## 비즈니스 지표

MAPE와 RMSE는 통계적으로 유용하지만, 비즈니스 의사결정자에게는 직관적이지 않습니다. "예측 정확도가 91.6%이고, 과대 예측 경향이 있다"라는 표현이 더 의미 있습니다. `BusinessMetrics`는 통계적 오차를 비즈니스 관점의 KPI — 편향, WAPE, MASE, 예측 정확도, 과대/과소 예측 비율 — 로 변환합니다.

```python
from vectrix.business import BusinessMetrics

result = BusinessMetrics().calculate(actual, predicted)

print(f"편향: {result['bias']:+.2f}")
print(f"WAPE: {result['wape']:.2f}%")
print(f"MASE: {result['mase']:.2f}")
print(f"예측 정확도: {result['forecastAccuracy']:.1f}%")
```

**예상 출력:**

```
편향: -2.35
WAPE: 8.45%
MASE: 0.87
예측 정확도: 91.6%
```

### 지표 참조

| 지표 | 키 | 설명 |
|------|-----|------|
| 편향 | `bias` | 양수 = 과대 예측 |
| 편향 % | `biasPercent` | 백분율 편향 |
| WAPE | `wape` | 가중 절대 백분율 오차 |
| MASE | `mase` | 1 미만이면 Naive보다 우수 |
| 정확도 | `forecastAccuracy` | 높을수록 우수 |
| 과대 예측 | `overForecastRatio` | 예측이 실제보다 큰 비율 |
| 과소 예측 | `underForecastRatio` | 예측이 실제보다 작은 비율 |

## 완전한 비즈니스 워크플로우

이상치 감지부터 예측, 백테스팅, 비즈니스 지표 산출까지 전체 파이프라인을 하나로 연결합니다.

```python
import pandas as pd
from vectrix import forecast
from vectrix.business import AnomalyDetector, Backtester, BusinessMetrics
from vectrix.engine.ets import AutoETS

df = pd.read_csv("daily_sales.csv")
data = df["sales"].values

detector = AnomalyDetector()
anomalies = detector.detect(data, method="auto", threshold=3.0, period=1)
print(f"1단계 - 이상치: {anomalies.nAnomalies}개 감지")

result = forecast(data, steps=30)
print(f"2단계 - 예측: {result.model} 선택됨")

backtester = Backtester(nFolds=5)
btResult = backtester.run(data, lambda: AutoETS())
print(f"3단계 - 백테스트: MAPE={btResult.avgMAPE:.2f}%, RMSE={btResult.avgRMSE:.2f}")

bizMetrics = BusinessMetrics().calculate(
    data[-30:],
    result.predictions[:30],
)
print(f"4단계 - 비즈니스 영향: WAPE={bizMetrics['wape']:.2f}%")
```

> **참고:** 모든 비즈니스 인텔리전스 도구는 원시 numpy 배열, pandas Series, DataFrame을 지원합니다. 특별한 데이터 형식이 필요하지 않습니다.

---
