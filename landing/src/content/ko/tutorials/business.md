---
title: "튜토리얼 06 — 비즈니스 인텔리전스"
---

# 튜토리얼 06 — 비즈니스 인텔리전스

예측 그 이상 -- 의사결정을 위한 도구입니다. 이상치 감지, 시나리오 분석, 백테스팅, 비즈니스 지표까지 실무에 필요한 모든 기능을 제공합니다.

## 이상치 감지

시계열에서 비정상적인 패턴을 자동으로 식별합니다

```python
import numpy as np
from vectrix.business import AnomalyDetector

data = np.random.normal(100, 10, 365)
data[50] = 250
data[200] = 20

detector = AnomalyDetector()
result = detector.detect(data, method="auto")

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

조건 변화가 예측에 미치는 영향을 시뮬레이션합니다

```python
from vectrix import forecast
from vectrix.business import WhatIfAnalyzer

result = forecast(data, steps=30)

analyzer = WhatIfAnalyzer()
scenarios = analyzer.analyze(result.predictions, data, [
    {"name": "낙관적", "trendChange": 0.1},
    {"name": "비관적", "trendChange": -0.15},
    {"name": "충격", "shockAt": 10, "shockMagnitude": -0.3, "shockDuration": 5},
    {"name": "수준 이동", "levelShift": 0.05},
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

워크포워드 교차 검증으로 예측 정확도를 평가합니다

```python
from vectrix.business import Backtester

bt = Backtester(nFolds=5, horizon=14, strategy='expanding')
result = bt.run(data, modelFunction)

print(f"평균 MAPE: {result.avgMAPE:.2f}%")
print(f"평균 RMSE: {result.avgRMSE:.2f}")
print(f"최고 폴드: #{result.bestFold}")
print(f"최악 폴드: #{result.worstFold}")

for f in result.folds:
    print(f"  폴드 {f.fold}: MAPE={f.mape:.2f}%")
```

**예상 출력:**

```
평균 MAPE: 5.23%
평균 RMSE: 12.45
최고 폴드: #3
최악 폴드: #1

  폴드 1: MAPE=7.12%
  폴드 2: MAPE=5.45%
  폴드 3: MAPE=3.89%
  폴드 4: MAPE=4.67%
  폴드 5: MAPE=5.02%
```

### 백테스팅 전략

| 전략 | 설명 |
|------|------|
| `expanding` | 원점을 앞으로 이동하며 학습 데이터 누적 확장 |
| `sliding` | 고정 크기 이동 창으로 학습 |

## 비즈니스 지표

통계적 정확도를 비즈니스 관점의 KPI로 변환합니다

```python
from vectrix.business import BusinessMetrics

metrics = BusinessMetrics()
result = metrics.calculate(actual, predicted)

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

전체 파이프라인을 하나로 연결합니다

```python
import pandas as pd
from vectrix import forecast
from vectrix.business import AnomalyDetector, Backtester, BusinessMetrics

df = pd.read_csv("daily_sales.csv")
data = df["sales"].values

detector = AnomalyDetector()
anomalies = detector.detect(data, method="auto")
print(f"1단계 - 이상치: {anomalies.nAnomalies}개 감지")

result = forecast(data, steps=30)
print(f"2단계 - 예측: {result.model} 선택됨")

backtester = Backtester()
btResult = backtester.run(data, horizon=30, nWindows=5)
print(f"3단계 - 백테스트: MAE={btResult.meanMae:.2f}, MAPE={btResult.meanMape:.2f}%")

metrics = BusinessMetrics()
bizMetrics = metrics.calculate(
    actual=data[-30:],
    forecast=result.predictions[:30],
)
print(f"4단계 - 비즈니스 영향: WAPE={bizMetrics['wape']:.2f}%")
```

> **참고:** 모든 비즈니스 인텔리전스 도구는 원시 numpy 배열, pandas Series, DataFrame을 지원합니다. 특별한 데이터 형식이 필요하지 않습니다.

---
