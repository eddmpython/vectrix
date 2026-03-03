---
title: "비즈니스 인텔리전스"
---

# 비즈니스 인텔리전스

엔드투엔드 비즈니스 워크플로우: 이상치 감지, 미래 예측, 시나리오 분석, 모델 백테스팅, 비즈니스 영향 측정까지 한 번에 처리합니다.

## 이상치 감지

시계열에서 비정상적인 패턴을 자동으로 식별합니다.

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

| 방법 | 설명 |
|------|------|
| `auto` | 최적 방법을 자동 선택 |
| `zscore` | Z-score 임계값 (기본 3.0) |
| `iqr` | 사분위수 범위 방법 |
| `rolling` | 이동 창 편차 |

## 예측

비즈니스 분석의 기준점이 되는 기본 예측을 생성합니다

```python
from vectrix import forecast

result = forecast(data, steps=30)
print(f"모델: {result.model}")
print(f"향후 30일 평균: {result.predictions.mean():.1f}")
```

## 시나리오 분석 (What-If)

조건 변화가 예측에 미치는 영향을 시뮬레이션합니다.

```python
from vectrix.business import WhatIfAnalyzer

analyzer = WhatIfAnalyzer()

scenarios = [
    {"name": "기본", "adjustment": 1.0},
    {"name": "10% 성장", "adjustment": 1.10},
    {"name": "20% 하락", "adjustment": 0.80},
    {"name": "충격 이벤트", "adjustment": 0.50},
]

for scenario in scenarios:
    adjusted = result.predictions * scenario["adjustment"]
    print(f"{scenario['name']:16s}  평균={adjusted.mean():8.1f}")
```

## 백테스팅

롤링 오리진 교차 검증으로 예측 정확도를 평가합니다.

```python
from vectrix.business import Backtester

backtester = Backtester()
btResult = backtester.run(
    data=data,
    horizon=12,
    nWindows=5,
)

print(f"평균 MAE:   {btResult.meanMae:.2f}")
print(f"평균 RMSE:  {btResult.meanRmse:.2f}")
print(f"평균 MAPE:  {btResult.meanMape:.2f}%")
print(f"평균 sMAPE: {btResult.meanSmape:.2f}%")
```

백테스팅은 원점을 앞으로 이동시키며 여러 학습/테스트 분할을 생성하여, 표본 외 성능의 현실적인 추정치를 제공합니다.

## 비즈니스 지표

통계적 정확도를 비즈니스 관점의 KPI로 변환합니다.

```python
from vectrix.business import BusinessMetrics

metrics = BusinessMetrics()
result = metrics.calculate(
    actual=actualSales,
    forecast=forecastedSales,
)

print(f"예측 편향: {result['bias']:+.2f}")
print(f"WAPE: {result['wape']:.2f}%")
print(f"MASE: {result['mase']:.2f}")
print(f"예측 정확도: {result['forecastAccuracy']:.1f}%")
```

## 전체 비즈니스 워크플로우

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
