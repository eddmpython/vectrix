---
title: 비즈니스 인텔리전스
---

# 비즈니스 인텔리전스

예측을 넘어 -- 의사결정을 위한 도구들.

## 이상치 탐지

```python
from vectrix.business import AnomalyDetector

detector = AnomalyDetector()
result = detector.detect(data, method="auto")

print(f"방법: {result.method}")
print(f"이상치: {result.nAnomalies}개")
print(f"비율: {result.anomalyRatio:.1%}")
print(f"위치: {result.indices}")
```

방법: `auto`, `zscore`, `iqr`, `rolling`

## What-If 분석

```python
from vectrix.business import WhatIfAnalyzer

analyzer = WhatIfAnalyzer()
results = analyzer.analyze(base_predictions, historical_data, [
    {"name": "낙관적", "trendChange": 0.1},
    {"name": "비관적", "trendChange": -0.15},
    {"name": "충격", "shockAt": 10, "shockMagnitude": -0.3, "shockDuration": 5},
    {"name": "수준 상승", "levelShift": 0.05},
])

for sr in results:
    print(f"{sr.name}: 평균={sr.predictions.mean():.2f}, 영향={sr.impact:+.1%}")
```

## 백테스팅

Walk-forward 검증:

```python
from vectrix.business import Backtester

bt = Backtester(nFolds=5, horizon=14, strategy='expanding')
result = bt.run(data, model_function)

print(f"평균 MAPE: {result.avgMAPE:.2f}%")
print(f"평균 RMSE: {result.avgRMSE:.2f}")

for f in result.folds:
    print(f"  Fold {f.fold}: MAPE={f.mape:.2f}%")
```

전략: `expanding`, `sliding`

## 비즈니스 지표

```python
from vectrix.business import BusinessMetrics

metrics = BusinessMetrics()
result = metrics.calculate(actual, predicted)

print(f"Bias: {result['bias']:+.2f}")
print(f"WAPE: {result['wape']:.2f}%")
print(f"MASE: {result['mase']:.2f}")
print(f"정확도: {result['forecastAccuracy']:.1f}%")
```

| 지표 | 키 | 설명 |
|------|-----|------|
| Bias | `bias` | 양수 = 과대예측 |
| WAPE | `wape` | 가중 절대 백분율 오차 |
| MASE | `mase` | 1 미만이면 Naive보다 우수 |
| 정확도 | `forecastAccuracy` | 높을수록 좋음 |

---

**인터랙티브 튜토리얼:** `marimo run docs/tutorials/ko/06_business.py`
