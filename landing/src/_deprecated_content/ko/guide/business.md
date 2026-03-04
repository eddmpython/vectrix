---
title: 비즈니스 인텔리전스
---

# 비즈니스 인텔리전스

**예측은 첫 번째 단계일 뿐입니다.** 실제 의사결정에는 이상치 탐지로 데이터를 정제하고, What-If 시나리오로 계획을 수립하고, 백테스팅으로 접근법을 검증하고, MAPE를 넘어선 비즈니스 특화 정확도 지표가 필요합니다.

Vectrix의 비즈니스 인텔리전스 모듈은 이 네 가지를 모두 제공합니다 — 운영 관리자, 분석가, 데이터 사이언티스트를 위한 프로덕션 수준의 예측 워크플로우.

## 이상치 탐지

예측하기 전에, 과거 데이터의 비정상적인 관측값을 식별하고 이해하세요. 이상치는 모델 학습을 왜곡하고 편향된 예측을 초래할 수 있습니다:

```python
from vectrix.business import AnomalyDetector

detector = AnomalyDetector()
result = detector.detect(data, method="auto", threshold=3.0, period=1)

print(f"방법: {result.method}")
print(f"이상치: {result.nAnomalies}개")
print(f"비율: {result.anomalyRatio:.1%}")
print(f"위치: {result.indices}")
```

방법: `auto`, `zscore`, `iqr`, `rolling`

## What-If 분석

기본 예측에 대해 가설적 시나리오를 탐색합니다 — 예산 계획, 리스크 평가, 이해관계자 보고에 필수적입니다. 낙관적, 비관적, 충격 시나리오를 정의하고 영향을 비교하세요:

```python
from vectrix.business import WhatIfAnalyzer

analyzer = WhatIfAnalyzer()
results = analyzer.analyze(base_predictions, historical_data, [
    {"name": "낙관적", "trend_change": 0.1},
    {"name": "비관적", "trend_change": -0.15},
    {"name": "충격", "shock_at": 10, "shock_magnitude": -0.3, "shock_duration": 5},
    {"name": "수준 상승", "level_shift": 0.05},
])

for sr in results:
    print(f"{sr.name}: 평균={sr.predictions.mean():.2f}, 영향={sr.impact:+.1f}%")
```

## 백테스팅

예측 접근법이 실제로 작동하는지 어떻게 확인할까요? **백테스팅**(Walk-forward 검증)은 과거 데이터를 반복적으로 학습하고 다음 구간을 예측하여, 역사적으로 모델이 얼마나 잘 작동했을지 시뮬레이션합니다:

```python
from vectrix.business import Backtester

from vectrix.engine.ets import AutoETS

bt = Backtester(nFolds=5)
result = bt.run(data, lambda: AutoETS())

print(f"평균 MAPE: {result.avgMAPE:.2f}%")
print(f"평균 RMSE: {result.avgRMSE:.2f}")

for f in result.folds:
    print(f"  Fold {f.fold}: MAPE={f.mape:.2f}%")
```

## 비즈니스 지표

MAPE와 RMSE는 통계적 정확도를 알려주지만, 비즈니스는 다른 것을 중요시합니다: **체계적으로 과대/과소 예측하고 있는가? 볼륨 가중 오차는 얼마인가? Naive 기준선을 이기고 있는가?** `BusinessMetrics`가 이 질문에 답합니다:

```python
from vectrix.business import BusinessMetrics

result = BusinessMetrics().calculate(actual, predicted)

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
