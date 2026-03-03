---
title: 다변량 예측
---

# 다변량 예측

Vectrix는 벡터 자기회귀(VAR)와 벡터 오차 수정 모델(VECM)을 통해 다변량 시계열 예측을 지원합니다.

## VAR 모델

VAR 모델은 여러 시계열 간의 선형 상호의존성을 포착합니다. 각 변수는 자신의 과거값과 다른 모든 변수의 과거값의 선형 결합으로 모델링됩니다.

```python
from vectrix.engine.var import VARModel
import numpy as np

# Y shape: (T, k) -- T개 시점, k개 변수
Y = np.column_stack([sales, advertising, inventory])

model = VARModel(maxLag=5, criterion="aic")
model.fit(Y)

predictions, lower, upper = model.predict(steps=12)
# predictions shape: (12, 3) -- 12개 시점, 3개 변수
```

### 자동 시차 선택

모델은 1부터 `maxLag`까지 모든 시차 차수를 테스트하고 정보 기준에 따라 최적 차수를 선택합니다

```python
model = VARModel(maxLag=8, criterion="aic")   # Akaike IC (기본)
model = VARModel(maxLag=8, criterion="bic")   # Bayesian IC (더 희소)

model.fit(Y)
print(f"선택된 시차 차수: {model.order}")
```

### 그랜저 인과성

한 변수가 다른 변수의 예측에 도움이 되는지 검정합니다

```python
model = VARModel()
result = model.grangerCausality(Y, cause=0, effect=1, maxLag=5)

print(f"F-통계량: {result['fStat']:.3f}")
print(f"p-값: {result['pValue']:.4f}")

if result['pValue'] < 0.05:
    print("변수 0이 변수 1을 그랜저 인과합니다")
```

## VECM 모델

벡터 오차 수정 모델은 변수들이 **공적분** 관계에 있을 때 사용합니다 -- 개별적으로는 비정상이지만 장기 균형 관계를 공유합니다.

```python
from vectrix.engine.var import VECMModel

# 공적분 시리즈 예시: 관련 자산의 가격
prices = np.column_stack([stock_a, stock_b])

model = VECMModel(maxLag=4)
model.fit(prices)

predictions, lower, upper = model.predict(steps=10)
```

### VECM vs VAR 선택 기준

| 데이터 특성 | 권장 모델 |
|:--|:--|
| 정상 시계열 | VAR |
| 비정상이지만 공적분 관계 | VECM |
| 비정상, 공적분 없음 | 차분된 데이터에 VAR |
| 불확실 | 둘 다 시도 후 비교 |

### 공적분 랭크

VECM은 공적분 랭크(독립적인 공적분 관계의 수)를 자동으로 추정합니다

```python
model = VECMModel(maxLag=4, rank=None)  # 자동 탐지
model = VECMModel(maxLag=4, rank=1)     # 랭크 1 고정

model.fit(Y)
print(f"공적분 랭크: {model._rank}")
```

## 전체 예제

```python
import numpy as np
from vectrix.engine.var import VARModel

rng = np.random.default_rng(42)
T = 200

# 두 개의 관련 시리즈 생성
Y = np.zeros((T, 2))
for t in range(1, T):
    Y[t, 0] = 0.5 * Y[t-1, 0] + 0.2 * Y[t-1, 1] + rng.normal(0, 0.1)
    Y[t, 1] = 0.3 * Y[t-1, 0] + 0.4 * Y[t-1, 1] + rng.normal(0, 0.1)

# 학습 및 예측
model = VARModel(maxLag=5, criterion="bic")
model.fit(Y)

print(f"선택된 차수: {model.order}")

pred, lo, hi = model.predict(steps=10)
print(f"예측 shape: {pred.shape}")  # (10, 2)

# 그랜저 인과성
gc = model.grangerCausality(Y, cause=0, effect=1, maxLag=3)
print(f"시리즈 0이 시리즈 1을 인과하는가? p={gc['pValue']:.4f}")
```
