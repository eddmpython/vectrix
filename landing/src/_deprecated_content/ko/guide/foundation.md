---
title: 기초 모델
---

# 기초 모델

Vectrix는 최신 사전학습 예측 모델을 위한 선택적 래퍼를 제공합니다. 이 모델들은 **제로샷 예측**을 수행합니다 -- 특정 데이터에 대한 학습이 필요 없습니다.

> **참고:** 기초 모델 래퍼는 추가 패키지가 필요합니다: `pip install "vectrix[foundation]"`

## Amazon Chronos-2

[Chronos](https://github.com/amazon-science/chronos-forecasting)는 Amazon이 개발한 사전학습 확률적 시계열 모델 제품군입니다. 시계열을 구간(bin)으로 토큰화하고 트랜스포머 아키텍처로 예측합니다.

```python
from vectrix import ChronosForecaster

model = ChronosForecaster(
    modelId="amazon/chronos-bolt-small",
    device="cpu",
)

model.fit(y)  # 컨텍스트 저장만 수행 -- 학습 없음
predictions, lower, upper = model.predict(steps=12)
```

### 사용 가능한 모델

| 모델 | 파라미터 | 속도 | 정확도 |
|:--|:--|:--|:--|
| `amazon/chronos-bolt-tiny` | 8M | 가장 빠름 | 양호 |
| `amazon/chronos-bolt-small` | 48M | 빠름 | 우수 |
| `amazon/chronos-bolt-base` | 205M | 보통 | 최고 |

### 분위수 예측

```python
import numpy as np

quantiles = model.predictQuantiles(
    steps=12,
    quantileLevels=[0.1, 0.5, 0.9],
)
# shape: (3, 12) -- 분위수 수준별 한 행
```

### 배치 예측

여러 시계열을 한 번에 예측

```python
series = [y1, y2, y3]
results = model.predictBatch(series, steps=12)
# (predictions, lower, upper) 튜플의 리스트 반환
```

## Google TimesFM 2.5

[TimesFM](https://github.com/google-research/timesfm)은 Google의 시계열 예측 기초 모델로, 최대 2048 컨텍스트 길이를 지원합니다.

```python
from vectrix import TimesFMForecaster

model = TimesFMForecaster(
    modelId="google/timesfm-2.5-200m-pytorch",
)

model.fit(y)
predictions, lower, upper = model.predict(steps=12)
```

### 공변량 활용

TimesFM은 외생 변수(공변량)를 지원합니다

```python
predictions, lower, upper = model.predictWithCovariates(
    steps=12,
    dynamicNumerical=future_numerical_features,  # (steps, n_features)
    dynamicCategorical=future_categorical_features,
)
```

## 가용성 확인

```python
from vectrix import CHRONOS_AVAILABLE, TIMESFM_AVAILABLE

if CHRONOS_AVAILABLE:
    model = ChronosForecaster()
else:
    print("설치: pip install 'vectrix[foundation]'")
```

## 기초 모델 사용 시점

| 시나리오 | 권장 |
|:--|:--|
| 충분한 과거 데이터 (100+ 포인트) | 통계 모델 (기본 Vectrix) |
| 콜드 스타트 / 매우 짧은 시리즈 | 기초 모델 |
| 설명 가능성 필요 | 통계 모델 |
| 이질적인 다수 시리즈 | 기초 모델 |
| 운영 지연시간 제약 | 통계 모델 |
