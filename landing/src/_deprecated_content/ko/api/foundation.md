---
title: 기초 모델 API
---

# 기초 모델 API

## ChronosForecaster

`ChronosForecaster(modelId="amazon/chronos-bolt-small", device="cpu")`

Amazon Chronos-2 래퍼.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `modelId` | `str` | 모델 ID (기본: `"amazon/chronos-bolt-small"`) |
| `device` | `str` | 디바이스 (기본: `"cpu"`) |

### 메서드

| 메서드 | 반환 | 설명 |
|---|---|---|
| `fit(y)` | `self` | 컨텍스트 저장 (학습 없음) |
| `predict(steps)` | `tuple` | (predictions, lower, upper) |
| `predictQuantiles(steps, quantileLevels)` | `ndarray` | 분위수별 예측 (shape: 분위수 수 x steps) |
| `predictBatch(series_list, steps)` | `list` | 다중 시리즈 배치 예측 |

## TimesFMForecaster

`TimesFMForecaster(modelId="google/timesfm-2.5-200m-pytorch")`

Google TimesFM 2.5 래퍼.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `modelId` | `str` | 모델 ID (기본: `"google/timesfm-2.5-200m-pytorch"`) |

### 메서드

| 메서드 | 반환 | 설명 |
|---|---|---|
| `fit(y)` | `self` | 컨텍스트 저장 |
| `predict(steps)` | `tuple` | (predictions, lower, upper) |
| `predictWithCovariates(steps, dynamicNumerical, dynamicCategorical)` | `tuple` | 공변량 포함 예측 |

**predictWithCovariates 매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `steps` | `int` | 예측 스텝 수 |
| `dynamicNumerical` | `ndarray` | 미래 수치 특성 (steps, n_features) |
| `dynamicCategorical` | `ndarray` | 미래 범주형 특성 |

## NeuralForecaster

`NeuralForecaster(architecture="nbeats", **kwargs)`

NeuralForecast 래퍼.

### 편의 클래스

| 클래스 | 설명 |
|---|---|
| `NBEATSForecaster(**kwargs)` | N-BEATS 아키텍처 |
| `NHITSForecaster(**kwargs)` | N-HiTS 아키텍처 |
| `TFTForecaster(**kwargs)` | Temporal Fusion Transformer |

## 가용성 플래그

```python
from vectrix import CHRONOS_AVAILABLE, TIMESFM_AVAILABLE, NEURALFORECAST_AVAILABLE
```

선택적 의존성이 설치되었는지 나타내는 불리언 플래그.
