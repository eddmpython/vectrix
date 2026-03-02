---
title: 파이프라인 API
---

# 파이프라인 API

## ForecastPipeline

`ForecastPipeline(steps)`

변환기와 예측기를 체이닝하는 순차 파이프라인.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `steps` | `list[tuple]` | `(이름, 변환기/예측기)` 튜플 리스트 |

### 메서드

| 메서드 | 반환 | 설명 |
|---|---|---|
| `fit(y)` | `self` | 변환기 학습 후 예측기 학습 |
| `predict(steps)` | `tuple` | (predictions, lower, upper) |
| `transform(y)` | `ndarray` | 변환된 데이터 |
| `inverseTransform(y)` | `ndarray` | 원래 스케일 데이터 |
| `listSteps()` | `list` | 스텝 이름 목록 |
| `getStep(name)` | `object` | 변환기/예측기 인스턴스 |
| `getParams()` | `dict` | 전체 매개변수 |

## 변환기

모든 변환기는 `fitTransform(y)`과 `inverseTransform(y)`을 구현합니다.

### Scaler

`Scaler(method='zscore')`

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `method` | `str` | `'zscore'` (평균=0, 표준편차=1) 또는 `'minmax'` (0--1 범위) |

### LogTransformer

`LogTransformer()`

`log(1 + y)` 변환. 음수값에 대한 자동 시프트.

### BoxCoxTransformer

`BoxCoxTransformer(lmbda=None)`

MLE 기반 최적 Box-Cox 람다. `lmbda`를 전달하면 고정.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `lmbda` | `float` | Box-Cox 람다 (None이면 자동 추정) |

### Differencer

`Differencer(d=1)`

d차 차분.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `d` | `int` | 차분 차수 (기본: 1) |

### Deseasonalizer

`Deseasonalizer(period=7)`

주기 평균으로 계절성 제거.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `period` | `int` | 계절 주기 (기본: 7) |

### Detrend

`Detrend()`

선형 추세 제거.

### OutlierClipper

`OutlierClipper(factor=3.0)`

IQR 기반 이상치 클리핑. 역변환 없음.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `factor` | `float` | IQR 배수 (기본: 3.0) |

### MissingValueImputer

`MissingValueImputer(method='linear')`

결측값 처리. 역변환 없음.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `method` | `str` | `'linear'` (선형 보간), `'mean'` (평균), `'ffill'` (전방 채움) |
