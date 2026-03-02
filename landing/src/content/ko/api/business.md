---
title: 비즈니스 API
---

# 비즈니스 API

이상치 탐지, What-If 분석, 백테스팅, 비즈니스 지표.

## AnomalyDetector

`AnomalyDetector()`

### 메서드

- `detect(data, method="auto", threshold=3.0)` -> `AnomalyResult`

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `data` | `ndarray` | 입력 시계열 |
| `method` | `str` | 탐지 방법: `auto`, `zscore`, `iqr`, `rolling` |
| `threshold` | `float` | 이상치 판정 임계값 (기본: 3.0) |

### AnomalyResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.indices` | `np.ndarray` | 이상치 인덱스 |
| `.scores` | `np.ndarray` | 이상치 점수 |
| `.method` | `str` | 사용된 방법 |
| `.nAnomalies` | `int` | 이상치 수 |
| `.anomalyRatio` | `float` | 이상치 비율 |

## WhatIfAnalyzer

`WhatIfAnalyzer()`

### 메서드

- `analyze(predictions, historical, scenarios, period=None)` -> `list[ScenarioResult]`
- `compareSummary(results)` -> `str`

**매개변수 (analyze):**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `predictions` | `ndarray` | 기본 예측값 |
| `historical` | `ndarray` | 과거 데이터 |
| `scenarios` | `list[dict]` | 시나리오 목록 |
| `period` | `int` | 계절 주기 (선택) |

### 시나리오 매개변수

| 매개변수 | 타입 | 설명 |
|-----------|------|------|
| `name` | `str` | 시나리오 이름 |
| `trend_change` | `float` | 추세 조정 |
| `seasonal_multiplier` | `float` | 계절성 스케일링 |
| `shock_at` | `int` | 충격 발생 시점 |
| `shock_magnitude` | `float` | 충격 크기 |
| `shock_duration` | `int` | 충격 지속 기간 |
| `level_shift` | `float` | 영구적 수준 변화 |

## Backtester

`Backtester(nFolds=5, horizon=30, strategy='expanding', minTrainSize=50)`

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `nFolds` | `int` | 폴드 수 (기본: 5) |
| `horizon` | `int` | 각 폴드의 예측 기간 (기본: 30) |
| `strategy` | `str` | `'expanding'` 또는 `'sliding'` |
| `minTrainSize` | `int` | 최소 학습 데이터 크기 (기본: 50) |

### 메서드

- `run(data, modelFactory)` -> `BacktestResult`
- `summary(result)` -> `str`

### BacktestResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.avgMAPE` | `float` | 평균 MAPE |
| `.avgRMSE` | `float` | 평균 RMSE |
| `.folds` | `list` | 폴드별 결과 |
| `.bestFold` | `int` | 최고 폴드 번호 |
| `.worstFold` | `int` | 최저 폴드 번호 |

## BusinessMetrics

`BusinessMetrics()`

### 메서드

- `calculate(actual, predicted)` -> `dict`

**반환 딕셔너리 키:**

| 키 | 타입 | 설명 |
|---|---|---|
| `mape` | `float` | 평균 절대 백분율 오차 |
| `rmse` | `float` | 평균 제곱근 오차 |
| `mae` | `float` | 평균 절대 오차 |
| `bias` | `float` | 바이어스 (양수 = 과대예측) |
| `biasPercent` | `float` | 백분율 바이어스 |
| `wape` | `float` | 가중 절대 백분율 오차 |
| `mase` | `float` | 1 미만이면 Naive보다 우수 |
| `forecastAccuracy` | `float` | 예측 정확도 (높을수록 좋음) |
| `overForecastRatio` | `float` | 과대예측 비율 |
| `underForecastRatio` | `float` | 과소예측 비율 |
