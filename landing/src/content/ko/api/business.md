---
title: 비즈니스 API
---

# 비즈니스 API

이상치 탐지, What-If 분석, 백테스팅, 비즈니스 지표.

## AnomalyDetector

`AnomalyDetector(y)`

생성자에 시계열 배열을 전달합니다.

### 메서드

- `detect(y, method="auto", threshold=3.0, period=1)` -> `AnomalyResult`

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `y` | `ndarray` | 입력 시계열 |
| `method` | `str` | 탐지 방법: `auto`, `zscore`, `iqr`, `rolling` |
| `threshold` | `float` | 이상치 판정 임계값 (기본: 3.0) |
| `period` | `int` | 계절 주기 (기본: 1) |

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

- `analyze(basePredictions, historicalData, scenarios, period=7)` -> `list[ScenarioResult]`
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

`Backtester(nFolds=5, testSize=None)`

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `nFolds` | `int` | 폴드 수 (기본: 5) |
| `testSize` | `int` | 테스트 크기 (선택) |

### 메서드

- `run(y, modelFactory)` -> `BacktestResult`

### BacktestResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.folds` | `list[FoldResult]` | 폴드별 결과 |
| `.avgMetrics` | `dict` | 평균 지표 |

### FoldResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.metrics` | `dict` | `'mape'`, `'smape'` 등 지표 딕셔너리 |

## BusinessMetrics

`BusinessMetrics`

### 메서드

- `BusinessMetrics.calculate(y_true, y_pred, **kwargs)` -> `dict`

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
