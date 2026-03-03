---
title: 적응형 API
---

# 적응형 API

레짐 감지, 자가치유 예측, 비즈니스 제약, Forecast DNA.

## RegimeDetector

`RegimeDetector(nRegimes=2)`

Hidden Markov Model을 사용한 통계적 레짐 감지.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `nRegimes` | `int` | 레짐 수 (기본: 2) |

### 메서드

- `detect(data)` -> `RegimeResult`

### RegimeResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.currentRegime` | `int` | 현재 레짐 인덱스 |
| `.regimeStats` | `dict` | 레짐별 통계 |
| `.regimes` | `np.ndarray` | 관측치별 레짐 레이블 |
| `.transitionMatrix` | `np.ndarray` | 레짐 전이 행렬 |

## RegimeAwareForecaster

`RegimeAwareForecaster()`

현재 레짐 컨텍스트를 사용하여 예측.

### 메서드

- `forecast(data, steps, period=None)` -> predictions

## ForecastDNA

`ForecastDNA()`

메타러닝을 위한 시계열 특성 프로파일링.

### 메서드

- `analyze(data, period=None)` -> `DNAProfile`

### DNAProfile

| 속성 | 타입 | 설명 |
|---|---|---|
| `.category` | `str` | 시리즈 유형 |
| `.difficulty` | `str` | 'easy', 'medium', 'hard' |
| `.difficultyScore` | `float` | 0--100 점수 |
| `.fingerprint` | `str` | 고유 핑거프린트 코드 |
| `.recommendedModels` | `list` | 추천 모델 ID 목록 |
| `.features` | `dict` | 추출된 특성 |

## SelfHealingForecast

`SelfHealingForecast(originalPredictions, lower, upper, tolerance=0.1)`

실시간 관측값으로 예측을 자동 교정.

### 메서드

- `observe(actuals)` -> `HealingStatus`
- `getUpdatedForecast()` -> `(predictions, lower, upper)`
- `getReport()` -> `HealingReport`

### HealingReport

| 속성 | 타입 | 설명 |
|---|---|---|
| `.healthScore` | `float` | 0--100 건강 점수 |
| `.overallHealth` | `str` | 건강 상태 레이블 |
| `.totalObserved` | `int` | 관측 횟수 |
| `.totalCorrected` | `int` | 교정 횟수 |

## ConstraintAwareForecaster

`ConstraintAwareForecaster()`

예측에 비즈니스 제약 조건을 적용.

### 메서드

- `apply(predictions, lower95, upper95, constraints, smoothing=True)` -> `ConstraintResult`

## Constraint

`Constraint(name, type, value, ...)`

| 유형 | 설명 | 예시 |
|------|------|------|
| `min` | 최솟값 제한 | `Constraint(name='floor', type='min', value=0)` |
| `max` | 최댓값 제한 | `Constraint(name='cap', type='max', value=5000)` |
| `range` | 최소/최대 범위 | `Constraint(name='bounds', type='range', value=(0, 5000))` |
| `monotonic` | 증가/감소만 허용 | `Constraint(name='inc', type='monotonic', value='increasing')` |
