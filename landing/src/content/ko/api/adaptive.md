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
| `.regimeStats` | `list` | 레짐별 통계 |
| `.labels` | `np.ndarray` | 관측치별 레짐 레이블 |

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

실시간 관측값으로 예측을 자동 교정.

### 메서드

- `heal(originalForecast, actuals, historicalData)` -> `HealingReport`

### HealingReport

| 속성 | 타입 | 설명 |
|---|---|---|
| `.healthScore` | `float` | 0--100 건강 점수 |
| `.overallHealth` | `str` | 건강 상태 레이블 |
| `.totalCorrected` | `int` | 교정 횟수 |
| `.correctedForecast` | `np.ndarray` | 갱신된 예측값 |

## ConstraintAwareForecaster

`ConstraintAwareForecaster()`

예측에 비즈니스 제약 조건을 적용.

### 메서드

- `apply(predictions, lower, upper, constraints)` -> 제약 적용된 예측값

## Constraint

`Constraint(type, params)`

| 유형 | 매개변수 | 설명 |
|------|-----------|------|
| `non_negative` | `{}` | 음수 불가 |
| `range` | `{'min': 0, 'max': 5000}` | 최소/최대 범위 |
| `capacity` | `{'capacity': 10000}` | 용량 상한 |
| `yoy_change` | `{'maxPct': 30, 'historicalData': array}` | 전년 대비 변화율 제한 |
| `sum` | `{'total': 1000}` | 합계 제약 |
| `monotone` | `{'direction': 'increasing'}` | 증가/감소만 허용 |
| `custom` | `{'func': callable}` | 커스텀 함수 |
