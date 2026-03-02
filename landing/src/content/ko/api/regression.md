---
title: 회귀분석 API
---

# 회귀분석 API

R 스타일 수식 회귀분석 + 전체 진단.

## LinearRegressor

`LinearRegressor()`

OLS 선형 회귀.

### 메서드

| 메서드 | 반환 | 설명 |
|---|---|---|
| `fit(y, X)` | `self` | 모델 학습 |
| `predict(X, interval='none', alpha=0.05)` | `DataFrame` | 예측 (구간 옵션: `'none'`, `'confidence'`, `'prediction'`) |
| `summary()` | `str` | 회귀 요약표 |
| `diagnose()` | `str` | 전체 진단 결과 |

**매개변수 (fit):**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `y` | `ndarray` | 종속변수 |
| `X` | `ndarray` | 독립변수 행렬 |

## RidgeRegressor

`RidgeRegressor(alpha=1.0)`

L2 정규화 회귀.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `alpha` | `float` | 정규화 강도 (기본: 1.0) |

## LassoRegressor

`LassoRegressor(alpha=1.0)`

L1 정규화 회귀 (희소 해 산출).

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `alpha` | `float` | 정규화 강도 (기본: 1.0) |

## HuberRegressor

`HuberRegressor(delta=1.345)`

강건 회귀 -- 이상치에 저항.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `delta` | `float` | Huber 임계값 (기본: 1.345) |

## QuantileRegressor

`QuantileRegressor(quantile=0.5)`

중위수 회귀 (quantile=0.5) 또는 임의 분위수 회귀.

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `quantile` | `float` | 분위수 (기본: 0.5, 중위수) |

## RegressionDiagnostics

`RegressionDiagnostics()`

회귀 모델 진단 도구 모음.

### 메서드

| 메서드 | 설명 |
|---|---|
| `vif(X)` | 분산 팽창 계수 (다중공선성 검사) |
| `shapiroWilk(residuals)` | Shapiro-Wilk 정규성 검정 |
| `breuschPagan(residuals, X)` | Breusch-Pagan 등분산성 검정 |
| `durbinWatson(residuals)` | Durbin-Watson 자기상관 검정 |
| `cooksDistance(y, X)` | Cook's Distance 영향력 분석 |
| `fullReport(y, X, residuals)` | 전체 진단 리포트 문자열 |
