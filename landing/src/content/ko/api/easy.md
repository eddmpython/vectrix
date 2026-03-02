---
title: Easy API
---

# Easy API

Vectrix를 사용하는 가장 간단한 방법. 각 작업에 함수 하나면 충분합니다.

## 함수

### `forecast(data, steps=10, **kwargs)`

자동 모델 선택을 통한 원콜 예측.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `data` | `list, ndarray, Series, DataFrame, dict, str` | 입력 데이터 (CSV 경로도 가능) |
| `steps` | `int` | 예측 스텝 수 (기본: 10) |
| `date` | `str` | 날짜 컬럼명 (선택, 자동 탐지) |
| `value` | `str` | 값 컬럼명 (선택, 자동 탐지) |
| `period` | `int` | 계절 주기 (선택, 자동 탐지) |

**반환:** `EasyForecastResult`

### `analyze(data, **kwargs)`

시계열 DNA 프로파일링, 변화점 탐지, 이상치 식별.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `data` | `list, ndarray, Series, DataFrame, dict, str` | 입력 데이터 (forecast와 동일 형식) |
| `date` | `str` | 날짜 컬럼명 (선택) |
| `value` | `str` | 값 컬럼명 (선택) |

**반환:** `EasyAnalysisResult`

### `regress(y=None, X=None, data=None, formula=None, method="ols", **kwargs)`

R 스타일 수식 회귀분석 + 전체 진단.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `y` | `ndarray` | 종속변수 |
| `X` | `ndarray` | 독립변수 |
| `data` | `DataFrame` | 데이터프레임 (formula와 함께 사용) |
| `formula` | `str` | R 스타일 수식 (예: `"y ~ x1 + x2"`) |
| `method` | `str` | `"ols"`, `"ridge"`, `"lasso"`, `"huber"`, `"quantile"` |
| `summary` | `bool` | 요약 자동 출력 (기본: True) |

**반환:** `EasyRegressionResult`

### `quick_report(data, steps=10, **kwargs)`

분석 + 예측 통합 리포트 생성.

**반환:** `str` -- 포맷된 리포트 문자열

## 결과 클래스

### EasyForecastResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.predictions` | `np.ndarray` | 예측값 |
| `.dates` | `list` | 예측 날짜 문자열 |
| `.lower` | `np.ndarray` | 95% 하한 |
| `.upper` | `np.ndarray` | 95% 상한 |
| `.model` | `str` | 선택된 모델명 |
| `.mape` | `float` | 검증 MAPE (%) |
| `.rmse` | `float` | 검증 RMSE |
| `.mae` | `float` | 검증 MAE |
| `.smape` | `float` | 검증 sMAPE |
| `.models` | `list` | 평가된 전체 모델명 |

### EasyForecastResult 메서드

| 메서드 | 반환 타입 | 설명 |
|---|---|---|
| `.compare()` | `DataFrame` | MAPE 기준 전체 모델 순위 |
| `.all_forecasts()` | `DataFrame` | 모든 모델의 예측값 |
| `.summary()` | `str` | 포맷된 텍스트 요약 |
| `.to_dataframe()` | `DataFrame` | date, prediction, lower95, upper95 |
| `.to_csv(path)` | `self` | CSV로 저장 |
| `.to_json(path)` | `str` | JSON으로 저장 |
| `.describe()` | `DataFrame` | Pandas 스타일 통계 |

### EasyAnalysisResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.dna` | `DNAProfile` | DNA 프로파일 객체 |
| `.changepoints` | `np.ndarray` | 변화점 인덱스 |
| `.anomalies` | `np.ndarray` | 이상치 인덱스 |
| `.features` | `dict` | 추출된 특성 |
| `.characteristics` | `DataCharacteristics` | 데이터 속성 |
| `.summary()` | `str` | 포맷된 리포트 |

### EasyRegressionResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.coefficients` | `np.ndarray` | 회귀 계수 |
| `.pvalues` | `np.ndarray` | P-값 |
| `.r_squared` | `float` | R² |
| `.adj_r_squared` | `float` | 수정 R² |
| `.f_stat` | `float` | F-통계량 |
| `.summary()` | `str` | 회귀 요약표 |
| `.diagnose()` | `str` | 전체 진단 결과 |
| `.predict(X)` | `DataFrame` | 구간 포함 예측값 |
