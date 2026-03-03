---
title: Vectrix 클래스
---

# Vectrix 클래스

모델 비교와 선택 기능을 갖춘 전체 예측 엔진.

## `Vectrix(verbose=False)`

메인 예측 클래스.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `verbose` | `bool` | 상세 로그 출력 (기본: False) |

### 메서드

#### `forecast(df, dateCol, valueCol, steps=30, trainRatio=0.8)`

전체 예측 파이프라인을 실행합니다.

**매개변수:**

| 매개변수 | 타입 | 설명 |
|---|---|---|
| `df` | `DataFrame` | 날짜와 값 컬럼을 포함한 pandas DataFrame |
| `dateCol` | `str` | 날짜 컬럼명 |
| `valueCol` | `str` | 값 컬럼명 |
| `steps` | `int` | 예측 스텝 수 (기본: 30) |
| `trainRatio` | `float` | 학습 데이터 비율 (기본: 0.8) |

**반환:** `ForecastResult`

### ForecastResult

| 속성 | 타입 | 설명 |
|---|---|---|
| `.success` | `bool` | 예측 성공 여부 |
| `.predictions` | `np.ndarray` | 최종 예측값 |
| `.dates` | `list` | 예측 날짜 문자열 |
| `.lower95` | `np.ndarray` | 95% 하한 |
| `.upper95` | `np.ndarray` | 95% 상한 |
| `.bestModelId` | `str` | 선택된 모델 ID |
| `.bestModelName` | `str` | 선택된 모델 표시명 |
| `.allModelResults` | `dict` | 전체 ModelResult 객체 |
| `.characteristics` | `DataCharacteristics` | 탐지된 데이터 속성 |
| `.flatRisk` | `FlatRiskAssessment` | 평탄 예측 위험도 |
| `.flatInfo` | `FlatPredictionInfo` | 평탄 예측 정보 |
| `.warnings` | `list` | 생성된 경고 목록 |

### ForecastResult 메서드

| 메서드 | 반환 | 설명 |
|---|---|---|
| `.getSummary()` | `str` | 포맷된 텍스트 요약 |
