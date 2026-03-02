---
title: 타입
---

# 타입

Vectrix 전반에서 사용되는 핵심 데이터 타입과 결과 객체.

## ForecastResult

`Vectrix.forecast()`의 메인 결과.

| 필드 | 타입 | 설명 |
|---|---|---|
| `success` | `bool` | 예측 성공 여부 |
| `predictions` | `np.ndarray` | 예측값 |
| `dates` | `list[str]` | 예측 날짜 |
| `lower95` | `np.ndarray` | 95% 하한 |
| `upper95` | `np.ndarray` | 95% 상한 |
| `bestModelId` | `str` | 선택된 모델 ID |
| `bestModelName` | `str` | 모델 표시명 |
| `allModelResults` | `dict` | 전체 모델 결과 |
| `characteristics` | `DataCharacteristics` | 데이터 속성 |

## ModelResult

개별 모델 결과.

| 필드 | 타입 | 설명 |
|---|---|---|
| `modelId` | `str` | 모델 식별자 |
| `modelName` | `str` | 표시명 |
| `isValid` | `bool` | 유효한 출력 생성 여부 |
| `mape` | `float` | 검증 MAPE |
| `predictions` | `np.ndarray` | 모델 예측값 |
| `lower95` | `np.ndarray` | 하한 |
| `upper95` | `np.ndarray` | 상한 |
| `trainingTime` | `float` | 학습 시간 (초) |
| `flatInfo` | `FlatPredictionInfo` | 평탄 예측 탐지 정보 |

## DataCharacteristics

| 필드 | 타입 | 설명 |
|---|---|---|
| `length` | `int` | 관측치 수 |
| `period` | `int` | 탐지된 계절 주기 |
| `frequency` | `str` | 빈도 레이블 (D, W, M 등) |
| `hasTrend` | `bool` | 추세 탐지 여부 |
| `trendDirection` | `str` | 'increasing', 'decreasing', 'none' |
| `trendStrength` | `float` | 0--1 강도 |
| `hasSeasonality` | `bool` | 계절성 탐지 여부 |
| `seasonalStrength` | `float` | 0--1 강도 |
| `predictabilityScore` | `float` | 0--100 예측 가능성 점수 |

## ModelInfo

사용 가능한 모델 카탈로그 항목.

| 필드 | 타입 | 설명 |
|---|---|---|
| `id` | `str` | 모델 식별자 |
| `name` | `str` | 표시명 |
| `category` | `str` | 모델 카테고리 |
| `description` | `str` | 간략 설명 |
| `strengths` | `list[str]` | 모델의 강점 |
