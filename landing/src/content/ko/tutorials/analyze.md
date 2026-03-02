---
title: "튜토리얼 02 — 분석 & DNA"
---

# 튜토리얼 02 — 분석 & DNA

모든 시계열 데이터에는 고유한 지문이 있습니다. Vectrix는 65개 이상의 통계 특성을 추출하여 데이터의 본질을 드러내는 "DNA 프로파일"을 생성하고, 최적의 예측 전략을 추천합니다.

## 기본 분석

`analyze()` 함수는 `forecast()`와 동일한 입력 형식을 받습니다:

```python
import pandas as pd
from vectrix import analyze

df = pd.read_csv("sales.csv")
report = analyze(df, date="date", value="sales")

print(report.summary())
```

**예상 출력:**

```
=== Vectrix Analysis Report ===

DNA Profile:
  Fingerprint: a3f7c2d1
  Difficulty: medium (42/100)
  Category: seasonal
  Recommended Models: ['AutoETS', 'MSTL', 'Theta']

Data Characteristics:
  Length: 365 observations
  Period: 7 (weekly)
  Trend: Yes (upward, strength 0.72)
  Seasonality: Yes (strength 0.85)
  Volatility: low (0.0312)

Changepoints: [89, 201]
Anomalies: [15, 156, 298]
```

리스트나 배열도 직접 전달할 수 있습니다:

```python
from vectrix import analyze

data = [120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215]
report = analyze(data)
print(report.summary())
```

## DNA 프로파일

DNA 프로파일은 Vectrix 지능의 핵심입니다. `report.dna`를 통해 접근합니다:

```python
dna = report.dna

print(f"지문: {dna.fingerprint}")
print(f"난이도: {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"카테고리: {dna.category}")
print(f"추천 모델: {dna.recommendedModels[:5]}")
```

**예상 출력:**

```
지문: a3f7c2d1
난이도: medium (42/100)
카테고리: seasonal
추천 모델: ['AutoETS', 'MSTL', 'Theta', 'DOT', 'AutoCES']
```

### DNA 속성

| 속성 | 타입 | 설명 |
|-----|------|------|
| `fingerprint` | `str` | 결정적 해시 -- 동일한 데이터는 항상 동일한 값 생성 |
| `difficulty` | `str` | `easy`, `medium`, `hard`, `very_hard` |
| `difficultyScore` | `float` | 수치 난이도 점수 (0-100, 높을수록 어려움) |
| `category` | `str` | `seasonal`, `trending`, `volatile`, `intermittent`, `stationary` |
| `recommendedModels` | `list[str]` | 이 데이터에 최적인 모델의 순서 목록 |

지문은 결정적입니다: 동일한 데이터는 항상 동일한 해시를 생성합니다. 캐싱과 재현성에 유용합니다.

## 변화점 감지

변화점은 시계열의 통계적 속성(평균, 분산, 추세)이 전환되는 위치입니다:

```python
print(f"변화점 인덱스: {report.changepoints}")
```

**예상 출력:**

```
변화점 인덱스: [89, 201]
```

이는 구조적 단절이 발생한 데이터 위치에 해당합니다 -- 정책 변경, 제품 출시, 시장 변동 등.

## 이상치 감지

이상치 인덱스는 예상 패턴에서 크게 벗어나는 개별 관측치를 표시합니다:

```python
print(f"이상치 인덱스: {report.anomalies}")
print(f"이상치 수: {len(report.anomalies)}")
```

**예상 출력:**

```
이상치 인덱스: [15, 156, 298]
이상치 수: 3
```

## 데이터 특성

`characteristics` 객체는 상세한 통계적 속성을 제공합니다:

```python
c = report.characteristics

print(f"길이: {c.length}")
print(f"주기: {c.period}")
print(f"추세: {c.hasTrend} ({c.trendDirection}, 강도 {c.trendStrength:.2f})")
print(f"계절성: {c.hasSeasonality} (강도 {c.seasonalStrength:.2f})")
print(f"변동성: {c.volatilityLevel} ({c.volatility:.4f})")
print(f"예측 가능성: {c.predictabilityScore}/100")
print(f"이상치: {c.outlierCount} ({c.outlierRatio:.1%})")
```

**예상 출력:**

```
길이: 365
주기: 7
추세: True (upward, 강도 0.72)
계절성: True (강도 0.85)
변동성: low (0.0312)
예측 가능성: 78/100
이상치: 3 (0.8%)
```

### 특성 참조

| 속성 | 타입 | 설명 |
|-----|------|------|
| `length` | `int` | 관측치 수 |
| `period` | `int` | 감지된 계절 주기 |
| `hasTrend` | `bool` | 추세 존재 여부 |
| `trendDirection` | `str` | `upward`, `downward`, `none` |
| `trendStrength` | `float` | 추세 강도 (0-1) |
| `hasSeasonality` | `bool` | 계절성 존재 여부 |
| `seasonalStrength` | `float` | 계절성 강도 (0-1) |
| `volatility` | `float` | 변동 계수 |
| `volatilityLevel` | `str` | `low`, `medium`, `high` |
| `predictabilityScore` | `int` | 0-100, 높을수록 예측 가능 |
| `outlierCount` | `int` | 감지된 이상치 수 |
| `outlierRatio` | `float` | 이상치 비율 |

## 추출된 특성

65개 이상의 원시 특성을 딕셔너리로 접근할 수 있습니다:

```python
features = report.features
for key, value in list(features.items())[:10]:
    print(f"  {key}: {value}")
```

**예상 출력:**

```
  trendStrength: 0.72
  seasonalStrength: 0.85
  acf1: 0.91
  hurstExponent: 0.78
  volatilityClustering: 0.15
  nonlinearAutocorr: 0.23
  demandDensity: 1.0
  seasonalPeakPeriod: 7
  entropy: 2.34
  stability: 0.88
```

## 실전 활용: 예측 전 분석

분석 결과를 활용하여 정보에 기반한 예측 결정을 내릴 수 있습니다:

```python
from vectrix import analyze, forecast

report = analyze(df, date="date", value="sales")

if report.dna.difficulty == "very_hard":
    print("경고: 이 시계열은 예측이 매우 어렵습니다.")
    print(f"추천 모델: {report.dna.recommendedModels[:3]}")

if len(report.changepoints) > 0:
    print(f"구조적 단절 감지: {report.changepoints}")
    print("최근 데이터가 과거 데이터보다 더 관련성이 높을 수 있습니다.")

if report.characteristics.hasSeasonality:
    period = report.characteristics.period
    print(f"계절 주기: {period}")
    result = forecast(df, date="date", value="sales", steps=period)
else:
    result = forecast(df, date="date", value="sales", steps=14)
```

## ForecastDNA 직접 접근

하위 레벨에서 직접 제어하려면 `ForecastDNA`를 사용합니다:

```python
from vectrix import ForecastDNA
import numpy as np

dna = ForecastDNA()
data = np.random.randn(200).cumsum() + 100
profile = dna.analyze(data, period=7)

print(f"지문: {profile.fingerprint}")
print(f"난이도: {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"추천: {profile.recommendedModels}")
```

---
