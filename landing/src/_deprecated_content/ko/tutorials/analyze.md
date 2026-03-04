---
title: "튜토리얼 02 — 분석 & DNA"
---

# 튜토리얼 02 — 분석 & DNA

**예측하기 전에 데이터를 이해하세요.** 모든 시계열에는 고유한 통계적 지문이 있습니다 — Vectrix는 이를 '시계열 DNA'라고 부릅니다. Vectrix는 65개 이상의 특성을 추출하여 추세의 강도, 계절성의 주기, 변동성의 패턴, 예측 난이도까지 데이터의 본질을 정량화합니다. 이 DNA 프로파일은 최적의 예측 모델을 자동으로 추천하는 메타러닝의 기반이 됩니다.

## 기본 분석

`analyze()` 함수는 `forecast()`와 동일한 입력 형식(리스트, 배열, Series, DataFrame, CSV 경로)을 받습니다. 데이터를 전달하면 DNA 프로파일, 변화점, 이상치, 통계적 특성을 한 번에 분석합니다.

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

리스트나 배열도 직접 전달할 수 있습니다

```python
from vectrix import analyze

data = [120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215]
report = analyze(data)
print(report.summary())
```

## DNA 프로파일

DNA 프로파일은 Vectrix 모델 선택 지능의 핵심입니다. 데이터의 통계적 특성을 분석하여 난이도를 평가하고, 해당 데이터 유형에 가장 적합한 모델을 순서대로 추천합니다. 예를 들어, 강한 계절성 데이터에는 MSTL이나 ETS를, 비선형 패턴에는 CES나 DOT를 우선 추천합니다. `report.dna`를 통해 접근합니다.

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

변화점(Changepoint)은 시계열의 통계적 속성 — 평균, 분산, 추세 방향 — 이 근본적으로 전환되는 위치를 말합니다. 시장 환경 변화, 정책 변경, 제품 출시, 경쟁사 진입 등 비즈니스 환경의 구조적 단절(structural break)을 데이터에서 자동으로 포착합니다. 변화점 이후의 데이터가 예측에 더 관련성이 높을 수 있으므로, 모델 학습 전략을 조정하는 데 활용됩니다.

```python
print(f"변화점 인덱스: {report.changepoints}")
```

**예상 출력:**

```
변화점 인덱스: [89, 201]
```

이는 구조적 단절이 발생한 데이터 위치에 해당합니다 -- 정책 변경, 제품 출시, 시장 변동 등.

## 이상치 감지

이상치(Anomaly)는 변화점과 다릅니다. 변화점이 데이터의 **영구적인 구조 변화**라면, 이상치는 **일시적으로** 예상 패턴에서 크게 벗어나는 개별 관측치입니다. 시스템 오류, 프로모션 효과, 일회성 이벤트 등이 원인일 수 있으며, 예측 모델의 학습을 왜곡할 수 있으므로 사전에 식별하는 것이 중요합니다.

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

`characteristics` 객체는 시계열의 포괄적인 통계적 프로파일을 제공합니다. 데이터의 길이, 감지된 계절 주기, 추세의 존재 여부와 방향, 계절성의 강도, 변동성 수준 등 예측 전략 수립에 필요한 핵심 정보가 포함됩니다.

```python
c = report.characteristics

print(f"길이: {c.length}")
print(f"주기: {c.period}")
print(f"추세: {c.hasTrend} ({c.trendDirection}, 강도 {c.trendStrength:.2f})")
print(f"계절성: {c.hasSeasonality} (주기 {c.seasonalPeriods})")
print(f"변동성: {c.volatility:.4f}")
```

**예상 출력:**

```
길이: 365
주기: 7
추세: True (upward, 강도 0.72)
계절성: True (주기 [7])
변동성: 0.0312
```

### 특성 참조

| 속성 | 타입 | 설명 |
|-----|------|------|
| `length` | `int` | 관측치 수 |
| `period` | `int` | 감지된 계절 주기 |
| `frequency` | `str` | 빈도 레이블 |
| `hasTrend` | `bool` | 추세 존재 여부 |
| `trendDirection` | `str` | `upward`, `downward`, `none` |
| `trendStrength` | `float` | 추세 강도 (0-1) |
| `hasSeasonality` | `bool` | 계절성 존재 여부 |
| `seasonalPeriods` | `list` | 감지된 계절 주기 목록 |
| `volatility` | `float` | 변동 계수 |

## 추출된 특성

65개 이상의 원시 통계 특성을 딕셔너리로 접근할 수 있습니다. `trendStrength`(추세 강도), `seasonalStrength`(계절 강도), `hurstExponent`(장기 기억 지수), `volatilityClustering`(변동성 군집), `nonlinearAutocorr`(비선형 자기상관), `demandDensity`(수요 밀도), `entropy`(정보 엔트로피) 등 시계열의 모든 측면을 정량화합니다. 이 특성들은 DNA 메타러닝에서 모델 추천의 입력 변수로 활용됩니다.

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

분석 결과를 활용하여 정보에 기반한 예측 결정을 내릴 수 있습니다. 난이도가 높은 데이터에는 경고를 표시하고, 구조적 단절이 있으면 최근 데이터의 가중치를 높이며, 계절 주기에 맞춰 예측 기간을 조정하는 등 데이터의 특성에 따라 전략을 분기할 수 있습니다.

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

Easy API(`analyze()`) 대신 하위 레벨에서 DNA 분석을 직접 제어하려면 `ForecastDNA` 클래스를 사용합니다. 주기(period)를 직접 지정하거나, 분석 파이프라인을 커스터마이즈할 때 유용합니다.

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
