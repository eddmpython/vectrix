---
title: 분석 & DNA
---

# 분석 & DNA

**예측하기 전에 데이터를 이해하세요.** Vectrix의 분석 시스템은 시계열의 DNA를 추출합니다 — 추세 방향, 계절성 강도, 변동성 수준, 구조적 변화점, 이상치, 그리고 예측 난이도. 이 정보가 자동 모델 선택의 핵심 근거가 됩니다.

## 빠른 분석

`analyze()` 함수는 `forecast()`와 동일한 입력 형식을 지원합니다 — 리스트, 배열, DataFrame, Series, CSV 경로:

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
```

## DNA 프로필

모든 시계열에는 고유한 통계적 지문이 있습니다. Vectrix는 65개 이상의 특성(자기상관 구조, 허스트 지수, 엔트로피, 변동성 클러스터링, 계절성 강도 등)을 추출하여 데이터의 "DNA 프로필"을 생성합니다. 이 프로필이 모델 선택과 난이도 추정을 결정합니다:

```python
dna = report.dna
print(f"핑거프린트: {dna.fingerprint}")
print(f"난이도: {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"카테고리: {dna.category}")
print(f"추천 모델: {dna.recommendedModels[:3]}")
```

| 속성 | 설명 |
|------|------|
| `fingerprint` | 결정적 해시 -- 동일 데이터는 항상 같은 값 |
| `difficulty` | easy / medium / hard / very_hard |
| `difficultyScore` | 0-100 수치 점수 |
| `category` | seasonal, trending, volatile, intermittent 등 |
| `recommendedModels` | 최적 모델 순서 목록 |

## 데이터 특성

`characteristics` 객체는 데이터의 포괄적인 통계 프로필을 제공합니다 — 추세 방향과 강도, 계절 패턴, 변동성 수준, 예측 가능성 점수:

```python
c = report.characteristics
print(f"길이: {c.length}")
print(f"주기: {c.period}")
print(f"추세: {c.hasTrend} ({c.trendDirection}, 강도 {c.trendStrength:.2f})")
print(f"계절성: {c.hasSeasonality} (주기 {c.seasonalPeriods})")
print(f"변동성: {c.volatility:.4f}")
```

## 변화점 & 이상치

**변화점**은 시계열의 통계적 특성이 변하는 위치입니다(평균, 분산, 추세의 급격한 변화). **이상치**는 개별 관측값 중 예상 패턴에서 크게 벗어난 값입니다:

```python
print(f"변화점: {report.changepoints}")
print(f"이상치: {report.anomalies}")
```

## 통합 리포트

분석과 예측을 한 번에 실행합니다. DNA 프로필링, 특성 추출, 모델 선택, 예측을 하나의 호출로 수행합니다:

```python
from vectrix import quick_report

report = quick_report(df, date="date", value="sales", steps=14)
forecast_result = report['forecast']
analysis_result = report['analysis']
```

## ForecastDNA 직접 접근

커스텀 모델 선택 로직 구축이나 DNA 프로필 캐싱 등 더 세밀한 제어가 필요할 때 `ForecastDNA` 클래스를 직접 사용합니다:

```python
from vectrix.adaptive import ForecastDNA

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(profile.fingerprint)
print(profile.recommendedModels)
```

---

**인터랙티브 튜토리얼:** `marimo run docs/tutorials/ko/02_analyze.py`
