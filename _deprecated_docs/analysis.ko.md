# 분석 & DNA

## 빠른 분석

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
```

## DNA 프로필

모든 시계열은 65+ 통계적 특성 기반의 고유한 "DNA"를 가집니다.

```python
dna = report.dna
print(f"핑거프린트: {dna.fingerprint}")
print(f"난이도: {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"카테고리: {dna.category}")
print(f"추천 모델: {dna.recommendedModels[:3]}")
```

| 속성 | 설명 |
|------|------|
| `fingerprint` | 결정적 해시 — 동일 데이터는 항상 같은 값 |
| `difficulty` | easy / medium / hard / very_hard |
| `difficultyScore` | 0-100 수치 점수 |
| `category` | seasonal, trending, volatile, intermittent 등 |
| `recommendedModels` | 최적 모델 순서 목록 |

## 데이터 특성

```python
c = report.characteristics
print(f"길이: {c.length}")
print(f"주기: {c.period}")
print(f"추세: {c.hasTrend} ({c.trendDirection}, 강도 {c.trendStrength:.2f})")
print(f"계절성: {c.hasSeasonality} (강도 {c.seasonalStrength:.2f})")
print(f"변동성: {c.volatilityLevel} ({c.volatility:.4f})")
print(f"예측가능성: {c.predictabilityScore}/100")
print(f"이상치: {c.outlierCount} ({c.outlierRatio:.1%})")
```

## 변화점 & 이상치

```python
print(f"변화점: {report.changepoints}")
print(f"이상치: {report.anomalies}")
```

## 통합 리포트

분석 + 예측을 한 번에:

```python
from vectrix import quick_report

report = quick_report(df, steps=14)
print(report['summary'])
forecast_result = report['forecast']
analysis_result = report['analysis']
```

## ForecastDNA 직접 접근

```python
from vectrix.adaptive import ForecastDNA

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(profile.fingerprint)
print(profile.recommendedModels)
```

---

**인터랙티브 튜토리얼:** `marimo run docs/tutorials/ko/02_analyze.py`
