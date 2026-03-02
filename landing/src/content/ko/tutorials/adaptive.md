---
title: "튜토리얼 05 — 적응형 지능"
---

# 튜토리얼 05 — 적응형 지능

Vectrix만의 고유한 기능 -- 다른 어떤 예측 라이브러리에도 없는 적응형 시스템입니다. 레짐 감지, 자가 치유, 비즈니스 제약 적용, DNA 지문까지 4가지 핵심 기능을 제공합니다.

## 레짐 감지

HMM(Hidden Markov Model)을 사용하여 레짐 변화(상승장/하락장, 성수기/비수기)를 감지합니다:

```python
from vectrix import RegimeDetector

detector = RegimeDetector(nRegimes=3)
result = detector.detect(data)

print(f"현재 레짐: {result.currentRegime}")
for label, stats in result.regimeStats.items():
    print(f"  {label}: 평균={stats['mean']:.2f}, 표준편차={stats['std']:.2f}")
```

**예상 출력:**

```
현재 레짐: Regime_2
  Regime_0: 평균=95.23, 표준편차=8.45
  Regime_1: 평균=142.67, 표준편차=12.31
  Regime_2: 평균=203.89, 표준편차=15.72
```

각 레짐은 서로 다른 통계적 특성(평균, 분산, 추세)을 가진 구간을 나타냅니다. 데이터의 구조적 변화를 파악하는 데 유용합니다.

## 레짐 인식 예측

레짐별로 자동으로 모델을 전환하여 예측합니다:

```python
from vectrix import RegimeAwareForecaster

raf = RegimeAwareForecaster()
result = raf.forecast(data, steps=30, period=7)

print(f"현재 레짐: {result.currentRegime}")
print(f"레짐별 모델: {result.modelPerRegime}")
```

## 자가 치유 예측

예측 오차를 실시간으로 모니터링하고 자동 교정합니다:

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast(predictions, lower, upper, historicalData)
healer.observe(actualValues)

report = healer.getReport()
print(f"건강 상태: {report.overallHealth} ({report.healthScore:.1f}/100)")
print(f"개선율: {report.improvementPct:.1f}%")
print(f"총 관측: {report.totalObserved}")
print(f"총 교정: {report.totalCorrected}")

updated = healer.getUpdatedForecast()
```

**예상 출력:**

```
건강 상태: good (82.5/100)
개선율: 15.3%
총 관측: 14
총 교정: 3
```

자가 치유 시스템은 CUSUM과 EWMA 드리프트 감지를 사용하여 예측이 실제 값에서 벗어나는 시점을 자동으로 포착하고, Conformal Prediction 기반 교정을 적용합니다.

## 제약 인식 예측

비즈니스 제약을 예측에 적용합니다:

```python
from vectrix import ConstraintAwareForecaster, Constraint

caf = ConstraintAwareForecaster()
result = caf.apply(predictions, lower, upper, constraints=[
    Constraint('non_negative', {}),
    Constraint('range', {'min': 0, 'max': 5000}),
    Constraint('capacity', {'capacity': 10000}),
    Constraint('yoy_change', {'maxPct': 30, 'historicalData': pastYear}),
])

print(f"원본 평균: {predictions.mean():.2f}")
print(f"교정 평균: {result.predictions.mean():.2f}")
print(f"위반 수: {result.violationCount}")
```

### 제약 유형

| 유형 | 설명 |
|------|------|
| `non_negative` | 음수 값 방지 |
| `range` | 최솟값/최댓값 범위 제한 |
| `capacity` | 용량 상한 제한 |
| `yoy_change` | 전년 대비 변화율 제한 |
| `sum` | 총합 제약 |
| `monotone` | 단조 증가/감소만 허용 |
| `ratio` | 시계열 간 비율 제약 |
| `custom` | 사용자 정의 함수 |

> **참고:** 제약은 순서대로 적용됩니다. 서로 충돌하는 제약이 있으면, 나중에 적용되는 제약이 우선합니다.

## Forecast DNA

시계열의 지문을 추출하여 메타러닝 기반 모델 추천에 활용합니다:

```python
from vectrix import ForecastDNA

dna = ForecastDNA()
profile = dna.analyze(data, period=7)

print(f"지문: {profile.fingerprint}")
print(f"난이도: {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"카테고리: {profile.category}")
print(f"추천 모델: {profile.recommendedModels}")
```

**예상 출력:**

```
지문: b7e4a1f3
난이도: medium (45/100)
카테고리: seasonal
추천 모델: ['DOT', 'AutoCES', 'AutoETS', 'MSTL', 'FourTheta']
```

65개 이상의 통계 특성을 추출하여 데이터의 고유한 지문을 생성합니다. 동일한 데이터는 항상 동일한 지문을 생성하므로, 캐싱과 재현성에 활용할 수 있습니다.

## 완전한 적응형 워크플로우

```python
import numpy as np
from vectrix import (
    forecast, RegimeDetector, ForecastDNA,
    SelfHealingForecast, ConstraintAwareForecaster, Constraint,
)

data = np.random.randn(365).cumsum() + 500

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"DNA 난이도: {profile.difficulty}")
print(f"추천 모델: {profile.recommendedModels[:3]}")

detector = RegimeDetector(nRegimes=2)
regimes = detector.detect(data)
print(f"\n현재 레짐: {regimes.currentRegime}")

result = forecast(data, steps=30)
print(f"\n최적 모델: {result.model}")
print(f"MAPE: {result.mape:.2f}%")

caf = ConstraintAwareForecaster()
constrained = caf.apply(
    result.predictions, result.lower, result.upper,
    constraints=[
        Constraint('non_negative', {}),
        Constraint('range', {'min': 200, 'max': 800}),
    ]
)
print(f"\n제약 적용 후 예측 범위: {constrained.predictions.min():.1f} ~ {constrained.predictions.max():.1f}")
```

---
