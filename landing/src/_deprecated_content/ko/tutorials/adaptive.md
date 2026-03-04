---
title: "튜토리얼 05 — 적응형 지능"
---

# 튜토리얼 05 — 적응형 지능

**이 기능들은 Vectrix에만 존재합니다** — statsforecast, Prophet, Darts 등 다른 예측 라이브러리에는 없는 적응형 시스템입니다. 레짐 감지, 자가 치유 예측, 비즈니스 제약 적용, DNA 지문 기반 메타러닝까지 4가지 핵심 기능이 예측의 신뢰성과 실무 적용성을 높여줍니다.

## 레짐 감지

시계열 데이터는 시간에 따라 서로 다른 '상태(레짐)'를 거칩니다. 주식 시장의 상승장/하락장, 소매업의 성수기/비수기, 정책 변경 전후의 경제 지표, 제품 라이프사이클의 성장기/성숙기 등이 대표적인 예입니다. Vectrix는 HMM(Hidden Markov Model)을 사용하여 이러한 레짐 변화를 자동으로 감지하고, 각 레짐의 통계적 특성(평균, 분산, 추세)을 정량화합니다.

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

각 레짐은 서로 다른 통계적 특성을 가진 구간을 나타냅니다. 현재 데이터가 어떤 상태에 있는지 파악하면, 해당 상태에 최적화된 예측 전략을 적용할 수 있습니다.

## 레짐 인식 예측

레짐별로 자동으로 최적 모델을 전환하여 예측합니다. 예를 들어, 안정 구간에서는 ETS가, 변동 구간에서는 GARCH가 선택될 수 있습니다. 전체 데이터에 하나의 모델을 적용하는 것보다 각 상태에 맞는 모델을 사용하는 것이 더 정확한 예측을 만들어냅니다.

```python
from vectrix import RegimeAwareForecaster

raf = RegimeAwareForecaster()
result = raf.forecast(data, steps=30, period=7)

print(f"현재 레짐: {result.currentRegime}")
print(f"레짐별 모델: {result.modelPerRegime}")
```

## 자가 치유 예측

예측은 시간이 지남에 따라 점차 정확도가 떨어집니다(forecast degradation). 시장 환경이 변하거나, 예기치 못한 이벤트가 발생하면 초기 예측과 실제 값 사이의 괴리가 커집니다. 자가 치유 시스템은 실제 값이 관측될 때마다 예측 오차를 실시간으로 모니터링하고, 드리프트가 감지되면 남은 예측을 자동으로 교정합니다.

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast(predictions, lower, upper, historicalData, period=7)
healer.observe(actualValues)

report = healer.getReport()
print(f"건강 상태: {report.overallHealth} ({report.healthScore:.1f}/100)")
print(f"총 관측: {report.totalObserved}")
print(f"총 교정: {report.totalCorrected}")

updated_predictions, updated_lower, updated_upper = healer.getUpdatedForecast()
```

**예상 출력:**

```
건강 상태: good (82.5/100)
총 관측: 14
총 교정: 3
```

자가 치유 시스템은 CUSUM과 EWMA 드리프트 감지를 사용하여 예측이 실제 값에서 벗어나는 시점을 자동으로 포착하고, Conformal Prediction 기반 교정을 적용합니다.

## 제약 인식 예측

통계 모델은 수학적으로 최적인 예측을 생성하지만, 비즈니스 현실에는 모델이 알지 못하는 규칙이 있습니다. 재고는 음수가 될 수 없고, 생산량에는 설비 용량 상한이 있으며, 매출은 전년 대비 50% 이상 급변하기 어렵습니다. 제약 인식 예측은 이러한 비즈니스 규칙을 통계적 예측에 반영하여 현실적인 결과를 만들어냅니다.

```python
from vectrix import ConstraintAwareForecaster, Constraint

caf = ConstraintAwareForecaster()
result = caf.apply(predictions, lower95, upper95, constraints=[
    Constraint(type='non_negative', params={}),
    Constraint(type='capacity', params={'max': 5000}),
    Constraint(type='monotone', params={'direction': 'increasing'}),
], smoothing=True)

print(f"원본 평균: {predictions.mean():.2f}")
print(f"교정 평균: {result.predictions.mean():.2f}")
```

### 제약 유형

| 유형 | 설명 |
|------|------|
| `non_negative` | 음수 불가 제약 |
| `range` | 최소/최대 범위 제한 |
| `monotone` | 단조 증가/감소만 허용 |
| `capacity` | 용량 상한 제한 |
| `yoy_change` | 전년 대비 변화율 제한 |
| `sum_constraint` | 합계 제약 |
| `custom` | 사용자 정의 제약 |

> **참고:** 제약은 순서대로 적용됩니다. 서로 충돌하는 제약이 있으면, 나중에 적용되는 제약이 우선합니다.

## Forecast DNA

시계열의 통계적 지문을 추출하여 메타러닝 기반 모델 추천에 활용합니다. 65개 이상의 특성(추세 강도, 계절 강도, Hurst 지수, 변동성 군집, 비선형 자기상관, 엔트로피 등)을 분석하여 데이터의 고유한 DNA 프로파일을 생성합니다.

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

DNA 분석, 레짐 감지, 예측, 제약 적용을 하나의 파이프라인으로 연결하는 예제입니다.

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
        Constraint(type='non_negative', params={}),
        Constraint(type='range', params={'min': 200, 'max': 800}),
    ],
    smoothing=True
)
print(f"\n제약 적용 후 예측 범위: {constrained.predictions.min():.1f} ~ {constrained.predictions.max():.1f}")
```

---
