---
title: 적응형 지능
---

# 적응형 지능

**이 기능들은 Vectrix에만 존재합니다** — statsforecast, Prophet, Darts 등 다른 예측 라이브러리에는 없습니다. 적응형 지능이란, 변화하는 조건에 실시간으로 반응하는 예측 시스템입니다: 레짐 전환 감지, 새 데이터가 도착할 때 자동 교정, 비즈니스 제약 조건 적용, 그리고 데이터 DNA 기반 지능형 모델 선택.

## 레짐 감지

현실 데이터는 하나의 패턴만 따르지 않습니다. 시장은 상승장과 하락장 사이를 오가고, 소매 수요는 성수기와 비수기가 다르며, 정책 변화 이후 데이터 동태가 바뀝니다. Vectrix는 은닉 마르코프 모델(HMM)을 사용하여 이러한 **레짐**을 자동 감지합니다:

```python
from vectrix import RegimeDetector

detector = RegimeDetector(nRegimes=3)
result = detector.detect(data)

print(f"현재 레짐: {result.currentRegime}")
for label, stats in result.regimeStats.items():
    print(f"  {label}: 평균={stats['mean']:.2f}, 표준편차={stats['std']:.2f}")
```

## 레짐 인식 예측

전통적 예측기는 모든 데이터에 하나의 모델을 사용합니다 — 중간에 데이터 동태가 극적으로 바뀌어도 말입니다. `RegimeAwareForecaster`는 현재 레짐을 식별하고, 유사한 과거 레짐에서 가장 잘 작동한 모델을 자동으로 선택합니다:

```python
from vectrix import RegimeAwareForecaster

raf = RegimeAwareForecaster()
result = raf.forecast(data, steps=30, period=7)
print(result.currentRegime)
print(result.modelPerRegime)
```

## 자가 치유 예측

예측은 시간이 지날수록 정확도가 떨어집니다. **자가 치유**는 실제 값이 도착할 때마다 오차를 모니터링하고, 나머지 예측을 자동으로 조정합니다. 처음 3일간 예측이 너무 낙관적이었다면, 나머지 기간에 대해 보정합니다:

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast(predictions, lower, upper, historicalData, period=7)
healer.observe(actual_values)

report = healer.getReport()
print(f"건강 상태: {report.overallHealth} ({report.healthScore:.1f}/100)")
print(f"총 관측: {report.totalObserved}")
print(f"총 교정: {report.totalCorrected}")

updated_predictions, updated_lower, updated_upper = healer.getUpdatedForecast()
```

## 제약 조건 인식 예측

통계 모델은 비즈니스 규칙을 모릅니다. 예측값이 음수가 되거나(판매량에는 불가능), 창고 용량을 초과하거나, 비현실적인 전년 대비 변동을 보일 수 있습니다. **제약 조건 인식 예측**은 기본 모델을 수정하지 않고 도메인 지식을 후처리 규칙으로 적용합니다:

```python
from vectrix import ConstraintAwareForecaster, Constraint

caf = ConstraintAwareForecaster()
result = caf.apply(predictions, lower95, upper95, constraints=[
    Constraint(name='non_negative', type='min', value=0),
    Constraint(name='max_cap', type='max', value=5000),
    Constraint(name='increasing', type='monotonic', value='increasing'),
], smoothing=True)
```

### 제약 조건 유형

| 유형 | 설명 |
|------|------|
| `min` | 최솟값 제한 (예: 음수 불가) |
| `max` | 최댓값 제한 (예: 용량 상한) |
| `range` | 최소/최대 범위 |
| `monotonic` | 증가/감소만 허용 |

## Forecast DNA

모든 시계열에는 고유한 통계적 서명이 있습니다. **DNA 프로파일링**은 65개 이상의 특성 — 자기상관 구조, 허스트 지수, 엔트로피, 변동성 클러스터링, 계절성 강도 등 — 을 추출하여 결정적 지문을 생성합니다. 이 지문이 지능적 모델 선택과 난이도 추정을 구동합니다:

```python
from vectrix import ForecastDNA

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"핑거프린트: {profile.fingerprint}")
print(f"난이도: {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"추천 모델: {profile.recommendedModels}")
```

---

**인터랙티브 튜토리얼:** `marimo run docs/tutorials/ko/05_adaptive.py`
