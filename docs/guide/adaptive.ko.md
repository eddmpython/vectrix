# 적응형 지능

Vectrix만의 차별화 기능 — 다른 예측 라이브러리에는 없습니다.

## 레짐 감지

HMM 기반 레짐 전환 감지 (상승/하락/횡보, 성수기/비수기):

```python
from vectrix import RegimeDetector

detector = RegimeDetector(nRegimes=3)
result = detector.detect(data)

print(f"현재 레짐: {result.currentRegime}")
for label, stats in result.regimeStats.items():
    print(f"  {label}: 평균={stats['mean']:.2f}, 표준편차={stats['std']:.2f}")
```

## 레짐 인식 예측

레짐별로 최적 모델을 자동 전환:

```python
from vectrix import RegimeAwareForecaster

raf = RegimeAwareForecaster()
result = raf.forecast(data, steps=30, period=7)
print(result.currentRegime)
print(result.modelPerRegime)
```

## 자가 치유 예측

실시간 오차 모니터링 및 자동 교정:

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast(predictions, lower, upper, historical_data)
healer.observe(actual_values)

report = healer.getReport()
print(f"건강 상태: {report.overallHealth} ({report.healthScore:.1f}/100)")
print(f"개선율: {report.improvementPct:.1f}%")

updated = healer.getUpdatedForecast()
```

## 제약 조건 인식 예측

비즈니스 제약 조건 적용:

```python
from vectrix import ConstraintAwareForecaster, Constraint

caf = ConstraintAwareForecaster()
result = caf.apply(predictions, lower, upper, constraints=[
    Constraint('non_negative', {}),
    Constraint('range', {'min': 0, 'max': 5000}),
    Constraint('capacity', {'capacity': 10000}),
    Constraint('yoy_change', {'maxPct': 30, 'historicalData': past_year}),
])
```

### 제약 조건 유형

| 유형 | 설명 |
|------|------|
| `non_negative` | 음수 불가 |
| `range` | 최소/최대 범위 |
| `capacity` | 용량 상한 |
| `yoy_change` | 전년 대비 변화율 제한 |
| `sum` | 합계 제약 |
| `monotone` | 증가/감소만 허용 |
| `ratio` | 시리즈 간 비율 |
| `custom` | 커스텀 함수 |

## Forecast DNA

메타러닝을 위한 시계열 핑거프린트:

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
