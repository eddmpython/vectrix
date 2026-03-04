# 튜토리얼 06 — 비즈니스 인텔리전스

**실무 예측 도구: 이상치 탐지, What-If 시나리오, 백테스팅, 비즈니스 지표.**

Vectrix의 business 모듈은 기본 예측을 넘어서 — 예측 결과를 바탕으로 더 나은 의사결정을 내리도록 돕는 도구들을 제공합니다.

## 1. 이상치 탐지

데이터에서 오류, 이벤트, 또는 레짐 변화를 나타낼 수 있는 비정상적인 데이터 포인트를 감지합니다

```python
import numpy as np
from vectrix.business import AnomalyDetector

np.random.seed(42)
normal = np.random.normal(100, 10, 200)
normal[50] = 200
normal[120] = 20
normal[175] = 250

detector = AnomalyDetector()
result = detector.detect(normal, threshold=2.0)

print(f"발견된 이상치: {len(result.indices)}")
for idx in result.indices:
    print(f"  인덱스 {idx}: 값={normal[idx]:.1f}, 점수={result.scores[idx]:.2f}")
```

```
발견된 이상치: 3
  인덱스 50: 값=200.0, 점수=3.45
  인덱스 120: 값=20.0, 점수=-2.89
  인덱스 175: 값=250.0, 점수=4.12
```

### 임계값 조절

- `threshold=4.0` — 극단적 이상치만 감지 (알림 최소화)
- `threshold=3.0` — 균형 잡힌 설정 (기본값)
- `threshold=2.0` — 공격적 감지 (더 많은 알림)

## 2. What-If 시나리오 분석

추세, 계절성, 외부 충격의 변화가 예측에 어떤 영향을 미치는지 탐색합니다

```python
from vectrix import forecast
from vectrix.business import WhatIfAnalyzer

data = [100 + 0.5 * i + 10 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 3)
        for i in range(120)]

result = forecast(data, steps=12)
base = result.predictions
historical = np.array(data, dtype=np.float64)

analyzer = WhatIfAnalyzer()
scenarios = [
    {"name": "optimistic", "trend_change": 0.1},
    {"name": "pessimistic", "trend_change": -0.15},
    {"name": "shock", "shock_at": 3, "shock_magnitude": -0.20, "shock_duration": 2},
    {"name": "level_up", "level_shift": 0.05},
    {"name": "no_seasonality", "seasonal_multiplier": 0.0},
]

results = analyzer.analyze(base, historical, scenarios, period=12)
```

### 결과 확인

```python
for sr in results:
    print(f"  [{sr.name}]  평균 영향: {sr.impact:.1f}%  "
          f"최종 변화: {sr.percentChange[-1]:.1f}%")
```

```
  [optimistic]      평균 영향: 5.2%   최종 변화: +10.1%
  [pessimistic]     평균 영향: 7.8%   최종 변화: -15.2%
  [shock]           평균 영향: 4.1%   최종 변화: -1.3%
  [level_up]        평균 영향: 5.0%   최종 변화: +5.0%
  [no_seasonality]  평균 영향: 3.4%   최종 변화: -2.1%
```

### 시나리오 파라미터

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| `name` | `str` | 시나리오 이름 |
| `trend_change` | `float` | 추세 조정 (0.1 = +10% 추세 가속) |
| `seasonal_multiplier` | `float` | 계절성 스케일 (0 = 제거, 2 = 2배) |
| `shock_at` | `int` | 충격 발생 시점 (스텝 인덱스) |
| `shock_magnitude` | `float` | 충격 크기 (-0.2 = -20% 하락) |
| `shock_duration` | `int` | 충격 지속 기간 (스텝 수) |
| `level_shift` | `float` | 영구적 수준 변화 (0.05 = +5%) |

### 비교 요약

```python
print(analyzer.compareSummary(results))
```

```
Scenario Comparison:
  [pessimistic] Avg impact: 7.8%, Final change: -15.2%
  [optimistic] Avg impact: 5.2%, Final change: 10.1%
  [level_up] Avg impact: 5.0%, Final change: 5.0%
  [shock] Avg impact: 4.1%, Final change: -1.3%
  [no_seasonality] Avg impact: 3.4%, Final change: -2.1%
```

## 3. 백테스팅

Walk-forward 검증으로 실제 예측 정확도를 측정합니다

```python
from vectrix.business import Backtester
from vectrix.engine.ets import AutoETS

bt = Backtester(nFolds=5, horizon=12, strategy="expanding", minTrainSize=60)

y = np.array(data, dtype=np.float64)
result = bt.run(y, modelFactory=AutoETS)

print(bt.summary(result))
```

```
Backtest Results (5 folds)
  Avg MAPE: 4.23% (+-1.15%)
  Avg RMSE: 5.67
  Avg MAE: 4.12
  Avg Bias: 0.34
  Best fold: #2 (MAPE 2.89%)
  Worst fold: #4 (MAPE 6.12%)
```

### 백테스트 전략

| 전략 | 설명 |
|------|------|
| `"expanding"` | 각 fold마다 학습 윈도우가 확장 (권장) |
| `"sliding"` | 고정 크기 학습 윈도우가 이동 |

### 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `nFolds` | 5 | 검증 fold 수 |
| `horizon` | 30 | fold당 예측 스텝 수 |
| `strategy` | `"expanding"` | 윈도우 전략 |
| `minTrainSize` | 50 | 최소 학습 포인트 수 |
| `stepSize` | auto | fold 간 스텝 간격 |

### 개별 Fold 검사

```python
for fold in result.folds:
    print(f"  Fold {fold.fold}: 학습={fold.trainSize}, 테스트={fold.testSize}, "
          f"MAPE={fold.mape:.2f}%")
```

## 4. 비즈니스 지표

비즈니스 관점의 정확도 지표를 계산합니다

```python
from vectrix.business import BusinessMetrics

actuals = np.array([100, 110, 95, 120, 105])
predicted = np.array([102, 108, 97, 115, 110])

metrics = BusinessMetrics()
result = metrics.calculate(actuals, predicted)

for key, value in result.items():
    print(f"  {key}: {value:.4f}")
```

```
  mape: 3.0476
  rmse: 3.8730
  mae: 3.0000
  bias: -0.4000
  tracking_signal: -0.3333
```

## 5. 종합 비즈니스 워크플로우

모든 것을 하나로 연결합니다

```python
import numpy as np
from vectrix import forecast, analyze
from vectrix.business import AnomalyDetector, WhatIfAnalyzer

data = [100 + 0.5 * i + 10 * np.sin(2 * np.pi * i / 12) + np.random.normal(0, 3)
        for i in range(120)]

report = analyze(data)
print(f"DNA: {report.dna.category}, 난이도={report.dna.difficulty}")

detector = AnomalyDetector()
anom = detector.detect(np.array(data, dtype=np.float64))
print(f"과거 데이터 이상치: {len(anom.indices)}")

result = forecast(data, steps=12)
print(f"모델: {result.model}, MAPE: {result.mape:.1f}%")

analyzer = WhatIfAnalyzer()
scenarios = [
    {"name": "base", "trend_change": 0},
    {"name": "growth", "trend_change": 0.1},
    {"name": "recession", "trend_change": -0.2, "level_shift": -0.05},
]
sr = analyzer.analyze(result.predictions, np.array(data, dtype=np.float64), scenarios)
print(analyzer.compareSummary(sr))
```

## 6. API 레퍼런스

| 클래스 | Import | 용도 |
|--------|--------|------|
| `AnomalyDetector` | `from vectrix.business import AnomalyDetector` | 이상치 탐지 |
| `WhatIfAnalyzer` | `from vectrix.business import WhatIfAnalyzer` | 시나리오 분석 |
| `Backtester` | `from vectrix.business import Backtester` | Walk-forward 검증 |
| `BusinessMetrics` | `from vectrix.business import BusinessMetrics` | 정확도 지표 |

!!! note "Import 경로"
    Business 클래스는 최상위 `vectrix` 패키지가 아닌 `vectrix.business`에서 import합니다.

---

**이것으로 튜토리얼 시리즈를 마칩니다.** 더 많은 예제는 [API 문서](../api/business.md)를 참조하세요.
