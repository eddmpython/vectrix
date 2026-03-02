# 쇼케이스 04 — 비즈니스 인텔리전스

**End-to-end 비즈니스 예측: 이상치 탐지, 시나리오, 백테스팅, 지표.**

## 개요

Vectrix의 `business` 모듈을 활용한 완전한 비즈니스 예측 워크플로우:

1. **이상치 탐지** — 예측 전 비정상 데이터 포인트 발견
2. **예측** — 30+ 모델 파이프라인 실행
3. **What-If 시나리오** — 성장, 경기침체, 충격 시나리오 탐색
4. **백테스팅** — Walk-forward 모델 정확도 검증
5. **비즈니스 지표** — MAPE, RMSE, MAE, 편향, 추적 신호 계산

## 인터랙티브 실행

```bash
pip install vectrix pandas numpy marimo
marimo run docs/showcase/ko/04_businessIntelligence.py
```

## 코드

### 설정

```python
import numpy as np
import pandas as pd
from vectrix import forecast
from vectrix.business import AnomalyDetector, WhatIfAnalyzer, Backtester, BusinessMetrics
from vectrix.engine.ets import AutoETS
```

### 1. 이상치 탐지

```python
np.random.seed(42)
n = 150
t = np.arange(n, dtype=np.float64)
values = 200 + 1.2 * t + 30 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 10, n)
values[45] = 600
values[90] = 50
values[130] = 700

detector = AnomalyDetector()
result = detector.detect(values, sensitivity=0.95)

for idx in result.indices:
    print(f"  인덱스 {idx}: 값={values[idx]:.1f}, 점수={result.scores[idx]:+.2f}")
```

### 2. 예측

```python
bizDf = pd.DataFrame({
    "date": pd.date_range("2013-01-01", periods=n, freq="MS"),
    "revenue": values,
})

fcResult = forecast(bizDf, date="date", value="revenue", steps=12)
print(f"모델: {fcResult.model}, MAPE: {fcResult.mape:.2f}%")
```

### 3. What-If 시나리오

```python
analyzer = WhatIfAnalyzer()
scenarios = [
    {"name": "base", "trend_change": 0},
    {"name": "growth_10pct", "trend_change": 0.10},
    {"name": "recession", "trend_change": -0.15, "level_shift": -0.05},
    {"name": "supply_shock", "shock_at": 3, "shock_magnitude": -0.25, "shock_duration": 3},
]

historical = np.array(bizDf["revenue"], dtype=np.float64)
results = analyzer.analyze(fcResult.predictions, historical, scenarios, period=12)
print(analyzer.compareSummary(results))
```

### 4. 백테스팅

```python
bt = Backtester(nFolds=4, horizon=12, strategy="expanding", minTrainSize=60)
btResult = bt.run(values, modelFactory=AutoETS)
print(bt.summary(btResult))
```

### 5. 비즈니스 지표

```python
actuals = np.array([320, 340, 310, 360, 345])
predicted = np.array([325, 335, 315, 355, 350])

metrics = BusinessMetrics()
result = metrics.calculate(actuals, predicted)
for key, value in result.items():
    print(f"  {key}: {value:.4f}")
```

!!! tip "모범 사례"
    이상치 탐지 + 백테스팅 + What-If 분석을 결합하면 견고한 비즈니스 계획을 세울 수 있습니다.
    이상치를 먼저 감지하고, 모델 정확도를 검증한 후, 시나리오를 탐색하세요.

---

**돌아가기:** [쇼케이스 목록](index.md)
