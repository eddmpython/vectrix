# 쇼케이스 03 — 모델 비교 & 적응형 지능

**30+ 예측 모델을 나란히 비교하고 적응형 기능을 탐색합니다.**

## 개요

Vectrix는 30+ 후보 중 자동으로 최적 모델을 선택하지만, 모든 결과를 직접 확인할 수 있습니다:

- **DNA 분석** — 데이터의 난이도, 카테고리, 핑거프린트 파악
- **모델 순위** — 모든 모델의 성능 확인 (MAPE, RMSE, MAE, sMAPE)
- **전체 예측값** — 모든 모델의 예측을 단일 DataFrame으로 확인
- **빠른 비교** — `compare()`로 한 줄 비교

## 인터랙티브 실행

```bash
pip install vectrix pandas numpy marimo
marimo run docs/showcase/ko/03_modelComparison.py
```

## 코드

### 데이터 생성

```python
import numpy as np
import pandas as pd
from vectrix import forecast, analyze, compare

np.random.seed(42)
n = 120
t = np.arange(n, dtype=np.float64)
trend = 100 + 0.8 * t
seasonal = 25 * np.sin(2 * np.pi * t / 12) + 10 * np.cos(2 * np.pi * t / 6)
noise = np.random.normal(0, 8, n)

salesDf = pd.DataFrame({
    "date": pd.date_range("2015-01-01", periods=n, freq="MS"),
    "revenue": trend + seasonal + noise,
})
```

### DNA 분석

```python
report = analyze(salesDf, date="date", value="revenue")
print(f"카테고리: {report.dna.category}")
print(f"난이도: {report.dna.difficulty} ({report.dna.difficultyScore:.0f}/100)")
print(f"추천 모델: {report.dna.recommendedModels[:5]}")
```

### 예측 & 비교

```python
result = forecast(salesDf, date="date", value="revenue", steps=12)

print(f"최적 모델: {result.model}")
print(f"MAPE: {result.mape:.2f}%")

ranking = result.compare()
print(ranking)
```

### 전체 모델 예측값

```python
allForecasts = result.all_forecasts()
print(allForecasts)
```

`date` 컬럼과 모델별 컬럼이 있는 DataFrame을 반환합니다 — 커스텀 앙상블 구축에 유용합니다.

### 한 줄 비교

```python
comparison = compare(salesDf, date="date", value="revenue", steps=12)
print(comparison)
```

`compare()` 함수는 전체 파이프라인을 실행하고 모델 비교 DataFrame을 직접 반환합니다.

!!! tip "모델 선택 기준"
    모델은 in-sample 적합이 아닌 **out-of-sample** 정확도(교차 검증)로 순위가 매겨집니다.
    CV 분할에 따라 순위가 달라질 수 있습니다.

---

**다음:** [쇼케이스 04 — 비즈니스 인텔리전스](04_businessIntelligence.md)
