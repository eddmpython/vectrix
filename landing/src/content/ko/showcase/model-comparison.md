---
title: "모델 비교"
---

# 모델 비교

단일 함수 호출로 30개 이상의 예측 모델을 나란히 비교합니다. Vectrix는 호환되는 모든 모델을 자동으로 평가하고 정확도 순으로 정렬합니다.

## 빠른 비교

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170, 180, 195], steps=5)
print(result.compare())
```

sMAPE, MAPE, RMSE, MAE 지표가 포함된 전체 모델 비교 테이블을 출력합니다.

## DNA 분석 먼저

모델 비교 전에 DNA 프로파일링으로 데이터를 파악합니다:

```python
from vectrix import analyze

report = analyze([100, 120, 130, 115, 140, 160, 150, 170, 180, 195])

print(f"추세: {report.characteristics.hasTrend}")
print(f"계절성: {report.characteristics.hasSeasonality}")
print(f"변동성: {report.characteristics.volatilityLevel}")
print(f"난이도: {report.dna.difficulty}")
print(f"추천 모델: {report.dna.recommendedModels[:5]}")
```

DNA 프로파일링은 65개 이상의 시계열 특성을 추출하고, 메타러닝 시스템을 기반으로 최적 모델을 추천합니다.

## 예측 및 비교

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=12)

print(f"최적 모델: {result.model}")
print(f"sMAPE: {result.smape:.2f}")
print(f"MAPE:  {result.mape:.2f}")
print(f"RMSE:  {result.rmse:.2f}")

comparison = result.compare()
print(comparison)
```

`compare()` 메서드는 DataFrame을 반환합니다:

```
              sMAPE     MAPE     RMSE      MAE
DOT           3.214    3.187   12.453    9.876
AutoCES       3.456    3.421   13.102   10.234
AutoETS       3.789    3.752   14.567   11.345
FourTheta     3.891    3.844   14.892   11.678
...
```

## 모든 모델의 예측값

각 모델이 생성한 원시 예측값을 확인합니다:

```python
allForecasts = result.all_forecasts()
print(allForecasts)
```

```
   step        DOT    AutoCES    AutoETS  FourTheta  ...
0     1    152.340    153.120    151.890    154.230  ...
1     2    155.670    156.440    154.230    157.560  ...
2     3    158.120    159.890    157.560    160.120  ...
...
```

## 내장 모델 목록

Vectrix에 포함된 30개 이상의 모델:

**통계 모델**
- AutoETS (30가지 상태 공간 조합)
- AutoARIMA (단계적 차수 선택)
- Theta / Dynamic Optimized Theta (DOT)
- AutoCES (복소 지수 평활)
- FourTheta (적응형 Theta 앙상블)
- AutoTBATS (삼각함수 다중 계절성)
- AutoMSTL (다중 계절 STL 분해)

**변동성 모델**
- GARCH / EGARCH / GJR-GARCH

**간헐적 수요 모델**
- Croston Classic / SBA / TSB / AutoCroston

**신규 모델**
- DTSF (Dynamic Time Scan Forecaster)
- ESN (Echo State Network)
- Lotka-Volterra Ensemble
- Phase Transition Forecaster
- Hawkes Intermittent Demand

**기준선 모델**
- Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

## 완전한 워크플로우

```python
import pandas as pd
from vectrix import forecast, analyze

df = pd.read_csv("monthly_sales.csv")

report = analyze(df, date="date", value="sales")
print(f"DNA 난이도: {report.dna.difficulty}")
print(f"추천: {report.dna.recommendedModels[:3]}")

result = forecast(df, date="date", value="sales", steps=12)
print(f"\n최적 모델: {result.model}")
print(f"\n전체 모델 비교:")
print(result.compare())

result.to_dataframe().to_csv("forecast_output.csv", index=False)
```

> **참고:** `compare()` 출력은 표준 pandas DataFrame입니다. 정렬, 필터링, 내보내기 등 모든 DataFrame 연산을 사용할 수 있습니다.
