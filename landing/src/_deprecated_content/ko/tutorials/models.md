---
title: "튜토리얼 04 — 30+ 모델"
---

# 튜토리얼 04 — 30+ 모델

Vectrix는 10개 카테고리에 걸쳐 30개 이상의 예측 모델을 내장하고 있습니다. 지수 평활(ETS), ARIMA, Theta 분해, 복소수 지수 평활(CES), 동적 최적화 Theta(DOT), 삼각함수 분해(TBATS), 다중 계절 분해(MSTL), 간헐적 수요(Croston), 변동성 모델(GARCH), 그리고 패턴 매칭(DTSF), Echo State Network(ESN) 등 최신 모델까지 — 하나의 라이브러리에서 모두 사용할 수 있습니다. 자동 선택, 수동 지정, 비교 분석까지 모든 워크플로우를 지원합니다.

## 자동 모델 선택

`forecast()` 함수는 데이터에 호환되는 모든 모델을 시간 순서 교차 검증으로 평가하고, 검증 오차가 가장 낮은 최적 모델을 자동으로 선택합니다. 사용자가 직접 모델을 고를 필요가 없습니다.

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170, 180, 195], steps=5)
print(f"선택된 모델: {result.model}")
print(f"MAPE: {result.mape:.2f}%")
```

## 모델 비교

`compare()` 메서드로 평가된 모든 모델의 sMAPE, MAPE, RMSE, MAE를 한눈에 비교할 수 있습니다. 반환되는 pandas DataFrame을 정렬, 필터링, 시각화하여 모델 성능을 분석하세요.

```python
from vectrix import forecast

result = forecast(data, steps=12)
comparison = result.compare()
print(comparison)
```

**예상 출력:**

```
              sMAPE     MAPE     RMSE      MAE
DOT           3.214    3.187   12.453    9.876
AutoCES       3.456    3.421   13.102   10.234
AutoETS       3.789    3.752   14.567   11.345
FourTheta     3.891    3.844   14.892   11.678
...
```

`compare()` 함수를 사용하면 더 간결하게 비교할 수 있습니다

```python
from vectrix import compare

ranking = compare(data, steps=12)
print(ranking)
```

## 모든 모델의 예측값

각 모델이 생성한 개별 예측값을 모두 확인할 수 있습니다. 모델 간 예측 패턴의 차이를 시각적으로 비교하거나, 사용자 정의 앙상블 가중치를 설계할 때 활용됩니다.

```python
allForecasts = result.all_forecasts()
print(allForecasts)
```

**예상 출력:**

```
   step        DOT    AutoCES    AutoETS  FourTheta  ...
0     1    152.340    153.120    151.890    154.230  ...
1     2    155.670    156.440    154.230    157.560  ...
2     3    158.120    159.890    157.560    160.120  ...
...
```

## Vectrix 클래스 직접 사용

Easy API(`forecast()`)는 간편하지만, 학습/검증 비율 조정, 개별 모델 결과 접근, Flat Defense 진단 등 세밀한 제어가 필요할 때는 `Vectrix` 클래스를 직접 사용합니다. 프로덕션 파이프라인이나 고급 분석 워크플로우에 적합합니다.

```python
from vectrix import Vectrix

vx = Vectrix()
result = vx.forecast(
    df,
    dateCol="date",
    valueCol="sales",
    steps=14,
    trainRatio=0.8
)

print(result.bestModelName)
print(result.predictions)
```

### 개별 모델 결과 확인

```python
for modelId, mr in result.allModelResults.items():
    print(f"{mr.modelName}: MAPE={mr.mape:.2f}%")
```

## 모델 카테고리

| 카테고리 | 모델 | 적합 대상 |
|---------|------|----------|
| **지수 평활** | AutoETS, ETS 변형 | 안정적인 패턴, 추세와 계절성의 가법/승법 조합 |
| **ARIMA** | AutoARIMA | 정상 시계열, 자기상관 구조가 있는 데이터 |
| **분해** | MSTL, AutoMSTL | 다중 계절성 (예: 일별 + 주별 + 연간) |
| **Theta** | Theta, DOT, FourTheta | 범용 — DOT는 M4 100K 벤치마크 OWA 0.905(최상위), FourTheta는 4개 theta line 가중 조합으로 Yearly 데이터에 특히 강세 |
| **삼각함수** | TBATS | 복잡한 계절성, 비정수 주기 |
| **복소수** | AutoCES | 비선형 패턴, 복소수 지수 평활 — M4 OWA 0.927 |
| **간헐적 수요** | Croston, SBA, TSB | 0이 빈번한 희소 수요 데이터 |
| **변동성** | GARCH, EGARCH, GJR | 금융 수익률, 변동성 군집이 있는 데이터 |
| **신규** | DTSF, ESN, FourTheta | DTSF는 비모수 패턴 매칭(Hourly 강세), ESN은 Echo State Network로 앙상블 다양성 기여, FourTheta는 M4 3위 방법론 재현 |
| **기준선** | Naive, Seasonal, Mean, RWD | 벤치마크 기준선 — 다른 모델의 성능을 평가하는 비교 대상 |

## 엔진 직접 접근

특정 모델을 직접 인스턴스화하여 fit/predict 파이프라인을 제어할 수 있습니다. 연구, 실험, 또는 커스텀 앙상블을 구성할 때 유용합니다.

```python
from vectrix.engine.ets import AutoETS
from vectrix.engine.arima import AutoARIMA
from vectrix.engine.dot import DynamicOptimizedTheta

ets = AutoETS(period=7)
ets.fit(data)
predictions, lower, upper = ets.predict(steps=30)
print(f"AutoETS 예측: {predictions[:5]}")
```

```python
dot = DynamicOptimizedTheta(period=12)
dot.fit(data)
predictions, lower, upper = dot.predict(steps=12)
print(f"DOT 예측: {predictions[:5]}")
```

## Flat Defense 시스템

많은 예측 모델이 어려운 데이터에서 모든 미래 값을 동일한 상수로 예측하는 "평탄 예측(flat forecast)" 문제를 겪습니다. 이는 실무적으로 무의미한 결과입니다. Vectrix는 이 문제를 해결하기 위해 고유한 4단계 방어 시스템을 내장하고 있습니다.

1. **FlatRiskDiagnostic** -- 데이터의 특성을 분석하여 평탄 예측이 발생할 위험도를 사전에 평가합니다.
2. **AdaptiveModelSelector** -- 평탄 위험이 높은 경우, 평탄 예측에 강건한 모델을 우선 선택합니다.
3. **FlatPredictionDetector** -- 예측이 생성된 후, 결과가 실질적으로 평탄한지 자동 감지합니다.
4. **FlatPredictionCorrector** -- 평탄 예측이 감지되면, 데이터의 추세와 계절성을 반영하여 자동 교정합니다.

```python
result = vx.forecast(df, dateCol="date", valueCol="value", steps=14)
fr = result.flatRisk
print(f"위험: {fr.riskLevel.name} ({fr.riskScore:.2f})")
print(f"전략: {fr.recommendedStrategy}")
```

## DNA 기반 모델 추천

DNA 메타러닝은 4단계로 작동합니다: (1) 65개 이상의 통계 특성을 추출하고, (2) 데이터를 카테고리(seasonal, trending, volatile, intermittent, stationary)로 분류하며, (3) 난이도 점수(0-100)를 산출하고, (4) 데이터 유형에 최적화된 모델 순위를 추천합니다. 분석 결과를 활용하여 예측 전에 어떤 모델이 적합한지 미리 확인할 수 있습니다.

```python
from vectrix import analyze, forecast

report = analyze(df, date="date", value="sales")
print(f"DNA 난이도: {report.dna.difficulty}")
print(f"추천 모델: {report.dna.recommendedModels[:3]}")

result = forecast(df, date="date", value="sales", steps=12)
print(f"\n최적 모델: {result.model}")
print(f"\n전체 모델 비교:")
print(result.compare())
```

## 완전한 워크플로우

DNA 분석부터 모델 비교, 예측, 결과 내보내기까지 전체 파이프라인을 하나로 연결하는 예제입니다.

```python
import pandas as pd
from vectrix import forecast, analyze

df = pd.read_csv("monthly_sales.csv")

report = analyze(df, date="date", value="sales")
print(f"카테고리: {report.dna.category}")
print(f"난이도: {report.dna.difficulty}")
print(f"추천: {report.dna.recommendedModels[:3]}")

result = forecast(df, date="date", value="sales", steps=12)
print(f"\n최적 모델: {result.model}")
print(result.compare())
print(result.all_forecasts())

result.to_dataframe().to_csv("forecast_output.csv", index=False)
```

> **참고:** `compare()` 출력은 표준 pandas DataFrame입니다. 정렬, 필터링, 내보내기 등 모든 DataFrame 연산을 사용할 수 있습니다.

---
