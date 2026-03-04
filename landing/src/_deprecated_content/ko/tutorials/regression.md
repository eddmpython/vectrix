---
title: "튜토리얼 03 — 회귀분석"
---

# 튜토리얼 03 — 회귀분석

**하나의 함수 호출로 완전한 회귀분석.** R 스타일 수식 문법(`y ~ x1 + x2`), 자동 진단(VIF, Breusch-Pagan, Jarque-Bera, Durbin-Watson), OLS/Ridge/Lasso/Huber/Quantile 5가지 방법론, 예측 구간을 모두 지원합니다. R이나 statsmodels의 복잡한 설정 없이, `regress()` 한 줄로 계수 추정부터 진단 검정까지 수행할 수 있습니다.

## 기본 회귀분석

R 스타일 수식으로 회귀분석을 실행합니다. 종속변수 `~` 독립변수 형식으로 모델을 정의하면, Vectrix가 계수 추정, 유의성 검정, 적합도 평가를 자동으로 수행합니다.

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ ads + price + promo")
print(model.summary())
```

**예상 출력:**

```
=== Regression Summary ===
Method: OLS
R-squared: 0.8542
Adj R-squared: 0.8501
F-statistic: 215.34

Coefficients:
  Intercept:  +125.340  (p=0.0000) ***
  ads:         +2.451   (p=0.0001) ***
  price:       -3.872   (p=0.0023) **
  promo:      +45.123   (p=0.0000) ***
```

## 배열 직접 입력

DataFrame이 없는 경우, NumPy 배열을 직접 전달할 수 있습니다. 수식 대신 종속변수(`y`)와 독립변수 행렬(`X`)을 명시적으로 지정합니다.

```python
import numpy as np
from vectrix import regress

y = np.array([100, 120, 140, 160, 180])
X = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])

model = regress(y=y, X=X)
print(f"R-squared: {model.r_squared:.4f}")
print(f"계수: {model.coefficients}")
```

## 수식 문법

R/Patsy 스타일의 다양한 수식 패턴을 지원합니다. 특정 변수 선택, 모든 변수 포함(`.`), 상호작용 항(`*`), 다항식 변환(`I()`) 등 유연한 모델 정의가 가능합니다.

```python
from vectrix import regress

regress(data=df, formula="y ~ x1 + x2")       # 특정 변수
regress(data=df, formula="y ~ .")              # 모든 변수
regress(data=df, formula="y ~ x1 * x2")       # 상호작용 항
regress(data=df, formula="y ~ x + I(x**2)")   # 다항식
```

## 회귀 방법

데이터의 특성에 따라 5가지 회귀 방법 중 적합한 것을 선택할 수 있습니다. 다중공선성이 의심되면 Ridge, 변수 선택이 필요하면 Lasso, 이상치가 많으면 Huber, 조건부 분위수를 추정하려면 Quantile을 사용합니다.

| 방법 | 설명 |
|------|------|
| `ols` | 최소제곱법 (기본값) |
| `ridge` | L2 정규화 |
| `lasso` | L1 정규화 |
| `huber` | 강건 회귀 (이상치에 덜 민감) |
| `quantile` | 분위수 회귀 |

```python
model = regress(data=df, formula="sales ~ ads + price", method="ridge")
print(f"방법: Ridge")
print(f"R-squared: {model.r_squared:.4f}")
```

## 결과 확인

회귀 결과 객체는 R-squared, 수정 R-squared, F-통계량, 계수, p-값 등 모델 평가에 필요한 모든 속성을 제공합니다.

```python
print(f"R-squared:     {model.r_squared:.4f}")
print(f"Adj R-squared: {model.adj_r_squared:.4f}")
print(f"F-statistic:   {model.f_stat:.2f}")
print(f"계수: {model.coefficients}")
print(f"P-값: {model.pvalues}")
```

## 진단 검정

회귀 모델의 신뢰성은 가정의 충족 여부에 달려 있습니다. `diagnose()` 메서드는 다중공선성(VIF), 이분산성(Breusch-Pagan), 잔차 정규성(Jarque-Bera), 자기상관(Durbin-Watson) 등 4가지 핵심 진단 검정을 한 번에 수행하고, 각 결과를 해석과 함께 보여줍니다.

```python
print(model.diagnose())
```

**예상 출력:**

```
=== Regression Diagnostics ===

VIF (다중공선성):
  ads:   1.23  OK
  price: 2.45  OK
  promo: 1.67  OK

Breusch-Pagan (이분산성):
  Statistic: 3.45, p-value: 0.0631
  Result: 이분산성 없음 (p > 0.05)

Jarque-Bera (잔차 정규성):
  Statistic: 1.23, p-value: 0.5401
  Result: 정규성 만족

Durbin-Watson (자기상관):
  Statistic: 1.95
  Result: 자기상관 없음 (1.5 < DW < 2.5)
```

각 검정이 진단하는 문제와 경고 기준입니다.

| 검정 | 대상 | 경고 기준 |
|------|------|----------|
| **VIF** | 다중공선성 | VIF > 10이면 문제 |
| **Breusch-Pagan** | 이분산성 | p가 0.05 미만이면 이분산 존재 |
| **Jarque-Bera** | 잔차 정규성 | p가 0.05 미만이면 비정규 |
| **Durbin-Watson** | 자기상관 | 2에서 멀수록 자기상관 존재 |

## 예측

학습된 모델을 새로운 독립변수 값에 적용하여 종속변수를 예측합니다. 마케팅 예산 시나리오별 매출 추정, 가격 변경에 따른 수요 예측 등에 활용됩니다.

```python
import pandas as pd

newData = pd.DataFrame({
    "ads": [50, 75, 90],
    "price": [20, 15, 10],
    "promo": [0, 1, 1],
})

predictions = model.predict(newData)
print(predictions)
```

**예상 출력:**

```
   prediction
0      285.3
1      412.7
2      523.1
```

## 계수 해석

계수(coefficient)는 각 독립변수가 종속변수에 미치는 영향의 크기와 방향을 나타냅니다. p-값이 작을수록 해당 변수의 효과가 통계적으로 유의합니다. 계수 테이블을 사용자 정의 형식으로 출력하여 비즈니스 맥락에서 해석할 수 있습니다.

```python
labels = ["절편", "광고비", "가격", "프로모션"]

for name, coef, pval in zip(labels, model.coefficients, model.pvalues):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{name:12s}  {coef:+10.3f}  p={pval:.4f} {sig}")
```

## 완전한 예제

마케팅 데이터에서 수식 기반 회귀분석, 진단 검정, 새 데이터 예측까지 전체 워크플로우를 하나로 연결하는 예제입니다.

```python
import pandas as pd
from vectrix import regress

df = pd.read_csv("marketing.csv")

model = regress(
    data=df,
    formula="revenue ~ tvAds + digitalAds + price + seasonality",
)

print(f"R-squared: {model.r_squared:.4f}")
print(f"Adj R-squared: {model.adj_r_squared:.4f}")

print()
print("=== 진단 ===")
print(model.diagnose())

print()
print("=== 예측 ===")
newData = pd.DataFrame({
    "tvAds": [100, 200],
    "digitalAds": [500, 800],
    "price": [29.99, 24.99],
    "seasonality": [1, 0],
})
print(model.predict(newData))
```

---
