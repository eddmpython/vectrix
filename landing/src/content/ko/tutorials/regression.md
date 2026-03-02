---
title: "튜토리얼 03 — 회귀분석"
---

# 튜토리얼 03 — 회귀분석

Vectrix는 R 스타일 수식 인터페이스를 제공하여 직관적인 회귀분석을 지원합니다. OLS부터 강건 회귀까지 5가지 방법과 완전한 진단 도구를 제공합니다.

## 기본 회귀분석

R 스타일 수식으로 회귀분석을 실행합니다:

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

DataFrame이 없는 경우, 배열을 직접 전달할 수 있습니다:

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

다양한 수식 패턴을 지원합니다:

```python
from vectrix import regress

regress(data=df, formula="y ~ x1 + x2")       # 특정 변수
regress(data=df, formula="y ~ .")              # 모든 변수
regress(data=df, formula="y ~ x1 * x2")       # 상호작용 항
regress(data=df, formula="y ~ x + I(x**2)")   # 다항식
```

## 회귀 방법

5가지 회귀 방법을 지원합니다:

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

회귀 결과 객체가 제공하는 속성들:

```python
print(f"R-squared:     {model.r_squared:.4f}")
print(f"Adj R-squared: {model.adj_r_squared:.4f}")
print(f"F-statistic:   {model.f_stat:.2f}")
print(f"계수: {model.coefficients}")
print(f"P-값: {model.pvalues}")
```

## 진단 검정

`diagnose()` 메서드는 4가지 핵심 진단 검정을 수행합니다:

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

각 검정의 의미:

| 검정 | 대상 | 경고 기준 |
|------|------|----------|
| **VIF** | 다중공선성 | VIF > 10이면 문제 |
| **Breusch-Pagan** | 이분산성 | p가 0.05 미만이면 이분산 존재 |
| **Jarque-Bera** | 잔차 정규성 | p가 0.05 미만이면 비정규 |
| **Durbin-Watson** | 자기상관 | 2에서 멀수록 자기상관 존재 |

## 예측

새 데이터에 대해 예측을 수행합니다:

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

계수 테이블을 사용자 정의 형식으로 출력할 수 있습니다:

```python
labels = ["절편", "광고비", "가격", "프로모션"]

for name, coef, pval in zip(labels, model.coefficients, model.pvalues):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{name:12s}  {coef:+10.3f}  p={pval:.4f} {sig}")
```

## 완전한 예제

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

print("\n=== 진단 ===")
print(model.diagnose())

print("\n=== 예측 ===")
newData = pd.DataFrame({
    "tvAds": [100, 200],
    "digitalAds": [500, 800],
    "price": [29.99, 24.99],
    "seasonality": [1, 0],
})
print(model.predict(newData))
```

---
