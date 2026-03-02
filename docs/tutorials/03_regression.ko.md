# 튜토리얼 03 — 회귀분석

**statsmodels 수준의 회귀분석을 한 줄로.**

Vectrix의 `regress()` 함수는 R 스타일 공식, 다양한 방법(OLS, Ridge, Lasso, Huber, Quantile), 완전한 진단, 예측 구간을 지원합니다 — statsmodels 설치 없이.

## 1. 직접 입력 모드

y(종속변수)와 X(독립변수) 배열을 직접 전달합니다:

```python
import numpy as np
from vectrix import regress

np.random.seed(42)
n = 100
x1 = np.random.uniform(10, 50, n)
x2 = np.random.uniform(1, 10, n)
noise = np.random.normal(0, 5, n)
y = 20 + 3.5 * x1 - 2.0 * x2 + noise

X = np.column_stack([x1, x2])

result = regress(y, X)
```

자동으로 계수, 표준오차, t-값, p-값이 포함된 요약 테이블이 출력됩니다.

### 핵심 결과

```python
print(f"R²:          {result.r_squared:.4f}")
print(f"조정 R²:     {result.adj_r_squared:.4f}")
print(f"F-통계량:    {result.f_stat:.2f}")
print(f"계수:        {result.coefficients}")
print(f"P-값:        {result.pvalues}")
```

```
R²:          0.9812
조정 R²:     0.9808
F-통계량:    2531.45
계수:        [19.23  3.51 -2.13]
P-값:        [0.000 0.000 0.001]
```

## 2. 공식 모드

R 스타일 공식 문자열과 DataFrame을 사용합니다 — 더 읽기 쉽고 강력합니다:

```python
import pandas as pd
from vectrix import regress

df = pd.DataFrame({"sales": y, "ads": x1, "price": x2})

result = regress(data=df, formula="sales ~ ads + price")
```

### 공식 문법

| 문법 | 예시 | 의미 |
|------|------|------|
| 기본 | `"y ~ x1 + x2"` | 선형 회귀 |
| 전체 컬럼 | `"y ~ ."` | y 외 모든 수치 컬럼 사용 |
| 상호작용 | `"y ~ x1 * x2"` | x1 + x2 + x1:x2 |
| 교차항만 | `"y ~ x1 : x2"` | x1·x2 곱만 |
| 다항식 | `"y ~ x + I(x**2)"` | 제곱항 추가 |
| 혼합 | `"sales ~ ads + I(ads**2) + price"` | 선형 + 비선형 혼합 |

### 다항식 예제

```python
np.random.seed(42)
x = np.random.uniform(0, 10, 80)
y = 5 + 2 * x - 0.3 * x**2 + np.random.normal(0, 2, 80)

df = pd.DataFrame({"y": y, "x": x})
result = regress(data=df, formula="y ~ x + I(x**2)")
```

## 3. 진단

VIF, 정규성, 등분산성, 자기상관, 영향력 분석을 한 번에 실행합니다:

```python
print(result.diagnose())
```

```
============================================
     Regression Diagnostics Report
============================================

  [Multicollinearity - VIF]
    ads:   1.02  (OK)
    price: 1.02  (OK)

  [Normality of Residuals]
    Shapiro-Wilk: W=0.993, p=0.891
    → 잔차가 정규분포를 따름

  [Homoscedasticity]
    Breusch-Pagan: stat=2.14, p=0.343
    → 이분산성 없음

  [Autocorrelation]
    Durbin-Watson: 2.03
    → 자기상관 없음

  [Influential Points]
    High leverage: 2 points
    High Cook's D: 0 points
============================================
```

## 4. 예측

새 데이터에 대해 신뢰구간 또는 예측구간과 함께 예측합니다:

```python
X_new = np.array([[30, 5], [40, 3], [25, 8]])

pred_df = result.predict(X_new, interval="prediction", alpha=0.05)
print(pred_df)
```

```
   prediction      lower      upper
0      115.23      104.89     125.57
1      149.87      139.45     160.29
2       91.45       81.12     101.78
```

### 구간 유형

| 유형 | 의미 |
|------|------|
| `'prediction'` | 새로운 개별 관측값의 구간 (더 넓음) |
| `'confidence'` | 평균 반응의 구간 (더 좁음) |
| `'none'` | 구간 없이 예측값만 |

## 5. 회귀 방법

```python
result_ols    = regress(y, X, method="ols")      # 기본값
result_ridge  = regress(y, X, method="ridge")    # L2 정규화
result_lasso  = regress(y, X, method="lasso")    # L1 정규화
result_huber  = regress(y, X, method="huber")    # 이상치에 강건
result_quant  = regress(y, X, method="quantile") # 중앙값 회귀
```

## 6. 자동 출력 끄기

기본적으로 `regress()`는 요약을 자동 출력합니다. 끄려면:

```python
result = regress(y, X, summary=False)
```

## 7. 결과 객체 레퍼런스

| 속성 / 메서드 | 타입 | 설명 |
|---|---|---|
| `.coefficients` | `np.ndarray` | 회귀 계수 (절편 포함) |
| `.pvalues` | `np.ndarray` | 각 계수의 p-값 |
| `.r_squared` | `float` | R² (결정계수) |
| `.adj_r_squared` | `float` | 조정 R² |
| `.f_stat` | `float` | F-통계량 |
| `.summary()` | `str` | 전체 회귀 테이블 |
| `.diagnose()` | `str` | VIF + 정규성 + 등분산성 + 자기상관 + 영향력 분석 |
| `.predict(X, interval, alpha)` | `DataFrame` | prediction, lower, upper 컬럼 |

---

**다음:** [튜토리얼 04 — 30+ 모델](04_models.ko.md) — 직접 모델 접근과 비교
