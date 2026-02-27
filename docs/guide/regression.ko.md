# 회귀분석

## R 스타일 수식

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ ads + price + promo")
print(model.summary())
```

## 방법

| 방법 | 설명 |
|------|------|
| `ols` | 최소자승법 (기본값) |
| `ridge` | L2 정규화 |
| `lasso` | L1 정규화 |
| `huber` | 로버스트 회귀 |
| `quantile` | 분위 회귀 |

```python
model = regress(data=df, formula="sales ~ ads + price", method="ridge")
```

## 결과

```python
print(model.r_squared)        # R²
print(model.adj_r_squared)    # Adjusted R²
print(model.f_stat)           # F-통계량
print(model.coefficients)     # 계수 배열
print(model.pvalues)          # p-값 배열
```

## 진단

```python
print(model.diagnose())
```

다음 항목을 포함한 텍스트 리포트:

- **VIF**: 다중공선성 (10 이상이면 문제)
- **Breusch-Pagan**: 이분산성 검정
- **Jarque-Bera**: 잔차 정규성 검정
- **Durbin-Watson**: 자기상관 검정

## 예측

```python
import pandas as pd

new_data = pd.DataFrame({
    "ads": [50, 75, 90],
    "price": [20, 15, 10],
    "promo": [0, 1, 1],
})
predictions = model.predict(new_data)  # DataFrame 반환
```

## 수식 문법

```python
regress(data=df, formula="y ~ x1 + x2")       # 특정 변수
regress(data=df, formula="y ~ .")              # 전체 변수
regress(data=df, formula="y ~ x1 * x2")       # 교호작용
regress(data=df, formula="y ~ x + I(x**2)")   # 다항식
```

## 배열 직접 입력

```python
model = regress(y=y_array, X=X_array)
```

---

**인터랙티브 튜토리얼:** `marimo run docs/tutorials/ko/03_regression.py`
