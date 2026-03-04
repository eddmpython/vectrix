# 튜토리얼 01 — 빠른 시작

**3분 만에 첫 예측을 만들어보세요.**

Vectrix는 설정이 필요 없는 시계열 예측 라이브러리입니다.
데이터를 넣으면 — 리스트, NumPy 배열, pandas DataFrame, CSV 파일 — 30개 이상의 모델 후보 중 최적의 모델을 자동으로 선택합니다.

## 설치

```bash
pip install vectrix
```

## 1. 리스트로 바로 예측

가장 간단한 방법입니다. 날짜도, 컬럼 이름도, 모델 선택도 필요 없습니다.

```python
from vectrix import forecast

sales = [
    120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
    172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
    245, 268, 258, 280, 270, 295, 285, 310, 300, 325,
]

result = forecast(sales, steps=10)
```

이게 전부입니다. Vectrix가 자동으로

- 날짜를 생성하고 (오늘 기준으로 역산)
- 여러 모델을 시도하고 (ETS, ARIMA, Theta, CES, DOT, …)
- 검증 MAPE 기준으로 최적 모델을 선택하고
- 95% 신뢰구간과 함께 예측값을 반환합니다

### 결과 확인

```python
print(result.model)         # 예: 'Dynamic Optimized Theta'
print(result.predictions)   # 10개의 예측값 배열
print(result.mape)          # 검증 MAPE (%)
```

출력 예시

```
Dynamic Optimized Theta
[331.2  337.8  344.5  ...]
5.14
```

## 2. DataFrame에서 예측

pandas DataFrame에 날짜 컬럼이 있으면, Vectrix가 날짜와 값 컬럼을 자동으로 감지합니다.

```python
import numpy as np
import pandas as pd
from vectrix import forecast

np.random.seed(42)
n = 120
t = np.arange(n, dtype=np.float64)
trend = 100 + 0.5 * t
seasonal = 20 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 5, n)

df = pd.DataFrame({
    "date": pd.date_range("2015-01-01", periods=n, freq="MS"),
    "sales": trend + seasonal + noise,
})

result = forecast(df, steps=12)
```

컬럼 이름이 명확하지 않으면 직접 지정할 수 있습니다

```python
result = forecast(df, date="date", value="sales", steps=12)
```

### 요약 보기

```python
print(result.summary())
```

```
==================================================
        Vectrix Forecast Summary
==================================================
  Model: 4Theta Ensemble
  Horizon: 12 steps
  Start: 2025-01-01
  End: 2025-12-01
  Mean: 163.42
  Min: 139.68
  Max: 185.88

  [Model Comparison]
    4Theta Ensemble: MAPE=3.21%
    Dynamic Optimized Theta: MAPE=3.58%
    AutoCES (Native): MAPE=4.12%
    AutoETS (Native): MAPE=5.67%
    AutoARIMA (Native): MAPE=8.94%
==================================================
```

## 3. 결과 내보내기

### DataFrame으로 변환

```python
pred_df = result.to_dataframe()
print(pred_df.head())
```

```
         date  prediction     lower95     upper95
0  2025-01-01      159.69      140.12      179.26
1  2025-02-01      169.33      145.87      192.79
2  2025-03-01      176.37      149.23      203.52
3  2025-04-01      185.88      155.31      216.45
4  2025-05-01      181.74      147.82      215.66
```

### CSV / JSON으로 저장

```python
result.to_csv("forecast.csv")

json_str = result.to_json()
result.to_json("forecast.json")
```

## 4. 모든 모델 비교

모든 모델의 성능을 한눈에 확인하세요

```python
print(result.compare())
```

```
                     model   mape   rmse    mae  smape  time_ms  selected
0          4Theta Ensemble   3.21  12.45   9.87   3.15      2.1      True
1  Dynamic Optimized Theta   3.58  14.23  11.02   3.49      5.5     False
2         AutoCES (Native)   4.12  16.78  13.45   4.03      9.3     False
3         AutoETS (Native)   5.67  21.34  17.89   5.52     28.6     False
4       AutoARIMA (Native)   8.94  32.56  26.78   8.67     15.2     False
```

### 모든 모델의 예측값 가져오기

```python
all_df = result.all_forecasts()
print(all_df.head())
```

```
         date  4Theta Ensemble  Dynamic Optimized Theta  AutoCES  AutoETS  AutoARIMA
0  2025-01-01           159.69                   157.74   154.04   156.83     153.26
1  2025-02-01           169.33                   166.34   162.24   163.97     153.26
2  2025-03-01           176.37                   172.87   168.37   168.81     153.26
...
```

## 5. 예측 기간 변경

예측 기간(steps)을 바꾸면 모델 선택과 결과가 달라질 수 있습니다

```python
for steps in [7, 14, 30]:
    r = forecast(df, steps=steps)
    print(f"steps={steps:>2}  모델={r.model:<30}  평균={r.predictions.mean():.1f}")
```

```
steps= 7  모델=Dynamic Optimized Theta    평균=168.3
steps=14  모델=4Theta Ensemble             평균=165.7
steps=30  모델=4Theta Ensemble             평균=158.2
```

## 6. 결과 객체 레퍼런스

| 속성 / 메서드 | 타입 | 설명 |
|---|---|---|
| `.predictions` | `np.ndarray` | 예측값 배열 |
| `.dates` | `list` | 예측 날짜 문자열 |
| `.lower` | `np.ndarray` | 95% 하한 |
| `.upper` | `np.ndarray` | 95% 상한 |
| `.model` | `str` | 선택된 모델 이름 |
| `.mape` | `float` | 검증 MAPE (%) |
| `.rmse` | `float` | 검증 RMSE |
| `.models` | `list` | 평가된 모든 모델 이름 (순위순) |
| `.to_dataframe()` | `DataFrame` | date, prediction, lower95, upper95 |
| `.compare()` | `DataFrame` | 모든 모델 MAPE 순위 비교 |
| `.all_forecasts()` | `DataFrame` | 모든 모델의 예측값 |
| `.summary()` | `str` | 텍스트 요약 |
| `.to_csv(path)` | `self` | CSV 저장 |
| `.to_json(path)` | `str` | JSON 저장 |
| `.describe()` | `DataFrame` | Pandas 스타일 통계 |

## 7. 지원하는 입력 형식

```python
forecast([1, 2, 3, 4, 5])                    # 리스트
forecast(np.array([1, 2, 3, 4, 5]))          # NumPy 배열
forecast(pd.Series([1, 2, 3, 4, 5]))         # pandas Series
forecast({"value": [1, 2, 3, 4, 5]})         # dict
forecast(df, date="날짜", value="매출")        # DataFrame
forecast("data.csv")                          # CSV 파일 경로
```

---

**다음:** [튜토리얼 02 — 분석 & DNA](02_analyze.ko.md) — 자동 시계열 프로파일링
