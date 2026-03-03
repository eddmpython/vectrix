---
title: "튜토리얼 01 — 빠른 시작"
---

# 튜토리얼 01 — 빠른 시작

Vectrix로 한 줄 예측하기. 설정도, 보일러플레이트도 필요 없습니다. 데이터만 전달하면 결과가 나옵니다.

## 설치

```bash
pip install vectrix
```

## 리스트에서 예측

가장 간단한 예측 방법 — Python 리스트와 예측 스텝 수만 전달하면 됩니다

```python
from vectrix import forecast

data = [120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215]
result = forecast(data, steps=5)

print(result.model)        # 자동 선택된 최적 모델
print(result.predictions)  # 예측값 (numpy 배열)
print(result.mape)         # 검증 MAPE (%)
```

**예상 출력:**

```
AutoETS
[221.3  228.7  235.1  241.6  248.0]
4.23
```

Vectrix는 내부적으로 30개 이상의 모델을 평가하고, 검증 오차가 가장 낮은 모델을 자동으로 선택합니다.

## DataFrame에서 예측

날짜가 포함된 실제 데이터는 DataFrame으로 전달하고 컬럼명을 지정합니다

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=30)

print(result.summary())
```

**예상 출력:**

```
=== Vectrix Forecast Summary ===
Model: AutoETS
Forecast Steps: 30
MAPE: 3.85%
RMSE: 42.17
MAE: 35.62

Predictions:
  2026-03-03: 1,245.3 [1,102.1 - 1,388.5]
  2026-03-04: 1,251.8 [1,095.4 - 1,408.2]
  ...
```

## CSV 파일에서 예측

파일 경로를 직접 전달하면 Vectrix가 날짜와 값 컬럼을 자동 감지합니다

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result.model)
print(result.predictions)
```

## 모델 비교

평가된 모든 모델의 성능을 확인할 수 있습니다

```python
result = forecast(data, steps=5)

comparison = result.compare()
print(comparison)
```

**예상 출력:**

```
          model   mape   rmse    mae  smape
0       AutoETS   4.23  12.41   9.87   4.15
1    AutoARIMA   4.87  14.23  11.42   4.72
2        Theta   5.12  15.67  12.31   5.01
3          DOT   5.34  16.12  12.89   5.22
4         MSTL   5.89  17.45  13.67   5.74
...
```

`compare()` 함수를 사용하면 더 빠르게 모델 순위를 확인할 수 있습니다

```python
from vectrix import compare

ranking = compare(data, steps=5)
print(ranking)
```

## 모든 모델의 예측값 가져오기

최적 모델뿐 아니라 모든 모델의 예측값을 가져올 수 있습니다

```python
allPreds = result.all_forecasts()
print(allPreds)
```

**예상 출력:**

```
   step  AutoETS  AutoARIMA  Theta    DOT   MSTL  ...
0     1   221.3     219.8   222.1  220.5  218.9  ...
1     2   228.7     226.4   229.3  227.8  225.1  ...
2     3   235.1     233.2   236.0  234.6  231.8  ...
...
```

## 결과 내보내기

다양한 형식으로 결과를 내보낼 수 있습니다

```python
dfResult = result.to_dataframe()
print(dfResult)
```

**예상 출력:**

```
         date  prediction   lower95   upper95
0  2026-03-03      221.3     198.2     244.4
1  2026-03-04      228.7     202.1     255.3
2  2026-03-05      235.1     205.8     264.4
...
```

```python
result.to_csv("forecast_output.csv")

jsonStr = result.to_json()
print(jsonStr[:100])
```

## 지원 입력 형식

Vectrix는 6가지 입력 형식을 지원합니다. 변환이 필요 없습니다

```python
import numpy as np
import pandas as pd
from vectrix import forecast

forecast([1, 2, 3, 4, 5], steps=3)                    # Python 리스트
forecast(np.array([1, 2, 3, 4, 5]), steps=3)           # NumPy 배열
forecast(pd.Series([1, 2, 3, 4, 5]), steps=3)          # Pandas Series
forecast({"value": [1, 2, 3, 4, 5]}, steps=3)          # 딕셔너리
forecast(df, date="date", value="sales", steps=3)      # DataFrame
forecast("data.csv", steps=3)                           # CSV 파일 경로
```

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `data` | (필수) | 지원되는 형식의 입력 데이터 |
| `steps` | `10` | 예측 스텝 수 |
| `date` | 자동 | 날짜 컬럼명 (DataFrame/CSV만 해당) |
| `value` | 자동 | 값 컬럼명 (DataFrame/CSV만 해당) |
| `frequency` | `'auto'` | 데이터 빈도 (생략 시 자동 감지) |

## 결과 객체 참조

`EasyForecastResult`가 제공하는 속성과 메서드

| 속성 / 메서드 | 타입 | 설명 |
|--------------|------|------|
| `.predictions` | `np.ndarray` | 예측값 |
| `.dates` | `list` | 예측 날짜 문자열 |
| `.lower` | `np.ndarray` | 95% 하한 신뢰 구간 |
| `.upper` | `np.ndarray` | 95% 상한 신뢰 구간 |
| `.model` | `str` | 선택된 모델명 |
| `.mape` | `float` | 검증 MAPE (%) |
| `.rmse` | `float` | 검증 RMSE |
| `.mae` | `float` | 검증 MAE |
| `.smape` | `float` | 검증 sMAPE |
| `.compare()` | `DataFrame` | MAPE 기준 전체 모델 순위 |
| `.all_forecasts()` | `dict` | 모든 모델의 예측값 |
| `.summary()` | `str` | 서식화된 텍스트 요약 |
| `.to_dataframe()` | `DataFrame` | date, prediction, lower95, upper95 |
| `.to_csv(path)` | `self` | CSV 파일 내보내기 |
| `.to_json()` | `str` | JSON 문자열 내보내기 |

## 완전한 예제

```python
from vectrix import forecast

monthlySales = [
    450, 470, 520, 540, 580, 620, 590, 610, 650, 680, 710, 750,
    460, 490, 530, 560, 600, 640, 610, 630, 670, 700, 730, 770,
]

result = forecast(monthlySales, steps=6, frequency='M')

print(f"모델: {result.model}")
print(f"MAPE: {result.mape:.2f}%")
print(f"향후 6개월: {result.predictions}")
print(f"하한: {result.lower}")
print(f"상한: {result.upper}")

result.to_csv("forecast.csv")
print()
print("평가된 전체 모델:")
print(result.compare().head(10))
```

---
