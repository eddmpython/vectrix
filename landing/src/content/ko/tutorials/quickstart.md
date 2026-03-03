---
title: "튜토리얼 01 — 빠른 시작"
---

# 튜토리얼 01 — 빠른 시작

**1분 안에 첫 예측을 만드세요.** 설정 불필요, 보일러플레이트 불필요 — 데이터를 전달하면 신뢰구간이 포함된 예측값이 반환됩니다.

Vectrix는 제로 설정 시계열 예측 라이브러리입니다. 30개 이상의 통계 모델(ETS, ARIMA, Theta, CES, DOT 등)을 평가하고, 검증 세트에서 최적 모델을 선택합니다 — 단 한 번의 함수 호출로.

## 설치

```bash
pip install vectrix
```

## 리스트에서 예측

가장 간단한 예측 방법입니다. Python 리스트와 예측 스텝 수만 전달하면, Vectrix가 나머지를 모두 처리합니다.

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

**내부에서 일어나는 일:** `forecast()` 한 줄이 호출되면 Vectrix는 내부적으로 다음 5단계를 수행합니다.

1. **데이터 검증** — 입력을 파싱하고, 결측치와 이상값을 자동 처리합니다.
2. **DNA 분석** — 65개 이상의 통계적 특성(추세, 계절성, 변동성, Hurst 지수 등)을 추출하여 데이터의 '지문'을 생성합니다.
3. **모델 평가** — 30개 이상의 예측 모델(ETS, ARIMA, Theta, DOT, CES, MSTL 등)을 시간 순서 교차 검증으로 평가합니다.
4. **최적 모델 선택** — 검증 오차(MAPE, sMAPE, RMSE)가 가장 낮은 모델을 자동으로 선택합니다.
5. **예측 및 신뢰구간 생성** — 선택된 모델로 미래 값을 예측하고, 95% 신뢰구간을 함께 반환합니다.

## DataFrame에서 예측

날짜 컬럼이 포함된 DataFrame을 전달하면, Vectrix가 날짜 형식과 빈도(일별, 주별, 월별 등)를 자동으로 감지합니다. `date`와 `value` 파라미터로 컬럼명을 지정할 수 있으며, 생략 시 Vectrix가 자동으로 적절한 컬럼을 찾습니다.

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

파일 경로 문자열을 직접 전달하면 Vectrix가 CSV를 읽고, 날짜 컬럼과 값 컬럼을 자동 감지합니다. DataFrame을 미리 만들 필요가 없어 탐색적 분석에 편리합니다.

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result.model)
print(result.predictions)
```

## 모델 비교

어떤 모델이 평가되었고, 각각의 정확도는 어떤지 한눈에 확인할 수 있습니다. `compare()`는 MAPE, RMSE, MAE, sMAPE 기준으로 정렬된 전체 모델 순위를 pandas DataFrame으로 반환합니다.

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

`compare()` 함수를 독립적으로 사용하면, 예측 결과 객체 없이도 데이터와 스텝 수만으로 모델 순위를 빠르게 확인할 수 있습니다.

```python
from vectrix import compare

ranking = compare(data, steps=5)
print(ranking)
```

## 모든 모델의 예측값 가져오기

최적 모델뿐 아니라, 평가된 모든 모델의 개별 예측값을 확인할 수 있습니다. 모델 간 예측 경향을 비교하거나, 사용자 정의 앙상블을 구성할 때 유용합니다.

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

예측 결과를 DataFrame, CSV, JSON 등 다양한 형식으로 내보내 다른 시스템이나 대시보드와 연동할 수 있습니다.

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

Vectrix는 5가지 입력 형식을 지원합니다. 데이터 변환 없이 가지고 있는 형식 그대로 전달하면 됩니다.

```python
import numpy as np
import pandas as pd
from vectrix import forecast

forecast([1, 2, 3, 4, 5], steps=3)                    # Python 리스트
forecast(np.array([1, 2, 3, 4, 5]), steps=3)           # NumPy 배열
forecast(pd.Series([1, 2, 3, 4, 5]), steps=3)          # Pandas Series
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

`forecast()` 함수가 반환하는 `EasyForecastResult` 객체는 예측값, 신뢰구간, 정확도 지표, 모델 비교 등 모든 정보에 접근할 수 있는 통합 인터페이스입니다.

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

월별 매출 데이터에서 6개월 미래를 예측하고, 모델 성능을 확인하는 전체 워크플로우입니다.

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
