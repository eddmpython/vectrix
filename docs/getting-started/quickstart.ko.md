# 빠른 시작

## 리스트로 예측

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170], steps=5)
print(result.model)          # 선택된 모델명
print(result.predictions)    # 예측값
print(result.summary())      # 텍스트 요약
```

## DataFrame에서 예측

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

## CSV 파일에서 예측

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
```

## 결과 활용

`EasyForecastResult` 제공 기능:

| 속성 / 메서드 | 설명 |
|--------------|------|
| `.predictions` | 예측값 (numpy 배열) |
| `.dates` | 예측 날짜 |
| `.lower` | 95% 하한 |
| `.upper` | 95% 상한 |
| `.model` | 선택된 모델명 |
| `.summary()` | 텍스트 요약 |
| `.to_dataframe()` | DataFrame 변환 |
| `.to_csv(path)` | CSV 내보내기 |
| `.to_json()` | JSON 문자열 내보내기 |
| `.plot()` | Matplotlib 시각화 |

## 지원 입력 형식

```python
forecast([1, 2, 3, 4, 5])                    # 리스트
forecast(np.array([1, 2, 3, 4, 5]))          # numpy 배열
forecast(pd.Series([1, 2, 3, 4, 5]))         # pandas Series
forecast({"value": [1, 2, 3, 4, 5]})         # dict
forecast(df, date="date", value="sales")      # DataFrame
forecast("data.csv")                           # CSV 파일 경로
```

## 빠른 분석

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
print(f"난이도: {report.dna.difficulty}")
print(f"카테고리: {report.dna.category}")
print(report.summary())
```

## 빠른 회귀분석

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ ads + price")
print(model.summary())
print(model.diagnose())
```

---

**다음:** [예측 가이드](../guide/forecasting.md)에서 심화 사용법을 확인하세요.

**인터랙티브:** `marimo run docs/tutorials/ko/01_quickstart.py`로 대화형 버전을 실행하세요.
