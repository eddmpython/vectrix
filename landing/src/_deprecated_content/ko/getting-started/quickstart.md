---
title: 빠른 시작
---

# 빠른 시작

3줄의 Python 코드로 첫 예측을 만드세요. 설정 없이, 모델 선택 없이, 파라미터 튜닝 없이 — Vectrix가 모든 것을 자동으로 처리합니다.

## 리스트로 예측

가장 간단한 사용법. 숫자 시퀀스와 예측 스텝 수를 전달하세요:

```python
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140, 160, 150, 170], steps=5)
print(result.model)          # 선택된 모델명
print(result.predictions)    # 예측값
print(result.summary())      # 텍스트 요약
```

뒤에서 Vectrix는 30개 이상의 모델 후보를 평가하고, 검증 세트에서 각각을 검증한 후, 95% 신뢰구간과 함께 최적 모델을 반환합니다.

## DataFrame에서 예측

실제 데이터에는 타임스탬프가 있습니다. pandas DataFrame을 전달하면 Vectrix가 날짜와 값 컬럼을 자동 감지합니다:

```python
import pandas as pd
from vectrix import forecast

df = pd.read_csv("sales.csv")
result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

## CSV 파일에서 예측

pandas 단계를 생략하세요. 파일 경로를 전달하면 Vectrix가 읽기, 컬럼 감지, 예측을 한 번에 수행합니다:

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
```

## 결과 활용

모든 예측은 `EasyForecastResult` 객체를 반환합니다. 예측값, 신뢰구간, 정확도 지표, 내보내기 메서드를 포함합니다:

| 속성 / 메서드 | 설명 |
|--------------|------|
| `.predictions` | 예측값 (numpy 배열) |
| `.dates` | 예측 날짜 |
| `.lower` | 95% 하한 |
| `.upper` | 95% 상한 |
| `.model` | 선택된 모델명 |
| `.mape` | 검증 MAPE (%) |
| `.rmse` | 검증 RMSE |
| `.summary()` | 포맷된 텍스트 보고서 |
| `.compare()` | 정확도 기준 전체 모델 순위 |
| `.all_forecasts()` | 모든 모델의 예측값 나란히 비교 |
| `.to_dataframe()` | DataFrame 변환 |
| `.to_csv(path)` | CSV 내보내기 |
| `.to_json()` | JSON 문자열 내보내기 |
| `.plot()` | Matplotlib 시각화 |

## 지원 입력 형식

Vectrix는 5가지 입력 형식을 지원합니다 — 수동 변환이 필요 없습니다:

```python
forecast([1, 2, 3, 4, 5])                    # 리스트
forecast(np.array([1, 2, 3, 4, 5]))          # numpy 배열
forecast(pd.Series([1, 2, 3, 4, 5]))         # pandas Series
forecast(df, date="date", value="sales")      # DataFrame
forecast("data.csv")                           # CSV 파일 경로
```

## 빠른 분석

예측 전에 데이터를 프로파일링하세요 — 난이도, 계절성, 추천 모델을 파악할 수 있습니다:

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
print(f"난이도: {report.dna.difficulty}")
print(f"카테고리: {report.dna.category}")
print(report.summary())
```

## 빠른 회귀분석

R 스타일 수식으로 회귀분석을 수행합니다. 자동 진단이 포함되어 있습니다:

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ ads + price")
print(model.summary())
print(model.diagnose())
```

## 다음 단계

- **[튜토리얼 01 — 빠른 시작](/docs/tutorials/quickstart)** — 예상 출력과 함께하는 상세 가이드
- **[튜토리얼 02 — 분석 & DNA](/docs/tutorials/analyze)** — 데이터의 DNA 지문 이해하기
- **[튜토리얼 04 — 30+ 모델](/docs/tutorials/models)** — Vectrix가 제공하는 모든 모델 심층 탐구

---

**인터랙티브:** `marimo run docs/tutorials/ko/01_quickstart.py`로 대화형 버전을 실행하세요.
