# 튜토리얼 04 — 30+ 모델

**모든 모델에 직접 접근하고 Vectrix의 내부 동작을 이해하세요.**

`forecast()` 함수가 모든 것을 자동으로 처리합니다. 하지만 때로는 모든 모델을 보고, 과정을 제어하고, 특정 모델이 왜 선택되었는지 이해하고 싶을 수 있습니다.

## 1. 한 줄 모델 비교

모든 모델을 비교하는 가장 쉬운 방법:

```python
from vectrix import compare

df = compare([
    120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
    172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
], steps=5)

print(df)
```

```
                     model   mape   rmse    mae  smape  time_ms  selected
0  Dynamic Optimized Theta   6.14  19.69  14.85    inf      5.5      True
1          4Theta Ensemble   7.11  24.39  17.59    inf      2.0     False
2         AutoCES (Native)   9.00  27.88  22.09    inf     14.1     False
3         AutoETS (Native)  14.74  39.39  35.52    inf     28.6     False
```

## 2. Vectrix 클래스

완전한 제어를 위해 `Vectrix` 클래스를 직접 사용하세요:

```python
import numpy as np
import pandas as pd
from vectrix import Vectrix

np.random.seed(42)
n = 150
t = np.arange(n, dtype=np.float64)
values = 500 + 2 * t + 30 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 10, n)

df = pd.DataFrame({
    "date": pd.date_range("2012-01-01", periods=n, freq="MS"),
    "revenue": values,
})

fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="revenue", steps=12)
```

### 결과 구조

```python
print(f"성공: {result.success}")
print(f"최적 모델: {result.bestModelName}")
print(f"테스트된 모델 수: {len(result.allModelResults)}")
print(f"예측값: {result.predictions[:3]}...")
```

```
성공: True
최적 모델: 4Theta Ensemble
테스트된 모델 수: 8
예측값: [812.3  825.7  831.2]...
```

### 모든 모델 결과

```python
for modelId, m in result.allModelResults.items():
    if m.isValid:
        flat = " (FLAT)" if m.flatInfo and m.flatInfo.isFlat else ""
        print(f"  {m.modelName:<30} MAPE={m.mape:6.2f}%  시간={m.trainingTime*1000:.0f}ms{flat}")
```

```
  4Theta Ensemble                MAPE=  2.73%  시간=3ms
  Dynamic Optimized Theta        MAPE=  3.15%  시간=6ms
  AutoCES (Native)               MAPE=  4.21%  시간=12ms
  AutoETS (Native)               MAPE=  5.89%  시간=35ms
  AutoARIMA (Native)             MAPE=  7.45%  시간=18ms
  AutoMSTL                       MAPE=  8.12%  시간=42ms
  DTSF                           MAPE= 11.34%  시간=8ms
  ESN                            MAPE= 14.56%  시간=15ms
```

## 3. 사용 가능한 모델

Vectrix에 포함된 모델 패밀리:

| 카테고리 | 모델 | 강점 |
|----------|------|------|
| **지수평활** | AutoETS, ETS-AAN, ETS-AAA | 추세 + 계절성, 범용 |
| **ARIMA** | AutoARIMA | Box-Jenkins 방법론, 유연함 |
| **Theta** | Optimized Theta, 4Theta | M3 챔피언, 단순하지만 강력 |
| **DOT** | Dynamic Optimized Theta | M4급 정확도, 자동 적응 |
| **CES** | AutoCES | 복소 지수평활 |
| **분해** | MSTL, AutoMSTL | 다중 계절 분해 |
| **GARCH** | GARCH, EGARCH, GJR-GARCH | 변동성 모델링 |
| **Croston** | AutoCroston | 간헐적 수요 |
| **TBATS** | AutoTBATS | 다중 계절성 |
| **패턴 매칭** | DTSF | 비모수적, 시간별 데이터에 강함 |
| **신경망** | ESN (Echo State) | 리저보어 컴퓨팅, 앙상블 다양성 |
| **기준선** | Naive, Seasonal Naive, Mean, RWD | 참조 벤치마크 |

## 4. 평탄 예측 방어

Vectrix의 독자 기능 중 하나: 평탄(상수) 예측의 자동 감지 및 교정.

일부 모델은 패턴을 포착하지 못할 때 일직선 예측을 생성합니다. Vectrix는 이를 감지하고 교정하거나 경고합니다.

```python
if result.flatInfo and result.flatInfo.isFlat:
    print(f"평탄 예측 감지!")
    print(f"교정: {result.flatInfo.message}")
```

### 작동 방식

1. **감지** — 예측 분산이 과거 분산 대비 거의 0인지 확인
2. **위험 평가** — 심각도 판단 (low / medium / high / critical)
3. **교정** — 과거 패턴을 이용한 분산 주입
4. **대체** — 교정 실패 시 다른 모델로 전환

## 5. 데이터 특성

`Vectrix` 클래스는 예측 전에 데이터를 분석합니다:

```python
c = result.characteristics
print(f"주기: {c.period}")
print(f"빈도: {c.frequency}")
print(f"추세: {c.hasTrend} ({c.trendDirection})")
print(f"계절성: {c.hasSeasonality}")
```

## 6. 앙상블 전략

여러 모델이 좋은 성능을 보이면, Vectrix는 분산 보존 앙상블을 생성합니다:

```python
if result.bestModelId == "ensemble":
    print("앙상블이 선택되었습니다!")
    print(f"앙상블 모델: {result.bestModelName}")
```

앙상블 로직:

1. 상위 3개 모델을 전체 데이터로 재학습
2. MAPE 역수 가중 조합
3. 원본 데이터의 분산을 단일 최적 모델보다 잘 보존할 때만 앙상블 선택

## 7. 상세 모드

과정의 모든 단계를 확인하세요:

```python
fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="revenue", steps=12)
```

모델 학습 진행, 시간, 검증 점수, 선택 근거가 출력됩니다.

## 8. 결과 객체 레퍼런스 (ForecastResult)

| 속성 | 타입 | 설명 |
|---|---|---|
| `.success` | `bool` | 예측 성공 여부 |
| `.predictions` | `np.ndarray` | 최종 예측값 |
| `.dates` | `list` | 예측 날짜 문자열 |
| `.lower95` | `np.ndarray` | 95% 하한 |
| `.upper95` | `np.ndarray` | 95% 상한 |
| `.bestModelId` | `str` | 선택된 모델 ID |
| `.bestModelName` | `str` | 선택된 모델 표시명 |
| `.allModelResults` | `dict` | 모든 ModelResult 객체 (ID 키) |
| `.characteristics` | `DataCharacteristics` | 감지된 데이터 속성 |
| `.flatRisk` | `FlatRiskAssessment` | 평탄 예측 위험 정보 |
| `.flatInfo` | `FlatPredictionInfo` | 평탄 감지/교정 상세 |
| `.warnings` | `list` | 생성된 경고 목록 |

---

**다음:** [튜토리얼 05 — 적응형 인텔리전스](05_adaptive.ko.md) — 레짐 감지, DNA, 자가치유, 제약 조건
