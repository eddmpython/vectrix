# 파이프라인

`ForecastPipeline`은 전처리 변환기와 예측 모델을 체이닝합니다. 변환은 예측 생성 시 자동으로 역변환되므로, 예측값이 원래 데이터 스케일로 반환됩니다.

## 기본 파이프라인

```python
from vectrix.pipeline import ForecastPipeline, Scaler, OutlierClipper

pipe = ForecastPipeline([
    ('clip', OutlierClipper()),
    ('scale', Scaler()),
    ('forecast', MyForecaster()),
])

pipe.fit(y)
predictions, lower, upper = pipe.predict(12)
```

파이프라인 동작

1. **fit** — 각 변환기를 순차적으로 `fitTransform` 후, 변환된 데이터로 예측기를 `fit`
2. **predict** — 예측기에서 예측값을 얻은 후, 역순으로 모든 변환기의 `inverseTransform` 적용

## 내장 변환기

| 변환기 | 기능 | 역변환? |
|:--|:--|:--:|
| `Scaler(method='zscore')` | Z-score 표준화 또는 MinMax 정규화 | :white_check_mark: |
| `LogTransformer()` | `log(1 + y)`, 음수 데이터 자동 시프트 | :white_check_mark: |
| `BoxCoxTransformer()` | MLE 기반 최적 Box-Cox 람다 추정 | :white_check_mark: |
| `Differencer(d=1)` | d차 차분 (정상성 확보) | :white_check_mark: |
| `Deseasonalizer(period=7)` | 기간 평균으로 계절 성분 제거 | :white_check_mark: |
| `Detrend()` | 선형 추세 제거 | :white_check_mark: |
| `OutlierClipper(factor=3.0)` | IQR 기반 이상치 클리핑 | :x: |
| `MissingValueImputer(method='linear')` | 선형 보간, 평균, 전방 채움으로 NaN 처리 | :x: |

## 다단계 전처리

복잡한 데이터 준비를 위해 여러 변환기를 체이닝

```python
from vectrix.pipeline import (
    ForecastPipeline, MissingValueImputer, OutlierClipper,
    LogTransformer, Deseasonalizer, Scaler
)

pipe = ForecastPipeline([
    ('impute', MissingValueImputer(method='linear')),
    ('clip', OutlierClipper(factor=3.0)),
    ('log', LogTransformer()),
    ('deseason', Deseasonalizer(period=7)),
    ('scale', Scaler(method='zscore')),
    ('forecast', MyForecaster()),
])
```

## 예측 없이 변환만 수행

파이프라인을 순수 전처리 도구로 사용

```python
pipe.fit(train_data)

transformed = pipe.transform(test_data)
original = pipe.inverseTransform(transformed)
```

## 파이프라인 검사

```python
pipe.listSteps()
# ['impute', 'clip', 'log', 'deseason', 'scale', 'forecast']

scaler = pipe.getStep('scale')
print(scaler._mean, scaler._std)

pipe.getParams()
# {'clip__factor': 3.0, 'scale__method': 'zscore', ...}
```

## Scaler 옵션

=== "Z-score (기본값)"

    ```python
    Scaler(method='zscore')
    ```
    평균=0, 표준편차=1로 중심화. 범용 표준화에 적합.

=== "MinMax"

    ```python
    Scaler(method='minmax')
    ```
    [0, 1] 범위로 스케일링. 제한된 출력이 필요할 때 적합.

## Box-Cox 변환

데이터 분포를 정규화하기 위한 최적 람다를 자동으로 탐색

```python
from vectrix.pipeline import BoxCoxTransformer

bc = BoxCoxTransformer()       # 자동 람다 추정
bc = BoxCoxTransformer(lmbda=0.5)  # 고정 람다 (제곱근)
```

## 결측치 처리 전략

```python
MissingValueImputer(method='linear')  # 선형 보간 (기본값)
MissingValueImputer(method='mean')    # 시리즈 평균으로 대체
MissingValueImputer(method='ffill')   # 전방 채움
```

---

**API 레퍼런스:** [파이프라인 API](../api/pipeline.md)
