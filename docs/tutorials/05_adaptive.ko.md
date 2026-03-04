# 튜토리얼 05 — 적응형 인텔리전스

**변화하는 조건에 적응하는 예측.**

Vectrix는 적응형 인텔리전스 레이어를 포함합니다 — 레짐 감지, Forecast DNA 프로파일링, 자가치유 예측, 비즈니스 제약 조건 적용.

## 1. 레짐 감지

현실 세계의 시계열은 시간이 지나면서 행동이 바뀝니다. `RegimeDetector`는 통계적 속성이 다른 구간(레짐)을 식별합니다.

```python
import numpy as np
from vectrix import RegimeDetector

np.random.seed(42)
regime1 = np.random.normal(100, 5, 80)
regime2 = np.random.normal(150, 15, 60)
regime3 = np.random.normal(120, 8, 60)
y = np.concatenate([regime1, regime2, regime3])

detector = RegimeDetector()
result = detector.detect(y)

print(f"발견된 레짐: {len(result.regimeStats)}개")
for i, stat in enumerate(result.regimeStats):
    print(f"  레짐 {i}: 평균={stat.mean:.1f}, 표준편차={stat.std:.1f}, "
          f"시작={stat.startIdx}, 끝={stat.endIdx}")
```

```
발견된 레짐: 3개
  레짐 0: 평균=100.2, 표준편차=4.8, 시작=0, 끝=79
  레짐 1: 평균=149.5, 표준편차=14.2, 시작=80, 끝=139
  레짐 2: 평균=120.3, 표준편차=7.9, 시작=140, 끝=199
```

### 레짐 인식 예측

```python
from vectrix import RegimeAwareForecaster

forecaster = RegimeAwareForecaster()
predictions = forecaster.forecast(y, steps=20)
print(f"예측 형상: {predictions.shape}")
print(f"현재 레짐(레짐 2)의 통계를 사용하여 예측합니다")
```

## 2. Forecast DNA

DNA 프로파일링은 각 시계열에 고유한 지문을 부여합니다 — 시계열 비교와 전략 선택에 유용합니다.

```python
from vectrix import ForecastDNA

dna = ForecastDNA()

profile = dna.analyze(y, period=1)
print(f"카테고리:   {profile.category}")
print(f"난이도:     {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"지문:       {profile.fingerprint}")
print(f"추천 모델:  {profile.recommendedModels[:3]}")
```

### DNA 특성

DNA 프로파일은 데이터를 특징짓는 통계적 특성을 추출합니다

```python
for key, value in list(profile.features.items())[:5]:
    print(f"  {key}: {value:.4f}")
```

주요 특성: `volatilityClustering`, `seasonalPeakPeriod`, `nonlinearAutocorr`, `demandDensity`, `hurstExponent`.

## 3. 자가치유 예측

실제 데이터가 예측과 다를 때, 자가치유 시스템이 자동으로 교정합니다

```python
from vectrix import SelfHealingForecast

healer = SelfHealingForecast()

original_forecast = np.array([100, 105, 110, 115, 120])
actuals_so_far = np.array([100, 112, 125])

report = healer.heal(
    originalForecast=original_forecast,
    actuals=actuals_so_far,
    historicalData=y
)

print(f"건강 점수: {report.healthScore:.0f}/100")
print(f"교정 횟수: {report.totalCorrected}")
print(f"교정된 예측: {report.correctedForecast}")
```

```
건강 점수: 72/100
교정 횟수: 2
교정된 예측: [100.  112.  125.  128.5 133.2]
```

자가치유 과정

1. 실측값과 원래 예측을 비교
2. 체계적 편향 감지 (과소/과대 예측)
3. 오차 패턴을 기반으로 남은 예측을 조정
4. 전체 예측 건강 상태를 보고

## 4. 비즈니스 제약 조건

예측에 실무 비즈니스 규칙을 적용하세요

```python
from vectrix import ConstraintAwareForecaster, Constraint

constraints = [
    Constraint(name="non-negative", minValue=0),
    Constraint(name="capacity", maxValue=500),
    Constraint(name="growth-limit", maxChangeRate=0.10),
]

forecaster = ConstraintAwareForecaster(constraints=constraints)
constrained = forecaster.apply(
    predictions=np.array([480, 520, -10, 450, 490]),
    historicalData=y
)

print(f"원본:       [480, 520, -10, 450, 490]")
print(f"제약 적용: {constrained}")
```

```
원본:       [480, 520, -10, 450, 490]
제약 적용: [480. 500.   0. 450. 490.]
```

### 자주 사용하는 제약 조건

| 제약 조건 | 사용 사례 |
|-----------|----------|
| `minValue=0` | 매출, 수요 (음수 불가) |
| `maxValue=N` | 생산 능력 한계, 재고 상한 |
| `maxChangeRate=0.1` | 기간 대비 최대 10% 변화 |

## 5. 통합 활용

완전한 적응형 워크플로우

```python
import numpy as np
from vectrix import forecast, analyze, ForecastDNA, RegimeDetector

data = [120, 135, 148, 132, 155, 167, 143, 178, 165, 190,
        172, 195, 185, 210, 198, 225, 215, 240, 230, 255,
        245, 268, 258, 280, 270, 295, 285, 310, 300, 325]

report = analyze(data)
print(f"1. DNA: {report.dna.difficulty} 난이도, {report.dna.category}")
print(f"   변화점: {len(report.changepoints)}개")

result = forecast(data, steps=10)
print(f"2. 최적 모델: {result.model} (MAPE={result.mape:.1f}%)")
print(f"   전체 모델: {result.models}")

comparison = result.compare()
print(f"3. 모델 비교:")
print(comparison.to_string(index=False))
```

## 6. API 레퍼런스

| 클래스 | 용도 |
|--------|------|
| `RegimeDetector` | 시계열의 통계적 레짐 감지 |
| `RegimeAwareForecaster` | 현재 레짐 맥락을 활용한 예측 |
| `ForecastDNA` | 시계열 특성 프로파일링 |
| `SelfHealingForecast` | 실측 데이터로 예측 자동 교정 |
| `ConstraintAwareForecaster` | 비즈니스 제약 조건 적용 |
| `Constraint` | 최소/최대/변화율 제약 정의 |

---

**다음:** [튜토리얼 06 — 비즈니스 인텔리전스](06_business.ko.md) — 이상치 감지, 시나리오 분석, 백테스트
