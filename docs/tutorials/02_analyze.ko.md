# 튜토리얼 02 — 분석 & DNA

**예측 전에 데이터를 이해하세요.**

Vectrix의 `analyze()` 함수는 한 줄로 시계열을 프로파일링합니다 — 난이도, 카테고리, 계절성, 변화점, 이상치, 추천 모델을 자동으로 감지합니다.

## 1. 기본 분석

```python
import numpy as np
import pandas as pd
from vectrix import analyze

np.random.seed(42)
n = 200
t = np.arange(n, dtype=np.float64)
values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=n, freq="D"),
    "value": values,
})

report = analyze(df)
```

`forecast()`와 마찬가지로 컬럼 감지는 자동입니다. 리스트, 배열, CSV 경로도 전달할 수 있습니다.

## 2. DNA 프로파일

모든 시계열에는 고유한 "DNA"가 있습니다 — 통계적 특성을 요약하는 지문입니다.

```python
dna = report.dna

print(f"카테고리:   {dna.category}")
print(f"난이도:     {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"지문:       {dna.fingerprint}")
print(f"추천 모델:  {', '.join(dna.recommendedModels[:3])}")
```

```
카테고리:   trend-seasonal
난이도:     easy (23/100)
지문:       TS-E023-P007-V012
추천 모델:  auto_ces, dot, four_theta
```

DNA가 알려주는 것:

- **카테고리** — 시계열의 유형 (추세, 계절성, 정상, 간헐적 수요, …)
- **난이도** — 예측 난이도 (0=매우 쉬움, 100=극히 어려움)
- **지문** — 시계열 특성을 인코딩한 고유 코드
- **추천 모델** — 이 시계열에 가장 잘 맞을 가능성이 높은 모델

## 3. 변화점 감지

변화점은 시계열의 행동이 급격히 바뀌는 순간입니다 — 새로운 추세, 수준 점프, 변동성 변화.

```python
print(f"변화점 발견: {len(report.changepoints)}개")
print(f"위치: {report.changepoints.tolist()}")
```

```
변화점 발견: 2개
위치: [67, 134]
```

이 인덱스들은 데이터에서 구조적 변화가 발생한 행을 가리킵니다.

## 4. 이상치 감지

이상치는 예상 패턴에서 크게 벗어나는 개별 데이터 포인트입니다 (3-시그마 규칙).

```python
print(f"이상치 발견: {len(report.anomalies)}개")
if len(report.anomalies) > 0:
    print(f"위치: {report.anomalies.tolist()}")
```

```
이상치 발견: 1개
위치: [142]
```

## 5. 데이터 특성

시계열의 세부 속성:

```python
c = report.characteristics

print(f"길이:       {c.length}")
print(f"주기:       {c.period}")
print(f"빈도:       {c.frequency}")
print(f"추세:       {c.hasTrend} ({c.trendDirection}, 강도={c.trendStrength:.2f})")
print(f"계절성:     {c.hasSeasonality} (강도={c.seasonalStrength:.2f})")
print(f"예측 가능성: {c.predictabilityScore:.0f}/100")
```

```
길이:       200
주기:       7
빈도:       D
추세:       True (increasing, 강도=0.85)
계절성:     True (강도=0.92)
예측 가능성: 78/100
```

## 6. 전체 요약

하나의 포맷된 보고서로 모든 것을 확인하세요:

```python
print(report.summary())
```

```
=======================================================
        Vectrix Time Series Analysis Report
=======================================================

  [DNA Analysis]
    Trend-seasonal series with weekly cycle
    Category: trend-seasonal
    Forecast Difficulty: easy (23.0/100)
    Fingerprint: TS-E023-P007-V012
    Recommended Models: auto_ces, dot, four_theta

  [Changepoint Detection]
    Changepoints found: 2
    Locations: [67, 134]

  [Anomaly Detection]
    Anomalies: 1
    Locations: [142]

  [Data Characteristics]
    Length: 200
    Period: 7
    Frequency: D
    Trend: increasing (strength: 0.85)
    Seasonality: present (strength: 0.92)
    Predictability: 78.0/100
=======================================================
```

## 7. 실전 활용: 분석 후 예측

분석으로 데이터를 이해한 다음, 확신을 갖고 예측하세요:

```python
from vectrix import analyze, forecast

report = analyze(df)

if report.dna.difficulty == "hard":
    print("경고: 예측이 어려운 시계열입니다. 가능하면 더 많은 데이터를 사용하세요.")

result = forecast(df, steps=14)
print(f"DNA 추천 모델: {report.dna.recommendedModels[:3]}")
print(f"Vectrix 선택: {result.model}")
```

## 8. 여러 시계열 비교

여러 시계열을 프로파일링해서 차이를 파악하세요:

```python
from vectrix import analyze

series_list = {
    "안정적": [100 + np.random.normal(0, 2) for _ in range(100)],
    "추세형": [100 + 0.5 * i + np.random.normal(0, 3) for i in range(100)],
    "변동형": [100 + 10 * np.sin(i / 5) + np.random.normal(0, 15) for i in range(100)],
}

for name, data in series_list.items():
    r = analyze(data)
    print(f"{name:>4}: {r.dna.difficulty:>6} ({r.dna.difficultyScore:5.1f}/100)  카테고리={r.dna.category}")
```

```
안정적:   easy ( 12.3/100)  카테고리=stationary
추세형: medium ( 35.7/100)  카테고리=trend-stationary
변동형:   hard ( 72.1/100)  카테고리=volatile
```

## 9. 결과 객체 레퍼런스

| 속성 | 타입 | 설명 |
|---|---|---|
| `.dna` | `DNAProfile` | DNA 프로파일 (난이도, 카테고리, 지문, 추천 모델) |
| `.dna.difficulty` | `str` | 'easy', 'medium', 'hard' |
| `.dna.difficultyScore` | `float` | 0–100 점수 |
| `.dna.category` | `str` | 시계열 유형 분류 |
| `.dna.fingerprint` | `str` | 고유 지문 코드 |
| `.dna.recommendedModels` | `list` | 추천 모델 ID 목록 |
| `.changepoints` | `np.ndarray` | 변화점 인덱스 |
| `.anomalies` | `np.ndarray` | 이상치 인덱스 |
| `.features` | `dict` | 추출된 통계 특성 |
| `.characteristics` | `DataCharacteristics` | 길이, 주기, 빈도, 추세, 계절성 |
| `.summary()` | `str` | 포맷된 분석 보고서 |

---

**다음:** [튜토리얼 03 — 회귀분석](03_regression.ko.md) — R 스타일 공식 회귀분석과 진단
