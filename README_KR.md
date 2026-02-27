<div align="center">

<br>

# Vectrix

**데이터를 넣으면 예측이 나옵니다. 설정 필요 없음.**

순수 Python 시계열 예측 -- 30+ 모델, 무거운 의존성 제로.

<br>

[![PyPI](https://img.shields.io/pypi/v/vectrix?style=flat-square&color=6366f1&label=PyPI)](https://pypi.org/project/vectrix/)
[![Python](https://img.shields.io/pypi/pyversions/vectrix?style=flat-square&label=Python)](https://pypi.org/project/vectrix/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-Buy%20Me%20a%20Coffee-orange?style=flat-square&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/eddmpython)

---

[설치](#설치) · [빠른 시작](#빠른-시작) · [주요 기능](#주요-기능) · [API](#api-레퍼런스) · [English](README.md)

</div>

<br>

```
  의존성 3개        모델 30+개     코드 1줄
  ─────────        ──────────    ──────────────
  numpy            AutoETS       from vectrix import forecast
  scipy            AutoARIMA     result = forecast(data, steps=12)
  pandas           Theta/DOT     print(result)
                   TBATS
                   GARCH
                   ...
```

<br>

## 빠른 시작

```bash
pip install vectrix
```

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
result.to_csv("output.csv")
```

한 줄이면 모델 선택, 일직선 예측 방지, 신뢰구간까지 포함된 예측이 완성됩니다.

<br>

## 왜 Vectrix?

| | Vectrix | statsforecast | Prophet | Darts |
|:--|:-:|:-:|:-:|:-:|
| **Zero-config 자동 예측** | **Yes** | Yes | -- | -- |
| **순수 Python (무거운 의존성 없음)** | **Yes** | -- | -- | -- |
| **30+ 모델 내장** | **Yes** | Yes | -- | Yes |
| **일직선 예측 방지** | **Yes** | -- | -- | -- |
| **적대적 스트레스 테스트** | **Yes** | -- | -- | -- |
| **Forecast DNA 지문** | **Yes** | -- | -- | -- |
| **비즈니스 제약 (8종)** | **Yes** | -- | -- | -- |
| **R 스타일 회귀분석** | **Yes** | -- | -- | -- |

> `numpy` + `scipy` + `pandas` -- 설치 끝.

<br>

## 주요 기능

<details open>
<summary><b>핵심 모델</b></summary>

| 모델 | 설명 |
|------|------|
| AutoETS | 30개 Error x Trend x Seasonal 조합, AICc 자동 선택 |
| AutoARIMA | 계절성 ARIMA, 단계적 차수 선택 |
| Theta / DOT | Original Theta + Dynamic Optimized Theta |
| AutoCES | Complex Exponential Smoothing (Svetunkov 2023) |
| AutoTBATS | 삼각함수 다중 계절성 분해 |
| GARCH | GARCH, EGARCH, GJR-GARCH 변동성 모델 |
| Croston | Classic, SBA, TSB 간헐적 수요 예측 |
| Logistic Growth | 용량 제한 포화 추세 |
| AutoMSTL | 다중 계절성 분해 + ARIMA 잔차 예측 |
| 베이스라인 | Naive, Seasonal Naive, Mean, Drift, Window Average |

</details>

<details>
<summary><b>세계 최초 방법론</b></summary>

| 방법 | 설명 |
|------|------|
| Lotka-Volterra Ensemble | 생태계 경쟁 역학 기반 모델 가중치 |
| Phase Transition | 임계 둔화 감지로 레짐 전환 예측 |
| Adversarial Stress | 5가지 섭동 연산자로 견고성 분석 |
| Hawkes Demand | 자기 흥분 점 과정으로 군집 수요 처리 |
| Entropic Confidence | Shannon 엔트로피 불확실성 정량화 |

</details>

<details>
<summary><b>적응형 지능</b></summary>

| 기능 | 설명 |
|------|------|
| 레짐 감지 | 순수 numpy HMM (Baum-Welch + Viterbi) |
| 자가 치유 | CUSUM + EWMA 드리프트 감지, 컨포멀 보정 |
| 제약 조건 | 8종: 비음수, 범위, 용량, YoY, 합계, 단조, 비율, 커스텀 |
| Forecast DNA | 65+ 특성 지문, 메타러닝 모델 추천 |
| 평탄 방어 | 4단계 방지 시스템 |

</details>

<details>
<summary><b>회귀분석 & 진단</b></summary>

| 기능 | 설명 |
|------|------|
| 방법 | OLS, Ridge, Lasso, Huber, Quantile |
| 수식 | R 스타일 `regress(data=df, formula="y ~ x1 + x2")` |
| 진단 | Durbin-Watson, Breusch-Pagan, VIF, 정규성 |
| 변수 선택 | Stepwise, 정규화 CV, 최적 부분집합 |
| 시계열 | Newey-West, Cochrane-Orcutt, Granger 인과성 |

</details>

<details>
<summary><b>비즈니스 인텔리전스</b></summary>

| 기능 | 설명 |
|------|------|
| 이상치 탐지 | 자동 이상값 식별 및 설명 |
| What-if 분석 | 시나리오 기반 예측 시뮬레이션 |
| 백테스팅 | Rolling origin 교차 검증 |
| 계층 조정 | Bottom-up, Top-down, MinTrace |
| 예측 구간 | Conformal + Bootstrap |

</details>

<br>

## 설치

```bash
pip install vectrix                # 핵심 (numpy + scipy + pandas)
pip install "vectrix[numba]"       # + Numba JIT (2-5배 가속)
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # 전체
```

**요구사항:** Python 3.10+

<br>

## 사용 예시

### Easy API

```python
from vectrix import forecast, analyze, regress

result = forecast([100, 120, 115, 130, 125, 140], steps=5)

report = analyze(df, date="date", value="sales")
print(f"난이도: {report.dna.difficulty}")

model = regress(data=df, formula="sales ~ temperature + promotion")
print(model.summary())
```

### DataFrame 워크플로우

```python
from vectrix import forecast, analyze
import pandas as pd

df = pd.read_csv("data.csv")

report = analyze(df, date="date", value="sales")
print(report.summary())

result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

### 엔진 직접 접근

```python
from vectrix.engine import AutoETS, AutoARIMA
from vectrix.adaptive import ForecastDNA

ets = AutoETS(period=7)
ets.fit(data)
pred, lower, upper = ets.predict(30)

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"난이도: {profile.difficulty}")
print(f"추천: {profile.recommendedModels}")
```

### Classic API

```python
from vectrix import Vectrix

fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="sales", steps=30)

if result.success:
    print(f"모델: {result.bestModelName}")
    print(f"예측: {result.predictions}")
```

<br>

## API 레퍼런스

### Easy API (권장)

| 함수 | 설명 |
|------|------|
| `forecast(data, steps=30)` | 자동 모델 선택 예측 |
| `analyze(data)` | DNA 프로파일링, 변환점, 이상치 |
| `regress(y, X)` / `regress(data=df, formula="y ~ x")` | 진단 포함 회귀분석 |
| `quick_report(data, steps=30)` | 분석 + 예측 통합 |

### Classic API

| 메서드 | 설명 |
|--------|------|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | 전체 파이프라인 |
| `Vectrix().analyze(df, dateCol, valueCol)` | 데이터 분석 |

### 반환 객체

| 객체 | 주요 속성 |
|------|----------|
| `EasyForecastResult` | `.predictions` `.dates` `.lower` `.upper` `.model` `.plot()` `.to_csv()` `.to_json()` |
| `EasyAnalysisResult` | `.dna` `.changepoints` `.anomalies` `.features` `.summary()` |
| `EasyRegressionResult` | `.coefficients` `.pvalues` `.r_squared` `.f_stat` `.summary()` `.diagnose()` |

<br>

## 후원

Vectrix가 유용하다면 프로젝트를 후원해주세요:

<a href="https://buymeacoffee.com/eddmpython">
  <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Vectrix%20후원하기-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee">
</a>

<br>

## 라이선스

[MIT](LICENSE) -- 개인 및 상업 프로젝트에서 자유롭게 사용 가능합니다.
