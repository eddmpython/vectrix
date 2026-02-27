<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Vectrix-%EC%8B%9C%EA%B3%84%EC%97%B4%20%EC%98%88%EC%B8%A1-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwb2x5bGluZSBwb2ludHM9IjIyIDEyIDE4IDEyIDE1IDE5IDkgNSA2IDEyIDIgMTIiLz48L3N2Zz4=">
  <img alt="Vectrix" src="https://img.shields.io/badge/Vectrix-%EC%8B%9C%EA%B3%84%EC%97%B4%20%EC%98%88%EC%B8%A1-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwb2x5bGluZSBwb2ludHM9IjIyIDEyIDE4IDEyIDE1IDE5IDkgNSA2IDEyIDIgMTIiLz48L3N2Zz4=">
</picture>

### 설정 없이 바로 사용하는 시계열 예측 라이브러리

순수 Python 시계열 예측 -- 30+ 모델, 무거운 의존성 제로.

<br>

[![PyPI](https://img.shields.io/pypi/v/vectrix?style=flat-square&color=6366f1)](https://pypi.org/project/vectrix/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/vectrix?style=flat-square)](https://pypi.org/project/vectrix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Buy Me a Coffee](https://img.shields.io/badge/Sponsor-Buy%20Me%20a%20Coffee-orange?style=flat-square&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/eddmpython)

[설치](#설치) &middot; [빠른 시작](#빠른-시작) &middot; [주요 기능](#주요-기능) &middot; [API 레퍼런스](#api-레퍼런스) &middot; [English](README.md)

</div>

<br>

> **의존성 3개. 모델 30+개. 코드 1줄.**
>
> Vectrix는 순수 NumPy + SciPy로 처음부터 구현한 시계열 예측 라이브러리입니다.
> statsforecast, statsmodels, Prophet 없이 -- 데이터만 넣으면 최적 예측과 신뢰구간을 제공합니다.

<br>

## 빠른 시작

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
```

한 줄이면 모델 선택, 일직선 예측 방지, 신뢰구간까지 포함된 예측이 완성됩니다.

## 왜 Vectrix?

| 라이브러리 | 문제점 |
|-----------|-------|
| **statsforecast** | 무거운 의존성, C 컴파일 문제 |
| **Prophet** | pystan/cmdstan 필요, 200MB+ 설치 |
| **statsmodels** | 수동 모델 선택, 자동 파이프라인 없음 |
| **NeuralProphet** | PyTorch 필요, GPU 지향 |
| **darts** | 과도한 기능, 느린 시작 |

Vectrix는 **순수 Python, 의존성 3개, MIT 라이선스**로 프로덕션 수준의 예측을 제공합니다.

## 주요 기능

<details>
<summary><b>핵심 예측 모델</b></summary>

<br>

- **AutoETS** -- 30개 Error x Trend x Seasonal 조합, AICc 자동 선택
- **AutoARIMA** -- 계절성 ARIMA, AICc 기반 단계적 차수 선택
- **Theta / DOT** -- Original Theta + Dynamic Optimized Theta (M3 대회 우승)
- **AutoCES** -- Complex Exponential Smoothing (Svetunkov 2023)
- **AutoTBATS** -- 삼각함수 계절성, 복잡한 다중 계절 데이터 처리
- **GARCH** -- GARCH, EGARCH, GJR-GARCH 변동성 모델링
- **Croston** -- Classic, SBA, TSB + AutoCroston 간헐적 수요 예측
- **Logistic Growth** -- Prophet 스타일 포화 추세
- **AutoMSTL** -- 다중 계절성 분해 + ARIMA 잔차 예측
- **베이스라인** -- Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

</details>

<details>
<summary><b>세계 최초 방법론</b></summary>

<br>

- **Lotka-Volterra Ensemble** -- 생태계 경쟁 역학 기반 적응형 모델 가중치
- **Phase Transition Forecaster** -- 임계 둔화 감지로 레짐 전환 예측
- **Adversarial Stress Tester** -- 5가지 섭동 연산자로 예측 견고성 분석
- **Hawkes Intermittent Demand** -- 자기 흥분 점 과정으로 군집 수요 패턴 처리
- **Entropic Confidence Scorer** -- Shannon 엔트로피 기반 예측 불확실성 정량화

</details>

<details>
<summary><b>적응형 지능</b></summary>

<br>

- **레짐 감지** -- 순수 numpy HMM (Baum-Welch + Viterbi)
- **자가 치유 예측** -- CUSUM + EWMA 드리프트 감지, 컨포멀 예측 보정
- **제약 조건 예측** -- 8가지 비즈니스 제약: 비음수, 범위, 용량, YoY 변화, 합계, 단조, 비율, 커스텀
- **Forecast DNA** -- 65+ 특성 지문으로 메타러닝 모델 추천 및 유사도 검색
- **평탄 방어** -- 4단계 시스템 (진단, 감지, 교정, 방지)

</details>

<details>
<summary><b>회귀분석 & 진단</b></summary>

<br>

- **5가지 회귀 방법** -- OLS, Ridge, Lasso, Huber, Quantile
- **R 스타일 수식** -- `regress(data=df, formula="sales ~ ads + price")`
- **전체 진단** -- Durbin-Watson, Breusch-Pagan, VIF, 정규성 검정
- **변수 선택** -- Stepwise, 정규화 CV, 최적 부분집합
- **시계열 회귀** -- Newey-West, Cochrane-Orcutt, Prais-Winsten, Granger 인과성

</details>

<details>
<summary><b>비즈니스 인텔리전스</b></summary>

<br>

- **이상치 탐지** -- 자동 이상값 식별 및 설명
- **What-if 분석** -- 시나리오 기반 예측 시뮬레이션
- **백테스팅** -- Rolling origin 교차 검증, 다중 메트릭
- **계층 조정** -- Bottom-up, Top-down, MinTrace 최적 조정
- **예측 구간** -- Conformal + Bootstrap 방법

</details>

## 설치

### uv 사용 (권장)

```bash
uv init my-forecast && cd my-forecast
uv add vectrix

uv add "vectrix[numba]"     # Numba JIT 가속
```

### pip 사용

```bash
pip install vectrix

pip install "vectrix[numba]"  # Numba 포함
pip install "vectrix[ml]"     # LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"    # 전체
```

**요구사항:** Python 3.10+, NumPy >= 1.24, Pandas >= 2.0, SciPy >= 1.10

## 사용 예시

### 기본 예측

```python
from vectrix import forecast

result = forecast([100, 120, 115, 130, 125, 140, 135, 150], steps=5)
print(result)
```

리스트, NumPy 배열, Pandas Series, DataFrame, dict, CSV 파일 경로 모두 지원합니다.

### DataFrame + 분석

```python
from vectrix import forecast, analyze
import pandas as pd

df = pd.read_csv("data.csv")

report = analyze(df, date="date", value="sales")
print(report.summary())
print(f"난이도: {report.dna.difficulty}")
print(f"추천 모델: {report.dna.recommendedModels}")

result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

### R 스타일 회귀분석

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ temperature + promotion + holiday")
print(model.summary())
print(model.diagnose())
```

### 엔진 직접 접근

```python
from vectrix.engine import AutoETS, AutoARIMA, AutoTBATS
from vectrix.adaptive import RegimeDetector, ForecastDNA

ets = AutoETS(period=7)
ets.fit(data)
pred, lower, upper = ets.predict(30)

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"지문: {profile.fingerprint}")
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
    print(f"95% CI: [{result.lower95}, {result.upper95}]")
```

## API 레퍼런스

### Easy API (권장)

| 함수 | 설명 |
|------|------|
| `forecast(data, steps=30)` | 자동 모델 선택 원콜 예측 |
| `analyze(data)` | 시계열 DNA 프로파일링, 변환점, 이상치 |
| `regress(y, X)` 또는 `regress(data=df, formula="y ~ x")` | 전체 진단 포함 회귀분석 |
| `quick_report(data, steps=30)` | 분석 + 예측 통합 리포트 |

### Classic API

| 메서드 | 설명 |
|--------|------|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | 상세 결과 객체 포함 전체 파이프라인 |
| `Vectrix().analyze(df, dateCol, valueCol)` | 데이터 특성 + 평탄 위험도 평가 |

## 의존성

| 패키지 | 필수 | 용도 |
|-------|------|------|
| numpy >= 1.24 | O | 핵심 연산 |
| pandas >= 2.0 | O | 데이터 처리 |
| scipy >= 1.10 | O | 파라미터 최적화 |
| numba | X | JIT 가속 (2-5배) |
| lightgbm / xgboost | X | ML 기반 예측 |
| scikit-learn | X | ML 유틸리티 |

## 테스트

```bash
uv run pytest
```

275개 테스트로 모든 모델, 엣지 케이스, 통합 시나리오를 커버합니다.

## 후원

Vectrix가 유용하다면 프로젝트를 후원해주세요:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-후원하기-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/eddmpython)

## 라이선스

[MIT](LICENSE) -- 개인 및 상업 프로젝트에서 자유롭게 사용 가능합니다.
