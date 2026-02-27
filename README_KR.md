<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/hero.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/hero.svg">
  <img alt="Vectrix — Navigate the Vector Space of Time" src=".github/assets/hero.svg" width="100%">
</picture>

<br>

<h3>순수 Python 시계열 예측 엔진</h3>

<p>
<img src="https://img.shields.io/badge/3-Dependencies-818cf8?style=for-the-badge&labelColor=0f172a" alt="Dependencies">
<img src="https://img.shields.io/badge/Pure-Python-6366f1?style=for-the-badge&labelColor=0f172a" alt="Pure Python">
<img src="https://img.shields.io/badge/No-Compiled%20Extensions-a78bfa?style=for-the-badge&labelColor=0f172a" alt="No Compiled Extensions">
</p>

<p>
<a href="https://pypi.org/project/vectrix/"><img src="https://img.shields.io/pypi/v/vectrix?style=for-the-badge&color=6366f1&labelColor=0f172a&logo=pypi&logoColor=white" alt="PyPI"></a>
<a href="https://pypi.org/project/vectrix/"><img src="https://img.shields.io/pypi/pyversions/vectrix?style=for-the-badge&labelColor=0f172a&logo=python&logoColor=white" alt="Python"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22d3ee?style=for-the-badge&labelColor=0f172a" alt="License"></a>
<img src="https://img.shields.io/badge/Tests-275%20passed-10b981?style=for-the-badge&labelColor=0f172a&logo=pytest&logoColor=white" alt="Tests">
</p>

<br>

<p>
<a href="https://eddmpython.github.io/vectrix/"><img src="https://img.shields.io/badge/Docs-eddmpython.github.io/vectrix-818cf8?style=for-the-badge&labelColor=0f172a&logo=readthedocs&logoColor=white" alt="Documentation"></a>
</p>

<p>
<a href="https://eddmpython.github.io/vectrix/">Documentation</a> ·
<a href="#-빠른-시작">빠른 시작</a> ·
<a href="#-모델">모델</a> ·
<a href="#-설치">설치</a> ·
<a href="#-사용-예시">사용 예시</a> ·
<a href="#-벤치마크">벤치마크</a> ·
<a href="#-api-레퍼런스">API</a> ·
<a href="README.md">English</a>
</p>

</div>

<br>

## ◈ Vectrix란?

Vectrix는 **의존성 3개**(NumPy, SciPy, Pandas)만으로 동작하는 시계열 예측 라이브러리입니다. C 컴파일러, cmdstan, 시스템 패키지 없이 `pip install`만 하면 됩니다.

### 예측

리스트, DataFrame, CSV 경로를 `forecast()`에 전달하면 됩니다. Vectrix가 여러 모델(ETS, ARIMA, Theta, TBATS, CES, MSTL)을 실행하고, 교차검증으로 최적 모델을 선택한 뒤, 신뢰구간이 포함된 예측을 반환합니다. 모델을 직접 고를 필요 없습니다.

```python
from vectrix import forecast
result = forecast("sales.csv", steps=12)
```

### 평탄 예측 방어

자동화된 예측에서 흔한 실패 패턴은 평탄한 직선 예측입니다. Vectrix는 4단계 감지-보정 시스템으로 이를 잡아내고, 실제 신호를 포착하는 모델로 대체합니다.

### Forecast DNA

모델을 피팅하기 전에 데이터를 65개 이상의 통계 특성(추세 강도, 계절성 강도, 엔트로피, 스펙트럼 밀도 등)으로 프로파일링하고, 어떤 모델이 적합한지 추천합니다.

### 회귀분석

R 스타일 수식 인터페이스와 진단을 지원합니다. OLS, Ridge, Lasso, Huber, Quantile 회귀가 포함됩니다.

```python
from vectrix import regress
model = regress(data=df, formula="sales ~ temperature + promotion")
print(model.summary())
```

진단: Durbin-Watson, Breusch-Pagan, VIF, 정규성 검정, 시계열 보정(Newey-West, Cochrane-Orcutt).

### 분석

`analyze()`로 데이터를 프로파일링하고 변환점, 이상치, 데이터 특성을 보고합니다.

```python
from vectrix import analyze
report = analyze(df, date="date", value="sales")
print(report.summary())
```

### 레짐 감지 & 자가 치유

순수 numpy HMM(Baum-Welch + Viterbi)으로 레짐 전환을 감지합니다. 레짐 변화가 발생하면 CUSUM + EWMA로 드리프트를 감지하고, 컨포멀 예측으로 예측을 재보정합니다.

### 비즈니스 제약조건

8종의 제약조건을 예측에 적용할 수 있습니다: 비음수, 범위, 용량, 전년 대비 변화 한도, 합계 제약, 단조성, 비율, 커스텀 함수.

### 계층적 조정

Bottom-up, Top-down, MinTrace로 계층적 시계열을 조정합니다.

### 전부 순수 Python

위의 모든 기능 — 예측 모델, 레짐 감지, 회귀분석 진단, 제약조건 적용, 계층적 조정 — 이 NumPy, SciPy, Pandas만으로 구현되어 있습니다. 컴파일된 확장이나 시스템 의존성이 없습니다.

<br>

## ◈ 빠른 시작

```bash
pip install vectrix
```

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
```

<br>

## ◈ 왜 Vectrix?

| | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| **순수 Python (C/Fortran 없음)** | ✅ | ❌ (numba) | ❌ (cmdstan) | ❌ (torch) |
| **의존성** | 3 | 5+ | 10+ | 20+ |
| **자동 모델 선택** | ✅ | ✅ | ❌ | ❌ |
| **평탄 예측 방어** | ✅ | ❌ | ❌ | ❌ |
| **비즈니스 제약조건** | 8종 | ❌ | ❌ | ❌ |
| **내장 회귀분석** | R 스타일 | ❌ | ❌ | ❌ |

<br>

## ◈ 모델

<details open>
<summary><b>핵심 예측 모델</b></summary>

<br>

| 모델 | 설명 |
|:-----|:-----|
| **AutoETS** | 30개 ExT×S 조합, AICc 자동 선택 |
| **AutoARIMA** | 계절성 ARIMA, 단계적 차수 선택 |
| **Theta / DOT** | Original + Dynamic Optimized Theta |
| **AutoCES** | Complex Exponential Smoothing |
| **AutoTBATS** | 삼각함수 다중 계절성 분해 |
| **GARCH** | GARCH, EGARCH, GJR-GARCH 변동성 |
| **Croston** | Classic, SBA, TSB 간헐적 수요 |
| **Logistic Growth** | 용량 제한 포화 추세 |
| **AutoMSTL** | 다중 계절성 STL + ARIMA 잔차 |
| **베이스라인** | Naive, Seasonal, Mean, Drift, Window Average |

</details>

<details>
<summary><b>실험적 방법론</b></summary>

<br>

| 방법 | 설명 |
|:-----|:-----|
| **Lotka-Volterra Ensemble** | 생태계 역학 기반 모델 가중치 |
| **Phase Transition** | 임계 둔화 → 레짐 전환 예측 |
| **Adversarial Stress** | 5가지 섭동 연산자 |
| **Hawkes Demand** | 자기 흥분 점 과정 |
| **Entropic Confidence** | Shannon 엔트로피 정량화 |

</details>

<details>
<summary><b>적응형 지능</b></summary>

<br>

| 시스템 | 설명 |
|:-------|:-----|
| **레짐 감지** | 순수 numpy HMM (Baum-Welch + Viterbi) |
| **자가 치유** | CUSUM + EWMA 드리프트 → 컨포멀 보정 |
| **제약 조건** | 8종: ≥0, 범위, 용량, YoY, Σ, ↑↓, 비율, 커스텀 |
| **Forecast DNA** | 65+ 특성 → 메타러닝 모델 추천 |
| **평탄 방어** | 4단계 방지 시스템 |

</details>

<details>
<summary><b>회귀분석 & 진단</b></summary>

<br>

| 기능 | 설명 |
|:-----|:-----|
| **방법** | OLS, Ridge, Lasso, Huber, Quantile |
| **수식** | R 스타일: `regress(data=df, formula="y ~ x")` |
| **진단** | Durbin-Watson, Breusch-Pagan, VIF, 정규성 |
| **변수 선택** | Stepwise, 정규화 CV, 최적 부분집합 |
| **시계열** | Newey-West, Cochrane-Orcutt, Granger |

</details>

<details>
<summary><b>비즈니스 인텔리전스</b></summary>

<br>

| 모듈 | 설명 |
|:-----|:-----|
| **이상치 탐지** | 자동 이상값 식별 및 설명 |
| **What-if** | 시나리오 기반 예측 시뮬레이션 |
| **백테스팅** | Rolling origin 교차 검증 |
| **계층 조정** | Bottom-up, Top-down, MinTrace |
| **예측 구간** | Conformal + Bootstrap |

</details>

<br>

## ◈ 설치

```bash
pip install vectrix                # 핵심 (numpy + scipy + pandas)
pip install "vectrix[numba]"       # + Numba JIT (2-5배 가속)
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # 전체
```

<br>

## ◈ 사용 예시

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

### 비즈니스 제약조건

```python
from vectrix.adaptive import ConstraintAwareForecaster, Constraint

caf = ConstraintAwareForecaster()
result = caf.apply(predictions, lower95, upper95, constraints=[
    Constraint('non_negative', {}),
    Constraint('range', {'min': 100, 'max': 5000}),
    Constraint('capacity', {'capacity': 10000}),
    Constraint('yoy_change', {'maxPct': 30, 'historicalData': past_year}),
])
```

<br>

## ◈ 벤치마크

M3, M4 대회 데이터셋으로 평가 (카테고리별 100개 시계열). OWA < 1.0이면 Naive2보다 우수.

**M3 Competition** — 4/4 카테고리에서 Naive2 능가:

| 카테고리 | OWA |
|:---------|:---:|
| Yearly | **0.848** |
| Quarterly | **0.825** |
| Monthly | **0.758** |
| Other | **0.819** |

**M4 Competition** — 4/6 빈도에서 Naive2 능가:

| 빈도 | OWA |
|:-----|:---:|
| Yearly | **0.974** |
| Quarterly | **0.797** |
| Monthly | **0.987** |
| Weekly | **0.737** |
| Daily | 1.207 |
| Hourly | 1.006 |

sMAPE/MASE 상세 결과: [벤치마크 상세](docs/benchmarks.ko.md)

<br>

## ◈ API 레퍼런스

### Easy API (권장)

| 함수 | 설명 |
|:-----|:-----|
| `forecast(data, steps=30)` | 자동 모델 선택 예측 |
| `analyze(data)` | DNA 프로파일링, 변환점, 이상치 |
| `regress(y, X)` / `regress(data=df, formula="y ~ x")` | 진단 포함 회귀분석 |
| `quick_report(data, steps=30)` | 분석 + 예측 통합 |

### Classic API

| 메서드 | 설명 |
|:-------|:-----|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | 전체 파이프라인 |
| `Vectrix().analyze(df, dateCol, valueCol)` | 데이터 분석 |

### 반환 객체

| 객체 | 주요 속성 |
|:-----|:---------|
| `EasyForecastResult` | `.predictions` `.dates` `.lower` `.upper` `.model` `.plot()` `.to_csv()` `.to_json()` |
| `EasyAnalysisResult` | `.dna` `.changepoints` `.anomalies` `.features` `.summary()` |
| `EasyRegressionResult` | `.coefficients` `.pvalues` `.r_squared` `.f_stat` `.summary()` `.diagnose()` |

<br>

## ◈ 아키텍처

```
vectrix/
├── easy.py               forecast(), analyze(), regress()
├── vectrix.py             Vectrix 클래스 — 전체 파이프라인
├── types.py               ForecastResult, DataCharacteristics
├── engine/                예측 모델
│   ├── ets.py               AutoETS (30개 조합)
│   ├── arima.py             AutoARIMA (AICc stepwise)
│   ├── theta.py             Theta method
│   ├── dot.py               Dynamic Optimized Theta
│   ├── ces.py               Complex Exponential Smoothing
│   ├── tbats.py             TBATS / AutoTBATS
│   ├── mstl.py              Multi-Seasonal Decomposition
│   ├── garch.py             GARCH / EGARCH / GJR-GARCH
│   ├── croston.py           Croston Classic / SBA / TSB
│   ├── logistic.py          Logistic Growth
│   ├── hawkes.py            Hawkes Intermittent Demand
│   ├── lotkaVolterra.py     Lotka-Volterra Ensemble
│   ├── phaseTransition.py   Phase Transition Forecaster
│   ├── adversarial.py       Adversarial Stress Tester
│   ├── entropic.py          Entropic Confidence Scorer
│   └── turbo.py             Numba JIT acceleration
├── adaptive/              레짐, 자가치유, 제약조건, DNA
├── regression/            OLS, Ridge, Lasso, Huber, Quantile
├── business/              이상치, 백테스트, 시나리오, 지표
├── flat_defense/          4단계 평탄 예측 방지
├── hierarchy/             Bottom-up, Top-down, MinTrace
├── intervals/             Conformal + Bootstrap 구간
├── ml/                    LightGBM, XGBoost 래퍼
└── global_model/          크로스시리즈 예측
```

<br>

## ◈ 기여

```bash
git clone https://github.com/eddmpython/vectrix.git
cd vectrix
uv sync --extra dev
uv run pytest
```

<br>

## ◈ 후원

Vectrix가 유용하다면 프로젝트를 지원해주세요:

<a href="https://buymeacoffee.com/eddmpython">
  <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Vectrix%20후원하기-f59e0b?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white&labelColor=0f172a" alt="Buy Me a Coffee">
</a>

<br><br>

## ◈ 라이선스

[MIT](LICENSE) — 개인 및 상업 프로젝트에서 자유롭게 사용 가능합니다.

<br>

<div align="center">

*데이터의 미지 차원을 탐색합니다.*

</div>
