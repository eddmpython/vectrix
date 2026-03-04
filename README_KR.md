<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/hero.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/hero.svg">
  <img alt="Vectrix — Navigate the Vector Space of Time" src=".github/assets/hero.svg" width="100%">
</picture>

<br>

<h3>시계열 예측 엔진 — 내장 Rust 가속</h3>

<p>
<img src="https://img.shields.io/badge/3-Dependencies-818cf8?style=for-the-badge&labelColor=0f172a" alt="Dependencies">
<img src="https://img.shields.io/badge/Built--in-Rust%20Engine-e45a33?style=for-the-badge&labelColor=0f172a&logo=rust&logoColor=white" alt="Built-in Rust Engine">
</p>

<p>
<a href="https://pypi.org/project/vectrix/"><img src="https://img.shields.io/pypi/v/vectrix?style=for-the-badge&color=6366f1&labelColor=0f172a&logo=pypi&logoColor=white" alt="PyPI"></a>
<a href="https://pypi.org/project/vectrix/"><img src="https://img.shields.io/pypi/pyversions/vectrix?style=for-the-badge&labelColor=0f172a&logo=python&logoColor=white" alt="Python"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-22d3ee?style=for-the-badge&labelColor=0f172a" alt="License"></a>
<img src="https://img.shields.io/badge/Tests-573%20passed-10b981?style=for-the-badge&labelColor=0f172a&logo=pytest&logoColor=white" alt="Tests">
</p>

<br>

<p>
<a href="https://eddmpython.github.io/vectrix/"><img src="https://img.shields.io/badge/Docs-eddmpython.github.io/vectrix-818cf8?style=for-the-badge&labelColor=0f172a&logo=readthedocs&logoColor=white" alt="Documentation"></a>
<a href="https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/01_quickstart.ipynb"><img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&labelColor=0f172a&logo=googlecolab&logoColor=white" alt="Open in Colab"></a>
</p>

<p>
<a href="https://eddmpython.github.io/vectrix/">Documentation</a> ·
<a href="#-빠른-시작">빠른 시작</a> ·
<a href="#-모델">모델</a> ·
<a href="#-설치">설치</a> ·
<a href="#-사용-예시">사용 예시</a> ·
<a href="#-벤치마크">벤치마크</a> ·
<a href="#-api-레퍼런스">API</a> ·
<a href="#-인터랙티브-노트북">노트북</a> ·
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

### 내장 Rust 엔진

25개 핵심 예측 핫 루프가 Rust로 가속되어 모든 wheel에 컴파일됩니다. Polars처럼 설치만 하면 자동으로 빨라집니다.

| 구성요소 | Python Only | Rust 적용 | 속도 향상 |
|:---------|:-----------|:----------|:---------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| DOT fit | 240ms | **10ms** | **24x** |
| ETS 필터 (핫 루프) | 0.17ms | **0.003ms** | **67x** |

Rust 컴파일러 불필요. `pip install vectrix`만 하면 가속이 자동 적용됩니다.

### 내장 샘플 데이터셋

즉시 테스트 가능한 7개 데이터셋 내장:

```python
from vectrix import loadSample, forecast

df = loadSample("airline")       # 144개 월간 관측
result = forecast(df, date="date", value="passengers", steps=12)
```

사용 가능: `airline`, `retail`, `stock`, `temperature`, `energy`, `web`, `intermittent`

### 최소 의존성, 최대 성능

위의 모든 기능 — 예측 모델, 레짐 감지, 회귀분석 진단, 제약조건 적용, 계층적 조정 — 이 NumPy, SciPy, Pandas + 내장 Rust 엔진으로 구현되어 있습니다. 설치 하나로 전체 성능을 즉시 사용할 수 있습니다.

<br>

## ◈ 빠른 시작

```bash
pip install vectrix
```

```python
from vectrix import forecast, loadSample

df = loadSample("airline")
result = forecast(df, date="date", value="passengers", steps=12)
print(result)
result.plot()
```

<br>

## ◈ 왜 Vectrix?

| | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| **내장 Rust 엔진** | ✅ (5-67x) | ❌ | ❌ | ❌ |
| **의존성** | 3 | 5+ | 10+ | 20+ |
| **자동 모델 선택** | ✅ | ✅ | ❌ | ❌ |
| **평탄 예측 방어** | ✅ | ❌ | ❌ | ❌ |
| **비즈니스 제약조건** | 8종 | ❌ | ❌ | ❌ |
| **내장 회귀분석** | R 스타일 | ❌ | ❌ | ❌ |
| **샘플 데이터셋** | 7종 내장 | ❌ | ❌ | ✅ |

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
| **4Theta** | M4 Competition 방법론, 4개 theta line 가중 |
| **DTSF** | Dynamic Time Scan, 비모수 패턴 매칭 |
| **ESN** | Echo State Network, 저수지 컴퓨팅 |
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
pip install vectrix                  # 30+ 모델 + Rust 엔진 내장
pip install "vectrix[ml]"            # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"    # + Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[all]"           # 전체
```

<br>

## ◈ 사용 예시

### Easy API

```python
from vectrix import forecast, analyze, regress, compare

# Level 1 — 제로 설정
result = forecast([100, 120, 115, 130, 125, 140], steps=5)

# Level 2 — 가이드 제어
result = forecast(df, date="date", value="sales", steps=12,
                  models=["dot", "auto_ets", "auto_ces"],
                  ensemble="mean",
                  confidence=0.90)

print(result.compare())          # 전체 모델 순위
print(result.all_forecasts())    # 모든 모델의 예측값

report = analyze(df, date="date", value="sales")
print(f"난이도: {report.dna.difficulty}")

comparison = compare(df, date="date", value="sales", steps=12)

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

**M4 Competition 100,000 시계열** 벤치마크 (빈도별 2,000 샘플, seed=42). OWA < 1.0이면 Naive2보다 우수.

**DOT-Hybrid** (단일 모델, OWA 0.877 — M4 #18 Theta 0.897 초과):

| 빈도 | OWA | vs Naive2 |
|:-----|:---:|:---------:|
| Yearly | **0.797** | -20.3% |
| Quarterly | **0.894** | -10.6% |
| Monthly | **0.897** | -10.3% |
| Weekly | **0.959** | -4.1% |
| Daily | **0.996** | -0.4% |
| Hourly | **0.722** | -27.8% |

**M4 Competition 리더보드 위치:**

| 순위 | 방법 | OWA |
|:-----|:-----|:---:|
| #1 | ES-RNN (Smyl) | 0.821 |
| #2 | FFORMA | 0.838 |
| #11 | 4Theta | 0.874 |
| — | **Vectrix DOT-Hybrid** | **0.877** |
| #18 | Theta | 0.897 |

sMAPE/MASE 상세 결과: [벤치마크 상세](https://eddmpython.github.io/vectrix/docs/benchmarks/)

<br>

## ◈ 인터랙티브 노트북

설치 없이 바로 체험 — Google Colab에서 클릭 한 번으로 실행.

| 노트북 | 설명 | 링크 |
|:-------|:-----|:----:|
| **Quick Start** | 예측, 분석, 회귀를 5분 안에 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/01_quickstart.ipynb) |
| **Models & DNA** | 30+ 모델 비교, DNA 프로파일링 심화 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/02_models_and_dna.ipynb) |
| **Try Your Data** | CSV 업로드만 하면 즉시 분석 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/03_try_your_data.ipynb) |

<br>

## ◈ API 레퍼런스

### Easy API (권장)

| 함수 | 설명 |
|:-----|:-----|
| `forecast(data, steps, models, ensemble, confidence)` | 자동 또는 가이드 예측 |
| `analyze(data, period, features)` | DNA 프로파일링, 변환점, 이상치 |
| `regress(y, X)` / `regress(data=df, formula="y ~ x")` | 진단 포함 회귀분석 |
| `compare(data, steps, models)` | 모델 비교 (DataFrame) |
| `quick_report(data, steps)` | 분석 + 예측 통합 |

`data` 이외의 모든 파라미터는 합리적인 기본값을 가진 선택 사항입니다. [Progressive Disclosure](#api-레이어) 설계를 참조하세요.

### Classic API

| 메서드 | 설명 |
|:-------|:-----|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | 전체 파이프라인 |
| `Vectrix().analyze(df, dateCol, valueCol)` | 데이터 분석 |

### 반환 객체

| 객체 | 주요 속성 |
|:-----|:---------|
| `EasyForecastResult` | `.predictions` `.dates` `.lower` `.upper` `.model` `.mape` `.rmse` `.models` `.compare()` `.all_forecasts()` `.plot()` `.to_csv()` `.to_json()` |
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
│   ├── fourTheta.py         4Theta (M4 Competition 방법론)
│   ├── dtsf.py              Dynamic Time Scan Forecaster
│   ├── esn.py               Echo State Network
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
├── global_model/          크로스시리즈 예측
└── datasets.py            7개 내장 샘플 데이터셋

rust/                         내장 Rust 엔진 (25개 가속 함수)
└── src/lib.rs             ETS, ARIMA, Theta, DOT, CES, GARCH 등 (PyO3)
```

<br>

## ◈ AI 통합

Vectrix는 AI 어시스턴트가 완벽하게 이해할 수 있도록 설계되었습니다. Claude, GPT, Copilot 등 어떤 AI 도구를 사용하든, 구조화된 컨텍스트 파일을 통해 전체 API를 한 번에 파악할 수 있습니다.

### llms.txt — AI용 문서

[`llms.txt`](https://llmstxt.org/) 표준은 AI 어시스턴트에게 프로젝트의 구조화된 개요를 제공하고, `llms-full.txt`는 모든 함수 시그니처, 파라미터, 반환 타입, 사용 패턴을 포함한 완전한 API 레퍼런스입니다.

| 파일 | URL | 내용 |
|:-----|:----|:-----|
| `llms.txt` | [eddmpython.github.io/vectrix/llms.txt](https://eddmpython.github.io/vectrix/llms.txt) | 프로젝트 개요 + 문서 링크 |
| `llms-full.txt` | [eddmpython.github.io/vectrix/llms-full.txt](https://eddmpython.github.io/vectrix/llms-full.txt) | 완전한 API 레퍼런스 — 모든 클래스, 메서드, 파라미터, 주의사항 |

AI 어시스턴트에게 `llms-full.txt`를 읽게 하면 세션이 바뀌어도 라이브러리를 즉시 이해합니다.

### MCP 서버 — AI 도구 호출

[Model Context Protocol](https://modelcontextprotocol.io/) 서버는 Vectrix를 Claude Desktop, Claude Code 등 MCP 호환 AI 어시스턴트에서 직접 호출할 수 있는 도구로 노출합니다.

**10개 도구**: `forecast_timeseries`, `forecast_csv`, `analyze_timeseries`, `compare_models`, `run_regression`, `detect_anomalies`, `backtest_model`, `list_sample_datasets`, `load_sample_dataset`

```bash
# Claude Code 설정
pip install "vectrix[mcp]"
claude mcp add --transport stdio vectrix -- uv run python mcp/server.py

# Claude Desktop 설정 (claude_desktop_config.json에 추가)
{
    "mcpServers": {
        "vectrix": {
            "command": "uv",
            "args": ["run", "python", "/path/to/mcp/server.py"]
        }
    }
}
```

연결 후 AI에게 *"이 매출 데이터를 12개월 예측해줘"*라고 말하면 AI가 Vectrix를 직접 호출합니다.

### Claude Code Skills

Claude Code 사용자를 위한 3개 전문 스킬:

| 스킬 | 명령어 | 설명 |
|:------|:--------|:-----|
| `vectrix-forecast` | `/vectrix-forecast` | 시계열 예측 워크플로우 |
| `vectrix-analyze` | `/vectrix-analyze` | DNA 프로파일링 및 이상치 탐지 |
| `vectrix-regress` | `/vectrix-regress` | R-스타일 회귀분석 + 진단 |

Vectrix 프로젝트 디렉토리에서 작업할 때 자동으로 로드됩니다.

<br>

## ◈ 철학 & 로드맵

### 정체성

Vectrix는 **내장 Rust 가속을 갖춘 제로 설정 예측 엔진**입니다. 설계 철학:

- **Python 문법, Rust 속도** — Polars처럼 Rust 엔진은 투명합니다. 사용자는 Python을 작성하고, 핫 루프는 자동으로 Rust에서 실행됩니다.
- **점진적 공개(Progressive Disclosure)** — 초보자는 `forecast(data, steps=12)`를 설정 없이 호출합니다. 전문가는 `models=`, `ensemble=`, `confidence=`를 전달하여 모든 측면을 제어합니다. 엔진 직접 접근(`AutoETS`, `AutoARIMA`)은 항상 가능합니다.
- **의존성 3개, 컴파일러 불필요** — NumPy, SciPy, Pandas. 시스템 패키지, Numba JIT 워밍업, CmdStan 없음. `pip install vectrix`로 끝.
- **기능보다 정확성** — 모든 빈도에서 Naive2를 이기는 15개 모델이, Daily/Hourly에서 실패하는 50개 모델보다 낫습니다.

### API 레이어

| 레이어 | 대상 | 예시 |
|:-------|:-----|:-----|
| **Level 1 — 제로 설정** | 초보자, 빠른 프로토타입 | `forecast(data, steps=12)` |
| **Level 2 — 가이드 제어** | 데이터 과학자, 프로덕션 | `forecast(data, steps=12, models=["dot", "auto_ets"], ensemble="mean", confidence=0.90)` |
| **Level 3 — 엔진 직접** | 연구자, 커스텀 파이프라인 | `AutoETS(period=7).fit(data).predict(30)` |

Level 2의 모든 파라미터에는 Level 1 동작을 재현하는 합리적인 기본값이 있습니다. 필수 파라미터는 없습니다.

### 로드맵

| 우선순위 | 영역 | 현재 | 목표 | 상태 |
|:---------|:-----|:-----|:-----|:-----|
| **P0** | M4 정확도 | OWA 0.877 | OWA < 0.850 | 진행 중 |
| **P1** | Easy API Progressive Disclosure | Level 1만 | Level 1-3 | 진행 중 |
| **P2** | 파이프라인 속도 | 48ms forecast() | < 10ms | 계획 |
| **P3** | Foundation Model 깊이 | 기본 래퍼 | 완전 통합 | 계획 |
| **P4** | 커뮤니티 성장 | 초기 단계 | 블로그, Reddit, Kaggle | 진행 중 |

### 확장 원칙

1. **정확도 우선, 속도 그다음** — 빠르게 전달된 잘못된 답은 여전히 잘못됩니다. M4 OWA 개선이 지연 시간 최적화보다 먼저.
2. **제로 설정을 절대 깨지 않는다** — 모든 새 파라미터에는 기본값이 있어야 합니다. `forecast(data, steps=12)`는 항상 작동해야 합니다.
3. **하나의 정체성** — "Python 문법, Rust 속도, 제로 설정." 모든 기능, 문서, 마케팅 메시지가 이에 부합합니다.
4. **벤치마크 기반** — 모든 엔진 변경은 M4 100K 시리즈로 검증합니다. "더 좋아 보인다"가 아닌 OWA로 보여주기.
5. **최소 의존성** — 의존성 추가에는 강력한 근거가 필요합니다. numpy/scipy로 구현 가능하면 그렇게 해야 합니다.

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
