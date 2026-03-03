<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/hero.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/hero.svg">
  <img alt="Vectrix — Navigate the Vector Space of Time" src=".github/assets/hero.svg" width="100%">
</picture>

<br>

<h3>Time Series Forecasting Engine — Built-in Rust Acceleration</h3>

<p>
<img src="https://img.shields.io/badge/3-Dependencies-818cf8?style=for-the-badge&labelColor=0f172a" alt="Dependencies">
<img src="https://img.shields.io/badge/Built--in-Rust%20Engine-e45a33?style=for-the-badge&labelColor=0f172a&logo=rust&logoColor=white" alt="Built-in Rust Engine">
<img src="https://img.shields.io/badge/Python-3.10+-6366f1?style=for-the-badge&labelColor=0f172a&logo=python&logoColor=white" alt="Python 3.10+">
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
</p>

<p>
<a href="https://eddmpython.github.io/vectrix/">Documentation</a> ·
<a href="#-quick-start">Quick Start</a> ·
<a href="#-models">Models</a> ·
<a href="#-installation">Installation</a> ·
<a href="#-usage">Usage</a> ·
<a href="#-benchmarks">Benchmarks</a> ·
<a href="#-api-reference">API Reference</a> ·
<a href="https://eddmpython.github.io/vectrix/docs/tutorials/">Tutorials</a> ·
<a href="https://eddmpython.github.io/vectrix/docs/showcase/">Showcase</a> ·
<a href="README_KR.md">한국어</a>
</p>

</div>

<br>

## ◈ What is Vectrix?

Vectrix is a time series forecasting library with a **built-in Rust engine** for blazing-fast performance. 3 dependencies (NumPy, SciPy, Pandas), no compiler needed — `pip install vectrix` and the Rust-accelerated engine is included in the wheel.

### Forecasting

Pass a list, DataFrame, or CSV path to `forecast()`. Vectrix runs multiple models (ETS, ARIMA, Theta, TBATS, CES, MSTL), evaluates each with cross-validation, and returns the best prediction with confidence intervals. You don't choose a model — it does.

```python
from vectrix import forecast
result = forecast("sales.csv", steps=12)
```

### Flat-Line Defense

A common failure mode in automated forecasting is flat predictions — the model outputs a constant line. Vectrix has a 4-level detection and correction system that catches this and falls back to a model that actually captures the signal.

### Forecast DNA

Before fitting any model, Vectrix profiles your data with 65+ statistical features (trend strength, seasonality strength, entropy, spectral density, etc.) and uses them to recommend which models are likely to work best.

### Regression

R-style formula interface with full diagnostics. OLS, Ridge, Lasso, Huber, and Quantile regression are included.

```python
from vectrix import regress
model = regress(data=df, formula="sales ~ temperature + promotion")
print(model.summary())
```

Diagnostics include Durbin-Watson, Breusch-Pagan, VIF, normality tests, and time series adjustments (Newey-West, Cochrane-Orcutt).

### Analysis

`analyze()` profiles the data and reports changepoints, anomalies, and data characteristics.

```python
from vectrix import analyze
report = analyze(df, date="date", value="sales")
print(report.summary())
```

### Regime Detection & Self-Healing

A pure-numpy HMM (Baum-Welch + Viterbi) detects regime shifts. When a regime change occurs, the self-healing system uses CUSUM + EWMA to detect drift and applies conformal prediction to recalibrate the forecast.

### Business Constraints

8 constraint types can be applied to any forecast: non-negative, range, capacity, year-over-year change limit, sum constraint, monotonicity, ratio, and custom functions.

### Hierarchical Reconciliation

Bottom-up, top-down, and MinTrace reconciliation for hierarchical time series.

### Built-in Rust Engine

Every `pip install vectrix` includes a pre-built Rust extension — like Polars, no compiler needed. 25 core hot loops are Rust-accelerated across all forecasting engines.

| Component | Python Only | With Rust | Speedup |
|:----------|:-----------|:----------|:--------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| DOT fit | 240ms | **10ms** | **24x** |
| ETS filter (hot loop) | 0.17ms | **0.003ms** | **67x** |

Pre-built wheels for Linux (x86_64), macOS (ARM + x86), and Windows. The Rust engine is included in the default installation — no extras, no flags, no `[turbo]`.

### Built-in Sample Datasets

7 ready-to-use datasets for quick testing:

```python
from vectrix import loadSample, forecast

df = loadSample("airline")       # 144 monthly observations
result = forecast(df, date="date", value="passengers", steps=12)
```

Available: `airline`, `retail`, `stock`, `temperature`, `energy`, `web`, `intermittent`

### Minimal Dependencies, Maximum Performance

All of the above — forecasting models, regime detection, regression diagnostics, constraint enforcement, hierarchical reconciliation — runs on just NumPy, SciPy, and Pandas. The Rust engine is compiled into the wheel and loaded automatically. No system dependencies, no compiler, no extra install steps.

<br>

## ◈ Quick Start

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

## ◈ Why Vectrix?

| | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| **Built-in Rust engine** | ✅ (5-67x) | ❌ | ❌ | ❌ |
| **No compiler needed** | ✅ | ❌ (numba) | ❌ (cmdstan) | ❌ (torch) |
| **Dependencies** | 3 | 5+ | 10+ | 20+ |
| **Auto model selection** | ✅ | ✅ | ❌ | ❌ |
| **Flat-line defense** | ✅ | ❌ | ❌ | ❌ |
| **Business constraints** | 8 types | ❌ | ❌ | ❌ |
| **Built-in regression** | R-style | ❌ | ❌ | ❌ |
| **Sample datasets** | 7 built-in | ❌ | ❌ | ✅ |

> **Comparison notes**: Dependencies counted as direct `pip install` requirements (not transitive). Vectrix's Rust engine is compiled into the wheel (like Polars) — no separate install needed. statsforecast requires Numba JIT compilation; Prophet requires CmdStan (C++ compiler); Darts requires PyTorch. Feature comparison based on statsforecast 2.0+, Prophet 1.1+, Darts 0.31+.

<br>

## ◈ Models

<details open>
<summary><b>Core Forecasting Models</b></summary>

<br>

| Model | Description |
|:------|:------------|
| **AutoETS** | 30 ExT×S combinations, AICc selection |
| **AutoARIMA** | Seasonal ARIMA, stepwise order selection |
| **Theta / DOT** | Original + Dynamic Optimized Theta |
| **AutoCES** | Complex Exponential Smoothing |
| **AutoTBATS** | Trigonometric multi-seasonal decomposition |
| **GARCH** | GARCH, EGARCH, GJR-GARCH volatility |
| **Croston** | Classic, SBA, TSB intermittent demand |
| **Logistic Growth** | Saturating trends with capacity constraints |
| **AutoMSTL** | Multi-seasonal STL + ARIMA residuals |
| **4Theta** | M4 Competition method, 4 theta lines weighted |
| **DTSF** | Dynamic Time Scan, non-parametric pattern matching |
| **ESN** | Echo State Network, reservoir computing |
| **Baselines** | Naive, Seasonal, Mean, Drift, Window Average |

</details>

<details>
<summary><b>Experimental Methods</b></summary>

<br>

| Method | Description |
|:-------|:------------|
| **Lotka-Volterra Ensemble** | Ecological dynamics for model weighting |
| **Phase Transition** | Critical slowing → regime shift |
| **Adversarial Stress** | 5 perturbation operators |
| **Hawkes Demand** | Self-exciting point process |
| **Entropic Confidence** | Shannon entropy quantification |

</details>

<details>
<summary><b>Adaptive Intelligence</b></summary>

<br>

| System | Description |
|:-------|:------------|
| **Regime Detection** | Pure numpy HMM (Baum-Welch + Viterbi) |
| **Self-Healing** | CUSUM + EWMA drift → conformal correction |
| **Constraints** | 8 types: ≥0, range, cap, YoY, Σ, ↑↓, ratio, fn |
| **Forecast DNA** | 65+ features → meta-learning recommendation |
| **Flat Defense** | 4-level prevention system |

</details>

<details>
<summary><b>Regression & Diagnostics</b></summary>

<br>

| Capability | Description |
|:-----------|:------------|
| **Methods** | OLS, Ridge, Lasso, Huber, Quantile |
| **Formula** | R-style: `regress(data=df, formula="y ~ x")` |
| **Diagnostics** | Durbin-Watson, Breusch-Pagan, VIF, normality |
| **Selection** | Stepwise, regularization CV, best subset |
| **Time Series** | Newey-West, Cochrane-Orcutt, Granger |

</details>

<details>
<summary><b>Business Intelligence</b></summary>

<br>

| Module | Description |
|:-------|:------------|
| **Anomaly** | Automated outlier detection & explanation |
| **What-if** | Scenario-based forecast simulation |
| **Backtesting** | Rolling origin cross-validation |
| **Hierarchy** | Bottom-up, top-down, MinTrace |
| **Intervals** | Conformal + bootstrap prediction |

</details>

<br>

## ◈ Installation

```bash
pip install vectrix                # Rust engine included — no extras needed
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # Everything
```

<br>

## ◈ Usage

### Easy API

```python
from vectrix import forecast, analyze, regress, compare

result = forecast([100, 120, 115, 130, 125, 140], steps=5)
print(result.compare())          # All model rankings
print(result.all_forecasts())    # Every model's predictions

report = analyze(df, date="date", value="sales")
print(f"Difficulty: {report.dna.difficulty}")

comparison = compare(df, date="date", value="sales", steps=12)

model = regress(data=df, formula="sales ~ temperature + promotion")
print(model.summary())
```

### DataFrame Workflow

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

### Direct Engine Access

```python
from vectrix.engine import AutoETS, AutoARIMA
from vectrix.adaptive import ForecastDNA

ets = AutoETS(period=7)
ets.fit(data)
pred, lower, upper = ets.predict(30)

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"Difficulty: {profile.difficulty}")
print(f"Recommended: {profile.recommendedModels}")
```

### Business Constraints

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

## ◈ Benchmarks

Evaluated on M3 and M4 competition datasets (first 100 series per category). OWA < 1.0 means better than Naive2.

**M3 Competition** — 4/4 categories beat Naive2:

| Category | OWA |
|:---------|:---:|
| Yearly | **0.848** |
| Quarterly | **0.825** |
| Monthly | **0.758** |
| Other | **0.819** |

**M4 Competition** — 4/6 frequencies beat Naive2:

| Frequency | OWA |
|:----------|:---:|
| Yearly | **0.974** |
| Quarterly | **0.797** |
| Monthly | **0.987** |
| Weekly | **0.737** |
| Daily | 1.207 |
| Hourly | 1.006 |

Full results with sMAPE/MASE breakdown: [benchmarks](https://eddmpython.github.io/vectrix/docs/benchmarks/)

<br>

## ◈ API Reference

### Easy API (Recommended)

| Function | Description |
|:---------|:------------|
| `forecast(data, steps=30)` | Auto model selection forecasting |
| `analyze(data)` | DNA profiling, changepoints, anomalies |
| `regress(y, X)` / `regress(data=df, formula="y ~ x")` | Regression with diagnostics |
| `compare(data, steps=30)` | All model comparison (DataFrame) |
| `quick_report(data, steps=30)` | Combined analysis + forecast |

### Classic API

| Method | Description |
|:-------|:------------|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | Full pipeline |
| `Vectrix().analyze(df, dateCol, valueCol)` | Data analysis |

### Return Objects

| Object | Key Attributes |
|:-------|:--------------|
| `EasyForecastResult` | `.predictions` `.dates` `.lower` `.upper` `.model` `.mape` `.rmse` `.models` `.compare()` `.all_forecasts()` `.plot()` `.to_csv()` `.to_json()` |
| `EasyAnalysisResult` | `.dna` `.changepoints` `.anomalies` `.features` `.summary()` |
| `EasyRegressionResult` | `.coefficients` `.pvalues` `.r_squared` `.f_stat` `.summary()` `.diagnose()` |

<br>

## ◈ Architecture

```
vectrix/
├── easy.py               forecast(), analyze(), regress()
├── vectrix.py             Vectrix class — full pipeline
├── types.py               ForecastResult, DataCharacteristics
├── engine/                Forecasting models
│   ├── ets.py               AutoETS (30 combinations)
│   ├── arima.py             AutoARIMA (AICc stepwise)
│   ├── theta.py             Theta method
│   ├── dot.py               Dynamic Optimized Theta
│   ├── ces.py               Complex Exponential Smoothing
│   ├── tbats.py             TBATS / AutoTBATS
│   ├── mstl.py              Multi-Seasonal Decomposition
│   ├── garch.py             GARCH / EGARCH / GJR-GARCH
│   ├── croston.py           Croston Classic / SBA / TSB
│   ├── fourTheta.py         4Theta (M4 Competition method)
│   ├── dtsf.py              Dynamic Time Scan Forecaster
│   ├── esn.py               Echo State Network
│   ├── logistic.py          Logistic Growth
│   ├── hawkes.py            Hawkes Intermittent Demand
│   ├── lotkaVolterra.py     Lotka-Volterra Ensemble
│   ├── phaseTransition.py   Phase Transition Forecaster
│   ├── adversarial.py       Adversarial Stress Tester
│   ├── entropic.py          Entropic Confidence Scorer
│   └── turbo.py             Numba JIT acceleration
├── adaptive/              Regime, self-healing, constraints, DNA
├── regression/            OLS, Ridge, Lasso, Huber, Quantile
├── business/              Anomaly, backtest, what-if, metrics
├── flat_defense/          4-level flat prediction prevention
├── hierarchy/             Bottom-up, top-down, MinTrace
├── intervals/             Conformal + bootstrap intervals
├── ml/                    LightGBM, XGBoost wrappers
├── global_model/          Cross-series forecasting
└── datasets.py            7 built-in sample datasets

rust/                         Built-in Rust engine (25 accelerated functions)
└── src/lib.rs             ETS, ARIMA, DOT, CES, GARCH, DTSF, ESN, 4Theta (PyO3)
```

<br>

## ◈ AI Integration

Vectrix is designed to be fully accessible to AI assistants. Whether you're using Claude, GPT, Copilot, or any other AI tool, Vectrix provides structured context files that allow any AI to understand the complete API in a single read.

### llms.txt — AI-Readable Documentation

The [`llms.txt`](https://llmstxt.org/) standard provides AI assistants with a structured overview of the project, and `llms-full.txt` contains the complete API reference with every function signature, parameter, return type, and common usage pattern.

| File | URL | Contents |
|:-----|:----|:---------|
| `llms.txt` | [eddmpython.github.io/vectrix/llms.txt](https://eddmpython.github.io/vectrix/llms.txt) | Project overview + documentation links |
| `llms-full.txt` | [eddmpython.github.io/vectrix/llms-full.txt](https://eddmpython.github.io/vectrix/llms-full.txt) | Complete API reference — every class, method, parameter, gotcha |

Point your AI assistant to `llms-full.txt` for instant, session-independent understanding of the entire library. No context loss between sessions.

### MCP Server — Tool Use for AI Assistants

The [Model Context Protocol](https://modelcontextprotocol.io/) server exposes Vectrix as callable tools for Claude Desktop, Claude Code, and other MCP-compatible AI assistants.

**10 tools**: `forecast_timeseries`, `forecast_csv`, `analyze_timeseries`, `compare_models`, `run_regression`, `detect_anomalies`, `backtest_model`, `list_sample_datasets`, `load_sample_dataset`

```bash
# Setup with Claude Code
pip install "vectrix[mcp]"
claude mcp add --transport stdio vectrix -- uv run python mcp/server.py

# Setup with Claude Desktop (add to claude_desktop_config.json)
{
    "mcpServers": {
        "vectrix": {
            "command": "uv",
            "args": ["run", "python", "/path/to/mcp/server.py"]
        }
    }
}
```

Once connected, ask your AI: *"Forecast the next 12 months of this sales data"* — the AI calls Vectrix directly.

### Claude Code Skills

Three specialized skills for Claude Code users:

| Skill | Command | Description |
|:------|:--------|:------------|
| `vectrix-forecast` | `/vectrix-forecast` | Time series forecasting workflow |
| `vectrix-analyze` | `/vectrix-analyze` | DNA profiling and anomaly detection |
| `vectrix-regress` | `/vectrix-regress` | R-style regression with diagnostics |

Skills are auto-loaded when working in the Vectrix project directory.

<br>

## ◈ Contributing

```bash
git clone https://github.com/eddmpython/vectrix.git
cd vectrix
uv sync --extra dev
uv run pytest
```

<br>

## ◈ Support

If Vectrix is useful to you, consider supporting the project:

<a href="https://buymeacoffee.com/eddmpython">
  <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support%20Vectrix-f59e0b?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white&labelColor=0f172a" alt="Buy Me a Coffee">
</a>

<br><br>

## ◈ License

[MIT](LICENSE) — Use freely in personal and commercial projects.

<br>

<div align="center">

*Mapping the unknown dimensions of your data.*

</div>
