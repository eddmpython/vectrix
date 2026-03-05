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
<img src="https://img.shields.io/badge/Tests-603%20passed-10b981?style=for-the-badge&labelColor=0f172a&logo=pytest&logoColor=white" alt="Tests">
</p>

<br>

<p>
<a href="https://eddmpython.github.io/vectrix/"><img src="https://img.shields.io/badge/Docs-eddmpython.github.io/vectrix-818cf8?style=for-the-badge&labelColor=0f172a&logo=readthedocs&logoColor=white" alt="Documentation"></a>
<a href="https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/01_quickstart.ipynb"><img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&labelColor=0f172a&logo=googlecolab&logoColor=white" alt="Open in Colab"></a>
</p>

<p>
<a href="https://eddmpython.github.io/vectrix/">Documentation</a> ·
<a href="#-quick-start">Quick Start</a> ·
<a href="#-models">Models</a> ·
<a href="#-benchmarks">Benchmarks</a> ·
<a href="#-research--understanding-first-forecasting">Research</a> ·
<a href="#-api-reference">API Reference</a> ·
<a href="#-interactive-notebooks">Notebooks</a>
</p>

</div>

<br>

## ◈ What is Vectrix?

Vectrix is a time series forecasting engine that **understands your data before predicting it**. Every series is profiled into a DNA fingerprint — 65+ statistical features — that drives model selection, ensemble strategy, and anomaly detection automatically. Built-in Rust acceleration, 3 dependencies (NumPy, SciPy, Pandas), no compiler needed. `pip install vectrix` and the Rust engine is included in the wheel.

Under the hood, there is an active research program: [Understanding-first Forecasting](#-research--understanding-first-forecasting) — proving that structural understanding beats pattern memorization, even against foundation models.

### Forecasting

Pass a list, DataFrame, or CSV path to `forecast()`. Vectrix runs multiple models (ETS, ARIMA, Theta, TBATS, CES, MSTL), evaluates each with cross-validation, and returns the best prediction with confidence intervals. You don't choose a model — it does.

```python
from vectrix import forecast
result = forecast("sales.csv", steps=12)
```

### Flat-Line Defense

A common failure mode in automated forecasting is flat predictions — the model outputs a constant line. Vectrix includes a detection and correction system that identifies flat outputs and falls back to a model that captures the signal. This is a heuristic defense layer — it reduces flat predictions significantly but is not a guarantee.

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

Every `pip install vectrix` includes a pre-built Rust extension — like Polars, no compiler needed. 29 core hot loops are Rust-accelerated across all forecasting engines.

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

### Interactive Visualization

Publication-quality dark-themed Plotly charts and a self-contained HTML dashboard, built into the library as an optional dependency.

```python
pip install vectrix[viz]
```

```python
from vectrix import forecast, analyze, compare, loadSample
from vectrix.viz import forecastChart, dnaRadar, dashboard

df = loadSample("airline")
result = forecast(df, steps=12)
analysis = analyze(df)
comparison = compare(df, steps=12)

forecastChart(result, historical=df).show()
dnaRadar(analysis).show()
```

9 chart functions — `forecastChart`, `dnaRadar`, `modelHeatmap`, `scenarioChart`, `backtestChart`, `metricsCard`, `forecastReport`, `analysisReport`, `dashboard` — all return standard `go.Figure` objects with a consistent brand theme (dark navy background, cyan-purple gradient). `dashboard()` generates a self-contained HTML report.

### HTML Dashboard

Generate a complete interactive report — data profile, forecast results, model comparison, and charts — in a single self-contained HTML file. No server needed.

```python
from vectrix import forecast, analyze, compare, loadSample
from vectrix.viz import dashboard

df = loadSample("airline")

report = dashboard(
    forecast=forecast(df, steps=12),
    analysis=analyze(df),
    comparison=compare(df, steps=12),
    historical=df,
    title="Airline Passengers — Monthly Forecast",
)
report.save("report.html")   # Self-contained HTML (embedded Plotly + CSS)
report.show()                 # Opens in browser or displays inline in Jupyter
```

The report includes: overview KPIs, DNA feature bars with descriptive stats, accuracy metrics with model comparison, and interactive forecast + radar charts. All parameters are optional — pass only what you have.

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
| **HTML dashboard** | Self-contained | ❌ | ❌ | ❌ |
| **Interactive viz** | 9 Plotly charts | ❌ | ❌ | ❌ |

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
pip install "vectrix[viz]"         # + Plotly charts & HTML dashboard
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # Everything
```

<br>

## ◈ Usage

### Easy API

```python
from vectrix import forecast, analyze, regress, compare

# Level 1 — Zero Config
result = forecast([100, 120, 115, 130, 125, 140], steps=5)

# Level 2 — Guided Control
result = forecast(df, date="date", value="sales", steps=12,
                  models=["dot", "auto_ets", "auto_ces"],
                  ensemble="mean",
                  confidence=0.90)

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

Evaluated on **M4 Competition 100,000 time series** (2,000 sample per frequency, seed=42). OWA < 1.0 means better than Naive2.

**DOT-Hybrid** (single model, OWA 0.848 — beats M4 #2 FFORMA 0.838):

| Frequency | OWA | vs Naive2 |
|:----------|:---:|:---------:|
| Yearly | **0.797** | -20.3% |
| Quarterly | **0.894** | -10.6% |
| Monthly | **0.897** | -10.3% |
| Weekly | **0.959** | -4.1% |
| Daily | **0.820** | -18.0% |
| Hourly | **0.722** | -27.8% |

**M4 Competition Leaderboard Context:**

| Rank | Method | OWA |
|:-----|:-------|:---:|
| #1 | ES-RNN (Smyl) | 0.821 |
| #2 | FFORMA | 0.838 |
| — | **Vectrix DOT-Hybrid** | **0.848** |
| #11 | 4Theta | 0.874 |
| #18 | Theta | 0.897 |

Full results with sMAPE/MASE breakdown: [benchmarks](https://eddmpython.github.io/vectrix/docs/benchmarks/)

<br>

## ◈ Interactive Notebooks

Try Vectrix instantly — no setup needed. Click to open in Google Colab.

### Tutorials

| Notebook | What you'll learn | |
|:---------|:-----------------|:-:|
| **01 Quickstart** | Forecast from list, DataFrame, CSV | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/01_quickstart.ipynb) |
| **02 Analysis & DNA** | DNA profiling, changepoints, anomalies | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/02_analyze.ipynb) |
| **03 Regression** | R-style formulas, diagnostics, 5 methods | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/03_regression.ipynb) |
| **04 Models** | Model comparison, direct engine, flat defense | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/04_models.ipynb) |
| **05 Adaptive** | Regime detection, DNA, healing, constraints | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/05_adaptive.ipynb) |
| **06 Business** | Anomalies, scenarios, backtest, metrics | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/06_business.ipynb) |
| **07 Visualization** | Charts, reports, HTML dashboard | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/07_visualization.ipynb) |

### Showcase (Plotly)

| Notebook | What you'll build | |
|:---------|:-----------------|:-:|
| **Sales Dashboard** | Interactive forecast + DNA radar + scenarios | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/showcase/01_sales_forecasting_dashboard.ipynb) |
| **Demand Planning** | Full workflow: quality check → forecast → budget | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/showcase/02_demand_planning_workflow.ipynb) |

<br>

## ◈ API Reference

### Easy API (Recommended)

| Function | Description |
|:---------|:------------|
| `forecast(data, steps, models, ensemble, confidence)` | Auto or guided forecasting |
| `analyze(data, period, features)` | DNA profiling, changepoints, anomalies |
| `regress(y, X)` / `regress(data=df, formula="y ~ x")` | Regression with diagnostics |
| `compare(data, steps, models)` | Model comparison (DataFrame) |
| `quick_report(data, steps)` | Combined analysis + forecast |

All parameters beyond `data` are optional with sensible defaults. See [Progressive Disclosure](#api-layers) for the Level 1 → 2 → 3 design.

### Visualization API (`pip install vectrix[viz]`)

| Function | Description |
|:---------|:------------|
| `forecastChart(result, historical, theme)` | Forecast line chart with confidence intervals |
| `dnaRadar(analysis, theme)` | 6-axis radar of DNA features |
| `modelHeatmap(comparison, top, theme)` | Normalized error metric heatmap |
| `scenarioChart(scenarios, dates, theme)` | What-if scenario comparison |
| `backtestChart(result, metric, theme)` | Fold-by-fold backtest bars |
| `metricsCard(metrics, thresholds, theme)` | Business metrics scorecard |
| `forecastReport(result, historical, theme)` | Forecast + error metrics (2-row composite) |
| `analysisReport(analysis, theme)` | DNA radar + features + difficulty (2x2 composite) |
| `dashboard(forecast, analysis, comparison, historical, title)` | Self-contained HTML report |

All chart functions return `go.Figure` (standard Plotly). `dashboard()` returns a `DashboardResult` with `.show()`, `.save(path)`, and `.html` property.

### Classic API

| Method | Description |
|:-------|:------------|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | Full pipeline |
| `Vectrix().analyze(df, dateCol, valueCol)` | Data analysis |

### Return Objects

| Object | Key Attributes |
|:-------|:--------------|
| `EasyForecastResult` | `.predictions` `.dates` `.lower` `.upper` `.model` `.mape` `.rmse` `.models` `.compare()` `.all_forecasts()` `.plot()` `.to_csv()` `.to_json()` `.toDataframe()` |
| `EasyAnalysisResult` | `.dna` `.changepoints` `.anomalies` `.features` `.summary()` |
| `EasyRegressionResult` | `.coefficients` `.pvalues` `.r_squared` `.f_stat` `.summary()` `.diagnose()` |
| `DashboardResult` | `.html` `.show()` `.save(path)` |

<br>

## ◈ Architecture

```
vectrix/
├── easy.py               forecast(), analyze(), regress(), compare()
├── vectrix.py             Vectrix class — full pipeline orchestrator
├── types.py               ForecastResult, DataCharacteristics
├── engine/                Forecasting models (21 registered)
│   ├── registry.py          Model registry — Single Source of Truth
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
├── viz/                   Interactive visualization (Plotly)
│   ├── theme.py             Brand colors, layout, applyTheme()
│   ├── charts.py            6 individual chart functions
│   ├── report.py            Composite reports (forecastReport, analysisReport)
│   └── dashboard.py         Self-contained HTML dashboard generator
├── ml/                    LightGBM, XGBoost wrappers
├── global_model/          Cross-series forecasting
└── datasets.py            7 built-in sample datasets

rust/                         Built-in Rust engine (29 accelerated functions)
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

## ◈ Philosophy

### Identity

Vectrix is a **zero-config forecasting engine with built-in Rust acceleration**. The design philosophy:

- **Python syntax, Rust speed** — Like Polars, the Rust engine is invisible. Users write Python; hot loops run in Rust automatically.
- **Progressive disclosure** — Beginners call `forecast(data, steps=12)` with zero configuration. Experts pass `models=`, `ensemble=`, `confidence=` to control every aspect. Engine-level access (`AutoETS`, `AutoARIMA`) is always available for full control.
- **3 dependencies, no compiler** — NumPy, SciPy, Pandas. No system packages, no Numba JIT warmup, no CmdStan. `pip install vectrix` and you're done.
- **Correctness over features** — We'd rather have 15 models that beat Naive2 on every frequency than 50 models that fail on Daily and Hourly.

### API Layers

| Layer | Target | Example |
|:------|:-------|:--------|
| **Level 1 — Zero Config** | Beginners, quick prototypes | `forecast(data, steps=12)` |
| **Level 2 — Guided Control** | Data scientists, production | `forecast(data, steps=12, models=["dot", "auto_ets"], ensemble="mean", confidence=0.90)` |
| **Level 3 — Engine Direct** | Researchers, custom pipelines | `AutoETS(period=7).fit(data).predict(30)` |

Every parameter at Level 2 has a sensible default that reproduces Level 1 behavior. No parameter is ever required.

<br>

## ◈ Research — Understanding-first Forecasting

Vectrix is a production forecasting library today. But beneath the surface, there is an active research program aimed at **breaking the ceiling of statistical forecasting**.

### The Problem

Foundation models (Chronos-2, TimesFM, Moirai) learn to predict by memorizing patterns from billions of data points. They're powerful, but they **don't understand the data** — they match patterns. When the structure shifts, they hallucinate.

Statistical models understand structure (trend, seasonality, error decomposition), but they **can't learn from experience** — each series is forecasted in isolation.

### Our Thesis — Understand First, Then Act

```
Foundation models:  data → [giant neural net] → prediction     (pattern memorization)
Vectrix approach:   data → [understand] → [decide] → prediction (structural reasoning)
```

Vectrix profiles every time series into a **DNA fingerprint** — 65+ statistical features that capture the structural essence of the data. This fingerprint drives every downstream decision: which models to run, how to blend them, where regime shifts occur.

The research question is: **can a system that understands data structure outperform a system that memorizes data patterns?**

### What We've Proven So Far

Experiments on [GIFT-Eval](https://huggingface.co/datasets/Salesforce/gift_eval) (144K+ series, 7 domains, 10 frequencies):

| Finding | Evidence |
|:--------|:---------|
| DNA fingerprints contain real structural information | 65 features classify 7 domains at **82.6%** accuracy (vs 14.3% random) |
| Structure predicts model performance | DNA features explain **27.3%** of MASE variance (linear only) |
| Learned model selection beats any single model | GBT selector achieves **+5.5%** over best single model, capturing 31.3% of Oracle gap. Domain-optimal routing pushes to **+7.7%** (43.4% of Oracle) |
| Selection > Blending | Choosing the right model beats mixing all models — bad model contamination is real |
| Statistical models win on annual/quarterly frequencies | Foundation models dominate high-frequency, but low-frequency is **contested territory** |

### Research Roadmap

| Phase | Goal | Status |
|:------|:-----|:-------|
| ~~Phase 0~~ | Baseline — DOT-Hybrid on GIFT-Eval, foundation model comparison | Done |
| ~~Phase 1~~ | Learned Profiling — DNA feature augmentation, domain classification | Done |
| ~~Phase 2~~ | Learned Selection — DNA → optimal model mapping via meta-learning | Done |
| Phase 3 | Learned Surgery — residual correction at regime boundaries | Planned |
| Phase 4 | Integration — unified pipeline (profile → select → correct → predict) | Planned |
| Phase 5 | Domain-specific defeat of foundation models | Planned |

### Where This Goes

If DNA representation quality improves — from handcrafted features to learned representations — the same principle scales:

- **Better DNA** → better model selection → better accuracy than brute-force memorization
- **CPU milliseconds** vs GPU inference — 100x+ speed advantage at comparable accuracy
- **Explainability** — every decision traces back to structural features, not a black box

The goal is not to build another foundation model. The goal is to prove that **understanding data structure is a more efficient path to accurate forecasting than memorizing data patterns**.

### Roadmap

| Priority | Area | Current | Target | Status |
|:---------|:-----|:--------|:-------|:-------|
| **P0** | M4 Accuracy | OWA 0.848 | OWA < 0.821 | In progress |
| **P1** | Beat Foundation Models | Phase 2 done, Phase 3 next | Win 3+ GIFT-Eval domains | In progress |
| **P2** | Pipeline Speed | 48ms forecast() | < 10ms | Planned |
| **P3** | Interactive Playground | — | GitHub Pages live demo | Planned |
| **P4** | Community Growth | Blog (5 posts) | Reddit, Kaggle, HN | In progress |

### Principles

1. **Accuracy first, speed second** — A wrong answer delivered fast is still wrong. Improve M4 OWA before optimizing latency.
2. **Never break zero-config** — Every new parameter must have a default. `forecast(data, steps=12)` must always work.
3. **Benchmark-driven** — Every engine change is validated against M4 100K series. No "it seems better" — show the OWA.
4. **Understanding over memorization** — Invest in DNA quality, not model count.
5. **Minimal dependencies** — Adding a dependency requires strong justification. If it can be implemented in numpy/scipy, it should be.

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
