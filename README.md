<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/hero.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/assets/hero.svg">
  <img alt="Vectrix — Navigate the Vector Space of Time" src=".github/assets/hero.svg" width="100%">
</picture>

<br>

<h3>Pure Python Time Series Forecasting Engine</h3>

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
<a href="#-quick-start">Quick Start</a> ·
<a href="#-models">Models</a> ·
<a href="#-installation">Installation</a> ·
<a href="#-usage">Usage</a> ·
<a href="#-benchmarks">Benchmarks</a> ·
<a href="#-api-reference">API Reference</a> ·
<a href="README_KR.md">한국어</a>
</p>

</div>

<br>

## ◈ What is Vectrix?

Vectrix is a time series forecasting library that runs on **3 dependencies** (NumPy, SciPy, Pandas) with **no compiled extensions**. No C compiler, no cmdstan, no system packages — `pip install` and it works.

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

### Everything is Pure Python

All of the above — forecasting models, regime detection, regression diagnostics, constraint enforcement, hierarchical reconciliation — is implemented in pure Python with only NumPy, SciPy, and Pandas. No compiled extensions, no system dependencies.

<br>

## ◈ Quick Start

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

## ◈ Why Vectrix?

| | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| **Pure Python (no C/Fortran)** | ✅ | ❌ (numba) | ❌ (cmdstan) | ❌ (torch) |
| **Dependencies** | 3 | 5+ | 10+ | 20+ |
| **Auto model selection** | ✅ | ✅ | ❌ | ❌ |
| **Flat-line defense** | ✅ | ❌ | ❌ | ❌ |
| **Business constraints** | 8 types | ❌ | ❌ | ❌ |
| **Built-in regression** | R-style | ❌ | ❌ | ❌ |

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
pip install vectrix                # Core (numpy + scipy + pandas)
pip install "vectrix[numba]"       # + Numba JIT (2-5x speedup)
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # Everything
```

<br>

## ◈ Usage

### Easy API

```python
from vectrix import forecast, analyze, regress

result = forecast([100, 120, 115, 130, 125, 140], steps=5)

report = analyze(df, date="date", value="sales")
print(f"Difficulty: {report.dna.difficulty}")

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
| Quarterly | **0.824** |
| Monthly | **0.756** |
| Other | **0.820** |

**M4 Competition** — 4/6 frequencies beat Naive2:

| Frequency | OWA |
|:----------|:---:|
| Yearly | **0.974** |
| Quarterly | **0.800** |
| Monthly | **0.989** |
| Weekly | **0.737** |
| Daily | 1.210 |
| Hourly | 1.007 |

Full results with sMAPE/MASE breakdown: [benchmarks](docs/benchmarks.md)

<br>

## ◈ API Reference

### Easy API (Recommended)

| Function | Description |
|:---------|:------------|
| `forecast(data, steps=30)` | Auto model selection forecasting |
| `analyze(data)` | DNA profiling, changepoints, anomalies |
| `regress(y, X)` / `regress(data=df, formula="y ~ x")` | Regression with diagnostics |
| `quick_report(data, steps=30)` | Combined analysis + forecast |

### Classic API

| Method | Description |
|:-------|:------------|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | Full pipeline |
| `Vectrix().analyze(df, dateCol, valueCol)` | Data analysis |

### Return Objects

| Object | Key Attributes |
|:-------|:--------------|
| `EasyForecastResult` | `.predictions` `.dates` `.lower` `.upper` `.model` `.plot()` `.to_csv()` `.to_json()` |
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
└── global_model/          Cross-series forecasting
```

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
