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
<img src="https://img.shields.io/badge/30+-Models-6366f1?style=for-the-badge&labelColor=0f172a" alt="Models">
<img src="https://img.shields.io/badge/3-Dependencies-818cf8?style=for-the-badge&labelColor=0f172a" alt="Dependencies">
<img src="https://img.shields.io/badge/1-Line%20of%20Code-a78bfa?style=for-the-badge&labelColor=0f172a" alt="One Line">
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
<a href="#-api-reference">API Reference</a> ·
<a href="README_KR.md">한국어</a>
</p>

</div>

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

> One function call. Auto model selection, flat-line defense, confidence intervals, and a plot.

<br>

## ◈ Why Vectrix?

<table>
<tr>
<td>

| Dimension | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| **Zero-config** | ✅ | ✅ | ❌ | ❌ |
| **Pure Python** | ✅ | ❌ | ❌ | ❌ |
| **30+ models** | ✅ | ✅ | ❌ | ✅ |
| **Flat defense** | ✅ | ❌ | ❌ | ❌ |
| **Stress testing** | ✅ | ❌ | ❌ | ❌ |
| **Forecast DNA** | ✅ | ❌ | ❌ | ❌ |
| **Constraints (8)** | ✅ | ❌ | ❌ | ❌ |
| **R-style regress** | ✅ | ❌ | ❌ | ❌ |

</td>
</tr>
</table>

> **Three vectors.** `numpy` · `scipy` · `pandas` — that's the entire orbit.

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
<summary><b>Novel Methods — Uncharted Territory</b></summary>

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
├── engine/                30+ statistical models
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

If Vectrix is useful to you, consider fueling the mission:

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
