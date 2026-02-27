<div align="center">

<br>

# Vectrix

**Feed data. Get forecasts. Zero config.**

Pure Python time series forecasting -- 30+ models, zero heavy dependencies.

<br>

[![PyPI](https://img.shields.io/pypi/v/vectrix?style=flat-square&color=6366f1&label=PyPI)](https://pypi.org/project/vectrix/)
[![Python](https://img.shields.io/pypi/pyversions/vectrix?style=flat-square&label=Python)](https://pypi.org/project/vectrix/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-275%20passed-brightgreen?style=flat-square)]()
[![Sponsor](https://img.shields.io/badge/Sponsor-Buy%20Me%20a%20Coffee-orange?style=flat-square&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/eddmpython)

---

[Installation](#installation) · [Quick Start](#quick-start) · [Features](#features) · [API](#api-reference) · [한국어](README_KR.md)

</div>

<br>

```
  3 dependencies    30+ models    1 line of code
  ─────────────     ──────────    ──────────────
  numpy             AutoETS       from vectrix import forecast
  scipy             AutoARIMA     result = forecast(data, steps=12)
  pandas            Theta/DOT     print(result)
                    TBATS
                    GARCH
                    ...
```

<br>

## Quick Start

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

One call. Auto model selection, flat-line prevention, confidence intervals, and a plot.

<br>

## Why Vectrix?

| | Vectrix | statsforecast | Prophet | Darts |
|:--|:-:|:-:|:-:|:-:|
| **Zero-config auto-forecast** | **Yes** | Yes | -- | -- |
| **Pure Python (no heavy deps)** | **Yes** | -- | -- | -- |
| **30+ models built-in** | **Yes** | Yes | -- | Yes |
| **Flat prediction defense** | **Yes** | -- | -- | -- |
| **Adversarial stress testing** | **Yes** | -- | -- | -- |
| **Forecast DNA fingerprinting** | **Yes** | -- | -- | -- |
| **Business constraints (8 types)** | **Yes** | -- | -- | -- |
| **R-style regression** | **Yes** | -- | -- | -- |

> `numpy` + `scipy` + `pandas` -- that's the entire install.

<br>

## Features

<details open>
<summary><b>Core Models</b></summary>

| Model | Description |
|-------|-------------|
| AutoETS | 30 Error x Trend x Seasonal combinations, AICc selection |
| AutoARIMA | Seasonal ARIMA, stepwise order selection |
| Theta / DOT | Original Theta + Dynamic Optimized Theta |
| AutoCES | Complex Exponential Smoothing (Svetunkov 2023) |
| AutoTBATS | Trigonometric multi-seasonal decomposition |
| GARCH | GARCH, EGARCH, GJR-GARCH volatility models |
| Croston | Classic, SBA, TSB for intermittent demand |
| Logistic Growth | Saturating trends with capacity constraints |
| AutoMSTL | Multi-seasonal decomposition + ARIMA residuals |
| Baselines | Naive, Seasonal Naive, Mean, Drift, Window Average |

</details>

<details>
<summary><b>Novel Methods</b></summary>

| Method | Description |
|--------|-------------|
| Lotka-Volterra Ensemble | Ecological competition dynamics for model weighting |
| Phase Transition | Critical slowing down for regime shift prediction |
| Adversarial Stress | 5 perturbation operators for robustness analysis |
| Hawkes Demand | Self-exciting point process for clustered demand |
| Entropic Confidence | Shannon entropy uncertainty quantification |

</details>

<details>
<summary><b>Adaptive Intelligence</b></summary>

| Feature | Description |
|---------|-------------|
| Regime Detection | Pure numpy HMM (Baum-Welch + Viterbi) |
| Self-Healing | CUSUM + EWMA drift detection, conformal correction |
| Constraints | 8 types: non-negative, range, capacity, YoY, sum, monotone, ratio, custom |
| Forecast DNA | 65+ feature fingerprinting, meta-learning recommendation |
| Flat Defense | 4-level prevention system |

</details>

<details>
<summary><b>Regression & Diagnostics</b></summary>

| Feature | Description |
|---------|-------------|
| Methods | OLS, Ridge, Lasso, Huber, Quantile |
| Formula | R-style `regress(data=df, formula="y ~ x1 + x2")` |
| Diagnostics | Durbin-Watson, Breusch-Pagan, VIF, normality |
| Variable Selection | Stepwise, regularization CV, best subset |
| Time Series | Newey-West, Cochrane-Orcutt, Granger causality |

</details>

<details>
<summary><b>Business Intelligence</b></summary>

| Feature | Description |
|---------|-------------|
| Anomaly Detection | Automated outlier identification and explanation |
| What-if Analysis | Scenario-based forecast simulation |
| Backtesting | Rolling origin cross-validation |
| Hierarchy | Bottom-up, top-down, MinTrace reconciliation |
| Intervals | Conformal + bootstrap prediction intervals |

</details>

<br>

## Installation

```bash
pip install vectrix                # Core (numpy + scipy + pandas)
pip install "vectrix[numba]"       # + Numba JIT (2-5x speedup)
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # Everything
```

**Requirements:** Python 3.10+

<br>

## Usage Examples

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

### Classic API

```python
from vectrix import Vectrix

fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="sales", steps=30)

if result.success:
    print(f"Model: {result.bestModelName}")
    print(f"Predictions: {result.predictions}")
```

<br>

## API Reference

### Easy API (Recommended)

| Function | Description |
|----------|-------------|
| `forecast(data, steps=30)` | Auto model selection forecasting |
| `analyze(data)` | DNA profiling, changepoints, anomalies |
| `regress(y, X)` / `regress(data=df, formula="y ~ x")` | Regression with diagnostics |
| `quick_report(data, steps=30)` | Combined analysis + forecast |

### Classic API

| Method | Description |
|--------|-------------|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | Full pipeline |
| `Vectrix().analyze(df, dateCol, valueCol)` | Data analysis |

### Return Objects

| Object | Key Attributes |
|--------|---------------|
| `EasyForecastResult` | `.predictions` `.dates` `.lower` `.upper` `.model` `.plot()` `.to_csv()` `.to_json()` |
| `EasyAnalysisResult` | `.dna` `.changepoints` `.anomalies` `.features` `.summary()` |
| `EasyRegressionResult` | `.coefficients` `.pvalues` `.r_squared` `.f_stat` `.summary()` `.diagnose()` |

<br>

## Architecture

```
vectrix/
├── easy.py               forecast(), analyze(), regress()
├── vectrix.py             Vectrix class (full pipeline)
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

## Contributing

```bash
git clone https://github.com/eddmpython/vectrix.git
cd vectrix
uv sync --extra dev
uv run pytest
```

<br>

## Support

If you find Vectrix useful, consider supporting the project:

<a href="https://buymeacoffee.com/eddmpython">
  <img src="https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support%20Vectrix-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white" alt="Buy Me a Coffee">
</a>

<br>

## License

[MIT](LICENSE) -- Use freely in personal and commercial projects.
