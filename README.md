<div align="center">

<br>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Vectrix-Time%20Series%20Forecasting-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwb2x5bGluZSBwb2ludHM9IjIyIDEyIDE4IDEyIDE1IDE5IDkgNSA2IDEyIDIgMTIiLz48L3N2Zz4=">
  <img alt="Vectrix" src="https://img.shields.io/badge/Vectrix-Time%20Series%20Forecasting-6366f1?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiPjxwb2x5bGluZSBwb2ludHM9IjIyIDEyIDE4IDEyIDE1IDE5IDkgNSA2IDEyIDIgMTIiLz48L3N2Zz4=">
</picture>

### Feed data. Get forecasts. Zero config.

Pure Python time series forecasting -- 30+ models, zero heavy dependencies.

<br>

[![PyPI](https://img.shields.io/pypi/v/vectrix?style=flat-square&color=6366f1)](https://pypi.org/project/vectrix/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/vectrix?style=flat-square)](https://pypi.org/project/vectrix/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-275%20passed-brightgreen?style=flat-square)]()
[![Buy Me a Coffee](https://img.shields.io/badge/Sponsor-Buy%20Me%20a%20Coffee-orange?style=flat-square&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/eddmpython)

[Installation](#installation) &middot; [Quick Start](#30-second-quick-start) &middot; [Features](#features) &middot; [API Reference](#api-reference) &middot; [Architecture](#architecture) &middot; [한국어](README_KR.md)

</div>

<br>

> **3 dependencies. 30+ models. 1 line of code.**
>
> Vectrix is a self-contained time series forecasting library built from scratch with pure NumPy + SciPy.
> No statsforecast, no statsmodels, no Prophet -- just feed your data and get optimal predictions with confidence intervals.

<br>

## 30-Second Quick Start

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

That's it. One call gets you automatic model selection, flat-line prevention, confidence intervals, and a plot.

## Why Vectrix?

| Feature | Vectrix | statsforecast | Prophet | Darts |
|---------|:---------:|:-------------:|:-------:|:-----:|
| Zero-config auto-forecast | **Yes** | Yes | -- | -- |
| Pure Python (no heavy deps) | **Yes** | -- | -- | -- |
| 30+ models built-in | **Yes** | Yes | -- | Yes |
| Flat prediction defense | **Yes** | -- | -- | -- |
| Adversarial stress testing | **Yes** | -- | -- | -- |
| Forecast DNA fingerprinting | **Yes** | -- | -- | -- |
| Business constraints (8 types) | **Yes** | -- | -- | -- |
| R-style regression (`y ~ x`) | **Yes** | -- | -- | -- |
| Korean market native | **Yes** | -- | -- | -- |

**Three dependencies.** `numpy`, `scipy`, `pandas`. That's the entire install.

## Features

<details>
<summary><b>Core Forecasting Models</b></summary>

<br>

- **AutoETS** -- 30 Error x Trend x Seasonal combinations with AICc selection (Hyndman-Khandakar)
- **AutoARIMA** -- Seasonal ARIMA with stepwise order selection via AICc
- **Theta / DOT** -- Original Theta + Dynamic Optimized Theta (M3 Competition winner)
- **AutoCES** -- Complex Exponential Smoothing (Svetunkov 2023)
- **AutoTBATS** -- Trigonometric seasonality for complex multi-seasonal data
- **GARCH** -- GARCH, EGARCH, GJR-GARCH for volatility modeling
- **Croston** -- Classic, SBA, TSB + AutoCroston for intermittent demand
- **Logistic Growth** -- Prophet-style saturating trends with capacity constraints
- **AutoMSTL** -- Multi-seasonal decomposition + ARIMA residual forecasting
- **Baselines** -- Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

</details>

<details>
<summary><b>Novel Methods (World First)</b></summary>

<br>

- **Lotka-Volterra Ensemble** -- Ecological competition dynamics for adaptive model weighting
- **Phase Transition Forecaster** -- Critical slowing down detection for regime shift prediction
- **Adversarial Stress Tester** -- 5 perturbation operators for forecast robustness analysis
- **Hawkes Intermittent Demand** -- Self-exciting point process for clustered demand patterns
- **Entropic Confidence Scorer** -- Shannon entropy-based forecast uncertainty quantification

</details>

<details>
<summary><b>Adaptive Intelligence</b></summary>

<br>

- **Regime Detection** -- Pure numpy Hidden Markov Model (Baum-Welch + Viterbi)
- **Self-Healing Forecast** -- CUSUM + EWMA drift detection with conformal prediction correction
- **Constraint-Aware Forecasting** -- 8 business constraints: non-negative, range, capacity, YoY change, sum, monotone, ratio, custom
- **Forecast DNA** -- 65+ feature fingerprinting with meta-learning model recommendation and similarity search
- **Flat Defense** -- 4-level system (diagnostic, detection, correction, prevention) against flat prediction failure

</details>

<details>
<summary><b>Regression & Diagnostics</b></summary>

<br>

- **5 regression methods** -- OLS, Ridge, Lasso, Huber, Quantile
- **R-style formula interface** -- `regress(data=df, formula="sales ~ ads + price")`
- **Full diagnostics** -- Durbin-Watson, Breusch-Pagan, VIF, normality tests
- **Variable selection** -- Stepwise, regularization CV, best subset
- **Time series regression** -- Newey-West, Cochrane-Orcutt, Prais-Winsten, Granger causality

</details>

<details>
<summary><b>Business Intelligence</b></summary>

<br>

- **Anomaly detection** -- Automated outlier identification and explanation
- **What-if analysis** -- Scenario-based forecast simulation
- **Backtesting** -- Rolling origin cross-validation with multiple metrics
- **Hierarchy reconciliation** -- Bottom-up, top-down, MinTrace optimal reconciliation
- **Prediction intervals** -- Conformal + bootstrap methods

</details>

## Installation

```bash
pip install vectrix
```

Optional dependencies:

```bash
pip install "vectrix[numba]"    # Numba JIT for 2-5x speedup
pip install "vectrix[ml]"      # LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"     # Everything
```

**Requirements:** Python 3.10+, NumPy >= 1.24, Pandas >= 2.0, SciPy >= 1.10

## Usage Examples

### Basic: 2-Line Forecast

```python
from vectrix import forecast

result = forecast([100, 120, 115, 130, 125, 140, 135, 150], steps=5)
print(result)
```

Pass a list, NumPy array, Pandas Series, DataFrame, dict, or a CSV file path. Vectrix auto-detects everything.

### Intermediate: DataFrame + Analysis

```python
from vectrix import forecast, analyze
import pandas as pd

df = pd.read_csv("data.csv")

report = analyze(df, date="date", value="sales")
print(report.summary())
print(f"Difficulty: {report.dna.difficulty}")
print(f"Recommended: {report.dna.recommendedModels}")

result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

### Advanced: R-Style Regression

```python
from vectrix import regress

model = regress(data=df, formula="sales ~ temperature + promotion + holiday")
print(model.summary())
print(model.diagnose())
```

### Expert: Direct Engine Access

```python
from vectrix.engine import AutoETS, AutoARIMA, AutoTBATS
from vectrix.adaptive import RegimeDetector, ForecastDNA

ets = AutoETS(period=7)
ets.fit(data)
pred, lower, upper = ets.predict(30)

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"Fingerprint: {profile.fingerprint}")
print(f"Difficulty: {profile.difficulty}")
print(f"Recommended: {profile.recommendedModels}")
```

### Stress Testing

```python
from vectrix.engine import AdversarialStressTester

tester = AdversarialStressTester(nPerturbations=50)
result = tester.analyze(data, steps=12)
print(f"Fragility: {result.fragilityScore:.2f}")
print(f"Grade: {result.summary()['grade']}")
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
    Constraint('monotone', {'direction': 'increasing'}),
])
```

### Full Pipeline (Classic API)

```python
from vectrix import Vectrix

fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="sales", steps=30)

if result.success:
    print(f"Model: {result.bestModelName}")
    print(f"Predictions: {result.predictions}")
    print(f"95% CI: [{result.lower95}, {result.upper95}]")
```

## API Reference

### Easy API (Recommended)

| Function | Description |
|----------|-------------|
| `forecast(data, steps=30)` | One-call forecasting with auto model selection |
| `analyze(data)` | Time series DNA profiling, changepoints, anomalies |
| `regress(y, X)` or `regress(data=df, formula="y ~ x")` | Regression with full diagnostics |
| `quick_report(data, steps=30)` | Combined analysis + forecast report |

### Classic API

| Method | Description |
|--------|-------------|
| `Vectrix().forecast(df, dateCol, valueCol, steps)` | Full pipeline with detailed result object |
| `Vectrix().analyze(df, dateCol, valueCol)` | Data characteristics + flat risk assessment |

### Return Objects

**`EasyForecastResult`** -- `.predictions`, `.dates`, `.lower`, `.upper`, `.model`, `.summary()`, `.plot()`, `.to_csv()`, `.to_dataframe()`, `.to_json()`

**`EasyAnalysisResult`** -- `.dna`, `.changepoints`, `.anomalies`, `.features`, `.characteristics`, `.summary()`

**`EasyRegressionResult`** -- `.coefficients`, `.pvalues`, `.r_squared`, `.adj_r_squared`, `.f_stat`, `.summary()`, `.diagnose()`

## Architecture

```
vectrix/
├── easy.py              # Zero-config API: forecast(), analyze(), regress()
├── vectrix.py         # Full pipeline: Vectrix class
├── types.py             # ForecastResult, DataCharacteristics, ModelResult
├── engine/              # 30+ statistical models
│   ├── ets.py           #   AutoETS (30 combinations)
│   ├── arima.py         #   AutoARIMA (AICc stepwise)
│   ├── theta.py         #   Theta method
│   ├── dot.py           #   Dynamic Optimized Theta
│   ├── ces.py           #   Complex Exponential Smoothing
│   ├── tbats.py         #   TBATS / AutoTBATS
│   ├── mstl.py          #   Multi-Seasonal Decomposition
│   ├── garch.py         #   GARCH / EGARCH / GJR-GARCH
│   ├── croston.py       #   Croston Classic / SBA / TSB
│   ├── logistic.py      #   Logistic Growth / Saturating Trend
│   ├── hawkes.py        #   Hawkes Intermittent Demand
│   ├── lotkaVolterra.py #   Lotka-Volterra Ensemble
│   ├── phaseTransition.py # Phase Transition Forecaster
│   ├── adversarial.py   #   Adversarial Stress Tester
│   ├── entropic.py      #   Entropic Confidence Scorer
│   ├── probabilistic.py #   Probabilistic Forecaster
│   └── turbo.py         #   Numba JIT acceleration
├── adaptive/            # Regime detection, self-healing, constraints, DNA
├── regression/          # 5 regressors + diagnostics + variable selection
├── business/            # Anomaly, backtest, what-if, explanation, metrics
├── flat_defense/        # 4-level flat prediction prevention
├── hierarchy/           # Bottom-up, top-down, MinTrace reconciliation
├── intervals/           # Conformal + bootstrap prediction intervals
├── ml/                  # LightGBM, XGBoost, scikit-learn wrappers
└── global_model/        # Global (cross-series) forecasting
```

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy >= 1.24 | Yes | Core numerical computation |
| pandas >= 2.0 | Yes | Data handling and I/O |
| scipy >= 1.10 | Yes | Optimization and statistical tests |
| numba | Optional | JIT acceleration (2-5x speedup) |
| lightgbm / xgboost | Optional | ML-based forecasting models |
| scikit-learn | Optional | ML utilities |

## Running Tests

```bash
uv run pytest
```

275 tests covering all models, edge cases, and integration scenarios.

## Contributing

Contributions are welcome. Fork, branch, and submit a pull request.

```bash
git clone https://github.com/eddmpython/vectrix.git
cd vectrix
uv sync --extra dev
uv run pytest
```

## Support

If you find Vectrix useful, consider supporting the project:

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Support-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/eddmpython)

## License

[MIT](LICENSE) -- Use freely in personal and commercial projects.
