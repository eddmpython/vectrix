<div align="center">

# ForecastX

**Feed data. Get forecasts. Zero config.**

Pure Python time series forecasting -- 30+ models, zero heavy dependencies.

[![PyPI](https://img.shields.io/pypi/v/forecastx)](https://pypi.org/project/forecastx/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/forecastx)](https://pypi.org/project/forecastx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-275%20passed-brightgreen)]()

[Installation](#installation) &middot; [Quick Start](#30-second-quick-start) &middot; [Features](#features) &middot; [API Reference](#api-reference) &middot; [Architecture](#architecture) &middot; [한국어](README_KR.md)

</div>

---

ForecastX is a self-contained time series forecasting library built from scratch with pure NumPy + SciPy. No statsforecast, no statsmodels, no Prophet -- just feed your data and get optimal predictions with confidence intervals.

## 30-Second Quick Start

```bash
pip install forecastx
```

```python
from forecastx import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
result.to_csv("output.csv")
```

That's it. One call gets you automatic model selection, flat-line prevention, confidence intervals, and a plot.

## Why ForecastX?

| Feature | ForecastX | statsforecast | Prophet | Darts |
|---------|:---------:|:-------------:|:-------:|:-----:|
| Zero-config auto-forecast | Yes | Yes | -- | -- |
| Pure Python (no heavy deps) | Yes | -- | -- | -- |
| 30+ models built-in | Yes | Yes | -- | Yes |
| Flat prediction defense | Yes | -- | -- | -- |
| Adversarial stress testing | Yes | -- | -- | -- |
| Forecast DNA fingerprinting | Yes | -- | -- | -- |
| Business constraints (8 types) | Yes | -- | -- | -- |
| R-style regression (`y ~ x`) | Yes | -- | -- | -- |
| Korean market native | Yes | -- | -- | -- |

**Three dependencies.** `numpy`, `scipy`, `pandas`. That's the entire install.

## Features

### Core Forecasting Models

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

### Novel Methods (World First)

- **Lotka-Volterra Ensemble** -- Ecological competition dynamics for adaptive model weighting
- **Phase Transition Forecaster** -- Critical slowing down detection for regime shift prediction
- **Adversarial Stress Tester** -- 5 perturbation operators for forecast robustness analysis
- **Hawkes Intermittent Demand** -- Self-exciting point process for clustered demand patterns
- **Entropic Confidence Scorer** -- Shannon entropy-based forecast uncertainty quantification

### Adaptive Intelligence

- **Regime Detection** -- Pure numpy Hidden Markov Model (Baum-Welch + Viterbi)
- **Self-Healing Forecast** -- CUSUM + EWMA drift detection with conformal prediction correction
- **Constraint-Aware Forecasting** -- 8 business constraints: non-negative, range, capacity, YoY change, sum, monotone, ratio, custom
- **Forecast DNA** -- 65+ feature fingerprinting with meta-learning model recommendation and similarity search
- **Flat Defense** -- 4-level system (diagnostic, detection, correction, prevention) against flat prediction failure

### Regression & Diagnostics

- **5 regression methods** -- OLS, Ridge, Lasso, Huber, Quantile
- **R-style formula interface** -- `regress(data=df, formula="sales ~ ads + price")`
- **Full diagnostics** -- Durbin-Watson, Breusch-Pagan, VIF, normality tests
- **Variable selection** -- Stepwise, regularization CV, best subset
- **Time series regression** -- Newey-West, Cochrane-Orcutt, Prais-Winsten, Granger causality

### Business Intelligence

- **Anomaly detection** -- Automated outlier identification and explanation
- **What-if analysis** -- Scenario-based forecast simulation
- **Backtesting** -- Rolling origin cross-validation with multiple metrics
- **Hierarchy reconciliation** -- Bottom-up, top-down, MinTrace optimal reconciliation
- **Prediction intervals** -- Conformal + bootstrap methods

## Installation

```bash
pip install forecastx
```

Optional dependencies:

```bash
pip install "forecastx[numba]"    # Numba JIT for 2-5x speedup
pip install "forecastx[ml]"      # LightGBM, XGBoost, scikit-learn
pip install "forecastx[all]"     # Everything
```

**Requirements:** Python 3.10+, NumPy >= 1.24, Pandas >= 2.0, SciPy >= 1.10

## Usage Examples

### Basic: 2-Line Forecast

```python
from forecastx import forecast

result = forecast([100, 120, 115, 130, 125, 140, 135, 150], steps=5)
print(result)
```

Pass a list, NumPy array, Pandas Series, DataFrame, dict, or a CSV file path. ForecastX auto-detects everything.

### Intermediate: DataFrame + Analysis

```python
from forecastx import forecast, analyze
import pandas as pd

df = pd.read_csv("data.csv")

# Analyze time series characteristics
report = analyze(df, date="date", value="sales")
print(report.summary())
print(f"Difficulty: {report.dna.difficulty}")
print(f"Recommended: {report.dna.recommendedModels}")

# Forecast with auto-detected columns
result = forecast(df, date="date", value="sales", steps=30)
result.plot()
result.to_csv("forecast.csv")
```

### Advanced: R-Style Regression

```python
from forecastx import regress

model = regress(data=df, formula="sales ~ temperature + promotion + holiday")
print(model.summary())    # statsmodels-style output
print(model.diagnose())   # Durbin-Watson, Breusch-Pagan, VIF, normality
```

### Expert: Direct Engine Access

```python
from forecastx.engine import AutoETS, AutoARIMA, AutoTBATS
from forecastx.adaptive import RegimeDetector, ForecastDNA

# ETS with specific configuration
ets = AutoETS(period=7)
ets.fit(data)
pred, lower, upper = ets.predict(30)

# Forecast DNA analysis
dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(f"Fingerprint: {profile.fingerprint}")
print(f"Difficulty: {profile.difficulty}")
print(f"Recommended: {profile.recommendedModels}")
```

### Stress Testing

```python
from forecastx.engine import AdversarialStressTester

tester = AdversarialStressTester(nPerturbations=50)
result = tester.analyze(data, steps=12)
print(f"Fragility: {result.fragilityScore:.2f}")
print(f"Grade: {result.summary()['grade']}")
```

### Business Constraints

```python
from forecastx.adaptive import ConstraintAwareForecaster, Constraint

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
from forecastx import ForecastX

fx = ForecastX(verbose=True)
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
| `ForecastX().forecast(df, dateCol, valueCol, steps)` | Full pipeline with detailed result object |
| `ForecastX().analyze(df, dateCol, valueCol)` | Data characteristics + flat risk assessment |

### Return Objects

**`EasyForecastResult`** -- `.predictions`, `.dates`, `.lower`, `.upper`, `.model`, `.summary()`, `.plot()`, `.to_csv()`, `.to_dataframe()`, `.to_json()`

**`EasyAnalysisResult`** -- `.dna`, `.changepoints`, `.anomalies`, `.features`, `.characteristics`, `.summary()`

**`EasyRegressionResult`** -- `.coefficients`, `.pvalues`, `.r_squared`, `.adj_r_squared`, `.f_stat`, `.summary()`, `.diagnose()`

## Architecture

```
forecastx/
├── easy.py              # Zero-config API: forecast(), analyze(), regress()
├── forecastx.py         # Full pipeline: ForecastX class
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
# Using uv
uv run pytest

# Using pip
pytest tests/ -q
```

275 tests covering all models, edge cases, and integration scenarios.

## Contributing

Contributions are welcome. Fork, branch, and submit a pull request.

```bash
git clone https://github.com/paxbun/forecastx.git
cd forecastx
uv sync --extra dev
uv run pytest
```

## License

[MIT](LICENSE) -- Use freely in personal and commercial projects.
