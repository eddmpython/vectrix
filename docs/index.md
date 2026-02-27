---
hide:
  - navigation
---

# Vectrix

<div style="text-align: center; margin: 2em 0;">
<p style="font-size: 1.4em; font-weight: 300; color: var(--md-default-fg-color--light);">
Zero-config time series forecasting for Python
</p>
<p style="font-size: 1.1em;">
<strong>30+ models</strong> · <strong>3 dependencies</strong> · <strong>1 line of code</strong>
</p>
</div>

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
```

That's it. Auto model selection, flat-line defense, confidence intervals, and a plot — all in one call.

[:material-download: Install](getting-started/installation.md){ .md-button .md-button--primary }
[:material-rocket-launch: Quickstart](getting-started/quickstart.md){ .md-button }
[:material-github: GitHub](https://github.com/eddmpython/vectrix){ .md-button }

---

## Why Vectrix?

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } **Zero Configuration**

    ---

    Pass your data, get forecasts. No hyperparameters, no manual model selection, no configuration files.

-   :material-language-python:{ .lg .middle } **Pure Python**

    ---

    Built on `numpy` + `scipy` + `pandas` only. No compiled extensions, no platform issues, installs everywhere.

-   :material-shield-check:{ .lg .middle } **Flat Defense System**

    ---

    Unique 4-level system prevents constant/flat predictions — the most common forecasting failure mode.

-   :material-dna:{ .lg .middle } **Forecast DNA**

    ---

    65+ feature fingerprinting. Know your data's difficulty, optimal model, and similarity to other series.

-   :material-brain:{ .lg .middle } **Adaptive Intelligence**

    ---

    Regime detection, self-healing forecasts, 8 business constraint types. Forecasts that adapt to reality.

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **Probabilistic Forecasting**

    ---

    Parametric distributions (Gaussian, Student-t, Log-Normal), quantile forecasts, CRPS scoring.

</div>

---

## Feature Comparison

| Capability | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| Zero-config forecasting | :white_check_mark: | :white_check_mark: | :x: | :x: |
| Pure Python (no compiled deps) | :white_check_mark: | :x: | :x: | :x: |
| 30+ statistical models | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| Flat prediction defense | :white_check_mark: | :x: | :x: | :x: |
| Forecast DNA fingerprinting | :white_check_mark: | :x: | :x: | :x: |
| Business constraints (8 types) | :white_check_mark: | :x: | :x: | :x: |
| R-style regression | :white_check_mark: | :x: | :x: | :x: |
| Foundation model wrappers | :white_check_mark: | :x: | :x: | :white_check_mark: |
| Pipeline system | :white_check_mark: | :x: | :x: | :white_check_mark: |
| VAR / VECM multivariate | :white_check_mark: | :x: | :x: | :white_check_mark: |

---

## Capabilities

### :material-chart-line: [Forecasting](guide/forecasting.md)
30+ models with automatic selection — ETS, ARIMA, Theta, MSTL, TBATS, GARCH, Croston, and more.

### :material-dna: [Analysis & DNA](guide/analysis.md)
Automatic time series fingerprinting with 65+ features, difficulty scoring, and optimal model recommendation.

### :material-function-variant: [Regression](guide/regression.md)
R-style formula interface: `regress(df, "sales ~ ads + price")` with OLS, Ridge, Lasso, Huber, Quantile.

### :material-brain: [Adaptive Intelligence](guide/adaptive.md)
Regime detection (HMM), self-healing forecasts (CUSUM + EWMA), and 8 business constraint types.

### :material-briefcase: [Business Intelligence](guide/business.md)
Anomaly detection, what-if scenarios, backtesting, and production-grade metrics (WAPE, MASE, bias).

### :material-pipe: [Pipeline](guide/pipeline.md)
sklearn-style `ForecastPipeline` — chain transformers (log, scale, deseasonalize) with automatic inverse on predictions.

### :material-robot: [Foundation Models](guide/foundation.md)
Optional wrappers for Amazon Chronos-2 and Google TimesFM 2.5 — zero-shot forecasting with pretrained models.

### :material-chart-multiple: [Multivariate](guide/multivariate.md)
VAR with automatic lag selection, Granger causality testing, and VECM for cointegrated series.

---

## Benchmarks

Tested against the M3 and M4 forecasting competitions (OWA < 1.0 = beats Naive2):

| Competition | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **M3** | **0.848** | **0.825** | **0.758** | — | — | **0.819** |
| **M4** | **0.974** | **0.797** | **0.987** | **0.737** | 1.207 | 1.006 |

[:material-chart-bar: Full benchmark results](benchmarks.md)

---

## Installation

=== "Core"

    ```bash
    pip install vectrix
    ```
    NumPy + SciPy + Pandas only. Works on Python 3.10+.

=== "With Numba"

    ```bash
    pip install "vectrix[numba]"
    ```
    2-5x faster with JIT compilation for core algorithms.

=== "With ML"

    ```bash
    pip install "vectrix[ml]"
    ```
    Adds LightGBM, XGBoost, scikit-learn ensemble models.

=== "Foundation Models"

    ```bash
    pip install "vectrix[foundation]"
    ```
    Amazon Chronos-2 and Google TimesFM 2.5 wrappers.

=== "Everything"

    ```bash
    pip install "vectrix[all]"
    ```
    All optional dependencies included.

---

<div style="text-align: center; margin: 2em 0; color: var(--md-default-fg-color--light);">
<a href="https://github.com/eddmpython/vectrix/blob/master/LICENSE">MIT License</a> — Use freely in personal and commercial projects.
</div>
