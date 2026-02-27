# Vectrix

**Pure Python Time Series Forecasting Engine**

30+ models · 3 dependencies · 1 line of code

---

## Quick Start

```bash
pip install vectrix
```

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
```

One function call. Auto model selection, flat-line defense, confidence intervals, and a plot.

---

## Why Vectrix?

| Dimension | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| **Zero-config** | :white_check_mark: | :white_check_mark: | :x: | :x: |
| **Pure Python** | :white_check_mark: | :x: | :x: | :x: |
| **30+ models** | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| **Flat defense** | :white_check_mark: | :x: | :x: | :x: |
| **Stress testing** | :white_check_mark: | :x: | :x: | :x: |
| **Forecast DNA** | :white_check_mark: | :x: | :x: | :x: |
| **Constraints (8)** | :white_check_mark: | :x: | :x: | :x: |
| **R-style regress** | :white_check_mark: | :x: | :x: | :x: |

**Three vectors.** `numpy` · `scipy` · `pandas` — that's the entire orbit.

---

## Features

### :material-chart-line: [Forecasting](guide/forecasting.md)
30+ models with automatic selection. ETS, ARIMA, Theta, MSTL, TBATS, GARCH, and more.

### :material-dna: [Analysis & DNA](guide/analysis.md)
Automatic time series fingerprinting, difficulty scoring, and optimal model recommendation.

### :material-function-variant: [Regression](guide/regression.md)
R-style formula interface with OLS, Ridge, Lasso, Huber, Quantile and full diagnostics.

### :material-brain: [Adaptive Intelligence](guide/adaptive.md)
Regime detection, self-healing forecasts, business constraints, and Forecast DNA.

### :material-briefcase: [Business Intelligence](guide/business.md)
Anomaly detection, what-if analysis, backtesting, and business metrics.

---

## Installation

```bash
pip install vectrix                # Core (numpy + scipy + pandas)
pip install "vectrix[numba]"       # + Numba JIT (2-5x speedup)
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # Everything
```

---

## License

[MIT](https://github.com/eddmpython/vectrix/blob/master/LICENSE) — Use freely in personal and commercial projects.
