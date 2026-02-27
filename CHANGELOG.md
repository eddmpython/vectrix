# Changelog

All notable changes to Vectrix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2026-02-27

### Added

**Core Forecasting Engine (30+ Models)**
- AutoETS: 30 Error x Trend x Seasonal combinations with AICc selection (Hyndman-Khandakar stepwise)
- AutoARIMA: Seasonal ARIMA with stepwise order selection, CSS objective with Numba JIT
- Theta / Dynamic Optimized Theta (DOT): M3 Competition winner methodology
- AutoCES: Complex Exponential Smoothing (Svetunkov 2023)
- AutoTBATS: Trigonometric multi-seasonal decomposition
- GARCH / EGARCH / GJR-GARCH: Volatility modeling
- Croston Classic / SBA / TSB / AutoCroston: Intermittent demand forecasting
- Logistic Growth: Prophet-style saturating trends with capacity constraints
- AutoMSTL: Multi-seasonal decomposition with ARIMA residual forecasting
- 5 Baseline models: Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

**Novel Methods (World First)**
- Lotka-Volterra Ensemble: Ecological competition dynamics for adaptive model weighting
- Phase Transition Forecaster: Critical slowing down detection for regime shift prediction
- Adversarial Stress Tester: 5 perturbation operators for forecast robustness analysis
- Hawkes Intermittent Demand: Self-exciting point process for clustered demand patterns
- Entropic Confidence Scorer: Shannon entropy-based forecast uncertainty quantification

**Adaptive Intelligence**
- Regime Detection: Pure numpy Hidden Markov Model (Baum-Welch + Viterbi)
- Self-Healing Forecast: CUSUM + EWMA drift detection with conformal prediction correction
- Constraint-Aware Forecasting: 8 business constraints (non-negative, range, capacity, YoY, sum, monotone, ratio, custom)
- Forecast DNA: 65+ feature fingerprinting with meta-learning model recommendation
- Flat Defense: 4-level system against flat prediction failure

**Easy API**
- `forecast()`: One-call forecasting with auto model selection
- `analyze()`: Time series DNA profiling and diagnostics
- `regress()`: R-style formula regression with full diagnostics
- `quick_report()`: Combined analysis + forecast report
- Input flexibility: str (CSV), DataFrame, Series, ndarray, list, tuple, dict
- Rich result objects: `.plot()`, `.to_csv()`, `.to_json()`, `.save()`, `.describe()`

**Regression & Diagnostics**
- 5 regression methods: OLS, Ridge, Lasso, Huber, Quantile
- R-style formula interface: `regress(data=df, formula="y ~ x1 + x2")`
- Full diagnostics: Durbin-Watson, Breusch-Pagan, VIF, normality tests
- Time series regression: Newey-West, Cochrane-Orcutt, Granger causality

**Business Intelligence**
- Anomaly detection, What-if analysis, Backtesting, Forecast explanation
- Hierarchy reconciliation: Bottom-up, Top-down, MinTrace (Wickramasuriya 2019)
- Prediction intervals: Conformal + Bootstrap methods
- HTML report generator: Self-contained SVG inline charts

**Infrastructure**
- Batch forecasting API with ThreadPoolExecutor parallelization
- Model persistence: `.fxm` binary format with save/load/info
- 275 tests with parametrized model coverage
- Numba JIT acceleration for core computations

[3.0.0]: https://github.com/eddmpython/vectrix/releases/tag/v3.0.0
