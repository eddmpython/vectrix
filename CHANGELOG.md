# Changelog

All notable changes to Vectrix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2026-02-27

Initial public release of Vectrix -- a zero-config time series forecasting library built with pure NumPy + SciPy.

### Added

**Core Forecasting Engine (30+ Models)**
- AutoETS: 30 Error x Trend x Seasonal combinations with AICc model selection (Hyndman-Khandakar stepwise algorithm)
- AutoARIMA: Seasonal ARIMA with stepwise order selection, CSS objective function
- Theta / Dynamic Optimized Theta (DOT): Original Theta method + M3 Competition winner methodology
- AutoCES: Complex Exponential Smoothing (Svetunkov 2023)
- AutoTBATS: Trigonometric seasonality for complex multi-seasonal time series
- GARCH / EGARCH / GJR-GARCH: Conditional volatility modeling with asymmetric effects
- Croston Classic / SBA / TSB / AutoCroston: Intermittent and lumpy demand forecasting
- Logistic Growth: Prophet-style saturating trends with user-defined capacity constraints
- AutoMSTL: Multi-seasonal STL decomposition with ARIMA residual forecasting
- Baseline models: Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

**Novel Methods**
- Lotka-Volterra Ensemble: Ecological competition dynamics for adaptive model weighting
- Phase Transition Forecaster: Critical slowing down detection for regime shift prediction
- Adversarial Stress Tester: 5 perturbation operators (spike, dropout, drift, noise, swap) for forecast robustness analysis
- Hawkes Intermittent Demand: Self-exciting point process for clustered demand patterns
- Entropic Confidence Scorer: Shannon entropy-based forecast uncertainty quantification

**Adaptive Intelligence**
- Regime Detection: Pure numpy Hidden Markov Model implementation (Baum-Welch + Viterbi)
- Self-Healing Forecast: CUSUM + EWMA drift detection with conformal prediction correction
- Constraint-Aware Forecasting: 8 business constraint types (non-negative, range, capacity, YoY change, sum, monotone, ratio, custom)
- Forecast DNA: 65+ time series feature fingerprinting with meta-learning model recommendation and similarity search
- Flat Defense: 4-level system (diagnostic, detection, correction, prevention) against flat prediction failure

**Easy API**
- `forecast()`: One-call forecasting with automatic model selection, accepts str/DataFrame/Series/ndarray/list/tuple/dict
- `analyze()`: Time series DNA profiling, changepoint detection, anomaly identification
- `regress()`: R-style formula regression (`y ~ x1 + x2`) with full diagnostics
- `quick_report()`: Combined analysis + forecast report generation
- Rich result objects with `.plot()`, `.to_csv()`, `.to_json()`, `.to_dataframe()`, `.summary()`, `.describe()`

**Regression & Diagnostics**
- 5 regression methods: OLS, Ridge, Lasso, Huber, Quantile
- R-style formula interface: `regress(data=df, formula="sales ~ ads + price")`
- Full diagnostics suite: Durbin-Watson, Breusch-Pagan, VIF, Jarque-Bera normality tests
- Variable selection: Stepwise (forward/backward), regularization CV, best subset
- Time series regression: Newey-West HAC, Cochrane-Orcutt, Prais-Winsten, Granger causality

**Business Intelligence**
- Anomaly detection with automated outlier identification and natural language explanation
- What-if analysis: Scenario-based forecast simulation with parameter perturbation
- Backtesting: Rolling origin cross-validation with MAE, RMSE, MAPE, SMAPE metrics
- Hierarchy reconciliation: Bottom-up, Top-down, MinTrace optimal (Wickramasuriya 2019)
- Prediction intervals: Conformal prediction + Bootstrap methods

**Infrastructure**
- Batch forecasting API with ThreadPoolExecutor parallelization
- Model persistence: `.fxm` binary format with save/load/info utilities
- TSFrame: Time series DataFrame wrapper with frequency detection
- Global model: Cross-series learning for related time series
- Numba JIT acceleration for core computations (optional dependency)
- 275 tests covering all models, edge cases, and integration scenarios
- GitHub Actions CI: Matrix testing (Python 3.10-3.13, Ubuntu + Windows)
- PyPI trusted publisher deployment via GitHub Actions

[0.0.1]: https://github.com/eddmpython/vectrix/releases/tag/v0.0.1
