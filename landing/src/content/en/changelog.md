---
title: Changelog
---

# Changelog

All notable changes to Vectrix are documented here. This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## [0.0.7] - 2026-03-02

AI integration release -- llms.txt, MCP server (10 tools, 2 resources, 2 prompts), Claude Code skills (3).

### Added

**llms.txt / llms-full.txt**
- `llms.txt`: Structured project overview following the [llms.txt standard](https://llmstxt.org/) with documentation links, quick start, and API sections
- `llms-full.txt`: Complete API reference (every class, method, parameter, return type) for full library understanding
- Deployed to GitHub Pages root and included in PyPI package

**MCP Server (Model Context Protocol)**
- 10 tools: `forecast_timeseries`, `forecast_csv`, `analyze_timeseries`, `compare_models`, `run_regression`, `detect_anomalies`, `backtest_model`, `list_sample_datasets`, `load_sample_dataset`
- 2 resources: `vectrix://models`, `vectrix://api-reference`
- 2 prompts: `forecast_workflow`, `regression_workflow`
- Compatible with Claude Desktop, Claude Code, and any MCP client
- Install: `pip install "vectrix[mcp]"`

**Claude Code Skills (3)**
- `vectrix-forecast`: Time series forecasting workflow with full API reference
- `vectrix-analyze`: DNA profiling, anomaly detection, regime analysis
- `vectrix-regress`: R-style regression, diagnostics, variable selection

## [0.0.6] - 2026-03-02

Documentation and deployment release -- tutorials, showcases, EasyForecastResult enhancements, unified SvelteKit + MkDocs GitHub Pages.

### Added

**EasyForecastResult Enhancements**
- `compare()`: Side-by-side model comparison table with sMAPE, MAPE, RMSE, MAE
- `all_forecasts()`: DataFrame of all valid model forecasts
- Accuracy attributes: `.mape`, `.rmse`, `.mae`, `.smape` on result objects
- `Vectrix._refitAllModels()`: Refit all valid models for compare/all_forecasts

**Tutorials (6 topics x 2 languages = 12 files)**
- Quickstart, analyze, regression, models, adaptive, business

**Showcases (marimo interactive notebooks)**
- Korean Economy Forecasting, Korean Regression, Model Comparison, Business Intelligence
- Companion markdown pages for GitHub Pages

### Changed

**Unified GitHub Pages Deployment**
- SvelteKit landing page at root (`/vectrix/`)
- MkDocs documentation at `/vectrix/docs/`
- Single `docs.yml` workflow builds and merges both

## [0.0.5] - 2026-03-02

Rust turbo extended -- DOT 24x, AutoCES 12x, 4Theta 11x acceleration. 13 total Rust-accelerated functions.

### Changed

**Rust Turbo Mode Extended (vectrix-core 0.2.0)**
- DOT (Dynamic Optimized Theta): 68ms to 2.8ms (24x faster)
- AutoCES (Complex Exponential Smoothing): 118ms to 9.6ms (12x faster)
- 4Theta (Adaptive Theta Ensemble): 63ms to 5.6ms (11x faster)
- Total: 13 Rust-accelerated functions (was 9, added 4 new)
- 3-tier fallback preserved: Rust > Numba JIT > Pure Python
- Bit-identical results with Python reference implementations

## [0.0.4] - 2026-03-02

Quality release -- English docstrings across 60+ modules, DOT/CES as default model candidates, 573 tests (+186).

### Changed

**English Docstring Conversion**
- Complete Korean to English conversion across all 60+ source modules
- All docstrings, error messages, comments now in English
- API Reference documentation renders correctly in English

**Model Selection Improvement**
- DOT and AutoCES now included as default model candidates
- M4-validated: DOT OWA 0.905, AutoCES OWA 0.927

### Added

**Test Coverage Expansion (387 to 573, +48%)**
- `test_new_models.py`: 45 tests for DTSF, ESN, 4Theta
- `test_business.py`: 45 tests for anomaly detection, backtesting, metrics, what-if
- `test_infrastructure.py`: 43 tests for flat defense, hierarchy, batch, persistence
- `test_engine_utils.py`: 53 tests for ARIMAX, cross-validation, decomposition

## [0.0.3] - 2026-02-28

Rust turbo mode -- 9 accelerated functions, 5-10x speedup, pre-built wheels for all platforms. Built-in sample datasets. pandas 2.x compatibility fixes.

### Added

**Rust Turbo Mode (vectrix-core)**
- Native Rust extension via PyO3 + maturin
- 9 accelerated functions: `ets_filter`, `ets_loglik`, `css_objective`, `seasonal_css_objective`, `ses_sse`, `ses_filter`, `theta_decompose`, `arima_css`, `batch_ets_filter`
- 3-tier fallback: Rust > Numba JIT > Pure Python
- Pre-built wheels for Linux, macOS (x86 + ARM), Windows
- Install: `pip install "vectrix[turbo]"`

**Built-in Sample Datasets**
- 7 datasets: `airline`, `retail`, `stock`, `temperature`, `energy`, `web`, `intermittent`
- `loadSample(name)` and `listSamples()` API

### Changed

**Performance Improvements**
- AutoETS: 348ms to 32ms (10.8x faster)
- AutoARIMA: 195ms to 35ms (5.6x faster)
- Theta: 1.3ms to 0.16ms (8.1x faster)
- `forecast()` end-to-end: 295ms to 52ms (5.6x faster)

### Fixed

- pandas 2.x frequency deprecation: `"M"` to `"ME"`, `"Q"` to `"QE"`, `"Y"` to `"YE"`, `"H"` to `"h"`

## [0.0.2] - 2026-02-28

Foundation models (Chronos, TimesFM), deep learning (NBEATS, NHITS, TFT), VAR/VECM multivariate, multi-country holidays, pipeline system, probabilistic distributions.

### Added

**Foundation Model Wrappers (Optional)**
- `ChronosForecaster`: Amazon Chronos-2 zero-shot forecasting
- `TimesFMForecaster`: Google TimesFM 2.5 with covariate support

**Deep Learning Model Wrappers (Optional)**
- `NeuralForecaster`: NeuralForecast wrapper for NBEATS, NHITS, TFT
- Convenience classes: `NBEATSForecaster`, `NHITSForecaster`, `TFTForecaster`

**Probabilistic Forecast Distributions**
- `ForecastDistribution`: Parametric distribution forecasting (Gaussian, Student-t, Log-Normal)
- `DistributionFitter`: Automatic distribution selection via AIC
- `empiricalCRPS`: Closed-form Gaussian CRPS + Monte Carlo CRPS

**Multivariate Models**
- `VARModel`: Vector AutoRegression with automatic lag selection
- `VECMModel`: Vector Error Correction with cointegration rank estimation

**Multi-Country Holiday Support**
- US, Japan, China, Korea holidays
- `getHolidays(year)` and `adjustForecast()` API

**Pipeline System**
- `ForecastPipeline`: sklearn-style sequential chaining
- 8 transformers: `Differencer`, `LogTransformer`, `BoxCoxTransformer`, `Scaler`, `Deseasonalizer`, `Detrend`, `OutlierClipper`, `MissingValueImputer`

### Changed

- Parallelized model evaluation via `ThreadPoolExecutor`
- 346 tests (up from 275)

## [0.0.1] - 2026-02-27

Initial release -- 30+ forecasting models, adaptive intelligence, Easy API, regression suite, business intelligence, 275 tests.

### Added

**Core Forecasting Engine (30+ Models)**
- AutoETS, AutoARIMA, Theta, DOT, AutoCES, AutoTBATS, GARCH/EGARCH/GJR-GARCH
- Croston (Classic/SBA/TSB/Auto), Logistic Growth, AutoMSTL
- Baseline: Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

**Novel Methods**
- Lotka-Volterra Ensemble, Phase Transition Forecaster, Adversarial Stress Tester
- Hawkes Intermittent Demand, Entropic Confidence Scorer

**Adaptive Intelligence**
- Regime Detection (HMM), Self-Healing Forecast (CUSUM + EWMA)
- Constraint-Aware Forecasting (8 constraint types)
- Forecast DNA (65+ features, meta-learning model recommendation)
- Flat Defense (4-level diagnostic/detection/correction/prevention)

**Easy API**
- `forecast()`: One-call forecasting with automatic model selection
- `analyze()`: Time series DNA profiling and anomaly identification
- `regress()`: R-style formula regression with full diagnostics
- `quick_report()`: Combined analysis + forecast report

**Regression and Diagnostics**
- 5 methods: OLS, Ridge, Lasso, Huber, Quantile
- Full diagnostics: Durbin-Watson, Breusch-Pagan, VIF, Jarque-Bera
- Variable selection: Stepwise, regularization CV, best subset
- Time series regression: Newey-West HAC, Cochrane-Orcutt, Prais-Winsten

**Business Intelligence**
- Anomaly detection, what-if analysis, backtesting
- Hierarchy reconciliation (Bottom-up, Top-down, MinTrace)
- Prediction intervals (Conformal + Bootstrap)

**Infrastructure**
- Batch forecasting with parallel execution
- Model persistence (`.fxm` format)
- TSFrame, Global model, Numba JIT acceleration
- GitHub Actions CI (Python 3.10-3.13, Ubuntu + Windows)
- PyPI trusted publisher deployment
