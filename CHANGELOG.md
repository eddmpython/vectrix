# Changelog

All notable changes to Vectrix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.5] - 2026-03-02

Performance release — Rust turbo acceleration extended to DOT, CES, and 4Theta models (vectrix-core 0.2.0).

### Changed

**Rust Turbo Mode Extended (vectrix-core 0.2.0)**
- DOT (Dynamic Optimized Theta): 68ms → 2.8ms (24x faster) — `dot_objective`, `dot_residuals` Rust hot paths
- AutoCES (Complex Exponential Smoothing): 118ms → 9.6ms (12x faster) — `ces_nonseasonal_sse`, `ces_seasonal_sse` Rust hot paths
- 4Theta (Adaptive Theta Ensemble): 63ms → 5.6ms (11x faster) — wired existing `ses_sse`/`ses_filter` Rust functions
- Total: 13 Rust-accelerated functions (was 9, added 4 new)
- 3-tier fallback preserved: Rust > Numba JIT > Pure Python
- All functions produce bit-identical results with Python reference implementations

[0.0.5]: https://github.com/eddmpython/vectrix/compare/v0.0.4...v0.0.5

## [0.0.4] - 2026-03-02

Quality & internationalization release — full English docstring conversion, 573 tests (+186), improved model selection with DOT/CES defaults.

### Changed

**English Docstring Conversion**
- Complete Korean→English conversion across all 60+ source modules
- All docstrings, error messages, comments, and user-facing strings now in English
- API Reference documentation (mkdocstrings) now renders correctly in English
- Korean column detection keywords (`'날짜', '일자', '일시'`) preserved for Korean DataFrame auto-detection

**Model Selection Improvement**
- DOT (Dynamic Optimized Theta) and AutoCES now included as default model candidates
- M4-validated: DOT OWA 0.905 (#18 level), AutoCES OWA 0.927 — both top-performing general-purpose models
- Hourly data: DTSF + MSTL prioritized for multi-seasonal pattern capture
- Fallback models upgraded from four_theta/esn to dot/auto_ces

### Added

**Test Coverage Expansion (387 → 573, +48%)**
- `test_new_models.py`: 45 tests for DTSF, ESN, 4Theta (pattern matching, nonlinear, M4-style holdout)
- `test_business.py`: 45 tests for anomaly detection, backtesting, metrics, what-if, reports, HTML reports
- `test_infrastructure.py`: 43 tests for flat defense, hierarchy reconciliation, batch, persistence, TSFrame, AutoAnalyzer
- `test_engine_utils.py`: 53 tests for ARIMAX, cross-validation, decomposition, diagnostics, periodic drop, comparison, imputation

### Fixed

- Test assertions updated for English error messages (pipeline, holiday names)
- FlatPredictionType enum comments translated

[0.0.4]: https://github.com/eddmpython/vectrix/compare/v0.0.3...v0.0.4

## [0.0.3] - 2026-02-28

Performance release — Rust-accelerated core loops (vectrix-core), built-in sample datasets, and pandas 2.x compatibility fixes.

### Added

**Rust Turbo Mode (vectrix-core)**
- Native Rust extension for core forecasting hot loops via PyO3 + maturin
- 9 accelerated functions: `ets_filter`, `ets_loglik`, `css_objective`, `seasonal_css_objective`, `ses_sse`, `ses_filter`, `theta_decompose`, `arima_css`, `batch_ets_filter`
- 3-tier fallback: Rust > Numba JIT > Pure Python — transparent, no code changes needed
- Pre-built wheels for Linux (manylinux), macOS (x86 + ARM), Windows (x86_64), Python 3.10-3.13
- Install via `pip install "vectrix[turbo]"` — no Rust compiler needed for users
- GitHub Actions CI workflow (`publish-core.yml`) for automated wheel builds on `core-v*` tags

**Built-in Sample Datasets**
- 7 deterministic sample datasets for quick testing: `airline` (144 monthly), `retail` (730 daily), `stock` (252 business daily), `temperature` (1095 daily), `energy` (720 hourly), `web` (180 daily), `intermittent` (365 daily)
- `loadSample(name)`: Load a sample dataset as DataFrame
- `listSamples()`: List all available datasets with metadata
- 41 tests covering all datasets

### Changed

**Performance Improvements**
- AutoETS: 348ms → 32ms (10.8x faster with Rust turbo)
- AutoARIMA: 195ms → 35ms (5.6x faster)
- Theta: 1.3ms → 0.16ms (8.1x faster)
- `forecast()` end-to-end: 295ms → 52ms (5.6x faster)
- ETS filter hot loop: 0.17ms → 0.003ms (67x faster)
- ARIMA CSS objective: 0.19ms → 0.001ms (157x faster)

### Fixed

- pandas 2.x frequency deprecation: `"M"` → `"ME"`, `"Q"` → `"QE"`, `"Y"` → `"YE"`, `"H"` → `"h"`

### Changed (Docs)

- Complete bilingual docs site (EN/KO) with i18n plugin
- Installation guide updated with Rust Turbo Mode section
- README updated with turbo benchmarks, sample datasets, comparison table
- 387 tests (up from 346), 5 skipped (optional dependency guards)

[0.0.3]: https://github.com/eddmpython/vectrix/compare/v0.0.2...v0.0.3

## [0.0.2] - 2026-02-28

Feature expansion release — Foundation Model wrappers, deep learning models, multivariate forecasting, probabilistic distributions, multi-country holidays, and pipeline system.

### Added

**Foundation Model Wrappers (Optional)**
- `ChronosForecaster`: Amazon Chronos-2 zero-shot forecasting wrapper with batch prediction and quantile output
- `TimesFMForecaster`: Google TimesFM 2.5 wrapper with covariate support and multi-horizon prediction
- Optional dependency groups: `foundation` (torch + chronos-forecasting), `neural` (neuralforecast)

**Deep Learning Model Wrappers (Optional)**
- `NeuralForecaster`: NeuralForecast wrapper supporting NBEATS, NHITS, TFT architectures
- Convenience classes: `NBEATSForecaster`, `NHITSForecaster`, `TFTForecaster`
- Automatic numpy ↔ NeuralForecast DataFrame conversion

**Probabilistic Forecast Distributions**
- `ForecastDistribution`: Parametric distribution forecasting (Gaussian, Student-t, Log-Normal)
- `DistributionFitter`: Automatic distribution selection via AIC comparison
- `empiricalCRPS`: Closed-form Gaussian CRPS + Monte Carlo CRPS for other distributions
- Full distribution API: quantile, interval, sample, pdf, crps methods

**Multivariate Models**
- `VARModel`: Vector AutoRegression with automatic lag selection (AIC/BIC) and Granger causality test
- `VECMModel`: Vector Error Correction Model with Johansen-style cointegration rank estimation

**Multi-Country Holiday Support**
- US holidays: 4 fixed (New Year, Independence Day, Veterans Day, Christmas) + 6 floating (MLK, Presidents', Memorial, Labor, Columbus, Thanksgiving)
- Japan holidays: 13 fixed national holidays (元日, 建国記念の日, etc.)
- China holidays: 5 fixed holidays (元旦, 劳动节, 国庆节)
- `getHolidays(year)`: Unified holiday query for KR/US/JP/CN
- `adjustForecast()`: Apply estimated event effects to point forecasts
- `_nthWeekdayOfMonth()`: Floating holiday date calculation helper

**Pipeline System**
- `ForecastPipeline`: sklearn-style sequential chaining with automatic inverse transform on predictions
- 8 built-in transformers: `Differencer`, `LogTransformer`, `BoxCoxTransformer`, `Scaler` (zscore/minmax), `Deseasonalizer`, `Detrend`, `OutlierClipper`, `MissingValueImputer`
- Named step access, parameter nesting (`step__param`), repr display

### Changed

**Speed Improvements**
- Parallelized model evaluation in `Vectrix._evaluateNativeModels` via `ThreadPoolExecutor`
- Parallelized cross-validation candidate loop in M3/M4 benchmark runners
- ~13% faster M3 Monthly benchmark (11.22s → 9.70s)

**Test Coverage**
- 346 tests (up from 275), 5 skipped (optional dependency guards)

[0.0.2]: https://github.com/eddmpython/vectrix/compare/v0.0.1...v0.0.2

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
