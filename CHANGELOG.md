# Changelog

All notable changes to Vectrix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.16] - 2026-03-05

Code quality and architecture cleanup release — eliminates MODEL_INFO indirection layer, enforces registry as single source of truth for all model metadata, fixes ETSModel refit bug, and adds refit contract tests.

### Fixed

**ETSModel refit() State Corruption**
- `engine/ets.py`: `_computeSSE()` called `_filter()` which mutated `self.level/trend/seasonal` as a side-effect during optimization
- This caused refit() on the same data to produce 175%+ divergence from original fit()
- Added state save/restore in `_computeSSE` using try/finally pattern
- All 6 refit-capable models now pass same-data similarity test (<5% divergence)

### Changed

**Registry as Single Source of Truth (SSoT)**
- `engine/registry.py`: Added `tier`, `seasonal`, `hourly` fields to `ModelSpec`
- Added `selectModels()` for data-characteristic-based model selection
- Added `getCoreModelIds()` for dynamic ensemble core pool
- `vectrix.py`: Replaced hardcoded model lists with `selectModels()` and `getCoreModelIds()`
- `flat_defense/diagnostic.py`: Replaced `MODEL_INFO` import with `getModelInfo()` direct call
- `models/selector.py`: Same — `MODEL_INFO` → `getModelInfo()` direct call
- `types.py`: Removed `_LazyModelInfo` class and `MODEL_INFO` variable entirely (48 lines deleted)

**Exception Handling**
- `vectrix.py`: Replaced 6 bare `except Exception:` with specific types (`ValueError`, `RuntimeError`, `np.linalg.LinAlgError`, etc.)

**Code Deduplication**
- `vectrix.py`: Unified sequential/parallel model evaluation result handling into `storeResult()` helper

### Added

**Refit Contract Tests**
- `tests/test_refit_contract.py`: 30 tests covering 6 refit-capable models
- Tests: returns_self, output_shape, output_finite, ci_ordering, same_data_similar_output
- Total tests: 573 → 603

## [0.0.15] - 2026-03-05

Accuracy & engine improvement release — FFT period detection bug fix eliminates spurious seasonal periods that degraded M4 Daily OWA from 0.820 to 0.996. Three new Rust DNA functions added. PeriodicDropDetector seasonal false positive fix. Visualization module (plotly). Overall M4 AVG OWA improved from 0.877 to 0.848.

### Fixed

**FFT Period Detection Bug (Root Cause of Daily OWA 0.996)**
- `analyzer/autoAnalyzer.py`: `_detectSeasonalPeriods()` used FFT which produced spurious periods (53, 144, 168 for Daily data)
- These wrong periods propagated through `forecast()` pipeline, degrading accuracy across all M4 frequency groups
- Replaced FFT-based detection with frequency-based lookup table (Daily→7, Monthly→12, etc.)
- M4 Daily OWA: 0.996 → 0.820 (17.7% improvement)
- M4 AVG OWA: 0.877 → 0.848 (3.3% improvement)

**PeriodicDropDetector Seasonal False Positives**
- `vectrix.py`: Normal seasonal patterns (e.g., 24h Hourly cycle) were falsely detected as "periodic drops"
- This triggered interpolation that destroyed training data, harming forecast accuracy
- Added check: if detected drop period matches or aligns with seasonal period, skip drop correction

**Ruff Lint Errors**
- Fixed import sort order in `engine/tsfeatures.py`
- Removed unused `numpy` imports in `viz/charts.py` and `viz/report.py`
- Removed unused `make_subplots` import in `viz/charts.py`

### Added

**Rust DNA Functions (3 new, total 29)**
- `sample_entropy`: Measures time series complexity/regularity
- `approximate_entropy`: Similar to sample entropy with different algorithm
- `hurst_exponent`: Long-range dependence measurement via R/S analysis
- Python vectorized fallbacks using numpy broadcasting + stride_tricks
- `engine/tsfeatures.py` auto-selects Rust or Python implementation

**Visualization Module (`viz/`)**
- `viz/charts.py`: Individual Plotly chart functions (forecastChart, analysisChart, etc.)
- `viz/report.py`: Composite report generators combining multiple charts
- `viz/theme.py`: Dark theme configuration with consistent color palette

**Experiments (E047-E054)**
- E047~E050: Speed optimization experiments (all rejected — Python overhead is not in hot loops)
- E051: Daily diagnosis — DOT engine alone gives 0.605 but pipeline gives 0.908 (50% gap)
- E052: Root cause identified — FFT produces spurious periods (53, 144, 168)
- E053: basePeriod-only wins all 6 frequency groups (-19.3% avg improvement)
- E054: Integrated pipeline benchmark with both fixes applied

### Changed

**AdaptiveModelSelector Registry-Based**
- `models/selector.py`: Replaced hardcoded FLAT_RESISTANCE and MIN_DATA dicts with dynamic lookup from `engine/registry.py`

**M4 Benchmark Numbers Updated Across All Files**
- Daily OWA: 0.996 → 0.820 (12 files updated)
- AVG OWA: 0.877 → 0.848 (12 files updated)
- Files: constants.json, README.md, docs/benchmarks.md, llms.txt, llms-full.txt, landing components, blog posts, API_SPEC.md

[0.0.15]: https://github.com/eddmpython/vectrix/compare/v0.0.14...v0.0.15

## [0.0.14] - 2026-03-04

Model refit() interface — each model owns its own refit logic, eliminating the last model-specific if/elif chain in vectrix.py. GitHub Deployments fix.

### Changed

**refit() Interface Added to All Auto Models**
- `AutoETS.refit(newData)`: reuses selected model structure (error/trend/seasonal type + smoothing params), skips model selection
- `ETSModel.refit(newData)`: reuses smoothing parameters, skips optimization
- `AutoARIMA.refit(newData)`: reuses selected order (p,d,q)(P,D,Q)[m], skips order selection
- `OptimizedTheta.refit(newData)`: reuses selected theta value, skips theta optimization
- `AutoMSTL.refit(newData)`: reuses detected periods, skips period detection

**vectrix.py `_refitModelOnFullData` Simplified (50 lines → 8 lines)**
- Removed 5-branch if/elif chain with model-specific refit logic
- Now uses `hasattr(model, 'refit')` — zero model-specific code in vectrix.py
- Removed `refitStrategy` field from `ModelSpec` — no longer needed

**GitHub Actions: Deployment Tracking Fix**
- Added `environment: pypi` to publish job in `.github/workflows/publish.yml`
- Enables GitHub Deployments section to track each release (previously showed only first deployment)

### Removed

- `ModelSpec.refitStrategy` field — replaced by model-owned `refit()` methods

[0.0.14]: https://github.com/eddmpython/vectrix/compare/v0.0.13...v0.0.14

## [0.0.13] - 2026-03-04

Model Registry architecture — eliminates coupling between model metadata, model creation, and orchestration. Adding a new model now requires editing only 1 file instead of 5.

### Changed

**Architecture: Model Registry Pattern (`engine/registry.py`)**
- Created `engine/registry.py` as the Single Source of Truth for all model metadata
- `ModelSpec` dataclass: modelId, name, description, factory, needsPeriod, minData, flatResistance, bestFor
- `createModel(modelId, period)`: unified model instantiation — replaces 200-line if/elif chain in vectrix.py
- `getRegistry()`, `getModelSpec()`, `listModelIds()`, `getModelInfo()`: centralized access to model information
- Adding a new forecasting model now requires only 1 edit: add a `ModelSpec` entry to `registry.py`

**vectrix.py Refactored (959 → 709 lines, -26%)**
- Removed `NATIVE_MODELS` class dict (80+ lines) — replaced by `getRegistry()`
- Removed `_fitAndPredictNativeWithCache` 200-line if/elif chain — replaced by `createModel()` + `fit()` + `predict()`
- `_getModelName()` static method replaces repeated dict lookups

**types.py MODEL_INFO Unified**
- `MODEL_INFO` is now a lazy-loading proxy that reads from `engine/registry.py` on first access
- Backward compatible: all existing code using `MODEL_INFO[modelId]` continues to work
- Eliminates duplicate model metadata between types.py and vectrix.py

### Added

- `engine/registry.py`: Model registry module with `ModelSpec`, `getRegistry()`, `createModel()`, `getModelInfo()`

[0.0.13]: https://github.com/eddmpython/vectrix/compare/v0.0.12...v0.0.13

## [0.0.12] - 2026-03-04

DOT-Hybrid holdout validation release — 8-way config selection for period>1 data now uses holdout validation instead of in-sample MAE, reducing overfitting on Quarterly (-1.25%) and Monthly (-2.55%) forecasts. AVG OWA improved from 0.8831 to ~0.876.

### Changed

**DOT-Hybrid Engine Holdout Validation**
- `engine/dot.py`: `_fitHybrid()` now uses holdout-based config selection when `period > 1` and sufficient data available
- When `period > 1`: splits data into train/validation, evaluates 8 variant configurations on held-out segment, selects best by validation MAE, then refits on full data
- When `period <= 1` (Yearly, Daily, Weekly): preserves original in-sample MAE selection — no behavioral change
- When `period >= 24` (Hourly): unchanged, uses classic DOT path as before
- Added `_predictVariantSteps()` helper method for multi-step holdout prediction
- Net effect: Quarterly OWA -1.25%, Monthly OWA -2.55%, zero regression on other groups

### Added

**Experiment Files (4 new DOT improvement experiments)**
- `modelCreation/043_dotAutoPeriodHoldout.py`: ACF-based auto period detection (REJECTED, +1.29%) + holdout validation (ACCEPTED, -0.79%)
- `modelCreation/044_dailyWeeklySpecialist.py`: Classic DOT for Weekly (ACCEPTED, -2.18%) + Core3 ensemble for Daily/Weekly (REJECTED, +21%/+8%)
- `modelCreation/045_integratedImprovement.py`: Integrated holdout + Weekly classic (AVG -0.94%, but Yearly +1.16% regression)
- `modelCreation/046_finalIntegration.py`: Final rule validation — period<=1 classic vs period>1 holdout isolation confirmed safe

### Key Findings

- ACF-based auto period detection detects spurious short periods (2,3) from noise — harmful for accuracy
- Holdout validation eliminates in-sample overfitting in 8-way config selection for seasonal data
- Core3 ensemble (DOT+CES+4Theta) is harmful for period=1 data — CES/4Theta struggle without seasonality
- Classic DOT is good for Weekly (period=1) but catastrophic for Yearly (period=1) — Yearly needs Hybrid's trend exploration
- Safe improvement scope: only `1 < period < 24` benefits from holdout validation

[0.0.12]: https://github.com/eddmpython/vectrix/compare/v0.0.11...v0.0.12

## [0.0.11] - 2026-03-04

Progressive Disclosure release — Easy API now supports Level 2 guided control with model selection, ensemble strategy, and confidence interval parameters, while maintaining full backward compatibility with Level 1 zero-config usage.

### Added

**Easy API Progressive Disclosure (Level 2 Parameters)**
- `forecast()`: `models=` (select specific model IDs), `ensemble=` ('mean', 'weighted', 'median', 'best'), `confidence=` (0.80~0.99 CI level)
- `analyze()`: `features=` (toggle feature extraction), `changepoints=` (toggle detection), `anomalies=` (toggle detection), `anomaly_threshold=` (z-score sensitivity)
- `regress()`: `alpha=` (regularization strength for ridge/lasso), `diagnostics=` (auto-run diagnostics)
- `compare()`: `models=` (compare specific model subset)

**Vectrix Class Level 2 Parameters**
- `Vectrix.forecast()`: `models=`, `ensembleMethod=`, `confidenceLevel=` parameters with full validation
- Ensemble methods: 'mean' (simple average), 'weighted' (inverse-MAPE), 'median', 'best' (single model)
- Confidence interval rescaling from 0.95 default to any level using scipy.stats.norm

**Documentation**
- README.md / README_KR.md: "Philosophy & Roadmap" section with identity, API layers (Level 1-3), roadmap, expansion principles
- CLAUDE.md: Expansion/maintenance principles (API design, engine, speed, accuracy, docs/marketing)
- Updated Easy API examples showing Level 1 and Level 2 usage side by side

### Changed

- `easy.py`: All functions now accept Level 2 parameters with sensible defaults preserving Level 1 behavior
- `vectrix.py`: `forecast()` accepts `models`, `ensembleMethod`, `confidenceLevel` with validation and error messages
- Identity principles updated: added Progressive Disclosure and benchmark-based honesty

[0.0.11]: https://github.com/eddmpython/vectrix/compare/v0.0.10...v0.0.11

## [0.0.10] - 2026-03-04

Research & stability release — 16 DOT improvement experiments (E020~E030, E013~E015), DOT-Hybrid engine OWA 0.885 re-verified, CES combination approach tested and rolled back.

### Added

**Experiment Files (16 new experiments)**
- `modelCreation/013~015`: Novel ensemble models (WDE, RGF, EPE) — all rejected, DOT/CES wall too high
- `modelCreation/020~024`: Fundamentally novel approaches (Compression, Topological, Gravitational, Evolutionary, Causal Entropy) — all rejected, structural decomposition models remain superior
- `modelCreation/025~030`: DOT engine improvement attempts (Multi-Season, DOT+CES Combined, Holdout Selection, Adaptive Theta Bounds, Residual Correction, Engine Verification)
  - E026 DOT+CES Combined showed within-experiment improvement but degraded vs E019 baseline (0.918 vs 0.885) — rolled back
  - E029 Residual Correction catastrophically harmful (Monthly +0.202 regression)
  - All 6 experiments documented with full results in docstrings

### Changed

- `engine/dot.py`: Minor docstring fix (OWA 0.884 → 0.885), code style consistency in fit/predict methods
- `CLAUDE.md`: Added Novel Approaches research section, updated experiment folder range to 001~019

### Key Findings

- DOT-Hybrid standalone remains the strongest engine (AVG OWA 0.885)
- DOT residuals are near white noise — post-processing (residual correction, period re-detection, theta bounds) cannot improve further
- CES combination adds 4.5x speed overhead while degrading accuracy vs baseline
- Model-free approaches (compression, topology, gravity) consistently fail — structural assumptions (trend + seasonality + error decomposition) are essential for forecasting accuracy

[0.0.10]: https://github.com/eddmpython/vectrix/compare/v0.0.9...v0.0.10

## [0.0.9] - 2026-03-03

Accuracy & speed release — DOT-Hybrid model achieves M4 Competition OWA 0.885 (beats #18 Theta 0.897), with 9.8x benchmark speedup from Rust `dot_hybrid_objective`.

### Added

**DOT-Hybrid Mode (M4 OWA: 0.905 → 0.885)**
- Period < 24: 8-way auto-select (2 trend × 2 model × 2 season types) with exponential trend and multiplicative theta line
- Period ≥ 24: Original 3-parameter DOT optimization (preserves Hourly OWA 0.722)
- Automatic mode switching at `_HYBRID_THRESHOLD = 24`
- Yearly OWA 0.797 (world-class, near M4 #1 ES-RNN 0.821)
- NumPy-vectorized combination functions replacing Python for-loops

**Rust Acceleration: `dot_hybrid_objective` (26th function)**
- Full DOT-Hybrid objective: buildThetaLine + golden section alpha optimization + SES filter + combine + MAE — all in Rust
- Golden section search (50 iterations) replaces scipy `minimize_scalar` inside Rust for alpha optimization
- M4 100K benchmark: 16.6 min → 1.7 min (9.8x faster)
- Per-group speedups: Yearly 3x, Quarterly 3.7x, Monthly 4.8x, Weekly 7.7x, Daily 9.8x, Hourly 53x

### Changed

- `engine/dot.py`: `DynamicOptimizedTheta` now includes hybrid mode with backward-compatible API
- `rust/src/lib.rs`: 25 → 26 Rust-accelerated functions
- Version sync: pyproject.toml, Cargo.toml, __init__.py all at 0.0.9

[0.0.9]: https://github.com/eddmpython/vectrix/compare/v0.0.8...v0.0.9

## [0.0.8] - 2026-03-03

Built-in Rust engine release — Rust acceleration expanded to all engines and compiled into every wheel. No `[turbo]` extra, no flags — `pip install vectrix` includes the Rust engine like Polars.

### Added

**Rust Engine Expansion (13 → 25 functions)**
- `garch_filter`, `egarch_filter`, `gjr_garch_filter`: GARCH family negative log-likelihood hot loops
- `tbats_filter`: Fourier harmonic state update loop
- `dtsf_distances`, `dtsf_fit_residuals`: O(n²) sliding window pattern matching (biggest speedup)
- `mstl_extract_seasonal`, `mstl_moving_average`: MSTL seasonal decomposition
- `croston_tsb_filter`: Croston TSB SES update loop
- `esn_reservoir_update`: Echo State Network reservoir state computation O(n×N²)
- `four_theta_fitted`, `four_theta_deseasonalize`: 4Theta fitted values + seasonal decomposition
- All 25 functions compiled into the default wheel — no separate `[turbo]` install needed

**CI/CD: macOS x86_64 wheel**
- Added `macos-13` build target for Intel Mac users
- Now 4 platform builds: Linux x86_64, macOS ARM, macOS x86_64, Windows x86_64

### Changed

**Messaging: "optional Rust turbo" → "built-in Rust engine"**
- `pyproject.toml`: Description updated to reflect built-in Rust engine
- `README.md`: Removed all "optional" Rust language, comparison table updated
- Landing page: Hero, Features, Install, Performance, Numbers sections rewritten
- Docs (EN/KO): Installation guides rewritten — no `[turbo]` extra mentioned
- SEO metadata: All "Rust turbo" → "built-in Rust engine" across meta tags, schema, OG tags

**Version sync**: pyproject.toml, Cargo.toml, __init__.py all at 0.0.8

[0.0.8]: https://github.com/eddmpython/vectrix/compare/v0.0.7...v0.0.8

## [0.0.7] - 2026-03-02

AI integration release — llms.txt for instant AI understanding, MCP server for tool use, Claude Code skills for workflow automation.

### Added

**llms.txt / llms-full.txt**
- `llms.txt`: Structured project overview following the [llms.txt standard](https://llmstxt.org/) — documentation links, quick start, API sections
- `llms-full.txt`: Complete API reference (every class, method, parameter, return type, common mistakes) — AI reads once, understands the full library
- Deployed to GitHub Pages root: `eddmpython.github.io/vectrix/llms.txt` and `llms-full.txt`
- Included in PyPI package for local access

**MCP Server (Model Context Protocol)**
- 10 tools: `forecast_timeseries`, `forecast_csv`, `analyze_timeseries`, `compare_models`, `run_regression`, `detect_anomalies`, `backtest_model`, `list_sample_datasets`, `load_sample_dataset`
- 2 resources: `vectrix://models`, `vectrix://api-reference`
- 2 prompts: `forecast_workflow`, `regression_workflow`
- Compatible with Claude Desktop, Claude Code, and any MCP client
- Setup: `pip install "vectrix[mcp]"` + `claude mcp add`

**Claude Code Skills (3)**
- `vectrix-forecast`: Time series forecasting workflow with full API reference
- `vectrix-analyze`: DNA profiling, anomaly detection, regime analysis
- `vectrix-regress`: R-style regression, diagnostics, variable selection
- Auto-loaded in project directory, invocable via `/vectrix-forecast` etc.

### Changed

- `pyproject.toml`: Added `mcp` optional dependency, `include` for llms.txt/mcp in wheel
- `docs.yml`: Copy llms.txt and llms-full.txt to GitHub Pages deploy directory
- README.md / README_KR.md: Added "AI Integration" section (llms.txt, MCP, Skills)
- CLAUDE.md: Added README update policy (mandatory update after every feature change)

[0.0.7]: https://github.com/eddmpython/vectrix/compare/v0.0.6...v0.0.7

## [0.0.6] - 2026-03-02

Documentation & deployment release — tutorials, showcases, EasyForecastResult enhancements, and unified SvelteKit landing + MkDocs GitHub Pages deployment.

### Added

**EasyForecastResult Enhancements**
- `compare()`: Side-by-side model comparison table with sMAPE, MAPE, RMSE, MAE metrics
- `all_forecasts()`: DataFrame of all valid model forecasts for manual analysis
- Accuracy attributes: `.mape`, `.rmse`, `.mae`, `.smape` on EasyForecastResult for quick access
- `Vectrix._refitAllModels()`: Refit all valid models after best model selection for compare/all_forecasts

**Tutorials (Markdown, 6 topics × 2 languages = 12 files)**
- 01_quickstart: One-line forecasting, result inspection, visualization
- 02_analyze: DNA profiling, feature fingerprinting, changepoint detection
- 03_regression: R-style formula regression, diagnostics, robust methods
- 04_models: 30+ model catalog, manual selection, comparison workflow
- 05_adaptive: Regime detection, self-healing, constraints, forecast DNA
- 06_business: Anomaly detection, what-if scenarios, backtesting, business metrics

**Showcases (marimo interactive notebooks)**
- 03_modelComparison: 30+ model comparison with DNA analysis
- 04_businessIntelligence: Anomaly detection, scenarios, backtesting
- Companion .md pages for GitHub Pages visibility (8 files)

### Changed

**Unified GitHub Pages Deployment**
- SvelteKit landing page now serves at root (`/vectrix/`)
- MkDocs documentation serves at `/vectrix/docs/`
- `docs.yml` workflow builds both SvelteKit + MkDocs and merges into single deployment
- All landing page links updated to point to `/vectrix/docs/` paths
- SvelteKit `paths.base` configured via `BASE_PATH` environment variable
- Header/Footer components use `{base}` import for correct asset paths

**Documentation Navigation**
- mkdocs.yml nav updated with tutorial and showcase sub-pages
- showcase/index and tutorials/index updated with content descriptions
- README.md and README_KR.md updated: 573 tests, compare API, new models, doc links

[0.0.6]: https://github.com/eddmpython/vectrix/compare/v0.0.5...v0.0.6

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
