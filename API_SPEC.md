# Vectrix API Specification (Single Source of Truth)

> **All code, docs, notebooks, and examples MUST reference this file.**
> When the API changes, update this file FIRST, then propagate to code/docs.

---

## Easy API (from vectrix import ...)

### forecast()

```python
forecast(
    data,                    # str | DataFrame | ndarray | list | tuple | Series | dict
    date=None,               # str — date column name
    value=None,              # str — value column name
    steps=30,                # int — forecast horizon
    verbose=False,           # bool
    models=None,             # list[str] | None — model IDs
    ensemble=None,           # str | None — 'mean', 'weighted', 'median', 'best'
    confidence=0.95          # float — 0.80, 0.90, 0.95, 0.99
) -> EasyForecastResult
```

**Available model IDs:** `'dot'`, `'auto_ets'`, `'auto_arima'`, `'auto_ces'`, `'four_theta'`, `'auto_mstl'`, `'tbats'`, `'theta'`, `'dtsf'`, `'esn'`, `'garch'`, `'croston'`, `'ets_aan'`, `'ets_aaa'`, `'naive'`, `'mean'`, `'rwd'`, `'window_avg'`, `'egarch'`, `'gjr_garch'`, `'seasonal_naive'`, `'mstl'`

### analyze()

```python
analyze(
    data,                    # str | DataFrame | ndarray | list | tuple | Series | dict
    date=None,               # str
    value=None,              # str
    period=None,             # int | None — seasonal period (auto if None)
    features=True,           # bool
    changepoints=True,       # bool
    anomalies=True,          # bool
    anomalyThreshold=3.0,    # float — z-score threshold
    anomaly_threshold=None   # float | None — snake_case alias for anomalyThreshold
) -> EasyAnalysisResult
```

### regress()

```python
regress(
    y=None,                  # ndarray | Series | str | None (direct mode)
    X=None,                  # ndarray | DataFrame | None (direct mode)
    data=None,               # DataFrame | None (formula mode)
    formula=None,            # str | None — "y ~ x1 + x2"
    method='ols',            # str — 'ols', 'ridge', 'lasso', 'huber', 'quantile'
    summary=True,            # bool — auto-print summary
    alpha=None,              # float | None — regularization strength
    diagnostics=False        # bool — auto-run diagnostics
) -> EasyRegressionResult
```

### compare()

```python
compare(
    data,                    # str | DataFrame | ndarray | list | Series | dict
    date=None,               # str
    value=None,              # str
    steps=30,                # int
    verbose=False,           # bool
    models=None              # list[str] | None
) -> pd.DataFrame           # !! Returns DataFrame directly, NOT a Result object
```

**Returned DataFrame columns:** `model`, `mape`, `rmse`, `mae`, `smape`, `time_ms`, `selected`

### quickReport()

```python
quickReport(
    data, date=None, value=None, steps=30
) -> dict                   # !! Returns dict, NOT a Result object
```

**Returned dict keys:** `'forecast'` (EasyForecastResult), `'analysis'` (EasyAnalysisResult), `'summary'` (str)

**Alias:** `quick_report` = `quickReport` (backward compatibility)

### loadSample() / listSamples()

```python
loadSample(name: str) -> pd.DataFrame
listSamples() -> pd.DataFrame
```

**Available samples:** `'airline'`, `'retail'`, `'stock'`, `'temperature'`, `'energy'`, `'web'`, `'intermittent'`

**Column names per sample:**
| Sample | date col | value col |
|--------|----------|-----------|
| airline | date | passengers |
| retail | date | sales |
| stock | date | close |
| temperature | date | temperature |
| energy | date | consumption_kwh |
| web | date | pageviews |
| intermittent | date | demand |

---

## Result Objects

### EasyForecastResult

**Attributes:**
| Name | Type | Description |
|------|------|-------------|
| predictions | np.ndarray | Forecast values |
| dates | list[str] | Forecast dates |
| lower | np.ndarray | Lower CI |
| upper | np.ndarray | Upper CI |
| model | str | Best model name |
| mape | float | MAPE % |
| rmse | float | RMSE |
| mae | float | MAE |
| smape | float | sMAPE |
| models | list[str] | All valid model names (sorted by MAPE) |

**Methods:**
| Method | Alias | Returns | Description |
|--------|-------|---------|-------------|
| summary() | — | str | Text summary |
| toDataframe() | to_dataframe() | DataFrame | date, prediction, lower95, upper95 |
| compare() | — | DataFrame | All models ranked by MAPE |
| allForecasts() | all_forecasts() | DataFrame | date + one col per model |
| describe() | — | DataFrame | .describe() style stats |
| toCsv(path) | to_csv(path) | self | Save to CSV |
| toJson(path=None) | to_json(path=None) | str | JSON string or save to file |
| save(path) | — | self | Alias for toJson(path) |
| plot() | — | Figure | matplotlib plot (optional dep) |

**NOT available:** ~~`.table()`~~

### EasyAnalysisResult

**Attributes:**
| Name | Type | Description |
|------|------|-------------|
| dna | DNAProfile | DNA profile object |
| changepoints | np.ndarray | Array of **int indices** (NOT dicts!) |
| anomalies | np.ndarray | Array of **int indices** (NOT dicts!) |
| features | dict | Statistical features dict |
| characteristics | DataCharacteristics | Data characteristics |

**Methods:**
| Method | Returns |
|--------|---------|
| summary() | str |

**IMPORTANT — anomalies/changepoints are int arrays, NOT dict lists!**
```python
# CORRECT
for idx in analysis.anomalies:
    print(f"Anomaly at index {idx}")

# WRONG — will crash
for a in analysis.anomalies:
    print(a['index'], a['value'])  # TypeError!
```

### EasyRegressionResult

**Attributes (camelCase is primary, snake_case aliases available):**
| Primary | Alias | Type | Description |
|---------|-------|------|-------------|
| coefficients | — | np.ndarray | Including intercept |
| pvalues | — | np.ndarray | P-values |
| rSquared | r_squared | float | R² |
| adjRSquared | adj_r_squared | float | Adjusted R² |
| fStat | f_stat | float | F-statistic |
| durbinWatson | durbin_watson | float | Durbin-Watson statistic |

**Methods:**
| Method | Returns |
|--------|---------|
| summary() | str |
| diagnose() | str |
| predict(X, interval, alpha) | DataFrame |

### DNAProfile

**Attributes:**
| Name | Type | Description |
|------|------|-------------|
| features | dict[str, float] | 65+ statistical features |
| fingerprint | str | 8-char hex hash |
| difficulty | str | 'easy', 'medium', 'hard', 'very_hard' |
| difficultyScore | float | 0-100 |
| recommendedModels | list[str] | Sorted by fitness |
| category | str | 'trending', 'seasonal', 'stationary', etc. |
| summary | str | Natural language summary |

**IMPORTANT — trendStrength etc. are inside features dict, NOT direct attributes!**
```python
# CORRECT
dna.features['trendStrength']
dna.features['seasonalStrength']
dna.features['hurstExponent']
dna.features['seasonalPeakPeriod']

# WRONG — will crash
dna.trendStrength    # AttributeError!
dna.seasonalStrength # AttributeError!
dna.noiseLevel       # AttributeError!
dna.isStationary     # AttributeError! (use dna.features['adfStatistic'])
```

**Key feature names (from features dict):**
`trendStrength`, `seasonalStrength`, `seasonalPeakPeriod`, `hurstExponent`, `volatility`, `cv`, `skewness`, `kurtosis`, `adfStatistic`, `spectralEntropy`, `approximateEntropy`, `garchEffect`, `volatilityClustering`, `demandDensity`, `nonlinearAutocorr`, `forecastability`, `trendSlope`, `trendDirection`, `trendLinearity`, `trendCurvature`

---

## Vectrix Class (Level 2 API)

```python
vx = Vectrix(locale='ko_KR', verbose=False, nJobs=-1)
```

### Vectrix.forecast()

```python
vx.forecast(
    df,                      # DataFrame
    dateCol='date',          # str (camelCase!)
    valueCol='value',        # str (camelCase!)
    steps=30,                # int
    trainRatio=0.8,          # float
    models=None,             # list[str] | None
    ensembleMethod=None,     # str | None (camelCase!)
    confidenceLevel=0.95     # float (camelCase!)
) -> ForecastResult          # Raw result, NOT EasyForecastResult
```

### Vectrix.analyze()

```python
vx.analyze(
    df, dateCol, valueCol
) -> dict                    # {'characteristics': DataCharacteristics, 'flatRisk': FlatRiskAssessment}
```

**NOTE:** Vectrix class does NOT have: ~~detectRegimes()~~, ~~fit()~~, ~~healForecast()~~

### ForecastResult (raw, from Vectrix class)

| Name | Type |
|------|------|
| success | bool |
| predictions | np.ndarray |
| dates | list[str] |
| lower95 | np.ndarray |
| upper95 | np.ndarray |
| bestModelId | str |
| bestModelName | str |
| allModelResults | dict[str, ModelResult] |
| characteristics | DataCharacteristics |

---

## Engine Models

All models follow the same interface:

```python
model.fit(y)                            # y: np.ndarray, returns self
predictions, lower, upper = model.predict(steps)  # all np.ndarray
model.refit(newData)                    # re-fit with cached hyperparams, returns self
```

### Import paths

```python
from vectrix.engine.ets import AutoETS
from vectrix.engine.arima import AutoARIMA, ARIMAModel
from vectrix.engine.theta import OptimizedTheta
from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ces import AutoCES
from vectrix.engine.mstl import AutoMSTL
from vectrix.engine.tbats import AutoTBATS
from vectrix.engine.garch import GARCHModel, EGARCHModel, GJRGARCHModel
from vectrix.engine.croston import AutoCroston
from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
from vectrix.engine.dtsf import DynamicTimeScanForecaster
from vectrix.engine.esn import EchoStateForecaster
from vectrix.engine.baselines import NaiveModel, SeasonalNaiveModel, MeanModel, RandomWalkDrift, WindowAverage
```

### Model catalog

| modelId | Class | needsPeriod | minData | Best for |
|---------|-------|-------------|---------|----------|
| `auto_ets` | AutoETS | Yes | 20 | Stable patterns, short-term |
| `auto_arima` | AutoARIMA | No | 30 | Stationary with complex autocorrelation |
| `theta` | OptimizedTheta | Yes | 10 | Simple trend extrapolation |
| `dot` | DynamicOptimizedTheta | Yes | 10 | General purpose (M4 OWA 0.848) |
| `auto_ces` | AutoCES | Yes | 20 | Nonlinear, complex seasonality |
| `auto_mstl` | AutoMSTL | No | 50 | Multiple seasonality |
| `mstl` | MSTLDecomposition | Yes | 50 | Multiple seasonality (explicit period) |
| `tbats` | AutoTBATS | Yes | 30 | Complex multi-seasonal |
| `four_theta` | AdaptiveThetaEnsemble | Yes | 10 | M4-validated ensemble (Yearly OWA 0.879) |
| `dtsf` | DynamicTimeScanForecaster | No | 30 | Pattern matching, hourly data |
| `esn` | EchoStateForecaster | No | 20 | Ensemble diversity (not standalone) |
| `garch` | GARCHModel | No | 50 | Financial volatility |
| `egarch` | EGARCHModel | No | 50 | Asymmetric volatility |
| `gjr_garch` | GJRGARCHModel | No | 50 | Leverage effect |
| `croston` | AutoCroston | No | 10 | Intermittent/lumpy demand |
| `ets_aan` | ETSModel('A','A','N') | Yes | 10 | Trending data |
| `ets_aaa` | ETSModel('A','A','A') | Yes | 20 | Seasonal data |
| `naive` | NaiveModel | No | 2 | Baseline |
| `seasonal_naive` | SeasonalNaiveModel | Yes | 14 | Seasonal baseline |
| `mean` | MeanModel | No | 2 | Baseline |
| `rwd` | RandomWalkDrift | No | 5 | Trending data baseline |
| `window_avg` | WindowAverage | Yes | 5 | Stable data baseline |

### Registry API

```python
from vectrix.engine.registry import getRegistry, getModelSpec, listModelIds, createModel, getModelInfo

getRegistry()              # -> Dict[str, ModelSpec]
getModelSpec('dot')        # -> ModelSpec or None
listModelIds()             # -> List[str]
createModel('dot', period=12)  # -> model instance
getModelInfo()             # -> backward-compatible MODEL_INFO dict
```

---

## Business Intelligence

```python
from vectrix.business import (
    AnomalyDetector, ForecastExplainer, WhatIfAnalyzer,
    Backtester, BusinessMetrics, ReportGenerator, HTMLReportGenerator
)
```

### AnomalyDetector

```python
detector = AnomalyDetector()
result = detector.detect(
    y,                    # np.ndarray
    method='auto',        # 'zscore', 'iqr', 'seasonal', 'rolling', 'auto'
    threshold=3.0,        # float
    period=1              # int — seasonal period
) -> AnomalyResult
```

**AnomalyResult attributes:** `indices`, `scores`, `method`, `threshold`, `nAnomalies`, `anomalyRatio`, `details`

### ForecastExplainer

```python
explainer = ForecastExplainer()
result = explainer.explain(
    y,                    # np.ndarray — historical data
    predictions,          # np.ndarray — forecast values
    period=7,             # int
    locale='ko'           # str — 'ko' or 'en'
) -> dict
```

**Returned dict keys:** `drivers`, `narrative`, `decomposition`, `confidence`, `summary`

### WhatIfAnalyzer

```python
analyzer = WhatIfAnalyzer()
results = analyzer.analyze(
    basePredictions,      # np.ndarray
    historicalData,       # np.ndarray
    scenarios,            # List[dict] — each with 'name', 'trend_change', etc.
    period=7              # int
) -> List[ScenarioResult]

summary = analyzer.compareSummary(results)  # -> str
```

**Scenario dict keys:** `name`, `trend_change`, `seasonal_multiplier`, `shock_at`, `shock_magnitude`, `shock_duration`, `level_shift`

**ScenarioResult attributes:** `name`, `predictions`, `baselinePredictions`, `difference`, `percentChange`, `impact`

### Backtester

```python
bt = Backtester(
    nFolds=5,             # int
    horizon=30,           # int
    strategy='expanding', # 'expanding' or 'sliding'
    minTrainSize=50,      # int
    stepSize=None         # int | None
)
result = bt.run(y, modelFactory=AutoETS)  # -> BacktestResult
summary = bt.summary(result)              # -> str
```

**BacktestResult attributes:** `nFolds`, `avgMAPE`, `avgRMSE`, `avgMAE`, `avgSMAPE`, `avgBias`, `mapeStd`, `folds`, `bestFold`, `worstFold`

### BusinessMetrics

```python
metrics = BusinessMetrics()
result = metrics.calculate(actual, predicted)  # -> dict
```

**Returned dict keys:** `bias`, `biasPercent`, `trackingSignal`, `wape`, `mase`, `overForecastRatio`, `underForecastRatio`, `forecastAccuracy`, `fillRateImpact`

### ReportGenerator / HTMLReportGenerator

```python
rg = ReportGenerator(locale='ko')
report = rg.generate(historicalData, predictions, lower95, upper95, period=7, modelName='Vectrix', dates=None)  # -> dict

html = HTMLReportGenerator()
path = html.generate(historicalData, predictions, lower95, upper95, modelName='Auto', title='Report', outputPath='report.html')  # -> str (file path)
```

---

## Prediction Intervals

```python
from vectrix.intervals import ConformalInterval, BootstrapInterval
from vectrix.intervals.distribution import ForecastDistribution, DistributionFitter, empiricalCRPS
```

### ConformalInterval

```python
ci = ConformalInterval(
    method='split',           # 'split' or 'jackknife'
    coverageLevel=0.95,       # float
    calibrationRatio=0.2      # float
)
ci.calibrate(y, modelFactory, steps=1)  # returns self
lower, upper = ci.predict(pointPredictions)
```

### BootstrapInterval

```python
bi = BootstrapInterval(
    nBoot=100,                # int
    coverageLevel=0.95        # float
)
bi.calibrate(y, modelFactory, steps=1)  # returns self
lower, upper = bi.predict(pointPredictions)
```

### DistributionFitter

```python
fitter = DistributionFitter()
dist = fitter.fit(residuals)  # -> ForecastDistribution
q50 = dist.quantile(0.5)
crps = dist.crps(actual)
score = empiricalCRPS(actual, samples)
```

---

## Hierarchical Reconciliation

```python
from vectrix.hierarchy import BottomUp, TopDown, MinTrace
```

### BottomUp

```python
bu = BottomUp()
reconciled = bu.reconcile(
    bottomForecasts,      # np.ndarray [nBottom, steps]
    summingMatrix         # np.ndarray [nTotal, nBottom]
) -> np.ndarray           # [nTotal, steps]
```

### TopDown

```python
td = TopDown(method='proportions')  # 'proportions' or 'forecast_proportions'
reconciled = td.reconcile(
    topForecast,          # np.ndarray [steps]
    proportions,          # np.ndarray [nBottom]
    summingMatrix         # np.ndarray [nTotal, nBottom]
) -> np.ndarray           # [nTotal, steps]

proportions = TopDown.computeProportions(historicalBottom)  # static method
```

### MinTrace

```python
mt = MinTrace(method='ols')  # 'ols' or 'wls'
reconciled = mt.reconcile(
    forecasts,            # np.ndarray [nTotal, steps]
    summingMatrix,        # np.ndarray [nTotal, nBottom]
    residuals=None        # np.ndarray [nTotal, T] — required for WLS
) -> np.ndarray           # [nTotal, steps]

S = MinTrace.buildSummingMatrix(structure)  # static, {parent: [children]}
```

---

## Pipeline System

```python
from vectrix.pipeline import (
    ForecastPipeline, Differencer, LogTransformer, BoxCoxTransformer,
    Scaler, Deseasonalizer, Detrend, OutlierClipper, MissingValueImputer
)
```

### Transformers

All transformers implement: `fit(y)`, `transform(y)`, `inverseTransform(y)`, `fitTransform(y)`

| Transformer | Constructor | Description |
|-------------|------------|-------------|
| Differencer | `Differencer(d=1)` | d-th order differencing |
| LogTransformer | `LogTransformer(shift=None)` | log(1+y), auto-shift for negatives |
| BoxCoxTransformer | `BoxCoxTransformer(lmbda=None)` | Auto Box-Cox lambda |
| Scaler | `Scaler(method='zscore')` | 'zscore' or 'minmax' |
| Deseasonalizer | `Deseasonalizer(period=7)` | Remove seasonal component |
| Detrend | `Detrend()` | Remove linear trend |
| OutlierClipper | `OutlierClipper(factor=3.0)` | IQR-based clipping |
| MissingValueImputer | `MissingValueImputer(method='linear')` | 'linear', 'mean', 'ffill' |

### ForecastPipeline

```python
pipe = ForecastPipeline([
    ("log", LogTransformer()),
    ("deseason", Deseasonalizer(period=12)),
    ("model", AutoETS()),
])
pipe.fit(y)
pred, lower, upper = pipe.predict(steps=12)
```

**Methods:** `fit(y)`, `predict(steps)`, `transform(y)`, `inverseTransform(y)`, `getStep(name)`, `getParams()`, `listSteps()`

---

## Datasets

```python
from vectrix import loadSample, listSamples
```

### loadSample(name) -> DataFrame

| name | frequency | rows | date col | value col |
|------|-----------|------|----------|-----------|
| `'airline'` | monthly | 144 | date | passengers |
| `'retail'` | daily | 730 | date | sales |
| `'stock'` | business_daily | 252 | date | close |
| `'temperature'` | daily | 1095 | date | temperature |
| `'energy'` | hourly | 720 | date | consumption_kwh |
| `'web'` | daily | 180 | date | pageviews |
| `'intermittent'` | daily | 365 | date | demand |

### listSamples() -> DataFrame

Returns DataFrame with columns: `name`, `description`, `valueCol`, `frequency`, `rows`

---

## Naming Convention Summary

| Layer | Parameters | Attributes | Methods |
|-------|-----------|------------|---------|
| easy.py functions | snake_case (`date=`, `value=`, `steps=`) + camelCase (`anomalyThreshold`) | — | snake_case (`forecast`, `analyze`) + camelCase (`quickReport`) |
| EasyForecastResult | — | camelCase-ish (`predictions`, `model`) | camelCase primary (`toDataframe`, `allForecasts`) + snake_case alias |
| EasyRegressionResult | — | camelCase primary (`rSquared`, `adjRSquared`, `durbinWatson`) + snake_case alias | snake_case (`summary`, `diagnose`) |
| DNAProfile | — | camelCase (`difficultyScore`, `recommendedModels`) | — |
| Vectrix class | camelCase (`dateCol`, `valueCol`, `ensembleMethod`) | camelCase | camelCase (`setProgressCallback`) |
| Internal types | camelCase | camelCase | — |

---

## Adaptive API (from vectrix import ...)

### RegimeDetector

```python
from vectrix import RegimeDetector
rd = RegimeDetector(nRegimes=2)
result = rd.detect(values)  # np.ndarray
# result.regimeStats: list[dict] with 'mean', 'std', 'size'
# result.labels: np.ndarray
```

### ConstraintAwareForecaster

```python
from vectrix import ConstraintAwareForecaster, Constraint
caf = ConstraintAwareForecaster()
result = caf.apply(
    predictions, lower95, upper95,
    constraints=[Constraint('non_negative', {})]
)
# result.predictions, result.lower95, result.upper95
```

**Constraint types:** `'non_negative'`, `'range'`, `'sum_constraint'`, `'yoy_change'`, `'monotone'`, `'capacity'`, `'ratio'`, `'custom'`

### ForecastDNA

```python
from vectrix import ForecastDNA
dna = ForecastDNA()
profile = dna.analyze(values, period=7)  # -> DNAProfile
```

## Utility API (from vectrix import ...)

### TimeSeriesCrossValidator

```python
from vectrix import TimeSeriesCrossValidator

cv = TimeSeriesCrossValidator(nSplits=5, horizon=30, strategy='expanding', minTrainSize=50, stepSize=None)
splits = cv.split(y)           # -> List[Tuple[ndarray, ndarray]]
result = cv.evaluate(y, modelFactory, period=7)  # -> dict
```

**evaluate() return dict keys:** `'mape'`, `'rmse'`, `'mae'`, `'smape'`, `'foldResults'`, `'nFolds'`

## Visualization API (from vectrix.viz import ...)

> **Optional dependency.** Install with: `pip install vectrix[viz]` (requires Plotly >= 5.0)

### Individual Charts

```python
from vectrix.viz import forecastChart, dnaRadar, modelHeatmap, scenarioChart, backtestChart, metricsCard
```

#### forecastChart()

```python
forecastChart(
    forecastResult,          # EasyForecastResult
    historical=None,         # pd.DataFrame | None — auto-detects date/value columns
    title=None,              # str | None — auto: "Forecast — {model} (MAPE {mape}%)"
    theme="dark"             # str — 'dark' or 'light'
) -> go.Figure
```

#### dnaRadar()

```python
dnaRadar(
    analysisResult,          # EasyAnalysisResult
    title=None,              # str | None — auto: "DNA — {category} ({difficulty}, {score}/100)"
    theme="dark"             # str
) -> go.Figure               # Polar chart with 6 features: Trend, Seasonality, Memory, Vol.Clustering, Nonlinear, Forecastability
```

#### modelHeatmap()

```python
modelHeatmap(
    comparisonDf,            # pd.DataFrame — from compare()
    top=10,                  # int — number of top models
    title=None,              # str | None
    theme="dark"             # str
) -> go.Figure               # Heatmap with min-max normalized errors (green=best, red=worst)
```

#### scenarioChart()

```python
scenarioChart(
    scenarios,               # list[ScenarioResult] — from WhatIfAnalyzer.analyze()
    dates=None,              # list | pd.DatetimeIndex | None — if None, uses numeric steps
    title=None,              # str | None
    theme="dark"             # str
) -> go.Figure               # Baseline=solid, scenarios=dashed
```

#### backtestChart()

```python
backtestChart(
    backtestResult,          # BacktestResult — from Backtester.run()
    metric="mape",           # str — 'mape' or 'rmse'
    title=None,              # str | None
    theme="dark"             # str
) -> go.Figure               # Bar per fold + average hline, best=green, worst=red
```

#### metricsCard()

```python
metricsCard(
    metricsDict,             # dict — from BusinessMetrics.calculate()
    title=None,              # str | None
    thresholds=None,         # dict | None — custom thresholds
    theme="dark"             # str
) -> go.Figure               # 4 indicator cards: Accuracy, Bias, WAPE, MASE
```

**Default thresholds:** `{'accuracy': 95, 'bias': 3, 'wape': 5, 'mase': 1.0}`. Values beyond threshold turn red.

### Composite Reports

```python
from vectrix.viz import forecastReport, analysisReport

forecastReport(
    forecastResult,          # EasyForecastResult
    historical=None,         # pd.DataFrame | None
    title=None,              # str | None
    theme="dark"             # str
) -> go.Figure               # 2-row: forecast line chart (75%) + 4 metric bars MAPE/RMSE/MAE/sMAPE (25%)

analysisReport(
    analysisResult,          # EasyAnalysisResult
    title=None,              # str | None
    theme="dark"             # str
) -> go.Figure               # 2x2: DNA radar (top-left) + feature bars (top-right) + difficulty indicator (bottom)
```

### Theme Utilities

```python
from vectrix.viz import COLORS, LIGHT_COLORS, PALETTE, LAYOUT, HEIGHT, applyTheme
```

| Export | Type | Description |
|--------|------|-------------|
| `COLORS` | dict | 10 dark theme colors: primary `#6366f1`, accent `#a855f7`, positive `#22c55e`, negative `#ef4444`, warning `#f59e0b`, muted `#94a3b8`, bg `#0f172a`, card `#1e293b`, text `#f1f5f9`, grid `rgba(255,255,255,0.06)` |
| `LIGHT_COLORS` | dict | 10 light theme colors: same keys, adjusted values (bg `#ffffff`, text `#0f172a`) |
| `PALETTE` | list | 10 cycling colors for multi-series charts |
| `LAYOUT` | dict | Plotly layout defaults (dark theme, Inter font, margins) |
| `HEIGHT` | dict | Standard heights: `chart` 450, `card` 220, `report` 600, `analysis` 650, `small` 350 |

```python
applyTheme(
    fig,                     # go.Figure
    title=None,              # str | None
    height=450,              # int
    theme="dark"             # str — 'dark' or 'light'
) -> go.Figure               # Applies brand theme, legend, grid styling
```

---

*Last updated: 2026-03-05*
*When modifying ANY public API, update this file FIRST.*
