# Vectrix API Specification (Single Source of Truth)

> **All code, docs, notebooks, and examples MUST reference this file.**
> When the API changes, update this file FIRST, then propagate to code/docs.

---

## Easy API (from vectrix import ...)

### forecast()

```python
forecast(
    data,                    # str | DataFrame | ndarray | list | Series | dict
    date=None,               # str — date column name
    value=None,              # str — value column name
    steps=30,                # int — forecast horizon
    frequency='auto',        # str — 'auto', 'D', 'W', 'M', 'H'
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
    data,                    # str | DataFrame | ndarray | list | Series | dict
    date=None,               # str
    value=None,              # str
    period=None,             # int | None — seasonal period (auto if None)
    features=True,           # bool
    changepoints=True,       # bool
    anomalies=True,          # bool
    anomalyThreshold=3.0     # float — z-score threshold
) -> EasyAnalysisResult
```

### regress()

```python
regress(
    y=None,                  # ndarray | Series | None (direct mode)
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
| durbinWatson | — | float | Durbin-Watson statistic |

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

---

*Last updated: 2026-03-04*
*When modifying ANY public API, update this file FIRST.*
