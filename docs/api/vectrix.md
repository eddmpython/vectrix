---
title: Vectrix Class
---

# Vectrix Class

The full-featured forecasting engine with model comparison and selection.

## `Vectrix(locale='ko_KR', verbose=False, nJobs=-1)`

Main forecasting class.

### Methods

#### `forecast(df, dateCol, valueCol, steps=30, trainRatio=0.8, models=None, ensembleMethod=None, confidenceLevel=0.95)`

Run the full forecasting pipeline.

**Parameters:**

- `df` — pandas DataFrame with date and value columns
- `dateCol` — Date column name (camelCase)
- `valueCol` — Value column name (camelCase)
- `steps` — Number of forecast steps (default: 30)
- `trainRatio` — Train/test split ratio (default: 0.8)
- `models` — list[str] | None — Model IDs to evaluate
- `ensembleMethod` — str | None — Ensemble strategy (camelCase)
- `confidenceLevel` — float — Confidence interval level (default: 0.95)

**Returns:** `ForecastResult`

#### `analyze(df, dateCol, valueCol)`

Run data analysis pipeline.

**Returns:** `dict` — `{'characteristics': DataCharacteristics, 'flatRisk': FlatRiskAssessment}`

!!! note "Vectrix class does NOT have"
    `detectRegimes()`, `fit()`, `healForecast()` — use the Adaptive API for these.

### ForecastResult

| Attribute | Type | Description |
|---|---|---|
| `.success` | `bool` | Whether forecasting succeeded |
| `.predictions` | `np.ndarray` | Final forecast values |
| `.dates` | `list` | Forecast date strings |
| `.lower95` | `np.ndarray` | 95% lower bound |
| `.upper95` | `np.ndarray` | 95% upper bound |
| `.bestModelId` | `str` | Selected model ID |
| `.bestModelName` | `str` | Selected model display name |
| `.allModelResults` | `dict[str, ModelResult]` | All ModelResult objects |
| `.characteristics` | `DataCharacteristics` | Detected data properties |
| `.flatRisk` | `FlatRiskAssessment` | Flat prediction risk |
| `.warnings` | `list` | Any warnings generated |
