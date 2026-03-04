---
title: Types
---

# Types

Core data types and result objects used throughout Vectrix.

## ForecastResult

Main result from `Vectrix.forecast()`. Not the same as `EasyForecastResult` from the Easy API.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether forecast succeeded |
| `predictions` | `np.ndarray` | Forecast values |
| `dates` | `list[str]` | Forecast dates |
| `lower95` | `np.ndarray` | 95% lower bound |
| `upper95` | `np.ndarray` | 95% upper bound |
| `bestModelId` | `str` | Selected model ID |
| `bestModelName` | `str` | Model display name |
| `allModelResults` | `dict[str, ModelResult]` | All model results |
| `characteristics` | `DataCharacteristics` | Data properties |

!!! note "EasyForecastResult vs ForecastResult"
    The Easy API returns `EasyForecastResult` with `.lower` / `.upper`.
    The Vectrix class returns `ForecastResult` with `.lower95` / `.upper95`.

## ModelResult

Per-model result stored in `ForecastResult.allModelResults`.

| Field | Type | Description |
|---|---|---|
| `modelId` | `str` | Model identifier |
| `modelName` | `str` | Display name |
| `isValid` | `bool` | Whether model produced valid output |
| `mape` | `float` | Validation MAPE |
| `rmse` | `float` | Validation RMSE |
| `mae` | `float` | Validation MAE |
| `smape` | `float` | Validation sMAPE |
| `predictions` | `np.ndarray` | Model predictions |
| `lower95` | `np.ndarray` | Lower bound |
| `upper95` | `np.ndarray` | Upper bound |
| `trainingTime` | `float` | Training time (seconds) |
| `flatInfo` | `FlatPredictionInfo` | Flat detection info |

## DataCharacteristics

| Field | Type | Description |
|---|---|---|
| `length` | `int` | Number of observations |
| `period` | `int` | Detected seasonal period |
| `frequency` | `str` | Frequency label (D, W, M, etc.) |
| `hasTrend` | `bool` | Whether trend detected |
| `trendDirection` | `str` | 'increasing', 'decreasing', 'none' |
| `trendStrength` | `float` | 0–1 strength |
| `hasSeasonality` | `bool` | Whether seasonality detected |
| `seasonalStrength` | `float` | 0–1 strength |
| `predictabilityScore` | `float` | 0–100 score |

## DNAProfile

Returned by `analyze().dna` or `ForecastDNA().analyze()`.

| Field | Type | Description |
|---|---|---|
| `features` | `dict[str, float]` | 65+ statistical features |
| `fingerprint` | `str` | 8-char hex hash |
| `difficulty` | `str` | 'easy', 'medium', 'hard', 'very_hard' |
| `difficultyScore` | `float` | 0–100 score |
| `recommendedModels` | `list[str]` | Sorted by fitness |
| `category` | `str` | 'trending', 'seasonal', 'stationary', etc. |
| `summary` | `str` | Natural language summary |

!!! warning "Feature values are inside the `features` dict"
    ```python
    # CORRECT
    dna.features['trendStrength']
    dna.features['seasonalStrength']
    dna.features['hurstExponent']

    # WRONG — AttributeError
    dna.trendStrength
    dna.seasonalStrength
    ```

**Key feature names:** `trendStrength`, `seasonalStrength`, `seasonalPeakPeriod`, `hurstExponent`, `volatility`, `cv`, `skewness`, `kurtosis`, `adfStatistic`, `spectralEntropy`, `approximateEntropy`, `garchEffect`, `volatilityClustering`, `demandDensity`, `nonlinearAutocorr`, `forecastability`, `trendSlope`, `trendDirection`, `trendLinearity`, `trendCurvature`

## ModelInfo

Available model catalog entry.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Model identifier |
| `name` | `str` | Display name |
| `category` | `str` | Model category |
| `description` | `str` | Brief description |
| `strengths` | `list[str]` | What the model is good at |
