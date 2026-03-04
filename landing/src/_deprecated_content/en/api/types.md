---
title: Types
---

# Types

Core data types and result objects used throughout Vectrix.

## ForecastResult

Main result from `Vectrix.forecast()`.

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether forecast succeeded |
| `predictions` | `np.ndarray` | Forecast values |
| `dates` | `list[str]` | Forecast dates |
| `lower95` | `np.ndarray` | 95% lower bound |
| `upper95` | `np.ndarray` | 95% upper bound |
| `bestModelId` | `str` | Selected model ID |
| `bestModelName` | `str` | Model display name |
| `allModelResults` | `dict` | All model results |
| `characteristics` | `DataCharacteristics` | Data properties |

## ModelResult

Per-model result.

| Field | Type | Description |
|---|---|---|
| `modelId` | `str` | Model identifier |
| `modelName` | `str` | Display name |
| `isValid` | `bool` | Whether model produced valid output |
| `mape` | `float` | Validation MAPE |
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

## ModelInfo

Available model catalog entry.

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Model identifier |
| `name` | `str` | Display name |
| `category` | `str` | Model category |
| `description` | `str` | Brief description |
| `strengths` | `list[str]` | What the model is good at |
