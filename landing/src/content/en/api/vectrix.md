---
title: Vectrix Class
---

# Vectrix Class

The full-featured forecasting engine with model comparison and selection.

## `Vectrix(verbose=False)`

Main forecasting class.

### Methods

#### `forecast(data, dateCol=None, valueCol=None, steps=10, period=None)`

Run the full forecasting pipeline.

**Parameters:**
- `data` — pandas DataFrame with date and value columns
- `dateCol` — Date column name (auto-detected if None)
- `valueCol` — Value column name (auto-detected if None)
- `steps` — Number of forecast steps
- `period` — Seasonal period (auto-detected if None)

**Returns:** `ForecastResult`

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
| `.allModelResults` | `dict` | All ModelResult objects |
| `.characteristics` | `DataCharacteristics` | Detected data properties |
| `.flatRisk` | `FlatRiskAssessment` | Flat prediction risk |
| `.warnings` | `list` | Any warnings generated |
