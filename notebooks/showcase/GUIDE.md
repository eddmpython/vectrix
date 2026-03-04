# Showcase Notebook Guide

How to create a new showcase notebook for Vectrix.

## Purpose

Each showcase solves a **real-world business problem** end-to-end with interactive Plotly charts.
The goal is twofold:
1. Demonstrate Vectrix capabilities in a specific domain
2. Add a new built-in dataset to `datasets.py` so users can reproduce the showcase instantly

## Workflow

### 1. Pick a Domain & Find Public Data

Choose a domain not yet covered. Find a public dataset with a clear time series signal.

**Good data sources**
- FRED (fred.stlouisfed.org) — macro economics, interest rates, unemployment
- EIA (eia.gov) — energy production, electricity demand, oil prices
- NOAA (noaa.gov) — temperature, precipitation, sea level
- WHO / CDC — disease incidence, flu trends
- US DOT / BTS — air passengers, freight volumes
- World Bank — GDP, population, trade
- Kaggle (public domain / CC0 only) — retail, taxi, bike sharing

**Requirements**
- License must allow redistribution (public domain, CC0, or US government data)
- At least 100 observations
- Clear date + numeric value structure
- Should cover a frequency or pattern not already in `loadSample()`

### 2. Add to datasets.py

Convert the raw data into a deterministic generator function in `src/vectrix/datasets.py`.

**Pattern** (follow existing examples exactly)

```python
def _generateNewDataset() -> pd.DataFrame:
    """
    One-line description (source, frequency, observation count).
    Key pattern characteristics.
    """
    # Hard-code the data or generate from parameters
    # Use np.random.default_rng(SEED) for reproducibility
    # Return pd.DataFrame with "date" column + one value column
    ...
    return pd.DataFrame({"date": dates, "valueColName": values})
```

**Two approaches for data**

A. **Embed real values** — Copy actual public data values into a numpy array.
   Best for small datasets (< 500 rows). Data lives in code, no external download.

B. **Parameterized simulation** — Fit the real data's statistical properties
   (trend slope, seasonal amplitude, noise variance, period) and generate
   synthetic data that matches the pattern. Best for large datasets.

**Registry entry**

```python
"datasetName": {
    "fn": _generateNewDataset,
    "desc": "Description (frequency, N obs) — key pattern",
    "dateCol": "date",
    "valueCol": "valueColName",
    "freq": "monthly",  # monthly, quarterly, yearly, daily, hourly, weekly
},
```

**Naming rules**
- Dataset name: lowercase, short, descriptive (e.g., `"gasPrice"`, `"usGdp"`, `"flu"`)
- Value column: camelCase, domain-specific (e.g., `"priceUsd"`, `"gdpBillions"`, `"cases"`)
- Generator function: `_generateCamelCase()`

### 3. Create the Notebook

File: `notebooks/showcase/NN_snake_case_title.ipynb`

**Required structure**

```
Cell 0 (markdown): Title + Colab badge + description + "What you'll build" list
Cell 1 (code):     !pip install -q vectrix plotly
Cell 2 (code):     Imports + COLORS dict + LAYOUT dict (copy from existing)
Cell 3 (markdown): --- Section 1 header
Cell 4+ (code):    Section implementation
...
Last-1 (markdown): "How to Use" / "Summary" section with adaptation instructions
Last (markdown):    Resources links
```

**Mandatory elements**
- Colab badge linking to this notebook's GitHub path
- `COLORS` and `LAYOUT` dicts (copy from existing showcase — keeps visual consistency)
- At least 3 Plotly charts (interactive, dark theme, publication quality)
- Use `loadSample("newDataset")` — NOT raw data loading
- Business context in every markdown cell (why this matters, how to interpret)
- "How to adapt" section at the end with `pd.read_csv()` example

**Simplicity is key**
- Each code cell should be ≤ 10 lines (excluding chart config)
- No custom helper functions — use Vectrix API directly
- No complex preprocessing — `loadSample()` → `forecast()` → chart
- If a cell gets long, split into "compute" cell + "chart" cell
- A reader should understand every cell in 10 seconds
- Avoid nested loops, list comprehensions with conditionals, or multi-step transforms
- Good: `result = forecast(df, date="date", value="sales", steps=30)`
- Bad: 20-line data cleaning pipeline before the first forecast

**Markdown explanations must be thorough**
- Every section starts with a markdown cell explaining WHY this step matters
- Don't just say "Step 2: Forecast" — explain what forecasting does, what the reader should look for
- After a chart, add a markdown cell interpreting the results (what does this chart tell us?)
- Use business language, not statistical jargon (say "sales will likely grow" not "positive trend coefficient")
- Include practical tips: "If your MAPE is above 15%, consider using more historical data"
- Target audience: a business analyst who has never used a forecasting library
- When showing metrics, explain what "good" and "bad" values look like

**Balance code and visualization**
- Alternate between code cells and markdown/chart cells
- Never have 3+ consecutive code-only cells without a visual break
- Every Vectrix API call should produce either printed output or a chart
- Charts tell the story; code is just the mechanism

**Use `vectrix.viz` for all charts**

The library provides a built-in visualization module. Always use it instead of manual Plotly setup.

```python
from vectrix.viz import forecastChart, dnaRadar, modelHeatmap, forecastReport, analysisReport

# Individual charts
fig = forecastChart(result, historical=df)
fig.show()

# DNA radar
fig = dnaRadar(analysisResult)
fig.show()

# Composite reports
fig = forecastReport(result, historical=df)
fig.show()
```

Available functions:
- `forecastChart()` — forecast line + CI + historical
- `dnaRadar()` — 6-axis DNA radar
- `modelHeatmap()` — model comparison heatmap
- `scenarioChart()` — what-if scenario comparison
- `backtestChart()` — fold MAPE bar chart
- `metricsCard()` — business metrics scorecard
- `forecastReport()` — forecast + metrics composite
- `analysisReport()` — DNA + features composite

All functions return `go.Figure` with brand theme pre-applied. You can further customize with standard Plotly methods.

If you need a custom chart not covered by `vectrix.viz`, follow these standards:
- Dark theme: `template="plotly_dark"`, bg `#0f172a`
- Primary color: `#6366f1` (indigo), accent: `#a855f7` (purple)
- Positive: `#22c55e`, negative: `#ef4444`, warning: `#f59e0b`
- Height: 400-500px
- Horizontal legend below chart: `legend=dict(orientation="h", y=-0.15)`
- Hover templates with formatted numbers
- Or use `from vectrix.viz import applyTheme, COLORS` for consistency

**Vectrix API usage**
- Read `API_SPEC.md` before writing any code
- Use Easy API: `forecast()`, `analyze()`, `compare()`, `regress()`
- Use business module: `AnomalyDetector`, `WhatIfAnalyzer`, `Backtester`, `BusinessMetrics`
- Use adaptive module: `RegimeDetector`, `ForecastDNA`, `ConstraintAwareForecaster`
- Never use methods/attributes that don't exist in API_SPEC.md

### 4. Update References

After creating the notebook and dataset:

1. **API_SPEC.md** — Add new dataset name to `loadSample()` available samples list
2. **docs/tutorials/index.md** — Add row to Showcase Notebooks table
3. **README.md** — Add row to Showcase (Plotly) table

### 5. Verify

```bash
uv run python -c "from vectrix import loadSample; df = loadSample('newName'); print(df.shape)"
uv run python -c "from vectrix import listSamples; print(listSamples())"
```

## Current Coverage

| # | Showcase | Dataset | Frequency | Pattern |
|---|---------|---------|-----------|---------|
| 01 | Sales Dashboard | retail | daily | weekly + annual + holidays |
| 02 | Demand Planning | retail | daily | weekly + annual + holidays |

## Gaps to Fill (Priority)

| Priority | Domain | Frequency | Why |
|----------|--------|-----------|-----|
| High | Macro economics (GDP, CPI) | quarterly/yearly | No quarterly/yearly dataset exists |
| High | Electricity demand | hourly | Real-world hourly with multi-seasonality |
| High | Healthcare (flu/disease) | weekly | Weekly frequency gap |
| Medium | Transportation (air/rail) | monthly | Real monthly with trend + seasonality |
| Medium | Finance (commodity prices) | daily | Non-stationary, volatility clustering |
| Medium | Climate/weather | daily | Long-range annual pattern |
| Low | Manufacturing (production index) | monthly | Industrial use case |
| Low | E-commerce (orders) | daily | Marketing spikes, growth |

## Checklist

Before submitting a new showcase:

- [ ] Public data source with redistribution-compatible license
- [ ] Generator function added to `datasets.py` with `_REGISTRY` entry
- [ ] `loadSample("name")` works and returns correct DataFrame
- [ ] `listSamples()` shows the new dataset
- [ ] Notebook runs top-to-bottom without errors
- [ ] At least 3 interactive Plotly charts
- [ ] Business context in markdown cells
- [ ] "How to adapt" section at the end
- [ ] API_SPEC.md updated
- [ ] docs/tutorials/index.md updated
- [ ] README.md updated
