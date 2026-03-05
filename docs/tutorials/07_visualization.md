---
title: "Tutorial 07 — Visualization"
---

# Tutorial 07 — Visualization

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eddmpython/vectrix/blob/master/notebooks/tutorials/07_visualization.ipynb)

**Numbers tell the story, but charts sell it.** Vectrix's visualization module (`vectrix.viz`) provides publication-quality interactive charts built on Plotly — designed to work seamlessly with every Vectrix result object.

All charts follow a consistent pattern: pass a Vectrix result, get back a Plotly `go.Figure` you can display inline, save to HTML, or embed in dashboards.

## Setup

Install the visualization extra (Plotly >= 5.0):

```bash
pip install vectrix[viz]
```

All chart functions live under `vectrix.viz`:

```python
from vectrix.viz import (
    forecastChart, dnaRadar, modelHeatmap,
    scenarioChart, backtestChart, metricsCard,
    forecastReport, analysisReport,
)
```

## Forecast Chart

The most common visualization — plot predictions with confidence intervals, optionally overlaying historical data for context.

```python
from vectrix import forecast, loadSample
from vectrix.viz import forecastChart

df = loadSample("airline")
result = forecast(df, steps=24)

fig = forecastChart(result, historical=df)
fig.show()
```

The chart automatically detects date and value columns from the historical DataFrame. The title defaults to `"Forecast — {model} (MAPE {mape}%)"`.

### Custom Title

```python
fig = forecastChart(result, historical=df, title="Airline Passenger Forecast — Next 24 Months")
fig.show()
```

### Light Theme

Every chart function accepts `theme="light"` for white-background presentations:

```python
fig = forecastChart(result, historical=df, theme="light")
fig.show()
```

### Without Historical Data

If you only want to show the forecast period:

```python
fig = forecastChart(result)
fig.show()
```

## DNA Radar Chart

Visualize a time series' statistical fingerprint — 6 normalized features on a polar chart. Instantly shows whether your data is trend-dominated, seasonal, volatile, or forecastable.

```python
from vectrix import analyze, loadSample
from vectrix.viz import dnaRadar

df = loadSample("airline")
analysis = analyze(df)

fig = dnaRadar(analysis)
fig.show()
```

The 6 radar axes are:

| Axis | Feature Key | Interpretation |
|------|-------------|---------------|
| Trend | `trendStrength` | Linear trend dominance (0=flat, 1=strong trend) |
| Seasonality | `seasonalStrength` | Seasonal pattern strength |
| Memory | `hurstExponent` | Long-range dependency (&gt;0.5 = persistent, &lt;0.5 = mean-reverting) |
| Vol. Clustering | `volatilityClustering` | GARCH-like variance clustering |
| Nonlinear | `nonlinearAutocorr` | Nonlinear autocorrelation strength |
| Forecastability | `forecastability` | Overall predictability (1 - spectral entropy) |

**Tip:** Compare radar shapes across different datasets to understand why some are easier to forecast than others. A high-forecastability series with strong seasonality behaves very differently from a volatile series with low memory.

## Model Heatmap

After running `compare()`, visualize which models perform best across different error metrics. The heatmap normalizes each column (min-max) so green = best and red = worst, regardless of the metric's scale.

```python
from vectrix import compare, loadSample
from vectrix.viz import modelHeatmap

df = loadSample("retail")
comparison = compare(df, steps=30)

fig = modelHeatmap(comparison, top=8)
fig.show()
```

The `comparisonDf` is expected to be sorted by MAPE (best first), which is exactly what `compare()` returns.

### Show Fewer Models

```python
fig = modelHeatmap(comparison, top=5, title="Top 5 Models — Retail Sales")
fig.show()
```

## Scenario Chart

Visualize what-if scenarios from `WhatIfAnalyzer`. The baseline scenario is drawn solid; alternatives are dashed for easy comparison.

```python
import numpy as np
from vectrix import forecast
from vectrix.business import WhatIfAnalyzer
from vectrix.viz import scenarioChart

data = np.random.randn(200).cumsum() + 500
result = forecast(data, steps=30)

analyzer = WhatIfAnalyzer()
scenarios = analyzer.analyze(
    result.predictions, data,
    [
        {"name": "Optimistic", "trend_change": 0.1},
        {"name": "Pessimistic", "trend_change": -0.15},
        {"name": "Shock", "shock_at": 10, "shock_magnitude": -0.3, "shock_duration": 5},
    ]
)

fig = scenarioChart(scenarios)
fig.show()
```

### With Forecast Dates

When your forecast has actual dates (from a DataFrame source), pass them for a proper time axis:

```python
import pandas as pd

dates = pd.date_range("2026-01-01", periods=30, freq="D")
fig = scenarioChart(scenarios, dates=dates, title="Revenue Scenarios — Q1 2026")
fig.show()
```

Without `dates`, the X-axis shows numeric step indices (1, 2, 3...).

## Backtest Chart

Visualize fold-by-fold performance from `Backtester.run()`. Each fold gets a bar; the best fold is green, worst is red, and a dashed line shows the average.

```python
from vectrix.business import Backtester
from vectrix.engine.ets import AutoETS
import numpy as np

data = np.random.randn(300).cumsum() + 200

bt = Backtester(nFolds=5, horizon=14, strategy="expanding")
result = bt.run(data, lambda: AutoETS())

fig = backtestChart(result)
fig.show()
```

### Switch to RMSE

By default the chart shows MAPE. To show RMSE instead:

```python
fig = backtestChart(result, metric="rmse", title="Backtest — RMSE by Fold")
fig.show()
```

## Metrics Card

A four-indicator scorecard for business metrics — Accuracy, Bias, WAPE, and MASE. Colors turn green when a metric meets its threshold, red when it doesn't.

```python
from vectrix.business import BusinessMetrics
import numpy as np

actual = np.array([100, 120, 110, 130, 140, 125, 135, 150, 145, 155])
predicted = np.array([105, 115, 112, 128, 145, 120, 138, 148, 140, 160])

metrics = BusinessMetrics()
result = metrics.calculate(actual, predicted)

from vectrix.viz import metricsCard

fig = metricsCard(result)
fig.show()
```

### Custom Thresholds

Default thresholds: Accuracy &gt;= 95%, |Bias| &lt; 3%, WAPE &lt; 5%, MASE &lt; 1.0. Override any of them:

```python
fig = metricsCard(result, thresholds={
    "accuracy": 90,
    "bias": 5,
    "wape": 8,
    "mase": 1.2,
})
fig.show()
```

## Composite Reports

For dashboards and presentations, composite reports combine multiple charts into a single figure with subplots.

### Forecast Report

A 2-row layout: forecast chart with confidence intervals (top 75%) and error metric bars (bottom 25%). Shows MAPE, RMSE, MAE, and sMAPE.

```python
from vectrix import forecast, loadSample
from vectrix.viz import forecastReport

df = loadSample("airline")
result = forecast(df, steps=24)

fig = forecastReport(result, historical=df)
fig.show()
```

The metric bars are color-coded: MAPE and sMAPE turn green below 10%, yellow below 20%, red above 20%.

### Analysis Report

A 2x2 layout: DNA radar chart (top-left), feature importance bars (top-right), and difficulty score indicator (bottom).

```python
from vectrix import analyze, loadSample
from vectrix.viz import analysisReport

df = loadSample("airline")
analysis = analyze(df)

fig = analysisReport(analysis)
fig.show()
```

The bottom indicator shows the difficulty score (0-100) with the data category and number of changepoints/anomalies detected.

## Theme Reference

All chart functions accept `theme="dark"` (default) or `theme="light"`.

### Dark Theme (Default)

Deep navy background (`#0f172a`), light text (`#f8fafc`). Best for Jupyter notebooks and dark-mode dashboards.

### Light Theme

White background (`#ffffff`), dark text (`#0f172a`). Best for printed reports and light-mode presentations.

### Using Theme Constants

Access brand colors directly for custom Plotly charts:

```python
from vectrix.viz import COLORS, LIGHT_COLORS, PALETTE, HEIGHT, applyTheme
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[1, 2, 3], y=[10, 20, 15],
    line=dict(color=COLORS["primary"], width=2),
))

fig = applyTheme(fig, title="Custom Chart", height=HEIGHT["chart"])
fig.show()
```

### Color Reference

| Key | Dark | Light | Usage |
|-----|------|-------|-------|
| `primary` | `#06b6d4` | `#06b6d4` | Main brand color (cyan) |
| `accent` | `#8b5cf6` | `#8b5cf6` | Secondary highlight (purple) |
| `positive` | `#10b981` | `#059669` | Good values, below threshold |
| `negative` | `#ef4444` | `#dc2626` | Bad values, above threshold |
| `warning` | `#f59e0b` | `#d97706` | Caution, averages |
| `muted` | `#94a3b8` | `#64748b` | Historical data, secondary |
| `bg` | `#0f172a` | `#ffffff` | Plot background |
| `bgDarker` | `#020617` | `#f8fafc` | Paper background |
| `card` | `#1e293b` | `#f1f5f9` | Card/panel background |
| `text` | `#f8fafc` | `#0f172a` | Primary text |
| `textMuted` | `#94a3b8` | `#64748b` | Secondary text |
| `border` | `#334155` | `#e2e8f0` | Borders, dividers |
| `grid` | `rgba(255,255,255,0.05)` | `rgba(0,0,0,0.05)` | Grid lines |

### Height Constants

Standard heights for consistent layouts:

```python
from vectrix.viz import HEIGHT

HEIGHT["chart"]     # 480 — individual charts
HEIGHT["card"]      # 240 — metrics cards
HEIGHT["report"]    # 640 — forecast report
HEIGHT["analysis"]  # 680 — analysis report
HEIGHT["small"]     # 360 — compact charts
```

## Saving Charts

Every chart returns a standard Plotly `go.Figure`. Use Plotly's built-in export methods:

```python
fig = forecastChart(result, historical=df)

fig.write_html("forecast.html")

fig.write_image("forecast.png", width=1200, height=600, scale=2)

fig.write_json("forecast.json")
```

**Note:** `write_image()` requires the `kaleido` package: `pip install kaleido`.

## HTML Dashboard

The `dashboard()` function generates a self-contained HTML report that combines all analysis into a single page — data profile, forecast results, model comparison, and interactive charts. No Plotly CDN dependency required in the output.

### Basic Usage

```python
from vectrix import forecast, analyze, compare, loadSample
from vectrix.viz import dashboard

df = loadSample("airline")

result = forecast(df, steps=12)
analysis = analyze(df)
comparison = compare(df, steps=12)

report = dashboard(
    forecast=result,
    analysis=analysis,
    comparison=comparison,
    historical=df,
)
report.show()  # Opens in browser (terminal) or displays inline (Jupyter)
```

### Custom Title

```python
report = dashboard(
    forecast=result,
    analysis=analysis,
    comparison=comparison,
    historical=df,
    title="Airline Passengers — Monthly Forecast",
)
report.show()
```

### Save to File

```python
report.save("forecast_report.html")
```

The HTML file is fully self-contained (embedded Plotly JS + inline CSS). Share it by email, upload to a web server, or open locally in any browser.

### Report Sections

The dashboard follows a narrative flow:

1. **Overview** — observation count, frequency, difficulty score, forecast horizon
2. **Data Profile** — DNA feature bars, descriptive statistics, key insights, recommended models
3. **Forecast Results** — accuracy KPIs (MAPE, RMSE, MAE, sMAPE), forecast details, model comparison table
4. **Visualizations** — interactive forecast chart with confidence intervals, DNA radar chart

### Partial Reports

Every parameter is optional. Generate a profile-only report (no forecast):

```python
report = dashboard(analysis=analysis, historical=df, title="Data Profile Report")
report.save("profile.html")
```

Or a forecast-only report (no DNA analysis):

```python
report = dashboard(forecast=result, historical=df, title="Forecast Report")
report.save("forecast.html")
```

### Terminal Progress

When running from a terminal (not Jupyter), the dashboard builder prints progress to stderr:

```
[vectrix] Building header... (0.0s)
[vectrix] Analyzing data profile... (0.0s)
[vectrix] Computing performance metrics... (0.0s)
[vectrix] Rendering charts... (0.2s)
[vectrix] Dashboard ready (0.3s, 48,244 bytes)
```

## Combining Individual Charts

For maximum flexibility, build your own dashboards by combining individual chart functions. Each returns a standard Plotly `go.Figure`:

```python
from vectrix import forecast, analyze, compare, loadSample
from vectrix.viz import (
    forecastChart, dnaRadar, modelHeatmap,
    forecastReport, analysisReport,
)

df = loadSample("airline")

result = forecast(df, steps=24)
analysis = analyze(df)
comparison = compare(df, steps=24)

fig1 = forecastReport(result, historical=df, title="Airline Forecast Report")
fig1.show()

fig2 = analysisReport(analysis, title="Airline DNA Analysis")
fig2.show()

fig3 = modelHeatmap(comparison, top=8, title="Model Comparison — Airline")
fig3.show()
```

Each figure is independent — display them in separate notebook cells, save to separate HTML files, or embed in a web application.

---
