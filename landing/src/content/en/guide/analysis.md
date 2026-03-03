---
title: Analysis & DNA
---

# Analysis & DNA

**Understand your data before you forecast it.** Vectrix's analysis system extracts 65+ statistical features to build a DNA profile — revealing trend direction, seasonal patterns, volatility regime, structural breaks, and forecasting difficulty. This profile drives automatic model selection.

## Quick Analysis

The `analyze()` function accepts the same input formats as `forecast()` — lists, arrays, DataFrames, Series, or CSV paths:

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
```

## DNA Profile

Every time series has a unique statistical fingerprint — its "DNA." Vectrix extracts 65+ features (autocorrelation structure, Hurst exponent, entropy, volatility clustering, seasonal strength, and more) to create a deterministic profile that drives model selection and difficulty estimation:

```python
dna = report.dna
print(f"Fingerprint: {dna.fingerprint}")
print(f"Difficulty: {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"Category: {dna.category}")
print(f"Recommended: {dna.recommendedModels[:3]}")
```

| Attribute | Description |
|-----------|-------------|
| `fingerprint` | Deterministic hash -- identical data always produces the same value |
| `difficulty` | easy / medium / hard / very_hard |
| `difficultyScore` | 0-100 numeric score |
| `category` | seasonal, trending, volatile, intermittent, etc. |
| `recommendedModels` | Ordered list of optimal models |

## Data Characteristics

The `characteristics` object provides a comprehensive statistical profile — trend direction and strength, seasonal patterns, volatility level, and predictability score:

```python
c = report.characteristics
print(f"Length: {c.length}")
print(f"Period: {c.period}")
print(f"Trend: {c.hasTrend} ({c.trendDirection}, strength {c.trendStrength:.2f})")
print(f"Seasonality: {c.hasSeasonality} (strength {c.seasonalStrength:.2f})")
print(f"Volatility: {c.volatilityLevel} ({c.volatility:.4f})")
print(f"Predictability: {c.predictabilityScore}/100")
print(f"Outliers: {c.outlierCount} ({c.outlierRatio:.1%})")
```

## Changepoints & Anomalies

**Changepoints** are locations where the time series undergoes a structural shift (sudden change in mean, variance, or trend). **Anomalies** are isolated observations that deviate from the expected pattern:

```python
print(f"Changepoints: {report.changepoints}")
print(f"Anomalies: {report.anomalies}")
```

## Quick Report

Run analysis and forecasting in a single call. DNA profiling, feature extraction, model selection, and forecasting — all at once:

```python
from vectrix import quick_report

report = quick_report(df, steps=14)
print(report['summary'])
forecast_result = report['forecast']
analysis_result = report['analysis']
```

## Direct ForecastDNA Access

For lower-level control — such as building custom model selection logic or caching DNA profiles — use the `ForecastDNA` class directly:

```python
from vectrix.adaptive import ForecastDNA

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(profile.fingerprint)
print(profile.recommendedModels)
```

---

**Interactive tutorial:** `marimo run docs/tutorials/en/02_analyze.py`
