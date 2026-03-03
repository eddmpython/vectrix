---
title: "Tutorial 02 — Analysis & DNA"
---

# Tutorial 02 — Analysis & DNA

Every time series has a unique fingerprint. Vectrix extracts 65+ statistical features to build a "DNA profile" that reveals the nature of your data and recommends the best forecasting approach.

## Basic Analysis

The `analyze()` function accepts the same input formats as `forecast()`

```python
import pandas as pd
from vectrix import analyze

df = pd.read_csv("sales.csv")
report = analyze(df, date="date", value="sales")

print(report.summary())
```

**Expected output:**

```
=== Vectrix Analysis Report ===

DNA Profile:
  Fingerprint: a3f7c2d1
  Difficulty: medium (42/100)
  Category: seasonal
  Recommended Models: ['AutoETS', 'MSTL', 'Theta']

Data Characteristics:
  Length: 365 observations
  Period: 7 (weekly)
  Trend: Yes (upward, strength 0.72)
  Seasonality: Yes (strength 0.85)
  Volatility: low (0.0312)

Changepoints: [89, 201]
Anomalies: [15, 156, 298]
```

You can also pass raw lists or arrays

```python
from vectrix import analyze

data = [120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215]
report = analyze(data)
print(report.summary())
```

## DNA Profile

The DNA profile is the core of Vectrix's intelligence. Access it through `report.dna`

```python
dna = report.dna

print(f"Fingerprint: {dna.fingerprint}")
print(f"Difficulty: {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"Category: {dna.category}")
print(f"Recommended: {dna.recommendedModels[:5]}")
```

**Expected output:**

```
Fingerprint: a3f7c2d1
Difficulty: medium (42/100)
Category: seasonal
Recommended: ['AutoETS', 'MSTL', 'Theta', 'DOT', 'AutoCES']
```

### DNA Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `fingerprint` | `str` | Deterministic hash -- identical data always produces the same value |
| `difficulty` | `str` | `easy`, `medium`, `hard`, or `very_hard` |
| `difficultyScore` | `float` | Numeric difficulty score (0-100, higher = harder) |
| `category` | `str` | `seasonal`, `trending`, `volatile`, `intermittent`, `stationary` |
| `recommendedModels` | `list[str]` | Ordered list of optimal models for this data |

The fingerprint is deterministic: the same data always produces the same hash. This makes it useful for caching and reproducibility.

## Changepoints

Changepoints are locations where the statistical properties of the time series shift (mean, variance, or trend)

```python
print(f"Changepoints at indices: {report.changepoints}")
```

**Expected output:**

```
Changepoints at indices: [89, 201]
```

These correspond to positions in your data where structural breaks occurred -- for example, a policy change, a product launch, or a market shift.

## Anomalies

Anomaly indices mark individual observations that deviate significantly from the expected pattern

```python
print(f"Anomaly indices: {report.anomalies}")
print(f"Number of anomalies: {len(report.anomalies)}")
```

**Expected output:**

```
Anomaly indices: [15, 156, 298]
Number of anomalies: 3
```

## Data Characteristics

The `characteristics` object provides detailed statistical properties

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

**Expected output:**

```
Length: 365
Period: 7
Trend: True (upward, strength 0.72)
Seasonality: True (strength 0.85)
Volatility: low (0.0312)
Predictability: 78/100
Outliers: 3 (0.8%)
```

### Characteristics Reference

| Attribute | Type | Description |
|-----------|------|-------------|
| `length` | `int` | Number of observations |
| `period` | `int` | Detected seasonal period |
| `hasTrend` | `bool` | Whether trend is present |
| `trendDirection` | `str` | `upward`, `downward`, or `none` |
| `trendStrength` | `float` | Trend strength (0-1) |
| `hasSeasonality` | `bool` | Whether seasonality is present |
| `seasonalStrength` | `float` | Seasonal strength (0-1) |
| `volatility` | `float` | Coefficient of variation |
| `volatilityLevel` | `str` | `low`, `medium`, or `high` |
| `predictabilityScore` | `int` | 0-100, higher = more predictable |
| `outlierCount` | `int` | Number of detected outliers |
| `outlierRatio` | `float` | Fraction of outliers |

## Full Summary

The `summary()` method combines all analysis components into a single formatted report

```python
full_report = report.summary()
print(full_report)
```

This includes the DNA profile, characteristics, changepoints, anomalies, and model recommendations in a human-readable format.

## Extracted Features

Access the raw 65+ features as a dictionary

```python
features = report.features
for key, value in list(features.items())[:10]:
    print(f"  {key}: {value}")
```

**Expected output:**

```
  trendStrength: 0.72
  seasonalStrength: 0.85
  acf1: 0.91
  hurstExponent: 0.78
  volatilityClustering: 0.15
  nonlinearAutocorr: 0.23
  demandDensity: 1.0
  seasonalPeakPeriod: 7
  entropy: 2.34
  stability: 0.88
```

## Practical Usage: Analyze Before Forecasting

Use analysis results to make informed forecasting decisions

```python
from vectrix import analyze, forecast

report = analyze(df, date="date", value="sales")

if report.dna.difficulty == "very_hard":
    print("Warning: This series is very hard to forecast.")
    print(f"Consider using: {report.dna.recommendedModels[:3]}")

if len(report.changepoints) > 0:
    print(f"Structural breaks detected at {report.changepoints}.")
    print("Recent data may be more relevant than older data.")

if report.characteristics.hasSeasonality:
    period = report.characteristics.period
    print(f"Seasonal period: {period}")
    result = forecast(df, date="date", value="sales", steps=period)
else:
    result = forecast(df, date="date", value="sales", steps=14)
```

## Direct ForecastDNA Access

For lower-level control, use `ForecastDNA` directly

```python
from vectrix import ForecastDNA
import numpy as np

dna = ForecastDNA()
data = np.random.randn(200).cumsum() + 100
profile = dna.analyze(data, period=7)

print(f"Fingerprint: {profile.fingerprint}")
print(f"Difficulty: {profile.difficulty} ({profile.difficultyScore:.0f}/100)")
print(f"Recommended: {profile.recommendedModels}")
```

## Complete Example

```python
import pandas as pd
from vectrix import analyze

df = pd.read_csv("monthly_revenue.csv")
report = analyze(df, date="month", value="revenue")

print("=== DNA ===")
print(f"Category: {report.dna.category}")
print(f"Difficulty: {report.dna.difficulty}")
print(f"Top models: {report.dna.recommendedModels[:3]}")

print()
print("=== Characteristics ===")
c = report.characteristics
print(f"Trend: {c.trendDirection} (strength {c.trendStrength:.2f})")
print(f"Seasonal: period={c.period}, strength={c.seasonalStrength:.2f}")
print(f"Predictability: {c.predictabilityScore}/100")

print()
print("=== Structural Breaks ===")
print(f"Changepoints: {report.changepoints}")
print(f"Anomalies: {report.anomalies}")
```

---
