# Analysis & DNA

## Quick Analysis

```python
from vectrix import analyze

report = analyze(df, date="date", value="sales")
```

## DNA Profile

Every time series has a unique "DNA" — a fingerprint based on 65+ statistical features.

```python
dna = report.dna
print(f"Fingerprint: {dna.fingerprint}")
print(f"Difficulty: {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"Category: {dna.category}")
print(f"Recommended: {dna.recommendedModels[:3]}")
```

| Attribute | Description |
|-----------|-------------|
| `fingerprint` | Deterministic hash — identical data always produces the same value |
| `difficulty` | easy / medium / hard / very_hard |
| `difficultyScore` | 0-100 numeric score |
| `category` | seasonal, trending, volatile, intermittent, etc. |
| `recommendedModels` | Ordered list of optimal models |

## Data Characteristics

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

```python
print(f"Changepoints: {report.changepoints}")
print(f"Anomalies: {report.anomalies}")
```

## Quick Report

Combined analysis + forecast in one call:

```python
from vectrix import quick_report

report = quick_report(df, steps=14)
print(report['summary'])
forecast_result = report['forecast']
analysis_result = report['analysis']
```

## Direct ForecastDNA Access

```python
from vectrix.adaptive import ForecastDNA

dna = ForecastDNA()
profile = dna.analyze(data, period=7)
print(profile.fingerprint)
print(profile.recommendedModels)
```

---

**Interactive tutorial:** `marimo run docs/tutorials/en/02_analyze.py`
