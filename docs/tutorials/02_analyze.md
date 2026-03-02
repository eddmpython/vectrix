# Tutorial 02 — Analysis & DNA

**Understand your data before forecasting.**

Vectrix's `analyze()` function profiles any time series in one line — detecting difficulty, category, seasonality, changepoints, anomalies, and recommended models.

## 1. Basic Analysis

```python
import numpy as np
import pandas as pd
from vectrix import analyze

np.random.seed(42)
n = 200
t = np.arange(n, dtype=np.float64)
values = 100 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 7) + np.random.normal(0, 3, n)

df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=n, freq="D"),
    "value": values,
})

report = analyze(df)
```

Just like `forecast()`, column detection is automatic. You can also pass lists, arrays, or CSV paths.

## 2. DNA Profile

Every time series has a unique "DNA" — a fingerprint that summarizes its statistical personality.

```python
dna = report.dna

print(f"Category:    {dna.category}")
print(f"Difficulty:  {dna.difficulty} ({dna.difficultyScore:.0f}/100)")
print(f"Fingerprint: {dna.fingerprint}")
print(f"Recommended: {', '.join(dna.recommendedModels[:3])}")
```

```
Category:    trend-seasonal
Difficulty:  easy (23/100)
Fingerprint: TS-E023-P007-V012
Recommended: auto_ces, dot, four_theta
```

The DNA tells you:

- **Category** — What type of series this is (trend, seasonal, stationary, intermittent, …)
- **Difficulty** — How hard this series is to forecast (0=trivial, 100=extremely hard)
- **Fingerprint** — A unique code encoding the series characteristics
- **Recommended Models** — Which models are likely to perform best

## 3. Changepoints

Changepoints are moments where the series behavior shifts abruptly — a new trend, a level jump, or a volatility change.

```python
print(f"Changepoints found: {len(report.changepoints)}")
print(f"Locations: {report.changepoints.tolist()}")
```

```
Changepoints found: 2
Locations: [67, 134]
```

These indices point to rows in your data where structural breaks occurred.

## 4. Anomalies

Anomalies are individual data points that deviate significantly from the expected pattern (3-sigma rule).

```python
print(f"Anomalies found: {len(report.anomalies)}")
if len(report.anomalies) > 0:
    print(f"Locations: {report.anomalies.tolist()}")
```

```
Anomalies found: 1
Locations: [142]
```

## 5. Data Characteristics

Low-level properties of the series:

```python
c = report.characteristics

print(f"Length:         {c.length}")
print(f"Period:         {c.period}")
print(f"Frequency:      {c.frequency}")
print(f"Has Trend:      {c.hasTrend} ({c.trendDirection}, strength={c.trendStrength:.2f})")
print(f"Has Seasonality:{c.hasSeasonality} (strength={c.seasonalStrength:.2f})")
print(f"Predictability: {c.predictabilityScore:.0f}/100")
```

```
Length:         200
Period:         7
Frequency:      D
Has Trend:      True (increasing, strength=0.85)
Has Seasonality:True (strength=0.92)
Predictability: 78/100
```

## 6. Full Summary

Get everything in one formatted report:

```python
print(report.summary())
```

```
=======================================================
        Vectrix Time Series Analysis Report
=======================================================

  [DNA Analysis]
    Trend-seasonal series with weekly cycle
    Category: trend-seasonal
    Forecast Difficulty: easy (23.0/100)
    Fingerprint: TS-E023-P007-V012
    Recommended Models: auto_ces, dot, four_theta

  [Changepoint Detection]
    Changepoints found: 2
    Locations: [67, 134]

  [Anomaly Detection]
    Anomalies: 1
    Locations: [142]

  [Data Characteristics]
    Length: 200
    Period: 7
    Frequency: D
    Trend: increasing (strength: 0.85)
    Seasonality: present (strength: 0.92)
    Predictability: 78.0/100
=======================================================
```

## 7. Practical Usage: Analyze Before Forecasting

Use analysis to understand your data, then forecast with confidence:

```python
from vectrix import analyze, forecast

report = analyze(df)

if report.dna.difficulty == "hard":
    print("Warning: This series is hard to predict. Use more data if possible.")

result = forecast(df, steps=14)
print(f"DNA recommended: {report.dna.recommendedModels[:3]}")
print(f"Vectrix selected: {result.model}")
```

## 8. Comparing Multiple Series

Profile several series to understand their differences:

```python
from vectrix import analyze

series_list = {
    "stable": [100 + np.random.normal(0, 2) for _ in range(100)],
    "trending": [100 + 0.5 * i + np.random.normal(0, 3) for i in range(100)],
    "volatile": [100 + 10 * np.sin(i / 5) + np.random.normal(0, 15) for i in range(100)],
}

for name, data in series_list.items():
    r = analyze(data)
    print(f"{name:>10}: {r.dna.difficulty:>6} ({r.dna.difficultyScore:5.1f}/100)  category={r.dna.category}")
```

```
    stable:   easy ( 12.3/100)  category=stationary
  trending: medium ( 35.7/100)  category=trend-stationary
  volatile:   hard ( 72.1/100)  category=volatile
```

## 9. Result Object Reference

| Attribute | Type | Description |
|---|---|---|
| `.dna` | `DNAProfile` | DNA profile (difficulty, category, fingerprint, recommendedModels) |
| `.dna.difficulty` | `str` | 'easy', 'medium', 'hard' |
| `.dna.difficultyScore` | `float` | 0–100 score |
| `.dna.category` | `str` | Series type classification |
| `.dna.fingerprint` | `str` | Unique fingerprint code |
| `.dna.recommendedModels` | `list` | Recommended model IDs |
| `.changepoints` | `np.ndarray` | Changepoint indices |
| `.anomalies` | `np.ndarray` | Anomaly indices |
| `.features` | `dict` | Extracted statistical features |
| `.characteristics` | `DataCharacteristics` | Length, period, frequency, trend, seasonality |
| `.summary()` | `str` | Formatted analysis report |

---

**Next:** [Tutorial 03 — Regression](03_regression.md) — R-style formula regression with full diagnostics
