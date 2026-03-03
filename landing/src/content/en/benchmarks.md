---
title: Benchmarks
---

# Benchmarks

Vectrix is benchmarked against the M3 and M4 Competition datasets, the gold standard for time series forecasting evaluation. All results use Naive2 as the baseline, following competition methodology.

## Metrics

| Metric | Description |
|--------|-------------|
| **sMAPE** | Symmetric Mean Absolute Percentage Error. Scale-independent accuracy measure, bounded between 0% and 200%. |
| **MASE** | Mean Absolute Scaled Error. Compares forecast errors against a naive seasonal benchmark. Values below 1.0 indicate the model outperforms the naive method. |
| **OWA** | Overall Weighted Average. Combines sMAPE and MASE relative to Naive2: `OWA = 0.5 × (sMAPE/sMAPE_naive2) + 0.5 × (MASE/MASE_naive2)`. **OWA below 1.0 means the model beats Naive2.** |

## M3 Competition Results

First 100 series per category. Lower is better for all metrics. **OWA below 1.0 beats Naive2.**

| Category | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | Vectrix OWA |
|----------|:---:|:---:|:---:|:---:|:---:|
| Yearly | 22.675 | 19.404 | 3.861 | 3.246 | **0.848** |
| Quarterly | 12.546 | 10.445 | 1.568 | 1.283 | **0.825** |
| Monthly | 37.872 | 30.731 | 1.214 | 0.856 | **0.758** |
| Other | 6.620 | 5.903 | 2.741 | 2.044 | **0.819** |

Vectrix consistently outperforms Naive2 across all M3 categories, with the strongest performance on Monthly data (OWA 0.758).

## M4 Competition Results

First 100 series per frequency. Lower is better for all metrics. **OWA below 1.0 beats Naive2.**

| Frequency | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | Vectrix OWA |
|-----------|:---:|:---:|:---:|:---:|:---:|
| Yearly | 13.493 | 13.540 | 4.369 | 4.125 | **0.974** |
| Quarterly | 3.714 | 3.120 | 1.244 | 0.937 | **0.797** |
| Monthly | 8.943 | 9.175 | 0.923 | 0.875 | **0.987** |
| Weekly | 10.534 | 8.598 | 0.857 | 0.563 | **0.737** |
| Daily | 2.652 | 3.254 | 1.122 | 1.331 | 1.207 |
| Hourly | 6.814 | 6.759 | 0.987 | 1.006 | 1.006 |

Vectrix beats Naive2 on 4 of 6 M4 frequencies. Weekly data shows the largest improvement (OWA 0.737). Daily and Hourly remain active areas of improvement (see below).

## Understanding the Results

**Strong performance (OWA well below 1.0):**
- M3 Monthly (0.758) and M3 Quarterly (0.825) demonstrate robust model selection on mid-frequency data.
- M4 Weekly (0.737) benefits from DTSF and MSTL multi-seasonal pattern capture.

**Competitive performance (OWA near 1.0):**
- M4 Yearly (0.974) and Monthly (0.987) show Vectrix is competitive but has room for improvement on these frequencies.

**Known weaknesses:**
- M4 Daily (OWA 1.207): High noise ratio and multi-seasonal patterns (day-of-week + annual) challenge the current model selection.
- M4 Hourly (OWA 1.006): Multi-level seasonality (hourly + daily + weekly) requires further MSTL optimization.

These weaknesses are documented transparently and are active research areas. See the model creation experiments in the repository for ongoing work.

## Running Benchmarks

### Reproducing with Vectrix 0.0.7

Install Vectrix

```bash
pip install vectrix==0.0.7
```

Run the M3 benchmark (first 100 series per category)

```python
from vectrix import Vectrix
from datasetsforecast.m3 import M3

trainDict, testDict = M3.load(directory="./data")

categories = ["Yearly", "Quarterly", "Monthly", "Other"]
for cat in categories:
    trainData = trainDict[cat]
    testData = testDict[cat]

    totalSmape = 0
    totalMase = 0
    nSeries = min(100, len(trainData))

    for i in range(nSeries):
        y = trainData[i]
        h = len(testData[i])
        vx = Vectrix()
        vx.fit(y)
        pred = vx.predict(steps=h)

    print(f"{cat}: sMAPE={totalSmape/nSeries:.3f}, MASE={totalMase/nSeries:.3f}")
```

Run the M4 benchmark (first 100 series per frequency)

```python
from datasetsforecast.m4 import M4

trainDict, testDict = M4.load(directory="./data")

frequencies = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
for freq in frequencies:
    trainData = trainDict[freq]
    testData = testDict[freq]

    nSeries = min(100, len(trainData))
    for i in range(nSeries):
        y = trainData[i]
        h = len(testData[i])
        vx = Vectrix()
        vx.fit(y)
        pred = vx.predict(steps=h)

    print(f"{freq}: sMAPE=..., MASE=...")
```

> **Note:** Full M4 benchmarks (100,000 series) take several hours. The 100-series subset provides representative results in a few minutes.

### Dependencies for Benchmarks

```bash
pip install vectrix datasetsforecast
```

> **Tip:** For faster M4 data loading, download the CSV files directly from the [M4 Competition repository](https://github.com/Mcompetitions/M4-methods) rather than using `M4.load()`, which can be slow due to wide-to-long data transformation.
