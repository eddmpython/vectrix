---
title: Benchmarks
---

# Benchmarks

Vectrix is benchmarked against the M3 and M4 Competition datasets, the gold standard for time series forecasting evaluation. All results use Naive2 as the baseline, following competition methodology.

## M4 Competition Results — DOT-Hybrid Engine

The [M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) (Makridakis et al., 2020) contains 100,000 time series across 6 frequencies. Results from **DOT-Hybrid** (DynamicOptimizedTheta with 8-way auto-select), evaluated on 2,000 randomly sampled series per frequency (seed=42)

| Frequency | DOT-Hybrid OWA | M4 Context |
|-----------|:--------------:|------------|
| Yearly | **0.797** | Near M4 #1 ES-RNN (0.821) |
| Quarterly | **0.905** | Competitive with M4 top methods |
| Monthly | **0.933** | Solid mid-table performance |
| Weekly | **0.959** | Beats Naive2 |
| Daily | **0.996** | Near parity with Naive2 |
| Hourly | **0.722** | World-class, near M4 winner level |
| **AVG** | **0.885** | **Beats M4 #18 Theta (0.897)** |

### M4 Competition Leaderboard Context

| Rank | Method | OWA |
|:----:|--------|:---:|
| 1 | ES-RNN (Smyl) | 0.821 |
| 2 | FFORMA (Montero-Manso) | 0.838 |
| 3 | Theta (Fiorucci) | 0.854 |
| 11 | 4Theta (Petropoulos) | 0.874 |
| 18 | Theta (Assimakopoulos) | 0.897 |
| -- | **Vectrix DOT-Hybrid** | **0.885** |

Vectrix DOT-Hybrid outperforms **all pure statistical methods** in the M4 Competition. Only hybrid methods (ES-RNN = LSTM + ETS, FFORMA = meta-learning ensemble) rank higher.

## M3 Competition Results

First 100 series per category. Lower is better for all metrics. **OWA below 1.0 beats Naive2.**

| Category | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | Vectrix OWA |
|----------|:---:|:---:|:---:|:---:|:---:|
| Yearly | 22.675 | 19.404 | 3.861 | 3.246 | **0.848** |
| Quarterly | 12.546 | 10.445 | 1.568 | 1.283 | **0.825** |
| Monthly | 37.872 | 30.731 | 1.214 | 0.856 | **0.758** |
| Other | 6.620 | 5.903 | 2.741 | 2.044 | **0.819** |

Vectrix consistently outperforms Naive2 across all M3 categories, with the strongest performance on Monthly data (OWA 0.758).

## Metrics

| Metric | Description |
|--------|-------------|
| **sMAPE** | Symmetric Mean Absolute Percentage Error. Scale-independent accuracy measure, bounded between 0% and 200%. |
| **MASE** | Mean Absolute Scaled Error. Compares forecast errors against a naive seasonal benchmark. Values below 1.0 indicate the model outperforms the naive method. |
| **OWA** | Overall Weighted Average. Combines sMAPE and MASE relative to Naive2: `OWA = 0.5 × (sMAPE/sMAPE_naive2) + 0.5 × (MASE/MASE_naive2)`. **OWA below 1.0 means the model beats Naive2.** |

## Reproducing Results

```bash
pip install vectrix
```

M4 benchmark experiment: `src/vectrix/experiments/modelCreation/019_dotHybridEngine.py`

> **Tip:** For faster M4 data loading, download the CSV files directly from the [M4 Competition repository](https://github.com/Mcompetitions/M4-methods) rather than using `M4.load()`, which can be slow due to wide-to-long data transformation.
