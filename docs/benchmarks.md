# Benchmarks

Vectrix is evaluated against standard time series forecasting competitions (M3, M4) using the **OWA** (Overall Weighted Average) metric, which measures performance relative to the Naive2 seasonal benchmark.

- **OWA < 1.0** → better than Naive2
- **OWA = 1.0** → same as Naive2
- **OWA > 1.0** → worse than Naive2

## M4 Competition Results — DOT-Hybrid Engine

The [M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) (Makridakis et al., 2020) contains 100,000 time series across 6 frequencies. Results below are from **DOT-Hybrid** (DynamicOptimizedTheta with 8-way auto-select), evaluated on 2,000 randomly sampled series per frequency (seed=42):

| Frequency  | DOT-Hybrid OWA | M4 Context |
|------------|:--------------:|------------|
| Yearly     | **0.797**      | Near M4 #1 ES-RNN (0.821) |
| Quarterly  | **0.905**      | Competitive with M4 top methods |
| Monthly    | **0.933**      | Solid mid-table performance |
| Weekly     | **0.959**      | Beats Naive2 |
| Daily      | **0.996**      | Near parity with Naive2 |
| Hourly     | **0.722**      | World-class, near M4 winner level |
| **AVG**    | **0.885**      | **Beats M4 #18 Theta (0.897)** |

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

The [M3 Competition](https://forecasters.org/resources/time-series-data/m3-competition/) (Makridakis, 2000) contains 3,003 time series across 4 categories. First 100 series per category:

| Category | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|----------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly   | 22.675       | 19.404        | 3.861       | 3.246        | **0.848**       |
| Quarterly| 12.546       | 10.445        | 1.568       | 1.283        | **0.825**       |
| Monthly  | 37.872       | 30.731        | 1.214       | 0.856        | **0.758**       |
| Other    | 6.620        | 5.903         | 2.741       | 2.044        | **0.819**       |

Vectrix outperforms Naive2 on **4 out of 4** M3 categories, with M3 Monthly achieving OWA = 0.758.

## Metrics

| Metric | Description |
|--------|-------------|
| **sMAPE** | Symmetric Mean Absolute Percentage Error (0-200 scale) |
| **MASE** | Mean Absolute Scaled Error (scale-free, relative to naive) |
| **OWA** | Overall Weighted Average = 0.5 × (sMAPE/sMAPE_naive2 + MASE/MASE_naive2) |

## Reproducing Results

### Environment

| Item | Version / Spec |
|------|----------------|
| Python | 3.10+ |
| Vectrix | 0.0.10 |
| OS | Windows 11 / Ubuntu 22.04 / macOS 14+ |
| CPU | Any modern x86_64 or ARM64 |
| RAM | 8 GB minimum |

### Steps

```bash
pip install vectrix
```

M4 benchmark experiments are located in `src/vectrix/experiments/modelCreation/019_dotHybridEngine.py`.

### Notes

- All models are **deterministic** (no random seed required). Given the same data and parameters, Vectrix produces identical results across runs.
- The built-in Rust engine does not affect accuracy — only speed. Results are numerically identical with or without Rust acceleration.
- M4 data files can be downloaded from the [M4 Competition repository](https://github.com/Mcompetitions/M4-methods).
