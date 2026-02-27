# Benchmarks

Vectrix is evaluated against standard time series forecasting competitions (M3, M4) using the **OWA** (Overall Weighted Average) metric, which measures performance relative to the Naive2 seasonal benchmark.

- **OWA < 1.0** → better than Naive2
- **OWA = 1.0** → same as Naive2
- **OWA > 1.0** → worse than Naive2

## M3 Competition Results

The [M3 Competition](https://forecasters.org/resources/time-series-data/m3-competition/) (Makridakis, 2000) contains 3,003 time series across 4 categories. First 100 series per category:

| Category | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|----------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly   | 22.675       | 19.404        | 3.861       | 3.246        | **0.848**       |
| Quarterly| 12.546       | 10.445        | 1.568       | 1.283        | **0.825**       |
| Monthly  | 37.872       | 30.731        | 1.214       | 0.856        | **0.758**       |
| Other    | 6.620        | 5.903         | 2.741       | 2.044        | **0.819**       |

Vectrix outperforms Naive2 on **4 out of 4** M3 categories, with M3 Monthly achieving OWA = 0.758.

## M4 Competition Results

The [M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) (Makridakis et al., 2020) contains 100,000 time series across 6 frequencies. First 100 series per frequency:

| Frequency  | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|------------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly     | 13.493       | 13.540        | 4.369       | 4.125        | **0.974**       |
| Quarterly  | 3.714        | 3.120         | 1.244       | 0.937        | **0.797**       |
| Monthly    | 8.943        | 9.175         | 0.923       | 0.875        | **0.987**       |
| Weekly     | 10.534       | 8.598         | 0.857       | 0.563        | **0.737**       |
| Daily      | 2.652        | 3.254         | 1.122       | 1.331        | 1.207           |
| Hourly     | 6.814        | 6.759         | 0.987       | 1.006        | 1.006           |

Vectrix outperforms Naive2 on **4 out of 6** M4 frequencies, with M4 Weekly achieving OWA = 0.737.

## Metrics

| Metric | Description |
|--------|-------------|
| **sMAPE** | Symmetric Mean Absolute Percentage Error (0-200 scale) |
| **MASE** | Mean Absolute Scaled Error (scale-free, relative to naive) |
| **OWA** | Overall Weighted Average = 0.5 × (sMAPE/sMAPE_naive2 + MASE/MASE_naive2) |

## Running Benchmarks

```bash
# M3 Competition
python benchmarks/runM3.py --cat M3Month --n 100
python benchmarks/runM3.py --all --n 50

# M4 Competition
python benchmarks/runM4.py --freq Monthly --n 100
python benchmarks/runM4.py --all --n 50
```

### Available Categories

**M3**: `M3Year`, `M3Quart`, `M3Month`, `M3Other`

**M4**: `Yearly`, `Quarterly`, `Monthly`, `Weekly`, `Daily`, `Hourly`

## Reproducing Results

```bash
git clone https://github.com/eddmpython/vectrix.git
cd vectrix
pip install -e .
python benchmarks/runM3.py --all --n 100
python benchmarks/runM4.py --all --n 100
```

Results are saved to `benchmarks/m3Results.csv` and `benchmarks/m4Results.csv`.
