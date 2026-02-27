# Benchmarks

Vectrix is evaluated against standard time series forecasting competitions (M3, M4) using the **OWA** (Overall Weighted Average) metric, which measures performance relative to the Naive2 seasonal benchmark.

- **OWA < 1.0** → better than Naive2
- **OWA = 1.0** → same as Naive2
- **OWA > 1.0** → worse than Naive2

## M3 Competition Results

The [M3 Competition](https://forecasters.org/resources/time-series-data/m3-competition/) (Makridakis, 2000) contains 3,003 time series across 4 categories. First 100 series per category:

| Category | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|----------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly   | 22.675       | 24.305        | 3.861       | 4.006        | 1.055           |
| Quarterly| 12.546       | 11.316        | 1.568       | 1.366        | **0.887**       |
| Monthly  | 37.872       | 28.461        | 1.214       | 0.751        | **0.685**       |
| Other    | 6.620        | 6.161         | 2.741       | 2.209        | **0.868**       |

Vectrix outperforms Naive2 on **3 out of 4** M3 categories, with M3 Monthly achieving OWA = 0.685.

## M4 Competition Results

The [M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) (Makridakis et al., 2020) contains 100,000 time series across 6 frequencies. First 100 series per frequency:

| Frequency  | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|------------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly     | 13.493       | 12.092        | 4.369       | 3.827        | **0.886**       |
| Quarterly  | 3.714        | 3.329         | 1.244       | 1.052        | **0.871**       |
| Monthly    | 8.943        | 9.386         | 0.923       | 0.999        | 1.066           |
| Weekly     | 10.534       | 7.785         | 0.857       | 0.554        | **0.693**       |
| Daily      | 2.652        | 3.196         | 1.122       | 1.361        | 1.209           |
| Hourly     | 6.814        | 6.524         | 0.987       | 0.973        | **0.972**       |

Vectrix outperforms Naive2 on **4 out of 6** M4 frequencies, with M4 Weekly achieving OWA = 0.693.

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
