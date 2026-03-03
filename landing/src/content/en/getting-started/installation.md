---
title: Installation
---

# Installation

Vectrix is a pure Python library with optional native acceleration. Most users are up and running in under 30 seconds.

## Requirements

- **Python 3.10 or higher** (3.11+ recommended for best performance)
- **OS:** Windows, macOS, Linux — all platforms supported including Apple Silicon

## Install

Choose your preferred package manager:

**pip** (most common)

```bash
pip install vectrix
```

**uv** (fastest)

```bash
uv add vectrix
```

**conda / mamba** (from PyPI)

```bash
pip install vectrix
```

This installs Vectrix with its 3 core dependencies (NumPy, pandas, SciPy) and nothing else. No C compiler, no CUDA, no heavy frameworks.

## Optional Extras

Vectrix follows a modular design — install only what you need:

```bash
pip install "vectrix[turbo]"       # Rust acceleration (5-10x speedup, no Rust compiler needed)
pip install "vectrix[numba]"       # Numba JIT acceleration (2-5x speedup)
pip install "vectrix[ml]"          # LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"  # Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[tutorials]"   # Interactive marimo tutorials
pip install "vectrix[all]"         # Everything
```

| Extra | What It Adds | When to Use |
|:------|:-------------|:------------|
| `turbo` | Rust-compiled native extension | Production workloads, large datasets |
| `numba` | JIT-compiled numerical loops | Alternative acceleration without Rust |
| `ml` | LightGBM, XGBoost, scikit-learn | Machine learning model candidates |
| `foundation` | Chronos-2, TimesFM 2.5 | Zero-shot foundation model forecasting |
| `tutorials` | marimo interactive notebooks | Learning and exploration |

## Rust Turbo Mode

The `turbo` extra installs `vectrix-core`, a Rust-compiled native extension that accelerates 13 core forecasting inner loops — ETS state filtering, ARIMA likelihood computation, Theta decomposition, and more. Pre-built binary wheels are available for all major platforms:

- **Linux** x86_64 (manylinux)
- **macOS** x86_64 + Apple Silicon (ARM64)
- **Windows** x86_64
- **Python** 3.10, 3.11, 3.12, 3.13

No Rust compiler is needed. The acceleration is completely transparent — your code doesn't change, it just runs faster. Vectrix auto-detects the native extension at import time and falls back to pure Python if unavailable.

| Component | Without Turbo | With Turbo | Speedup |
|:----------|:-------------|:-----------|:--------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| AutoARIMA fit | 195ms | **35ms** | **5.6x** |

## Verify Installation

After installing, confirm everything works:

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

If you installed `turbo`, check that the native extension loaded:

```python
import vectrix
print(vectrix.__turbo__)  # True if Rust acceleration is active
```

## Core Dependencies

Vectrix is designed to be lightweight. Only 3 packages are required — all widely used, well-maintained scientific Python libraries:

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | >= 1.24.0 | Array operations and linear algebra |
| pandas | >= 2.0.0 | Time series data handling |
| scipy | >= 1.10.0 | Statistical distributions and optimization |
