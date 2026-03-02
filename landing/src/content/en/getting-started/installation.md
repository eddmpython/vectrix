---
title: Installation
---

# Installation

## Requirements

- Python 3.10+
- OS: Windows, macOS, Linux

## Install

**pip**

```bash
pip install vectrix
```

**uv**

```bash
uv add vectrix
```

## Optional Extras

```bash
pip install "vectrix[turbo]"       # Rust acceleration (5-10x speedup, no Rust compiler needed)
pip install "vectrix[numba]"       # Numba JIT acceleration (2-5x speedup)
pip install "vectrix[ml]"          # LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"  # Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[tutorials]"   # Interactive marimo tutorials
pip install "vectrix[all]"         # Everything
```

## Rust Turbo Mode

The `turbo` extra installs `vectrix-core`, a Rust-compiled native extension that accelerates core forecasting loops by 5-10x. Pre-built wheels are available for:

- Linux (x86_64, manylinux)
- macOS (x86_64 + Apple Silicon ARM)
- Windows (x86_64)
- Python 3.10, 3.11, 3.12, 3.13

No Rust compiler is needed. The acceleration is transparent -- your code doesn't change, it just runs faster.

| Component | Without Turbo | With Turbo | Speedup |
|:----------|:-------------|:-----------|:--------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| AutoARIMA fit | 195ms | **35ms** | **5.6x** |

## Verify

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

## Core Dependencies

Vectrix has only 3 required dependencies:

| Package | Minimum Version |
|---------|----------------|
| numpy | >= 1.24.0 |
| pandas | >= 2.0.0 |
| scipy | >= 1.10.0 |
