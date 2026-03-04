---
title: Installation
---

# Installation

Vectrix ships with a **built-in Rust engine** — like Polars, the Rust extension is compiled into the wheel. No compiler, no extras, just `pip install` and it's fast.

## Requirements

- **Python 3.10 or higher** (3.11+ recommended for best performance)
- **OS:** Windows, macOS, Linux — all platforms supported including Apple Silicon

## Install

Choose your preferred package manager

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

This installs Vectrix with its 3 core dependencies (NumPy, pandas, SciPy) **and the Rust engine**. No C compiler, no CUDA, no heavy frameworks.

## Optional Extras

Vectrix follows a modular design — install only what you need

```bash
pip install vectrix                  # All 30+ models + built-in Rust engine
pip install "vectrix[ml]"            # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"    # + Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[all]"           # Everything
```

| Extra | What It Adds | When to Use |
|:------|:-------------|:------------|
| `ml` | LightGBM, XGBoost, scikit-learn | Machine learning model candidates |
| `foundation` | Chronos-2, TimesFM 2.5 | Zero-shot foundation model forecasting |
| `neural` | NeuralForecast (N-BEATS, N-HiTS, TFT) | Deep learning models |

## Built-in Rust Engine

25 core forecasting hot loops are Rust-accelerated and compiled into every wheel. Pre-built binary wheels are available for all major platforms

- **Linux** x86_64 (manylinux)
- **macOS** Apple Silicon (ARM64) + x86_64
- **Windows** x86_64
- **Python** 3.10, 3.11, 3.12, 3.13

No Rust compiler is needed. The acceleration is completely transparent — your code doesn't change, it just runs faster. Vectrix auto-detects the Rust engine at import time.

| Component | Python Only | With Rust | Speedup |
|:----------|:-----------|:----------|:--------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| DOT fit | 240ms | **10ms** | **24x** |
| ETS filter (hot loop) | 0.17ms | **0.003ms** | **67x** |

## Verify Installation

After installing, confirm everything works

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

Check that the Rust engine loaded

```python
import vectrix
print(vectrix.TURBO_AVAILABLE)  # True if Rust engine is active
```

## Core Dependencies

Vectrix is designed to be lightweight. Only 3 packages are required — all widely used, well-maintained scientific Python libraries

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| numpy | >= 1.24.0 | Array operations and linear algebra |
| pandas | >= 2.0.0 | Time series data handling |
| scipy | >= 1.10.0 | Statistical distributions and optimization |
