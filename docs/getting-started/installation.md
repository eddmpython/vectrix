# Installation

## Requirements

- Python 3.10+
- OS: Windows, macOS, Linux

## Install

=== "pip"

    ```bash
    pip install vectrix
    ```

=== "uv"

    ```bash
    uv add vectrix
    ```

The Rust engine is **built into the wheel** — no extras, no compiler needed. Like Polars, just install and it's fast.

## Optional Extras

```bash
pip install vectrix                  # 30+ models + built-in Rust engine
pip install "vectrix[ml]"            # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"    # + Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[all]"           # Everything
```

## Built-in Rust Engine

25 core forecasting hot loops are Rust-accelerated and compiled into every wheel:

- Linux (x86_64, manylinux)
- macOS (Apple Silicon ARM + x86_64)
- Windows (x86_64)
- Python 3.10, 3.11, 3.12, 3.13

No Rust compiler is needed. The acceleration is transparent — your code doesn't change, it just runs faster.

| Component | Python Only | With Rust | Speedup |
|:----------|:-----------|:----------|:--------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| DOT fit | 240ms | **10ms** | **24x** |
| ETS filter (hot loop) | 0.17ms | **0.003ms** | **67x** |

## Verify

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

Check that Rust engine is active:

```python
print(vectrix.TURBO_AVAILABLE)  # True
```

## Core Dependencies

Vectrix has only 3 required dependencies:

| Package | Minimum Version |
|---------|----------------|
| numpy | >= 1.24.0 |
| pandas | >= 2.0.0 |
| scipy | >= 1.10.0 |
