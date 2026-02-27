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

## Optional Extras

```bash
pip install "vectrix[numba]"       # Numba JIT acceleration (2-5x speedup)
pip install "vectrix[ml]"          # LightGBM, XGBoost, scikit-learn
pip install "vectrix[tutorials]"   # Interactive marimo tutorials
pip install "vectrix[all]"         # Everything
```

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
