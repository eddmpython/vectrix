"""
TSFrame: Time Series DataFrame

Time-series-aware DataFrame wrapper:
- Automatic date/frequency detection
- Built-in forecast/decompose methods
- Missing value handling, resampling
- Train/test split
"""

from .tsframe import TSFrame

__all__ = ["TSFrame"]
