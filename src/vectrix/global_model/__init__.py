"""
Global Model Module

Global models that learn from multiple time series simultaneously:
- GlobalForecaster: Learn common patterns across multiple series
- PanelData: Multi-series data management
"""

from .global_forecaster import GlobalForecaster
from .panel import PanelData

__all__ = [
    "GlobalForecaster",
    "PanelData",
]
