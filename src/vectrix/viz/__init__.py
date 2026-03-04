"""
Vectrix Visualization Module.

Interactive Plotly charts for time series analysis and forecasting.

Requires: pip install vectrix[viz]
"""

try:
    import plotly  # noqa: F401
except ImportError:
    raise ImportError(
        "plotly is required for vectrix.viz. "
        "Install it with: pip install vectrix[viz]"
    )

from .charts import (
    backtestChart,
    dnaRadar,
    forecastChart,
    metricsCard,
    modelHeatmap,
    scenarioChart,
)
from .report import analysisReport, forecastReport
from .theme import COLORS, LAYOUT, PALETTE, applyTheme

__all__ = [
    "forecastChart",
    "dnaRadar",
    "modelHeatmap",
    "scenarioChart",
    "backtestChart",
    "metricsCard",
    "forecastReport",
    "analysisReport",
    "COLORS",
    "PALETTE",
    "LAYOUT",
    "applyTheme",
]
