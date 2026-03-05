"""
Vectrix Visualization Module.

Interactive Plotly charts for time series analysis and forecasting.
Design tokens unified with the landing page (Cyan→Purple gradient).

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
from .dashboard import dashboard
from .report import analysisReport, forecastReport
from .theme import (
    COLORS,
    GRADIENT_COLORSCALE,
    HEATMAP_COLORSCALE,
    HEIGHT,
    LIGHT_COLORS,
    PALETTE,
    applyTheme,
)

__all__ = [
    "forecastChart",
    "dnaRadar",
    "modelHeatmap",
    "scenarioChart",
    "backtestChart",
    "metricsCard",
    "forecastReport",
    "analysisReport",
    "dashboard",
    "COLORS",
    "LIGHT_COLORS",
    "PALETTE",
    "GRADIENT_COLORSCALE",
    "HEATMAP_COLORSCALE",
    "HEIGHT",
    "applyTheme",
]
