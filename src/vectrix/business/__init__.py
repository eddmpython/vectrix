"""
Business Intelligence Module (Vectrix Differentiator)

Business features not found in competing libraries:
- AnomalyDetector: Anomaly detection
- ForecastExplainer: Forecast explanation
- WhatIfAnalyzer: Scenario analysis
- Backtester: Backtesting
- BusinessMetrics: Business metrics
- ReportGenerator: Comprehensive reports
"""

from .anomaly import AnomalyDetector
from .backtest import Backtester
from .explain import ForecastExplainer
from .htmlReport import HTMLReportGenerator
from .metrics import BusinessMetrics
from .report import ReportGenerator
from .whatif import WhatIfAnalyzer

__all__ = [
    "AnomalyDetector",
    "ForecastExplainer",
    "WhatIfAnalyzer",
    "Backtester",
    "BusinessMetrics",
    "ReportGenerator",
    "HTMLReportGenerator",
]
