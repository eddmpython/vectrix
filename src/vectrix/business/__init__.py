"""
Business Intelligence Module (Vectrix Differentiator)

경쟁 라이브러리에 없는 비즈니스 기능:
- AnomalyDetector: 이상치 탐지
- ForecastExplainer: 예측 설명
- WhatIfAnalyzer: 시나리오 분석
- Backtester: 백테스팅
- BusinessMetrics: 비즈니스 지표
- ReportGenerator: 종합 보고서
"""

from .anomaly import AnomalyDetector
from .explain import ForecastExplainer
from .whatif import WhatIfAnalyzer
from .backtest import Backtester
from .metrics import BusinessMetrics
from .report import ReportGenerator
from .htmlReport import HTMLReportGenerator

__all__ = [
    "AnomalyDetector",
    "ForecastExplainer",
    "WhatIfAnalyzer",
    "Backtester",
    "BusinessMetrics",
    "ReportGenerator",
    "HTMLReportGenerator",
]
