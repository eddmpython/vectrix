"""
Global Model Module

다중 시계열을 동시에 학습하는 글로벌 모델:
- GlobalForecaster: 여러 시계열에서 공통 패턴 학습
- PanelData: 다중 시계열 데이터 관리
"""

from .global_forecaster import GlobalForecaster
from .panel import PanelData

__all__ = [
    "GlobalForecaster",
    "PanelData",
]
