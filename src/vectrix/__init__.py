"""
Vectrix: Zero-config time series forecasting library.

Pure numpy + scipy implementation with no external forecasting library dependencies.

Usage:
    >>> from vectrix import Vectrix
    >>> fx = Vectrix()
    >>> result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
    >>> print(result.predictions)
"""

from .adaptive import (
    Constraint,
    ConstraintAwareForecaster,
    ForecastDNA,
    RegimeAwareForecaster,
    RegimeDetector,
    SelfHealingForecast,
)
from .batch import BatchForecastResult, batchForecast
from .easy import (
    EasyAnalysisResult,
    EasyForecastResult,
    EasyRegressionResult,
    analyze,
    forecast,
    quick_report,
    regress,
)
from .engine.baselines import MeanModel, NaiveModel, RandomWalkDrift, SeasonalNaiveModel, WindowAverage
from .engine.changepoint import ChangePointDetector
from .engine.crossval import TimeSeriesCrossValidator
from .engine.events import EventEffect
from .engine.tsfeatures import TSFeatureExtractor
from .persistence import ModelPersistence
from .tsframe import TSFrame
from .types import (
    DataCharacteristics,
    FlatPredictionInfo,
    FlatRiskAssessment,
    ForecastResult,
    ModelResult,
    RiskLevel,
)
from .vectrix import Vectrix

__version__ = "3.0.0"
__all__ = [
    "Vectrix",
    "ForecastResult",
    "DataCharacteristics",
    "FlatRiskAssessment",
    "ModelResult",
    "FlatPredictionInfo",
    "RiskLevel",
    "NaiveModel",
    "SeasonalNaiveModel",
    "MeanModel",
    "RandomWalkDrift",
    "WindowAverage",
    "TimeSeriesCrossValidator",
    "TSFrame",
    # v3.0 - Engine extensions
    "ChangePointDetector",
    "EventEffect",
    "TSFeatureExtractor",
    # v3.0 - Adaptive (world-first)
    "RegimeDetector",
    "RegimeAwareForecaster",
    "SelfHealingForecast",
    "ConstraintAwareForecaster",
    "Constraint",
    "ForecastDNA",
    # v3.0 - Easy API (초보자 간편 인터페이스)
    "forecast",
    "analyze",
    "regress",
    "quick_report",
    "EasyForecastResult",
    "EasyAnalysisResult",
    "EasyRegressionResult",
    # v3.0 - Batch & Persistence
    "batchForecast",
    "BatchForecastResult",
    "ModelPersistence",
]
