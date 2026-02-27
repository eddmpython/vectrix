"""
ForecastX: Zero-config time series forecasting library.

Pure numpy + scipy implementation with no external forecasting library dependencies.

Usage:
    >>> from forecastx import ForecastX
    >>> fx = ForecastX()
    >>> result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
    >>> print(result.predictions)
"""

from .forecastx import ForecastX
from .types import (
    ForecastResult,
    DataCharacteristics,
    FlatRiskAssessment,
    ModelResult,
    FlatPredictionInfo,
    RiskLevel,
)
from .engine.baselines import NaiveModel, SeasonalNaiveModel, MeanModel, RandomWalkDrift, WindowAverage
from .engine.crossval import TimeSeriesCrossValidator
from .engine.changepoint import ChangePointDetector
from .engine.events import EventEffect
from .engine.tsfeatures import TSFeatureExtractor
from .tsframe import TSFrame
from .adaptive import (
    RegimeDetector,
    RegimeAwareForecaster,
    SelfHealingForecast,
    ConstraintAwareForecaster,
    Constraint,
    ForecastDNA,
)
from .easy import (
    forecast,
    analyze,
    regress,
    quick_report,
    EasyForecastResult,
    EasyAnalysisResult,
    EasyRegressionResult,
)
from .batch import batchForecast, BatchForecastResult
from .persistence import ModelPersistence

__version__ = "3.0.0"
__all__ = [
    "ForecastX",
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
