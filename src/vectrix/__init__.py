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
from .datasets import listSamples, loadSample
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
from .ml import (
    CHRONOS_AVAILABLE,
    NEURALFORECAST_AVAILABLE,
    TIMESFM_AVAILABLE,
    ChronosForecaster,
    NBEATSForecaster,
    NeuralForecaster,
    NHITSForecaster,
    TFTForecaster,
    TimesFMForecaster,
)
from .persistence import ModelPersistence
from .pipeline import (
    BaseTransformer,
    BoxCoxTransformer,
    Deseasonalizer,
    Detrend,
    Differencer,
    ForecastPipeline,
    LogTransformer,
    MissingValueImputer,
    OutlierClipper,
    Scaler,
)
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

__version__ = "0.0.5"
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
    # v3.0 - Easy API
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
    # Foundation Models (optional)
    "ChronosForecaster",
    "TimesFMForecaster",
    "CHRONOS_AVAILABLE",
    "TIMESFM_AVAILABLE",
    # Deep Learning Models (optional)
    "NeuralForecaster",
    "NBEATSForecaster",
    "NHITSForecaster",
    "TFTForecaster",
    "NEURALFORECAST_AVAILABLE",
    # Pipeline
    "ForecastPipeline",
    "BaseTransformer",
    "Differencer",
    "LogTransformer",
    "BoxCoxTransformer",
    "Scaler",
    "Deseasonalizer",
    "Detrend",
    "OutlierClipper",
    "MissingValueImputer",
    # Datasets
    "loadSample",
    "listSamples",
]
