"""
Vectrix: Zero-config time series forecasting library.

Built-in Rust engine for blazing-fast performance. No Rust compiler needed — pre-built wheels included.

Usage:
    >>> from vectrix import Vectrix
    >>> fx = Vectrix()
    >>> result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
    >>> print(result.predictions)
"""

import sys

try:
    from . import _core
    sys.modules["vectrix_core"] = _core
    TURBO_AVAILABLE = True
except ImportError:
    TURBO_AVAILABLE = False

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
    compare,
    forecast,
    quick_report,
    regress,
)
from .engine.baselines import MeanModel, NaiveModel, RandomWalkDrift, SeasonalNaiveModel, WindowAverage
from .engine.dtsf import DynamicTimeScanForecaster
from .engine.esn import EchoStateForecaster
from .engine.fourTheta import AdaptiveThetaEnsemble
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

__version__ = "0.0.10"
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
    # New engines (v0.0.9+)
    "AdaptiveThetaEnsemble",
    "DynamicTimeScanForecaster",
    "EchoStateForecaster",
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
    "compare",
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
    # Rust acceleration
    "TURBO_AVAILABLE",
]
