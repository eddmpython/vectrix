"""
Vectrix Forecasting Engine

Time series forecasting engine implemented in pure numpy + numba without external library dependencies
"""

from .adversarial import AdversarialStressTester, StressTestResult
from .arima import ARIMAModel
from .baselines import MeanModel, NaiveModel, RandomWalkDrift, SeasonalNaiveModel, WindowAverage
from .ces import AutoCES, CESModel
from .changepoint import ChangePointDetector, ChangePointResult
from .comparison import ModelComparison
from .crossval import TimeSeriesCrossValidator
from .croston import AutoCroston, CrostonClassic, CrostonSBA, CrostonTSB
from .decomposition import SeasonalDecomposition
from .diagnostics import ForecastDiagnostics, ForecastDiagnosticsResult
from .dot import DynamicOptimizedTheta
from .dtsf import DynamicTimeScanForecaster
from .entropic import EntropicConfidenceScorer, EntropyResult
from .esn import EchoStateForecaster
from .ets import ETSModel
from .events import EventEffect
from .fourTheta import AdaptiveThetaEnsemble
from .garch import EGARCHModel, GARCHModel, GJRGARCHModel
from .hawkes import HawkesIntermittentDemand
from .impute import TimeSeriesImputer
from .logistic import LogisticGrowthModel, SaturatingTrendModel
from .lotkaVolterra import LotkaVolterraEnsemble
from .mstl import MSTL, AutoMSTL
from .periodic_drop import PeriodicDropDetector
from .phaseTransition import PhaseTransitionForecaster
from .probabilistic import ProbabilisticForecaster
from .tbats import TBATS, AutoTBATS
from .theta import ThetaModel
from .tsfeatures import TSFeatureExtractor
from .registry import ModelSpec, createModel, getModelInfo, getModelSpec, getRegistry, listModelIds
from .turbo import TurboCore
from .var import VARModel, VECMModel

__all__ = [
    "TurboCore",
    "ETSModel",
    "ARIMAModel",
    "ThetaModel",
    "SeasonalDecomposition",
    "MSTL",
    "AutoMSTL",
    "PeriodicDropDetector",
    "NaiveModel",
    "SeasonalNaiveModel",
    "MeanModel",
    "RandomWalkDrift",
    "WindowAverage",
    "TimeSeriesCrossValidator",
    "CESModel",
    "AutoCES",
    "CrostonClassic",
    "CrostonSBA",
    "CrostonTSB",
    "AutoCroston",
    "DynamicOptimizedTheta",
    "TBATS",
    "AutoTBATS",
    "GARCHModel",
    "EGARCHModel",
    "GJRGARCHModel",
    "ChangePointDetector",
    "ChangePointResult",
    "EventEffect",
    "TSFeatureExtractor",
    "ProbabilisticForecaster",
    "TimeSeriesImputer",
    "LogisticGrowthModel",
    "SaturatingTrendModel",
    "ModelComparison",
    "ForecastDiagnostics",
    "ForecastDiagnosticsResult",
    "HawkesIntermittentDemand",
    "EntropicConfidenceScorer",
    "EntropyResult",
    "LotkaVolterraEnsemble",
    "PhaseTransitionForecaster",
    "AdversarialStressTester",
    "StressTestResult",
    "VARModel",
    "VECMModel",
    "AdaptiveThetaEnsemble",
    "EchoStateForecaster",
    "DynamicTimeScanForecaster",
    "ModelSpec",
    "getRegistry",
    "getModelSpec",
    "listModelIds",
    "createModel",
    "getModelInfo",
]
