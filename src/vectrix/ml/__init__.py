"""
ML Model Wrappers for Time Series Forecasting

Optional dependencies: lightgbm, xgboost, scikit-learn.
Uses the same feature engineering from regression module.

Foundation Model wrappers: chronos-forecasting, timesfm.
Zero-shot forecasting with pre-trained transformers.
"""

from .chronos_model import CHRONOS_AVAILABLE, ChronosForecaster
from .lightgbm_model import LIGHTGBM_AVAILABLE, LightGBMForecaster
from .neuralforecast_model import (
    NEURALFORECAST_AVAILABLE,
    NBEATSForecaster,
    NeuralForecaster,
    NHITSForecaster,
    TFTForecaster,
)
from .sklearn_model import SKLEARN_AVAILABLE, SklearnForecaster
from .timesfm_model import TIMESFM_AVAILABLE, TimesFMForecaster
from .xgboost_model import XGBOOST_AVAILABLE, XGBoostForecaster

__all__ = [
    "LightGBMForecaster",
    "XGBoostForecaster",
    "SklearnForecaster",
    "ChronosForecaster",
    "TimesFMForecaster",
    "NeuralForecaster",
    "NBEATSForecaster",
    "NHITSForecaster",
    "TFTForecaster",
    "LIGHTGBM_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "SKLEARN_AVAILABLE",
    "CHRONOS_AVAILABLE",
    "TIMESFM_AVAILABLE",
    "NEURALFORECAST_AVAILABLE",
]
