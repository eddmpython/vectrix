"""
ML Model Wrappers for Time Series Forecasting

Optional dependencies: lightgbm, xgboost, scikit-learn.
Uses the same feature engineering from regression module.
"""

from .lightgbm_model import LightGBMForecaster, LIGHTGBM_AVAILABLE
from .xgboost_model import XGBoostForecaster, XGBOOST_AVAILABLE
from .sklearn_model import SklearnForecaster, SKLEARN_AVAILABLE

__all__ = [
    "LightGBMForecaster",
    "XGBoostForecaster",
    "SklearnForecaster",
    "LIGHTGBM_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "SKLEARN_AVAILABLE",
]
