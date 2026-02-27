"""
ML Model Wrappers for Time Series Forecasting

Optional dependencies: lightgbm, xgboost, scikit-learn.
Uses the same feature engineering from regression module.
"""

from .lightgbm_model import LIGHTGBM_AVAILABLE, LightGBMForecaster
from .sklearn_model import SKLEARN_AVAILABLE, SklearnForecaster
from .xgboost_model import XGBOOST_AVAILABLE, XGBoostForecaster

__all__ = [
    "LightGBMForecaster",
    "XGBoostForecaster",
    "SklearnForecaster",
    "LIGHTGBM_AVAILABLE",
    "XGBOOST_AVAILABLE",
    "SKLEARN_AVAILABLE",
]
