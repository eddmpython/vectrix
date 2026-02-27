"""
Regression module for time series forecasting

Feature engineering + regularized regression models + reduction strategies.
Diagnostics, robust models, model selection, time series regression.
Pure numpy/scipy implementation (no sklearn dependency).
"""

from .diagnostics import (
    DiagnosticResult,
    RegressionDiagnostics,
)
from .features import (
    CalendarFeatures,
    FourierFeatures,
    LagFeatures,
    RollingFeatures,
    autoFeatureEngineering,
)
from .inference import (
    OLSInference,
    RegressionResult,
)
from .linear import (
    ElasticNetRegressor,
    LassoRegressor,
    LinearRegressor,
    RidgeRegressor,
)
from .reduction import (
    DirectReduction,
    RecursiveReduction,
)
from .robust import (
    HuberRegressor,
    QuantileRegressor,
    RANSACRegressor,
    WLSRegressor,
)
from .selection import (
    BestSubsetResult,
    BestSubsetSelector,
    RegularizationCV,
    RegularizationCVResult,
    StepwiseResult,
    StepwiseSelector,
)
from .timeseries_regression import (
    CochraneOrcutt,
    DistributedLagModel,
    GrangerCausality,
    GrangerResult,
    NeweyWestOLS,
    PraisWinsten,
    TSRegressionResult,
)

__all__ = [
    # Features
    "LagFeatures",
    "RollingFeatures",
    "CalendarFeatures",
    "FourierFeatures",
    "autoFeatureEngineering",
    # Linear models
    "LinearRegressor",
    "RidgeRegressor",
    "LassoRegressor",
    "ElasticNetRegressor",
    # Reduction strategies
    "DirectReduction",
    "RecursiveReduction",
    # Inference
    "OLSInference",
    "RegressionResult",
    # Diagnostics
    "DiagnosticResult",
    "RegressionDiagnostics",
    # Robust models
    "WLSRegressor",
    "HuberRegressor",
    "RANSACRegressor",
    "QuantileRegressor",
    # Model selection
    "StepwiseResult",
    "RegularizationCVResult",
    "BestSubsetResult",
    "StepwiseSelector",
    "RegularizationCV",
    "BestSubsetSelector",
    # Time series regression
    "TSRegressionResult",
    "GrangerResult",
    "NeweyWestOLS",
    "CochraneOrcutt",
    "PraisWinsten",
    "GrangerCausality",
    "DistributedLagModel",
]
