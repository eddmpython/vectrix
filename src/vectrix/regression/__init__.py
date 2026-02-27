"""
Regression module for time series forecasting

Feature engineering + regularized regression models + reduction strategies.
Diagnostics, robust models, model selection, time series regression.
Pure numpy/scipy implementation (no sklearn dependency).
"""

from .features import (
    LagFeatures,
    RollingFeatures,
    CalendarFeatures,
    FourierFeatures,
    autoFeatureEngineering,
)
from .linear import (
    LinearRegressor,
    RidgeRegressor,
    LassoRegressor,
    ElasticNetRegressor,
)
from .reduction import (
    DirectReduction,
    RecursiveReduction,
)
from .inference import (
    OLSInference,
    RegressionResult,
)
from .diagnostics import (
    DiagnosticResult,
    RegressionDiagnostics,
)
from .robust import (
    WLSRegressor,
    HuberRegressor,
    RANSACRegressor,
    QuantileRegressor,
)
from .selection import (
    StepwiseResult,
    RegularizationCVResult,
    BestSubsetResult,
    StepwiseSelector,
    RegularizationCV,
    BestSubsetSelector,
)
from .timeseries_regression import (
    TSRegressionResult,
    GrangerResult,
    NeweyWestOLS,
    CochraneOrcutt,
    PraisWinsten,
    GrangerCausality,
    DistributedLagModel,
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
