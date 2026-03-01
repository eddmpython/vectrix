"""
Core data type definitions for Vectrix.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FlatPredictionType(Enum):
    """Flat prediction types."""
    NONE = "none"
    HORIZONTAL = "horizontal"      # Flat horizontal line: ────────
    DIAGONAL = "diagonal"          # Flat diagonal line: ╱╱╱╱╱╱
    MEAN_REVERSION = "mean_reversion"  # Mean reversion: ∿→───


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Frequency(Enum):
    """Data frequency."""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    HOURLY = "H"
    UNKNOWN = "unknown"


@dataclass
class DataCharacteristics:
    """Data characteristics analysis result."""

    # Basic info
    length: int = 0
    frequency: Frequency = Frequency.UNKNOWN
    period: int = 1
    dateRange: Tuple[str, str] = ("", "")

    # Trend analysis
    hasTrend: bool = False
    trendDirection: str = "none"  # "up", "down", "none"
    trendStrength: float = 0.0    # 0.0 ~ 1.0

    # Seasonality analysis
    hasSeasonality: bool = False
    seasonalStrength: float = 0.0
    seasonalPeriods: List[int] = field(default_factory=list)
    hasMultipleSeasonality: bool = False

    # Stationarity
    isStationary: bool = False

    # Volatility
    volatility: float = 0.0
    volatilityLevel: str = "normal"  # "low", "normal", "high"

    # Quality
    missingRatio: float = 0.0
    outlierCount: int = 0
    outlierRatio: float = 0.0

    # Predictability
    predictabilityScore: float = 0.0  # 0 ~ 100


@dataclass
class FlatRiskAssessment:
    """Flat prediction risk assessment result."""

    # Overall risk
    riskScore: float = 0.0  # 0.0 ~ 1.0
    riskLevel: RiskLevel = RiskLevel.LOW

    # Individual risk factors
    riskFactors: Dict[str, bool] = field(default_factory=dict)

    # Recommended strategy
    recommendedStrategy: str = "standard"
    recommendedModels: List[str] = field(default_factory=list)

    # Warning messages
    warnings: List[str] = field(default_factory=list)


@dataclass
class FlatPredictionInfo:
    """Flat prediction detection info."""

    isFlat: bool = False
    flatType: FlatPredictionType = FlatPredictionType.NONE

    # Detection metrics
    predictionStd: float = 0.0
    originalStd: float = 0.0
    stdRatio: float = 0.0
    varianceRatio: float = 0.0

    # Correction info
    correctionApplied: bool = False
    correctionMethod: str = ""
    correctionStrength: float = 0.0

    # Messages
    message: str = ""
    suggestion: str = ""


@dataclass
class ModelResult:
    """Individual model forecast result."""

    modelId: str = ""
    modelName: str = ""

    # Predictions
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    upper95: np.ndarray = field(default_factory=lambda: np.array([]))

    # Evaluation metrics
    mape: float = float('inf')
    rmse: float = float('inf')
    mae: float = float('inf')
    smape: float = float('inf')

    # Flat prediction info
    flatInfo: Optional[FlatPredictionInfo] = None

    # Metadata
    trainingTime: float = 0.0
    isValid: bool = True


@dataclass
class ForecastResult:
    """Final forecast result."""

    success: bool = False

    # Predictions
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    dates: List[str] = field(default_factory=list)
    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    upper95: np.ndarray = field(default_factory=lambda: np.array([]))

    # Selected model
    bestModelId: str = ""
    bestModelName: str = ""

    # All model results
    allModelResults: Dict[str, ModelResult] = field(default_factory=dict)

    # Data characteristics
    characteristics: Optional[DataCharacteristics] = None

    # Flat prediction risk
    flatRisk: Optional[FlatRiskAssessment] = None

    # Flat detection/correction info
    flatInfo: Optional[FlatPredictionInfo] = None

    # Interpretation
    interpretation: Dict[str, Any] = field(default_factory=dict)

    # Warnings
    warnings: List[str] = field(default_factory=list)

    # Error
    error: Optional[str] = None

    def hasWarning(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0 or (self.flatInfo and self.flatInfo.isFlat)

    def getSummary(self) -> Dict[str, Any]:
        """Return a summary dict of the result."""
        return {
            'success': self.success,
            'bestModel': self.bestModelName,
            'mape': self.allModelResults.get(self.bestModelId, ModelResult()).mape,
            'predictionLength': len(self.predictions),
            'hasWarning': self.hasWarning(),
            'flatDetected': self.flatInfo.isFlat if self.flatInfo else False,
            'warningCount': len(self.warnings)
        }


MODEL_INFO = {
    'seasonal_naive': {
        'name': 'Seasonal Naive',
        'description': 'Repeats values from the same point in the last season.',
        'flatResistance': 0.95,
        'bestFor': ['strong seasonality', 'high flat risk'],
        'minData': 14
    },
    'snaive_drift': {
        'name': 'Seasonal Naive + Drift',
        'description': 'Seasonal pattern repetition with trend adjustment.',
        'flatResistance': 0.90,
        'bestFor': ['seasonality + trend', 'high flat risk'],
        'minData': 14
    },
    'mstl': {
        'name': 'MSTL',
        'description': 'Multiple seasonal decomposition (LOESS) + ARIMA.',
        'flatResistance': 0.85,
        'bestFor': ['multiple seasonality', 'complex patterns'],
        'minData': 50
    },
    'holt_winters': {
        'name': 'Holt-Winters',
        'description': 'Triple exponential smoothing (level + trend + season).',
        'flatResistance': 0.80,
        'bestFor': ['seasonal data', 'medium-term forecasting'],
        'minData': 24
    },
    'theta': {
        'name': 'Theta',
        'description': 'M3 Competition winner. Theta decomposition.',
        'flatResistance': 0.75,
        'bestFor': ['general purpose', 'fast forecasting'],
        'minData': 10
    },
    'auto_arima': {
        'name': 'AutoARIMA',
        'description': 'Automatic ARIMA. AICc-based optimal parameter selection.',
        'flatResistance': 0.60,
        'bestFor': ['stationary data', 'trend forecasting'],
        'minData': 30
    },
    'auto_ets': {
        'name': 'AutoETS',
        'description': 'Automatic exponential smoothing. 30 model combinations.',
        'flatResistance': 0.55,
        'bestFor': ['stable patterns', 'short-term forecasting'],
        'minData': 20
    },
    'ensemble': {
        'name': 'Variability-Preserving Ensemble',
        'description': 'Variability-preserving ensemble of top models.',
        'flatResistance': 0.85,
        'bestFor': ['uncertain patterns', 'stable forecasting'],
        'minData': 30
    },
    'naive': {
        'name': 'Naive',
        'description': 'Repeats last observation. Simplest benchmark.',
        'flatResistance': 0.10,
        'bestFor': ['benchmark', 'random walk data'],
        'minData': 2
    },
    'mean': {
        'name': 'Mean',
        'description': 'Historical mean forecast. Stationary benchmark.',
        'flatResistance': 0.05,
        'bestFor': ['benchmark', 'stationary series'],
        'minData': 2
    },
    'rwd': {
        'name': 'Random Walk with Drift',
        'description': 'Last value + average trend. Trending benchmark.',
        'flatResistance': 0.60,
        'bestFor': ['trending data', 'benchmark'],
        'minData': 5
    },
    'window_avg': {
        'name': 'Window Average',
        'description': 'Recent window average forecast.',
        'flatResistance': 0.15,
        'bestFor': ['benchmark', 'stable data'],
        'minData': 5
    },
    'auto_ces': {
        'name': 'AutoCES',
        'description': 'Complex exponential smoothing. Auto N/S/P/F selection.',
        'flatResistance': 0.65,
        'bestFor': ['nonlinear patterns', 'complex seasonality'],
        'minData': 20
    },
    'croston': {
        'name': 'Croston (Auto)',
        'description': 'Intermittent demand forecasting. Auto Classic/SBA/TSB.',
        'flatResistance': 0.30,
        'bestFor': ['intermittent demand', 'zero-inflated series'],
        'minData': 10
    },
    'dot': {
        'name': 'Dynamic Optimized Theta',
        'description': 'Joint Theta+alpha+drift L-BFGS-B optimization.',
        'flatResistance': 0.80,
        'bestFor': ['trending data', 'general purpose'],
        'minData': 10
    },
    'tbats': {
        'name': 'TBATS',
        'description': 'Trigonometric Seasonal, Box-Cox, ARMA, Trend, Damping.',
        'flatResistance': 0.85,
        'bestFor': ['multiple seasonality', 'hourly data', 'complex patterns'],
        'minData': 30
    },
    'garch': {
        'name': 'GARCH(1,1)',
        'description': 'Conditional variance model for financial volatility.',
        'flatResistance': 0.50,
        'bestFor': ['financial data', 'volatility forecasting', 'returns'],
        'minData': 50
    },
    'egarch': {
        'name': 'EGARCH',
        'description': 'Asymmetric volatility model with leverage effect.',
        'flatResistance': 0.50,
        'bestFor': ['financial data', 'asymmetric volatility'],
        'minData': 50
    },
    'gjr_garch': {
        'name': 'GJR-GARCH',
        'description': 'Threshold asymmetric GARCH for negative shock response.',
        'flatResistance': 0.50,
        'bestFor': ['financial data', 'asymmetric volatility'],
        'minData': 50
    },
    'four_theta': {
        'name': '4Theta Ensemble',
        'description': 'Weighted combination of 4 theta lines. Top stability.',
        'flatResistance': 0.80,
        'bestFor': ['general purpose', 'trending data', 'stable forecasting'],
        'minData': 10
    },
    'esn': {
        'name': 'Echo State Network',
        'description': 'Reservoir Computing nonlinear forecasting. Top accuracy.',
        'flatResistance': 0.70,
        'bestFor': ['nonlinear patterns', 'regime switching', 'high volatility'],
        'minData': 20
    },
    'dtsf': {
        'name': 'Dynamic Time Scan',
        'description': 'Non-parametric pattern matching. Best ensemble diversity.',
        'flatResistance': 0.65,
        'bestFor': ['repeating patterns', 'hourly data', 'seasonal data'],
        'minData': 30
    }
}
