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


def _getModelInfo():
    """Get MODEL_INFO from the registry (single source of truth)."""
    from .engine.registry import getModelInfo
    return getModelInfo()


class _LazyModelInfo(dict):
    """Lazy-loading MODEL_INFO that initializes from registry on first access."""

    _loaded = False

    def _ensureLoaded(self):
        if not self._loaded:
            self.update(_getModelInfo())
            self._loaded = True

    def __getitem__(self, key):
        self._ensureLoaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensureLoaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensureLoaded()
        return super().__iter__()

    def keys(self):
        self._ensureLoaded()
        return super().keys()

    def values(self):
        self._ensureLoaded()
        return super().values()

    def items(self):
        self._ensureLoaded()
        return super().items()

    def get(self, key, default=None):
        self._ensureLoaded()
        return super().get(key, default)


MODEL_INFO = _LazyModelInfo()
