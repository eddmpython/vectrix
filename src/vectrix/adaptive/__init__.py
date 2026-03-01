"""
Regime-Aware Adaptive Forecasting & Self-Healing Forecast

HMM-based time series regime detection and regime-specific adaptive forecasting system.
Real-time forecast error monitoring and automatic correction (Self-Healing Forecast).
Constraint-aware forecast post-processing (Constraint-Aware Forecasting).
Time series DNA analysis (Forecast DNA Fingerprinting).

Detects time series "regimes" in real-time and automatically switches to
the optimal forecast model per regime, generating transition probability
weighted ensemble forecasts.

Usage:
    >>> from vectrix.adaptive import RegimeDetector, RegimeAwareForecaster
    >>> detector = RegimeDetector(nRegimes=3)
    >>> result = detector.detect(y)
    >>> print(result.currentRegime)
    >>>
    >>> raf = RegimeAwareForecaster()
    >>> forecast = raf.forecast(y, steps=30, period=7)
    >>> print(forecast.currentRegime, forecast.modelPerRegime)

    >>> from vectrix.adaptive import SelfHealingForecast
    >>> healer = SelfHealingForecast(predictions, lower, upper, data)
    >>> healer.observe(actual_values)
    >>> updated = healer.getUpdatedForecast()

    >>> from vectrix.adaptive import ConstraintAwareForecaster, Constraint
    >>> caf = ConstraintAwareForecaster()
    >>> result = caf.apply(predictions, lower, upper, constraints=[
    ...     Constraint('non_negative', {}),
    ...     Constraint('range', {'min': 100, 'max': 5000}),
    ... ])

    >>> from vectrix.adaptive import ForecastDNA
    >>> dna = ForecastDNA()
    >>> profile = dna.analyze(y, period=7)
    >>> print(profile.fingerprint, profile.difficulty, profile.recommendedModels)
"""

from .constraints import (
    Constraint,
    ConstraintAwareForecaster,
    ConstraintResult,
)
from .dna import (
    DNAProfile,
    ForecastDNA,
)
from .healing import (
    HealingReport,
    HealingStatus,
    SelfHealingForecast,
)
from .regime import (
    RegimeAwareForecaster,
    RegimeDetector,
    RegimeForecastResult,
    RegimeResult,
)

__all__ = [
    "RegimeDetector",
    "RegimeAwareForecaster",
    "RegimeResult",
    "RegimeForecastResult",
    "SelfHealingForecast",
    "HealingStatus",
    "HealingReport",
    "ConstraintAwareForecaster",
    "Constraint",
    "ConstraintResult",
    "ForecastDNA",
    "DNAProfile",
]
