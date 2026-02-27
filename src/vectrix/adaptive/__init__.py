"""
Regime-Aware Adaptive Forecasting & Self-Healing Forecast

HMM 기반 시계열 레짐(국면) 감지 및 레짐별 적응 예측 시스템.
실시간 예측 오차 모니터링 및 자동 교정 (Self-Healing Forecast).
제약 조건 인식 예측 후처리 (Constraint-Aware Forecasting).
시계열 DNA 분석 (Forecast DNA Fingerprinting).

시계열의 "레짐"을 실시간 감지하고, 레짐별로 최적의 예측 모델을
자동 전환하여 전이 확률 가중 앙상블 예측을 생성한다.

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
