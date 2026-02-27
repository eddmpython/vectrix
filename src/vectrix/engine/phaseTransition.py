"""
Phase Transition Forecaster

Applies phase transition theory from statistical physics to time series forecasting.
Detects "critical slowing down" precursors that signal an impending regime shift:
  - Rising lag-1 autocorrelation (AR(1) coefficient approaching 1)
  - Increasing variance (flickering)
  - Changing skewness (asymmetry toward alternative stable state)

Kendall tau trend tests on each early warning indicator (EWI) produce a
composite transition score. When composite exceeds a threshold the system
is declared "critical" and prediction uncertainty is amplified.

References:
  - Scheffer et al. (2009), "Early-warning signals for critical transitions"
  - Dakos et al. (2012), "Methods for detecting early warnings"
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from scipy.stats import kendalltau


COMPOSITE_THRESHOLD = 0.5
CRITICAL_UNCERTAINTY_MULTIPLIER = 2.0
EWI_WEIGHT_AR1 = 0.45
EWI_WEIGHT_VAR = 0.35
EWI_WEIGHT_SKEW = 0.20


def _lag1Autocorrelation(segment: np.ndarray) -> float:
    if len(segment) < 3:
        return 0.0
    x = segment[:-1]
    y = segment[1:]
    stdX = np.std(x)
    stdY = np.std(y)
    if stdX < 1e-10 or stdY < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _windowVariance(segment: np.ndarray) -> float:
    if len(segment) < 2:
        return 0.0
    return float(np.var(segment, ddof=1))


def _windowSkewness(segment: np.ndarray) -> float:
    if len(segment) < 3:
        return 0.0
    n = len(segment)
    mean = np.mean(segment)
    std = np.std(segment, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.sum(((segment - mean) / std) ** 3) * n / ((n - 1) * (n - 2) + 1e-15))


def _kendallTrend(values: np.ndarray) -> float:
    if len(values) < 4:
        return 0.0
    tau, _ = kendalltau(np.arange(len(values)), values)
    if np.isnan(tau):
        return 0.0
    return float(tau)


class PhaseTransitionForecaster:
    """
    Phase transition theory regime-shift forecaster.
    Detects critical slowing down precursors (rising autocorrelation,
    variance, skewness) and amplifies uncertainty when a transition is imminent.
    """

    def __init__(
        self,
        windowSize: Optional[int] = None,
        baseModel=None
    ):
        self._windowSize = windowSize
        self._baseModelSpec = baseModel

        self._baseModel = None
        self._ar1Series: np.ndarray = np.array([])
        self._varSeries: np.ndarray = np.array([])
        self._skewSeries: np.ndarray = np.array([])
        self._tauAr1: float = 0.0
        self._tauVar: float = 0.0
        self._tauSkew: float = 0.0
        self._compositeScore: float = 0.0
        self._criticalState: bool = False
        self._windowCenters: np.ndarray = np.array([])
        self._isFitted: bool = False
        self._y: np.ndarray = np.array([])

    def fit(self, y: np.ndarray) -> 'PhaseTransitionForecaster':
        """Fit base model and compute early warning indicators via sliding windows."""
        self._y = y.copy()
        n = len(y)

        if self._baseModelSpec is not None:
            self._baseModel = self._baseModelSpec
        else:
            from .ets import ETSModel
            self._baseModel = ETSModel()

        self._baseModel.fit(y)

        windowSize = self._windowSize if self._windowSize is not None else max(10, n // 5)
        windowSize = min(windowSize, n - 2)
        windowSize = max(windowSize, 5)

        if hasattr(self._baseModel, 'residuals') and self._baseModel.residuals is not None and len(self._baseModel.residuals) > 0:
            workData = self._baseModel.residuals
        else:
            workData = np.diff(y) if n > 1 else y.copy()

        nWindows = len(workData) - windowSize + 1
        if nWindows < 4:
            self._ar1Series = np.zeros(1)
            self._varSeries = np.zeros(1)
            self._skewSeries = np.zeros(1)
            self._tauAr1 = 0.0
            self._tauVar = 0.0
            self._tauSkew = 0.0
            self._compositeScore = 0.0
            self._criticalState = False
            self._windowCenters = np.array([0])
            self._isFitted = True
            return self

        ar1Vals = np.zeros(nWindows)
        varVals = np.zeros(nWindows)
        skewVals = np.zeros(nWindows)
        centers = np.zeros(nWindows)

        for i in range(nWindows):
            segment = workData[i:i + windowSize]
            ar1Vals[i] = _lag1Autocorrelation(segment)
            varVals[i] = _windowVariance(segment)
            skewVals[i] = _windowSkewness(segment)
            centers[i] = i + windowSize / 2.0

        self._ar1Series = ar1Vals
        self._varSeries = varVals
        self._skewSeries = skewVals
        self._windowCenters = centers

        self._tauAr1 = _kendallTrend(ar1Vals)
        self._tauVar = _kendallTrend(varVals)
        self._tauSkew = _kendallTrend(skewVals)

        self._compositeScore = (
            EWI_WEIGHT_AR1 * self._tauAr1
            + EWI_WEIGHT_VAR * self._tauVar
            + EWI_WEIGHT_SKEW * abs(self._tauSkew)
        )

        self._criticalState = self._compositeScore > COMPOSITE_THRESHOLD

        self._isFitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with amplified uncertainty if critical state is detected."""
        if not self._isFitted:
            raise ValueError("Model not fitted.")

        predictions, lower95, upper95 = self._baseModel.predict(steps)

        if self._criticalState:
            center = predictions.copy()
            halfWidth = (upper95 - lower95) / 2.0

            amplificationFactor = 1.0 + (CRITICAL_UNCERTAINTY_MULTIPLIER - 1.0) * min(self._compositeScore / 1.0, 1.0)
            halfWidth *= amplificationFactor

            recentTrend = 0.0
            if len(self._y) >= 3:
                recentSegment = self._y[-min(len(self._y), 20):]
                diffs = np.diff(recentSegment)
                recentTrend = np.mean(diffs)

            trendMomentum = recentTrend * np.arange(1, steps + 1)
            skewBias = self._tauSkew * np.std(self._y) * 0.1 * np.sqrt(np.arange(1, steps + 1))

            predictions = center + skewBias
            lower95 = predictions - halfWidth
            upper95 = predictions + halfWidth

        return predictions, lower95, upper95

    def getTransitionIndicators(self) -> Dict[str, Any]:
        """Return early warning indicators and transition state diagnostics."""
        if not self._isFitted:
            raise ValueError("Model not fitted.")

        return {
            "ar1Series": self._ar1Series.copy(),
            "varianceSeries": self._varSeries.copy(),
            "skewnessSeries": self._skewSeries.copy(),
            "windowCenters": self._windowCenters.copy(),
            "kendallTauAr1": self._tauAr1,
            "kendallTauVariance": self._tauVar,
            "kendallTauSkewness": self._tauSkew,
            "compositeScore": self._compositeScore,
            "criticalState": self._criticalState,
            "threshold": COMPOSITE_THRESHOLD,
            "interpretation": self._interpretState(),
        }

    def _interpretState(self) -> str:
        if self._compositeScore > 0.8:
            return "Strong critical slowing down detected. Regime shift highly probable."
        if self._compositeScore > COMPOSITE_THRESHOLD:
            return "Moderate critical slowing down. Regime shift possible."
        if self._compositeScore > 0.3:
            return "Weak early warning signals. System may be approaching transition."
        return "System appears stable. No significant transition indicators."
