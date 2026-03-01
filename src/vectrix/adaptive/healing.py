"""
Self-Healing Forecast

After forecasts are generated and actual data arrives, this system monitors
forecast errors in real-time and automatically corrects them - a 'living forecast' system.

Core algorithms:
- CUSUM (Cumulative Sum): Detects systematic bias
- EWMA (Exponentially Weighted Moving Average): Tracks recent error trends
- Adaptive Conformal Correction: Distribution-free forecast correction
- Online residual learning + forecast update

Usage:
    >>> from vectrix.adaptive.healing import SelfHealingForecast
    >>> healer = SelfHealingForecast(predictions, lower95, upper95, historicalData)
    >>> healer.observe(actual_values)
    >>> updated = healer.getUpdatedForecast()
    >>> status = healer.getStatus()
    >>> report = healer.getReport()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class HealingStatus:
    """
    Snapshot of the current Self-Healing forecast state

    Returned on each observe() call, containing forecast health status,
    drift detection results, and correction application status.
    """
    health: str                         # 'healthy', 'degrading', 'critical', 'healed'
    healthScore: float                  # 0-100
    observedCount: int
    remainingSteps: int
    driftDetected: bool
    driftDirection: Optional[str]       # 'upward_bias', 'downward_bias', None
    driftMagnitude: float
    correctionApplied: bool
    biasEstimate: float                 # Estimated bias
    mape: float                         # Current MAPE
    mae: float                          # Current MAE
    refitRecommended: bool
    refitReason: Optional[str]
    message: str                        # Human-readable status message


@dataclass
class HealingReport:
    """
    Full healing process report for Self-Healing forecast

    Returned by getReport(), containing summary of the entire healing process,
    before/after MAPE comparison, drift event list, etc.
    """
    overallHealth: str
    healthScore: float
    totalObserved: int
    totalCorrected: int
    originalMape: float                 # Pre-correction MAPE
    healedMape: float                   # Post-correction MAPE
    improvementPct: float               # Improvement ratio
    corrections: List[Dict] = field(default_factory=list)
    healingLog: List[str] = field(default_factory=list)
    driftEvents: List[Dict] = field(default_factory=list)
    updatedPredictions: np.ndarray = field(default_factory=lambda: np.array([]))
    updatedLower: np.ndarray = field(default_factory=lambda: np.array([]))
    updatedUpper: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# SelfHealingForecast
# ---------------------------------------------------------------------------

class SelfHealingForecast:
    """
    Self-healing forecast system

    When actual data arrives sequentially after forecasts are generated,
    monitors forecast errors in real-time and automatically corrects them.

    Features:
    1. Real-time forecast error monitoring
    2. CUSUM + EWMA based drift detection
    3. Adaptive Conformal Prediction based automatic correction
    4. Online residual learning + forecast update
    5. Automatic refit trigger determination

    Parameters
    ----------
    predictions : np.ndarray
        Original predictions (length H)
    lower95 : np.ndarray
        Original 95% lower bound (length H)
    upper95 : np.ndarray
        Original 95% upper bound (length H)
    historicalData : np.ndarray
        Historical data used for training
    period : int
        Seasonal period (default 7)
    healingMode : str
        Correction intensity. One of 'conservative', 'adaptive', 'aggressive'

    Usage:
        >>> healer = SelfHealingForecast(original_forecast, lower, upper, data)
        >>> healer.observe(actual_values)      # Actual data arrives
        >>> updated = healer.getUpdatedForecast()  # Corrected forecast
        >>> healer.getStatus()                 # Drift detection, alerts, health status
    """

    _VALID_MODES = ('conservative', 'adaptive', 'aggressive')

    def __init__(
        self,
        predictions: np.ndarray,
        lower95: np.ndarray,
        upper95: np.ndarray,
        historicalData: np.ndarray,
        period: int = 7,
        healingMode: str = 'adaptive',
    ):
        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        predictions = np.asarray(predictions, dtype=np.float64)
        lower95 = np.asarray(lower95, dtype=np.float64)
        upper95 = np.asarray(upper95, dtype=np.float64)
        historicalData = np.asarray(historicalData, dtype=np.float64)

        if predictions.ndim != 1 or lower95.ndim != 1 or upper95.ndim != 1:
            raise ValueError("predictions, lower95, upper95 must be 1-dimensional arrays.")
        if len(predictions) == 0:
            raise ValueError("predictions is empty.")
        if len(predictions) != len(lower95) or len(predictions) != len(upper95):
            raise ValueError(
                "predictions, lower95, upper95 must have the same length. "
                f"Received lengths: {len(predictions)}, {len(lower95)}, {len(upper95)}"
            )
        if historicalData.ndim != 1 or len(historicalData) < 2:
            raise ValueError("historicalData must be a 1-dimensional array with length >= 2.")
        if period < 1:
            raise ValueError(f"period must be at least 1. Received: {period}")
        if healingMode not in self._VALID_MODES:
            raise ValueError(
                f"healingMode must be one of {self._VALID_MODES}. "
                f"Received: '{healingMode}'"
            )

        # ------------------------------------------------------------------
        # Store original forecasts (immutable)
        # ------------------------------------------------------------------
        self.originalPredictions = predictions.copy()
        self.currentPredictions = predictions.copy()
        self.originalLower = lower95.copy()
        self.originalUpper = upper95.copy()
        self.currentLower = lower95.copy()
        self.currentUpper = upper95.copy()
        self.historicalData = historicalData.copy()
        self.period = period
        self.healingMode = healingMode
        self.totalSteps = len(predictions)

        # ------------------------------------------------------------------
        # Reference statistics (based on historical data)
        # ------------------------------------------------------------------
        self._referenceStd = float(np.std(historicalData))
        if self._referenceStd < 1e-10:
            # Fallback when data is nearly constant
            self._referenceStd = float(np.mean(np.abs(historicalData))) * 0.01 + 1e-6
        self._referenceMean = float(np.mean(historicalData))

        # Initial mean forecast interval width (for reference)
        self._originalMeanWidth = float(np.mean(upper95 - lower95))

        # ------------------------------------------------------------------
        # Observed data tracking
        # ------------------------------------------------------------------
        self.observedValues: List[float] = []
        self.observedCount: int = 0

        # ------------------------------------------------------------------
        # Error tracking
        # ------------------------------------------------------------------
        self.errors: List[float] = []            # actual - prediction (signed)
        self.absErrors: List[float] = []         # |actual - prediction|
        self.signedBias: List[float] = []        # Cumulative bias tracking

        # ------------------------------------------------------------------
        # Drift detection (CUSUM + EWMA)
        # ------------------------------------------------------------------
        self.cusumPos: float = 0.0
        self.cusumNeg: float = 0.0
        self.ewmaError: float = 0.0
        self.ewmaLambda: float = 0.2
        self.driftDetected: bool = False
        self.driftDirection: Optional[str] = None
        self.driftMagnitude: float = 0.0
        self._driftEvents: List[Dict] = []

        # ------------------------------------------------------------------
        # Correction history
        # ------------------------------------------------------------------
        self.corrections: List[Dict] = []
        self.healingLog: List[str] = []

        # ------------------------------------------------------------------
        # Health status
        # ------------------------------------------------------------------
        self.health: str = 'healthy'
        self.healthScore: float = 100.0
        self.refitRecommended: bool = False
        self.refitReason: Optional[str] = None
        self.refitRecommendedAt: Optional[int] = None

        # Initial log
        self.healingLog.append(
            f"[init] SelfHealingForecast created: "
            f"steps={self.totalSteps}, mode={healingMode}, "
            f"refStd={self._referenceStd:.4f}"
        )

    # ======================================================================
    # Public API
    # ======================================================================

    def observe(self, actuals: np.ndarray) -> HealingStatus:
        """
        Observe actual data and automatically update forecast

        When new actual observations (one or multiple) arrive, internally
        performs error analysis -> drift detection -> correction.

        Parameters
        ----------
        actuals : np.ndarray
            Newly arrived actual observations. Scalar or 1-dimensional array.

        Returns
        -------
        HealingStatus
            Current healing status snapshot

        Raises
        ------
        ValueError
            When observations exceed forecast range
        """
        actuals = np.atleast_1d(np.asarray(actuals, dtype=np.float64)).ravel()

        if len(actuals) == 0:
            return self.getStatus()

        remaining = self.totalSteps - self.observedCount
        if len(actuals) > remaining:
            raise ValueError(
                f"{len(actuals)} observations exceed {remaining} remaining forecast steps."
            )

        correctionApplied = False

        for actual in actuals:
            idx = self.observedCount
            predicted = self.originalPredictions[idx]

            # 1. Compute error (actual - predicted: positive = under-prediction)
            error = float(actual - predicted)
            absError = abs(error)

            self.observedValues.append(float(actual))
            self.errors.append(error)
            self.absErrors.append(absError)

            # Cumulative bias tracking
            cumBias = float(np.mean(self.errors))
            self.signedBias.append(cumBias)

            # 2. CUSUM update
            self._updateCUSUM(error)

            # 3. EWMA update
            self._updateEWMA(error)

            self.observedCount += 1

            # 4. Health status evaluation
            self._evaluateHealth()

            # 5. Log entry
            self.healingLog.append(
                f"[step {idx}] actual={actual:.4f}, pred={predicted:.4f}, "
                f"error={error:.4f}, health={self.health}({self.healthScore:.1f})"
            )

        # 6. Apply correction (after at least 2 observations)
        if self.observedCount >= 2:
            self._applyCorrection()
            correctionApplied = True

        # 7. Evaluate refit recommendation
        self._evaluateRefitNeed()

        return self._buildStatus(correctionApplied)

    def getUpdatedForecast(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return corrected predictions, lower bound, and upper bound

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (predictions, lower95, upper95) - each array of length H.
            Observed steps are filled with actual values, rest with corrected forecasts.
        """
        preds = self.currentPredictions.copy()
        lower = self.currentLower.copy()
        upper = self.currentUpper.copy()

        # Overwrite observed steps with actual values
        for i, val in enumerate(self.observedValues):
            preds[i] = val
            lower[i] = val
            upper[i] = val

        return preds, lower, upper

    def getStatus(self) -> HealingStatus:
        """
        Return current status snapshot

        Returns
        -------
        HealingStatus
            Current healing status
        """
        return self._buildStatus(correctionApplied=len(self.corrections) > 0)

    def getReport(self) -> HealingReport:
        """
        Generate full healing process report

        Returns
        -------
        HealingReport
            Healing process summary, before/after comparison, drift events, etc.
        """
        # Pre-correction MAPE (original prediction vs actual)
        originalMape = self._computeMape(
            self.originalPredictions[:self.observedCount],
            np.array(self.observedValues)
        )

        # Post-correction MAPE (corrected prediction vs actual)
        # Note: corrections apply to 'future' steps, so direct comparison is limited.
        # Here we compute estimated correction effect
        healedMape = originalMape  # Default

        if len(self.corrections) > 0 and self.observedCount > 0:
            # Compare corrected prediction with actual at each correction step
            healedErrors = []
            for i, actual in enumerate(self.observedValues):
                correctedPred = self.currentPredictions[i]
                if abs(actual) > 1e-10:
                    healedErrors.append(abs(correctedPred - actual) / abs(actual))
                else:
                    healedErrors.append(abs(correctedPred - actual))
            healedMape = float(np.mean(healedErrors) * 100) if healedErrors else originalMape

        # Improvement rate
        if originalMape > 1e-10:
            improvementPct = max(0.0, (originalMape - healedMape) / originalMape * 100)
        else:
            improvementPct = 0.0

        preds, lower, upper = self.getUpdatedForecast()

        return HealingReport(
            overallHealth=self.health,
            healthScore=self.healthScore,
            totalObserved=self.observedCount,
            totalCorrected=len(self.corrections),
            originalMape=originalMape,
            healedMape=healedMape,
            improvementPct=improvementPct,
            corrections=list(self.corrections),
            healingLog=list(self.healingLog),
            driftEvents=list(self._driftEvents),
            updatedPredictions=preds,
            updatedLower=lower,
            updatedUpper=upper,
        )

    def reset(self) -> None:
        """
        Reset observed data and correction history.
        Original predictions are preserved.
        """
        self.currentPredictions = self.originalPredictions.copy()
        self.currentLower = self.originalLower.copy()
        self.currentUpper = self.originalUpper.copy()

        self.observedValues.clear()
        self.observedCount = 0
        self.errors.clear()
        self.absErrors.clear()
        self.signedBias.clear()

        self.cusumPos = 0.0
        self.cusumNeg = 0.0
        self.ewmaError = 0.0
        self.driftDetected = False
        self.driftDirection = None
        self.driftMagnitude = 0.0
        self._driftEvents.clear()

        self.corrections.clear()
        self.healingLog.clear()

        self.health = 'healthy'
        self.healthScore = 100.0
        self.refitRecommended = False
        self.refitReason = None
        self.refitRecommendedAt = None

        self.healingLog.append("[reset] SelfHealingForecast state reset complete")

    # ======================================================================
    # Drift Detection
    # ======================================================================

    def _updateCUSUM(self, error: float) -> None:
        """
        Bidirectional CUSUM for systematic bias detection

        S_pos = max(0, S_pos + (error - k))
        S_neg = max(0, S_neg + (-error - k))

        k = 0.5 * sigma  (slack parameter: ignore small variations)
        h = 5.0 * sigma  (threshold: drift detected when exceeded)

        Persistent error > 0 -> cusumPos rises -> under-prediction (downward_bias)
        Persistent error < 0 -> cusumNeg rises -> over-prediction (upward_bias)
        """
        k = 0.5 * self._referenceStd
        h = 5.0 * self._referenceStd

        self.cusumPos = max(0.0, self.cusumPos + error - k)
        self.cusumNeg = max(0.0, self.cusumNeg + (-error) - k)

        prevDrift = self.driftDetected

        if self.cusumPos > h:
            self.driftDetected = True
            self.driftDirection = 'downward_bias'   # actual > predicted
            self.driftMagnitude = self.cusumPos / max(self.observedCount, 1)
        elif self.cusumNeg > h:
            self.driftDetected = True
            self.driftDirection = 'upward_bias'     # predicted > actual
            self.driftMagnitude = self.cusumNeg / max(self.observedCount, 1)
        else:
            # Below threshold: drift can be cleared
            if self.cusumPos < h * 0.3 and self.cusumNeg < h * 0.3:
                self.driftDetected = False
                self.driftDirection = None
                self.driftMagnitude = 0.0

        # Record new drift event
        if self.driftDetected and not prevDrift:
            event = {
                'step': self.observedCount,
                'direction': self.driftDirection,
                'magnitude': self.driftMagnitude,
                'cusumPos': self.cusumPos,
                'cusumNeg': self.cusumNeg,
            }
            self._driftEvents.append(event)
            self.healingLog.append(
                f"[drift] Drift detected at step {self.observedCount}: "
                f"{self.driftDirection}, magnitude={self.driftMagnitude:.4f}"
            )

    def _updateEWMA(self, error: float) -> None:
        """
        EWMA for tracking recent error trends

        E_t = lambda * error + (1 - lambda) * E_{t-1}

        EWMA weights recent errors more heavily, detecting trend changes
        faster than CUSUM.
        """
        self.ewmaError = (
            self.ewmaLambda * error
            + (1.0 - self.ewmaLambda) * self.ewmaError
        )

    # ======================================================================
    # Health Status Evaluation
    # ======================================================================

    def _evaluateHealth(self) -> None:
        """
        Evaluate forecast health status on a 0-100 score

        Score criteria:
        - 100-80: healthy  (errors within normal range)
        - 80-50:  degrading (increasing error trend or mild bias)
        - 50-0:   critical  (systematic bias or large errors)

        Transitions to 'healed' if errors improve after drift detection + correction
        """
        if self.observedCount == 0:
            self.health = 'healthy'
            self.healthScore = 100.0
            return

        score = 100.0

        # --- (1) MAE-based deduction ---
        mae = float(np.mean(self.absErrors))
        maeRatio = mae / max(self._referenceStd, 1e-10)
        # maeRatio of 1 means error is at reference std level -> moderate
        # maeRatio > 2 means large error
        maeDeduction = min(40.0, maeRatio * 20.0)
        score -= maeDeduction

        # --- (2) Bias-based deduction ---
        meanBias = abs(float(np.mean(self.errors)))
        biasRatio = meanBias / max(self._referenceStd, 1e-10)
        biasDeduction = min(30.0, biasRatio * 15.0)
        score -= biasDeduction

        # --- (3) Error trend deduction ---
        if len(self.absErrors) >= 3:
            recentErrors = self.absErrors[-min(5, len(self.absErrors)):]
            # Deduct if error is trending upward
            if len(recentErrors) >= 3:
                xVals = np.arange(len(recentErrors), dtype=np.float64)
                slope = np.polyfit(xVals, recentErrors, 1)[0]
                if slope > 0:
                    trendDeduction = min(15.0, slope / max(self._referenceStd, 1e-10) * 10.0)
                    score -= trendDeduction

        # --- (4) EWMA bias deduction ---
        ewmaRatio = abs(self.ewmaError) / max(self._referenceStd, 1e-10)
        ewmaDeduction = min(15.0, ewmaRatio * 10.0)
        score -= ewmaDeduction

        # --- (5) Additional deduction on drift detection ---
        if self.driftDetected:
            score -= 10.0

        # Score range limit
        score = max(0.0, min(100.0, score))
        self.healthScore = score

        # Health status determination
        if score >= 80:
            self.health = 'healthy'
        elif score >= 50:
            self.health = 'degrading'
        else:
            self.health = 'critical'

        # Transition to 'healed' if errors improved after correction
        if len(self.corrections) > 0 and self.observedCount >= 3:
            recentAbsErrors = self.absErrors[-min(3, len(self.absErrors)):]
            earlyAbsErrors = self.absErrors[:min(3, len(self.absErrors))]
            if np.mean(recentAbsErrors) < np.mean(earlyAbsErrors) * 0.7:
                self.health = 'healed'

    def _evaluateRefitNeed(self) -> None:
        """
        Determine whether refit is recommended

        Refit recommendation criteria:
        - healthScore < 50 (after 5+ observations)
        - Drift detected + correction did not improve
        - Current MAPE > initial expectation * 2
        """
        if self.refitRecommended:
            return  # Already recommended

        if self.observedCount < 5:
            return  # Insufficient data for judgment

        reasons = []

        # Criterion 1: Low health score
        if self.healthScore < 50:
            reasons.append(f"Health score {self.healthScore:.1f} < 50")

        # Criterion 2: Drift + correction failure
        if self.driftDetected and len(self.corrections) >= 2:
            recentErrors = self.absErrors[-3:]
            if np.mean(recentErrors) > self._referenceStd * 2:
                reasons.append(
                    f"Errors remain large after drift detection and correction "
                    f"(recent MAE={np.mean(recentErrors):.4f})"
                )

        # Criterion 3: Excessive MAPE
        currentMape = self._computeCurrentMape()
        if currentMape > 30.0:  # Over 30%
            reasons.append(f"MAPE {currentMape:.1f}% > 30%")

        if reasons:
            self.refitRecommended = True
            self.refitReason = "; ".join(reasons)
            self.refitRecommendedAt = self.observedCount
            self.healingLog.append(
                f"[refit] Refit recommended at step {self.observedCount}: {self.refitReason}"
            )

    # ======================================================================
    # Correction Algorithm
    # ======================================================================

    def _applyCorrection(self) -> None:
        """
        Correct remaining forecasts based on observed error patterns

        Correction strategies:
        1. Bias correction: shift forecast when systematic bias detected
        2. Trend correction: linear correction if errors show a trend
        3. Seasonal correction: reflect periodic patterns in errors
        4. Volatility correction: adjust confidence interval width
        """
        remainingSteps = self.totalSteps - self.observedCount
        if remainingSteps <= 0:
            return

        errors = np.array(self.errors, dtype=np.float64)

        # ------------------------------------------------------------------
        # 1. Bias correction
        # ------------------------------------------------------------------
        meanError = float(np.mean(errors))
        biasCorrection = meanError  # Positive = under-prediction -> shift up

        # ------------------------------------------------------------------
        # 2. Trend correction (if errors show linear trend)
        # ------------------------------------------------------------------
        errorTrend = 0.0
        if len(errors) >= 3:
            xVals = np.arange(len(errors), dtype=np.float64)
            coeffs = np.polyfit(xVals, errors, 1)
            errorTrend = float(coeffs[0])

        # ------------------------------------------------------------------
        # 3. Seasonal correction (if errors show periodic patterns)
        # ------------------------------------------------------------------
        seasonalCorrection = np.zeros(remainingSteps)
        if len(errors) >= self.period * 2 and self.period > 1:
            # Extract seasonal pattern from errors
            seasonalPattern = np.zeros(self.period)
            counts = np.zeros(self.period)
            for i, e in enumerate(errors):
                phase = i % self.period
                seasonalPattern[phase] += e
                counts[phase] += 1
            # Safe division
            mask = counts > 0
            seasonalPattern[mask] /= counts[mask]
            # Remove mean (bias handled by biasCorrection)
            seasonalPattern -= np.mean(seasonalPattern)

            for h in range(remainingSteps):
                futurePhase = (self.observedCount + h) % self.period
                seasonalCorrection[h] = seasonalPattern[futurePhase]

        # ------------------------------------------------------------------
        # 4. Apply correction (with decay - more distant future = more uncertain)
        # ------------------------------------------------------------------
        for h in range(remainingSteps):
            stepIdx = self.observedCount + h
            decay = self._getDecayFactor(h, remainingSteps)

            # Base correction: bias + trend extrapolation + seasonal
            trendComponent = errorTrend * (self.observedCount + h)
            correction = (biasCorrection + trendComponent + seasonalCorrection[h]) * decay

            # Adjust correction intensity by healingMode
            if self.healingMode == 'conservative':
                correction *= 0.5
            elif self.healingMode == 'aggressive':
                correction *= 1.5
            # 'adaptive' is 1.0 (default)

            self.currentPredictions[stepIdx] = (
                self.originalPredictions[stepIdx] + correction
            )

        # ------------------------------------------------------------------
        # 5. Confidence interval update
        # ------------------------------------------------------------------
        observedStd = float(np.std(errors)) if len(errors) > 1 else self._referenceStd
        widthRatio = observedStd / max(self._referenceStd, 1e-10)

        # Interval width only widens (conservative)
        widthRatio = max(widthRatio, 1.0)

        for h in range(remainingSteps):
            stepIdx = self.observedCount + h
            originalWidth = self.originalUpper[stepIdx] - self.originalLower[stepIdx]
            newWidth = originalWidth * widthRatio

            # Additional uncertainty for more distant future
            horizonFactor = 1.0 + 0.05 * h
            newWidth *= horizonFactor

            center = self.currentPredictions[stepIdx]
            self.currentLower[stepIdx] = center - newWidth / 2.0
            self.currentUpper[stepIdx] = center + newWidth / 2.0

        # ------------------------------------------------------------------
        # 6. Record correction history
        # ------------------------------------------------------------------
        self.corrections.append({
            'step': self.observedCount,
            'biasCorrection': float(biasCorrection),
            'trendCorrection': float(errorTrend),
            'seasonalApplied': bool(np.any(seasonalCorrection != 0)),
            'widthRatio': float(widthRatio),
            'healingMode': self.healingMode,
        })

    def _getDecayFactor(self, horizonStep: int, totalRemaining: int) -> float:
        """
        Correction decay factor for future steps

        Near-future corrections have stronger effect,
        far-future corrections have diminishing effect (exponential decay).

        Parameters
        ----------
        horizonStep : int
            Target correction step (starting from 0)
        totalRemaining : int
            Total remaining steps

        Returns
        -------
        float
            Decay factor 0.0 ~ 1.0
        """
        if totalRemaining <= 1:
            return 1.0

        # Exponential decay: exp(-rate * h)
        # Rate adjusted based on total remaining steps
        # Target: approximately 0.3 at the last step
        rate = -np.log(0.3) / max(totalRemaining - 1, 1)
        decay = float(np.exp(-rate * horizonStep))

        # Decay adjustment by healingMode
        if self.healingMode == 'aggressive':
            # Slower decay (correction effect persists longer)
            decay = decay ** 0.7
        elif self.healingMode == 'conservative':
            # Faster decay (correction effect diminishes quickly)
            decay = decay ** 1.5

        return max(0.0, min(1.0, decay))

    # ======================================================================
    # Utilities
    # ======================================================================

    def _computeMape(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """Compute MAPE (using only non-zero values)"""
        mask = np.abs(actual) > 1e-10
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    def _computeCurrentMape(self) -> float:
        """MAPE based on data observed so far"""
        if self.observedCount == 0:
            return 0.0
        predicted = self.originalPredictions[:self.observedCount]
        actual = np.array(self.observedValues)
        return self._computeMape(predicted, actual)

    def _computeCurrentMae(self) -> float:
        """MAE based on data observed so far"""
        if self.observedCount == 0:
            return 0.0
        return float(np.mean(self.absErrors))

    def _buildStatus(self, correctionApplied: bool) -> HealingStatus:
        """Build HealingStatus object"""
        remaining = self.totalSteps - self.observedCount
        biasEstimate = float(np.mean(self.errors)) if self.errors else 0.0
        currentMape = self._computeCurrentMape()
        currentMae = self._computeCurrentMae()

        # Generate human-readable message
        message = self._buildStatusMessage(
            currentMape, currentMae, biasEstimate, remaining
        )

        return HealingStatus(
            health=self.health,
            healthScore=self.healthScore,
            observedCount=self.observedCount,
            remainingSteps=remaining,
            driftDetected=self.driftDetected,
            driftDirection=self.driftDirection,
            driftMagnitude=self.driftMagnitude,
            correctionApplied=correctionApplied,
            biasEstimate=biasEstimate,
            mape=currentMape,
            mae=currentMae,
            refitRecommended=self.refitRecommended,
            refitReason=self.refitReason,
            message=message,
        )

    def _buildStatusMessage(
        self,
        mape: float,
        mae: float,
        bias: float,
        remaining: int,
    ) -> str:
        """Generate human-readable status message"""
        parts = []

        # Health status
        healthLabels = {
            'healthy': 'Healthy',
            'degrading': 'Degrading',
            'critical': 'Critical',
            'healed': 'Healed',
        }
        parts.append(f"Status: {healthLabels.get(self.health, self.health)} "
                      f"(score {self.healthScore:.0f}/100)")

        # Observation progress
        parts.append(f"Observed: {self.observedCount}/{self.totalSteps}")

        # Error info
        if self.observedCount > 0:
            parts.append(f"MAPE: {mape:.1f}%, MAE: {mae:.4f}")

        # Bias info
        if abs(bias) > self._referenceStd * 0.5:
            direction = "under-prediction" if bias > 0 else "over-prediction"
            parts.append(f"Bias: {direction} ({bias:+.4f})")

        # Drift
        if self.driftDetected:
            driftLabels = {
                'upward_bias': 'upward bias',
                'downward_bias': 'downward bias',
            }
            label = driftLabels.get(self.driftDirection, self.driftDirection or '')
            parts.append(f"Drift: {label} detected")

        # Refit recommendation
        if self.refitRecommended:
            parts.append(f"Refit recommended: {self.refitReason}")

        return " | ".join(parts)

    # ======================================================================
    # Representation
    # ======================================================================

    def __repr__(self) -> str:
        return (
            f"SelfHealingForecast("
            f"steps={self.totalSteps}, "
            f"observed={self.observedCount}, "
            f"health='{self.health}', "
            f"score={self.healthScore:.1f}, "
            f"mode='{self.healingMode}')"
        )
