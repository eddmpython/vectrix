"""
Level 3-4: Flat prediction correction

Intelligently corrects predictions when flat patterns are detected.
"""

from typing import Optional, Tuple

import numpy as np

from ..types import FlatPredictionInfo, FlatPredictionType


class FlatPredictionCorrector:
    """
    Flat prediction corrector

    When flat predictions are detected, corrects them using patterns from the original data.
    Not simply adding noise, but correcting based on actual patterns.
    """

    def __init__(
        self,
        seasonalStrength: float = 0.5,
        variationStrength: float = 0.3,
        maxCorrection: float = 0.5
    ):
        """
        Parameters
        ----------
        seasonalStrength : float
            Seasonal pattern injection strength (0.0 ~ 1.0)
        variationStrength : float
            Variation addition strength (0.0 ~ 1.0)
        maxCorrection : float
            Maximum correction ratio (relative to original std)
        """
        self.seasonalStrength = seasonalStrength
        self.variationStrength = variationStrength
        self.maxCorrection = maxCorrection

    def correct(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        flatInfo: FlatPredictionInfo,
        period: int = 7
    ) -> Tuple[np.ndarray, FlatPredictionInfo]:
        """
        Correct flat predictions

        Parameters
        ----------
        predictions : np.ndarray
            Original predictions
        originalData : np.ndarray
            Original time series data
        flatInfo : FlatPredictionInfo
            Flat detection info
        period : int
            Seasonal period

        Returns
        -------
        Tuple[np.ndarray, FlatPredictionInfo]
            (corrected predictions, updated detection info)
        """
        if not flatInfo.isFlat:
            return predictions, flatInfo

        corrected = predictions.copy()
        correctionMethod = ""

        if flatInfo.flatType == FlatPredictionType.HORIZONTAL:
            corrected, correctionMethod = self._correctHorizontal(
                predictions, originalData, period
            )

        elif flatInfo.flatType == FlatPredictionType.DIAGONAL:
            corrected, correctionMethod = self._correctDiagonal(
                predictions, originalData, period
            )

        elif flatInfo.flatType == FlatPredictionType.MEAN_REVERSION:
            corrected, correctionMethod = self._correctMeanReversion(
                predictions, originalData, period
            )

        updatedInfo = FlatPredictionInfo(
            isFlat=flatInfo.isFlat,
            flatType=flatInfo.flatType,
            predictionStd=flatInfo.predictionStd,
            originalStd=flatInfo.originalStd,
            stdRatio=flatInfo.stdRatio,
            varianceRatio=flatInfo.varianceRatio,
            correctionApplied=True,
            correctionMethod=correctionMethod,
            correctionStrength=self.seasonalStrength,
            message=flatInfo.message,
            suggestion=f'Correction applied: {correctionMethod}'
        )

        return corrected, updatedInfo

    def _correctHorizontal(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        period: int
    ) -> Tuple[np.ndarray, str]:
        """
        Horizontal flat correction: seasonal pattern injection

        Extracts seasonal patterns from original data and injects into predictions.
        """
        seasonal = self._extractSeasonalPattern(originalData, period)

        if seasonal is None:
            return self._addSimpleVariation(predictions, originalData), "simple_variation"

        # Repeat seasonal pattern to match prediction length
        nPred = len(predictions)
        seasonalExtended = np.tile(seasonal, nPred // len(seasonal) + 1)[:nPred]

        # Inject with adjusted strength
        originalStd = np.std(originalData)
        maxAdjustment = originalStd * self.maxCorrection

        # Scale seasonal pattern
        seasonalAdjustment = seasonalExtended * self.seasonalStrength
        seasonalAdjustment = np.clip(seasonalAdjustment, -maxAdjustment, maxAdjustment)

        corrected = predictions + seasonalAdjustment

        return corrected, "seasonal_injection"

    def _correctDiagonal(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        period: int
    ) -> Tuple[np.ndarray, str]:
        """
        Diagonal flat correction: add seasonal variation

        Adds seasonal variation while preserving the trend.
        """
        seasonal = self._extractSeasonalPattern(originalData, period)

        if seasonal is None:
            return self._addSimpleVariation(predictions, originalData), "simple_variation"

        # Preserve prediction trend
        nPred = len(predictions)
        trend = np.linspace(predictions[0], predictions[-1], nPred)

        # Add seasonal pattern
        seasonalExtended = np.tile(seasonal, nPred // len(seasonal) + 1)[:nPred]

        originalStd = np.std(originalData)
        seasonalAdjustment = seasonalExtended * self.variationStrength * originalStd / (np.std(seasonal) + 1e-10)

        corrected = trend + seasonalAdjustment

        return corrected, "trend_plus_seasonal"

    def _correctMeanReversion(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        period: int
    ) -> Tuple[np.ndarray, str]:
        """
        Mean reversion correction: preserve variability

        Corrects to maintain variability even in long-horizon forecasts.
        """
        nPred = len(predictions)
        originalStd = np.std(originalData[-min(30, len(originalData)):])

        # Calculate current prediction variability
        predStd = np.std(predictions)

        if predStd < 1e-10:
            # Inject seasonal pattern if completely flat
            return self._correctHorizontal(predictions, originalData, period)

        # Target variability (maintain a fraction of original)
        targetStd = originalStd * 0.7

        # Variability scaling
        predMean = np.mean(predictions)
        scaleFactor = targetStd / predStd

        corrected = predMean + (predictions - predMean) * scaleFactor

        return corrected, "variance_scaling"

    def _extractSeasonalPattern(
        self,
        data: np.ndarray,
        period: int
    ) -> Optional[np.ndarray]:
        """
        Extract seasonal pattern from data (simple average-based)
        """
        n = len(data)

        if n < period:
            return None

        # Extract seasonal pattern from recent data
        recentData = data[-min(period * 3, n):]
        nRecent = len(recentData)

        # Calculate per-period average
        seasonal = np.zeros(period)
        counts = np.zeros(period)

        for i in range(nRecent):
            idx = i % period
            seasonal[idx] += recentData[i]
            counts[idx] += 1

        counts[counts == 0] = 1
        seasonal = seasonal / counts

        # Remove mean (seasonal component only)
        seasonal = seasonal - np.mean(seasonal)

        return seasonal

    def _addSimpleVariation(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray
    ) -> np.ndarray:
        """
        Add simple variation (when seasonal pattern extraction fails)

        Not purely random, but mimics variation patterns from recent data.
        """
        n = len(predictions)
        originalStd = np.std(originalData[-min(30, len(originalData)):])

        # Recent variation pattern
        recentDiffs = np.diff(originalData[-min(n + 10, len(originalData)):])

        if len(recentDiffs) < n:
            # Repeat pattern
            recentDiffs = np.tile(recentDiffs, n // len(recentDiffs) + 1)[:n]
        else:
            recentDiffs = recentDiffs[:n]

        # Adjust variation scale
        variation = recentDiffs * self.variationStrength
        maxVar = originalStd * self.maxCorrection
        variation = np.clip(variation, -maxVar, maxVar)

        corrected = predictions.copy()
        for i in range(1, n):
            corrected[i] = corrected[i - 1] + variation[i - 1]

        # Level adjustment (preserve original prediction mean)
        corrected = corrected - np.mean(corrected) + np.mean(predictions)

        return corrected


def correctWithConfidenceInterval(
    predictions: np.ndarray,
    lower95: np.ndarray,
    upper95: np.ndarray,
    flatInfo: FlatPredictionInfo,
    originalData: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correct confidence intervals along with predictions

    Expands confidence intervals for flat predictions.
    """
    if not flatInfo.isFlat:
        return predictions, lower95, upper95

    corrector = FlatPredictionCorrector()
    correctedPred, _ = corrector.correct(
        predictions, originalData, flatInfo
    )

    # Expand confidence intervals
    originalStd = np.std(originalData)
    steps = np.arange(1, len(predictions) + 1)

    # Greater uncertainty for flat predictions
    uncertaintyMultiplier = 1.5 if flatInfo.flatType != FlatPredictionType.NONE else 1.0

    margin = 1.96 * originalStd * np.sqrt(steps) * uncertaintyMultiplier

    correctedLower = correctedPred - margin
    correctedUpper = correctedPred + margin

    return correctedPred, correctedLower, correctedUpper
