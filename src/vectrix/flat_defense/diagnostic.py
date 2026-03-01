"""
Level 1: Pre-diagnostic for flat predictions

Diagnoses the risk of flat predictions before forecasting.
"""

from typing import Dict, List, Optional

import numpy as np

from ..types import MODEL_INFO, DataCharacteristics, FlatRiskAssessment, RiskLevel


class FlatRiskDiagnostic:
    """
    Pre-diagnostic for flat prediction risk

    Analyzes data before forecasting to assess the likelihood of flat predictions.
    Recommends appropriate model selection strategies based on risk level.
    """

    def __init__(self, period: int = 7):
        self.period = period

    def diagnose(
        self,
        values: np.ndarray,
        characteristics: Optional[DataCharacteristics] = None
    ) -> FlatRiskAssessment:
        """
        Diagnose flat prediction risk

        Parameters
        ----------
        values : np.ndarray
            Time series data
        characteristics : DataCharacteristics, optional
            Pre-analyzed data characteristics (analyzed simply if not provided)

        Returns
        -------
        FlatRiskAssessment
            Risk assessment result
        """
        n = len(values)

        if n < 4:
            return FlatRiskAssessment(
                riskScore=1.0,
                riskLevel=RiskLevel.CRITICAL,
                riskFactors={'shortData': True},
                warnings=['Insufficient data (minimum 4 required)'],
                recommendedStrategy='naive_only',
                recommendedModels=['naive']
            )

        riskFactors = {
            'lowVariance': self._checkLowVariance(values),
            'weakSeasonality': self._checkWeakSeasonality(values),
            'noTrend': self._checkNoTrend(values),
            'shortData': self._checkShortData(values),
            'highNoise': self._checkHighNoise(values),
            'flatRecent': self._checkFlatRecent(values)
        }

        weights = {
            'lowVariance': 0.25,
            'weakSeasonality': 0.20,
            'noTrend': 0.15,
            'shortData': 0.15,
            'highNoise': 0.15,
            'flatRecent': 0.10
        }

        riskScore = sum(
            weights[k] * (1.0 if v else 0.0)
            for k, v in riskFactors.items()
        )

        if riskScore >= 0.7:
            riskLevel = RiskLevel.CRITICAL
        elif riskScore >= 0.5:
            riskLevel = RiskLevel.HIGH
        elif riskScore >= 0.3:
            riskLevel = RiskLevel.MEDIUM
        else:
            riskLevel = RiskLevel.LOW

        strategy, models = self._getRecommendation(riskLevel, riskFactors, n)
        warnings = self._generateWarnings(riskFactors, riskLevel)

        return FlatRiskAssessment(
            riskScore=riskScore,
            riskLevel=riskLevel,
            riskFactors=riskFactors,
            recommendedStrategy=strategy,
            recommendedModels=models,
            warnings=warnings
        )

    def _checkLowVariance(self, values: np.ndarray) -> bool:
        """Check if variability is too low"""
        std = np.std(values)
        mean = np.mean(np.abs(values))

        if mean < 1e-10:
            return True

        cv = std / mean
        return cv < 0.05

    def _checkWeakSeasonality(self, values: np.ndarray) -> bool:
        """Check if seasonality is weak"""
        n = len(values)
        period = self.period

        if n < period * 2:
            return True

        try:
            seasonalMeans = []
            for i in range(period):
                indices = list(range(i, n, period))
                if indices:
                    seasonalMeans.append(np.mean(values[indices]))

            if not seasonalMeans:
                return True

            seasonalVar = np.var(seasonalMeans)
            totalVar = np.var(values)

            if totalVar < 1e-10:
                return True

            seasonalStrength = seasonalVar / totalVar
            return seasonalStrength < 0.15

        except Exception:
            return True

    def _checkNoTrend(self, values: np.ndarray) -> bool:
        """Check if there is no trend"""
        n = len(values)

        if n < 10:
            return True

        try:
            x = np.arange(n)
            slope, _ = np.polyfit(x, values, 1)

            valueRange = np.max(values) - np.min(values)
            if valueRange < 1e-10:
                return True

            trendStrength = abs(slope * n) / valueRange
            return trendStrength < 0.1

        except Exception:
            return True

    def _checkShortData(self, values: np.ndarray) -> bool:
        """Check if data is insufficient"""
        n = len(values)
        minRequired = self.period * 2
        return n < max(minRequired, 20)

    def _checkHighNoise(self, values: np.ndarray) -> bool:
        """Check if noise is excessive"""
        n = len(values)

        if n < 5:
            return False

        try:
            # Estimate trend via moving average
            windowSize = min(5, n // 3)
            if windowSize < 2:
                return False

            smoothed = np.convolve(
                values,
                np.ones(windowSize) / windowSize,
                mode='valid'
            )

            # Difference between original and smoothed data (noise)
            startIdx = windowSize // 2
            endIdx = startIdx + len(smoothed)
            noise = values[startIdx:endIdx] - smoothed

            noiseRatio = np.std(noise) / (np.std(values) + 1e-10)
            return noiseRatio > 0.7

        except Exception:
            return False

    def _checkFlatRecent(self, values: np.ndarray) -> bool:
        """Check if recent data is flat"""
        n = len(values)
        recentN = min(10, n // 2)

        if recentN < 3:
            return False

        recent = values[-recentN:]
        recentStd = np.std(recent)
        totalStd = np.std(values)

        if totalStd < 1e-10:
            return True

        return (recentStd / totalStd) < 0.3

    def _getRecommendation(
        self,
        riskLevel: RiskLevel,
        riskFactors: Dict[str, bool],
        dataLength: int
    ) -> tuple:
        """Strategy and model recommendation by risk level"""

        if riskLevel == RiskLevel.CRITICAL:
            strategy = "force_seasonal"
            models = ['seasonal_naive', 'snaive_drift']

        elif riskLevel == RiskLevel.HIGH:
            strategy = "seasonal_priority"
            models = ['seasonal_naive', 'snaive_drift', 'mstl', 'theta']

        elif riskLevel == RiskLevel.MEDIUM:
            strategy = "balanced"
            models = ['mstl', 'holt_winters', 'theta', 'auto_arima']

        else:
            strategy = "standard"
            models = ['auto_arima', 'auto_ets', 'theta', 'mstl']

        models = [
            m for m in models
            if dataLength >= MODEL_INFO.get(m, {}).get('minData', 10)
        ]

        if not models:
            models = ['seasonal_naive'] if dataLength >= 7 else ['naive']

        return strategy, models

    def _generateWarnings(
        self,
        riskFactors: Dict[str, bool],
        riskLevel: RiskLevel
    ) -> List[str]:
        """Generate warning messages"""
        warnings = []

        warningMessages = {
            'lowVariance': 'Data variability is very low. Predictions may be flat.',
            'weakSeasonality': 'No clear seasonal pattern detected.',
            'noTrend': 'No upward/downward trend detected.',
            'shortData': 'Insufficient data. A longer time period is needed.',
            'highNoise': 'High noise makes pattern detection difficult.',
            'flatRecent': 'Recent data is flat. Possible pattern change.'
        }

        for factor, isRisk in riskFactors.items():
            if isRisk and factor in warningMessages:
                warnings.append(warningMessages[factor])

        if riskLevel in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            warnings.insert(0, f'Flat prediction risk level: {riskLevel.value.upper()}')

        return warnings
