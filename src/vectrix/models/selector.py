"""
Level 2: Adaptive model selector

Selects optimal models based on flat prediction risk and data characteristics.
"""

from typing import Any, Dict, List, Optional

from ..engine.registry import getRegistry
from ..types import MODEL_INFO, DataCharacteristics, FlatRiskAssessment, RiskLevel


def _buildFromRegistry(field: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Build lookup dict from registry, with fallback for unregistered models."""
    result = dict(fallback)
    for modelId, spec in getRegistry().items():
        result[modelId] = getattr(spec, field)
    return result


_EXTRA_FLAT_RESISTANCE = {
    'snaive_drift': 0.90,
    'holt_winters': 0.80,
    'prophet': 0.70,
    'auto_theta': 0.70,
}

_EXTRA_MIN_DATA = {
    'snaive_drift': 14,
    'holt_winters': 24,
    'prophet': 60,
    'auto_theta': 10,
}


class AdaptiveModelSelector:
    """
    Adaptive model selector

    Prioritizes seasonal-forcing models (e.g. Seasonal Naive) when flat prediction
    risk is high, and uses Auto models when risk is low.

    FLAT_RESISTANCE and MIN_DATA are populated from engine/registry.py at init time.
    Extra entries for unregistered models (snaive_drift, holt_winters, prophet, auto_theta)
    are provided as fallbacks.
    """

    def __init__(self):
        self.FLAT_RESISTANCE = _buildFromRegistry('flatResistance', _EXTRA_FLAT_RESISTANCE)
        self.MIN_DATA = _buildFromRegistry('minData', _EXTRA_MIN_DATA)

    def selectModels(
        self,
        flatRisk: FlatRiskAssessment,
        characteristics: DataCharacteristics,
        maxModels: int = 5
    ) -> List[str]:
        """
        Risk-based model selection

        Parameters
        ----------
        flatRisk : FlatRiskAssessment
            Flat prediction risk assessment result
        characteristics : DataCharacteristics
            Data characteristics
        maxModels : int
            Maximum number of models

        Returns
        -------
        List[str]
            List of selected model IDs
        """
        dataLength = characteristics.length
        riskLevel = flatRisk.riskLevel

        # Base model set by risk level
        if riskLevel == RiskLevel.CRITICAL:
            candidates = self._getCriticalModels()
        elif riskLevel == RiskLevel.HIGH:
            candidates = self._getHighRiskModels()
        elif riskLevel == RiskLevel.MEDIUM:
            candidates = self._getMediumRiskModels()
        else:
            candidates = self._getLowRiskModels()

        # Filter by data length
        validModels = [
            m for m in candidates
            if dataLength >= self.MIN_DATA.get(m, 10)
        ]

        # Adjust by seasonality
        if characteristics.hasSeasonality:
            validModels = self._prioritizeSeasonalModels(validModels)
        else:
            # Exclude seasonal-forcing models if no seasonality
            validModels = [
                m for m in validModels
                if m not in ['seasonal_naive', 'snaive_drift']
            ]

        # Fallback
        if not validModels:
            validModels = ['theta'] if dataLength >= 10 else ['naive']

        return validModels[:maxModels]

    def _getCriticalModels(self) -> List[str]:
        """Critical risk: seasonal-forcing models only"""
        return ['seasonal_naive', 'snaive_drift', 'mstl']

    def _getHighRiskModels(self) -> List[str]:
        """High risk: seasonal models prioritized"""
        return ['seasonal_naive', 'snaive_drift', 'mstl', 'theta', 'holt_winters']

    def _getMediumRiskModels(self) -> List[str]:
        """Medium risk: balanced"""
        return ['mstl', 'holt_winters', 'theta', 'auto_arima', 'auto_theta']

    def _getLowRiskModels(self) -> List[str]:
        """Low risk: standard Auto models"""
        return ['auto_arima', 'auto_ets', 'auto_theta', 'theta', 'mstl']

    def _prioritizeSeasonalModels(self, models: List[str]) -> List[str]:
        """Prioritize seasonal models when seasonality is present"""
        seasonalModels = ['mstl', 'seasonal_naive', 'snaive_drift', 'holt_winters']
        otherModels = [m for m in models if m not in seasonalModels]
        prioritized = [m for m in seasonalModels if m in models] + otherModels
        return prioritized

    def getModelInfo(self, modelId: str) -> Dict[str, Any]:
        """Return model information"""
        return MODEL_INFO.get(modelId, {
            'name': modelId,
            'description': '',
            'flatResistance': 0.5,
            'bestFor': [],
            'minData': 10
        })

    def rankByFlatResistance(self, models: List[str]) -> List[str]:
        """Sort by flat resistance score"""
        return sorted(
            models,
            key=lambda m: self.FLAT_RESISTANCE.get(m, 0.5),
            reverse=True
        )

    def getRecommendation(
        self,
        flatRisk: FlatRiskAssessment,
        characteristics: DataCharacteristics
    ) -> Dict[str, Any]:
        """
        Detailed model recommendation

        Returns
        -------
        Dict
            Recommended model list with reasons
        """
        models = self.selectModels(flatRisk, characteristics)

        recommendations = []
        for i, modelId in enumerate(models):
            info = self.getModelInfo(modelId)
            recommendations.append({
                'rank': i + 1,
                'modelId': modelId,
                'name': info.get('name', modelId),
                'flatResistance': self.FLAT_RESISTANCE.get(modelId, 0.5),
                'reason': self._getRecommendationReason(
                    modelId, flatRisk.riskLevel, characteristics
                )
            })

        return {
            'riskLevel': flatRisk.riskLevel.value,
            'strategy': flatRisk.recommendedStrategy,
            'models': recommendations,
            'warning': self._getStrategyWarning(flatRisk.riskLevel)
        }

    def _getRecommendationReason(
        self,
        modelId: str,
        riskLevel: RiskLevel,
        characteristics: DataCharacteristics
    ) -> str:
        """Generate recommendation reason"""
        reasons = {
            'seasonal_naive': 'Prevents flat predictions by forcing seasonal pattern repetition',
            'snaive_drift': 'Stable forecasts with seasonal pattern + trend',
            'mstl': 'Captures complex patterns via multiple seasonal decomposition',
            'holt_winters': 'Separates level/trend/seasonality modeling',
            'theta': 'M3 Competition winner, general-purpose performance',
            'auto_arima': 'Optimal ARIMA via automatic parameter tuning',
            'auto_ets': 'Automatic best selection among 30 combinations',
            'auto_theta': 'Automatic Theta decomposition',
            'prophet': 'Robust to outliers with holiday effects'
        }

        baseReason = reasons.get(modelId, 'General-purpose model')

        if riskLevel in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if self.FLAT_RESISTANCE.get(modelId, 0) >= 0.8:
                baseReason += ' (flat prediction risk mitigation)'

        return baseReason

    def _getStrategyWarning(self, riskLevel: RiskLevel) -> Optional[str]:
        """Strategy warning message"""
        if riskLevel == RiskLevel.CRITICAL:
            return 'Flat prediction risk is very high. Only seasonal-forcing models will be used.'
        elif riskLevel == RiskLevel.HIGH:
            return 'Flat prediction risk is high. Seasonal models will be prioritized.'
        elif riskLevel == RiskLevel.MEDIUM:
            return 'Flat prediction is possible. Review results carefully.'
        return None
