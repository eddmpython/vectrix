"""
Level 4: Variability-preserving ensemble

Standard ensembles reduce variability through averaging.
This ensemble preserves variability while maintaining accuracy.
"""

from typing import Dict, List, Tuple

import numpy as np

from ...types import ModelResult


class VariabilityPreservingEnsemble:
    """
    Variability-preserving ensemble

    Core idea:
    1. Combine accuracy-based weights + variability preservation weights
    2. Reduce weights for flat prediction models
    3. Scale up if variability drops excessively after ensembling
    """

    def __init__(
        self,
        variabilityWeight: float = 0.3,
        minVariabilityRatio: float = 0.5,
        excludeFlatModels: bool = True
    ):
        """
        Parameters
        ----------
        variabilityWeight : float
            Variability preservation weight ratio (0.0 ~ 1.0)
        minVariabilityRatio : float
            Minimum variability ratio (relative to original)
        excludeFlatModels : bool
            Whether to exclude flat prediction models
        """
        self.variabilityWeight = variabilityWeight
        self.minVariabilityRatio = minVariabilityRatio
        self.excludeFlatModels = excludeFlatModels

    def ensemble(
        self,
        modelResults: Dict[str, ModelResult],
        originalData: np.ndarray,
        topK: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Perform variability-preserving ensemble

        Parameters
        ----------
        modelResults : Dict[str, ModelResult]
            Results by model
        originalData : np.ndarray
            Original data
        topK : int
            Use only top K models

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]
            (predictions, lower95, upper95, metadata)
        """
        originalStd = np.std(originalData[-min(30, len(originalData)):])

        # Filter valid models
        validResults = self._filterValidModels(modelResults)

        if not validResults:
            return self._fallbackPrediction(originalData, modelResults)

        # Sort by MAPE
        sortedModels = sorted(
            validResults.items(),
            key=lambda x: x[1].mape
        )[:topK]

        if not sortedModels:
            return self._fallbackPrediction(originalData, modelResults)

        # Calculate weights
        weights = self._calculateWeights(sortedModels, originalStd)

        # Weighted ensemble
        predictions = self._weightedAverage(sortedModels, weights)

        # Variability correction
        predictions = self._correctVariability(predictions, originalStd)

        # Calculate confidence interval
        lower95, upper95 = self._calculateConfidenceInterval(
            sortedModels, predictions, originalStd
        )

        metadata = {
            'modelsUsed': [m[0] for m in sortedModels],
            'weights': {m[0]: w for m, w in zip(sortedModels, weights)},
            'originalStd': originalStd,
            'ensembleStd': np.std(predictions),
            'variabilityRatio': np.std(predictions) / originalStd if originalStd > 0 else 0
        }

        return predictions, lower95, upper95, metadata

    def _filterValidModels(
        self,
        modelResults: Dict[str, ModelResult]
    ) -> Dict[str, ModelResult]:
        """Filter valid models"""
        valid = {}

        for modelId, result in modelResults.items():
            if modelId == 'ensemble':
                continue

            if not result.isValid:
                continue

            if self.excludeFlatModels and result.flatInfo:
                if result.flatInfo.isFlat:
                    continue

            if result.mape == float('inf') or np.isnan(result.mape):
                continue

            if len(result.predictions) == 0:
                continue

            valid[modelId] = result

        return valid

    def _calculateWeights(
        self,
        sortedModels: List[Tuple[str, ModelResult]],
        originalStd: float
    ) -> np.ndarray:
        """
        Calculate weights: accuracy x variability preservation

        Parameters
        ----------
        sortedModels : List[Tuple[str, ModelResult]]
            (modelID, result) tuple list
        originalStd : float
            Original data standard deviation

        Returns
        -------
        np.ndarray
            Normalized weights
        """
        n = len(sortedModels)
        accuracyWeights = np.zeros(n)
        variabilityWeights = np.zeros(n)

        for i, (modelId, result) in enumerate(sortedModels):
            # Accuracy weight (inverse MAPE)
            accuracyWeights[i] = 1.0 / (result.mape + 1e-6)

            # Variability preservation weight
            predStd = np.std(result.predictions)
            if originalStd > 0:
                varRatio = predStd / originalStd
                # Higher score for variability similar to original
                variabilityWeights[i] = 1.0 / (1.0 + abs(varRatio - 1.0))
            else:
                variabilityWeights[i] = 1.0

        # Normalize
        if accuracyWeights.sum() > 0:
            accuracyWeights /= accuracyWeights.sum()
        if variabilityWeights.sum() > 0:
            variabilityWeights /= variabilityWeights.sum()

        # Combine
        alpha = self.variabilityWeight
        finalWeights = (1 - alpha) * accuracyWeights + alpha * variabilityWeights

        # Re-normalize
        if finalWeights.sum() > 0:
            finalWeights /= finalWeights.sum()
        else:
            finalWeights = np.ones(n) / n

        return finalWeights

    def _weightedAverage(
        self,
        sortedModels: List[Tuple[str, ModelResult]],
        weights: np.ndarray
    ) -> np.ndarray:
        """Calculate weighted average"""
        # Unify prediction length
        predLength = min(len(m[1].predictions) for m in sortedModels)

        predictions = np.zeros(predLength)
        for i, (modelId, result) in enumerate(sortedModels):
            predictions += weights[i] * result.predictions[:predLength]

        return predictions

    def _correctVariability(
        self,
        predictions: np.ndarray,
        originalStd: float
    ) -> np.ndarray:
        """
        Variability correction

        Scale up if variability decreased too much after ensembling
        """
        ensembleStd = np.std(predictions)

        if originalStd <= 0 or ensembleStd <= 0:
            return predictions

        varRatio = ensembleStd / originalStd

        # Scale up if variability is below minimum ratio
        if varRatio < self.minVariabilityRatio:
            targetStd = originalStd * self.minVariabilityRatio * 0.8
            scaleFactor = targetStd / ensembleStd

            predMean = np.mean(predictions)
            predictions = predMean + (predictions - predMean) * scaleFactor

        return predictions

    def _calculateConfidenceInterval(
        self,
        sortedModels: List[Tuple[str, ModelResult]],
        predictions: np.ndarray,
        originalStd: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate confidence interval"""
        nPred = len(predictions)

        # Inter-model uncertainty
        modelPreds = [m[1].predictions[:nPred] for m in sortedModels]
        modelStd = np.std(modelPreds, axis=0)

        # Uncertainty increasing over time
        steps = np.arange(1, nPred + 1)
        timeUncertainty = originalStd * np.sqrt(steps) * 0.5

        # Combine
        totalUncertainty = np.sqrt(modelStd ** 2 + timeUncertainty ** 2)

        margin = 1.96 * totalUncertainty

        lower95 = predictions - margin
        upper95 = predictions + margin

        return lower95, upper95

    def _fallbackPrediction(
        self,
        originalData: np.ndarray,
        modelResults: Dict[str, ModelResult]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Fallback prediction (when no valid models available)"""
        # Select any available model
        if modelResults:
            firstResult = next(iter(modelResults.values()))
            predictions = firstResult.predictions
        else:
            # Last resort: Naive
            lastVal = originalData[-1]
            predictions = np.full(30, lastVal)

        originalStd = np.std(originalData[-30:])
        steps = np.arange(1, len(predictions) + 1)
        margin = 1.96 * originalStd * np.sqrt(steps)

        return (
            predictions,
            predictions - margin,
            predictions + margin,
            {'warning': 'No valid models available, using fallback prediction'}
        )


def quickEnsemble(
    modelPredictions: Dict[str, np.ndarray],
    modelMapes: Dict[str, float],
    originalStd: float
) -> np.ndarray:
    """
    Simple ensemble (without result objects)

    Parameters
    ----------
    modelPredictions : Dict[str, np.ndarray]
        Predictions by model
    modelMapes : Dict[str, float]
        MAPE by model
    originalStd : float
        Original data standard deviation

    Returns
    -------
    np.ndarray
        Ensemble predictions
    """
    if not modelPredictions:
        return np.array([])

    # Unify prediction length
    predLength = min(len(p) for p in modelPredictions.values())

    # MAPE-based weights
    weights = {}
    totalWeight = 0

    for modelId, pred in modelPredictions.items():
        mape = modelMapes.get(modelId, 100)
        w = 1.0 / (mape + 1e-6)

        # Variability bonus
        predStd = np.std(pred)
        if originalStd > 0:
            varRatio = predStd / originalStd
            varBonus = 1.0 / (1.0 + abs(varRatio - 1.0))
            w *= (0.7 + 0.3 * varBonus)

        weights[modelId] = w
        totalWeight += w

    # Normalize
    for modelId in weights:
        weights[modelId] /= totalWeight

    # Ensemble
    ensemble = np.zeros(predLength)
    for modelId, pred in modelPredictions.items():
        ensemble += weights[modelId] * pred[:predLength]

    return ensemble
