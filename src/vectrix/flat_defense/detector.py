"""
Level 3: Flat prediction detection

Detects whether prediction results are flat after forecasting.
"""

from typing import Optional

import numpy as np

from ..types import FlatPredictionInfo, FlatPredictionType


class FlatPredictionDetector:
    """
    Flat prediction detector

    Analyzes prediction results to determine if they are flat (horizontal, diagonal, mean reversion).
    """

    def __init__(
        self,
        horizontalThreshold: float = 0.01,
        diagonalThreshold: float = 1e-8,
        varianceThreshold: float = 0.0001
    ):
        """
        Parameters
        ----------
        horizontalThreshold : float
            Horizontal flat threshold (prediction std / original std)
        diagonalThreshold : float
            Diagonal flat threshold (variance of differences)
        varianceThreshold : float
            Relative variance threshold
        """
        self.horizontalThreshold = horizontalThreshold
        self.diagonalThreshold = diagonalThreshold
        self.varianceThreshold = varianceThreshold

    def detect(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        originalStd: Optional[float] = None
    ) -> FlatPredictionInfo:
        """
        Detect flat predictions

        Parameters
        ----------
        predictions : np.ndarray
            Predicted values
        originalData : np.ndarray
            Original data
        originalStd : float, optional
            Original data standard deviation (calculated if not provided)

        Returns
        -------
        FlatPredictionInfo
            Detection result
        """
        if len(predictions) < 3:
            return FlatPredictionInfo(
                isFlat=False,
                flatType=FlatPredictionType.NONE,
                message='Prediction length too short for detection'
            )

        if originalStd is None:
            originalStd = np.std(originalData)

        predStd = np.std(predictions)
        predVar = np.var(predictions)
        predMean = np.mean(np.abs(predictions))

        # Horizontal flat detection
        if originalStd > 0:
            stdRatio = predStd / originalStd
            if stdRatio < self.horizontalThreshold:
                return FlatPredictionInfo(
                    isFlat=True,
                    flatType=FlatPredictionType.HORIZONTAL,
                    predictionStd=predStd,
                    originalStd=originalStd,
                    stdRatio=stdRatio,
                    message='Horizontal flat prediction detected: model failed to capture seasonality/variation',
                    suggestion='Consider using Seasonal Naive or MSTL model'
                )

        # Horizontal flat detection via relative variance
        if predMean > 0:
            relativeVar = predVar / (predMean ** 2)
            if relativeVar < self.varianceThreshold:
                return FlatPredictionInfo(
                    isFlat=True,
                    flatType=FlatPredictionType.HORIZONTAL,
                    predictionStd=predStd,
                    originalStd=originalStd,
                    varianceRatio=relativeVar,
                    message='Horizontal flat prediction detected: predictions barely change',
                    suggestion='Consider forced seasonal pattern injection'
                )

        # Diagonal flat detection (constant slope)
        diffs = np.diff(predictions)
        diffVar = np.var(diffs)

        if diffVar < self.diagonalThreshold and predVar > 1e-6:
            return FlatPredictionInfo(
                isFlat=True,
                flatType=FlatPredictionType.DIAGONAL,
                predictionStd=predStd,
                originalStd=originalStd,
                varianceRatio=diffVar,
                message='Diagonal flat prediction detected: only trend captured, no seasonality',
                suggestion='Consider adding seasonal variation'
            )

        # Mean reversion detection (variation decrease in long-horizon forecasts)
        if len(predictions) >= 10:
            firstHalfStd = np.std(predictions[:len(predictions)//2])
            secondHalfStd = np.std(predictions[len(predictions)//2:])

            if firstHalfStd > 0 and secondHalfStd / firstHalfStd < 0.3:
                return FlatPredictionInfo(
                    isFlat=True,
                    flatType=FlatPredictionType.MEAN_REVERSION,
                    predictionStd=predStd,
                    originalStd=originalStd,
                    stdRatio=secondHalfStd / firstHalfStd,
                    message='Mean reversion detected: variation drops sharply in long-horizon forecast',
                    suggestion='Consider shortening forecast horizon or expanding uncertainty'
                )

        return FlatPredictionInfo(
            isFlat=False,
            flatType=FlatPredictionType.NONE,
            predictionStd=predStd,
            originalStd=originalStd,
            stdRatio=predStd / originalStd if originalStd > 0 else 0
        )

    def detectMultiple(
        self,
        modelPredictions: dict,
        originalData: np.ndarray
    ) -> dict:
        """
        Batch detection of multiple model predictions

        Parameters
        ----------
        modelPredictions : dict
            {modelID: predictions} dictionary
        originalData : np.ndarray
            Original data

        Returns
        -------
        dict
            {modelID: FlatPredictionInfo} dictionary
        """
        originalStd = np.std(originalData)
        results = {}

        for modelId, predictions in modelPredictions.items():
            results[modelId] = self.detect(predictions, originalData, originalStd)

        return results

    def getFlatModels(self, detectionResults: dict) -> list:
        """Return list of models that produced flat predictions"""
        return [
            modelId
            for modelId, info in detectionResults.items()
            if info.isFlat
        ]

    def getValidModels(self, detectionResults: dict) -> list:
        """Return list of models that produced valid predictions"""
        return [
            modelId
            for modelId, info in detectionResults.items()
            if not info.isFlat
        ]
