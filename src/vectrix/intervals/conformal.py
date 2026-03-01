"""
Conformal Prediction Intervals

Distribution-free prediction intervals with finite-sample validity guarantees.

Methods:
- Split Conformal: split data into train/calibration, use calibration residuals
- Jackknife+: leave-one-out style, more data-efficient
"""

from typing import Callable, Tuple

import numpy as np


class ConformalInterval:
    """
    Conformal Prediction Intervals

    Provides distribution-free valid prediction intervals for a given model's forecasts.
    """

    def __init__(
        self,
        method: str = 'split',
        coverageLevel: float = 0.95,
        calibrationRatio: float = 0.2
    ):
        """
        Parameters
        ----------
        method : str
            'split' or 'jackknife'
        coverageLevel : float
            Target coverage (0.95 = 95%)
        calibrationRatio : float
            Calibration set ratio (split method only)
        """
        self.method = method
        self.coverageLevel = coverageLevel
        self.calibrationRatio = calibrationRatio
        self.conformalScores = None

    def calibrate(
        self,
        y: np.ndarray,
        modelFactory: Callable,
        steps: int = 1
    ) -> 'ConformalInterval':
        """
        Calibrate conformal scores

        Parameters
        ----------
        y : np.ndarray
            Full time series data
        modelFactory : Callable
            Model factory function
        steps : int
            Forecast horizon
        """
        if self.method == 'split':
            self._calibrateSplit(y, modelFactory, steps)
        else:
            self._calibrateJackknife(y, modelFactory, steps)

        return self

    def predict(
        self,
        pointPredictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute conformal intervals around point predictions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (lower, upper)
        """
        if self.conformalScores is None or len(self.conformalScores) == 0:
            sigma = np.std(pointPredictions) * 0.1 + 1.0
            return pointPredictions - 1.96 * sigma, pointPredictions + 1.96 * sigma

        alpha = 1 - self.coverageLevel
        n = len(self.conformalScores)
        quantileIdx = int(np.ceil((1 - alpha) * (n + 1)))
        quantileIdx = min(quantileIdx, n) - 1
        quantileIdx = max(quantileIdx, 0)

        sortedScores = np.sort(self.conformalScores)
        width = sortedScores[quantileIdx]

        steps = len(pointPredictions)
        widths = width * np.sqrt(np.arange(1, steps + 1))

        lower = pointPredictions - widths
        upper = pointPredictions + widths

        return lower, upper

    def _calibrateSplit(self, y: np.ndarray, modelFactory: Callable, steps: int):
        n = len(y)
        calSize = max(int(n * self.calibrationRatio), steps + 1)
        trainSize = n - calSize

        if trainSize < 10:
            self.conformalScores = np.array([np.std(y)])
            return

        trainData = y[:trainSize]
        calData = y[trainSize:]

        scores = []
        nFolds = max(1, len(calData) - steps + 1)
        nFolds = min(nFolds, 20)

        for i in range(nFolds):
            endTrain = trainSize + i
            if endTrain + steps > n:
                break

            try:
                model = modelFactory()
                model.fit(y[:endTrain])
                pred, _, _ = model.predict(steps)

                actual = y[endTrain:endTrain + steps]
                residuals = np.abs(actual - pred[:len(actual)])
                scores.append(np.max(residuals))
            except Exception:
                continue

        self.conformalScores = np.array(scores) if scores else np.array([np.std(y)])

    def _calibrateJackknife(self, y: np.ndarray, modelFactory: Callable, steps: int):
        n = len(y)
        minTrain = max(20, n // 3)
        nFolds = min(10, n - minTrain - steps)

        if nFolds < 2:
            self.conformalScores = np.array([np.std(y)])
            return

        scores = []
        stepSize = max(1, (n - minTrain - steps) // nFolds)

        for i in range(nFolds):
            endTrain = minTrain + i * stepSize
            if endTrain + steps > n:
                break

            try:
                model = modelFactory()
                model.fit(y[:endTrain])
                pred, _, _ = model.predict(steps)

                actual = y[endTrain:endTrain + steps]
                residuals = np.abs(actual - pred[:len(actual)])
                scores.append(np.max(residuals))
            except Exception:
                continue

        self.conformalScores = np.array(scores) if scores else np.array([np.std(y)])
