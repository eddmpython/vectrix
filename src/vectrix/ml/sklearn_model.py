"""
Sklearn Generic Forecaster

Optional dependency: scikit-learn
Wraps any sklearn regressor for time series forecasting.
"""

from typing import Any, Optional, Tuple

import numpy as np

try:
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..regression.features import FourierFeatures, autoFeatureEngineering


class SklearnForecaster:
    """
    Generic sklearn-based time series forecaster

    Wraps any sklearn regressor that has .fit(X, y) and .predict(X).
    Requires: pip install scikit-learn
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        period: int = 7,
        maxLag: int = 14
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required. Install: pip install scikit-learn")

        if estimator is None:
            from sklearn.ensemble import GradientBoostingRegressor
            self.estimator = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        else:
            self.estimator = estimator

        self.period = period
        self.maxLag = maxLag
        self.metadata = None
        self.fitted = False
        self._y = None

    def fit(self, y: np.ndarray) -> 'SklearnForecaster':
        n = len(y)
        self._y = y.copy()

        X, target, self.metadata = autoFeatureEngineering(
            y, period=self.period, maxLag=min(self.maxLag, n // 3)
        )

        if len(target) < 10:
            self.fitted = True
            return self

        self.estimator.fit(X, target)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted.")

        predictions = np.zeros(steps)
        extendedY = list(self._y)

        for h in range(steps):
            yArr = np.array(extendedY)
            features = self._buildFeatures(yArr, len(yArr) + h)
            pred = self.estimator.predict(features.reshape(1, -1))[0]
            predictions[h] = pred
            extendedY.append(pred)

        sigma = np.std(np.diff(self._y[-min(30, len(self._y)):]))
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def _buildFeatures(self, y: np.ndarray, timeIdx: int) -> np.ndarray:
        meta = self.metadata
        lagGen = meta['lagGen']
        rollGen = meta['rollGen']
        period = meta['period']
        useFourier = meta['useFourier']

        parts = [lagGen.transformLast(y), rollGen.transformLast(y)]

        if useFourier and period > 1:
            nTerms = min(3, period // 2)
            fourierGen = FourierFeatures(period=period, nTerms=nTerms)
            parts.append(fourierGen.transformSingle(timeIdx))

        return np.concatenate(parts)
