"""
LightGBM Forecaster

Optional dependency: lightgbm
Uses recursive reduction with auto feature engineering.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..regression.features import FourierFeatures, autoFeatureEngineering


class LightGBMForecaster:
    """
    LightGBM-based time series forecaster

    Recursive strategy with auto feature engineering.
    Requires: pip install lightgbm
    """

    def __init__(
        self,
        period: int = 7,
        maxLag: int = 14,
        params: Optional[Dict[str, Any]] = None
    ):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("lightgbm is required. Install: pip install lightgbm")

        self.period = period
        self.maxLag = maxLag
        self.params = params or {
            'objective': 'regression',
            'metric': 'mape',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'verbose': -1,
        }
        self.model = None
        self.metadata = None
        self.fitted = False
        self._y = None

    def fit(self, y: np.ndarray) -> 'LightGBMForecaster':
        n = len(y)
        self._y = y.copy()

        X, target, self.metadata = autoFeatureEngineering(
            y, period=self.period, maxLag=min(self.maxLag, n // 3)
        )

        if len(target) < 10:
            self.fitted = True
            return self

        nEstimators = self.params.pop('n_estimators', 100)
        self.model = lgb.LGBMRegressor(n_estimators=nEstimators, **self.params)
        self.model.fit(X, target)

        self.params['n_estimators'] = nEstimators
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted.")

        if self.model is None:
            pred = np.full(steps, self._y[-1])
            sigma = np.std(self._y[-30:])
            margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
            return pred, pred - margin, pred + margin

        predictions = np.zeros(steps)
        extendedY = list(self._y)

        for h in range(steps):
            yArr = np.array(extendedY)
            features = self._buildFeatures(yArr, len(yArr) + h)
            pred = self.model.predict(features.reshape(1, -1))[0]
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
