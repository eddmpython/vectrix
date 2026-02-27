"""
Reduction Strategies for Time Series Forecasting

Convert any regressor into a forecaster:
- DirectReduction: one model per forecast horizon
- RecursiveReduction: single model, iterate forward
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, List

from .features import LagFeatures, RollingFeatures, FourierFeatures, autoFeatureEngineering
from .linear import LinearRegressor, RidgeRegressor


class DirectReduction:
    """
    Direct Multi-step Forecasting

    각 예측 시점 h에 대해 별도 모델 학습:
    model_h: X_t → y_{t+h}
    """

    def __init__(self, modelFactory: Optional[Callable] = None, period: int = 7, maxLag: int = 14):
        self.modelFactory = modelFactory or (lambda: RidgeRegressor(alpha=1.0))
        self.period = period
        self.maxLag = maxLag
        self.models = {}
        self.metadata = None
        self.fitted = False
        self._y = None

    def fit(self, y: np.ndarray, maxHorizon: int = 30) -> 'DirectReduction':
        n = len(y)
        self._y = y.copy()

        X, target, self.metadata = autoFeatureEngineering(
            y, period=self.period, maxLag=min(self.maxLag, n // 3)
        )

        if len(target) == 0:
            self.fitted = True
            return self

        offset = self.metadata['offset']

        for h in range(1, maxHorizon + 1):
            if offset + h >= n:
                break

            hTarget = y[offset + h:]
            hX = X[:len(hTarget)]

            if len(hX) < 5:
                break

            model = self.modelFactory()
            model.fit(hX, hTarget)
            self.models[h] = model

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted.")

        predictions = np.zeros(steps)
        y = self._y

        for h in range(1, steps + 1):
            if h in self.models:
                features = self._buildFeaturesForStep(y, h)
                predictions[h - 1] = self.models[h].predict(features.reshape(1, -1))[0]
            elif self.models:
                lastModel = self.models[max(self.models.keys())]
                features = self._buildFeaturesForStep(y, h)
                predictions[h - 1] = lastModel.predict(features.reshape(1, -1))[0]
            else:
                predictions[h - 1] = y[-1]

        sigma = np.std(np.diff(y[-min(30, len(y)):]))
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def _buildFeaturesForStep(self, y: np.ndarray, h: int) -> np.ndarray:
        n = len(y)
        meta = self.metadata
        lagGen = meta['lagGen']
        rollGen = meta['rollGen']
        period = meta['period']
        useFourier = meta['useFourier']

        lagF = lagGen.transformLast(y)
        rollF = rollGen.transformLast(y)

        parts = [lagF, rollF]

        if useFourier and period > 1:
            nTerms = min(3, period // 2)
            fourierGen = FourierFeatures(period=period, nTerms=nTerms)
            fourierF = fourierGen.transformSingle(n + h - 1)
            parts.append(fourierF)

        return np.concatenate(parts)


class RecursiveReduction:
    """
    Recursive Multi-step Forecasting

    단일 모델을 반복 적용하여 다단계 예측:
    1. model: X_t → y_{t+1}
    2. y_{t+1}을 다음 입력에 추가
    3. 반복
    """

    def __init__(self, modelFactory: Optional[Callable] = None, period: int = 7, maxLag: int = 14):
        self.modelFactory = modelFactory or (lambda: RidgeRegressor(alpha=1.0))
        self.period = period
        self.maxLag = maxLag
        self.model = None
        self.metadata = None
        self.fitted = False
        self._y = None

    def fit(self, y: np.ndarray) -> 'RecursiveReduction':
        n = len(y)
        self._y = y.copy()

        X, target, self.metadata = autoFeatureEngineering(
            y, period=self.period, maxLag=min(self.maxLag, n // 3)
        )

        if len(target) < 5:
            self.fitted = True
            return self

        self.model = self.modelFactory()
        self.model.fit(X, target)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted.")

        if self.model is None:
            pred = np.full(steps, self._y[-1])
            return pred, pred - 1, pred + 1

        predictions = np.zeros(steps)
        extendedY = list(self._y)

        for h in range(steps):
            yArr = np.array(extendedY)
            features = self._buildFeatures(yArr, len(yArr) + h)
            pred = self.model.predict(features.reshape(1, -1))[0]
            predictions[h] = pred
            extendedY.append(pred)

        sigma = np.std(np.diff(self._y[-min(30, len(self._y)):]))
        hRange = np.arange(1, steps + 1)
        margin = 1.96 * sigma * np.sqrt(hRange)
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def _buildFeatures(self, y: np.ndarray, timeIdx: int) -> np.ndarray:
        meta = self.metadata
        lagGen = meta['lagGen']
        rollGen = meta['rollGen']
        period = meta['period']
        useFourier = meta['useFourier']

        lagF = lagGen.transformLast(y)
        rollF = rollGen.transformLast(y)

        parts = [lagF, rollF]

        if useFourier and period > 1:
            nTerms = min(3, period // 2)
            fourierGen = FourierFeatures(period=period, nTerms=nTerms)
            fourierF = fourierGen.transformSingle(timeIdx)
            parts.append(fourierF)

        return np.concatenate(parts)
