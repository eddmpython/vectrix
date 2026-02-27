"""
Global Forecaster

여러 시계열에서 하나의 모델을 학습하여 공통 패턴 추출.
Cross-learning: 시계열 간 지식 공유.

Strategy:
1. 모든 시계열에서 피처 추출
2. 하나의 회귀 모델 학습
3. 각 시계열에 대해 개별 예측
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np

from ..regression.features import FourierFeatures, autoFeatureEngineering
from ..regression.linear import RidgeRegressor


class GlobalForecaster:
    """
    Global Forecaster

    Usage:
        >>> gf = GlobalForecaster(period=7)
        >>> gf.fit(series_dict)  # {'series1': y1, 'series2': y2, ...}
        >>> predictions = gf.predict(steps=30)  # {'series1': (pred, lo, hi), ...}
    """

    def __init__(
        self,
        period: int = 7,
        maxLag: int = 14,
        modelFactory: Optional[Callable] = None
    ):
        self.period = period
        self.maxLag = maxLag
        self.modelFactory = modelFactory or (lambda: RidgeRegressor(alpha=1.0))
        self.model = None
        self.seriesData = {}
        self.seriesScales = {}
        self.metadata = None
        self.fitted = False

    def fit(self, series: Dict[str, np.ndarray]) -> 'GlobalForecaster':
        """
        여러 시계열을 동시에 학습

        Parameters
        ----------
        series : Dict[str, np.ndarray]
            시계열 딕셔너리 {이름: 데이터}
        """
        self.seriesData = {k: v.copy() for k, v in series.items()}

        allX = []
        allY = []

        for name, y in series.items():
            n = len(y)
            mean = np.mean(y)
            std = np.std(y)
            if std < 1e-10:
                std = 1.0
            self.seriesScales[name] = (mean, std)

            normalized = (y - mean) / std

            X, target, meta = autoFeatureEngineering(
                normalized, period=self.period,
                maxLag=min(self.maxLag, n // 3)
            )

            if len(target) > 0:
                allX.append(X)
                allY.append(target)
                if self.metadata is None:
                    self.metadata = meta

        if not allX:
            self.fitted = True
            return self

        XCombined = np.vstack(allX)
        yCombined = np.concatenate(allY)

        self.model = self.modelFactory()
        self.model.fit(XCombined, yCombined)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        모든 시계열에 대해 예측

        Returns
        -------
        Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
            {이름: (예측, 하한, 상한)}
        """
        if not self.fitted:
            raise ValueError("Model not fitted.")

        results = {}

        for name, y in self.seriesData.items():
            mean, std = self.seriesScales.get(name, (0, 1))
            normalized = (y - mean) / std

            predictions = np.zeros(steps)
            extendedY = list(normalized)

            if self.model is not None and self.metadata is not None:
                for h in range(steps):
                    yArr = np.array(extendedY)
                    features = self._buildFeatures(yArr, len(yArr) + h)
                    pred = self.model.predict(features.reshape(1, -1))[0]
                    predictions[h] = pred
                    extendedY.append(pred)

            predictions = predictions * std + mean

            sigma = np.std(np.diff(y[-min(30, len(y)):]))
            margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
            lower = predictions - margin
            upper = predictions + margin

            results[name] = (predictions, lower, upper)

        return results

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
