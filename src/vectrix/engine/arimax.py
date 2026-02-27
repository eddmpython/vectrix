"""
ARIMAX: ARIMA with Exogenous Variables

ARIMA 모델에 외생변수(exogenous regressors)를 추가.
y_t = β'X_t + ARIMA(p,d,q) noise

구현:
1. 외생 회귀로 y에서 X의 영향 제거
2. 잔차에 ARIMA 적용
3. 예측 시 X_future + ARIMA 예측 결합
"""

import numpy as np
from typing import Tuple, Optional

from .arima import ARIMAModel
from .turbo import TurboCore


class ARIMAXModel:
    """
    ARIMA with Exogenous Variables

    Usage:
        >>> model = ARIMAXModel(order=(1,1,1))
        >>> model.fit(y, X)
        >>> pred, lo, hi = model.predict(steps, X_future)
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.regCoef = None
        self.arimaModel = None
        self.fitted = False
        self._yMean = 0.0

    def fit(self, y: np.ndarray, X: np.ndarray) -> 'ARIMAXModel':
        """
        Parameters
        ----------
        y : np.ndarray
            시계열 [n]
        X : np.ndarray
            외생변수 행렬 [n, p]
        """
        n = len(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        try:
            self.regCoef = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            self.regCoef = np.zeros(X.shape[1])

        residual = y - X @ self.regCoef

        self.arimaModel = ARIMAModel(order=self.order)
        self.arimaModel.fit(residual)

        self.fitted = True
        return self

    def predict(
        self,
        steps: int,
        XFuture: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        steps : int
            예측 기간
        XFuture : np.ndarray, optional
            미래 외생변수 [steps, p]. None이면 0으로 가정.
        """
        if not self.fitted:
            raise ValueError("Model not fitted.")

        arimaPred, arimaLo, arimaHi = self.arimaModel.predict(steps)

        if XFuture is not None:
            if XFuture.ndim == 1:
                XFuture = XFuture.reshape(-1, 1)
            exogEffect = XFuture @ self.regCoef
        else:
            exogEffect = np.zeros(steps)

        predictions = exogEffect + arimaPred
        lower = exogEffect + arimaLo
        upper = exogEffect + arimaHi

        return predictions, lower, upper


class AutoARIMAX:
    """
    Auto ARIMAX: 자동 order 선택 + 외생변수

    AutoARIMA로 최적 order를 찾고 외생변수 회귀와 결합.
    """

    def __init__(self, maxP: int = 3, maxD: int = 2, maxQ: int = 3):
        self.maxP = maxP
        self.maxD = maxD
        self.maxQ = maxQ
        self.bestOrder = None
        self.model = None
        self.fitted = False

    def fit(self, y: np.ndarray, X: np.ndarray) -> 'AutoARIMAX':
        from .arima import AutoARIMA

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        regCoef = np.linalg.lstsq(X, y, rcond=None)[0]
        residual = y - X @ regCoef

        autoArima = AutoARIMA(maxP=self.maxP, maxD=self.maxD, maxQ=self.maxQ)
        autoArima.fit(residual)
        self.bestOrder = autoArima.bestOrder

        self.model = ARIMAXModel(order=self.bestOrder)
        self.model.fit(y, X)

        self.fitted = True
        return self

    def predict(
        self,
        steps: int,
        XFuture: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.predict(steps, XFuture)
