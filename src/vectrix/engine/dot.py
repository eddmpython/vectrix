"""
Dynamic Optimized Theta (DOT)

Fiorucci et al. (2016) 기반.
기존 Theta 모델을 확장하여 theta, alpha, drift를
동시에 L-BFGS-B로 최적화.

기존 ThetaModel 대비 장점:
- theta 값을 연속 최적화 (그리드 서치 아님)
- drift 파라미터 추가로 추세 조정 가능
- 더 넓은 탐색 공간
"""

import numpy as np
from typing import Tuple
from scipy.optimize import minimize

from .turbo import TurboCore


class DynamicOptimizedTheta:
    """
    Dynamic Optimized Theta Model

    theta, alpha, drift를 동시에 최적화하여
    기존 Theta/OptimizedTheta 대비 더 정확한 예측.
    """

    def __init__(self, period: int = 1):
        self.period = period

        self.theta = 2.0
        self.alpha = 0.3
        self.drift = 0.0
        self.slope = 0.0
        self.intercept = 0.0
        self.lastLevel = 0.0
        self.seasonal = None
        self.residuals = None
        self.fitted = False
        self._n = 0

    def fit(self, y: np.ndarray) -> 'DynamicOptimizedTheta':
        n = len(y)
        self._n = n

        if n < 5:
            self.intercept = np.mean(y)
            self.lastLevel = y[-1] if n > 0 else 0
            self.residuals = np.zeros(n)
            self.fitted = True
            return self

        if self.period > 1 and n >= self.period * 2:
            workData, self.seasonal = self._deseasonalize(y)
        else:
            workData = y
            self.seasonal = None

        x = np.arange(n, dtype=np.float64)
        self.slope, self.intercept = TurboCore.linearRegression(x, workData)

        def objective(params):
            theta, alpha, drift = params[0], params[1], params[2]
            linearTrend = self.intercept + self.slope * x
            thetaLine = theta * workData + (1 - theta) * linearTrend

            level = thetaLine[0]
            sse = 0.0
            for t in range(1, n):
                trendPred = self.intercept + self.slope * t
                pred = (trendPred + level + drift * t) / 2
                error = workData[t] - pred
                sse += error ** 2
                level = alpha * thetaLine[t] + (1 - alpha) * level

            return sse

        x0 = [2.0, 0.3, 0.0]
        bounds = [(0.5, 5.0), (0.01, 0.99), (-1.0, 1.0)]
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 30, 'ftol': 1e-4})

        self.theta = result.x[0]
        self.alpha = result.x[1]
        self.drift = result.x[2]

        linearTrend = self.intercept + self.slope * x
        thetaLine = self.theta * workData + (1 - self.theta) * linearTrend

        self.lastLevel = thetaLine[0]
        residuals = []
        for t in range(1, n):
            trendPred = self.intercept + self.slope * t
            pred = (trendPred + self.lastLevel + self.drift * t) / 2
            residuals.append(workData[t] - pred)
            self.lastLevel = self.alpha * thetaLine[t] + (1 - self.alpha) * self.lastLevel

        self.residuals = np.array(residuals)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        n = self._n
        predictions = np.zeros(steps)

        for h in range(1, steps + 1):
            t = n + h - 1
            trendPred = self.intercept + self.slope * t
            sesPred = self.lastLevel + self.drift * (n + h)
            predictions[h - 1] = (trendPred + sesPred) / 2

        if self.seasonal is not None:
            for h in range(steps):
                sidx = (n + h) % self.period
                predictions[h] += self.seasonal[sidx]

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 1 else 1.0
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def _deseasonalize(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(y)
        m = self.period

        seasonal = np.zeros(m)
        for i in range(m):
            vals = y[i::m]
            seasonal[i] = np.mean(vals) - np.mean(y)

        deseasonalized = np.zeros(n)
        for t in range(n):
            deseasonalized[t] = y[t] - seasonal[t % m]

        return deseasonalized, seasonal
