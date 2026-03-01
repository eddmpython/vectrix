"""
Dynamic Optimized Theta (DOT)

Based on Fiorucci et al. (2016).
Extends the Theta model by simultaneously optimizing
theta, alpha, and drift via L-BFGS-B.

Advantages over ThetaModel:
- Continuous optimization of theta (not grid search)
- Drift parameter enables trend adjustment
- Broader search space
"""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from vectrix_core import dot_objective as _rustDotObjective
    from vectrix_core import dot_residuals as _rustDotResiduals
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .turbo import TurboCore


@jit(nopython=True, cache=True)
def _dotObjectivePython(
    workData: np.ndarray,
    intercept: float,
    slope: float,
    theta: float,
    alpha: float,
    drift: float
) -> float:
    n = len(workData)
    x = np.arange(n, dtype=np.float64)
    linearTrend = np.empty(n)
    for i in range(n):
        linearTrend[i] = intercept + slope * x[i]

    thetaLine = np.empty(n)
    for i in range(n):
        thetaLine[i] = theta * workData[i] + (1.0 - theta) * linearTrend[i]

    level = thetaLine[0]
    sse = 0.0
    for t in range(1, n):
        trendPred = intercept + slope * t
        pred = (trendPred + level + drift * t) / 2.0
        error = workData[t] - pred
        sse += error * error
        level = alpha * thetaLine[t] + (1.0 - alpha) * level

    return sse


@jit(nopython=True, cache=True)
def _dotComputeResidualsPython(
    workData: np.ndarray,
    intercept: float,
    slope: float,
    theta: float,
    alpha: float,
    drift: float
) -> Tuple[np.ndarray, float]:
    n = len(workData)
    x = np.arange(n, dtype=np.float64)
    linearTrend = np.empty(n)
    for i in range(n):
        linearTrend[i] = intercept + slope * x[i]

    thetaLine = np.empty(n)
    for i in range(n):
        thetaLine[i] = theta * workData[i] + (1.0 - theta) * linearTrend[i]

    level = thetaLine[0]
    residuals = np.empty(n - 1)
    for t in range(1, n):
        trendPred = intercept + slope * t
        pred = (trendPred + level + drift * t) / 2.0
        residuals[t - 1] = workData[t] - pred
        level = alpha * thetaLine[t] + (1.0 - alpha) * level

    return residuals, level


def _dotObjectiveJIT(workData, intercept, slope, theta, alpha, drift):
    if RUST_AVAILABLE:
        return _rustDotObjective(workData, intercept, slope, theta, alpha, drift)
    return _dotObjectivePython(workData, intercept, slope, theta, alpha, drift)


def _dotComputeResiduals(workData, intercept, slope, theta, alpha, drift):
    if RUST_AVAILABLE:
        return _rustDotResiduals(workData, intercept, slope, theta, alpha, drift)
    return _dotComputeResidualsPython(workData, intercept, slope, theta, alpha, drift)


class DynamicOptimizedTheta:
    """
    Dynamic Optimized Theta Model

    Simultaneously optimizes theta, alpha, and drift for
    more accurate forecasts compared to Theta/OptimizedTheta.
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

        intercept = self.intercept
        slope = self.slope

        def objective(params):
            return _dotObjectiveJIT(workData, intercept, slope, params[0], params[1], params[2])

        x0 = [2.0, 0.3, 0.0]
        bounds = [(0.5, 5.0), (0.01, 0.99), (-1.0, 1.0)]
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 30, 'ftol': 1e-4})

        self.theta = result.x[0]
        self.alpha = result.x[1]
        self.drift = result.x[2]

        self.residuals, self.lastLevel = _dotComputeResiduals(
            workData, self.intercept, self.slope, self.theta, self.alpha, self.drift
        )
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
