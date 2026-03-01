"""
Theta Model Implementation

M3 Competition winning model
Decomposes time series into Theta lines for forecasting

Reference: Assimakopoulos & Nikolopoulos (2000)
"""

from typing import Tuple

import numpy as np

try:
    from vectrix_core import ses_filter as _sesFilterRust
    from vectrix_core import ses_sse as _sesSSERust
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
def _sesSSEJIT(y: np.ndarray, alpha: float) -> float:
    n = len(y)
    level = y[0]
    sse = 0.0

    for t in range(1, n):
        error = y[t] - level
        sse += error * error
        level = alpha * y[t] + (1.0 - alpha) * level

    return sse


@jit(nopython=True, cache=True)
def _sesFilterJIT(y: np.ndarray, alpha: float) -> np.ndarray:
    n = len(y)
    result = np.zeros(n)
    result[0] = y[0]

    for t in range(1, n):
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1]

    return result


class ThetaModel:
    """
    Theta Model Implementation

    Theta method:
    1. Decompose time series into theta=0 (linear trend) and theta=2 (double curvature) lines
    2. theta=0 forecasted by linear regression, theta=2 by SES
    3. Combine the two forecasts
    """

    def __init__(self, theta: float = 2.0, period: int = 1):
        """
        Parameters
        ----------
        theta : float
            Theta parameter (default 2.0)
        period : int
            Seasonal period (for seasonal adjustment)
        """
        self.theta = theta
        self.period = period

        self.slope = 0.0
        self.intercept = 0.0
        self.alpha = 0.3
        self.lastLevel = 0.0

        self.seasonal = None
        self.deseasonalized = None

        self.fitted = False
        self.residuals = None

    def fit(self, y: np.ndarray) -> 'ThetaModel':
        """
        Fit the model

        Parameters
        ----------
        y : np.ndarray
            Time series data
        """
        n = len(y)

        if n < 5:
            self._simpleFit(y)
            self.fitted = True
            return self

        if self.period > 1 and n >= self.period * 2:
            self.deseasonalized, self.seasonal = self._deseasonalize(y)
            workData = self.deseasonalized
        else:
            workData = y
            self.seasonal = None

        x = np.arange(n, dtype=np.float64)
        self.slope, self.intercept = TurboCore.linearRegression(x, workData)

        thetaLine = self._computeThetaLine(workData, self.theta)

        self.alpha = self._optimizeAlpha(thetaLine)
        self.lastLevel = self._sesFilter(thetaLine, self.alpha)[-1]

        fitted = self._computeFitted(workData, n)
        self.residuals = workData - fitted

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast

        Parameters
        ----------
        steps : int
            Forecast horizon

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (predictions, lower95, upper95)
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        n = len(self.deseasonalized) if self.deseasonalized is not None else 0
        if n == 0:
            n = 10

        predictions = np.zeros(steps)

        for h in range(1, steps + 1):
            t = n + h - 1
            trendPred = self.intercept + self.slope * t

            sesPred = self.lastLevel

            pred = (trendPred + sesPred) / 2

            predictions[h - 1] = pred

        if self.seasonal is not None:
            for h in range(steps):
                seasonIdx = h % self.period
                predictions[h] += self.seasonal[seasonIdx]

        if self.residuals is not None:
            sigma = np.std(self.residuals)
        else:
            sigma = np.std(predictions) * 0.1

        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower95 = predictions - margin
        upper95 = predictions + margin

        return predictions, lower95, upper95

    def _deseasonalize(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Classical seasonal decomposition"""
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

    def _computeThetaLine(self, y: np.ndarray, theta: float) -> np.ndarray:
        """
        Compute Theta line

        Z_theta(t) = theta * y(t) + (1 - theta) * L(t)
        where L(t) is the linear trend
        """
        n = len(y)
        x = np.arange(n, dtype=np.float64)

        linearTrend = self.intercept + self.slope * x

        thetaLine = theta * y + (1 - theta) * linearTrend

        return thetaLine

    def _optimizeAlpha(self, y: np.ndarray) -> float:
        """Optimize SES alpha"""
        bestAlpha = 0.3
        bestSSE = np.inf
        yF64 = y.astype(np.float64, copy=False)

        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if RUST_AVAILABLE:
                sse = _sesSSERust(yF64, alpha)
            else:
                sse = _sesSSEJIT(y, alpha)
            if sse < bestSSE:
                bestSSE = sse
                bestAlpha = alpha

        return bestAlpha

    def _sesSSE(self, y: np.ndarray, alpha: float) -> float:
        """Compute SES SSE"""
        if RUST_AVAILABLE:
            return _sesSSERust(y.astype(np.float64, copy=False), alpha)
        return _sesSSEJIT(y, alpha)

    def _sesFilter(self, y: np.ndarray, alpha: float) -> np.ndarray:
        """SES filtering"""
        if RUST_AVAILABLE:
            return _sesFilterRust(y.astype(np.float64, copy=False), alpha)
        return _sesFilterJIT(y, alpha)

    def _computeFitted(self, y: np.ndarray, n: int) -> np.ndarray:
        """Compute fitted values"""
        x = np.arange(n, dtype=np.float64)

        trendFitted = self.intercept + self.slope * x

        thetaLine = self._computeThetaLine(y, self.theta)
        sesFitted = self._sesFilter(thetaLine, self.alpha)

        fitted = (trendFitted + sesFitted) / 2

        return fitted

    def _simpleFit(self, y: np.ndarray):
        """Simple fit"""
        self.slope = 0.0
        self.intercept = np.mean(y)
        self.lastLevel = y[-1] if len(y) > 0 else 0
        self.deseasonalized = y
        self.residuals = np.zeros(len(y))


class OptimizedTheta:
    """
    Optimized Theta Model (OTM)

    Tries multiple theta values to select the optimal one
    """

    def __init__(self, period: int = 1):
        self.period = period
        self.bestTheta = 2.0
        self.bestModel = None

    def fit(self, y: np.ndarray) -> ThetaModel:
        """Select optimal theta (3 candidates, refit best on full data)"""
        n = len(y)
        trainSize = int(n * 0.8)

        if trainSize < 10:
            self.bestModel = ThetaModel(theta=2.0, period=self.period)
            self.bestModel.fit(y)
            return self.bestModel

        trainData = y[:trainSize]
        testData = y[trainSize:]
        testSteps = len(testData)

        bestMAPE = np.inf

        for theta in [1.0, 2.0, 3.0]:
            try:
                model = ThetaModel(theta=theta, period=self.period)
                model.fit(trainData)
                pred, _, _ = model.predict(testSteps)

                mape = TurboCore.mape(testData, pred[:len(testData)])

                if mape < bestMAPE:
                    bestMAPE = mape
                    self.bestTheta = theta

            except Exception:
                continue

        self.bestModel = ThetaModel(theta=self.bestTheta, period=self.period)
        self.bestModel.fit(y)

        return self.bestModel

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast"""
        if self.bestModel is None:
            raise ValueError("Model has not been fitted.")
        return self.bestModel.predict(steps)
