"""
Dynamic Optimized Theta (DOT)

Based on Fiorucci et al. (2016).
Extends the Theta model by simultaneously optimizing
theta, alpha, and drift via L-BFGS-B.

DOT-Hybrid mode (E018): For period<=12, applies 8-way auto-select
(2 trend types x 2 model types x 2 season types) for improved
accuracy on low-frequency data. For period>=24, uses original
3-parameter optimization which excels on high-frequency data.

M4 Competition benchmark: OWA 0.885 (DOT-Hybrid) vs 0.905 (original).
"""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

try:
    from vectrix_core import dot_objective as _rustDotObjective
    from vectrix_core import dot_residuals as _rustDotResiduals
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    from vectrix_core import ses_filter as _rustSesFilter
    from vectrix_core import ses_sse as _rustSesSSE
    SES_RUST_AVAILABLE = True
except ImportError:
    SES_RUST_AVAILABLE = False

try:
    from vectrix_core import dot_hybrid_objective as _rustDotHybridObjective
    HYBRID_RUST_AVAILABLE = True
except ImportError:
    HYBRID_RUST_AVAILABLE = False

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


def _sesFilterPython(y, alpha):
    n = len(y)
    result = np.zeros(n)
    result[0] = y[0]
    for t in range(1, n):
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1]
    return result


def _sesSSEPython(y, alpha):
    n = len(y)
    level = y[0]
    sse = 0.0
    for t in range(1, n):
        error = y[t] - level
        sse += error * error
        level = alpha * y[t] + (1.0 - alpha) * level
    return sse


def _sesFilter(y, alpha):
    if SES_RUST_AVAILABLE:
        return np.asarray(_rustSesFilter(y, alpha))
    return _sesFilterPython(y, alpha)


def _sesSSE(y, alpha):
    if SES_RUST_AVAILABLE:
        return _rustSesSSE(y, alpha)
    return _sesSSEPython(y, alpha)


def _optimizeAlpha(y):
    if len(y) < 3:
        return 0.3
    result = minimize_scalar(lambda a: _sesSSE(y, a), bounds=(0.001, 0.999), method='bounded')
    return result.x if result.success else 0.3


def _linearRegressionSimple(x, y):
    xMean = np.mean(x)
    yMean = np.mean(y)
    num = np.sum((x - xMean) * (y - yMean))
    den = np.sum((x - xMean) ** 2)
    slope = num / max(den, 1e-10)
    intercept = yMean - slope * xMean
    return slope, intercept


_HYBRID_THRESHOLD = 24


class DynamicOptimizedTheta:
    """
    Dynamic Optimized Theta Model

    Simultaneously optimizes theta, alpha, and drift for
    more accurate forecasts compared to Theta/OptimizedTheta.

    DOT-Hybrid: For period < 24, uses 8-way auto-select
    (exponential/linear trend x additive/multiplicative model x A/M season)
    to capture nonlinear growth patterns. For period >= 24, uses
    the original 3-parameter optimization which excels on high-frequency data.
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

        self._hybridMode = False
        self._hybridConfig = None
        self._hybridModel = None

    def fit(self, y: np.ndarray) -> 'DynamicOptimizedTheta':
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        self._n = n
        self._y = y.copy()

        if n < 5:
            self.intercept = np.mean(y)
            self.lastLevel = y[-1] if n > 0 else 0
            self.residuals = np.zeros(n)
            self.fitted = True
            return self

        if self.period < _HYBRID_THRESHOLD:
            self._fitHybrid(y)
        else:
            self._fitClassic(y)

        return self

    def _fitClassic(self, y: np.ndarray) -> 'DynamicOptimizedTheta':
        n = len(y)

        if self.period > 1 and n >= self.period * 2:
            workData, self.seasonal = self._deseasonalizeClassic(y)
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
        self._hybridMode = False
        self.fitted = True
        return self

    def _fitHybrid(self, y: np.ndarray) -> 'DynamicOptimizedTheta':
        n = len(y)

        hasSeason = self.period > 1 and n >= self.period * 3
        seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']

        scaled = y.copy()
        base = np.mean(np.abs(scaled))
        if base > 0:
            scaled = scaled / base
        else:
            base = 1.0

        bestMae = np.inf
        bestConfig = None
        bestModel = None

        for seasonType in seasonTypes:
            if seasonType != 'none':
                seasonal, deseasonalized = self._deseasonalizeAdvanced(scaled, self.period, seasonType)
            else:
                seasonal = None
                deseasonalized = scaled

            for trendType in ['linear', 'exponential']:
                thetaLine0 = self._fitTrendLine(deseasonalized, trendType)
                if thetaLine0 is None:
                    continue

                for modelType in ['additive', 'multiplicative']:
                    if modelType == 'multiplicative' and np.any(thetaLine0 <= 0):
                        continue
                    if modelType == 'multiplicative' and np.any(deseasonalized <= 0):
                        continue

                    result = self._fitVariant(deseasonalized, thetaLine0, trendType, modelType)
                    if result is None:
                        continue

                    fittedVals = result['fittedValues']
                    if seasonal is not None:
                        fittedVals = self._reseasonalize(fittedVals, seasonal, seasonType)

                    mae = np.mean(np.abs(scaled - fittedVals))
                    if mae < bestMae:
                        bestMae = mae
                        bestConfig = (trendType, modelType, seasonType)
                        bestModel = result
                        bestModel['seasonal'] = seasonal
                        bestModel['base'] = base

        if bestModel is None:
            return self._fitClassic(y)

        self._hybridMode = True
        self._hybridConfig = bestConfig
        self._hybridModel = bestModel
        self.theta = bestModel['theta']
        self.intercept = bestModel['intercept']
        self.slope = bestModel['slope']
        self.lastLevel = bestModel['lastLevel']
        self.residuals = y - bestModel['fittedValues'] * base
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        if self._hybridMode:
            return self._predictHybrid(steps)
        else:
            return self._predictClassic(steps)

    def _predictClassic(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _predictHybrid(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self._hybridModel
        trendType, modelType, seasonType = self._hybridConfig
        n = self._n
        base = m['base']

        forecastTrend = self._extrapolateTrend(m, trendType, steps)
        forecastSES = np.full(steps, m['lastLevel'])

        if modelType == 'additive':
            wses = 1.0 / max(m['theta'], 1.0)
            wtrend = 1.0 - wses
            combined = wses * forecastSES + wtrend * forecastTrend
        else:
            invTheta = 1.0 / max(m['theta'], 1.0)
            combined = np.power(np.maximum(forecastSES, 1e-10), invTheta) * \
                       np.power(np.maximum(forecastTrend, 1e-10), 1.0 - invTheta)

        if m['seasonal'] is not None:
            for h in range(steps):
                idx = (n + h) % self.period
                if seasonType == 'multiplicative':
                    combined[h] *= m['seasonal'][idx]
                else:
                    combined[h] += m['seasonal'][idx]

        predictions = combined * base

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 1 else 1.0
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def _fitVariant(self, y, thetaLine0, trendType, modelType):
        n = len(y)
        if n < 5:
            return None

        isAdd = modelType == 'additive'
        ySafe = np.maximum(y, 1e-10)
        t0Safe = np.maximum(thetaLine0, 1e-10)

        def buildThetaLine(theta):
            if isAdd:
                return theta * y + (1.0 - theta) * thetaLine0
            return np.power(ySafe, theta) * np.power(t0Safe, 1.0 - theta)

        def combineFitted(filtered, theta):
            if isAdd:
                w = 1.0 / max(theta, 1.0)
                return w * filtered + (1.0 - w) * thetaLine0
            inv = 1.0 / max(theta, 1.0)
            return np.power(np.maximum(filtered, 1e-10), inv) * \
                   np.power(t0Safe, 1.0 - inv)

        if HYBRID_RUST_AVAILABLE:
            def objective(params):
                return _rustDotHybridObjective(y, thetaLine0, params[0], isAdd)
        else:
            def objective(params):
                theta = params[0]
                thetaLine = buildThetaLine(theta)
                alpha = _optimizeAlpha(thetaLine)
                filtered = _sesFilter(thetaLine, alpha)
                fittedVals = combineFitted(filtered, theta)
                return np.mean(np.abs(y - fittedVals))

        result = minimize(objective, x0=[2.0], bounds=[(1.0, 50.0)],
                          method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
        theta = result.x[0]

        thetaLine = buildThetaLine(theta)
        alpha = _optimizeAlpha(thetaLine)
        filtered = _sesFilter(thetaLine, alpha)
        lastLevel = filtered[-1]
        fittedVals = combineFitted(filtered, theta)

        x = np.arange(n, dtype=np.float64)
        if trendType == 'exponential':
            logY = np.log(ySafe)
            slope, intercept = _linearRegressionSimple(x, logY)
        else:
            slope, intercept = _linearRegressionSimple(x, y)

        return {
            'theta': theta,
            'alpha': alpha,
            'intercept': intercept,
            'slope': slope,
            'lastLevel': lastLevel,
            'n': n,
            'residStd': max(np.std(y - fittedVals), 1e-8),
            'fittedValues': fittedVals,
        }

    def _fitTrendLine(self, y, trendType):
        n = len(y)
        x = np.arange(n, dtype=np.float64)
        if trendType == 'exponential':
            if np.any(y <= 0):
                return None
            logY = np.log(y)
            slope, intercept = _linearRegressionSimple(x, logY)
            return np.exp(intercept + slope * x)
        else:
            slope, intercept = _linearRegressionSimple(x, y)
            return intercept + slope * x

    def _extrapolateTrend(self, m, trendType, steps):
        n = m['n']
        futureX = np.arange(n, n + steps, dtype=np.float64)
        if trendType == 'exponential':
            return np.exp(m['intercept'] + m['slope'] * futureX)
        else:
            return m['intercept'] + m['slope'] * futureX

    def _deseasonalizeClassic(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    def _deseasonalizeAdvanced(self, y, period, seasonType):
        n = len(y)
        seasonal = np.zeros(period)
        counts = np.zeros(period)
        trend = np.convolve(y, np.ones(period) / period, mode='valid')
        offset = (period - 1) // 2

        if seasonType == 'multiplicative':
            for i in range(len(trend)):
                idx = i + offset
                if idx < n and trend[i] > 0:
                    seasonal[idx % period] += y[idx] / trend[i]
                    counts[idx % period] += 1
            for i in range(period):
                seasonal[i] = seasonal[i] / max(counts[i], 1)
            meanS = np.mean(seasonal)
            if meanS > 0:
                seasonal /= meanS
            seasonal = np.maximum(seasonal, 0.01)
            deseasonalized = y / seasonal[np.arange(n) % period]
        else:
            for i in range(len(trend)):
                idx = i + offset
                if idx < n:
                    seasonal[idx % period] += y[idx] - trend[i]
                    counts[idx % period] += 1
            for i in range(period):
                seasonal[i] = seasonal[i] / max(counts[i], 1)
            seasonal -= np.mean(seasonal)
            deseasonalized = y - seasonal[np.arange(n) % period]

        return seasonal, deseasonalized

    def _reseasonalize(self, y, seasonal, seasonType):
        n = len(y)
        result = y.copy()
        period = len(seasonal)
        for t in range(n):
            idx = t % period
            if seasonType == 'multiplicative':
                result[t] *= seasonal[idx]
            else:
                result[t] += seasonal[idx]
        return result
