"""
ETS (Error-Trend-Seasonal) Model Implementation

Exponential Smoothing family:
- Simple Exponential Smoothing (N,N)
- Holt's Linear (A,N), (M,N)
- Holt-Winters (A,A), (A,M), (M,A), (M,M)
- Damped variants

Hyndman-Khandakar style automatic model selection:
- Stepwise search (4 seed models -> neighbor search)
- Correct multiplicative error log-likelihood
- Numerical stability guaranteed

Pure implementation optimized with Numba
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from vectrix_core import ets_filter as _etsFilterRust
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


@jit(nopython=True, cache=True)
def _etsFilterJIT(
    y: np.ndarray,
    level0: float,
    trend0: float,
    seasonal0: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    phi: float,
    period: int,
    errorType: int,
    trendType: int,
    seasonalType: int
) -> tuple:
    """Numba JIT optimized ETS state-space filter"""
    n = len(y)
    fitted = np.zeros(n)
    residuals = np.zeros(n)
    seasonal = seasonal0.copy()

    level = level0
    trend = trend0

    for t in range(n):
        seasonIdx = t % period

        if seasonalType == 1:
            s = seasonal[seasonIdx]
        elif seasonalType == 2:
            s = seasonal[seasonIdx]
            if s < 1e-6:
                s = 1e-6
                seasonal[seasonIdx] = s
        else:
            s = 0.0

        if trendType == 0:
            if seasonalType == 1:
                yhat = level + s
            elif seasonalType == 2:
                yhat = level * s
            else:
                yhat = level
        elif trendType == 1 or trendType == 2:
            if seasonalType == 1:
                yhat = level + phi * trend + s
            elif seasonalType == 2:
                yhat = (level + phi * trend) * s
            else:
                yhat = level + phi * trend
        else:
            if seasonalType == 1:
                yhat = level * trend + s
            elif seasonalType == 2:
                yhat = level * trend * s
            else:
                yhat = level * trend

        fitted[t] = yhat
        error = y[t] - yhat
        residuals[t] = error

        levelOld = level
        trendOld = trend

        if errorType == 0:
            if seasonalType == 1:
                level = levelOld + phi * trendOld + alpha * error
            elif seasonalType == 2:
                level = (levelOld + phi * trendOld) + alpha * error / (s + 1e-10)
            else:
                level = levelOld + phi * trendOld + alpha * error
        else:
            eRatio = error / (yhat + 1e-10)
            if seasonalType == 2:
                level = (levelOld + phi * trendOld) * (1.0 + alpha * eRatio)
            elif seasonalType == 1:
                level = (levelOld + phi * trendOld) * (1.0 + alpha * eRatio)
            else:
                level = (levelOld + phi * trendOld) * (1.0 + alpha * eRatio)

        if level > 1e15:
            level = 1e15
        elif level < -1e15:
            level = -1e15

        if trendType == 1 or trendType == 2:
            trend = phi * trendOld + beta * (level - levelOld)
        elif trendType == 3:
            trend = trendOld * (level / (levelOld + 1e-10)) ** beta

        if seasonalType == 1:
            seasonal[seasonIdx] = s + gamma * error
        elif seasonalType == 2:
            if errorType == 0:
                seasonal[seasonIdx] = s + gamma * error / (levelOld + phi * trendOld + 1e-10)
            else:
                seasonal[seasonIdx] = s * (1.0 + gamma * error / (yhat + 1e-10))
            if seasonal[seasonIdx] < 1e-6:
                seasonal[seasonIdx] = 1e-6

    return fitted, residuals, level, trend, seasonal


class ETSModel:
    """
    ETS Model Implementation

    Model notation: ETS(Error, Trend, Seasonal)
    - Error: A(Additive), M(Multiplicative)
    - Trend: N(None), A(Additive), Ad(Additive Damped), M(Multiplicative), Md(Multiplicative Damped)
    - Seasonal: N(None), A(Additive), M(Multiplicative)
    """

    def __init__(
        self,
        errorType: str = 'A',
        trendType: str = 'A',
        seasonalType: str = 'A',
        period: int = 7,
        damped: bool = False
    ):
        """
        Parameters
        ----------
        errorType : str
            'A' (Additive) or 'M' (Multiplicative)
        trendType : str
            'N' (None), 'A' (Additive), 'M' (Multiplicative)
        seasonalType : str
            'N' (None), 'A' (Additive), 'M' (Multiplicative)
        period : int
            Seasonal period
        damped : bool
            Whether to use damped trend
        """
        self.errorType = errorType
        self.trendType = trendType
        self.seasonalType = seasonalType
        self.period = period
        self.damped = damped

        self.alpha = 0.3
        self.beta = 0.1
        self.gamma = 0.1
        self.phi = 0.98

        self.level = None
        self.trend = None
        self.seasonal = None

        self.fitted = False
        self.residuals = None

    def fit(self, y: np.ndarray, optimize: bool = True) -> 'ETSModel':
        """
        Fit the model

        Parameters
        ----------
        y : np.ndarray
            Time series data
        optimize : bool
            Whether to optimize parameters
        """
        n = len(y)
        m = self.period

        if n < max(m * 2, 10):
            self._simpleFit(y)
            self.fitted = True
            return self

        self._initializeState(y)

        if optimize:
            self._optimizeParameters(y)

        self._fitWithParams(y)
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

        predictions = np.zeros(steps)
        m = self.period

        level = self.level
        trend = self.trend if self.trendType != 'N' else 0
        seasonal = self.seasonal.copy() if self.seasonalType != 'N' else None

        phi = self.phi if self.damped else 1.0

        for h in range(1, steps + 1):
            if self.damped and self.trendType != 'N':
                trendCumsum = trend * (1 - phi ** h) / (1 - phi) if phi != 1 else trend * h
            else:
                trendCumsum = trend * h

            if self.seasonalType != 'N':
                seasonIdx = (h - 1) % m
                seasonVal = seasonal[seasonIdx]
            else:
                seasonVal = 0 if self.seasonalType == 'A' else 1

            if self.trendType == 'N':
                if self.seasonalType == 'A':
                    predictions[h - 1] = level + seasonVal
                elif self.seasonalType == 'M':
                    predictions[h - 1] = level * seasonVal
                else:
                    predictions[h - 1] = level
            elif self.trendType in ['A', 'Ad']:
                if self.seasonalType == 'A':
                    predictions[h - 1] = level + trendCumsum + seasonVal
                elif self.seasonalType == 'M':
                    predictions[h - 1] = (level + trendCumsum) * seasonVal
                else:
                    predictions[h - 1] = level + trendCumsum
            else:
                if self.seasonalType == 'A':
                    predictions[h - 1] = level * (trend ** h) + seasonVal
                elif self.seasonalType == 'M':
                    predictions[h - 1] = level * (trend ** h) * seasonVal
                else:
                    predictions[h - 1] = level * (trend ** h)

        if self.residuals is not None and len(self.residuals) > 0:
            sigma = np.std(self.residuals)
        else:
            sigma = np.std(predictions) * 0.1

        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower95 = predictions - margin
        upper95 = predictions + margin

        return predictions, lower95, upper95

    def _initializeState(self, y: np.ndarray):
        """Initialize state values"""
        n = len(y)
        m = self.period

        if self.seasonalType != 'N' and n >= m:
            self.level = np.mean(y[:m])
        else:
            self.level = y[0]

        if self.trendType != 'N':
            if n >= m * 2:
                self.trend = (np.mean(y[m:m*2]) - np.mean(y[:m])) / m
            elif n >= 2:
                self.trend = (y[-1] - y[0]) / (n - 1)
            else:
                self.trend = 0.0

            if self.trendType == 'M':
                self.trend = 1.0 + self.trend / (self.level + 1e-10)
        else:
            self.trend = 0.0

        if self.seasonalType != 'N' and n >= m:
            self.seasonal = np.zeros(m)
            for i in range(m):
                indices = range(i, min(n, m * 3), m)
                vals = [y[j] for j in indices if j < n]
                if vals:
                    if self.seasonalType == 'A':
                        self.seasonal[i] = np.mean(vals) - self.level
                    else:
                        self.seasonal[i] = np.mean(vals) / (self.level + 1e-10)
        else:
            if self.seasonalType == 'A':
                self.seasonal = np.zeros(m)
            else:
                self.seasonal = np.ones(m)

    def _optimizeParameters(self, y: np.ndarray):
        """Optimize parameters with scipy"""
        bounds = [(0.001, 0.999)]

        if self.trendType != 'N':
            bounds.append((0.001, 0.999))

        if self.seasonalType != 'N':
            bounds.append((0.001, 0.999))

        if self.damped:
            bounds.append((0.8, 0.999))

        def objective(params):
            return self._computeSSE(y, params)

        x0 = [0.3]
        if self.trendType != 'N':
            x0.append(0.1)
        if self.seasonalType != 'N':
            x0.append(0.1)
        if self.damped:
            x0.append(0.98)

        try:
            result = minimize(
                objective, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 30, 'ftol': 1e-4}
            )
            self._setParams(result.x)
        except Exception:
            pass

    def _computeSSE(self, y: np.ndarray, params: np.ndarray) -> float:
        """
        Compute Sum of Squared Errors

        Scales residuals by data standard deviation for numerical stability
        """
        self._setParams(params)

        try:
            fitted, residuals = self._filter(y)
            yStd = np.std(y)
            if yStd < 1e-10:
                yStd = 1.0
            scaledResiduals = residuals / yStd
            return np.sum(scaledResiduals ** 2) * (yStd ** 2)
        except Exception:
            return np.inf

    def _setParams(self, params: np.ndarray):
        """Set parameters"""
        idx = 0
        self.alpha = params[idx]
        idx += 1

        if self.trendType != 'N':
            self.beta = params[idx]
            idx += 1

        if self.seasonalType != 'N':
            self.gamma = params[idx]
            idx += 1

        if self.damped:
            self.phi = params[idx]

    def _fitWithParams(self, y: np.ndarray):
        """Fit with current parameters"""
        fitted, residuals = self._filter(y)
        self.residuals = residuals

    def _filter(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """State-space filtering (Rust > Numba JIT > Pure Python)"""
        errorInt = 0 if self.errorType == 'A' else 1
        trendMap = {'N': 0, 'A': 1, 'Ad': 2, 'M': 3}
        trendKey = self.trendType + ('d' if self.damped else '')
        trendInt = trendMap.get(trendKey, 0)
        seasonalInt = {'N': 0, 'A': 1, 'M': 2}.get(self.seasonalType, 0)

        trend0 = self.trend if self.trendType != 'N' else 0.0
        seasonal0 = self.seasonal.copy() if self.seasonalType != 'N' else np.zeros(self.period)
        phi = self.phi if self.damped else 1.0

        if RUST_AVAILABLE:
            fitted, residuals, level, trend, seasonal = _etsFilterRust(
                y.astype(np.float64, copy=False), float(self.level), float(trend0),
                seasonal0.astype(np.float64, copy=False),
                float(self.alpha), float(self.beta), float(self.gamma), float(phi),
                int(self.period), int(errorInt), int(trendInt), int(seasonalInt)
            )
        else:
            fitted, residuals, level, trend, seasonal = _etsFilterJIT(
                y, self.level, trend0, seasonal0,
                self.alpha, self.beta, self.gamma, phi,
                self.period, errorInt, trendInt, seasonalInt
            )

        self.level = level
        self.trend = trend
        if self.seasonalType != 'N':
            self.seasonal = seasonal

        return fitted, residuals

    def _simpleFit(self, y: np.ndarray):
        """Simple fit (when data is insufficient)"""
        self.level = np.mean(y)
        self.trend = 0.0 if self.trendType == 'N' else (y[-1] - y[0]) / max(len(y) - 1, 1)
        self.seasonal = np.zeros(self.period)
        self.residuals = y - self.level


class AutoETS:
    """
    Automatic ETS Model Selection

    Hyndman-Khandakar style stepwise search:
    - Start from 4 seed models
    - Search neighbors of the best seed (change one component at a time)
    - Repeat until no improvement
    - Select optimal model based on AICc

    Error: {A, M}
    Trend: {N, A, Ad}  (M/Md excluded due to stability issues)
    Seasonal: {N, A, M}
    Valid combinations: up to 18 (some excluded based on data conditions)
    """

    SEED_MODELS = [
        ('A', 'N', 'N'),
        ('A', 'Ad', 'N'),
        ('A', 'Ad', 'A'),
        ('A', 'N', 'A'),
    ]

    FULL_MODELS = [
        ('A', 'N', 'N'),
        ('A', 'A', 'N'),
        ('A', 'Ad', 'N'),
        ('A', 'N', 'A'),
        ('A', 'A', 'A'),
        ('A', 'Ad', 'A'),
        ('A', 'N', 'M'),
        ('A', 'A', 'M'),
        ('A', 'Ad', 'M'),
        ('M', 'N', 'N'),
        ('M', 'A', 'N'),
        ('M', 'Ad', 'N'),
        ('M', 'N', 'A'),
        ('M', 'A', 'A'),
        ('M', 'Ad', 'A'),
        ('M', 'N', 'M'),
        ('M', 'A', 'M'),
        ('M', 'Ad', 'M'),
    ]

    def __init__(self, period: int = 7, exhaustive: bool = False):
        """
        Parameters
        ----------
        period : int
            Seasonal period
        exhaustive : bool
            If True, search all 18 models; if False, use stepwise search
        """
        self.period = period
        self.exhaustive = exhaustive
        self.bestModel = None
        self.bestAIC = np.inf
        self.allResults = {}

    def _isValidModel(
        self,
        error: str,
        trend: str,
        seasonal: str,
        y: np.ndarray
    ) -> bool:
        """
        Model validity check

        Multiplicative error/seasonal allowed only when all y > 0
        Seasonal models require sufficient data
        """
        n = len(y)
        hasNonPositive = np.any(y <= 0)

        if seasonal != 'N' and n < self.period * 2:
            return False

        if error == 'M' and hasNonPositive:
            return False

        if seasonal == 'M' and hasNonPositive:
            return False

        return True

    def _getNeighbors(
        self,
        error: str,
        trend: str,
        seasonal: str
    ) -> List[Tuple[str, str, str]]:
        """
        Generate neighbor models (change one component at a time)

        Returns
        -------
        List[Tuple[str, str, str]]
            List of neighbor models
        """
        errorOptions = ['A', 'M']
        trendOptions = ['N', 'A', 'Ad']
        seasonalOptions = ['N', 'A', 'M']

        neighbors = []

        for e in errorOptions:
            if e != error:
                neighbors.append((e, trend, seasonal))

        for t in trendOptions:
            if t != trend:
                neighbors.append((error, t, seasonal))

        for s in seasonalOptions:
            if s != seasonal:
                neighbors.append((error, trend, s))

        return neighbors

    def _evaluateModel(
        self,
        error: str,
        trend: str,
        seasonal: str,
        y: np.ndarray
    ) -> Optional[Tuple[ETSModel, float]]:
        """
        Evaluate a single model

        Returns
        -------
        Optional[Tuple[ETSModel, float]]
            (fitted model, AICc) or None on failure
        """
        modelKey = (error, trend, seasonal)
        if modelKey in self.allResults:
            return self.allResults[modelKey]

        if not self._isValidModel(error, trend, seasonal, y):
            self.allResults[modelKey] = None
            return None

        try:
            damped = 'd' in trend
            trendType = trend.replace('d', '')

            model = ETSModel(
                errorType=error,
                trendType=trendType,
                seasonalType=seasonal,
                period=self.period,
                damped=damped
            )
            model.fit(y)

            aic = self._computeAICc(y, model)

            if np.isfinite(aic):
                result = (model, aic)
                self.allResults[modelKey] = result
                return result
            else:
                self.allResults[modelKey] = None
                return None
        except Exception:
            self.allResults[modelKey] = None
            return None

    def fit(self, y: np.ndarray) -> ETSModel:
        """
        Automatic optimal ETS model selection and fitting

        Stepwise algorithm:
        1. Evaluate 4 seed models
        2. Search neighbors of the best seed
        3. Repeat until no improvement

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        ETSModel
            Optimal model
        """
        self.allResults = {}
        self.bestModel = None
        self.bestAIC = np.inf

        if self.exhaustive:
            return self._fitExhaustive(y)

        return self._fitStepwise(y)

    def _fitStepwise(self, y: np.ndarray) -> ETSModel:
        """Hyndman-Khandakar stepwise search"""
        bestError = None
        bestTrend = None
        bestSeasonal = None

        for error, trend, seasonal in self.SEED_MODELS:
            result = self._evaluateModel(error, trend, seasonal, y)
            if result is not None:
                model, aic = result
                if aic < self.bestAIC:
                    self.bestAIC = aic
                    self.bestModel = model
                    bestError = error
                    bestTrend = trend
                    bestSeasonal = seasonal

        if self.bestModel is None:
            return self._fallback(y)

        improved = True
        while improved:
            improved = False
            neighbors = self._getNeighbors(bestError, bestTrend, bestSeasonal)

            for error, trend, seasonal in neighbors:
                result = self._evaluateModel(error, trend, seasonal, y)
                if result is not None:
                    model, aic = result
                    if aic < self.bestAIC:
                        self.bestAIC = aic
                        self.bestModel = model
                        bestError = error
                        bestTrend = trend
                        bestSeasonal = seasonal
                        improved = True

        return self.bestModel

    def _fitExhaustive(self, y: np.ndarray) -> ETSModel:
        """Exhaustive model search"""
        for error, trend, seasonal in self.FULL_MODELS:
            result = self._evaluateModel(error, trend, seasonal, y)
            if result is not None:
                model, aic = result
                if aic < self.bestAIC:
                    self.bestAIC = aic
                    self.bestModel = model

        if self.bestModel is None:
            return self._fallback(y)

        return self.bestModel

    def _fallback(self, y: np.ndarray) -> ETSModel:
        """Fallback when all models fail"""
        self.bestModel = ETSModel(period=self.period)
        self.bestModel.fit(y, optimize=False)
        return self.bestModel

    def _computeAICc(self, y: np.ndarray, model: ETSModel) -> float:
        """
        Compute AICc

        Additive error: logL = -n/2 * log(SSE/n)
        Multiplicative error: logL = -n/2 * log(SSE_rel/n) - sum(log|y|)
          Uses relative error based log-likelihood

        AIC = -2*logL + 2*k
        AICc = AIC + 2*k*(k+1)/(n-k-1)
        """
        n = len(y)

        if model.residuals is None:
            return np.inf

        k = 1
        if model.trendType != 'N':
            k += 1
        if model.seasonalType != 'N':
            k += 1
        if model.damped:
            k += 1

        k += 1
        if model.trendType != 'N':
            k += 1
        if model.seasonalType != 'N':
            k += self.period

        if n - k - 1 <= 0:
            return np.inf

        if model.errorType == 'A':
            sse = np.sum(model.residuals ** 2)
            sigma2 = sse / n
            if sigma2 <= 0:
                return np.inf
            logL = -n / 2.0 * np.log(sigma2)
        else:
            fittedVals = y - model.residuals
            safeFitted = np.where(np.abs(fittedVals) < 1e-10, 1e-10, fittedVals)
            relativeErrors = model.residuals / safeFitted
            sseRelative = np.sum(relativeErrors ** 2)
            sigma2Relative = sseRelative / n
            if sigma2Relative <= 0:
                return np.inf
            logL = -n / 2.0 * np.log(sigma2Relative) - np.sum(np.log(np.abs(y) + 1e-10))

        aic = -2.0 * logL + 2.0 * k
        aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)

        return aicc

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast"""
        if self.bestModel is None:
            raise ValueError("Model has not been fitted.")
        return self.bestModel.predict(steps)
