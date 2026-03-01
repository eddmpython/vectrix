"""
Time Series Decomposition Implementation

- Classical Decomposition (Additive/Multiplicative)
- STL-like Decomposition (LOESS based)
- Multiple Seasonality Decomposition (MSTL-like)

All implemented directly with numpy + numba
"""

from typing import List, NamedTuple

import numpy as np

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


class DecompositionResult(NamedTuple):
    """Decomposition result"""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    observed: np.ndarray


class SeasonalDecomposition:
    """
    Seasonal Decomposition Implementation

    Supports both Classical and STL-like methods
    """

    def __init__(
        self,
        period: int = 7,
        model: str = 'additive',
        method: str = 'classical'
    ):
        """
        Parameters
        ----------
        period : int
            Seasonal period
        model : str
            'additive' or 'multiplicative'
        method : str
            'classical', 'stl', 'mstl'
        """
        self.period = period
        self.model = model
        self.method = method

    def decompose(self, y: np.ndarray) -> DecompositionResult:
        """
        Decompose time series

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        DecompositionResult
            (trend, seasonal, residual, observed)
        """
        if self.method == 'classical':
            return self._classicalDecomposition(y)
        elif self.method == 'stl':
            return self._stlDecomposition(y)
        else:
            return self._classicalDecomposition(y)

    def _classicalDecomposition(self, y: np.ndarray) -> DecompositionResult:
        """Classical decomposition"""
        n = len(y)
        m = self.period

        trend = self._computeTrend(y, m)

        if self.model == 'additive':
            detrended = y - trend
        else:  # multiplicative
            detrended = y / (trend + 1e-10)

        seasonal = self._computeSeasonal(detrended, m)

        if self.model == 'additive':
            residual = y - trend - seasonal
        else:
            residual = y / ((trend + 1e-10) * (seasonal + 1e-10))

        return DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            observed=y
        )

    def _stlDecomposition(self, y: np.ndarray) -> DecompositionResult:
        """
        STL-like decomposition (simplified)

        Actual STL uses iterative LOESS,
        here a simplified version is implemented
        """
        n = len(y)
        m = self.period

        trend = np.zeros(n)
        seasonal = np.zeros(n)

        trend = self._loessSmooth(y, span=max(7, m))

        for iteration in range(3):
            detrended = y - trend

            seasonal = self._computeSubseriesSeasonal(detrended, m)

            seasonAdjusted = y - seasonal

            trend = self._loessSmooth(seasonAdjusted, span=max(7, m))

        residual = y - trend - seasonal

        return DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            observed=y
        )

    def _computeTrend(self, y: np.ndarray, m: int) -> np.ndarray:
        """Extract trend via centered moving average"""
        n = len(y)
        trend = np.full(n, np.nan)

        if m % 2 == 1:
            halfM = m // 2
            for t in range(halfM, n - halfM):
                trend[t] = np.mean(y[t - halfM:t + halfM + 1])
        else:
            halfM = m // 2
            for t in range(halfM, n - halfM):
                if t - halfM >= 0 and t + halfM < n:
                    ma = np.mean(y[t - halfM:t + halfM])
                    if t + halfM + 1 < n:
                        ma2 = np.mean(y[t - halfM + 1:t + halfM + 1])
                        trend[t] = (ma + ma2) / 2
                    else:
                        trend[t] = ma

        trend = self._interpolateNaN(trend, y)

        return trend

    def _computeSeasonal(self, detrended: np.ndarray, m: int) -> np.ndarray:
        """Extract seasonal component"""
        n = len(detrended)
        seasonal = np.zeros(n)

        seasonalMeans = np.zeros(m)
        for i in range(m):
            vals = detrended[i::m]
            validVals = vals[~np.isnan(vals)]
            if len(validVals) > 0:
                seasonalMeans[i] = np.mean(validVals)

        seasonalMeans = seasonalMeans - np.mean(seasonalMeans)

        for t in range(n):
            seasonal[t] = seasonalMeans[t % m]

        return seasonal

    def _computeSubseriesSeasonal(self, detrended: np.ndarray, m: int) -> np.ndarray:
        """Extract seasonal component via subseries smoothing (STL style)"""
        n = len(detrended)
        seasonal = np.zeros(n)

        for i in range(m):
            subseries = detrended[i::m]
            if len(subseries) >= 3:
                smoothed = self._loessSmooth(subseries, span=max(3, len(subseries) // 2))
            else:
                smoothed = subseries

            for j, val in enumerate(smoothed):
                idx = i + j * m
                if idx < n:
                    seasonal[idx] = val

        seasonalMean = np.zeros(m)
        for i in range(m):
            seasonalMean[i] = np.mean(seasonal[i::m])
        overallMean = np.mean(seasonalMean)

        for t in range(n):
            seasonal[t] -= overallMean

        return seasonal

    def _loessSmooth(self, y: np.ndarray, span: int = 7) -> np.ndarray:
        """
        Simplified LOESS smoothing

        Actual LOESS uses weighted regression,
        here approximated with weighted moving average
        """
        n = len(y)
        smoothed = np.zeros(n)

        halfSpan = span // 2

        for i in range(n):
            start = max(0, i - halfSpan)
            end = min(n, i + halfSpan + 1)

            weights = np.zeros(end - start)
            for j in range(len(weights)):
                dist = abs((start + j) - i)
                weights[j] = 1 - (dist / (halfSpan + 1))

            weights = weights / weights.sum()

            windowData = y[start:end]
            validMask = ~np.isnan(windowData)

            if validMask.any():
                validWeights = weights[validMask]
                validWeights = validWeights / validWeights.sum()
                smoothed[i] = np.sum(windowData[validMask] * validWeights)
            else:
                smoothed[i] = y[i] if not np.isnan(y[i]) else 0

        return smoothed

    def _interpolateNaN(self, y: np.ndarray, original: np.ndarray) -> np.ndarray:
        """NaN linear interpolation"""
        result = y.copy()
        n = len(y)

        for i in range(n):
            if np.isnan(result[i]):
                result[i] = original[i]
            else:
                break

        for i in range(n - 1, -1, -1):
            if np.isnan(result[i]):
                result[i] = original[i]
            else:
                break

        for i in range(n):
            if np.isnan(result[i]):
                prevIdx = i - 1
                while prevIdx >= 0 and np.isnan(result[prevIdx]):
                    prevIdx -= 1

                nextIdx = i + 1
                while nextIdx < n and np.isnan(result[nextIdx]):
                    nextIdx += 1

                if prevIdx >= 0 and nextIdx < n:
                    ratio = (i - prevIdx) / (nextIdx - prevIdx)
                    result[i] = result[prevIdx] + ratio * (result[nextIdx] - result[prevIdx])
                elif prevIdx >= 0:
                    result[i] = result[prevIdx]
                elif nextIdx < n:
                    result[i] = result[nextIdx]
                else:
                    result[i] = 0

        return result

    def extractSeasonal(self, y: np.ndarray) -> np.ndarray:
        """Extract only the seasonal component"""
        result = self.decompose(y)
        return result.seasonal

    def extractTrend(self, y: np.ndarray) -> np.ndarray:
        """Extract only the trend component"""
        result = self.decompose(y)
        return result.trend

    def deseasonalize(self, y: np.ndarray) -> np.ndarray:
        """Seasonally adjusted time series"""
        result = self.decompose(y)
        if self.model == 'additive':
            return y - result.seasonal
        else:
            return y / (result.seasonal + 1e-10)


class MSTLDecomposition:
    """
    Multiple Seasonality Decomposition (MSTL-like)

    Sequentially decomposes multiple seasonal periods
    """

    def __init__(self, periods: List[int], model: str = 'additive'):
        """
        Parameters
        ----------
        periods : List[int]
            List of seasonal periods (e.g., [7, 30, 365])
        model : str
            'additive' or 'multiplicative'
        """
        self.periods = sorted(periods)
        self.model = model

    def decompose(self, y: np.ndarray) -> dict:
        """
        Multiple seasonal decomposition

        Returns
        -------
        dict
            {'trend': array, 'seasonals': {period: array}, 'residual': array}
        """
        n = len(y)
        remaining = y.copy()
        seasonals = {}

        for period in self.periods:
            if n < period * 2:
                continue

            decomposer = SeasonalDecomposition(
                period=period,
                model=self.model,
                method='stl'
            )

            result = decomposer.decompose(remaining)
            seasonals[period] = result.seasonal

            if self.model == 'additive':
                remaining = remaining - result.seasonal
            else:
                remaining = remaining / (result.seasonal + 1e-10)

        trend = TurboCore.rollingMean(remaining, min(7, n // 4))
        residual = remaining - trend

        return {
            'trend': trend,
            'seasonals': seasonals,
            'residual': residual,
            'observed': y
        }

    def predict(self, y: np.ndarray, steps: int) -> np.ndarray:
        """
        Forecast based on multiple seasonal decomposition
        """
        decomposition = self.decompose(y)
        n = len(y)

        x = np.arange(n, dtype=np.float64)
        slope, intercept = TurboCore.linearRegression(x, decomposition['trend'])

        predictions = np.zeros(steps)

        for h in range(steps):
            t = n + h

            pred = intercept + slope * t

            for period, seasonal in decomposition['seasonals'].items():
                seasonIdx = (n + h) % period
                if seasonIdx < len(seasonal):
                    if self.model == 'additive':
                        pred += seasonal[seasonIdx]
                    else:
                        pred *= seasonal[seasonIdx]

            predictions[h] = pred

        return predictions
