"""
TurboCore: Numba-based High-Performance Computation Core

Core operations for time series analysis optimized with Numba JIT
- ACF/PACF computation
- FFT-based seasonality detection
- Rolling statistics
- Differencing/Integration
- Evaluation metrics (MAPE, RMSE, MAE)
"""

from typing import Tuple

import numpy as np

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


class TurboCore:
    """Numba-based high-performance computation core"""

    @staticmethod
    @jit(nopython=True, cache=True)
    def acf(x: np.ndarray, maxLag: int) -> np.ndarray:
        """
        Autocorrelation Function (ACF) computation

        Parameters
        ----------
        x : np.ndarray
            Time series data
        maxLag : int
            Maximum lag

        Returns
        -------
        np.ndarray
            ACF values (lag 0 to maxLag)
        """
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)

        if var < 1e-10:
            return np.zeros(maxLag + 1)

        acf = np.zeros(maxLag + 1)
        acf[0] = 1.0

        for lag in range(1, min(maxLag + 1, n)):
            cov = 0.0
            for i in range(n - lag):
                cov += (x[i] - mean) * (x[i + lag] - mean)
            acf[lag] = cov / (n * var)

        return acf

    @staticmethod
    @jit(nopython=True, cache=True)
    def pacf(x: np.ndarray, maxLag: int) -> np.ndarray:
        """
        Partial Autocorrelation Function (PACF) - Durbin-Levinson algorithm

        Parameters
        ----------
        x : np.ndarray
            Time series data
        maxLag : int
            Maximum lag

        Returns
        -------
        np.ndarray
            PACF values
        """
        n = len(x)
        acfVals = TurboCore.acf(x, maxLag)

        pacf = np.zeros(maxLag + 1)
        pacf[0] = 1.0

        if maxLag < 1:
            return pacf

        phi = np.zeros((maxLag + 1, maxLag + 1))
        phi[1, 1] = acfVals[1]
        pacf[1] = acfVals[1]

        for k in range(2, maxLag + 1):
            num = acfVals[k]
            den = 1.0

            for j in range(1, k):
                num -= phi[k-1, j] * acfVals[k - j]
                den -= phi[k-1, j] * acfVals[j]

            if abs(den) < 1e-10:
                phi[k, k] = 0.0
            else:
                phi[k, k] = num / den

            pacf[k] = phi[k, k]

            for j in range(1, k):
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k - j]

        return pacf

    @staticmethod
    @jit(nopython=True, cache=True)
    def diff(x: np.ndarray, d: int = 1) -> np.ndarray:
        """
        Differencing

        Parameters
        ----------
        x : np.ndarray
            Time series data
        d : int
            Differencing order

        Returns
        -------
        np.ndarray
            Differenced data
        """
        result = x.copy()
        for _ in range(d):
            newResult = np.zeros(len(result) - 1)
            for i in range(len(newResult)):
                newResult[i] = result[i + 1] - result[i]
            result = newResult
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def integrate(x: np.ndarray, initial: float, d: int = 1) -> np.ndarray:
        """
        Integration (inverse of differencing)

        Parameters
        ----------
        x : np.ndarray
            Differenced data
        initial : float
            Initial value
        d : int
            Integration order

        Returns
        -------
        np.ndarray
            Restored data
        """
        result = x.copy()
        for _ in range(d):
            newResult = np.zeros(len(result) + 1)
            newResult[0] = initial
            for i in range(len(result)):
                newResult[i + 1] = newResult[i] + result[i]
            result = newResult
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def seasonalDiff(x: np.ndarray, period: int) -> np.ndarray:
        """Seasonal differencing"""
        n = len(x)
        result = np.zeros(n - period)
        for i in range(n - period):
            result[i] = x[i + period] - x[i]
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def rollingMean(x: np.ndarray, window: int) -> np.ndarray:
        """Rolling mean"""
        n = len(x)
        result = np.zeros(n)

        cumSum = 0.0
        for i in range(min(window, n)):
            cumSum += x[i]
            result[i] = cumSum / (i + 1)

        for i in range(window, n):
            cumSum += x[i] - x[i - window]
            result[i] = cumSum / window

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def rollingStd(x: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation - O(n) Welford online algorithm"""
        n = len(x)
        result = np.zeros(n)

        sumX = 0.0
        sumX2 = 0.0

        for i in range(n):
            sumX += x[i]
            sumX2 += x[i] * x[i]

            if i >= window:
                sumX -= x[i - window]
                sumX2 -= x[i - window] * x[i - window]

            count = min(i + 1, window)
            if count > 1:
                mean = sumX / count
                variance = sumX2 / count - mean * mean
                if variance > 0.0:
                    result[i] = np.sqrt(variance)

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
        """Exponentially Weighted Moving Average (EWMA)"""
        n = len(x)
        result = np.zeros(n)
        result[0] = x[0]

        for i in range(1, n):
            result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Compute MAPE"""
        n = len(actual)
        total = 0.0
        count = 0

        for i in range(n):
            if abs(actual[i]) > 1e-10:
                total += abs((actual[i] - predicted[i]) / actual[i])
                count += 1

        if count == 0:
            return np.inf

        return total / count * 100

    @staticmethod
    @jit(nopython=True, cache=True)
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Compute RMSE"""
        n = len(actual)
        total = 0.0

        for i in range(n):
            total += (actual[i] - predicted[i]) ** 2

        return np.sqrt(total / n)

    @staticmethod
    @jit(nopython=True, cache=True)
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Compute MAE"""
        n = len(actual)
        total = 0.0

        for i in range(n):
            total += abs(actual[i] - predicted[i])

        return total / n

    @staticmethod
    @jit(nopython=True, cache=True)
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Compute SMAPE"""
        n = len(actual)
        total = 0.0
        count = 0

        for i in range(n):
            denom = abs(actual[i]) + abs(predicted[i])
            if denom > 1e-10:
                total += 2 * abs(actual[i] - predicted[i]) / denom
                count += 1

        if count == 0:
            return np.inf

        return total / count * 100

    @staticmethod
    def fftSeasonality(x: np.ndarray, topK: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        FFT-based seasonal period detection

        Parameters
        ----------
        x : np.ndarray
            Time series data
        topK : int
            Number of top periods to return

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (period array, strength array)
        """
        n = len(x)

        if n < 10:
            return np.array([7]), np.array([0.0])

        detrended = x - np.linspace(x[0], x[-1], n)

        fft = np.fft.fft(detrended)
        magnitudes = np.abs(fft)
        magnitudes[0] = 0
        magnitudes[n // 2:] = 0

        indices = np.argsort(magnitudes)[::-1][:topK * 2]

        periods = []
        strengths = []

        for idx in indices:
            if idx > 0:
                period = int(round(n / idx))
                if 2 <= period <= n // 2 and period not in periods:
                    periods.append(period)
                    strengths.append(magnitudes[idx])

                    if len(periods) >= topK:
                        break

        if not periods:
            periods = [7]
            strengths = [0.0]

        return np.array(periods), np.array(strengths)

    @staticmethod
    @jit(nopython=True, cache=True)
    def adfStatistic(x: np.ndarray, maxLag: int = 12) -> float:
        """
        ADF test statistic computation (simplified)

        Returns
        -------
        float
            ADF statistic (more negative indicates stationarity)
        """
        n = len(x)

        if n < maxLag + 2:
            return 0.0

        dx = np.zeros(n - 1)
        for i in range(n - 1):
            dx[i] = x[i + 1] - x[i]

        lag = min(maxLag, int((n - 1) ** (1/3)))

        xLag = x[lag:-1]
        dxCurrent = dx[lag:]

        m = len(dxCurrent)
        if m < 10:
            return 0.0

        meanX = np.mean(xLag)
        meanDx = np.mean(dxCurrent)

        num = 0.0
        denX = 0.0
        denDx = 0.0

        for i in range(m):
            devX = xLag[i] - meanX
            devDx = dxCurrent[i] - meanDx
            num += devX * devDx
            denX += devX * devX
            denDx += devDx * devDx

        if denX < 1e-10:
            return 0.0

        beta = num / denX

        residuals = np.zeros(m)
        for i in range(m):
            residuals[i] = dxCurrent[i] - beta * (xLag[i] - meanX)

        residVar = np.var(residuals)
        seBeta = np.sqrt(residVar / denX) if denX > 0 else 1.0

        if seBeta < 1e-10:
            return 0.0

        tStat = beta / seBeta

        return tStat

    @staticmethod
    @jit(nopython=True, cache=True)
    def linearRegression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Simple linear regression

        Returns
        -------
        Tuple[float, float]
            (slope, intercept)
        """
        n = len(x)
        meanX = np.mean(x)
        meanY = np.mean(y)

        num = 0.0
        den = 0.0

        for i in range(n):
            devX = x[i] - meanX
            num += devX * (y[i] - meanY)
            den += devX * devX

        if den < 1e-10:
            return 0.0, meanY

        slope = num / den
        intercept = meanY - slope * meanX

        return slope, intercept


def isNumbaAvailable() -> bool:
    return NUMBA_AVAILABLE
