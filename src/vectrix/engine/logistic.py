"""
Logistic Growth Model (Saturating Growth)

Pure numpy/scipy implementation of Prophet's core saturating growth model
- LogisticGrowthModel: Basic logistic curve fitting and forecasting
- SaturatingTrendModel: Saturating trend + seasonality combination

Reference: Taylor & Letham (2018) "Forecasting at Scale"
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def _logisticCurve(x, cap, floor, k, m):
    """Logistic function: floor + (cap - floor) / (1 + exp(-k*(x - m)))"""
    return floor + (cap - floor) / (1.0 + np.exp(-k * (x - m)))


class LogisticGrowthModel:
    """Logistic growth model (saturating growth)"""

    def __init__(self, cap: Optional[float] = None, floor: float = 0.0):
        """
        Parameters
        ----------
        cap : float or None
            Saturation cap. If None, auto-estimated as max(y) * 1.2
        floor : float
            Saturation floor
        """
        self.cap = cap
        self.floor = floor

        self.k = 0.0
        self.m = 0.0
        self.fittedCap = None
        self.residuals = None
        self.nObs = 0
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'LogisticGrowthModel':
        """
        Fit logistic curve

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        LogisticGrowthModel
            Fitted model
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)
        self.nObs = n

        if n < 4:
            self._fallbackFit(y)
            return self

        if self.cap is not None:
            self.fittedCap = float(self.cap)
        else:
            yMax = np.max(y)
            yMin = np.min(y)
            if np.abs(yMax - yMin) < 1e-10:
                self.fittedCap = yMax * 1.2 if yMax > 0 else 1.0
            else:
                self.fittedCap = yMax * 1.2

        if self.fittedCap <= self.floor:
            self.fittedCap = self.floor + np.abs(self.floor) + 1.0

        x = np.arange(n, dtype=np.float64)

        kInit = 4.0 / max(n - 1, 1)
        mInit = n / 2.0

        def _fixedCapLogistic(xVal, kParam, mParam):
            return _logisticCurve(xVal, self.fittedCap, self.floor, kParam, mParam)

        try:
            popt, _ = curve_fit(
                _fixedCapLogistic,
                x,
                y,
                p0=[kInit, mInit],
                maxfev=5000
            )
            self.k = popt[0]
            self.m = popt[1]
        except (RuntimeError, ValueError):
            self.k = kInit
            self.m = mInit

        fittedValues = _logisticCurve(x, self.fittedCap, self.floor, self.k, self.m)
        self.residuals = y - fittedValues
        self.fitted = True

        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast with confidence intervals

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
            raise ValueError("Model has not been fitted. Call fit() first.")

        if steps <= 0:
            return np.array([]), np.array([]), np.array([])

        futureX = np.arange(self.nObs, self.nObs + steps, dtype=np.float64)
        predictions = _logisticCurve(futureX, self.fittedCap, self.floor, self.k, self.m)

        if self.residuals is not None and len(self.residuals) > 1:
            sigma = np.std(self.residuals, ddof=1)
        else:
            sigma = 0.0

        horizons = np.arange(1, steps + 1, dtype=np.float64)
        margin = 1.96 * sigma * np.sqrt(horizons)
        lower95 = predictions - margin
        upper95 = predictions + margin

        return predictions, lower95, upper95

    def _fallbackFit(self, y: np.ndarray):
        """Simple fitting when data is insufficient"""
        n = len(y)
        if n == 0:
            self.fittedCap = 1.0
            self.k = 0.0
            self.m = 0.0
            self.residuals = np.array([])
            self.fitted = True
            return

        self.fittedCap = np.max(y) * 1.2 if np.max(y) > 0 else 1.0
        if self.fittedCap <= self.floor:
            self.fittedCap = self.floor + 1.0

        self.k = 0.01
        self.m = n / 2.0
        x = np.arange(n, dtype=np.float64)
        fittedValues = _logisticCurve(x, self.fittedCap, self.floor, self.k, self.m)
        self.residuals = y - fittedValues
        self.fitted = True


class SaturatingTrendModel:
    """Saturating trend + seasonality combination model"""

    def __init__(
        self,
        cap: Optional[float] = None,
        floor: float = 0.0,
        period: int = 1
    ):
        """
        Parameters
        ----------
        cap : float or None
            Saturation cap. If None, auto-estimated
        floor : float
            Saturation floor
        period : int
            Seasonal period (1 means no seasonality)
        """
        self.cap = cap
        self.floor = floor
        self.period = period

        self.logisticModel = None
        self.seasonal = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'SaturatingTrendModel':
        """
        Deseasonalize -> logistic fitting -> store seasonality

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        SaturatingTrendModel
            Fitted model
        """
        y = np.asarray(y, dtype=np.float64)
        n = len(y)

        if self.period > 1 and n >= self.period * 2:
            deseasonalized, self.seasonal = self._deseasonalize(y)
        else:
            deseasonalized = y
            self.seasonal = None

        self.logisticModel = LogisticGrowthModel(cap=self.cap, floor=self.floor)
        self.logisticModel.fit(deseasonalized)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast with seasonality restoration

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
            raise ValueError("Model has not been fitted. Call fit() first.")

        if steps <= 0:
            return np.array([]), np.array([]), np.array([])

        predictions, lower95, upper95 = self.logisticModel.predict(steps)

        if self.seasonal is not None:
            m = self.period
            for h in range(steps):
                seasonIdx = h % m
                predictions[h] += self.seasonal[seasonIdx]
                lower95[h] += self.seasonal[seasonIdx]
                upper95[h] += self.seasonal[seasonIdx]

        return predictions, lower95, upper95

    def _deseasonalize(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classical seasonal decomposition (additive model)

        Parameters
        ----------
        y : np.ndarray
            Original time series

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (deseasonalized time series, seasonal index array)
        """
        n = len(y)
        m = self.period
        globalMean = np.mean(y)

        seasonal = np.zeros(m)
        for i in range(m):
            vals = y[i::m]
            seasonal[i] = np.mean(vals) - globalMean

        deseasonalized = np.zeros(n)
        for t in range(n):
            deseasonalized[t] = y[t] - seasonal[t % m]

        return deseasonalized, seasonal
