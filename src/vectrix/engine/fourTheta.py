"""
Adaptive Theta Ensemble (4Theta)

Independent forecasts from 4 theta lines (theta=0,1,2,3) with holdout sMAPE-based weighted combination.
Extension of the M4 Competition 3rd place methodology.

E034 experiment result -> E041 stress test passed -> engine integration.
- Average rank 2.73 (1st): exceeds mstl(3.27)
- Safety 100%, Seed CV < 20%, Speed 53ms (n=1000)
"""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar


def _sesFilter(y: np.ndarray, alpha: float) -> np.ndarray:
    n = len(y)
    result = np.zeros(n)
    result[0] = y[0]
    for t in range(1, n):
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1]
    return result


def _sesSSE(y: np.ndarray, alpha: float) -> float:
    n = len(y)
    level = y[0]
    sse = 0.0
    for t in range(1, n):
        error = y[t] - level
        sse += error * error
        level = alpha * y[t] + (1.0 - alpha) * level
    return sse


def _optimizeAlpha(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.3
    result = minimize_scalar(lambda a: _sesSSE(y, a), bounds=(0.001, 0.999), method='bounded')
    return result.x if result.success else 0.3


def _linearRegression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    xMean = np.mean(x)
    yMean = np.mean(y)
    num = np.sum((x - xMean) * (y - yMean))
    den = np.sum((x - xMean) ** 2)
    slope = num / max(den, 1e-10)
    intercept = yMean - slope * xMean
    return slope, intercept


class AdaptiveThetaEnsemble:
    """
    4Theta: Multi Theta Line Ensemble

    Fits theta=0 (pure trend), theta=1 (original), theta=2 (2x curvature), theta=3 (3x curvature)
    with SES and combines using inverse holdout sMAPE weights.

    Includes automatic seasonal decomposition (multiplicative/additive).
    """

    def __init__(self, thetaValues=None, period=None, holdoutRatio=0.15):
        self.thetaValues = thetaValues if thetaValues else [0, 1, 2, 3]
        self.period = period
        self.holdoutRatio = holdoutRatio

        self._y = None
        self._seasonal = None
        self._seasonType = None
        self._models = []
        self._weights = None
        self._n = 0

        self.fitted = False
        self.residuals = None

    def fit(self, y: np.ndarray) -> 'AdaptiveThetaEnsemble':
        self._y = np.asarray(y, dtype=np.float64).copy()
        self._n = len(self._y)

        if self.period is None:
            self.period = self._detectPeriod(self._y)

        if self.period > 1 and self._n >= self.period * 3:
            self._seasonal, self._seasonType, deseasonalized = self._deseasonalize(self._y, self.period)
        else:
            deseasonalized = self._y
            self._seasonal = None
            self._seasonType = 'none'

        holdoutSize = max(1, int(len(deseasonalized) * self.holdoutRatio))
        holdoutSize = min(holdoutSize, len(deseasonalized) // 3)
        trainPart = deseasonalized[:len(deseasonalized) - holdoutSize]
        valPart = deseasonalized[len(deseasonalized) - holdoutSize:]

        self._models = []
        smapes = []

        for theta in self.thetaValues:
            model = self._fitThetaLine(trainPart, theta)
            pred = self._predictThetaLine(model, holdoutSize)
            smape = np.mean(2.0 * np.abs(valPart - pred) / (np.abs(valPart) + np.abs(pred) + 1e-10))
            smapes.append(smape)
            self._models.append(model)

        smapes = np.array(smapes)
        invSmapes = 1.0 / np.maximum(smapes, 1e-10)
        self._weights = invSmapes / invSmapes.sum()

        self._models = []
        for theta in self.thetaValues:
            model = self._fitThetaLine(deseasonalized, theta)
            self._models.append(model)

        fittedValues = np.zeros(self._n)
        for i, model in enumerate(self._models):
            for t in range(self._n):
                trendPred = model['intercept'] + model['slope'] * t
                sesPred = model['filtered'][t] if t < len(model['filtered']) else model['lastLevel']
                if model['theta'] == 0:
                    fittedValues[t] += self._weights[i] * trendPred
                else:
                    fittedValues[t] += self._weights[i] * (trendPred + sesPred) / 2.0

        if self._seasonal is not None:
            for t in range(self._n):
                seasonIdx = t % self.period
                if self._seasonType == 'multiplicative':
                    fittedValues[t] *= self._seasonal[seasonIdx]
                else:
                    fittedValues[t] += self._seasonal[seasonIdx]

        self.residuals = self._y - fittedValues
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("fit() must be called before predict()")

        allPreds = np.zeros((len(self._models), steps))
        for i, model in enumerate(self._models):
            allPreds[i] = self._predictThetaLine(model, steps)

        combined = np.average(allPreds, axis=0, weights=self._weights)

        if self._seasonal is not None:
            for h in range(steps):
                seasonIdx = h % self.period
                if self._seasonType == 'multiplicative':
                    combined[h] *= self._seasonal[seasonIdx]
                else:
                    combined[h] += self._seasonal[seasonIdx]

        allResidStds = [m['residStd'] for m in self._models]
        avgStd = np.average(allResidStds, weights=self._weights)
        sigma = avgStd * np.sqrt(np.arange(1, steps + 1))
        lower = combined - 1.96 * sigma
        upper = combined + 1.96 * sigma

        return combined, lower, upper

    def _fitThetaLine(self, y, theta):
        n = len(y)
        x = np.arange(n, dtype=np.float64)
        slope, intercept = _linearRegression(x, y)

        if theta == 0:
            thetaLine = intercept + slope * x
        elif theta == 1:
            thetaLine = y.copy()
        else:
            trendLine = intercept + slope * x
            thetaLine = theta * y - (theta - 1) * trendLine

        alpha = _optimizeAlpha(thetaLine)
        filtered = _sesFilter(thetaLine, alpha)
        lastLevel = filtered[-1]

        fittedVals = np.zeros(n)
        for t in range(n):
            trendPred = intercept + slope * t
            sesPred = filtered[t]
            if theta == 0:
                fittedVals[t] = trendPred
            else:
                fittedVals[t] = (trendPred + sesPred) / 2.0

        residuals = y - fittedVals
        residStd = max(np.std(residuals), 1e-8)

        return {
            'theta': theta,
            'slope': slope,
            'intercept': intercept,
            'alpha': alpha,
            'lastLevel': lastLevel,
            'filtered': filtered,
            'n': n,
            'residStd': residStd,
        }

    def _predictThetaLine(self, model, steps):
        predictions = np.zeros(steps)
        for h in range(steps):
            t = model['n'] + h
            trendPred = model['intercept'] + model['slope'] * t
            sesPred = model['lastLevel']

            if model['theta'] == 0:
                predictions[h] = trendPred
            else:
                predictions[h] = (trendPred + sesPred) / 2.0

        return predictions

    def _detectPeriod(self, y):
        n = len(y)
        if n < 14:
            return 1
        from scipy.signal import periodogram
        detrended = y - np.linspace(y[0], y[-1], n)
        freqs, power = periodogram(detrended)
        validMask = freqs > 0
        freqs = freqs[validMask]
        power = power[validMask]
        if len(freqs) == 0:
            return 1
        peakIdx = np.argmax(power)
        freq = freqs[peakIdx]
        if freq > 0:
            period = int(round(1.0 / freq))
            if 2 <= period <= n // 4:
                return period
        return 1

    def _deseasonalize(self, y, period):
        n = len(y)
        seasonal = np.zeros(period)
        counts = np.zeros(period)

        trend = np.convolve(y, np.ones(period) / period, mode='valid')
        offset = (period - 1) // 2

        minVal = np.min(y)
        useMultiplicative = minVal > 0

        if useMultiplicative:
            for i in range(len(trend)):
                idx = i + offset
                if idx < n and trend[i] > 0:
                    ratio = y[idx] / trend[i]
                    seasonal[idx % period] += ratio
                    counts[idx % period] += 1

            for i in range(period):
                seasonal[i] = seasonal[i] / max(counts[i], 1)

            meanSeasonal = np.mean(seasonal)
            if meanSeasonal > 0:
                seasonal /= meanSeasonal

            seasonal = np.maximum(seasonal, 0.01)
            deseasonalized = np.zeros(n)
            for i in range(n):
                deseasonalized[i] = y[i] / seasonal[i % period]

            return seasonal, 'multiplicative', deseasonalized

        for i in range(len(trend)):
            idx = i + offset
            if idx < n:
                diff = y[idx] - trend[i]
                seasonal[idx % period] += diff
                counts[idx % period] += 1

        for i in range(period):
            seasonal[i] = seasonal[i] / max(counts[i], 1)

        seasonal -= np.mean(seasonal)
        deseasonalized = np.zeros(n)
        for i in range(n):
            deseasonalized[i] = y[i] - seasonal[i % period]

        return seasonal, 'additive', deseasonalized
