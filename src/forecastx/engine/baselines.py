"""
Baseline forecasting models

Simple reference models for benchmarking:
- NaiveModel: Last value repeated
- SeasonalNaiveModel: Last season repeated
- MeanModel: Historical mean
- RandomWalkDrift: Last value + average drift
- WindowAverage: Moving window average
"""

import numpy as np
from typing import Tuple


class NaiveModel:
    """
    Naive (Random Walk) forecaster

    Predicts the last observed value for all future steps.
    """

    def __init__(self):
        self.lastValue = 0.0
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'NaiveModel':
        n = len(y)
        self.lastValue = y[-1]
        self.residuals = np.diff(y)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.full(steps, self.lastValue)

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 0 else 1.0
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper


class SeasonalNaiveModel:
    """
    Seasonal Naive forecaster

    Repeats the last observed seasonal cycle.
    """

    def __init__(self, period: int = 7):
        self.period = max(1, period)
        self.lastSeason = None
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'SeasonalNaiveModel':
        n = len(y)
        p = min(self.period, n)
        self.lastSeason = y[-p:].copy()

        if n > p:
            seasonalFitted = np.array([y[i - p] if i >= p else y[i] for i in range(n)])
            self.residuals = y[p:] - seasonalFitted[p:]
        else:
            self.residuals = np.zeros(1)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        p = len(self.lastSeason)
        predictions = np.tile(self.lastSeason, steps // p + 1)[:steps]

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 0 else 1.0
        k = np.array([(h // p) + 1 for h in range(steps)], dtype=np.float64)
        margin = 1.96 * sigma * np.sqrt(k)
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper


class MeanModel:
    """
    Mean forecaster

    Predicts the historical mean for all future steps.
    """

    def __init__(self):
        self.mean = 0.0
        self.sigma = 1.0
        self.n = 0
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'MeanModel':
        self.n = len(y)
        self.mean = np.mean(y)
        self.sigma = np.std(y, ddof=1) if self.n > 1 else 1.0
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.full(steps, self.mean)

        margin = 1.96 * self.sigma * np.sqrt(1 + 1.0 / self.n)
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper


class RandomWalkDrift:
    """
    Random Walk with Drift

    Predicts the last value plus average drift (trend).
    Equivalent to a line extending from the last point with slope = mean of differences.
    """

    def __init__(self):
        self.lastValue = 0.0
        self.drift = 0.0
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'RandomWalkDrift':
        n = len(y)
        self.lastValue = y[-1]
        self.drift = (y[-1] - y[0]) / max(n - 1, 1)

        if n > 1:
            diffs = np.diff(y)
            fittedDiffs = np.full(n - 1, self.drift)
            self.residuals = diffs - fittedDiffs
        else:
            self.residuals = np.zeros(1)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        h = np.arange(1, steps + 1, dtype=np.float64)
        predictions = self.lastValue + self.drift * h

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 0 else 1.0
        margin = 1.96 * sigma * np.sqrt(h)
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper


class WindowAverage:
    """
    Window Average forecaster

    Predicts the average of the last `window` observations.
    """

    def __init__(self, window: int = 7):
        self.window = max(1, window)
        self.windowMean = 0.0
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'WindowAverage':
        n = len(y)
        w = min(self.window, n)
        self.windowMean = np.mean(y[-w:])

        if n > w:
            fittedVals = np.array([
                np.mean(y[max(0, i - w):i]) if i >= w else np.mean(y[:i + 1])
                for i in range(n)
            ])
            self.residuals = y - fittedVals
        else:
            self.residuals = np.zeros(1)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.full(steps, self.windowMean)

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 0 else 1.0
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper
