"""
Business Metrics

Business-oriented forecast metrics for practical use:
- Bias: Forecast bias (over/under prediction)
- Tracking Signal: Cumulative bias signal
- WAPE: Weighted Absolute Percentage Error
- Value-Weighted Accuracy: Value-weighted accuracy
- Fill Rate Impact: Inventory fill rate impact
"""

from typing import Any, Dict, Optional

import numpy as np


class BusinessMetrics:
    """
    Business metrics calculator

    Usage:
        >>> metrics = BusinessMetrics()
        >>> result = metrics.calculate(actual, predicted)
    """

    def calculate(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
        values: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate all business metrics

        Parameters
        ----------
        actual : np.ndarray
            Actual values
        predicted : np.ndarray
            Predicted values
        values : np.ndarray, optional
            Monetary/value weights (for value-weighted metrics)
        """
        n = min(len(actual), len(predicted))
        actual = actual[:n]
        predicted = predicted[:n]

        errors = predicted - actual
        absErrors = np.abs(errors)

        result = {
            'bias': self.bias(actual, predicted),
            'biasPercent': self.biasPercent(actual, predicted),
            'trackingSignal': self.trackingSignal(actual, predicted),
            'wape': self.wape(actual, predicted),
            'mase': self.mase(actual, predicted),
            'overForecastRatio': self.overForecastRatio(actual, predicted),
            'underForecastRatio': self.underForecastRatio(actual, predicted),
        }

        if values is not None:
            result['valueWeightedAccuracy'] = self.valueWeightedAccuracy(
                actual, predicted, values[:n]
            )

        result['fillRateImpact'] = self.fillRateImpact(actual, predicted)
        result['forecastAccuracy'] = max(0, 100 - result['wape'])

        return result

    @staticmethod
    def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
        return float(np.mean(predicted - actual))

    @staticmethod
    def biasPercent(actual: np.ndarray, predicted: np.ndarray) -> float:
        totalActual = np.sum(np.abs(actual))
        if totalActual < 1e-10:
            return 0.0
        return float(np.sum(predicted - actual) / totalActual * 100)

    @staticmethod
    def trackingSignal(actual: np.ndarray, predicted: np.ndarray, smoothing: float = 0.1) -> float:
        errors = predicted - actual
        n = len(errors)

        cfe = 0.0
        mad = 0.0

        for t in range(n):
            cfe += errors[t]
            if t == 0:
                mad = abs(errors[t])
            else:
                mad = smoothing * abs(errors[t]) + (1 - smoothing) * mad

        if mad < 1e-10:
            return 0.0
        return float(cfe / mad)

    @staticmethod
    def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
        totalActual = np.sum(np.abs(actual))
        if totalActual < 1e-10:
            return 0.0
        return float(np.sum(np.abs(predicted - actual)) / totalActual * 100)

    @staticmethod
    def mase(actual: np.ndarray, predicted: np.ndarray, seasonalPeriod: int = 1) -> float:
        n = len(actual)
        if n <= seasonalPeriod:
            return np.inf

        naiveErrors = np.abs(actual[seasonalPeriod:] - actual[:-seasonalPeriod])
        naiveMAE = np.mean(naiveErrors)

        if naiveMAE < 1e-10:
            return 0.0

        forecastMAE = np.mean(np.abs(actual - predicted))
        return float(forecastMAE / naiveMAE)

    @staticmethod
    def overForecastRatio(actual: np.ndarray, predicted: np.ndarray) -> float:
        overCount = np.sum(predicted > actual)
        return float(overCount / len(actual)) if len(actual) > 0 else 0.0

    @staticmethod
    def underForecastRatio(actual: np.ndarray, predicted: np.ndarray) -> float:
        underCount = np.sum(predicted < actual)
        return float(underCount / len(actual)) if len(actual) > 0 else 0.0

    @staticmethod
    def valueWeightedAccuracy(
        actual: np.ndarray,
        predicted: np.ndarray,
        values: np.ndarray
    ) -> float:
        totalValue = np.sum(values)
        if totalValue < 1e-10:
            return 0.0

        weightedErrors = np.abs(predicted - actual) * values
        return float(100 - np.sum(weightedErrors) / totalValue * 100)

    @staticmethod
    def fillRateImpact(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        underForecast = np.maximum(actual - predicted, 0)
        totalDemand = np.sum(actual)

        if totalDemand < 1e-10:
            return {'potentialStockout': 0.0, 'fillRate': 100.0}

        stockoutRisk = np.sum(underForecast) / totalDemand * 100
        fillRate = max(0, 100 - stockoutRisk)

        return {
            'potentialStockout': float(stockoutRisk),
            'fillRate': float(fillRate)
        }
