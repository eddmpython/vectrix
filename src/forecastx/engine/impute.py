"""
Time Series Missing Value Imputation

Provides multiple strategies for handling missing values in time series:
linear interpolation, seasonal interpolation, LOCF, NOCB, and
automatic method selection.
"""

import numpy as np
from typing import Dict, Any


class TimeSeriesImputer:
    """시계열 결측값 처리"""

    def impute(self, y: np.ndarray, method: str = 'auto', period: int = 1) -> np.ndarray:
        """결측값 보간"""
        y = np.asarray(y, dtype=np.float64).copy()

        if len(y) == 0:
            return y

        if not np.any(np.isnan(y)):
            return y

        if np.all(np.isnan(y)):
            return np.zeros_like(y)

        if method == 'linear':
            return self.linearInterpolate(y)
        elif method == 'seasonal':
            return self.seasonalInterpolate(y, period)
        elif method == 'locf':
            return self.locf(y)
        elif method == 'nocb':
            return self.nocb(y)
        elif method == 'auto':
            return self._autoImpute(y, period)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _autoImpute(self, y: np.ndarray, period: int) -> np.ndarray:
        """자동 보간 전략 선택"""
        y = y.copy()

        if period > 1 and len(y) >= 2 * period:
            result = self.seasonalInterpolate(y, period)
            if not np.any(np.isnan(result)):
                return result
            y = result

        if np.any(np.isnan(y)):
            result = self.linearInterpolate(y)
            if not np.any(np.isnan(result)):
                return result
            y = result

        if np.any(np.isnan(y)):
            y = self.locf(y)

        if np.any(np.isnan(y)):
            y = self.nocb(y)

        return y

    def linearInterpolate(self, y: np.ndarray) -> np.ndarray:
        """선형 보간 (interior NaN)"""
        y = np.asarray(y, dtype=np.float64).copy()

        if len(y) == 0:
            return y

        if np.all(np.isnan(y)):
            return np.zeros_like(y)

        nanMask = np.isnan(y)
        if not np.any(nanMask):
            return y

        validIdx = np.where(~nanMask)[0]
        if len(validIdx) < 2:
            fillVal = y[validIdx[0]] if len(validIdx) == 1 else 0.0
            y[nanMask] = fillVal
            return y

        nanIdx = np.where(nanMask)[0]

        interiorMask = (nanIdx >= validIdx[0]) & (nanIdx <= validIdx[-1])
        interiorNans = nanIdx[interiorMask]

        if len(interiorNans) > 0:
            y[interiorNans] = np.interp(interiorNans, validIdx, y[validIdx])

        return y

    def seasonalInterpolate(self, y: np.ndarray, period: int) -> np.ndarray:
        """계절 패턴 기반 보간"""
        y = np.asarray(y, dtype=np.float64).copy()
        period = max(1, period)

        if len(y) == 0:
            return y

        if np.all(np.isnan(y)):
            return np.zeros_like(y)

        nanMask = np.isnan(y)
        if not np.any(nanMask):
            return y

        n = len(y)
        nanIdx = np.where(nanMask)[0]

        for t in nanIdx:
            seasonPos = t % period
            sameSeasonIdx = np.arange(seasonPos, n, period)
            sameSeasonValues = y[sameSeasonIdx]
            validValues = sameSeasonValues[~np.isnan(sameSeasonValues)]

            if len(validValues) > 0:
                y[t] = np.mean(validValues)

        return y

    def locf(self, y: np.ndarray) -> np.ndarray:
        """Last Observation Carried Forward"""
        y = np.asarray(y, dtype=np.float64).copy()

        if len(y) == 0:
            return y

        if np.all(np.isnan(y)):
            return np.zeros_like(y)

        lastValid = np.nan
        for i in range(len(y)):
            if np.isnan(y[i]):
                if np.isfinite(lastValid):
                    y[i] = lastValid
            else:
                lastValid = y[i]

        return y

    def nocb(self, y: np.ndarray) -> np.ndarray:
        """Next Observation Carried Backward"""
        y = np.asarray(y, dtype=np.float64).copy()

        if len(y) == 0:
            return y

        if np.all(np.isnan(y)):
            return np.zeros_like(y)

        nextValid = np.nan
        for i in range(len(y) - 1, -1, -1):
            if np.isnan(y[i]):
                if np.isfinite(nextValid):
                    y[i] = nextValid
            else:
                nextValid = y[i]

        return y

    def detectMissing(self, y: np.ndarray) -> Dict[str, Any]:
        """결측 패턴 분석"""
        y = np.asarray(y, dtype=np.float64)

        n = len(y)
        if n == 0:
            return {
                'nMissing': 0,
                'missingRatio': 0.0,
                'maxConsecutiveMissing': 0,
                'pattern': 'empty',
            }

        nanMask = np.isnan(y)
        nMissing = int(np.sum(nanMask))
        missingRatio = nMissing / n

        if nMissing == 0:
            return {
                'nMissing': 0,
                'missingRatio': 0.0,
                'maxConsecutiveMissing': 0,
                'pattern': 'complete',
            }

        maxConsecutive = 0
        current = 0
        for val in nanMask:
            if val:
                current += 1
                if current > maxConsecutive:
                    maxConsecutive = current
            else:
                current = 0

        if nMissing == n:
            pattern = 'all_missing'
        elif missingRatio > 0.5:
            pattern = 'mostly_missing'
        elif maxConsecutive > n * 0.2:
            pattern = 'block_missing'
        else:
            pattern = 'scattered'

        return {
            'nMissing': nMissing,
            'missingRatio': missingRatio,
            'maxConsecutiveMissing': maxConsecutive,
            'pattern': pattern,
        }
