"""
로지스틱 성장 모델 (Saturating Growth)

Prophet의 핵심 기능인 포화 성장 모델을 순수 numpy/scipy로 구현
- LogisticGrowthModel: 기본 로지스틱 곡선 피팅 및 예측
- SaturatingTrendModel: 포화 추세 + 계절성 결합

참조: Taylor & Letham (2018) "Forecasting at Scale"
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


def _logisticCurve(x, cap, floor, k, m):
    """로지스틱 함수: floor + (cap - floor) / (1 + exp(-k*(x - m)))"""
    return floor + (cap - floor) / (1.0 + np.exp(-k * (x - m)))


class LogisticGrowthModel:
    """로지스틱 성장 모델 (포화 성장)"""

    def __init__(self, cap: Optional[float] = None, floor: float = 0.0):
        """
        Parameters
        ----------
        cap : float or None
            포화 상한. None이면 자동 추정 (max(y) * 1.2)
        floor : float
            포화 하한
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
        로지스틱 곡선 피팅

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터

        Returns
        -------
        LogisticGrowthModel
            학습된 모델
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
        예측 + 신뢰구간

        Parameters
        ----------
        steps : int
            예측 기간

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (예측값, lower95, upper95)
        """
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

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
        """데이터 부족 시 단순 피팅"""
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
    """포화 추세 + 계절성 결합 모델"""

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
            포화 상한. None이면 자동 추정
        floor : float
            포화 하한
        period : int
            계절 주기 (1이면 계절성 없음)
        """
        self.cap = cap
        self.floor = floor
        self.period = period

        self.logisticModel = None
        self.seasonal = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'SaturatingTrendModel':
        """
        비계절화 -> 로지스틱 피팅 -> 계절성 저장

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터

        Returns
        -------
        SaturatingTrendModel
            학습된 모델
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
        예측 with 계절성 복원

        Parameters
        ----------
        steps : int
            예측 기간

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (예측값, lower95, upper95)
        """
        if not self.fitted:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

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
        고전적 계절 분해 (가법 모델)

        Parameters
        ----------
        y : np.ndarray
            원본 시계열

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (비계절화된 시계열, 계절 지수 배열)
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
