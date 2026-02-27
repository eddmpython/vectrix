"""
Time Series Feature Engineering

시계열 데이터에서 회귀 모델용 피처를 자동 생성:
- Lag features: y_{t-1}, y_{t-2}, ...
- Rolling features: rolling mean, rolling std
- Calendar features: day-of-week, month, etc.
- Fourier features: sin/cos terms for seasonality
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


class LagFeatures:
    """Lag feature 생성기"""

    def __init__(self, lags: Optional[List[int]] = None, maxLag: int = 7):
        self.lags = lags or list(range(1, maxLag + 1))

    def transform(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (features [n-maxLag, nLags], target [n-maxLag])
        """
        n = len(y)
        maxLag = max(self.lags)
        nSamples = n - maxLag
        nFeatures = len(self.lags)

        if nSamples <= 0:
            return np.empty((0, nFeatures)), np.empty(0)

        X = np.zeros((nSamples, nFeatures))
        for j, lag in enumerate(self.lags):
            for i in range(nSamples):
                X[i, j] = y[maxLag + i - lag]

        target = y[maxLag:]
        return X, target

    def transformLast(self, y: np.ndarray) -> np.ndarray:
        """마지막 시점의 lag features"""
        features = np.zeros(len(self.lags))
        n = len(y)
        for j, lag in enumerate(self.lags):
            if lag <= n:
                features[j] = y[-lag]
        return features


class RollingFeatures:
    """Rolling statistics feature 생성기"""

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [7, 14, 30]

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            [n, nWindows * 2]  (rolling mean + rolling std per window)
        """
        n = len(y)
        nFeatures = len(self.windows) * 2
        X = np.zeros((n, nFeatures))

        for j, w in enumerate(self.windows):
            for i in range(n):
                start = max(0, i - w + 1)
                window = y[start:i + 1]
                X[i, j * 2] = np.mean(window)
                X[i, j * 2 + 1] = np.std(window) if len(window) > 1 else 0.0

        return X

    def transformLast(self, y: np.ndarray) -> np.ndarray:
        """마지막 시점의 rolling features"""
        features = np.zeros(len(self.windows) * 2)
        n = len(y)
        for j, w in enumerate(self.windows):
            window = y[-min(w, n):]
            features[j * 2] = np.mean(window)
            features[j * 2 + 1] = np.std(window) if len(window) > 1 else 0.0
        return features


class CalendarFeatures:
    """Calendar feature 생성기 (index 기반)"""

    def __init__(self, period: int = 7):
        self.period = period

    def transform(self, n: int, startIdx: int = 0) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            [n, period]  one-hot encoded day-of-period
        """
        X = np.zeros((n, self.period))
        for i in range(n):
            idx = (startIdx + i) % self.period
            X[i, idx] = 1.0
        return X

    def transformSingle(self, idx: int) -> np.ndarray:
        """단일 시점의 calendar features"""
        features = np.zeros(self.period)
        features[idx % self.period] = 1.0
        return features


class FourierFeatures:
    """Fourier term feature 생성기"""

    def __init__(self, period: int = 7, nTerms: int = 3):
        self.period = period
        self.nTerms = min(nTerms, period // 2)

    def transform(self, n: int, startIdx: int = 0) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            [n, nTerms * 2]  (sin + cos per term)
        """
        X = np.zeros((n, self.nTerms * 2))
        for i in range(n):
            t = startIdx + i
            for k in range(self.nTerms):
                freq = 2 * np.pi * (k + 1) / self.period
                X[i, k * 2] = np.sin(freq * t)
                X[i, k * 2 + 1] = np.cos(freq * t)
        return X

    def transformSingle(self, idx: int) -> np.ndarray:
        """단일 시점의 fourier features"""
        features = np.zeros(self.nTerms * 2)
        for k in range(self.nTerms):
            freq = 2 * np.pi * (k + 1) / self.period
            features[k * 2] = np.sin(freq * idx)
            features[k * 2 + 1] = np.cos(freq * idx)
        return features


def autoFeatureEngineering(
    y: np.ndarray,
    period: int = 7,
    maxLag: int = 14,
    rollingWindows: Optional[List[int]] = None,
    useFourier: bool = True,
    useCalendar: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    자동 피처 엔지니어링

    Parameters
    ----------
    y : np.ndarray
        시계열 데이터
    period : int
        계절 주기
    maxLag : int
        최대 lag
    rollingWindows : List[int], optional
        롤링 윈도우 크기들
    useFourier : bool
        Fourier 항 사용 여부
    useCalendar : bool
        Calendar 항 사용 여부

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Dict]
        (X, y_target, metadata)
    """
    n = len(y)
    if rollingWindows is None:
        rollingWindows = [min(7, n // 4), min(14, n // 3)]
        rollingWindows = [w for w in rollingWindows if w >= 2]
        if not rollingWindows:
            rollingWindows = [2]

    lagGen = LagFeatures(maxLag=min(maxLag, n // 3))
    lagX, target = lagGen.transform(y)
    offset = max(lagGen.lags)
    nSamples = len(target)

    if nSamples == 0:
        return np.empty((0, 1)), np.empty(0), {'featureNames': [], 'offset': 0}

    featureSets = [lagX]
    featureNames = [f'lag_{l}' for l in lagGen.lags]

    rollGen = RollingFeatures(windows=rollingWindows)
    rollX = rollGen.transform(y)
    featureSets.append(rollX[offset:offset + nSamples])
    for w in rollingWindows:
        featureNames.extend([f'roll_mean_{w}', f'roll_std_{w}'])

    if useFourier and period > 1:
        nTerms = min(3, period // 2)
        fourierGen = FourierFeatures(period=period, nTerms=nTerms)
        fourierX = fourierGen.transform(nSamples, startIdx=offset)
        featureSets.append(fourierX)
        for k in range(nTerms):
            featureNames.extend([f'fourier_sin_{k+1}', f'fourier_cos_{k+1}'])

    if useCalendar and period > 1:
        calGen = CalendarFeatures(period=period)
        calX = calGen.transform(nSamples, startIdx=offset)
        featureSets.append(calX)
        for p in range(period):
            featureNames.append(f'cal_{p}')

    X = np.hstack(featureSets)

    metadata = {
        'featureNames': featureNames,
        'offset': offset,
        'lagGen': lagGen,
        'rollGen': rollGen,
        'period': period,
        'useFourier': useFourier,
        'useCalendar': useCalendar,
        'rollingWindows': rollingWindows,
    }

    return X, target, metadata
