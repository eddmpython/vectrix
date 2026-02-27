"""
시계열 분해 자체 구현

- Classical Decomposition (Additive/Multiplicative)
- STL-like Decomposition (LOESS 기반)
- 다중 계절성 분해 (MSTL-like)

모두 numpy + numba로 직접 구현
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
    """분해 결과"""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    observed: np.ndarray


class SeasonalDecomposition:
    """
    계절 분해 자체 구현

    Classical과 STL-like 방법 모두 지원
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
            계절 주기
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
        시계열 분해

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터

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
        """고전적 분해"""
        n = len(y)
        m = self.period

        # 1. 추세 추출 (중심 이동평균)
        trend = self._computeTrend(y, m)

        # 2. 추세 제거
        if self.model == 'additive':
            detrended = y - trend
        else:  # multiplicative
            detrended = y / (trend + 1e-10)

        # 3. 계절 성분 추출
        seasonal = self._computeSeasonal(detrended, m)

        # 4. 잔차
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
        STL-like 분해 (간소화)

        실제 STL은 반복적 LOESS를 사용하지만,
        여기서는 간소화된 버전 구현
        """
        n = len(y)
        m = self.period

        # 반복 분해
        trend = np.zeros(n)
        seasonal = np.zeros(n)

        # 초기 추세
        trend = self._loessSmooth(y, span=max(7, m))

        for iteration in range(3):  # 3회 반복
            # 추세 제거
            detrended = y - trend

            # 계절 성분 (서브시리즈 평활)
            seasonal = self._computeSubseriesSeasonal(detrended, m)

            # 계절 조정
            seasonAdjusted = y - seasonal

            # 추세 재추정
            trend = self._loessSmooth(seasonAdjusted, span=max(7, m))

        # 잔차
        residual = y - trend - seasonal

        return DecompositionResult(
            trend=trend,
            seasonal=seasonal,
            residual=residual,
            observed=y
        )

    def _computeTrend(self, y: np.ndarray, m: int) -> np.ndarray:
        """중심 이동평균으로 추세 추출"""
        n = len(y)
        trend = np.full(n, np.nan)

        # 홀수면 단순 이동평균
        if m % 2 == 1:
            halfM = m // 2
            for t in range(halfM, n - halfM):
                trend[t] = np.mean(y[t - halfM:t + halfM + 1])
        else:
            # 짝수면 2xm MA
            halfM = m // 2
            for t in range(halfM, n - halfM):
                # 2xm 이동평균
                if t - halfM >= 0 and t + halfM < n:
                    ma = np.mean(y[t - halfM:t + halfM])
                    if t + halfM + 1 < n:
                        ma2 = np.mean(y[t - halfM + 1:t + halfM + 1])
                        trend[t] = (ma + ma2) / 2
                    else:
                        trend[t] = ma

        # NaN 보간
        trend = self._interpolateNaN(trend, y)

        return trend

    def _computeSeasonal(self, detrended: np.ndarray, m: int) -> np.ndarray:
        """계절 성분 추출"""
        n = len(detrended)
        seasonal = np.zeros(n)

        # 각 계절 위치별 평균
        seasonalMeans = np.zeros(m)
        for i in range(m):
            vals = detrended[i::m]
            validVals = vals[~np.isnan(vals)]
            if len(validVals) > 0:
                seasonalMeans[i] = np.mean(validVals)

        # 평균 제거 (합이 0이 되도록)
        seasonalMeans = seasonalMeans - np.mean(seasonalMeans)

        # 전체 길이로 확장
        for t in range(n):
            seasonal[t] = seasonalMeans[t % m]

        return seasonal

    def _computeSubseriesSeasonal(self, detrended: np.ndarray, m: int) -> np.ndarray:
        """서브시리즈 평활로 계절 성분 추출 (STL 스타일)"""
        n = len(detrended)
        seasonal = np.zeros(n)

        # 각 계절 위치별 평활
        for i in range(m):
            subseries = detrended[i::m]
            if len(subseries) >= 3:
                smoothed = self._loessSmooth(subseries, span=max(3, len(subseries) // 2))
            else:
                smoothed = subseries

            # 원래 위치에 배치
            for j, val in enumerate(smoothed):
                idx = i + j * m
                if idx < n:
                    seasonal[idx] = val

        # 평균 제거
        seasonalMean = np.zeros(m)
        for i in range(m):
            seasonalMean[i] = np.mean(seasonal[i::m])
        overallMean = np.mean(seasonalMean)

        for t in range(n):
            seasonal[t] -= overallMean

        return seasonal

    def _loessSmooth(self, y: np.ndarray, span: int = 7) -> np.ndarray:
        """
        간소화된 LOESS 평활

        실제 LOESS는 가중 회귀를 사용하지만,
        여기서는 가중 이동평균으로 근사
        """
        n = len(y)
        smoothed = np.zeros(n)

        halfSpan = span // 2

        for i in range(n):
            start = max(0, i - halfSpan)
            end = min(n, i + halfSpan + 1)

            # 삼각 가중치
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
        """NaN 선형 보간"""
        result = y.copy()
        n = len(y)

        # 앞쪽 NaN
        for i in range(n):
            if np.isnan(result[i]):
                result[i] = original[i]
            else:
                break

        # 뒤쪽 NaN
        for i in range(n - 1, -1, -1):
            if np.isnan(result[i]):
                result[i] = original[i]
            else:
                break

        # 중간 NaN 선형 보간
        for i in range(n):
            if np.isnan(result[i]):
                # 이전 유효값 찾기
                prevIdx = i - 1
                while prevIdx >= 0 and np.isnan(result[prevIdx]):
                    prevIdx -= 1

                # 다음 유효값 찾기
                nextIdx = i + 1
                while nextIdx < n and np.isnan(result[nextIdx]):
                    nextIdx += 1

                if prevIdx >= 0 and nextIdx < n:
                    # 선형 보간
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
        """계절 성분만 추출"""
        result = self.decompose(y)
        return result.seasonal

    def extractTrend(self, y: np.ndarray) -> np.ndarray:
        """추세 성분만 추출"""
        result = self.decompose(y)
        return result.trend

    def deseasonalize(self, y: np.ndarray) -> np.ndarray:
        """계절 조정된 시계열"""
        result = self.decompose(y)
        if self.model == 'additive':
            return y - result.seasonal
        else:
            return y / (result.seasonal + 1e-10)


class MSTLDecomposition:
    """
    다중 계절성 분해 (MSTL-like)

    여러 계절 주기를 순차적으로 분해
    """

    def __init__(self, periods: List[int], model: str = 'additive'):
        """
        Parameters
        ----------
        periods : List[int]
            계절 주기 목록 (예: [7, 30, 365])
        model : str
            'additive' or 'multiplicative'
        """
        self.periods = sorted(periods)
        self.model = model

    def decompose(self, y: np.ndarray) -> dict:
        """
        다중 계절 분해

        Returns
        -------
        dict
            {'trend': array, 'seasonals': {period: array}, 'residual': array}
        """
        n = len(y)
        remaining = y.copy()
        seasonals = {}

        # 각 계절 주기에 대해 순차적으로 분해
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

            # 계절 성분 제거
            if self.model == 'additive':
                remaining = remaining - result.seasonal
            else:
                remaining = remaining / (result.seasonal + 1e-10)

        # 남은 부분에서 추세 추출
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
        다중 계절 분해 기반 예측
        """
        decomposition = self.decompose(y)
        n = len(y)

        # 추세 예측 (선형 외삽)
        x = np.arange(n, dtype=np.float64)
        slope, intercept = TurboCore.linearRegression(x, decomposition['trend'])

        predictions = np.zeros(steps)

        for h in range(steps):
            t = n + h

            # 추세
            pred = intercept + slope * t

            # 계절성 추가
            for period, seasonal in decomposition['seasonals'].items():
                seasonIdx = (n + h) % period
                if seasonIdx < len(seasonal):
                    if self.model == 'additive':
                        pred += seasonal[seasonIdx]
                    else:
                        pred *= seasonal[seasonIdx]

            predictions[h] = pred

        return predictions
