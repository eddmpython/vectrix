"""
TurboCore: Numba 기반 고속 연산 코어

시계열 분석에 필요한 핵심 연산을 Numba JIT으로 최적화
- ACF/PACF 계산
- FFT 기반 계절성 탐지
- 롤링 통계량
- 차분/적분
- 평가 지표 (MAPE, RMSE, MAE)
"""

import numpy as np
from typing import Tuple, Optional

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
    """Numba 기반 고속 연산 코어"""

    @staticmethod
    @jit(nopython=True, cache=True)
    def acf(x: np.ndarray, maxLag: int) -> np.ndarray:
        """
        자기상관함수 (ACF) 계산

        Parameters
        ----------
        x : np.ndarray
            시계열 데이터
        maxLag : int
            최대 래그

        Returns
        -------
        np.ndarray
            ACF 값 (lag 0 ~ maxLag)
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
        편자기상관함수 (PACF) 계산 - Durbin-Levinson 알고리즘

        Parameters
        ----------
        x : np.ndarray
            시계열 데이터
        maxLag : int
            최대 래그

        Returns
        -------
        np.ndarray
            PACF 값
        """
        n = len(x)
        acfVals = TurboCore.acf(x, maxLag)

        pacf = np.zeros(maxLag + 1)
        pacf[0] = 1.0

        if maxLag < 1:
            return pacf

        # Durbin-Levinson 재귀
        phi = np.zeros((maxLag + 1, maxLag + 1))
        phi[1, 1] = acfVals[1]
        pacf[1] = acfVals[1]

        for k in range(2, maxLag + 1):
            # phi[k,k] 계산
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

            # phi[k, 1:k-1] 업데이트
            for j in range(1, k):
                phi[k, j] = phi[k-1, j] - phi[k, k] * phi[k-1, k - j]

        return pacf

    @staticmethod
    @jit(nopython=True, cache=True)
    def diff(x: np.ndarray, d: int = 1) -> np.ndarray:
        """
        차분

        Parameters
        ----------
        x : np.ndarray
            시계열 데이터
        d : int
            차분 차수

        Returns
        -------
        np.ndarray
            차분된 데이터
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
        적분 (차분의 역연산)

        Parameters
        ----------
        x : np.ndarray
            차분된 데이터
        initial : float
            초기값
        d : int
            적분 차수

        Returns
        -------
        np.ndarray
            복원된 데이터
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
        """계절 차분"""
        n = len(x)
        result = np.zeros(n - period)
        for i in range(n - period):
            result[i] = x[i + period] - x[i]
        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def rollingMean(x: np.ndarray, window: int) -> np.ndarray:
        """롤링 평균"""
        n = len(x)
        result = np.zeros(n)

        # 초기 윈도우
        cumSum = 0.0
        for i in range(min(window, n)):
            cumSum += x[i]
            result[i] = cumSum / (i + 1)

        # 롤링
        for i in range(window, n):
            cumSum += x[i] - x[i - window]
            result[i] = cumSum / window

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def rollingStd(x: np.ndarray, window: int) -> np.ndarray:
        """롤링 표준편차 — O(n) Welford 온라인 알고리즘"""
        n = len(x)
        result = np.zeros(n)

        # 초기 윈도우 누적
        sumX = 0.0
        sumX2 = 0.0

        for i in range(n):
            sumX += x[i]
            sumX2 += x[i] * x[i]

            if i >= window:
                # 윈도우 밖 값 제거
                sumX -= x[i - window]
                sumX2 -= x[i - window] * x[i - window]

            count = min(i + 1, window)
            if count > 1:
                mean = sumX / count
                variance = sumX2 / count - mean * mean
                # 수치 안정성
                if variance > 0.0:
                    result[i] = np.sqrt(variance)

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def ewma(x: np.ndarray, alpha: float) -> np.ndarray:
        """지수가중이동평균 (EWMA)"""
        n = len(x)
        result = np.zeros(n)
        result[0] = x[0]

        for i in range(1, n):
            result[i] = alpha * x[i] + (1 - alpha) * result[i - 1]

        return result

    @staticmethod
    @jit(nopython=True, cache=True)
    def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """MAPE 계산"""
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
        """RMSE 계산"""
        n = len(actual)
        total = 0.0

        for i in range(n):
            total += (actual[i] - predicted[i]) ** 2

        return np.sqrt(total / n)

    @staticmethod
    @jit(nopython=True, cache=True)
    def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
        """MAE 계산"""
        n = len(actual)
        total = 0.0

        for i in range(n):
            total += abs(actual[i] - predicted[i])

        return total / n

    @staticmethod
    @jit(nopython=True, cache=True)
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """SMAPE 계산"""
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
        FFT 기반 계절 주기 탐지

        Parameters
        ----------
        x : np.ndarray
            시계열 데이터
        topK : int
            상위 K개 주기 반환

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (주기 배열, 강도 배열)
        """
        n = len(x)

        if n < 10:
            return np.array([7]), np.array([0.0])

        # 추세 제거
        detrended = x - np.linspace(x[0], x[-1], n)

        # FFT
        fft = np.fft.fft(detrended)
        magnitudes = np.abs(fft)
        magnitudes[0] = 0  # DC 성분 제거
        magnitudes[n // 2:] = 0  # 나이퀴스트 이상 제거

        # 상위 K개 주파수
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
        ADF 검정 통계량 계산 (간소화)

        Returns
        -------
        float
            ADF 통계량 (음수일수록 정상성)
        """
        n = len(x)

        if n < maxLag + 2:
            return 0.0

        # 1차 차분
        dx = np.zeros(n - 1)
        for i in range(n - 1):
            dx[i] = x[i + 1] - x[i]

        # 래그 선택 (AIC)
        lag = min(maxLag, int((n - 1) ** (1/3)))

        # 회귀: dx_t = alpha + beta * x_{t-1} + sum(gamma_i * dx_{t-i})
        # 간소화: beta만 추정

        # x_{t-1}과 dx_t의 상관
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

        # 표준오차 추정
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
        단순 선형회귀

        Returns
        -------
        Tuple[float, float]
            (기울기, 절편)
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


# Numba 사용 가능 여부 확인
def isNumbaAvailable() -> bool:
    return NUMBA_AVAILABLE
