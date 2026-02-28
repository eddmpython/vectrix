"""
Theta 모델 자체 구현

M3 Competition 우승 모델
시계열을 Theta lines로 분해하여 예측

참조: Assimakopoulos & Nikolopoulos (2000)
"""

from typing import Tuple

import numpy as np

try:
    from vectrix_core import ses_sse as _sesSSERust
    from vectrix_core import ses_filter as _sesFilterRust
    from vectrix_core import theta_decompose as _thetaDecomposeRust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

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


@jit(nopython=True, cache=True)
def _sesSSEJIT(y: np.ndarray, alpha: float) -> float:
    n = len(y)
    level = y[0]
    sse = 0.0

    for t in range(1, n):
        error = y[t] - level
        sse += error * error
        level = alpha * y[t] + (1.0 - alpha) * level

    return sse


@jit(nopython=True, cache=True)
def _sesFilterJIT(y: np.ndarray, alpha: float) -> np.ndarray:
    n = len(y)
    result = np.zeros(n)
    result[0] = y[0]

    for t in range(1, n):
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1]

    return result


class ThetaModel:
    """
    자체 구현 Theta 모델

    Theta 방법:
    1. 시계열을 theta=0 (선형 추세)와 theta=2 (곡률 2배) 두 라인으로 분해
    2. theta=0는 선형회귀, theta=2는 SES로 예측
    3. 두 예측을 결합
    """

    def __init__(self, theta: float = 2.0, period: int = 1):
        """
        Parameters
        ----------
        theta : float
            Theta 파라미터 (기본 2.0)
        period : int
            계절 주기 (계절 조정용)
        """
        self.theta = theta
        self.period = period

        # 학습 결과
        self.slope = 0.0
        self.intercept = 0.0
        self.alpha = 0.3
        self.lastLevel = 0.0

        # 계절 성분
        self.seasonal = None
        self.deseasonalized = None

        self.fitted = False
        self.residuals = None

    def fit(self, y: np.ndarray) -> 'ThetaModel':
        """
        모델 학습

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        """
        n = len(y)

        if n < 5:
            self._simpleFit(y)
            self.fitted = True
            return self

        # 계절 조정 (period > 1인 경우)
        if self.period > 1 and n >= self.period * 2:
            self.deseasonalized, self.seasonal = self._deseasonalize(y)
            workData = self.deseasonalized
        else:
            workData = y
            self.seasonal = None

        # Theta line 분해
        # theta=0: 선형 추세
        x = np.arange(n, dtype=np.float64)
        self.slope, self.intercept = TurboCore.linearRegression(x, workData)

        # theta=2: 곡률 강조
        thetaLine = self._computeThetaLine(workData, self.theta)

        # SES로 theta line 예측
        self.alpha = self._optimizeAlpha(thetaLine)
        self.lastLevel = self._sesFilter(thetaLine, self.alpha)[-1]

        # 잔차
        fitted = self._computeFitted(workData, n)
        self.residuals = workData - fitted

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        예측

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
            raise ValueError("모델이 학습되지 않았습니다.")

        n = len(self.deseasonalized) if self.deseasonalized is not None else 0
        if n == 0:
            n = 10  # 폴백

        predictions = np.zeros(steps)

        for h in range(1, steps + 1):
            # Theta=0 (선형 추세) 예측
            t = n + h - 1
            trendPred = self.intercept + self.slope * t

            # Theta=2 (SES) 예측
            sesPred = self.lastLevel  # SES는 수평

            # 결합 (동일 가중치)
            pred = (trendPred + sesPred) / 2

            predictions[h - 1] = pred

        # 계절성 복원
        if self.seasonal is not None:
            for h in range(steps):
                seasonIdx = h % self.period
                predictions[h] += self.seasonal[seasonIdx]

        # 신뢰구간
        if self.residuals is not None:
            sigma = np.std(self.residuals)
        else:
            sigma = np.std(predictions) * 0.1

        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower95 = predictions - margin
        upper95 = predictions + margin

        return predictions, lower95, upper95

    def _deseasonalize(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """고전적 계절 분해"""
        n = len(y)
        m = self.period

        # 계절 평균 계산
        seasonal = np.zeros(m)
        for i in range(m):
            vals = y[i::m]
            seasonal[i] = np.mean(vals) - np.mean(y)

        # 계절 조정
        deseasonalized = np.zeros(n)
        for t in range(n):
            deseasonalized[t] = y[t] - seasonal[t % m]

        return deseasonalized, seasonal

    def _computeThetaLine(self, y: np.ndarray, theta: float) -> np.ndarray:
        """
        Theta line 계산

        Z_theta(t) = theta * y(t) + (1 - theta) * L(t)
        여기서 L(t)는 선형 추세
        """
        n = len(y)
        x = np.arange(n, dtype=np.float64)

        # 선형 추세
        linearTrend = self.intercept + self.slope * x

        # Theta line
        thetaLine = theta * y + (1 - theta) * linearTrend

        return thetaLine

    def _optimizeAlpha(self, y: np.ndarray) -> float:
        """SES 알파 최적화"""
        bestAlpha = 0.3
        bestSSE = np.inf
        yF64 = y.astype(np.float64, copy=False)

        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            if RUST_AVAILABLE:
                sse = _sesSSERust(yF64, alpha)
            else:
                sse = _sesSSEJIT(y, alpha)
            if sse < bestSSE:
                bestSSE = sse
                bestAlpha = alpha

        return bestAlpha

    def _sesSSE(self, y: np.ndarray, alpha: float) -> float:
        """SES SSE 계산"""
        if RUST_AVAILABLE:
            return _sesSSERust(y.astype(np.float64, copy=False), alpha)
        return _sesSSEJIT(y, alpha)

    def _sesFilter(self, y: np.ndarray, alpha: float) -> np.ndarray:
        """SES 필터링"""
        if RUST_AVAILABLE:
            return _sesFilterRust(y.astype(np.float64, copy=False), alpha)
        return _sesFilterJIT(y, alpha)

    def _computeFitted(self, y: np.ndarray, n: int) -> np.ndarray:
        """적합값 계산"""
        x = np.arange(n, dtype=np.float64)

        # 선형 추세
        trendFitted = self.intercept + self.slope * x

        # SES 적합값
        thetaLine = self._computeThetaLine(y, self.theta)
        sesFitted = self._sesFilter(thetaLine, self.alpha)

        # 결합
        fitted = (trendFitted + sesFitted) / 2

        return fitted

    def _simpleFit(self, y: np.ndarray):
        """간단한 학습"""
        self.slope = 0.0
        self.intercept = np.mean(y)
        self.lastLevel = y[-1] if len(y) > 0 else 0
        self.deseasonalized = y
        self.residuals = np.zeros(len(y))


class OptimizedTheta:
    """
    최적화된 Theta 모델 (OTM)

    여러 theta 값을 시도하여 최적 선택
    """

    def __init__(self, period: int = 1):
        self.period = period
        self.bestTheta = 2.0
        self.bestModel = None

    def fit(self, y: np.ndarray) -> ThetaModel:
        """최적 theta 선택 (3 후보로 축소 + 최적 모델 전체 데이터 재학습)"""
        n = len(y)
        trainSize = int(n * 0.8)

        if trainSize < 10:
            self.bestModel = ThetaModel(theta=2.0, period=self.period)
            self.bestModel.fit(y)
            return self.bestModel

        trainData = y[:trainSize]
        testData = y[trainSize:]
        testSteps = len(testData)

        bestMAPE = np.inf

        # 6→3 후보로 축소 (핵심 theta 값만)
        for theta in [1.0, 2.0, 3.0]:
            try:
                model = ThetaModel(theta=theta, period=self.period)
                model.fit(trainData)
                pred, _, _ = model.predict(testSteps)

                mape = TurboCore.mape(testData, pred[:len(testData)])

                if mape < bestMAPE:
                    bestMAPE = mape
                    self.bestTheta = theta

            except Exception:
                continue

        # 전체 데이터로 재학습
        self.bestModel = ThetaModel(theta=self.bestTheta, period=self.period)
        self.bestModel.fit(y)

        return self.bestModel

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측"""
        if self.bestModel is None:
            raise ValueError("모델이 학습되지 않았습니다.")
        return self.bestModel.predict(steps)
