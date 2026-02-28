"""
Dynamic Time Scan Forecaster (DTSF)

비모수 패턴 매칭 예측 모델.
과거에서 현재와 가장 유사한 패턴을 찾아 그 후속값으로 예측.

E031 실험 결과 → E041 스트레스 테스트 통과 → 엔진 통합.
- 승률 38.2% (1위): 기존 엔진 대비 가장 높은 승률
- 잔차 상관 0.1~0.5: 기존 모델과 근본적으로 다른 예측 원리
- n >= 30 조건 (패턴 매칭에 충분한 이력 필요)
"""

from typing import Tuple

import numpy as np
from scipy.signal import periodogram


class DynamicTimeScanForecaster:
    """
    Dynamic Time Scan Forecaster

    1. 현재 시점의 윈도우(query)와 과거 모든 윈도우 간 정규화된 유클리드 거리 계산
    2. 가장 유사한 K개 패턴의 후속값 중앙값 = 예측
    3. 시간 가중(timeDecay)으로 최근 패턴에 더 높은 가중치
    """

    def __init__(self, windowSize=None, nNeighbors=5, normalize=True, timeDecay=0.001, computeResiduals=True):
        self.windowSize = windowSize
        self.nNeighbors = nNeighbors
        self.normalize = normalize
        self.timeDecay = timeDecay
        self._computeResiduals = computeResiduals

        self._y = None
        self._detectedPeriod = None
        self._residStd = 1.0

        self.fitted = False
        self.residuals = None

    def fit(self, y: np.ndarray) -> 'DynamicTimeScanForecaster':
        self._y = np.asarray(y, dtype=np.float64).copy()
        n = len(self._y)

        if self.windowSize is None:
            self._detectedPeriod = self._detectPeriod(self._y)
            self.windowSize = self._detectedPeriod

        self._residStd = max(np.std(np.diff(self._y)), 1e-8)

        if self._computeResiduals:
            W = min(self.windowSize, n // 3)
            W = max(W, 2)

            if n >= W + 2:
                residuals = np.zeros(n)
                for t in range(W, n):
                    query = self._y[t - W:t]
                    if self.normalize:
                        qMean = np.mean(query)
                        qStd = max(np.std(query), 1e-8)
                        queryNorm = (query - qMean) / qStd
                    else:
                        queryNorm = query

                    maxStart = t - W
                    if maxStart < 1:
                        continue

                    distances = np.zeros(maxStart)
                    for i in range(maxStart):
                        window = self._y[i:i + W]
                        if self.normalize:
                            wMean = np.mean(window)
                            wStd = max(np.std(window), 1e-8)
                            windowNorm = (window - wMean) / wStd
                        else:
                            windowNorm = window
                        distances[i] = np.sqrt(np.mean((queryNorm - windowNorm) ** 2))

                    K = min(self.nNeighbors, maxStart)
                    if K >= len(distances):
                        neighborIdx = np.arange(len(distances))
                    else:
                        neighborIdx = np.argpartition(distances, K)[:K]

                    preds = []
                    for idx in neighborIdx:
                        nextIdx = idx + W
                        if nextIdx < n:
                            preds.append(self._y[nextIdx])
                    if preds:
                        residuals[t] = self._y[t] - np.median(preds)

                self.residuals = residuals
            else:
                self.residuals = np.zeros(n)
        else:
            self.residuals = np.zeros(n)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("fit() must be called before predict()")

        y = self._y
        n = len(y)
        W = min(self.windowSize, n // 3)
        W = max(W, 2)

        if n < W + steps + 1:
            pred = np.full(steps, np.mean(y))
            sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
            return pred, pred - 1.96 * sigma, pred + 1.96 * sigma

        if self.normalize:
            qMean = np.mean(y[-W:])
            qStd = max(np.std(y[-W:]), 1e-8)
            queryNorm = (y[-W:] - qMean) / qStd
        else:
            queryNorm = y[-W:]

        maxStart = n - W - steps
        if maxStart < 1:
            pred = np.full(steps, np.mean(y))
            sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
            return pred, pred - 1.96 * sigma, pred + 1.96 * sigma

        distances = np.zeros(maxStart)
        for i in range(maxStart):
            window = y[i:i + W]
            if self.normalize:
                wMean = np.mean(window)
                wStd = max(np.std(window), 1e-8)
                windowNorm = (window - wMean) / wStd
            else:
                windowNorm = window
            shapeDist = np.sqrt(np.mean((queryNorm - windowNorm) ** 2))
            timeWeight = np.exp(-self.timeDecay * (n - i))
            distances[i] = shapeDist / max(timeWeight, 1e-10)

        K = min(self.nNeighbors, maxStart)
        if K >= len(distances):
            neighborIdx = np.arange(len(distances))
        else:
            neighborIdx = np.argpartition(distances, K)[:K]

        futures = np.zeros((K, steps))
        weights = np.zeros(K)
        for j, idx in enumerate(neighborIdx):
            segment = y[idx + W: idx + W + steps]
            futures[j, :len(segment)] = segment
            if len(segment) < steps:
                futures[j, len(segment):] = segment[-1] if len(segment) > 0 else np.mean(y)
            weights[j] = 1.0 / max(distances[idx], 1e-10)

        weights /= weights.sum()
        predictions = np.average(futures, axis=0, weights=weights)

        pctLow = np.percentile(futures, 10, axis=0)
        pctHigh = np.percentile(futures, 90, axis=0)
        sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
        lower = np.minimum(pctLow, predictions - 1.96 * sigma)
        upper = np.maximum(pctHigh, predictions + 1.96 * sigma)

        return predictions, lower, upper

    def _detectPeriod(self, y):
        n = len(y)
        if n < 10:
            return 7

        detrended = y - np.linspace(y[0], y[-1], n)
        freqs, power = periodogram(detrended)

        if len(freqs) < 3:
            return 7

        validMask = freqs > 0
        freqs = freqs[validMask]
        power = power[validMask]

        if len(freqs) == 0:
            return 7

        peakIdx = np.argmax(power)
        dominantFreq = freqs[peakIdx]

        if dominantFreq > 0:
            period = int(round(1.0 / dominantFreq))
            period = max(2, min(period, n // 4))
            return period

        return 7
