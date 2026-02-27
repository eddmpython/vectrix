"""
Croston's Method for Intermittent Demand Forecasting

간헐적 수요 데이터 (0이 많은 시계열)를 위한 전문 모델.

Variants:
- Classic: Croston (1972) — 수요크기와 수요간격을 별도 SES
- SBA: Syntetos-Boylan Approximation — 편향 보정
- TSB: Teunter-Syntetos-Babai — 수요 확률 직접 추정
"""

import numpy as np
from typing import Tuple


class CrostonClassic:
    """
    Croston's Classic Method

    비영 수요 발생 시:
    - z_t: 수요 크기 SES 업데이트
    - p_t: 수요 간격 SES 업데이트
    예측: z_t / p_t
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.z = 0.0
        self.p = 0.0
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'CrostonClassic':
        n = len(y)
        demandSizes = []
        demandIntervals = []

        q = 0
        for t in range(n):
            q += 1
            if y[t] > 0:
                demandSizes.append(y[t])
                demandIntervals.append(q)
                q = 0

        if len(demandSizes) < 2:
            self.z = np.mean(y[y > 0]) if np.any(y > 0) else 0.0
            self.p = n / max(np.sum(y > 0), 1)
            self.residuals = np.zeros(1)
            self.fitted = True
            return self

        self.z = demandSizes[0]
        self.p = demandIntervals[0]

        residuals = []
        for i in range(1, len(demandSizes)):
            forecast = self.z / self.p if self.p > 0 else 0.0
            residuals.append(demandSizes[i] - forecast * demandIntervals[i])
            self.z = self.alpha * demandSizes[i] + (1 - self.alpha) * self.z
            self.p = self.alpha * demandIntervals[i] + (1 - self.alpha) * self.p

        self.residuals = np.array(residuals) if residuals else np.zeros(1)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        forecast = self.z / self.p if self.p > 0 else 0.0
        predictions = np.full(steps, max(forecast, 0.0))

        sigma = np.std(self.residuals) if len(self.residuals) > 1 else abs(forecast) * 0.5 + 0.1
        margin = 1.96 * sigma
        lower = np.maximum(predictions - margin, 0.0)
        upper = predictions + margin

        return predictions, lower, upper


class CrostonSBA:
    """
    Syntetos-Boylan Approximation

    Croston Classic의 편향 보정:
    forecast = (1 - alpha/2) * z/p
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._classic = CrostonClassic(alpha=alpha)
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'CrostonSBA':
        self._classic.fit(y)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        z, p = self._classic.z, self._classic.p
        biasCorrection = 1 - self.alpha / 2
        forecast = biasCorrection * z / p if p > 0 else 0.0
        predictions = np.full(steps, max(forecast, 0.0))

        sigma = np.std(self._classic.residuals) if len(self._classic.residuals) > 1 else abs(forecast) * 0.5 + 0.1
        margin = 1.96 * sigma
        lower = np.maximum(predictions - margin, 0.0)
        upper = predictions + margin

        return predictions, lower, upper


class CrostonTSB:
    """
    Teunter-Syntetos-Babai Method

    수요 확률 d_t를 직접 추정:
    - d_t: 수요 발생 확률 SES
    - z_t: 수요 크기 SES (발생 시만)
    예측: d_t * z_t
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.1):
        self.alpha = alpha
        self.beta = beta
        self.z = 0.0
        self.d = 0.0
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'CrostonTSB':
        n = len(y)

        self.z = np.mean(y[y > 0]) if np.any(y > 0) else 1.0
        self.d = np.mean(y > 0)

        residuals = []
        for t in range(n):
            forecast = self.d * self.z
            residuals.append(y[t] - forecast)

            if y[t] > 0:
                self.z = self.alpha * y[t] + (1 - self.alpha) * self.z
                self.d = self.beta * 1.0 + (1 - self.beta) * self.d
            else:
                self.d = (1 - self.beta) * self.d

        self.residuals = np.array(residuals)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        forecast = self.d * self.z
        predictions = np.full(steps, max(forecast, 0.0))

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 1 else abs(forecast) * 0.5 + 0.1
        margin = 1.96 * sigma
        lower = np.maximum(predictions - margin, 0.0)
        upper = predictions + margin

        return predictions, lower, upper


class AutoCroston:
    """
    Automatic Croston variant selection

    데이터 특성에 따라 Classic/SBA/TSB 중 최적 선택.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.bestModel = None
        self.bestVariant = 'classic'

    def fit(self, y: np.ndarray) -> 'AutoCroston':
        n = len(y)
        trainSize = int(n * 0.8)

        if trainSize < 10:
            self.bestModel = CrostonSBA(alpha=self.alpha)
            self.bestModel.fit(y)
            self.bestVariant = 'sba'
            return self

        trainData = y[:trainSize]
        testData = y[trainSize:]
        testSteps = len(testData)

        bestMAE = np.inf
        variants = {
            'classic': CrostonClassic(alpha=self.alpha),
            'sba': CrostonSBA(alpha=self.alpha),
            'tsb': CrostonTSB(alpha=self.alpha, beta=self.alpha),
        }

        for name, model in variants.items():
            try:
                model.fit(trainData)
                pred, _, _ = model.predict(testSteps)
                mae = np.mean(np.abs(testData - pred[:len(testData)]))
                if mae < bestMAE:
                    bestMAE = mae
                    self.bestVariant = name
            except Exception:
                continue

        self.bestModel = variants.get(self.bestVariant, CrostonSBA(alpha=self.alpha))
        self.bestModel.fit(y)
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.bestModel is None:
            raise ValueError("Model not fitted.")
        return self.bestModel.predict(steps)
