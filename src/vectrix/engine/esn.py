"""
Echo State Network (ESN) Forecaster

Reservoir Computing 기반 비선형 시계열 예측 모델.
랜덤 고정 reservoir + 학습 가능 출력 가중치(Ridge regression).

E037 실험 결과 → E041 스트레스 테스트 통과 → 엔진 통합.
- 평균 순위 3.47 (1위): mstl(3.71), 4theta(3.62) 초과
- Safety 100%, Seed CV < 40%, Speed 11ms (n=1000)
- 비선형 동역학 포착, 잔차 상관 0.13~0.66
"""

from typing import Tuple

import numpy as np


class EchoStateForecaster:
    """
    Echo State Network for Time Series Forecasting

    sparse random reservoir (density = min(0.1, 10/N)) +
    spectral radius scaling + leaky integration +
    extended state [x, x[:1]**2] + Ridge regression output.

    Autoregressive multi-step prediction with prediction clamping.
    """

    def __init__(
        self,
        reservoirSize: int = 100,
        spectralRadius: float = 0.9,
        inputScaling: float = 0.5,
        leakRate: float = 0.3,
        ridgeAlpha: float = 1e-4,
        seed: int = 42
    ):
        self._reservoirSize = reservoirSize
        self._spectralRadius = spectralRadius
        self._inputScaling = inputScaling
        self._leakRate = leakRate
        self._ridgeAlpha = ridgeAlpha
        self._seed = seed

        self._Win = None
        self._W = None
        self._Wout = None
        self._lastState = None
        self._lastInput = None
        self._yMean = 0.0
        self._yStd = 1.0
        self._residStd = 0.0
        self._predClamp = 3.0

        self.fitted = False
        self.residuals = None

    def fit(self, y: np.ndarray) -> 'EchoStateForecaster':
        y = np.asarray(y, dtype=np.float64).copy()
        n = len(y)

        self._yMean = np.mean(y)
        self._yStd = max(np.std(y), 1e-8)
        yNorm = (y - self._yMean) / self._yStd

        rng = np.random.default_rng(self._seed)
        N = self._reservoirSize

        self._Win = rng.uniform(-1, 1, (N, 1)) * self._inputScaling

        density = min(0.1, 10.0 / N)
        W = rng.standard_normal((N, N))
        mask = rng.random((N, N)) < density
        W *= mask

        eigenvalues = np.linalg.eigvals(W)
        maxEig = np.max(np.abs(eigenvalues))
        if maxEig > 1e-10:
            W = W * (self._spectralRadius / maxEig)
        self._W = W

        washout = min(n // 5, 100)
        states = np.zeros((n - washout, N))
        x = np.zeros(N)

        for t in range(n):
            u = yNorm[t]
            xNew = np.tanh(self._Win.flatten() * u + self._W @ x)
            x = (1.0 - self._leakRate) * x + self._leakRate * xNew
            if t >= washout:
                states[t - washout] = x

        targets = yNorm[washout + 1:]
        stateMatrix = states[:-1]

        if len(stateMatrix) == 0 or len(targets) == 0:
            self._Wout = np.zeros(N + 1)
            self._lastState = x
            self._lastInput = yNorm[-1]
            self._residStd = self._yStd * 0.1
            self._predClamp = 3.0
            self.residuals = np.zeros(n)
            self.fitted = True
            return self

        extStates = np.hstack([stateMatrix, stateMatrix[:, :1] ** 2])

        noiseLvl = np.std(np.diff(yNorm))
        alpha = max(self._ridgeAlpha, noiseLvl ** 2 * 0.1)

        self._Wout = np.linalg.solve(
            extStates.T @ extStates + alpha * np.eye(extStates.shape[1]),
            extStates.T @ targets
        )

        fittedNorm = extStates @ self._Wout
        residualsNorm = targets - fittedNorm
        self._residStd = max(np.std(residualsNorm) * self._yStd, 1e-8)
        self._predClamp = max(3.0, 3.0 * np.std(yNorm))

        fullResiduals = np.zeros(n)
        fullResiduals[washout + 1:washout + 1 + len(residualsNorm)] = residualsNorm * self._yStd
        self.residuals = fullResiduals

        self._lastState = x
        self._lastInput = yNorm[-1]

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("fit() must be called before predict()")

        predictions = np.zeros(steps)
        x = self._lastState.copy()
        u = self._lastInput
        clamp = self._predClamp

        for h in range(steps):
            xNew = np.tanh(self._Win.flatten() * u + self._W @ x)
            x = (1.0 - self._leakRate) * x + self._leakRate * xNew
            extState = np.concatenate([x, x[:1] ** 2])
            yPred = extState @ self._Wout
            yPred = np.clip(yPred, -clamp, clamp)
            predictions[h] = yPred
            u = yPred

        predictions = predictions * self._yStd + self._yMean

        sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - 1.96 * sigma
        upper = predictions + 1.96 * sigma

        return predictions, lower, upper
