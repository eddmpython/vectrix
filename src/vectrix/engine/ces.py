"""
Complex Exponential Smoothing (CES)

Svetunkov (2023) 기반 복소수 지수평활법.
기존 ETS의 확장으로, 복소수 평활 파라미터를 사용하여
추세와 계절성을 동시에 모델링.

Forms:
- N (None): 단순 CES
- S (Simple): 단순 계절 CES
- P (Partial): 부분 계절 CES
- F (Full): 전체 계절 CES
"""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, cache=True)
def _cesNonSeasonalSSE(y: np.ndarray, a0Real: float, a0Imag: float) -> float:
    n = len(y)
    levelReal = y[0]
    levelImag = 0.0
    sse = 0.0

    for t in range(1, n):
        forecast = levelReal
        error = y[t] - forecast
        sse += error * error
        newReal = a0Real * y[t] + (1.0 - a0Real) * levelReal + a0Imag * levelImag
        newImag = a0Imag * (y[t] - levelReal) + (1.0 - a0Real) * levelImag
        levelReal = newReal
        levelImag = newImag

    return sse


@jit(nopython=True, cache=True)
def _cesSeasonalSSE(y: np.ndarray, a0Real: float, a0Imag: float, gamma: float,
                    seasonalInit: np.ndarray, m: int) -> float:
    n = len(y)
    levelReal = y[0] - seasonalInit[0]
    levelImag = 0.0
    seasonal = seasonalInit.copy()
    sse = 0.0

    for t in range(1, n):
        sidx = t % m
        forecast = levelReal + seasonal[sidx]
        error = y[t] - forecast
        sse += error * error
        yAdj = y[t] - seasonal[sidx]
        newReal = a0Real * yAdj + (1.0 - a0Real) * levelReal + a0Imag * levelImag
        newImag = a0Imag * (yAdj - levelReal) + (1.0 - a0Real) * levelImag
        levelReal = newReal
        levelImag = newImag
        seasonal[sidx] += gamma * error

    return sse


class CESModel:
    """
    Complex Exponential Smoothing

    복소수 평활 파라미터 (a0, a1)을 사용하여
    시계열의 수준과 잠재적 성장을 모델링.
    """

    def __init__(self, form: str = 'N', period: int = 1):
        """
        Parameters
        ----------
        form : str
            'N' (None), 'S' (Simple), 'P' (Partial), 'F' (Full)
        period : int
            계절 주기
        """
        self.form = form.upper()
        self.period = max(1, period)

        self.a0 = complex(0.1, 0.1)
        self.a1 = complex(0.1, 0.0)
        self.level = complex(0.0, 0.0)
        self.seasonal = None
        self.residuals = None
        self.fitted = False
        self._y = None

    def fit(self, y: np.ndarray) -> 'CESModel':
        n = len(y)
        self._y = y.copy()

        if self.form in ('S', 'P', 'F') and self.period > 1 and n >= self.period * 2:
            self._fitSeasonal(y)
        else:
            self._fitNonSeasonal(y)

        self.fitted = True
        return self

    def _fitNonSeasonal(self, y: np.ndarray):
        n = len(y)

        def objective(params):
            return _cesNonSeasonalSSE(y, params[0], params[1])

        x0 = [0.3, 0.1]
        bounds = [(0.01, 0.99), (-0.5, 0.5)]
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 30, 'ftol': 1e-4})

        self.a0 = complex(result.x[0], result.x[1])
        self.level = complex(y[0], 0.0)

        residuals = []
        for t in range(1, n):
            forecast = self.level.real
            error = y[t] - forecast
            residuals.append(error)
            self.level = self.a0 * complex(y[t], 0.0) + (1 - self.a0) * self.level

        self.residuals = np.array(residuals)

    def _fitSeasonal(self, y: np.ndarray):
        n = len(y)
        m = self.period

        seasonalInit = np.zeros(m)
        for i in range(m):
            vals = y[i::m]
            seasonalInit[i] = np.mean(vals) - np.mean(y)

        def objective(params):
            return _cesSeasonalSSE(y, params[0], params[1], params[2], seasonalInit, m)

        x0 = [0.3, 0.1, 0.1]
        bounds = [(0.01, 0.99), (-0.5, 0.5), (0.001, 0.5)]
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 30, 'ftol': 1e-4})

        self.a0 = complex(result.x[0], result.x[1])
        gammaOpt = result.x[2]

        self.level = complex(y[0] - seasonalInit[0], 0.0)
        self.seasonal = seasonalInit.copy()
        residuals = []

        for t in range(1, n):
            sidx = t % m
            forecast = self.level.real + self.seasonal[sidx]
            error = y[t] - forecast
            residuals.append(error)
            self.level = self.a0 * complex(y[t] - self.seasonal[sidx], 0.0) + (1 - self.a0) * self.level
            self.seasonal[sidx] += gammaOpt * error

        self.residuals = np.array(residuals)

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.zeros(steps)

        for h in range(steps):
            pred = self.level.real
            if self.seasonal is not None:
                sidx = (len(self._y) + h) % self.period
                pred += self.seasonal[sidx]
            predictions[h] = pred

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 1 else 1.0
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper


class AutoCES:
    """
    Automatic CES model selection

    N, S, P, F 4가지 form을 시도하고 AICc로 최적 선택.
    """

    FORMS = ['N', 'S', 'P', 'F']

    def __init__(self, period: int = 1):
        self.period = period
        self.bestModel = None
        self.bestForm = 'N'

    def fit(self, y: np.ndarray) -> CESModel:
        n = len(y)
        bestAIC = np.inf

        forms = self.FORMS
        if self.period <= 1 or n < self.period * 2:
            forms = ['N']

        for form in forms:
            try:
                model = CESModel(form=form, period=self.period)
                model.fit(y)

                if model.residuals is not None and len(model.residuals) > 0:
                    sse = np.sum(model.residuals ** 2)
                    k = 2 if form == 'N' else 3
                    nRes = len(model.residuals)
                    aic = nRes * np.log(sse / nRes + 1e-10) + 2 * k
                    if nRes - k - 1 > 0:
                        aic += 2 * k * (k + 1) / (nRes - k - 1)

                    if aic < bestAIC:
                        bestAIC = aic
                        self.bestModel = model
                        self.bestForm = form
            except Exception:
                continue

        if self.bestModel is None:
            self.bestModel = CESModel(form='N', period=self.period)
            self.bestModel.fit(y)
            self.bestForm = 'N'

        return self.bestModel

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.bestModel is None:
            raise ValueError("Model not fitted.")
        return self.bestModel.predict(steps)
