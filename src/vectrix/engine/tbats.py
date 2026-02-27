"""
TBATS: Trigonometric seasonality, Box-Cox transformation,
       ARMA errors, Trend and Seasonal components

De Livera et al. (2011) 기반.
복잡한 다중 계절성 (예: 시간 데이터의 일별+주별+연별)을 처리.
Fourier 급수로 계절성을 표현하여 상태 공간 폭발을 방지.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.optimize import minimize


class TBATS:
    """
    TBATS Model

    각 계절 주기에 Fourier 항을 사용하여
    상태 공간 크기를 제어하면서 다중 계절성 처리.
    """

    def __init__(
        self,
        periods: Optional[List[int]] = None,
        useBoxCox: bool = False,
        useTrend: bool = True,
        useDamping: bool = False,
        maxHarmonics: Optional[List[int]] = None
    ):
        self.periods = periods or [7]
        self.useBoxCox = useBoxCox
        self.useTrend = useTrend
        self.useDamping = useDamping
        self.maxHarmonics = maxHarmonics

        self.boxCoxLambda = 1.0
        self.alpha = 0.3
        self.beta = 0.01
        self.phi = 0.98
        self.arCoeffs = []
        self.maCoeffs = []

        self.level = 0.0
        self.trend = 0.0
        self.harmonicStates = {}
        self.harmonicCounts = {}
        self.residuals = None
        self.fitted = False
        self._n = 0

    def fit(self, y: np.ndarray) -> 'TBATS':
        n = len(y)
        self._n = n

        if n < 10:
            self.level = np.mean(y)
            self.residuals = y - self.level
            self.fitted = True
            return self

        workData = y.copy()

        if self.useBoxCox:
            self.boxCoxLambda = self._estimateBoxCox(y)
            workData = self._boxCoxTransform(y, self.boxCoxLambda)

        self._determineHarmonics(workData)
        self._optimizeParams(workData)
        self._filterFinal(workData)

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.zeros(steps)
        level = self.level
        trend = self.trend

        harmonicStates = {}
        for period in self.periods:
            if period in self.harmonicStates:
                harmonicStates[period] = [
                    (s.copy(), g.copy()) for s, g in self.harmonicStates[period]
                ]

        for h in range(steps):
            pred = level
            if self.useTrend:
                dampFactor = self.phi ** (h + 1) if self.useDamping else 1.0
                pred += trend * (h + 1) * dampFactor / max(h + 1, 1) if self.useDamping else trend

            for period in self.periods:
                if period in harmonicStates:
                    for sj, sj_star in harmonicStates[period]:
                        pred += sj[0]

            if self.useBoxCox and self.boxCoxLambda != 1.0:
                pred = self._inverseBoxCox(np.array([pred]), self.boxCoxLambda)[0]

            predictions[h] = pred

            if self.useTrend and self.useDamping:
                trend = self.phi * trend

        sigma = np.std(self.residuals) if self.residuals is not None and len(self.residuals) > 1 else 1.0
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

        if self.useBoxCox and self.boxCoxLambda != 1.0:
            margin *= abs(np.mean(predictions)) * 0.01 + 1.0

        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def _determineHarmonics(self, y: np.ndarray):
        n = len(y)
        for period in self.periods:
            if self.maxHarmonics is not None and len(self.maxHarmonics) > self.periods.index(period):
                k = self.maxHarmonics[self.periods.index(period)]
            else:
                k = min(max(1, period // 2), 5)

            k = min(k, n // (2 * period + 1))
            k = max(k, 1)
            self.harmonicCounts[period] = k

    def _optimizeParams(self, y: np.ndarray):
        n = len(y)

        nParams = 2
        if self.useTrend:
            nParams += 1
        if self.useDamping:
            nParams += 1

        def objective(params):
            alpha = params[0]
            beta = params[1] if self.useTrend else 0.0
            phi = params[2] if self.useDamping and self.useTrend else 0.98

            level = y[0]
            trend = 0.0 if self.useTrend else 0.0

            states = {}
            for period in self.periods:
                k = self.harmonicCounts.get(period, 1)
                states[period] = []
                for j in range(k):
                    freq = 2 * np.pi * (j + 1) / period
                    sj = np.array([0.0])
                    sj_star = np.array([0.0])
                    states[period].append((sj, sj_star, freq))

            sse = 0.0
            for t in range(1, n):
                pred = level
                if self.useTrend:
                    pred += trend

                for period in self.periods:
                    for sj, sj_star, freq in states[period]:
                        pred += sj[0]

                error = y[t] - pred
                sse += error ** 2

                level = level + alpha * error
                if self.useTrend:
                    if self.useDamping:
                        trend = phi * trend + beta * error
                    else:
                        trend = trend + beta * error

                gamma = alpha * 0.5
                for period in self.periods:
                    for sj, sj_star, freq in states[period]:
                        cosF = np.cos(freq)
                        sinF = np.sin(freq)
                        newSj = sj[0] * cosF + sj_star[0] * sinF + gamma * error
                        newSjStar = -sj[0] * sinF + sj_star[0] * cosF
                        sj[0] = newSj
                        sj_star[0] = newSjStar

            return sse

        bounds = [(0.001, 0.99)]
        x0 = [0.3]
        if self.useTrend:
            bounds.append((0.0001, 0.5))
            x0.append(0.01)
        if self.useDamping and self.useTrend:
            bounds.append((0.8, 0.999))
            x0.append(0.98)

        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 30, 'ftol': 1e-4})

        self.alpha = result.x[0]
        idx = 1
        if self.useTrend:
            self.beta = result.x[idx]
            idx += 1
        if self.useDamping and self.useTrend:
            self.phi = result.x[idx]

    def _filterFinal(self, y: np.ndarray):
        n = len(y)
        self.level = y[0]
        self.trend = 0.0

        self.harmonicStates = {}
        for period in self.periods:
            k = self.harmonicCounts.get(period, 1)
            self.harmonicStates[period] = []
            for j in range(k):
                freq = 2 * np.pi * (j + 1) / period
                sj = np.array([0.0])
                sj_star = np.array([0.0])
                self.harmonicStates[period].append((sj, sj_star))

        gamma = self.alpha * 0.5
        residuals = []

        for t in range(1, n):
            pred = self.level
            if self.useTrend:
                pred += self.trend

            for period in self.periods:
                for sj, sj_star in self.harmonicStates[period]:
                    pred += sj[0]

            error = y[t] - pred
            residuals.append(error)

            self.level += self.alpha * error
            if self.useTrend:
                if self.useDamping:
                    self.trend = self.phi * self.trend + self.beta * error
                else:
                    self.trend += self.beta * error

            for period in self.periods:
                k = self.harmonicCounts.get(period, 1)
                for j, (sj, sj_star) in enumerate(self.harmonicStates[period]):
                    freq = 2 * np.pi * (j + 1) / period
                    cosF = np.cos(freq)
                    sinF = np.sin(freq)
                    newSj = sj[0] * cosF + sj_star[0] * sinF + gamma * error
                    newSjStar = -sj[0] * sinF + sj_star[0] * cosF
                    sj[0] = newSj
                    sj_star[0] = newSjStar

        self.residuals = np.array(residuals) if residuals else np.zeros(1)

    def _estimateBoxCox(self, y: np.ndarray) -> float:
        if np.any(y <= 0):
            return 1.0
        from scipy.stats import boxcox_normmax
        try:
            lam = boxcox_normmax(y, method='mle')
            return np.clip(lam, 0.0, 1.0)
        except Exception:
            return 1.0

    def _boxCoxTransform(self, y: np.ndarray, lam: float) -> np.ndarray:
        if abs(lam) < 1e-6:
            return np.log(np.maximum(y, 1e-10))
        return (np.power(np.maximum(y, 1e-10), lam) - 1) / lam

    def _inverseBoxCox(self, y: np.ndarray, lam: float) -> np.ndarray:
        if abs(lam) < 1e-6:
            return np.exp(y)
        return np.power(np.maximum(lam * y + 1, 1e-10), 1 / lam)


class AutoTBATS:
    """
    Automatic TBATS model selection

    trend/damping/boxcox 조합을 시도하여 최적 선택.
    """

    def __init__(self, periods: Optional[List[int]] = None):
        self.periods = periods or [7]
        self.bestModel = None

    def fit(self, y: np.ndarray) -> TBATS:
        n = len(y)
        bestAIC = np.inf

        configs = [
            {'useTrend': True, 'useDamping': False, 'useBoxCox': False},
            {'useTrend': True, 'useDamping': True, 'useBoxCox': False},
            {'useTrend': False, 'useDamping': False, 'useBoxCox': False},
        ]

        if np.all(y > 0):
            configs.append({'useTrend': True, 'useDamping': True, 'useBoxCox': True})

        for cfg in configs:
            try:
                model = TBATS(periods=self.periods, **cfg)
                model.fit(y)

                if model.residuals is not None and len(model.residuals) > 0:
                    sse = np.sum(model.residuals ** 2)
                    k = 2 + int(cfg['useTrend']) + int(cfg['useDamping']) + int(cfg['useBoxCox'])
                    nRes = len(model.residuals)
                    aic = nRes * np.log(sse / nRes + 1e-10) + 2 * k

                    if aic < bestAIC:
                        bestAIC = aic
                        self.bestModel = model
            except Exception:
                continue

        if self.bestModel is None:
            self.bestModel = TBATS(periods=self.periods)
            self.bestModel.fit(y)

        return self.bestModel

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.bestModel is None:
            raise ValueError("Model not fitted.")
        return self.bestModel.predict(steps)
