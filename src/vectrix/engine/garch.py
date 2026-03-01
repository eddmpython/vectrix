"""
GARCH Family Models

Financial/volatility time series forecasting:
- GARCH(1,1): Standard conditional variance model
- EGARCH: Asymmetric volatility (leverage effect)
- GJR-GARCH: Asymmetric volatility (threshold)

MLE-based parameter estimation (scipy.optimize)
"""

from typing import Tuple

import numpy as np
from scipy.optimize import minimize


class GARCHModel:
    """
    GARCH(1,1) Model

    Conditional variance: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

    Used for return/volatility forecasting.
    Mean model is AR(1) or constant.
    """

    def __init__(self, meanModel: str = 'constant'):
        """
        Parameters
        ----------
        meanModel : str
            'constant' or 'ar1'
        """
        self.meanModel = meanModel

        self.mu = 0.0
        self.ar1 = 0.0
        self.omega = 0.0001
        self.alphaG = 0.1
        self.betaG = 0.85

        self.lastVariance = 0.0
        self.lastResidual = 0.0
        self.unconditionalVar = 0.0
        self.residuals = None
        self.condVariances = None
        self.fitted = False
        self._y = None

    def fit(self, y: np.ndarray) -> 'GARCHModel':
        n = len(y)
        self._y = y.copy()

        if n < 20:
            self.mu = np.mean(y)
            self.unconditionalVar = np.var(y)
            self.lastVariance = self.unconditionalVar
            self.residuals = y - self.mu
            self.condVariances = np.full(n, self.unconditionalVar)
            self.fitted = True
            return self

        self.mu = np.mean(y)
        self.unconditionalVar = np.var(y)

        def negLogLik(params):
            mu = params[0]
            omega = params[1]
            alphaG = params[2]
            betaG = params[3]

            if omega <= 0 or alphaG < 0 or betaG < 0 or alphaG + betaG >= 1:
                return 1e10

            sigma2 = self.unconditionalVar
            nll = 0.0

            for t in range(n):
                if self.meanModel == 'ar1' and t > 0:
                    meanPred = mu + self.ar1 * (y[t - 1] - mu)
                else:
                    meanPred = mu

                eps = y[t] - meanPred
                nll += 0.5 * (np.log(max(sigma2, 1e-20)) + eps ** 2 / max(sigma2, 1e-20))
                sigma2 = omega + alphaG * eps ** 2 + betaG * sigma2

            return nll

        x0 = [self.mu, self.unconditionalVar * 0.05, 0.1, 0.85]
        bounds = [
            (np.min(y) - np.std(y), np.max(y) + np.std(y)),
            (1e-8, np.var(y) * 10),
            (1e-6, 0.5),
            (0.01, 0.999)
        ]

        result = minimize(negLogLik, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-6})

        self.mu = result.x[0]
        self.omega = result.x[1]
        self.alphaG = result.x[2]
        self.betaG = result.x[3]

        sigma2 = self.unconditionalVar
        residuals = []
        condVars = []

        for t in range(n):
            if self.meanModel == 'ar1' and t > 0:
                meanPred = self.mu + self.ar1 * (y[t - 1] - self.mu)
            else:
                meanPred = self.mu

            eps = y[t] - meanPred
            residuals.append(eps)
            condVars.append(sigma2)
            sigma2 = self.omega + self.alphaG * eps ** 2 + self.betaG * sigma2

        self.residuals = np.array(residuals)
        self.condVariances = np.array(condVars)
        self.lastVariance = sigma2
        self.lastResidual = residuals[-1] if residuals else 0.0

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.full(steps, self.mu)
        variances = np.zeros(steps)

        sigma2 = self.lastVariance
        for h in range(steps):
            variances[h] = sigma2
            sigma2 = self.omega + (self.alphaG + self.betaG) * sigma2

        margin = 1.96 * np.sqrt(variances)
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def forecastVariance(self, steps: int) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        variances = np.zeros(steps)
        sigma2 = self.lastVariance

        for h in range(steps):
            variances[h] = sigma2
            sigma2 = self.omega + (self.alphaG + self.betaG) * sigma2

        return variances


class EGARCHModel:
    """
    EGARCH(1,1) Model

    log(σ²_t) = ω + α·g(z_{t-1}) + β·log(σ²_{t-1})
    g(z) = θ·z + γ·(|z| - E|z|)

    Asymmetric volatility modeling (leverage effect).
    """

    def __init__(self):
        self.mu = 0.0
        self.omega = 0.0
        self.alphaE = 0.1
        self.betaE = 0.95
        self.gamma = -0.1

        self.lastLogVar = 0.0
        self.unconditionalVar = 0.0
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'EGARCHModel':
        n = len(y)
        self.mu = np.mean(y)
        self.unconditionalVar = np.var(y)

        if n < 20:
            self.lastLogVar = np.log(max(self.unconditionalVar, 1e-10))
            self.residuals = y - self.mu
            self.fitted = True
            return self

        def negLogLik(params):
            omega = params[0]
            alphaE = params[1]
            betaE = params[2]
            gamma = params[3]

            logSigma2 = np.log(max(self.unconditionalVar, 1e-10))
            nll = 0.0
            sqrt2pi = np.sqrt(2 / np.pi)

            for t in range(n):
                sigma2 = np.exp(logSigma2)
                sigma = np.sqrt(max(sigma2, 1e-20))
                eps = y[t] - self.mu
                z = eps / max(sigma, 1e-10)

                nll += 0.5 * (logSigma2 + eps ** 2 / max(sigma2, 1e-20))

                g = alphaE * z + gamma * (abs(z) - sqrt2pi)
                logSigma2 = omega + g + betaE * logSigma2

                logSigma2 = np.clip(logSigma2, -20, 20)

            return nll

        x0 = [np.log(max(self.unconditionalVar, 1e-10)) * 0.05, 0.1, 0.95, -0.1]
        bounds = [(-5, 5), (-0.5, 0.5), (0.01, 0.999), (-0.5, 0.5)]

        result = minimize(negLogLik, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-6})

        self.omega = result.x[0]
        self.alphaE = result.x[1]
        self.betaE = result.x[2]
        self.gamma = result.x[3]

        logSigma2 = np.log(max(self.unconditionalVar, 1e-10))
        sqrt2pi = np.sqrt(2 / np.pi)
        residuals = []

        for t in range(n):
            sigma2 = np.exp(logSigma2)
            sigma = np.sqrt(max(sigma2, 1e-20))
            eps = y[t] - self.mu
            z = eps / max(sigma, 1e-10)
            residuals.append(eps)

            g = self.alphaE * z + self.gamma * (abs(z) - sqrt2pi)
            logSigma2 = self.omega + g + self.betaE * logSigma2
            logSigma2 = np.clip(logSigma2, -20, 20)

        self.lastLogVar = logSigma2
        self.residuals = np.array(residuals)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.full(steps, self.mu)
        variances = np.zeros(steps)

        logSigma2 = self.lastLogVar
        for h in range(steps):
            variances[h] = np.exp(logSigma2)
            logSigma2 = self.omega + self.betaE * logSigma2

        margin = 1.96 * np.sqrt(variances)
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper


class GJRGARCHModel:
    """
    GJR-GARCH(1,1) Model

    σ²_t = ω + (α + γ·I_{t-1})·ε²_{t-1} + β·σ²_{t-1}
    I_{t-1} = 1 if ε_{t-1} < 0 (asymmetric response)

    Larger volatility response to negative shocks.
    """

    def __init__(self):
        self.mu = 0.0
        self.omega = 0.0001
        self.alphaGJR = 0.05
        self.betaGJR = 0.85
        self.gammaGJR = 0.1

        self.lastVariance = 0.0
        self.lastResidual = 0.0
        self.unconditionalVar = 0.0
        self.residuals = None
        self.fitted = False

    def fit(self, y: np.ndarray) -> 'GJRGARCHModel':
        n = len(y)
        self.mu = np.mean(y)
        self.unconditionalVar = np.var(y)

        if n < 20:
            self.lastVariance = self.unconditionalVar
            self.residuals = y - self.mu
            self.fitted = True
            return self

        def negLogLik(params):
            omega = params[0]
            alphaGJR = params[1]
            betaGJR = params[2]
            gammaGJR = params[3]

            if omega <= 0 or alphaGJR < 0 or betaGJR < 0 or gammaGJR < 0:
                return 1e10
            if alphaGJR + betaGJR + 0.5 * gammaGJR >= 1:
                return 1e10

            sigma2 = self.unconditionalVar
            nll = 0.0

            for t in range(n):
                eps = y[t] - self.mu
                nll += 0.5 * (np.log(max(sigma2, 1e-20)) + eps ** 2 / max(sigma2, 1e-20))
                indicator = 1.0 if eps < 0 else 0.0
                sigma2 = omega + (alphaGJR + gammaGJR * indicator) * eps ** 2 + betaGJR * sigma2

            return nll

        x0 = [self.unconditionalVar * 0.05, 0.05, 0.85, 0.1]
        bounds = [(1e-8, np.var(y) * 10), (1e-6, 0.5), (0.01, 0.999), (1e-6, 0.5)]

        result = minimize(negLogLik, x0, bounds=bounds, method='L-BFGS-B',
                          options={'maxiter': 50, 'ftol': 1e-6})

        self.omega = result.x[0]
        self.alphaGJR = result.x[1]
        self.betaGJR = result.x[2]
        self.gammaGJR = result.x[3]

        sigma2 = self.unconditionalVar
        residuals = []

        for t in range(n):
            eps = y[t] - self.mu
            residuals.append(eps)
            indicator = 1.0 if eps < 0 else 0.0
            sigma2 = self.omega + (self.alphaGJR + self.gammaGJR * indicator) * eps ** 2 + self.betaGJR * sigma2

        self.lastVariance = sigma2
        self.lastResidual = residuals[-1] if residuals else 0.0
        self.residuals = np.array(residuals)
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise ValueError("Model not fitted.")

        predictions = np.full(steps, self.mu)
        variances = np.zeros(steps)

        sigma2 = self.lastVariance
        for h in range(steps):
            variances[h] = sigma2
            sigma2 = self.omega + (self.alphaGJR + self.betaGJR + 0.5 * self.gammaGJR) * sigma2

        margin = 1.96 * np.sqrt(variances)
        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper
