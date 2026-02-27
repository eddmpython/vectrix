"""
Linear Regression Models (pure numpy/scipy)

No sklearn dependency. Implements:
- OLS Linear Regression (via normal equations)
- Ridge Regression (L2 regularization)
- Lasso Regression (coordinate descent)
- ElasticNet Regression (L1 + L2)
"""


import numpy as np


class LinearRegressor:
    """
    Ordinary Least Squares via normal equations

    β = (X'X)^{-1} X'y
    """

    def __init__(self, fitIntercept: bool = True):
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressor':
        n, p = X.shape

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X

        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(Xa.shape[1])
            beta[0] = np.mean(y) if self.fitIntercept else 0.0

        if self.fitIntercept:
            self.intercept = beta[0]
            self.coef = beta[1:]
        else:
            self.coef = beta

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercept


class RidgeRegressor:
    """
    Ridge Regression (L2 regularized)

    β = (X'X + λI)^{-1} X'y
    """

    def __init__(self, alpha: float = 1.0, fitIntercept: bool = True):
        self.alpha = alpha
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeRegressor':
        n, p = X.shape

        if self.fitIntercept:
            xMean = np.mean(X, axis=0)
            yMean = np.mean(y)
            Xc = X - xMean
            yc = y - yMean
        else:
            Xc = X
            yc = y

        I = np.eye(p)
        try:
            self.coef = np.linalg.solve(Xc.T @ Xc + self.alpha * I, Xc.T @ yc)
        except np.linalg.LinAlgError:
            self.coef = np.zeros(p)

        if self.fitIntercept:
            self.intercept = yMean - xMean @ self.coef
        else:
            self.intercept = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercept


class LassoRegressor:
    """
    Lasso Regression (L1 regularized)

    Coordinate descent implementation.
    """

    def __init__(self, alpha: float = 1.0, maxIter: int = 1000, tol: float = 1e-4,
                 fitIntercept: bool = True):
        self.alpha = alpha
        self.maxIter = maxIter
        self.tol = tol
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LassoRegressor':
        n, p = X.shape

        if self.fitIntercept:
            xMean = np.mean(X, axis=0)
            yMean = np.mean(y)
            Xc = X - xMean
            yc = y - yMean
        else:
            Xc = X
            yc = y

        colNorms = np.sum(Xc ** 2, axis=0)
        beta = np.zeros(p)

        for iteration in range(self.maxIter):
            betaOld = beta.copy()

            for j in range(p):
                if colNorms[j] < 1e-10:
                    beta[j] = 0.0
                    continue

                residual = yc - Xc @ beta + Xc[:, j] * beta[j]
                rho = Xc[:, j] @ residual

                beta[j] = self._softThreshold(rho, self.alpha * n) / colNorms[j]

            if np.max(np.abs(beta - betaOld)) < self.tol:
                break

        self.coef = beta
        if self.fitIntercept:
            self.intercept = yMean - xMean @ self.coef
        else:
            self.intercept = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercept

    @staticmethod
    def _softThreshold(x: float, lam: float) -> float:
        if x > lam:
            return x - lam
        elif x < -lam:
            return x + lam
        return 0.0


class ElasticNetRegressor:
    """
    ElasticNet Regression (L1 + L2)

    Coordinate descent with both penalties.
    """

    def __init__(self, alpha: float = 1.0, l1Ratio: float = 0.5,
                 maxIter: int = 1000, tol: float = 1e-4, fitIntercept: bool = True):
        self.alpha = alpha
        self.l1Ratio = l1Ratio
        self.maxIter = maxIter
        self.tol = tol
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ElasticNetRegressor':
        n, p = X.shape
        l1Penalty = self.alpha * self.l1Ratio
        l2Penalty = self.alpha * (1 - self.l1Ratio)

        if self.fitIntercept:
            xMean = np.mean(X, axis=0)
            yMean = np.mean(y)
            Xc = X - xMean
            yc = y - yMean
        else:
            Xc = X
            yc = y

        colNorms = np.sum(Xc ** 2, axis=0)
        beta = np.zeros(p)

        for iteration in range(self.maxIter):
            betaOld = beta.copy()

            for j in range(p):
                denom = colNorms[j] + l2Penalty * n
                if denom < 1e-10:
                    beta[j] = 0.0
                    continue

                residual = yc - Xc @ beta + Xc[:, j] * beta[j]
                rho = Xc[:, j] @ residual

                beta[j] = self._softThreshold(rho, l1Penalty * n) / denom

            if np.max(np.abs(beta - betaOld)) < self.tol:
                break

        self.coef = beta
        if self.fitIntercept:
            self.intercept = yMean - xMean @ self.coef
        else:
            self.intercept = 0.0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.coef + self.intercept

    @staticmethod
    def _softThreshold(x: float, lam: float) -> float:
        if x > lam:
            return x - lam
        elif x < -lam:
            return x + lam
        return 0.0
