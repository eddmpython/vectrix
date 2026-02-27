"""
Vector AutoRegression (VAR) Model

Multivariate time series forecasting.
Estimates VAR(p) coefficients via OLS:
    Y_t = c + A_1 * Y_{t-1} + ... + A_p * Y_{t-p} + e_t

Includes automatic lag order selection via AIC/BIC.
"""

from typing import Optional, Tuple

import numpy as np


class VARModel:
    """
    Vector AutoRegression (VAR)

    Multivariate time series forecasting via OLS estimation.

    Parameters
    ----------
    maxLag : int
        Maximum lag order to consider.
    criterion : str
        Information criterion for lag selection ('aic' or 'bic').
    """

    def __init__(self, maxLag: int = 5, criterion: str = "aic"):
        self.maxLag = maxLag
        self.criterion = criterion
        self.order = 1
        self.intercept = None
        self.coefficients = None
        self.residuals = None
        self.sigma = None
        self._k = 0
        self._lastValues = None
        self.fitted = False

    def fit(self, Y: np.ndarray) -> 'VARModel':
        """
        Fit VAR model.

        Parameters
        ----------
        Y : np.ndarray
            Multivariate data, shape (T, k) where T = time, k = variables.

        Returns
        -------
        self
        """
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T, k = Y.shape
        self._k = k

        bestIC = np.inf
        bestP = 1

        maxP = min(self.maxLag, T // (k + 2))
        maxP = max(maxP, 1)

        for p in range(1, maxP + 1):
            ic = self._fitWithOrder(Y, p, returnIC=True)
            if ic is not None and ic < bestIC:
                bestIC = ic
                bestP = p

        self.order = bestP
        self._fitWithOrder(Y, bestP, returnIC=False)
        self._lastValues = Y[-self.order:].copy()
        self.fitted = True
        return self

    def _fitWithOrder(self, Y: np.ndarray, p: int, returnIC: bool = False):
        T, k = Y.shape
        if T <= p + 1:
            return np.inf if returnIC else None

        nObs = T - p
        Z = np.ones((nObs, 1 + k * p))
        Ymat = np.zeros((nObs, k))

        for t in range(nObs):
            Ymat[t] = Y[t + p]
            for lag in range(p):
                Z[t, 1 + lag * k: 1 + (lag + 1) * k] = Y[t + p - 1 - lag]

        ZtZ = Z.T @ Z
        reg = 1e-8 * np.eye(ZtZ.shape[0])
        try:
            B = np.linalg.solve(ZtZ + reg, Z.T @ Ymat)
        except np.linalg.LinAlgError:
            return np.inf if returnIC else None

        residuals = Ymat - Z @ B
        sigmaHat = (residuals.T @ residuals) / nObs

        if returnIC:
            logDet = np.log(np.linalg.det(sigmaHat) + 1e-300)
            nParams = k * (1 + k * p)
            if self.criterion == "bic":
                return logDet + nParams * np.log(nObs) / nObs
            else:
                return logDet + 2 * nParams / nObs

        self.intercept = B[0]
        self.coefficients = []
        for lag in range(p):
            self.coefficients.append(B[1 + lag * k: 1 + (lag + 1) * k])
        self.residuals = residuals
        self.sigma = sigmaHat
        return None

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate multivariate forecasts.

        Parameters
        ----------
        steps : int
            Number of future steps.

        Returns
        -------
        predictions : np.ndarray
            Shape (steps, k).
        lower : np.ndarray
            Shape (steps, k).
        upper : np.ndarray
            Shape (steps, k).
        """
        if not self.fitted:
            raise ValueError("Model not fitted.")

        k = self._k
        p = self.order
        history = self._lastValues.copy()
        predictions = np.zeros((steps, k))

        for h in range(steps):
            pred = self.intercept.copy()
            for lag in range(p):
                idx = len(history) - 1 - lag
                if idx >= 0:
                    pred = pred + self.coefficients[lag] @ history[idx]
            predictions[h] = pred
            history = np.vstack([history, pred])

        sigDiag = np.sqrt(np.diag(self.sigma)) if self.sigma is not None else np.ones(k)
        horizonScale = np.sqrt(np.arange(1, steps + 1)).reshape(-1, 1)
        margin = 1.96 * sigDiag * horizonScale

        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper

    def grangerCausality(self, Y: np.ndarray, cause: int, effect: int, maxLag: int = 5) -> dict:
        """
        Granger causality test.

        Tests if variable 'cause' Granger-causes variable 'effect'.

        Parameters
        ----------
        Y : np.ndarray
            Multivariate data, shape (T, k).
        cause : int
            Index of causal variable.
        effect : int
            Index of effect variable.
        maxLag : int
            Maximum lag order.

        Returns
        -------
        dict
            fStat, pValue, lag
        """
        from scipy.stats import f as fdist

        Y = np.asarray(Y, dtype=np.float64)
        T, k = Y.shape
        p = min(maxLag, T // 4)

        nObs = T - p
        if nObs < p + 2:
            return {"fStat": 0.0, "pValue": 1.0, "lag": p}

        yTarget = Y[p:, effect]

        Xfull = np.ones((nObs, 1 + 2 * p))
        Xrestricted = np.ones((nObs, 1 + p))

        for t in range(nObs):
            for lag in range(p):
                Xfull[t, 1 + lag] = Y[t + p - 1 - lag, effect]
                Xfull[t, 1 + p + lag] = Y[t + p - 1 - lag, cause]
                Xrestricted[t, 1 + lag] = Y[t + p - 1 - lag, effect]

        reg = 1e-8
        bFull = np.linalg.solve(Xfull.T @ Xfull + reg * np.eye(Xfull.shape[1]), Xfull.T @ yTarget)
        bRestr = np.linalg.solve(Xrestricted.T @ Xrestricted + reg * np.eye(Xrestricted.shape[1]), Xrestricted.T @ yTarget)

        sseFull = np.sum((yTarget - Xfull @ bFull) ** 2)
        sseRestr = np.sum((yTarget - Xrestricted @ bRestr) ** 2)

        dfNum = p
        dfDen = nObs - 2 * p - 1
        if dfDen <= 0 or sseFull <= 0:
            return {"fStat": 0.0, "pValue": 1.0, "lag": p}

        fStat = ((sseRestr - sseFull) / dfNum) / (sseFull / dfDen)
        pValue = 1 - fdist.cdf(fStat, dfNum, dfDen)

        return {"fStat": float(fStat), "pValue": float(pValue), "lag": p}


class VECMModel:
    """
    Vector Error Correction Model (VECM)

    For cointegrated multivariate time series.
    VECM(p) = alpha * beta' * Y_{t-1} + Gamma_1 * dY_{t-1} + ... + e_t

    Uses Johansen-style rank estimation.

    Parameters
    ----------
    maxLag : int
        Maximum lag for the underlying VAR.
    rank : int, optional
        Cointegration rank. If None, estimated automatically.
    """

    def __init__(self, maxLag: int = 4, rank: Optional[int] = None):
        self.maxLag = maxLag
        self.rank = rank
        self.alpha = None
        self.beta = None
        self.gamma = None
        self.intercept = None
        self.sigma = None
        self._k = 0
        self._lastLevels = None
        self._lastDiffs = None
        self.fitted = False

    def fit(self, Y: np.ndarray) -> 'VECMModel':
        """
        Fit VECM model.

        Parameters
        ----------
        Y : np.ndarray
            Multivariate data, shape (T, k). Must be I(1).

        Returns
        -------
        self
        """
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        T, k = Y.shape
        self._k = k

        dY = np.diff(Y, axis=0)

        if self.rank is None:
            self.rank = self._estimateRank(Y)
        self.rank = min(self.rank, k - 1)
        self.rank = max(self.rank, 1)

        p = min(self.maxLag, T // (k + 3))
        p = max(p, 1)

        nObs = T - p - 1
        if nObs < k + 2:
            self._fallbackFit(Y)
            return self

        Ylag = Y[p:-1]
        dYt = dY[p:]

        nCols = k + k * (p - 1) + 1
        Z = np.ones((nObs, nCols))
        Z[:, 1:k + 1] = Ylag

        for lag in range(1, p):
            startCol = 1 + k + (lag - 1) * k
            endCol = startCol + k
            if endCol <= nCols:
                Z[:, startCol:endCol] = dY[p - lag:T - 1 - lag]

        reg = 1e-8 * np.eye(nCols)
        try:
            B = np.linalg.solve(Z.T @ Z + reg, Z.T @ dYt)
        except np.linalg.LinAlgError:
            self._fallbackFit(Y)
            return self

        Pi = B[1:k + 1]

        U, S, Vt = np.linalg.svd(Pi, full_matrices=False)
        r = self.rank
        self.beta = Vt[:r].T
        self.alpha = U[:, :r] @ np.diag(S[:r])

        self.intercept = B[0]
        self.gamma = []
        for lag in range(1, p):
            startCol = 1 + k + (lag - 1) * k
            endCol = startCol + k
            if endCol <= nCols:
                self.gamma.append(B[startCol:endCol])

        residuals = dYt - Z @ B
        self.sigma = (residuals.T @ residuals) / nObs

        self._lastLevels = Y[-p - 1:].copy()
        self._lastDiffs = dY[-p:].copy()
        self.fitted = True
        return self

    def _estimateRank(self, Y: np.ndarray) -> int:
        T, k = Y.shape
        dY = np.diff(Y, axis=0)

        nObs = T - 2
        if nObs < k + 2:
            return 1

        Z0 = dY[1:]
        Z1 = Y[1:-1]

        reg = 1e-8

        M0 = np.eye(nObs) - Z0 @ np.linalg.solve(Z0.T @ Z0 + reg * np.eye(Z0.shape[1]), Z0.T)
        R0 = M0 @ Z0
        R1 = M0 @ Z1

        S00 = R0.T @ R0 / nObs
        S11 = R1.T @ R1 / nObs
        S01 = R0.T @ R1 / nObs
        S10 = S01.T

        try:
            S00inv = np.linalg.inv(S00 + reg * np.eye(k))
            S11inv = np.linalg.inv(S11 + reg * np.eye(k))
            eigVals = np.linalg.eigvalsh(S11inv @ S10 @ S00inv @ S01)
            eigVals = np.sort(np.abs(eigVals))[::-1]
        except np.linalg.LinAlgError:
            return 1

        rank = 0
        for i in range(k):
            if eigVals[i] > 0.1:
                rank += 1
            else:
                break

        return max(rank, 1)

    def _fallbackFit(self, Y: np.ndarray):
        T, k = Y.shape
        self._k = k
        self.intercept = np.zeros(k)
        self.alpha = np.zeros((k, 1))
        self.beta = np.zeros((k, 1))
        self.gamma = []
        self.sigma = np.cov(np.diff(Y, axis=0).T) if T > 2 else np.eye(k)
        self._lastLevels = Y[-2:].copy()
        self._lastDiffs = np.diff(Y[-2:], axis=0)
        self.fitted = True

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate multivariate forecasts.

        Parameters
        ----------
        steps : int
            Number of future steps.

        Returns
        -------
        predictions : np.ndarray
            Shape (steps, k).
        lower : np.ndarray
            Shape (steps, k).
        upper : np.ndarray
            Shape (steps, k).
        """
        if not self.fitted:
            raise ValueError("Model not fitted.")

        k = self._k
        levels = list(self._lastLevels)
        diffs = list(self._lastDiffs) if self._lastDiffs is not None else []

        predictions = np.zeros((steps, k))

        for h in range(steps):
            lastLevel = levels[-1]

            ecTerm = self.alpha @ (self.beta.T @ lastLevel) if self.alpha is not None and self.beta is not None else np.zeros(k)

            dY = self.intercept + ecTerm
            for i, g in enumerate(self.gamma):
                diffIdx = len(diffs) - 1 - i
                if diffIdx >= 0:
                    dY = dY + g @ diffs[diffIdx]

            newLevel = lastLevel + dY
            predictions[h] = newLevel
            levels.append(newLevel)
            diffs.append(dY)

        sigDiag = np.sqrt(np.diag(self.sigma)) if self.sigma is not None else np.ones(k)
        horizonScale = np.sqrt(np.arange(1, steps + 1)).reshape(-1, 1)
        margin = 1.96 * sigDiag * horizonScale

        lower = predictions - margin
        upper = predictions + margin

        return predictions, lower, upper
