"""
Time Series Regression

Time series specific regression methods:
- Newey-West HAC standard errors
- Cochrane-Orcutt (AR residual correction)
- Prais-Winsten (Cochrane-Orcutt + first observation correction)
- Granger Causality Test
- Distributed Lag Model

Pure numpy/scipy implementation (no sklearn dependency).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class TSRegressionResult:
    """Regression result"""
    coef: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept: float = 0.0
    stdErrors: np.ndarray = field(default_factory=lambda: np.array([]))
    tStats: np.ndarray = field(default_factory=lambda: np.array([]))
    pValues: np.ndarray = field(default_factory=lambda: np.array([]))
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))
    fittedValues: np.ndarray = field(default_factory=lambda: np.array([]))
    r2: float = 0.0
    adjR2: float = 0.0
    fStatistic: float = 0.0
    fPValue: float = 0.0
    sigma2: float = 0.0
    nObs: int = 0
    nParams: int = 0
    covarianceType: str = 'nonrobust'

    def summary(self) -> str:
        """Result summary"""
        lines = []
        lines.append("=" * 65)
        lines.append(f"  Regression Results ({self.covarianceType} standard errors)")
        lines.append("=" * 65)
        lines.append(f"  Observations: {self.nObs}, Parameters: {self.nParams}")
        lines.append(f"  R^2: {self.r2:.4f}, Adj R^2: {self.adjR2:.4f}")
        lines.append(f"  F-stat: {self.fStatistic:.4f}, p-value: {self.fPValue:.6f}")
        lines.append(f"  Residual std (sigma): {np.sqrt(self.sigma2):.4f}")
        lines.append("-" * 65)
        lines.append(f"  {'Variable':>10} {'Coef':>12} {'Std Err':>12} {'t-stat':>10} {'p-value':>10}")
        lines.append("-" * 65)

        # intercept
        if len(self.stdErrors) > 0:
            lines.append(
                f"  {'intercept':>10} {self.intercept:>12.4f} "
                f"{self.stdErrors[0]:>12.4f} {self.tStats[0]:>10.4f} {self.pValues[0]:>10.4f}"
            )
            for j in range(len(self.coef)):
                lines.append(
                    f"  {f'X{j}':>10} {self.coef[j]:>12.4f} "
                    f"{self.stdErrors[j+1]:>12.4f} {self.tStats[j+1]:>10.4f} "
                    f"{self.pValues[j+1]:>10.4f}"
                )
        lines.append("=" * 65)
        return "\n".join(lines)


@dataclass
class GrangerResult:
    """Granger causality test result"""
    fStatistic: float = 0.0
    pValue: float = 0.0
    optimalLag: int = 1
    aicPerLag: Dict[int, float] = field(default_factory=dict)
    bicPerLag: Dict[int, float] = field(default_factory=dict)
    fStatPerLag: Dict[int, float] = field(default_factory=dict)
    pValuePerLag: Dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Result summary"""
        lines = []
        lines.append("=" * 55)
        lines.append("  Granger Causality Test Results")
        lines.append("=" * 55)
        lines.append(f"  Optimal lag: {self.optimalLag}")
        lines.append(f"  F-statistic: {self.fStatistic:.4f}")
        lines.append(f"  p-value: {self.pValue:.6f}")

        if self.pValue < 0.01:
            lines.append("  Conclusion: Strong Granger causality present (p < 0.01)")
        elif self.pValue < 0.05:
            lines.append("  Conclusion: Granger causality present (p < 0.05)")
        else:
            lines.append("  Conclusion: No Granger causality")

        lines.append("")
        lines.append(f"  {'lag':>5} {'F-stat':>12} {'p-value':>12} {'AIC':>12} {'BIC':>12}")
        lines.append("-" * 55)
        for lag in sorted(self.fStatPerLag.keys()):
            lines.append(
                f"  {lag:>5} {self.fStatPerLag[lag]:>12.4f} "
                f"{self.pValuePerLag[lag]:>12.4f} "
                f"{self.aicPerLag.get(lag, 0):>12.4f} "
                f"{self.bicPerLag.get(lag, 0):>12.4f}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)


class NeweyWestOLS:
    """
    HAC (Heteroscedasticity and Autocorrelation Consistent) standard errors

    Bartlett kernel:
        K(j, L) = 1 - j/(L+1) for j <= L, 0 otherwise

    HAC covariance:
        V_HAC = (X'X)^{-1} S (X'X)^{-1}
        where S = sum_{j=-L}^{L} K(j,L) * Gamma_j
        Gamma_j = (1/n) sum_t (x_t * e_t) * (x_{t-j} * e_{t-j})'

    OLS coefficients remain the same, but standard errors are robustly corrected
    for heteroscedasticity and autocorrelation.

    Parameters
    ----------
    maxLags : int, optional
        Maximum number of lags. If None, computed as floor(4*(n/100)^(2/9))
    fitIntercept : bool
        Whether to include intercept (default: True)
    """

    def __init__(self, maxLags: Optional[int] = None, fitIntercept: bool = True):
        self.maxLags = maxLags
        self.fitIntercept = fitIntercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> TSRegressionResult:
        """
        OLS coefficient estimation + Newey-West HAC standard errors for t-stat/p-value

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable

        Returns
        -------
        TSRegressionResult
        """
        n, p = X.shape

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]

        # OLS estimation
        try:
            XtXinv = np.linalg.inv(Xa.T @ Xa)
        except np.linalg.LinAlgError:
            XtXinv = np.linalg.pinv(Xa.T @ Xa)

        beta = XtXinv @ (Xa.T @ y)
        residuals = y - Xa @ beta

        # Residual variance
        sigma2 = np.sum(residuals ** 2) / max(n - k, 1)

        # Newey-West lag count determination
        L = self.maxLags
        if L is None:
            L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        L = max(L, 1)

        # HAC S matrix computation
        S = self._computeHACMatrix(Xa, residuals, n, k, L)

        # HAC covariance matrix
        hacCov = XtXinv @ S @ XtXinv

        # Standard errors
        stdErrors = np.sqrt(np.maximum(np.diag(hacCov), 0.0))

        # t-statistics, p-values
        tStats = np.where(stdErrors > 1e-15, beta / stdErrors, 0.0)
        pValues = 2.0 * (1.0 - stats.t.cdf(np.abs(tStats), df=max(n - k, 1)))

        # R^2, F-statistic
        fittedValues = Xa @ beta
        ssTot = np.sum((y - np.mean(y)) ** 2)
        ssRes = np.sum(residuals ** 2)

        r2 = 1.0 - ssRes / ssTot if ssTot > 1e-15 else 0.0
        adjR2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

        dfModel = k - 1 if self.fitIntercept else k
        if dfModel > 0 and ssRes > 1e-15:
            ssReg = ssTot - ssRes
            fStat = (ssReg / dfModel) / (ssRes / max(n - k, 1))
            fPValue = 1.0 - stats.f.cdf(fStat, dfModel, max(n - k, 1))
        else:
            fStat = 0.0
            fPValue = 1.0

        if self.fitIntercept:
            intercept = beta[0]
            coef = beta[1:]
        else:
            intercept = 0.0
            coef = beta

        return TSRegressionResult(
            coef=coef,
            intercept=intercept,
            stdErrors=stdErrors,
            tStats=tStats,
            pValues=pValues,
            residuals=residuals,
            fittedValues=fittedValues,
            r2=r2,
            adjR2=adjR2,
            fStatistic=fStat,
            fPValue=fPValue,
            sigma2=sigma2,
            nObs=n,
            nParams=k,
            covarianceType='HAC (Newey-West)',
        )

    def _computeHACMatrix(
        self,
        Xa: np.ndarray,
        residuals: np.ndarray,
        n: int,
        k: int,
        L: int
    ) -> np.ndarray:
        """
        HAC S matrix computation (Bartlett kernel)

        S = Gamma_0 + sum_{j=1}^{L} w_j * (Gamma_j + Gamma_j')
        w_j = 1 - j/(L+1) (Bartlett)
        Gamma_j = (1/n) sum_{t=j+1}^{n} (x_t * e_t) * (x_{t-j} * e_{t-j})'
        """
        # Score vectors: g_t = x_t * e_t
        scores = Xa * residuals[:, np.newaxis]  # (n, k)

        # Gamma_0
        S = (scores.T @ scores) / n

        for j in range(1, L + 1):
            bartlettWeight = 1.0 - j / (L + 1.0)

            # Gamma_j = (1/n) sum_{t=j}^{n-1} scores[t] * scores[t-j]'
            gammaJ = (scores[j:].T @ scores[:-j]) / n

            S += bartlettWeight * (gammaJ + gammaJ.T)

        return S

    def predict(self, X: np.ndarray) -> np.ndarray:
        """This class returns results from fit(), use result.coef/intercept instead"""
        raise NotImplementedError(
            "NeweyWestOLS returns TSRegressionResult from fit(). "
            "Use result.coef and result.intercept for prediction."
        )


class CochraneOrcutt:
    """
    Cochrane-Orcutt iterative estimation

    GLS estimation that corrects for AR(1) autocorrelated residuals.

    Model: y_t = X_t'b + u_t, u_t = rho*u_{t-1} + e_t

    Iterative algorithm:
        1. Estimate beta via OLS
        2. Estimate rho from residuals (AR(1) coefficient)
        3. Transform y* = y_t - rho*y_{t-1}, X* = X_t - rho*X_{t-1}
        4. OLS on y*, X*
        5. Repeat until convergence

    Note: First observation is dropped (differs from Prais-Winsten)

    Parameters
    ----------
    maxIter : int
        Maximum iterations (default: 100)
    tol : float
        Convergence tolerance (default: 1e-6)
    fitIntercept : bool
        Whether to include intercept (default: True)
    """

    def __init__(self, maxIter: int = 100, tol: float = 1e-6, fitIntercept: bool = True):
        self.maxIter = maxIter
        self.tol = tol
        self.fitIntercept = fitIntercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[TSRegressionResult, float]:
        """
        Perform Cochrane-Orcutt iterative estimation

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable

        Returns
        -------
        Tuple[TSRegressionResult, float]
            (regression result, estimated rho)
        """
        n, p = X.shape

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]

        # Step 1: Initial OLS
        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(k)

        rho = 0.0

        for iteration in range(self.maxIter):
            betaOld = beta.copy()
            rhoOld = rho

            # Step 2: Estimate rho from residuals
            residuals = y - Xa @ beta
            rho = self._estimateRho(residuals)

            # Step 3: Transform data (drop first observation)
            yTransformed = y[1:] - rho * y[:-1]
            XaTransformed = Xa[1:] - rho * Xa[:-1]

            # Step 4: OLS on transformed data
            try:
                beta = np.linalg.lstsq(XaTransformed, yTransformed, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            # Step 5: Check convergence
            betaDiff = np.max(np.abs(beta - betaOld))
            rhoDiff = abs(rho - rhoOld)
            if betaDiff < self.tol and rhoDiff < self.tol:
                break

        # Compute final results
        residuals = y - Xa @ beta
        fittedValues = Xa @ beta
        sigma2 = np.sum((yTransformed - XaTransformed @ beta) ** 2) / max(n - 1 - k, 1)

        # Standard errors (based on transformed data)
        try:
            XtXinv = np.linalg.inv(XaTransformed.T @ XaTransformed)
        except np.linalg.LinAlgError:
            XtXinv = np.linalg.pinv(XaTransformed.T @ XaTransformed)

        covBeta = sigma2 * XtXinv
        stdErrors = np.sqrt(np.maximum(np.diag(covBeta), 0.0))

        tStats = np.where(stdErrors > 1e-15, beta / stdErrors, 0.0)
        pValues = 2.0 * (1.0 - stats.t.cdf(np.abs(tStats), df=max(n - 1 - k, 1)))

        ssTot = np.sum((y - np.mean(y)) ** 2)
        ssRes = np.sum(residuals ** 2)
        r2 = 1.0 - ssRes / ssTot if ssTot > 1e-15 else 0.0
        adjR2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

        dfModel = k - 1 if self.fitIntercept else k
        if dfModel > 0 and ssRes > 1e-15:
            ssReg = ssTot - ssRes
            fStat = (ssReg / dfModel) / (ssRes / max(n - k, 1))
            fPValue = 1.0 - stats.f.cdf(fStat, dfModel, max(n - k, 1))
        else:
            fStat = 0.0
            fPValue = 1.0

        if self.fitIntercept:
            intercept = beta[0]
            coef = beta[1:]
        else:
            intercept = 0.0
            coef = beta

        result = TSRegressionResult(
            coef=coef,
            intercept=intercept,
            stdErrors=stdErrors,
            tStats=tStats,
            pValues=pValues,
            residuals=residuals,
            fittedValues=fittedValues,
            r2=r2,
            adjR2=adjR2,
            fStatistic=fStat,
            fPValue=fPValue,
            sigma2=sigma2,
            nObs=n,
            nParams=k,
            covarianceType='Cochrane-Orcutt GLS',
        )

        return result, float(rho)

    @staticmethod
    def _estimateRho(residuals: np.ndarray) -> float:
        """Estimate AR(1) coefficient from residuals"""
        n = len(residuals)
        if n < 3:
            return 0.0
        denom = np.sum(residuals[:-1] ** 2)
        if denom < 1e-15:
            return 0.0
        rho = np.sum(residuals[1:] * residuals[:-1]) / denom
        # Constrain rho to (-1, 1) interval
        return float(np.clip(rho, -0.999, 0.999))


class PraisWinsten:
    """
    Prais-Winsten estimation (Cochrane-Orcutt + first observation correction)

    Same as Cochrane-Orcutt but preserves the first observation for better efficiency.

    First observation transform:
        y*_1 = sqrt(1 - rho^2) * y_1
        X*_1 = sqrt(1 - rho^2) * X_1

    Remaining:
        y*_t = y_t - rho * y_{t-1}
        X*_t = X_t - rho * X_{t-1}

    Parameters
    ----------
    maxIter : int
        Maximum iterations (default: 100)
    tol : float
        Convergence tolerance (default: 1e-6)
    fitIntercept : bool
        Whether to include intercept (default: True)
    """

    def __init__(self, maxIter: int = 100, tol: float = 1e-6, fitIntercept: bool = True):
        self.maxIter = maxIter
        self.tol = tol
        self.fitIntercept = fitIntercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[TSRegressionResult, float]:
        """
        Perform Prais-Winsten iterative estimation

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable

        Returns
        -------
        Tuple[TSRegressionResult, float]
            (regression result, estimated rho)
        """
        n, p = X.shape

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]

        # Initial OLS
        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(k)

        rho = 0.0

        for iteration in range(self.maxIter):
            betaOld = beta.copy()
            rhoOld = rho

            # Estimate rho from residuals
            residuals = y - Xa @ beta
            rho = self._estimateRho(residuals)

            # Prais-Winsten transformation
            yTransformed, XaTransformed = self._transform(y, Xa, rho)

            # OLS on transformed data
            try:
                beta = np.linalg.lstsq(XaTransformed, yTransformed, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            # Check convergence
            betaDiff = np.max(np.abs(beta - betaOld))
            rhoDiff = abs(rho - rhoOld)
            if betaDiff < self.tol and rhoDiff < self.tol:
                break

        # Compute final results
        yTransformed, XaTransformed = self._transform(y, Xa, rho)
        residTransformed = yTransformed - XaTransformed @ beta
        sigma2 = np.sum(residTransformed ** 2) / max(n - k, 1)

        residuals = y - Xa @ beta
        fittedValues = Xa @ beta

        try:
            XtXinv = np.linalg.inv(XaTransformed.T @ XaTransformed)
        except np.linalg.LinAlgError:
            XtXinv = np.linalg.pinv(XaTransformed.T @ XaTransformed)

        covBeta = sigma2 * XtXinv
        stdErrors = np.sqrt(np.maximum(np.diag(covBeta), 0.0))

        tStats = np.where(stdErrors > 1e-15, beta / stdErrors, 0.0)
        pValues = 2.0 * (1.0 - stats.t.cdf(np.abs(tStats), df=max(n - k, 1)))

        ssTot = np.sum((y - np.mean(y)) ** 2)
        ssRes = np.sum(residuals ** 2)
        r2 = 1.0 - ssRes / ssTot if ssTot > 1e-15 else 0.0
        adjR2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

        dfModel = k - 1 if self.fitIntercept else k
        if dfModel > 0 and ssRes > 1e-15:
            ssReg = ssTot - ssRes
            fStat = (ssReg / dfModel) / (ssRes / max(n - k, 1))
            fPValue = 1.0 - stats.f.cdf(fStat, dfModel, max(n - k, 1))
        else:
            fStat = 0.0
            fPValue = 1.0

        if self.fitIntercept:
            intercept = beta[0]
            coef = beta[1:]
        else:
            intercept = 0.0
            coef = beta

        result = TSRegressionResult(
            coef=coef,
            intercept=intercept,
            stdErrors=stdErrors,
            tStats=tStats,
            pValues=pValues,
            residuals=residuals,
            fittedValues=fittedValues,
            r2=r2,
            adjR2=adjR2,
            fStatistic=fStat,
            fPValue=fPValue,
            sigma2=sigma2,
            nObs=n,
            nParams=k,
            covarianceType='Prais-Winsten GLS',
        )

        return result, float(rho)

    @staticmethod
    def _transform(
        y: np.ndarray,
        Xa: np.ndarray,
        rho: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform Prais-Winsten transformation

        First observation: sqrt(1-rho^2) * original value
        Remaining: x_t - rho * x_{t-1}
        """
        n = len(y)
        yT = np.zeros(n)
        XaT = np.zeros_like(Xa)

        # First observation transformation
        sqrtFactor = np.sqrt(max(1.0 - rho ** 2, 1e-15))
        yT[0] = sqrtFactor * y[0]
        XaT[0] = sqrtFactor * Xa[0]

        # Remaining transformation
        yT[1:] = y[1:] - rho * y[:-1]
        XaT[1:] = Xa[1:] - rho * Xa[:-1]

        return yT, XaT

    @staticmethod
    def _estimateRho(residuals: np.ndarray) -> float:
        """Estimate AR(1) coefficient from residuals"""
        n = len(residuals)
        if n < 3:
            return 0.0
        denom = np.sum(residuals[:-1] ** 2)
        if denom < 1e-15:
            return 0.0
        rho = np.sum(residuals[1:] * residuals[:-1]) / denom
        return float(np.clip(rho, -0.999, 0.999))


class GrangerCausality:
    """
    Granger causality test

    H0: X does not Granger-cause Y

    Algorithm:
        1. Restricted model: Y_t = a + sum(b_i * Y_{t-i})
        2. Unrestricted model: Y_t = a + sum(b_i * Y_{t-i}) + sum(c_j * X_{t-j})
        3. F = ((SSR_r - SSR_u) / q) / (SSR_u / (n - k))

    Automatic optimal lag selection via AIC/BIC.

    Parameters
    ----------
    maxLag : int
        Maximum number of lags (default: 4)
    """

    def __init__(self, maxLag: int = 4):
        if maxLag < 1:
            raise ValueError(f"maxLag must be >= 1: {maxLag}")
        self.maxLag = maxLag

    def test(self, y: np.ndarray, x: np.ndarray) -> GrangerResult:
        """
        Perform Granger causality test

        H0: x does not Granger-cause y

        Parameters
        ----------
        y : np.ndarray, shape (T,)
            Dependent time series (effect)
        x : np.ndarray, shape (T,)
            Independent time series (candidate cause)

        Returns
        -------
        GrangerResult
        """
        y = np.asarray(y, dtype=float).ravel()
        x = np.asarray(x, dtype=float).ravel()

        if len(y) != len(x):
            raise ValueError(f"y({len(y)}) and x({len(x)}) have different lengths")

        T = len(y)
        if T <= 2 * self.maxLag + 2:
            raise ValueError(
                f"Data length({T}) is too short for maximum lag({self.maxLag})"
            )

        aicPerLag = {}
        bicPerLag = {}
        fStatPerLag = {}
        pValuePerLag = {}

        bestAIC = np.inf
        optimalLag = 1

        for lag in range(1, self.maxLag + 1):
            # Effective data range: t = lag, lag+1, ..., T-1
            nEff = T - lag

            if nEff <= 2 * lag + 2:
                continue

            # Dependent variable
            yTarget = y[lag:]

            # Restricted model design matrix: intercept + Y lags
            XrParts = [np.ones((nEff, 1))]
            for j in range(1, lag + 1):
                XrParts.append(y[lag - j: T - j].reshape(-1, 1))
            Xr = np.hstack(XrParts)

            # Unrestricted model design matrix: intercept + Y lags + X lags
            XuParts = [Xr]
            for j in range(1, lag + 1):
                XuParts.append(x[lag - j: T - j].reshape(-1, 1))
            Xu = np.hstack(XuParts)

            # Restricted model OLS
            try:
                betaR = np.linalg.lstsq(Xr, yTarget, rcond=None)[0]
                ssrR = np.sum((yTarget - Xr @ betaR) ** 2)
            except np.linalg.LinAlgError:
                continue

            # Unrestricted model OLS
            try:
                betaU = np.linalg.lstsq(Xu, yTarget, rcond=None)[0]
                ssrU = np.sum((yTarget - Xu @ betaU) ** 2)
            except np.linalg.LinAlgError:
                continue

            # F test
            q = lag  # Number of X lag variables
            kU = Xu.shape[1]  # Unrestricted model parameter count
            dfResid = nEff - kU

            if dfResid <= 0 or ssrU < 1e-15:
                continue

            fStat = ((ssrR - ssrU) / q) / (ssrU / dfResid)
            pValue = 1.0 - stats.f.cdf(fStat, q, dfResid)

            # AIC/BIC (based on unrestricted model)
            if ssrU < 1e-15:
                logLik = 0.0
            else:
                logLik = -nEff / 2.0 * (np.log(2 * np.pi) + np.log(ssrU / nEff) + 1)
            aic = -2 * logLik + 2 * kU
            bic = -2 * logLik + np.log(nEff) * kU

            fStatPerLag[lag] = float(fStat)
            pValuePerLag[lag] = float(pValue)
            aicPerLag[lag] = float(aic)
            bicPerLag[lag] = float(bic)

            if aic < bestAIC:
                bestAIC = aic
                optimalLag = lag

        # Results at optimal lag
        bestFStat = fStatPerLag.get(optimalLag, 0.0)
        bestPValue = pValuePerLag.get(optimalLag, 1.0)

        return GrangerResult(
            fStatistic=bestFStat,
            pValue=bestPValue,
            optimalLag=optimalLag,
            aicPerLag=aicPerLag,
            bicPerLag=bicPerLag,
            fStatPerLag=fStatPerLag,
            pValuePerLag=pValuePerLag,
        )


class DistributedLagModel:
    """
    Distributed Lag Model

    Y_t = a + sum_{j=0}^{q} beta_j * X_{t-j} + epsilon_t

    Simultaneously estimates the effect of current and past values (lags) of X on Y.
    Enables analysis of long-run multiplier and lag structure.

    Parameters
    ----------
    maxLag : int
        Maximum lag (default: 4)
    fitIntercept : bool
        Whether to include intercept (default: True)
    """

    def __init__(self, maxLag: int = 4, fitIntercept: bool = True):
        self.maxLag = maxLag
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0
        self.lagCoefficients = None
        self.longRunMultiplier = 0.0

    def fit(self, y: np.ndarray, x: np.ndarray) -> TSRegressionResult:
        """
        Distributed lag model estimation

        Parameters
        ----------
        y : np.ndarray, shape (T,)
            Dependent time series
        x : np.ndarray, shape (T,) or (T, p)
            Independent time series (if 1D, treated as single variable)

        Returns
        -------
        TSRegressionResult
        """
        y = np.asarray(y, dtype=float).ravel()
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        T, px = x.shape
        if len(y) != T:
            raise ValueError(f"y({len(y)}) and x({T}) have different lengths")

        nEff = T - self.maxLag
        if nEff <= self.maxLag * px + 2:
            raise ValueError("Data length is too short for the number of lags")

        # Design matrix construction: X_{t}, X_{t-1}, ..., X_{t-q}
        parts = []
        if self.fitIntercept:
            parts.append(np.ones((nEff, 1)))

        for j in range(self.maxLag + 1):
            start = self.maxLag - j
            end = T - j
            parts.append(x[start:end])

        Xa = np.hstack(parts)
        yTarget = y[self.maxLag:]

        k = Xa.shape[1]

        # OLS estimation
        try:
            XtXinv = np.linalg.inv(Xa.T @ Xa)
        except np.linalg.LinAlgError:
            XtXinv = np.linalg.pinv(Xa.T @ Xa)

        beta = XtXinv @ (Xa.T @ yTarget)

        residuals = yTarget - Xa @ beta
        fittedValues = Xa @ beta
        sigma2 = np.sum(residuals ** 2) / max(nEff - k, 1)

        covBeta = sigma2 * XtXinv
        stdErrors = np.sqrt(np.maximum(np.diag(covBeta), 0.0))

        tStats = np.where(stdErrors > 1e-15, beta / stdErrors, 0.0)
        pValues = 2.0 * (1.0 - stats.t.cdf(np.abs(tStats), df=max(nEff - k, 1)))

        ssTot = np.sum((yTarget - np.mean(yTarget)) ** 2)
        ssRes = np.sum(residuals ** 2)
        r2 = 1.0 - ssRes / ssTot if ssTot > 1e-15 else 0.0
        adjR2 = 1.0 - (1.0 - r2) * (nEff - 1) / max(nEff - k, 1)

        dfModel = k - 1 if self.fitIntercept else k
        if dfModel > 0 and ssRes > 1e-15:
            ssReg = ssTot - ssRes
            fStat = (ssReg / dfModel) / (ssRes / max(nEff - k, 1))
            fPValue = 1.0 - stats.f.cdf(fStat, dfModel, max(nEff - k, 1))
        else:
            fStat = 0.0
            fPValue = 1.0

        # Separate lag coefficients
        if self.fitIntercept:
            self.intercept = beta[0]
            self.lagCoefficients = beta[1:]
        else:
            self.intercept = 0.0
            self.lagCoefficients = beta.copy()

        self.coef = self.lagCoefficients.copy()
        self.longRunMultiplier = float(np.sum(self.lagCoefficients))

        # Full residuals (original length)
        fullResiduals = np.full(T, np.nan)
        fullFitted = np.full(T, np.nan)
        fullResiduals[self.maxLag:] = residuals
        fullFitted[self.maxLag:] = fittedValues

        result = TSRegressionResult(
            coef=self.lagCoefficients,
            intercept=self.intercept,
            stdErrors=stdErrors,
            tStats=tStats,
            pValues=pValues,
            residuals=fullResiduals,
            fittedValues=fullFitted,
            r2=r2,
            adjR2=adjR2,
            fStatistic=fStat,
            fPValue=fPValue,
            sigma2=sigma2,
            nObs=nEff,
            nParams=k,
            covarianceType='OLS (Distributed Lag)',
        )

        return result

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Return predictions

        Parameters
        ----------
        x : np.ndarray, shape (T, p) or (T,)
            Independent time series (requires minimum length of maxLag+1)

        Returns
        -------
        np.ndarray
            Predictions (length: T - maxLag)
        """
        if self.lagCoefficients is None:
            raise ValueError("Model has not been fitted yet.")

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        T = x.shape[0]
        nEff = T - self.maxLag

        if nEff <= 0:
            raise ValueError("Data length must be greater than the number of lags")

        parts = []
        if self.fitIntercept:
            parts.append(np.ones((nEff, 1)))

        for j in range(self.maxLag + 1):
            start = self.maxLag - j
            end = T - j
            parts.append(x[start:end])

        Xa = np.hstack(parts)

        if self.fitIntercept:
            allCoef = np.concatenate([[self.intercept], self.lagCoefficients])
        else:
            allCoef = self.lagCoefficients

        return Xa @ allCoef
