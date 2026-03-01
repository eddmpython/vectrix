"""
Statistical Inference Engine for Regression

Complete regression statistical inference at statsmodels level:
- R-squared, Adjusted R-squared
- Standard errors, t-statistics, p-values
- F-statistic
- Confidence intervals (coefficients and predictions)
- AIC / BIC / Log-likelihood
- Residuals (raw, standardized, studentized)
- Durbin-Watson statistic
- Condition Number

Pure numpy/scipy only. No sklearn dependency.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats


@dataclass
class RegressionResult:
    """
    Regression result object - statsmodels.OLSResults level

    All statistics computed from OLS estimation accessible in one place.
    summary() method provides statsmodels-style formatted text output.
    """

    # ── Coefficients ──
    coefficients: np.ndarray          # beta (including intercept)
    standardErrors: np.ndarray        # SE(beta)
    tValues: np.ndarray               # t = beta / SE(beta)
    pValues: np.ndarray               # P(|t| > t_obs)
    confidenceIntervals: np.ndarray   # [lower, upper] per coef, shape (k, 2)

    # ── Goodness of Fit ──
    rSquared: float
    adjustedRSquared: float
    fStatistic: float
    fPValue: float

    # ── Information Criteria ──
    logLikelihood: float
    aic: float
    bic: float

    # ── Residuals ──
    residuals: np.ndarray              # raw residuals
    standardizedResiduals: np.ndarray  # standardized
    studentizedResiduals: np.ndarray   # studentized (leave-one-out)
    fittedValues: np.ndarray

    # ── Matrix Information ──
    hatMatrix: np.ndarray              # diag(H) = leverage
    covarianceMatrix: np.ndarray       # Var(beta)
    conditionNumber: float

    # ── Meta ──
    nObs: int
    nParams: int                       # Including intercept
    degreesOfFreedom: int              # n - k
    featureNames: Optional[List[str]] = None
    sigma: float = 0.0                 # residual standard error
    ssRes: float = 0.0
    ssTot: float = 0.0
    ssReg: float = 0.0
    durbinWatson: float = 0.0

    def summary(self, title: str = "OLS Regression Results") -> str:
        """
        statsmodels-style formatted text output

        Returns:
            Regression result summary string (78 chars wide for monospace font)
        """
        width = 78
        halfWidth = width // 2

        # Determine feature names
        if self.featureNames is not None:
            names = list(self.featureNames)
        else:
            names = [f"x{i}" for i in range(self.nParams)]

        # ── Top Header ──
        lines: List[str] = []
        lines.append(title.center(width))
        lines.append("=" * width)

        # ── Model Info Block (left/right 2 columns) ──
        def _leftRight(leftLabel: str, leftVal: str,
                       rightLabel: str, rightVal: str) -> str:
            """Format left/right info in 78-char width"""
            leftPart = f"{leftLabel:<22s}{leftVal:>17s}"
            rightPart = f"   {rightLabel:<22s}{rightVal:>14s}"
            return leftPart + rightPart

        fPStr = f"{self.fPValue:.2e}" if self.fPValue < 0.001 else f"{self.fPValue:.4g}"

        lines.append(_leftRight(
            "Dep. Variable:", "y",
            "R-squared:", f"{self.rSquared:.3f}"))
        lines.append(_leftRight(
            "Model:", "OLS",
            "Adj. R-squared:", f"{self.adjustedRSquared:.3f}"))
        lines.append(_leftRight(
            "Method:", "Least Squares",
            "F-statistic:", f"{self.fStatistic:.4g}"))
        lines.append(_leftRight(
            "No. Observations:", f"{self.nObs:d}",
            "Prob (F-statistic):", fPStr))
        lines.append(_leftRight(
            "Df Residuals:", f"{self.degreesOfFreedom:d}",
            "Log-Likelihood:", f"{self.logLikelihood:.2f}"))

        dfModel = self.nParams - 1 if self.nParams > 1 else self.nParams
        lines.append(_leftRight(
            "Df Model:", f"{dfModel:d}",
            "AIC:", f"{self.aic:.1f}"))
        # BIC on right side only
        leftPad = " " * 39
        rightPart = f"   {'BIC:':<22s}{self.bic:>14.1f}"
        lines.append(leftPad + rightPart)
        lines.append("=" * width)

        # ── Coefficient Table Header ──
        alpha = 0.05  # Default confidence interval
        halfAlpha = alpha / 2.0
        lowerLabel = f"[{halfAlpha:.3f}"
        upperLabel = f"{1 - halfAlpha:.3f}]"

        header = (f"{'':>15s} {'coef':>10s} {'std err':>10s} "
                  f"{'t':>10s} {'P>|t|':>10s} {lowerLabel:>10s} {upperLabel:>10s}")
        lines.append(header)
        lines.append("-" * width)

        # ── Coefficient Rows ──
        for i in range(self.nParams):
            name = names[i] if i < len(names) else f"x{i}"
            coef = self.coefficients[i]
            se = self.standardErrors[i]
            t = self.tValues[i]
            p = self.pValues[i]
            ciLow = self.confidenceIntervals[i, 0]
            ciHigh = self.confidenceIntervals[i, 1]

            pStr = f"{p:.3f}" if p >= 0.0005 else f"{p:.3e}"

            row = (f"{name:>15s} {coef:>10.4f} {se:>10.3f} "
                   f"{t:>10.3f} {pStr:>10s} {ciLow:>10.3f} {ciHigh:>10.3f}")
            lines.append(row)

        lines.append("=" * width)

        # ── Bottom Diagnostic Statistics ──
        # Omnibus test (normality)
        if len(self.residuals) >= 8:
            try:
                omnibusStatistic, omnibusPValue = stats.normaltest(self.residuals)
            except Exception:
                omnibusStatistic, omnibusPValue = float('nan'), float('nan')
        else:
            omnibusStatistic, omnibusPValue = float('nan'), float('nan')

        lines.append(_leftRight(
            "Omnibus:", f"{omnibusStatistic:.3f}",
            "Durbin-Watson:", f"{self.durbinWatson:.3f}"))
        lines.append(_leftRight(
            "Prob(Omnibus):", f"{omnibusPValue:.3f}",
            "Condition No.:", f"{self.conditionNumber:.2f}"))
        lines.append("=" * width)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Simple string representation"""
        return (
            f"<RegressionResult nObs={self.nObs} nParams={self.nParams} "
            f"R2={self.rSquared:.4f} adjR2={self.adjustedRSquared:.4f}>"
        )


class OLSInference:
    """
    OLS Statistical Inference Engine

    Pure numpy/scipy implementation providing statsmodels OLS-level statistical inference.
    Uses SVD-based pseudo-inverse as fallback for numerical stability.

    Usage:
        >>> engine = OLSInference()
        >>> result = engine.fit(X, y)
        >>> print(result.summary())
        >>> result.pValues       # p-value for each coefficient
        >>> result.rSquared      # coefficient of determination
        >>> yHat, lower, upper = engine.predict(Xnew, interval='prediction')
    """

    def __init__(self, fitIntercept: bool = True, alpha: float = 0.05):
        """
        Initialize OLS inference engine

        Args:
            fitIntercept: Whether to include intercept (True adds a column of 1s to X)
            alpha: Default significance level (1 - alpha), e.g. 0.05 -> 95% confidence interval
        """
        self.fitIntercept = fitIntercept
        self.alpha = alpha

        # Internal state after fit()
        self._Xa: Optional[np.ndarray] = None     # augmented design matrix
        self._beta: Optional[np.ndarray] = None
        self._XtXinv: Optional[np.ndarray] = None  # (X'X)^{-1}
        self._sigma: float = 0.0                    # residual std error
        self._result: Optional[RegressionResult] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            featureNames: Optional[List[str]] = None) -> RegressionResult:
        """
        OLS regression fit + complete statistical inference computation

        Computation steps:
          1. beta = (X'X)^{-1} X'y
          2. e = y - X*beta
          3. s^2 = e'e / (n-k)
          4. Var(beta) = s^2 * (X'X)^{-1}
          5. SE = sqrt(diag(Var(beta)))
          6. t = beta / SE
          7. p = 2 * P(T > |t|, df=n-k)
          8. R^2 = 1 - SS_res / SS_tot
          9. F = (SS_reg / (k-1)) / (SS_res / (n-k))
          10. AIC = n*ln(SS_res/n) + 2k
          11. BIC = n*ln(SS_res/n) + k*ln(n)

        Args:
            X: Design matrix, shape (n, p). Without intercept column.
            y: Dependent variable vector, shape (n,)
            featureNames: Feature name list (optional). May include intercept.

        Returns:
            RegressionResult object

        Raises:
            ValueError: When input dimensions mismatch or insufficient samples
        """
        # ── Input Validation ──
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        if n != y.shape[0]:
            raise ValueError(
                f"Row count mismatch between X and y: X.shape[0]={n}, y.shape[0]={y.shape[0]}"
            )

        # ── Add Intercept ──
        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n, dtype=np.float64), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]  # Parameter count including intercept
        df = n - k        # Residual degrees of freedom

        if df <= 0:
            raise ValueError(
                f"Degrees of freedom <= 0: n={n}, k={k}. "
                f"At least {k + 1} observations are required."
            )

        # ── Feature Names ──
        if featureNames is not None:
            if self.fitIntercept and len(featureNames) == p:
                names = ["const"] + list(featureNames)
            elif len(featureNames) == k:
                names = list(featureNames)
            else:
                names = ["const"] + [f"x{i + 1}" for i in range(p)] if self.fitIntercept \
                    else [f"x{i + 1}" for i in range(p)]
        else:
            if self.fitIntercept:
                names = ["const"] + [f"x{i + 1}" for i in range(p)]
            else:
                names = [f"x{i + 1}" for i in range(p)]

        # ── Step 1: OLS estimation (lstsq + SVD fallback for numerical stability) ──
        XtX = Xa.T @ Xa

        try:
            # Try Cholesky decomposition (fastest and most stable)
            L = np.linalg.cholesky(XtX)
            XtXinv = np.linalg.inv(L.T) @ np.linalg.inv(L)
            beta = XtXinv @ (Xa.T @ y)
        except np.linalg.LinAlgError:
            # SVD pseudo-inverse fallback
            XtXinv = np.linalg.pinv(XtX)
            beta = XtXinv @ (Xa.T @ y)

        # ── Step 2: Residuals ──
        yHat = Xa @ beta
        residuals = y - yHat

        # ── Step 3: Residual variance (s^2) ──
        ssRes = float(residuals @ residuals)
        s2 = ssRes / df
        sigma = np.sqrt(s2)

        # ── Step 4: Covariance matrix Var(beta) ──
        covBeta = s2 * XtXinv

        # ── Step 5: Standard errors ──
        seRaw = np.diag(covBeta)
        # Numerical stability: prevent very small negative values
        seRaw = np.maximum(seRaw, 0.0)
        standardErrors = np.sqrt(seRaw)

        # ── Step 6: t-statistics ──
        # Prevent division by zero
        safeStdErrors = np.where(standardErrors > 1e-15, standardErrors, 1e-15)
        tValues = beta / safeStdErrors

        # ── Step 7: p-values (two-sided test) ──
        pValues = 2.0 * stats.t.sf(np.abs(tValues), df)

        # ── Step 8: R-squared ──
        yMean = np.mean(y)
        ssTot = float(np.sum((y - yMean) ** 2))
        ssReg = ssTot - ssRes

        if ssTot > 1e-15:
            rSquared = 1.0 - ssRes / ssTot
        else:
            rSquared = 0.0

        # Adjusted R-squared
        if n - k > 0 and ssTot > 1e-15:
            adjustedRSquared = 1.0 - (ssRes / df) / (ssTot / (n - 1))
        else:
            adjustedRSquared = 0.0

        # ── Step 9: F-statistic ──
        # dfModel = k - 1 (regression parameters excluding intercept)
        dfModel = k - 1 if self.fitIntercept else k
        if dfModel > 0 and df > 0 and ssRes > 1e-15:
            fStatistic = (ssReg / dfModel) / (ssRes / df)
            fPValue = float(stats.f.sf(fStatistic, dfModel, df))
        else:
            fStatistic = 0.0
            fPValue = 1.0

        # ── Step 10: Information criteria ──
        # Log-likelihood (normal distribution assumption)
        # L = -(n/2)*ln(2*pi) - (n/2)*ln(s^2_ML) - n/2
        # where s^2_ML = SS_res / n
        if n > 0 and ssRes > 0:
            s2ML = ssRes / n
            logLikelihood = -0.5 * n * (np.log(2.0 * np.pi) + np.log(s2ML) + 1.0)
        else:
            logLikelihood = 0.0

        # AIC = -2*logL + 2*k
        aic = -2.0 * logLikelihood + 2.0 * k
        # BIC = -2*logL + k*ln(n)
        bic = -2.0 * logLikelihood + k * np.log(n)

        # ── Confidence Intervals ──
        tCrit = stats.t.ppf(1.0 - self.alpha / 2.0, df)
        ciLower = beta - tCrit * standardErrors
        ciUpper = beta + tCrit * standardErrors
        confidenceIntervals = np.column_stack([ciLower, ciUpper])

        # ── Hat Matrix (leverage) ──
        # H = X (X'X)^{-1} X'
        # leverage h_ii = diag(H)
        # Efficiently compute only h_ii: h_ii = x_i' (X'X)^{-1} x_i
        hatDiag = np.sum((Xa @ XtXinv) * Xa, axis=1)

        # ── Residual Types ──
        # Standardized residuals: e_i / (s * sqrt(1 - h_ii))
        denomStd = sigma * np.sqrt(np.maximum(1.0 - hatDiag, 1e-15))
        standardizedResiduals = residuals / denomStd

        # External (leave-one-out) studentized residuals
        # s_{(i)}^2 = ((n-k)*s^2 - e_i^2 / (1 - h_ii)) / (n-k-1)
        if df > 1:
            eLoo = residuals ** 2 / np.maximum(1.0 - hatDiag, 1e-15)
            s2Loo = (df * s2 - eLoo) / (df - 1)
            # Numerical stability: prevent negatives
            s2Loo = np.maximum(s2Loo, 1e-15)
            sLoo = np.sqrt(s2Loo)
            studentizedResiduals = residuals / (sLoo * np.sqrt(np.maximum(1.0 - hatDiag, 1e-15)))
        else:
            # Leave-one-out not possible when df = 1
            studentizedResiduals = standardizedResiduals.copy()

        # ── Condition Number ──
        sv = np.linalg.svd(Xa, compute_uv=False)
        if sv[-1] > 1e-10:
            conditionNumber = float(sv[0] / sv[-1])
        else:
            conditionNumber = float('inf')

        # ── Durbin-Watson Statistic ──
        # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        if ssRes > 1e-15 and n > 1:
            diffResiduals = np.diff(residuals)
            durbinWatson = float(np.sum(diffResiduals ** 2) / ssRes)
        else:
            durbinWatson = 0.0

        # ── Save internal state (used in predict) ──
        self._Xa = Xa
        self._beta = beta
        self._XtXinv = XtXinv
        self._sigma = sigma

        # ── Create result object ──
        result = RegressionResult(
            coefficients=beta,
            standardErrors=standardErrors,
            tValues=tValues,
            pValues=pValues,
            confidenceIntervals=confidenceIntervals,
            rSquared=rSquared,
            adjustedRSquared=adjustedRSquared,
            fStatistic=fStatistic,
            fPValue=fPValue,
            logLikelihood=logLikelihood,
            aic=aic,
            bic=bic,
            residuals=residuals,
            standardizedResiduals=standardizedResiduals,
            studentizedResiduals=studentizedResiduals,
            fittedValues=yHat,
            hatMatrix=hatDiag,
            covarianceMatrix=covBeta,
            conditionNumber=conditionNumber,
            nObs=n,
            nParams=k,
            degreesOfFreedom=df,
            featureNames=names,
            sigma=sigma,
            ssRes=ssRes,
            ssTot=ssTot,
            ssReg=ssReg,
            durbinWatson=durbinWatson,
        )

        self._result = result
        return result

    def predict(self, X: np.ndarray,
                interval: str = 'none',
                alpha: float = 0.05
                ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prediction on new data + optional confidence/prediction intervals

        Confidence interval: uncertainty of mean response
            y_hat +/- t_{alpha/2, df} * s * sqrt(x' (X'X)^{-1} x)

        Prediction interval: uncertainty of individual response
            y_hat +/- t_{alpha/2, df} * s * sqrt(1 + x' (X'X)^{-1} x)

        Args:
            X: New design matrix, shape (m, p). Without intercept column.
            interval: One of 'none', 'confidence', 'prediction'
            alpha: Significance level for interval (default 0.05 -> 95% interval)

        Returns:
            (yPred, lower, upper) tuple.
            If interval='none', lower and upper are None.

        Raises:
            RuntimeError: If fit() has not been called
            ValueError: If interval argument is invalid
        """
        if self._beta is None or self._XtXinv is None:
            raise RuntimeError(
                "fit() must be called before predict()."
            )

        validIntervals = ('none', 'confidence', 'prediction')
        if interval not in validIntervals:
            raise ValueError(
                f"interval must be one of {validIntervals}. "
                f"Got: '{interval}'"
            )

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Add intercept
        m = X.shape[0]
        if self.fitIntercept:
            Xa = np.column_stack([np.ones(m, dtype=np.float64), X])
        else:
            Xa = X.copy()

        # Predictions
        yPred = Xa @ self._beta

        if interval == 'none':
            return yPred, None, None

        # Interval computation
        df = self._result.degreesOfFreedom
        tCrit = stats.t.ppf(1.0 - alpha / 2.0, df)

        # x' (X'X)^{-1} x for each row
        # Efficient computation: sum((Xa @ XtXinv) * Xa, axis=1)
        leverage = np.sum((Xa @ self._XtXinv) * Xa, axis=1)

        if interval == 'confidence':
            # Standard error of mean response
            seInterval = self._sigma * np.sqrt(leverage)
        else:  # prediction
            # Standard error of individual response
            seInterval = self._sigma * np.sqrt(1.0 + leverage)

        lower = yPred - tCrit * seInterval
        upper = yPred + tCrit * seInterval

        return yPred, lower, upper
