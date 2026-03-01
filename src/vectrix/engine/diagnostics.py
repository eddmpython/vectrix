"""
Forecast Model Residual Diagnostics

Ljung-Box, Jarque-Bera, ARCH tests, etc.
Verifies whether forecast model residuals are white noise

Evaluates the quality of time series forecast model residuals
to determine if the model adequately captures data structure.
Non-white-noise residuals suggest model improvement is needed.

References:
- Ljung & Box (1978)
- Jarque & Bera (1987)
- Engle (1982) ARCH test
- Durbin & Watson (1950, 1951)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.stats import chi2


@dataclass
class ForecastDiagnosticsResult:
    """Forecast residual diagnostics result"""

    isWhiteNoise: bool = True
    ljungBoxStat: float = 0.0
    ljungBoxPvalue: float = 1.0
    ljungBoxLag: int = 10
    isNormal: bool = True
    jarqueBeraStat: float = 0.0
    jarqueBeraPvalue: float = 1.0
    skewness: float = 0.0
    kurtosis: float = 3.0
    hasHeteroscedasticity: bool = False
    archStat: float = 0.0
    archPvalue: float = 1.0
    durbinWatson: float = 2.0
    acfValues: np.ndarray = field(default_factory=lambda: np.array([]))
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Diagnostics summary"""
        lines = []
        lines.append("=" * 60)
        lines.append("       Forecast Model Residual Diagnostics Summary")
        lines.append("=" * 60)

        lines.append("\n[ Ljung-Box Autocorrelation Test ]")
        lbStatus = "PASS" if self.isWhiteNoise else "FAIL"
        sig = self._significance(self.ljungBoxPvalue)
        lines.append(f"  Q({self.ljungBoxLag}) = {self.ljungBoxStat:.4f}, "
                     f"p = {self.ljungBoxPvalue:.4f} {sig}")
        lines.append(f"  Result: {lbStatus}")
        if self.isWhiteNoise:
            lines.append("  Interpretation: No significant autocorrelation in residuals (white noise)")
        else:
            lines.append("  Interpretation: Autocorrelation present in residuals -> Model does not fully capture time series structure")

        lines.append("\n[ Jarque-Bera Normality Test ]")
        jbStatus = "PASS" if self.isNormal else "FAIL"
        sig = self._significance(self.jarqueBeraPvalue)
        lines.append(f"  JB = {self.jarqueBeraStat:.4f}, "
                     f"p = {self.jarqueBeraPvalue:.4f} {sig}")
        lines.append(f"  Skewness = {self.skewness:.4f}, "
                     f"Kurtosis = {self.kurtosis:.4f}")
        lines.append(f"  Result: {jbStatus}")
        if self.isNormal:
            lines.append("  Interpretation: Residuals follow normal distribution -> Prediction intervals are reliable")
        else:
            lines.append("  Interpretation: Non-normal residuals -> Prediction interval coverage may be inaccurate")

        lines.append("\n[ ARCH Heteroscedasticity Test ]")
        archStatus = "PASS" if not self.hasHeteroscedasticity else "FAIL"
        sig = self._significance(self.archPvalue)
        lines.append(f"  ARCH LM = {self.archStat:.4f}, "
                     f"p = {self.archPvalue:.4f} {sig}")
        lines.append(f"  Result: {archStatus}")
        if not self.hasHeteroscedasticity:
            lines.append("  Interpretation: Residual variance is constant (homoscedastic)")
        else:
            lines.append("  Interpretation: Conditional heteroscedasticity present in residuals -> Consider GARCH model")

        lines.append("\n[ Durbin-Watson Statistic ]")
        lines.append(f"  DW = {self.durbinWatson:.4f}")
        if self.durbinWatson < 1.5:
            lines.append("  Interpretation: Suspected positive autocorrelation (DW < 1.5)")
        elif self.durbinWatson > 2.5:
            lines.append("  Interpretation: Suspected negative autocorrelation (DW > 2.5)")
        else:
            lines.append("  Interpretation: No first-order autocorrelation (normal)")

        if self.issues:
            lines.append("\n[ Issues Found ]")
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("\n[ Conclusion: Residuals satisfy white noise conditions ]")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _significance(pValue: float) -> str:
        """p-value significance indicator"""
        if pValue < 0.001:
            return "***"
        if pValue < 0.01:
            return "**"
        if pValue < 0.05:
            return "*"
        return ""


class ForecastDiagnostics:
    """
    Forecast Model Residual Diagnostics

    Comprehensively verifies whether time series forecast model
    residuals are white noise. Remaining patterns in residuals
    indicate that model improvement is needed.

    Usage:
        >>> diag = ForecastDiagnostics()
        >>> result = diag.analyze(residuals, period=12)
        >>> print(result.summary())
        >>> result.isWhiteNoise
    """

    def analyze(
        self,
        residuals: np.ndarray,
        period: int = 1,
        alpha: float = 0.05
    ) -> ForecastDiagnosticsResult:
        """
        Comprehensive residual diagnostics

        Parameters
        ----------
        residuals : np.ndarray
            Forecast model residuals (1-dimensional)
        period : int
            Seasonal period (1 if non-seasonal)
        alpha : float
            Significance level (default 0.05)

        Returns
        -------
        ForecastDiagnosticsResult
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(residuals)

        if n < 4:
            return ForecastDiagnosticsResult(
                issues=["Too few residuals for diagnostics (n < 4)"]
            )

        residualStd = np.std(residuals)
        if residualStd < 1e-15:
            return ForecastDiagnosticsResult(
                isWhiteNoise=True,
                isNormal=True,
                hasHeteroscedasticity=False,
                durbinWatson=2.0,
                issues=["Residuals are constant (variance = 0). Check for perfect fit or data error"]
            )

        maxLagLB = self._defaultLjungBoxLag(n, period)
        lbStat, lbPvalue = self.ljungBoxTest(residuals, maxLag=maxLagLB)

        jbStat, jbPvalue, skew, kurt = self.jarqueBeraTest(residuals)

        archLags = min(5, n // 4 - 1)
        archLags = max(archLags, 1)
        archStat, archPvalue = self.archTest(residuals, lags=archLags)

        dwStat = self.durbinWatsonTest(residuals)

        acfMaxLag = min(max(20, 2 * period), n // 2 - 1)
        acfMaxLag = max(acfMaxLag, 1)
        acfVals = self.acf(residuals, maxLag=acfMaxLag)

        isWhiteNoise = lbPvalue >= alpha
        isNormal = jbPvalue >= alpha
        hasHeteroscedasticity = archPvalue < alpha

        issues = self._identifyIssues(
            isWhiteNoise, lbPvalue, lbStat, maxLagLB,
            isNormal, jbPvalue, skew, kurt,
            hasHeteroscedasticity, archPvalue,
            dwStat, acfVals, n, period, alpha
        )

        return ForecastDiagnosticsResult(
            isWhiteNoise=isWhiteNoise,
            ljungBoxStat=lbStat,
            ljungBoxPvalue=lbPvalue,
            ljungBoxLag=maxLagLB,
            isNormal=isNormal,
            jarqueBeraStat=jbStat,
            jarqueBeraPvalue=jbPvalue,
            skewness=skew,
            kurtosis=kurt,
            hasHeteroscedasticity=hasHeteroscedasticity,
            archStat=archStat,
            archPvalue=archPvalue,
            durbinWatson=dwStat,
            acfValues=acfVals,
            issues=issues,
        )

    def ljungBoxTest(
        self,
        residuals: np.ndarray,
        maxLag: Optional[int] = None
    ) -> tuple:
        """
        Ljung-Box Q test (residual autocorrelation test)

        Parameters
        ----------
        residuals : np.ndarray
            Residual array
        maxLag : int, optional
            Maximum lag. If None, uses min(10, n//5)

        Returns
        -------
        tuple
            (Q statistic, p-value)
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(residuals)

        if maxLag is None:
            maxLag = min(10, n // 5)
        maxLag = max(1, min(maxLag, n - 1))

        if n <= maxLag:
            return 0.0, 1.0

        mean = np.mean(residuals)
        centered = residuals - mean
        gamma0 = np.sum(centered ** 2) / n

        if gamma0 < 1e-15:
            return 0.0, 1.0

        qStat = 0.0
        for k in range(1, maxLag + 1):
            rk = np.sum(centered[k:] * centered[:-k]) / (n * gamma0)
            qStat += (rk ** 2) / (n - k)

        qStat *= n * (n + 2)
        pValue = 1.0 - chi2.cdf(qStat, df=maxLag)

        return float(qStat), float(pValue)

    def jarqueBeraTest(self, residuals: np.ndarray) -> tuple:
        """
        Jarque-Bera normality test

        Parameters
        ----------
        residuals : np.ndarray
            Residual array

        Returns
        -------
        tuple
            (JB statistic, p-value, skewness, kurtosis)
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(residuals)

        if n < 3:
            return 0.0, 1.0, 0.0, 3.0

        mean = np.mean(residuals)
        centered = residuals - mean
        m2 = np.mean(centered ** 2)
        m3 = np.mean(centered ** 3)
        m4 = np.mean(centered ** 4)

        if m2 < 1e-15:
            return 0.0, 1.0, 0.0, 3.0

        skewness = m3 / (m2 ** 1.5)
        kurtosis = m4 / (m2 ** 2)

        jb = (n / 6.0) * (skewness ** 2 + (kurtosis - 3.0) ** 2 / 4.0)
        pValue = 1.0 - chi2.cdf(jb, df=2)

        return float(jb), float(pValue), float(skewness), float(kurtosis)

    def archTest(self, residuals: np.ndarray, lags: int = 5) -> tuple:
        """
        ARCH test (heteroscedasticity)

        Engle (1982) ARCH LM test.
        Regresses squared residuals on their own lags
        to test for conditional heteroscedasticity.

        Parameters
        ----------
        residuals : np.ndarray
            Residual array
        lags : int
            Number of ARCH lags (default 5)

        Returns
        -------
        tuple
            (LM statistic, p-value)
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(residuals)
        lags = max(1, min(lags, n // 4 - 1))

        if n < lags + 2:
            return 0.0, 1.0

        eSq = residuals ** 2

        eSqStd = np.std(eSq)
        if eSqStd < 1e-15:
            return 0.0, 1.0

        nObs = n - lags
        Y = eSq[lags:]
        X = np.ones((nObs, lags + 1))
        for j in range(1, lags + 1):
            X[:, j] = eSq[lags - j:n - j]

        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0, 1.0

        fitted = X @ beta
        ssTot = np.sum((Y - np.mean(Y)) ** 2)

        if ssTot < 1e-15:
            return 0.0, 1.0

        ssRes = np.sum((Y - fitted) ** 2)
        r2 = 1.0 - ssRes / ssTot
        r2 = max(r2, 0.0)

        lmStat = nObs * r2
        pValue = 1.0 - chi2.cdf(lmStat, df=lags)

        return float(lmStat), float(pValue)

    def durbinWatsonTest(self, residuals: np.ndarray) -> float:
        """
        Durbin-Watson test (first-order autocorrelation)

        Parameters
        ----------
        residuals : np.ndarray
            Residual array

        Returns
        -------
        float
            DW statistic (range 0~4, closer to 2 means no autocorrelation)
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()

        if len(residuals) < 2:
            return 2.0

        ssResid = np.sum(residuals ** 2)
        if ssResid < 1e-15:
            return 2.0

        diff = np.diff(residuals)
        return float(np.sum(diff ** 2) / ssResid)

    def acf(self, residuals: np.ndarray, maxLag: int = 20) -> np.ndarray:
        """
        Compute residual ACF

        Parameters
        ----------
        residuals : np.ndarray
            Residual array
        maxLag : int
            Maximum lag (default 20)

        Returns
        -------
        np.ndarray
            Autocorrelation values from lag 0 to maxLag (length maxLag+1)
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(residuals)
        maxLag = max(1, min(maxLag, n - 1))

        mean = np.mean(residuals)
        centered = residuals - mean
        gamma0 = np.sum(centered ** 2) / n

        if gamma0 < 1e-15:
            result = np.zeros(maxLag + 1)
            result[0] = 1.0
            return result

        acfValues = np.zeros(maxLag + 1)
        acfValues[0] = 1.0
        for k in range(1, maxLag + 1):
            gammaK = np.sum(centered[k:] * centered[:-k]) / n
            acfValues[k] = gammaK / gamma0

        return acfValues

    def _defaultLjungBoxLag(self, n: int, period: int) -> int:
        """Determine default lag for Ljung-Box test"""
        if period > 1:
            candidateLag = min(2 * period, n // 5)
        else:
            candidateLag = min(10, n // 5)
        return max(1, candidateLag)

    def _identifyIssues(
        self,
        isWhiteNoise: bool,
        lbPvalue: float,
        lbStat: float,
        lbLag: int,
        isNormal: bool,
        jbPvalue: float,
        skew: float,
        kurt: float,
        hasHeteroscedasticity: bool,
        archPvalue: float,
        dwStat: float,
        acfVals: np.ndarray,
        n: int,
        period: int,
        alpha: float
    ) -> List[str]:
        """Generate list of identified issues"""
        issues = []

        if not isWhiteNoise:
            issues.append(
                f"Residual autocorrelation present (Ljung-Box Q({lbLag})={lbStat:.2f}, "
                f"p={lbPvalue:.4f}). "
                "Consider increasing ARIMA order or adding seasonal components."
            )

        if period > 1 and len(acfVals) > period:
            seasonalACF = abs(acfVals[period])
            bartlettBound = 1.96 / np.sqrt(n)
            if seasonalACF > bartlettBound:
                issues.append(
                    f"Residual seasonal autocorrelation (ACF[{period}]={acfVals[period]:.4f}, "
                    f"threshold=+/-{bartlettBound:.4f}). "
                    "Consider seasonal differencing or adding seasonal ARIMA terms."
                )

        if not isNormal:
            interpretation = []
            if abs(skew) > 1.0:
                direction = "positive" if skew > 0 else "negative"
                interpretation.append(f"strong {direction} skewness({skew:.2f})")
            if kurt > 4.0:
                interpretation.append(f"heavy tails(kurtosis={kurt:.2f})")
            elif kurt < 2.0:
                interpretation.append(f"thin tails(kurtosis={kurt:.2f})")
            detail = ", ".join(interpretation) if interpretation else "mild non-normality"
            issues.append(
                f"Non-normal residuals (JB p={jbPvalue:.4f}, {detail}). "
                "Prediction interval coverage may be inaccurate. "
                "Consider Box-Cox transform or nonparametric prediction intervals."
            )

        if hasHeteroscedasticity:
            issues.append(
                f"Conditional heteroscedasticity present (ARCH p={archPvalue:.4f}). "
                "Consider GARCH model or volatility-adjusted prediction intervals."
            )

        if dwStat < 1.5:
            issues.append(
                f"Suspected positive first-order autocorrelation (DW={dwStat:.4f}). "
                "Consider adding an AR(1) term."
            )
        elif dwStat > 2.5:
            issues.append(
                f"Suspected negative first-order autocorrelation (DW={dwStat:.4f}). "
                "Check for over-differencing."
            )

        if len(acfVals) > 1:
            bartlettBound = 1.96 / np.sqrt(n)
            nSignificant = np.sum(np.abs(acfVals[1:]) > bartlettBound)
            expectedFalse = max(1, int(0.05 * (len(acfVals) - 1)))
            if nSignificant > 2 * expectedFalse:
                issues.append(
                    f"{nSignificant} significant ACF lags (expected ~{expectedFalse}). "
                    "Systematic patterns may remain in residuals."
                )

        return issues
