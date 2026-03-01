"""
Regression Diagnostics

Residual diagnostics, tests, and influence analysis:
- VIF (Variance Inflation Factor)
- Durbin-Watson statistic
- Breusch-Pagan test (homoscedasticity)
- White test (homoscedasticity)
- Jarque-Bera test (normality)
- Goldfeld-Quandt test (homoscedasticity)
- Cook's Distance
- Leverage (Hat values)
- DFFITS / DFBETAS
- Omnibus test
- Diagnostic plot data

Pure numpy/scipy implementation (no sklearn dependency).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import stats


@dataclass
class DiagnosticResult:
    """Comprehensive diagnostic results"""

    # Homoscedasticity
    breuschPagan: Dict[str, float] = field(default_factory=dict)
    white: Dict[str, float] = field(default_factory=dict)
    goldfeldQuandt: Dict[str, float] = field(default_factory=dict)

    # Normality
    jarqueBera: Dict[str, float] = field(default_factory=dict)
    shapiroWilk: Dict[str, float] = field(default_factory=dict)
    omnibus: Dict[str, float] = field(default_factory=dict)

    # Autocorrelation
    durbinWatson: float = 0.0
    ljungBox: Dict[str, float] = field(default_factory=dict)

    # Multicollinearity
    vif: np.ndarray = field(default_factory=lambda: np.array([]))
    conditionNumber: float = 0.0

    # Influential observations
    cooksDistance: np.ndarray = field(default_factory=lambda: np.array([]))
    leverage: np.ndarray = field(default_factory=lambda: np.array([]))
    dffits: np.ndarray = field(default_factory=lambda: np.array([]))
    dfbetas: np.ndarray = field(default_factory=lambda: np.array([]))

    # Diagnostic plot data
    plotData: Dict[str, Dict] = field(default_factory=dict)

    # Overall assessment
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Text summary of diagnostic results"""
        lines = []
        lines.append("=" * 60)
        lines.append("         Regression Diagnostic Summary")
        lines.append("=" * 60)

        # Homoscedasticity tests
        lines.append("\n[ Homoscedasticity Tests ]")
        if self.breuschPagan:
            bp = self.breuschPagan
            sig = "***" if bp.get('pValue', 1) < 0.01 else "**" if bp.get('pValue', 1) < 0.05 else ""
            lines.append(f"  Breusch-Pagan: LM={bp.get('statistic', 0):.4f}, "
                         f"p={bp.get('pValue', 0):.4f} {sig}")
        if self.white:
            w = self.white
            sig = "***" if w.get('pValue', 1) < 0.01 else "**" if w.get('pValue', 1) < 0.05 else ""
            lines.append(f"  White:         LM={w.get('statistic', 0):.4f}, "
                         f"p={w.get('pValue', 0):.4f} {sig}")
        if self.goldfeldQuandt:
            gq = self.goldfeldQuandt
            sig = "***" if gq.get('pValue', 1) < 0.01 else "**" if gq.get('pValue', 1) < 0.05 else ""
            lines.append(f"  Goldfeld-Quandt: F={gq.get('statistic', 0):.4f}, "
                         f"p={gq.get('pValue', 0):.4f} {sig}")

        # Normality tests
        lines.append("\n[ Normality Tests ]")
        if self.jarqueBera:
            jb = self.jarqueBera
            sig = "***" if jb.get('pValue', 1) < 0.01 else "**" if jb.get('pValue', 1) < 0.05 else ""
            lines.append(f"  Jarque-Bera:  JB={jb.get('statistic', 0):.4f}, "
                         f"p={jb.get('pValue', 0):.4f} {sig}")
            lines.append(f"    Skewness={jb.get('skewness', 0):.4f}, "
                         f"Kurtosis={jb.get('kurtosis', 0):.4f}")
        if self.shapiroWilk:
            sw = self.shapiroWilk
            sig = "***" if sw.get('pValue', 1) < 0.01 else "**" if sw.get('pValue', 1) < 0.05 else ""
            lines.append(f"  Shapiro-Wilk: W={sw.get('statistic', 0):.4f}, "
                         f"p={sw.get('pValue', 0):.4f} {sig}")
        if self.omnibus:
            om = self.omnibus
            sig = "***" if om.get('pValue', 1) < 0.01 else "**" if om.get('pValue', 1) < 0.05 else ""
            lines.append(f"  Omnibus:      K2={om.get('statistic', 0):.4f}, "
                         f"p={om.get('pValue', 0):.4f} {sig}")

        # Autocorrelation
        lines.append("\n[ Autocorrelation Tests ]")
        lines.append(f"  Durbin-Watson: {self.durbinWatson:.4f}")
        if self.durbinWatson < 1.5:
            lines.append("    -> Positive autocorrelation suspected")
        elif self.durbinWatson > 2.5:
            lines.append("    -> Negative autocorrelation suspected")
        else:
            lines.append("    -> No autocorrelation (normal)")
        if self.ljungBox:
            lb = self.ljungBox
            sig = "***" if lb.get('pValue', 1) < 0.01 else "**" if lb.get('pValue', 1) < 0.05 else ""
            lines.append(f"  Ljung-Box (lag 1): Q={lb.get('statistic', 0):.4f}, "
                         f"p={lb.get('pValue', 0):.4f} {sig}")

        # Multicollinearity
        lines.append("\n[ Multicollinearity ]")
        if len(self.vif) > 0:
            maxVif = np.max(self.vif)
            lines.append(f"  Max VIF: {maxVif:.2f}")
            highVifIdx = np.where(self.vif > 10)[0]
            if len(highVifIdx) > 0:
                lines.append(f"  VIF > 10 variable indices: {highVifIdx.tolist()}")
            else:
                lines.append("  VIF > 10 variables: None (good)")
        lines.append(f"  Condition Number: {self.conditionNumber:.2f}")

        # Influential observations
        lines.append("\n[ Influence Analysis ]")
        if len(self.cooksDistance) > 0:
            n = len(self.cooksDistance)
            threshold = 4.0 / n
            nInfluential = np.sum(self.cooksDistance > threshold)
            lines.append(f"  Cook's D > 4/n observations: {nInfluential}")
            if nInfluential > 0:
                topIdx = np.argsort(self.cooksDistance)[-min(5, nInfluential):][::-1]
                lines.append(f"  Top influential indices: {topIdx.tolist()}")
        if len(self.leverage) > 0:
            n = len(self.leverage)
            p = self.dfbetas.shape[1] if len(self.dfbetas.shape) == 2 else 1
            levThreshold = 2.0 * p / n
            nHighLev = np.sum(self.leverage > levThreshold)
            lines.append(f"  High Leverage (> 2p/n) observations: {nHighLev}")

        # Overall assessment
        if self.issues:
            lines.append("\n[ Issues Found ]")
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("\n[ Conclusion: No major issues found ]")

        lines.append("=" * 60)
        return "\n".join(lines)


class RegressionDiagnostics:
    """
    Regression residual diagnostics

    Usage:
        >>> diag = RegressionDiagnostics()
        >>> result = diag.diagnose(X, y, residuals, hatMatrix, beta, fittedValues)
        >>> print(result.summary())
        >>> result.vif  # VIF values
    """

    def diagnose(
        self,
        X: np.ndarray,
        y: np.ndarray,
        residuals: np.ndarray,
        hatMatrix: np.ndarray,
        beta: np.ndarray,
        fittedValues: np.ndarray
    ) -> DiagnosticResult:
        """
        Perform comprehensive diagnostics

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix (without intercept)
        y : np.ndarray, shape (n,)
            Response variable
        residuals : np.ndarray, shape (n,)
            Residuals (y - yHat)
        hatMatrix : np.ndarray, shape (n, n)
            Hat matrix H = X(X'X)^{-1}X'
        beta : np.ndarray, shape (k,)
            Regression coefficients (k = p+1 if intercept is included)
        fittedValues : np.ndarray, shape (n,)
            Fitted values (yHat)

        Returns
        -------
        DiagnosticResult
        """
        n, p = X.shape
        k = len(beta)  # number of variables including intercept

        # Residual variance estimation
        sigma2 = np.sum(residuals ** 2) / max(n - k, 1)

        # leverage (hat matrix diagonal)
        leverage = hatMatrix if hatMatrix.ndim == 1 else np.diag(hatMatrix)

        # Standardized residuals
        stdResiduals = self._standardizedResiduals(residuals, sigma2, leverage)

        # --- Homoscedasticity tests ---
        bpResult = self._breuschPagan(X, residuals)
        whiteResult = self._whiteTest(X, residuals)
        gqResult = self._goldfeldQuandt(X, y, k)

        # --- Normality tests ---
        jbResult = self._jarqueBera(residuals)
        swResult = self._shapiroWilk(residuals)
        omnibusResult = self._omnibus(residuals)

        # --- Autocorrelation ---
        dwStat = self._durbinWatson(residuals)
        lbResult = self._ljungBox(residuals, lag=1)

        # --- Multicollinearity ---
        vifValues = self._computeVIF(X)
        condNum = self._conditionNumber(X)

        # --- Influential observations ---
        cooksD = self._cooksDistance(residuals, leverage, k, sigma2)
        dffitsValues = self._computeDFFITS(residuals, leverage, k, sigma2)
        dfbetasValues = self._computeDFBETAS(X, residuals, hatMatrix, sigma2)

        # --- Diagnostic plot data ---
        plotData = self._computePlotData(fittedValues, residuals, stdResiduals, leverage, cooksD)

        # --- Overall assessment ---
        issues = self._identifyIssues(
            bpResult, whiteResult, gqResult,
            jbResult, swResult, omnibusResult,
            dwStat, lbResult,
            vifValues, condNum,
            cooksD, leverage, n, k
        )

        return DiagnosticResult(
            breuschPagan=bpResult,
            white=whiteResult,
            goldfeldQuandt=gqResult,
            jarqueBera=jbResult,
            shapiroWilk=swResult,
            omnibus=omnibusResult,
            durbinWatson=dwStat,
            ljungBox=lbResult,
            vif=vifValues,
            conditionNumber=condNum,
            cooksDistance=cooksD,
            leverage=leverage,
            dffits=dffitsValues,
            dfbetas=dfbetasValues,
            plotData=plotData,
            issues=issues,
        )

    # ----------------------------------------------------------------
    # Standardized Residuals
    # ----------------------------------------------------------------

    def _standardizedResiduals(
        self,
        residuals: np.ndarray,
        sigma2: float,
        leverage: np.ndarray
    ) -> np.ndarray:
        """
        Internally studentized residuals: r_i = e_i / (s * sqrt(1 - h_ii))
        """
        s = np.sqrt(max(sigma2, 1e-15))
        denom = s * np.sqrt(np.maximum(1.0 - leverage, 1e-15))
        return residuals / denom

    # ----------------------------------------------------------------
    # VIF (Variance Inflation Factor)
    # ----------------------------------------------------------------

    def _computeVIF(self, X: np.ndarray) -> np.ndarray:
        """
        For each variable j:
        1. Regress X_j on the remaining variables
        2. Compute R_j^2
        3. VIF_j = 1 / (1 - R_j^2)

        VIF > 10 indicates severe multicollinearity
        """
        n, p = X.shape
        vifValues = np.zeros(p)

        if p < 2:
            vifValues[:] = 1.0
            return vifValues

        for j in range(p):
            # j-th variable as dependent, rest as independent
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            Xj = X[:, j]
            Xrest = X[:, mask]

            # Add intercept
            Xa = np.column_stack([np.ones(n), Xrest])

            try:
                betaAux = np.linalg.lstsq(Xa, Xj, rcond=None)[0]
                fitted = Xa @ betaAux
                ssTot = np.sum((Xj - np.mean(Xj)) ** 2)
                ssRes = np.sum((Xj - fitted) ** 2)

                if ssTot < 1e-15:
                    # Variable is constant
                    vifValues[j] = np.inf
                else:
                    r2 = 1.0 - ssRes / ssTot
                    r2 = min(r2, 1.0 - 1e-15)  # Guard against being too close to 1
                    vifValues[j] = 1.0 / (1.0 - r2)
            except np.linalg.LinAlgError:
                vifValues[j] = np.inf

        return vifValues

    # ----------------------------------------------------------------
    # Condition Number
    # ----------------------------------------------------------------

    def _conditionNumber(self, X: np.ndarray) -> float:
        """
        Condition number of design matrix (with intercept)
        condition number > 30 suggests multicollinearity
        """
        n = X.shape[0]
        Xa = np.column_stack([np.ones(n), X])
        try:
            singularValues = np.linalg.svd(Xa, compute_uv=False)
            if singularValues[-1] < 1e-15:
                return np.inf
            return float(singularValues[0] / singularValues[-1])
        except np.linalg.LinAlgError:
            return np.inf

    # ----------------------------------------------------------------
    # Durbin-Watson
    # ----------------------------------------------------------------

    def _durbinWatson(self, residuals: np.ndarray) -> float:
        """
        Durbin-Watson statistic
        DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        DW ~ 2: no autocorrelation, DW < 2: positive autocorrelation, DW > 2: negative autocorrelation
        """
        diff = np.diff(residuals)
        ssResid = np.sum(residuals ** 2)
        if ssResid < 1e-15:
            return 2.0
        return float(np.sum(diff ** 2) / ssResid)

    # ----------------------------------------------------------------
    # Ljung-Box Test
    # ----------------------------------------------------------------

    def _ljungBox(self, residuals: np.ndarray, lag: int = 1) -> Dict[str, float]:
        """
        Ljung-Box test (autocorrelation test)
        Q = n(n+2) * sum_{k=1}^{m} (r_k^2 / (n-k))
        Q ~ chi2(m)
        """
        n = len(residuals)
        if n <= lag:
            return {'statistic': 0.0, 'pValue': 1.0}

        mean = np.mean(residuals)
        centered = residuals - mean
        var = np.sum(centered ** 2) / n

        if var < 1e-15:
            return {'statistic': 0.0, 'pValue': 1.0}

        qStat = 0.0
        for k in range(1, lag + 1):
            autoCorr = np.sum(centered[k:] * centered[:-k]) / (n * var)
            qStat += (autoCorr ** 2) / (n - k)

        qStat *= n * (n + 2)
        pValue = 1.0 - stats.chi2.cdf(qStat, df=lag)

        return {'statistic': float(qStat), 'pValue': float(pValue)}

    # ----------------------------------------------------------------
    # Breusch-Pagan Test
    # ----------------------------------------------------------------

    def _breuschPagan(self, X: np.ndarray, residuals: np.ndarray) -> Dict[str, float]:
        """
        H0: homoscedasticity
        1. Auxiliary regression of e^2 on X
        2. LM = n * R^2 of auxiliary regression
        3. LM ~ chi2(p)

        Also computes F-statistic
        """
        n, p = X.shape

        if n <= p + 1:
            return {'statistic': 0.0, 'pValue': 1.0, 'fStatistic': 0.0, 'fPValue': 1.0}

        eSq = residuals ** 2

        # Auxiliary regression: e^2 ~ 1 + X
        Xa = np.column_stack([np.ones(n), X])
        try:
            betaAux = np.linalg.lstsq(Xa, eSq, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {'statistic': 0.0, 'pValue': 1.0, 'fStatistic': 0.0, 'fPValue': 1.0}

        fittedAux = Xa @ betaAux
        ssTot = np.sum((eSq - np.mean(eSq)) ** 2)
        ssRes = np.sum((eSq - fittedAux) ** 2)

        if ssTot < 1e-15:
            return {'statistic': 0.0, 'pValue': 1.0, 'fStatistic': 0.0, 'fPValue': 1.0}

        r2 = 1.0 - ssRes / ssTot

        # LM statistic
        lm = n * r2
        pValue = 1.0 - stats.chi2.cdf(lm, df=p)

        # F statistic
        ssReg = ssTot - ssRes
        dfReg = p
        dfRes = n - p - 1
        if dfRes > 0 and ssRes > 1e-15:
            fStat = (ssReg / dfReg) / (ssRes / dfRes)
            fPValue = 1.0 - stats.f.cdf(fStat, dfReg, dfRes)
        else:
            fStat = 0.0
            fPValue = 1.0

        return {
            'statistic': float(lm),
            'pValue': float(pValue),
            'fStatistic': float(fStat),
            'fPValue': float(fPValue),
        }

    # ----------------------------------------------------------------
    # White Test
    # ----------------------------------------------------------------

    def _whiteTest(self, X: np.ndarray, residuals: np.ndarray) -> Dict[str, float]:
        """
        H0: homoscedasticity
        1. Auxiliary regression e^2 ~ X + X^2 + X_i*X_j (cross terms)
        2. nR^2 ~ chi2(q)
        """
        n, p = X.shape
        eSq = residuals ** 2

        # Auxiliary design matrix: 1 + X + X^2 + cross terms
        auxParts = [np.ones((n, 1)), X]

        # Squared terms
        auxParts.append(X ** 2)

        # Cross terms
        for i in range(p):
            for j in range(i + 1, p):
                auxParts.append((X[:, i] * X[:, j]).reshape(-1, 1))

        Xa = np.hstack(auxParts)
        q = Xa.shape[1] - 1  # Excluding intercept

        if n <= q + 1:
            # Insufficient degrees of freedom
            return {'statistic': 0.0, 'pValue': 1.0}

        try:
            betaAux = np.linalg.lstsq(Xa, eSq, rcond=None)[0]
        except np.linalg.LinAlgError:
            return {'statistic': 0.0, 'pValue': 1.0}

        fittedAux = Xa @ betaAux
        ssTot = np.sum((eSq - np.mean(eSq)) ** 2)
        ssRes = np.sum((eSq - fittedAux) ** 2)

        if ssTot < 1e-15:
            return {'statistic': 0.0, 'pValue': 1.0}

        r2 = 1.0 - ssRes / ssTot
        lm = n * r2
        pValue = 1.0 - stats.chi2.cdf(lm, df=q)

        return {'statistic': float(lm), 'pValue': float(pValue)}

    # ----------------------------------------------------------------
    # Goldfeld-Quandt Test
    # ----------------------------------------------------------------

    def _goldfeldQuandt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int,
        dropFraction: float = 0.2
    ) -> Dict[str, float]:
        """
        H0: homoscedasticity
        1. Sort by first column of X
        2. Drop middle dropFraction of observations
        3. OLS on upper/lower groups separately
        4. F = SSR2 / SSR1 ~ F(df2, df1)
        """
        n, p = X.shape

        # Sort by first independent variable
        sortIdx = np.argsort(X[:, 0])
        Xs = X[sortIdx]
        ys = y[sortIdx]

        nDrop = int(n * dropFraction)
        nGroup = (n - nDrop) // 2

        if nGroup <= k + 1:
            return {'statistic': 0.0, 'pValue': 1.0}

        # Lower group
        X1 = Xs[:nGroup]
        y1 = ys[:nGroup]
        # Upper group
        X2 = Xs[n - nGroup:]
        y2 = ys[n - nGroup:]

        # OLS on each group
        def _ssr(Xg, yg):
            Xa = np.column_stack([np.ones(len(yg)), Xg])
            try:
                betaG = np.linalg.lstsq(Xa, yg, rcond=None)[0]
                resid = yg - Xa @ betaG
                return np.sum(resid ** 2)
            except np.linalg.LinAlgError:
                return 0.0

        ssr1 = _ssr(X1, y1)
        ssr2 = _ssr(X2, y2)

        df1 = nGroup - k
        df2 = nGroup - k

        if ssr1 < 1e-15 or df1 <= 0 or df2 <= 0:
            return {'statistic': 0.0, 'pValue': 1.0}

        fStat = (ssr2 / df2) / (ssr1 / df1)
        pValue = 1.0 - stats.f.cdf(fStat, df2, df1)

        return {'statistic': float(fStat), 'pValue': float(pValue)}

    # ----------------------------------------------------------------
    # Jarque-Bera Test
    # ----------------------------------------------------------------

    def _jarqueBera(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        H0: Residuals follow a normal distribution
        JB = (n/6) * (S^2 + (K-3)^2 / 4)
        JB ~ chi2(2)
        """
        n = len(residuals)
        if n < 3:
            return {'statistic': 0.0, 'pValue': 1.0, 'skewness': 0.0, 'kurtosis': 3.0}

        mean = np.mean(residuals)
        centered = residuals - mean
        m2 = np.mean(centered ** 2)
        m3 = np.mean(centered ** 3)
        m4 = np.mean(centered ** 4)

        if m2 < 1e-15:
            return {'statistic': 0.0, 'pValue': 1.0, 'skewness': 0.0, 'kurtosis': 3.0}

        skewness = m3 / (m2 ** 1.5)
        kurtosis = m4 / (m2 ** 2)

        jb = (n / 6.0) * (skewness ** 2 + (kurtosis - 3.0) ** 2 / 4.0)
        pValue = 1.0 - stats.chi2.cdf(jb, df=2)

        return {
            'statistic': float(jb),
            'pValue': float(pValue),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
        }

    # ----------------------------------------------------------------
    # Shapiro-Wilk Test
    # ----------------------------------------------------------------

    def _shapiroWilk(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Shapiro-Wilk test (using scipy)
        H0: Residuals follow a normal distribution
        Valid for n <= 5000
        """
        n = len(residuals)
        if n < 3:
            return {'statistic': 0.0, 'pValue': 1.0}

        # Shapiro-Wilk warns for n > 5000 in scipy -> subsampling
        sample = residuals
        if n > 5000:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, 5000, replace=False)
            sample = residuals[idx]

        try:
            stat, pValue = stats.shapiro(sample)
        except Exception:
            return {'statistic': 0.0, 'pValue': 1.0}

        return {'statistic': float(stat), 'pValue': float(pValue)}

    # ----------------------------------------------------------------
    # Omnibus Test (D'Agostino-Pearson)
    # ----------------------------------------------------------------

    def _omnibus(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        D'Agostino-Pearson Omnibus test
        H0: Residuals follow a normal distribution
        Joint test of skewness and kurtosis
        K^2 = Z_s^2 + Z_k^2 ~ chi2(2)
        """
        n = len(residuals)
        if n < 8:
            # scipy normaltest requires n >= 8
            return {'statistic': 0.0, 'pValue': 1.0}

        try:
            stat, pValue = stats.normaltest(residuals)
        except Exception:
            return {'statistic': 0.0, 'pValue': 1.0}

        return {'statistic': float(stat), 'pValue': float(pValue)}

    # ----------------------------------------------------------------
    # Cook's Distance
    # ----------------------------------------------------------------

    def _cooksDistance(
        self,
        residuals: np.ndarray,
        leverage: np.ndarray,
        k: int,
        sigma2: float
    ) -> np.ndarray:
        """
        D_i = (e_i^2 / (k * s^2)) * (h_ii / (1 - h_ii)^2)
        D_i > 4/n indicates influential observation
        """
        if sigma2 < 1e-15 or k < 1:
            return np.zeros(len(residuals))

        hii = np.clip(leverage, 0, 1.0 - 1e-10)
        cooksD = (residuals ** 2 / (k * sigma2)) * (hii / (1.0 - hii) ** 2)
        return cooksD

    # ----------------------------------------------------------------
    # DFFITS
    # ----------------------------------------------------------------

    def _computeDFFITS(
        self,
        residuals: np.ndarray,
        leverage: np.ndarray,
        k: int,
        sigma2: float
    ) -> np.ndarray:
        """
        DFFITS_i = e_i / (s * sqrt(1-h_ii)) * sqrt(h_ii / (1-h_ii))
        Threshold: |DFFITS| > 2 * sqrt(k/n)
        """
        n = len(residuals)
        if sigma2 < 1e-15:
            return np.zeros(n)

        s = np.sqrt(sigma2)
        hii = np.clip(leverage, 0, 1.0 - 1e-10)

        # Internally studentized residuals
        stdRes = residuals / (s * np.sqrt(1.0 - hii))

        dffitsValues = stdRes * np.sqrt(hii / (1.0 - hii))
        return dffitsValues

    # ----------------------------------------------------------------
    # DFBETAS
    # ----------------------------------------------------------------

    def _computeDFBETAS(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        hatMatrix: np.ndarray,
        sigma2: float
    ) -> np.ndarray:
        """
        DFBETAS_{j,i} = (beta_j - beta_j(-i)) / (s(-i) * sqrt(c_jj))
        Simplified formula: DFBETAS = (X'X)^{-1} X' diag(e_i/(1-h_ii)) / s_i

        Approximate computation (computationally efficient):
        DFBETAS_{j,i} = (c_j' x_i * e_i) / (s * (1 - h_ii) * sqrt(c_jj))
        where c_j is the j-th column of (X'X)^{-1}
        """
        n, p = X.shape
        k = p + 1  # Including intercept

        Xa = np.column_stack([np.ones(n), X])

        try:
            XtXinv = np.linalg.inv(Xa.T @ Xa)
        except np.linalg.LinAlgError:
            return np.zeros((n, k))

        s = np.sqrt(max(sigma2, 1e-15))
        hii = np.clip(hatMatrix if hatMatrix.ndim == 1 else np.diag(hatMatrix), 0, 1.0 - 1e-10)

        # DFBETAS approximation: (XtXinv @ x_i) * e_i / (s * (1 - h_ii))
        # Compute per observation
        dfbetasMatrix = np.zeros((n, k))
        sqrtCjj = np.sqrt(np.diag(XtXinv))
        sqrtCjj = np.where(sqrtCjj < 1e-15, 1.0, sqrtCjj)

        for i in range(n):
            xi = Xa[i, :]
            ei = residuals[i]
            denom = s * (1.0 - hii[i])
            if np.abs(denom) < 1e-15:
                continue
            dfbetasMatrix[i, :] = (XtXinv @ xi) * ei / (denom * sqrtCjj)

        return dfbetasMatrix

    # ----------------------------------------------------------------
    # Diagnostic Plot Data
    # ----------------------------------------------------------------

    def _computePlotData(
        self,
        fittedValues: np.ndarray,
        residuals: np.ndarray,
        stdResiduals: np.ndarray,
        leverage: np.ndarray,
        cooksD: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Four diagnostic plot datasets:
        1. residualsVsFitted: {'x': fitted, 'y': residuals}
        2. normalQQ: {'theoretical': quantiles, 'sample': sorted_std_resid}
        3. scaleLocation: {'x': fitted, 'y': sqrt(abs(std_resid))}
        4. residualsVsLeverage: {'x': leverage, 'y': std_resid, 'cooksD': cooksD}
        """
        n = len(residuals)

        # 1. Residuals vs Fitted
        residualsVsFitted = {
            'x': fittedValues.tolist(),
            'y': residuals.tolist(),
        }

        # 2. Normal Q-Q plot
        sortedStdResid = np.sort(stdResiduals)
        theoreticalQuantiles = stats.norm.ppf(
            (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        )
        normalQQ = {
            'theoretical': theoreticalQuantiles.tolist(),
            'sample': sortedStdResid.tolist(),
        }

        # 3. Scale-Location plot
        scaleLocation = {
            'x': fittedValues.tolist(),
            'y': np.sqrt(np.abs(stdResiduals)).tolist(),
        }

        # 4. Residuals vs Leverage
        residualsVsLeverage = {
            'x': leverage.tolist(),
            'y': stdResiduals.tolist(),
            'cooksD': cooksD.tolist(),
        }

        return {
            'residualsVsFitted': residualsVsFitted,
            'normalQQ': normalQQ,
            'scaleLocation': scaleLocation,
            'residualsVsLeverage': residualsVsLeverage,
        }

    # ----------------------------------------------------------------
    # Overall Assessment
    # ----------------------------------------------------------------

    def _identifyIssues(
        self,
        bpResult: Dict[str, float],
        whiteResult: Dict[str, float],
        gqResult: Dict[str, float],
        jbResult: Dict[str, float],
        swResult: Dict[str, float],
        omnibusResult: Dict[str, float],
        dwStat: float,
        lbResult: Dict[str, float],
        vifValues: np.ndarray,
        condNum: float,
        cooksD: np.ndarray,
        leverage: np.ndarray,
        n: int,
        k: int
    ) -> List[str]:
        """Generate list of identified issues"""
        issues = []
        alpha = 0.05

        # Homoscedasticity
        if bpResult.get('pValue', 1) < alpha:
            issues.append(
                f"Heteroscedasticity suspected (Breusch-Pagan p={bpResult['pValue']:.4f}). "
                "Consider using WLS or HC standard errors."
            )
        if whiteResult.get('pValue', 1) < alpha:
            issues.append(
                f"Heteroscedasticity suspected (White p={whiteResult['pValue']:.4f}). "
                "Nonlinear heteroscedastic patterns may be present."
            )

        # Normality
        normFail = 0
        if jbResult.get('pValue', 1) < alpha:
            normFail += 1
        if swResult.get('pValue', 1) < alpha:
            normFail += 1
        if omnibusResult.get('pValue', 1) < alpha:
            normFail += 1
        if normFail >= 2:
            issues.append(
                "Residual non-normality suspected (2 or more normality tests rejected). "
                "Consider transformations (log, Box-Cox) or robust regression."
            )

        # Autocorrelation
        if dwStat < 1.5:
            issues.append(
                f"Positive autocorrelation suspected (DW={dwStat:.4f}). "
                "Consider Newey-West standard errors or Cochrane-Orcutt correction."
            )
        elif dwStat > 2.5:
            issues.append(
                f"Negative autocorrelation suspected (DW={dwStat:.4f}). "
                "Review the model specification."
            )
        if lbResult.get('pValue', 1) < alpha:
            issues.append(
                f"Autocorrelation present (Ljung-Box p={lbResult['pValue']:.4f})."
            )

        # Multicollinearity
        if len(vifValues) > 0 and np.any(vifValues > 10):
            highVifIdx = np.where(vifValues > 10)[0]
            issues.append(
                f"Severe multicollinearity (VIF > 10: variable indices {highVifIdx.tolist()}). "
                "Consider removing variables or using ridge regression."
            )
        if condNum > 30:
            issues.append(
                f"High condition number (CN={condNum:.1f}). Multicollinearity suspected."
            )

        # Influential observations
        if len(cooksD) > 0:
            threshold = 4.0 / n
            nInfluential = np.sum(cooksD > threshold)
            if nInfluential > 0:
                ratio = nInfluential / n
                if ratio > 0.05:
                    issues.append(
                        f"Many influential points found (Cook's D > 4/n: {nInfluential}, {ratio:.1%}). "
                        "Consider reviewing outliers or using robust regression."
                    )

        return issues
