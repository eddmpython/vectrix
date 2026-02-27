"""
Regression Diagnostics

잔차 진단, 검정, 영향점 분석:
- VIF (Variance Inflation Factor)
- Durbin-Watson 통계량
- Breusch-Pagan 검정 (등분산성)
- White 검정 (등분산성)
- Jarque-Bera 검정 (정규성)
- Goldfeld-Quandt 검정 (등분산성)
- Cook's Distance
- Leverage (Hat values)
- DFFITS / DFBETAS
- Omnibus 검정
- 진단 플롯용 데이터

Pure numpy/scipy implementation (no sklearn dependency).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy import stats


@dataclass
class DiagnosticResult:
    """진단 결과 종합"""

    # 등분산성
    breuschPagan: Dict[str, float] = field(default_factory=dict)
    white: Dict[str, float] = field(default_factory=dict)
    goldfeldQuandt: Dict[str, float] = field(default_factory=dict)

    # 정규성
    jarqueBera: Dict[str, float] = field(default_factory=dict)
    shapiroWilk: Dict[str, float] = field(default_factory=dict)
    omnibus: Dict[str, float] = field(default_factory=dict)

    # 자기상관
    durbinWatson: float = 0.0
    ljungBox: Dict[str, float] = field(default_factory=dict)

    # 다중공선성
    vif: np.ndarray = field(default_factory=lambda: np.array([]))
    conditionNumber: float = 0.0

    # 영향점
    cooksDistance: np.ndarray = field(default_factory=lambda: np.array([]))
    leverage: np.ndarray = field(default_factory=lambda: np.array([]))
    dffits: np.ndarray = field(default_factory=lambda: np.array([]))
    dfbetas: np.ndarray = field(default_factory=lambda: np.array([]))

    # 진단 플롯 데이터
    plotData: Dict[str, Dict] = field(default_factory=dict)

    # 종합 판단
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """진단 결과 텍스트 요약"""
        lines = []
        lines.append("=" * 60)
        lines.append("         회귀분석 진단 결과 요약")
        lines.append("=" * 60)

        # 등분산성 검정
        lines.append("\n[ 등분산성 검정 ]")
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

        # 정규성 검정
        lines.append("\n[ 정규성 검정 ]")
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

        # 자기상관
        lines.append("\n[ 자기상관 검정 ]")
        lines.append(f"  Durbin-Watson: {self.durbinWatson:.4f}")
        if self.durbinWatson < 1.5:
            lines.append("    → 양의 자기상관 의심")
        elif self.durbinWatson > 2.5:
            lines.append("    → 음의 자기상관 의심")
        else:
            lines.append("    → 자기상관 없음 (정상)")
        if self.ljungBox:
            lb = self.ljungBox
            sig = "***" if lb.get('pValue', 1) < 0.01 else "**" if lb.get('pValue', 1) < 0.05 else ""
            lines.append(f"  Ljung-Box (lag 1): Q={lb.get('statistic', 0):.4f}, "
                         f"p={lb.get('pValue', 0):.4f} {sig}")

        # 다중공선성
        lines.append("\n[ 다중공선성 ]")
        if len(self.vif) > 0:
            maxVif = np.max(self.vif)
            lines.append(f"  VIF 최대값: {maxVif:.2f}")
            highVifIdx = np.where(self.vif > 10)[0]
            if len(highVifIdx) > 0:
                lines.append(f"  VIF > 10 변수 인덱스: {highVifIdx.tolist()}")
            else:
                lines.append("  VIF > 10 변수: 없음 (양호)")
        lines.append(f"  Condition Number: {self.conditionNumber:.2f}")

        # 영향점
        lines.append("\n[ 영향점 분석 ]")
        if len(self.cooksDistance) > 0:
            n = len(self.cooksDistance)
            threshold = 4.0 / n
            nInfluential = np.sum(self.cooksDistance > threshold)
            lines.append(f"  Cook's D > 4/n 관측값: {nInfluential}개")
            if nInfluential > 0:
                topIdx = np.argsort(self.cooksDistance)[-min(5, nInfluential):][::-1]
                lines.append(f"  상위 영향점 인덱스: {topIdx.tolist()}")
        if len(self.leverage) > 0:
            n = len(self.leverage)
            p = self.dfbetas.shape[1] if len(self.dfbetas.shape) == 2 else 1
            levThreshold = 2.0 * p / n
            nHighLev = np.sum(self.leverage > levThreshold)
            lines.append(f"  High Leverage (> 2p/n) 관측값: {nHighLev}개")

        # 종합 판단
        if self.issues:
            lines.append("\n[ 발견된 문제점 ]")
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("\n[ 결론: 주요 문제 없음 ]")

        lines.append("=" * 60)
        return "\n".join(lines)


class RegressionDiagnostics:
    """
    회귀분석 잔차 진단

    Usage:
        >>> diag = RegressionDiagnostics()
        >>> result = diag.diagnose(X, y, residuals, hatMatrix, beta, fittedValues)
        >>> print(result.summary())
        >>> result.vif  # VIF 값들
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
        종합 진단 수행

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬 (intercept 미포함)
        y : np.ndarray, shape (n,)
            반응변수
        residuals : np.ndarray, shape (n,)
            잔차 (y - yHat)
        hatMatrix : np.ndarray, shape (n, n)
            햇 행렬 H = X(X'X)^{-1}X'
        beta : np.ndarray, shape (k,)
            회귀계수 (intercept 포함 시 k = p+1)
        fittedValues : np.ndarray, shape (n,)
            적합값 (yHat)

        Returns
        -------
        DiagnosticResult
        """
        n, p = X.shape
        k = len(beta)  # intercept 포함 변수 수

        # 잔차 분산 추정
        sigma2 = np.sum(residuals ** 2) / max(n - k, 1)

        # leverage (hat matrix 대각)
        leverage = hatMatrix if hatMatrix.ndim == 1 else np.diag(hatMatrix)

        # 표준화 잔차
        stdResiduals = self._standardizedResiduals(residuals, sigma2, leverage)

        # --- 등분산성 검정 ---
        bpResult = self._breuschPagan(X, residuals)
        whiteResult = self._whiteTest(X, residuals)
        gqResult = self._goldfeldQuandt(X, y, k)

        # --- 정규성 검정 ---
        jbResult = self._jarqueBera(residuals)
        swResult = self._shapiroWilk(residuals)
        omnibusResult = self._omnibus(residuals)

        # --- 자기상관 ---
        dwStat = self._durbinWatson(residuals)
        lbResult = self._ljungBox(residuals, lag=1)

        # --- 다중공선성 ---
        vifValues = self._computeVIF(X)
        condNum = self._conditionNumber(X)

        # --- 영향점 ---
        cooksD = self._cooksDistance(residuals, leverage, k, sigma2)
        dffitsValues = self._computeDFFITS(residuals, leverage, k, sigma2)
        dfbetasValues = self._computeDFBETAS(X, residuals, hatMatrix, sigma2)

        # --- 진단 플롯 데이터 ---
        plotData = self._computePlotData(fittedValues, residuals, stdResiduals, leverage, cooksD)

        # --- 종합 판단 ---
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
    # 표준화 잔차
    # ----------------------------------------------------------------

    def _standardizedResiduals(
        self,
        residuals: np.ndarray,
        sigma2: float,
        leverage: np.ndarray
    ) -> np.ndarray:
        """
        내적 스튜던트화 잔차: r_i = e_i / (s * sqrt(1 - h_ii))
        """
        s = np.sqrt(max(sigma2, 1e-15))
        denom = s * np.sqrt(np.maximum(1.0 - leverage, 1e-15))
        return residuals / denom

    # ----------------------------------------------------------------
    # VIF (Variance Inflation Factor)
    # ----------------------------------------------------------------

    def _computeVIF(self, X: np.ndarray) -> np.ndarray:
        """
        각 변수 j에 대해:
        1. X_j를 나머지 변수에 대해 회귀
        2. R_j^2 계산
        3. VIF_j = 1 / (1 - R_j^2)

        VIF > 10이면 심각한 다중공선성
        """
        n, p = X.shape
        vifValues = np.zeros(p)

        if p < 2:
            vifValues[:] = 1.0
            return vifValues

        for j in range(p):
            # j번째 변수를 종속변수로, 나머지를 독립변수로
            mask = np.ones(p, dtype=bool)
            mask[j] = False
            Xj = X[:, j]
            Xrest = X[:, mask]

            # intercept 추가
            Xa = np.column_stack([np.ones(n), Xrest])

            try:
                betaAux = np.linalg.lstsq(Xa, Xj, rcond=None)[0]
                fitted = Xa @ betaAux
                ssTot = np.sum((Xj - np.mean(Xj)) ** 2)
                ssRes = np.sum((Xj - fitted) ** 2)

                if ssTot < 1e-15:
                    # 변수가 상수인 경우
                    vifValues[j] = np.inf
                else:
                    r2 = 1.0 - ssRes / ssTot
                    r2 = min(r2, 1.0 - 1e-15)  # 1에 너무 가까우면 보호
                    vifValues[j] = 1.0 / (1.0 - r2)
            except np.linalg.LinAlgError:
                vifValues[j] = np.inf

        return vifValues

    # ----------------------------------------------------------------
    # Condition Number
    # ----------------------------------------------------------------

    def _conditionNumber(self, X: np.ndarray) -> float:
        """
        설계 행렬의 조건수 (intercept 포함)
        condition number > 30이면 다중공선성 의심
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
        Durbin-Watson 통계량
        DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        DW ~ 2이면 자기상관 없음, DW < 2 양의 자기상관, DW > 2 음의 자기상관
        """
        diff = np.diff(residuals)
        ssResid = np.sum(residuals ** 2)
        if ssResid < 1e-15:
            return 2.0
        return float(np.sum(diff ** 2) / ssResid)

    # ----------------------------------------------------------------
    # Ljung-Box 검정
    # ----------------------------------------------------------------

    def _ljungBox(self, residuals: np.ndarray, lag: int = 1) -> Dict[str, float]:
        """
        Ljung-Box 검정 (자기상관 검정)
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
    # Breusch-Pagan 검정
    # ----------------------------------------------------------------

    def _breuschPagan(self, X: np.ndarray, residuals: np.ndarray) -> Dict[str, float]:
        """
        H0: 등분산 (homoscedasticity)
        1. e^2를 X에 대해 보조회귀
        2. LM = n * R^2 of auxiliary regression
        3. LM ~ chi2(p)

        추가로 F-statistic도 계산
        """
        n, p = X.shape

        if n <= p + 1:
            return {'statistic': 0.0, 'pValue': 1.0, 'fStatistic': 0.0, 'fPValue': 1.0}

        eSq = residuals ** 2

        # 보조회귀: e^2 ~ 1 + X
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
    # White 검정
    # ----------------------------------------------------------------

    def _whiteTest(self, X: np.ndarray, residuals: np.ndarray) -> Dict[str, float]:
        """
        H0: 등분산
        1. e^2 ~ X + X^2 + X_i*X_j (교차항) 보조회귀
        2. nR^2 ~ chi2(q)
        """
        n, p = X.shape
        eSq = residuals ** 2

        # 보조회귀 설계행렬: 1 + X + X^2 + 교차항
        auxParts = [np.ones((n, 1)), X]

        # 제곱항
        auxParts.append(X ** 2)

        # 교차항
        for i in range(p):
            for j in range(i + 1, p):
                auxParts.append((X[:, i] * X[:, j]).reshape(-1, 1))

        Xa = np.hstack(auxParts)
        q = Xa.shape[1] - 1  # intercept 제외

        if n <= q + 1:
            # 자유도 부족 시
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
    # Goldfeld-Quandt 검정
    # ----------------------------------------------------------------

    def _goldfeldQuandt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k: int,
        dropFraction: float = 0.2
    ) -> Dict[str, float]:
        """
        H0: 등분산
        1. X의 첫번째 열 기준 정렬
        2. 가운데 dropFraction 비율의 관측값 제거
        3. 상위/하위 그룹 각각 OLS
        4. F = SSR2 / SSR1 ~ F(df2, df1)
        """
        n, p = X.shape

        # 정렬 (첫 번째 독립변수 기준)
        sortIdx = np.argsort(X[:, 0])
        Xs = X[sortIdx]
        ys = y[sortIdx]

        nDrop = int(n * dropFraction)
        nGroup = (n - nDrop) // 2

        if nGroup <= k + 1:
            return {'statistic': 0.0, 'pValue': 1.0}

        # 하위 그룹
        X1 = Xs[:nGroup]
        y1 = ys[:nGroup]
        # 상위 그룹
        X2 = Xs[n - nGroup:]
        y2 = ys[n - nGroup:]

        # 각 그룹에 OLS
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
    # Jarque-Bera 검정
    # ----------------------------------------------------------------

    def _jarqueBera(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        H0: 잔차가 정규분포를 따름
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
    # Shapiro-Wilk 검정
    # ----------------------------------------------------------------

    def _shapiroWilk(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        Shapiro-Wilk 검정 (scipy 활용)
        H0: 잔차가 정규분포를 따름
        n <= 5000에서 유효
        """
        n = len(residuals)
        if n < 3:
            return {'statistic': 0.0, 'pValue': 1.0}

        # Shapiro-Wilk는 n > 5000이면 scipy가 경고 → 서브샘플링
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
    # Omnibus 검정 (D'Agostino-Pearson)
    # ----------------------------------------------------------------

    def _omnibus(self, residuals: np.ndarray) -> Dict[str, float]:
        """
        D'Agostino-Pearson Omnibus 검정
        H0: 잔차가 정규분포를 따름
        왜도(skewness)와 첨도(kurtosis) 동시 검정
        K^2 = Z_s^2 + Z_k^2 ~ chi2(2)
        """
        n = len(residuals)
        if n < 8:
            # scipy의 normaltest는 n >= 8 필요
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
        D_i > 4/n이면 영향점
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
        기준: |DFFITS| > 2 * sqrt(k/n)
        """
        n = len(residuals)
        if sigma2 < 1e-15:
            return np.zeros(n)

        s = np.sqrt(sigma2)
        hii = np.clip(leverage, 0, 1.0 - 1e-10)

        # 내적 스튜던트화 잔차
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
        간편 공식: DFBETAS = (X'X)^{-1} X' diag(e_i/(1-h_ii)) / s_i

        근사 계산 (computationally efficient):
        DFBETAS_{j,i} = (c_j' x_i * e_i) / (s * (1 - h_ii) * sqrt(c_jj))
        여기서 c_j는 (X'X)^{-1}의 j번째 열
        """
        n, p = X.shape
        k = p + 1  # intercept 포함

        Xa = np.column_stack([np.ones(n), X])

        try:
            XtXinv = np.linalg.inv(Xa.T @ Xa)
        except np.linalg.LinAlgError:
            return np.zeros((n, k))

        s = np.sqrt(max(sigma2, 1e-15))
        hii = np.clip(hatMatrix if hatMatrix.ndim == 1 else np.diag(hatMatrix), 0, 1.0 - 1e-10)

        # DFBETAS 근사: (XtXinv @ x_i) * e_i / (s * (1 - h_ii))
        # 각 관측값별 계산
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
    # 진단 플롯 데이터
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
        4대 진단 플롯 데이터:
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
    # 종합 판단
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
        """발견된 문제점 목록 생성"""
        issues = []
        alpha = 0.05

        # 등분산성
        if bpResult.get('pValue', 1) < alpha:
            issues.append(
                f"이분산성 의심 (Breusch-Pagan p={bpResult['pValue']:.4f}). "
                "WLS 또는 HC 표준오차를 사용하세요."
            )
        if whiteResult.get('pValue', 1) < alpha:
            issues.append(
                f"이분산성 의심 (White p={whiteResult['pValue']:.4f}). "
                "비선형 이분산 패턴이 있을 수 있습니다."
            )

        # 정규성
        normFail = 0
        if jbResult.get('pValue', 1) < alpha:
            normFail += 1
        if swResult.get('pValue', 1) < alpha:
            normFail += 1
        if omnibusResult.get('pValue', 1) < alpha:
            normFail += 1
        if normFail >= 2:
            issues.append(
                "잔차 비정규성 의심 (정규성 검정 2개 이상 기각). "
                "변환(log, Box-Cox) 또는 강건 회귀를 고려하세요."
            )

        # 자기상관
        if dwStat < 1.5:
            issues.append(
                f"양의 자기상관 의심 (DW={dwStat:.4f}). "
                "Newey-West 표준오차 또는 Cochrane-Orcutt 보정을 사용하세요."
            )
        elif dwStat > 2.5:
            issues.append(
                f"음의 자기상관 의심 (DW={dwStat:.4f}). "
                "모형 설정을 재검토하세요."
            )
        if lbResult.get('pValue', 1) < alpha:
            issues.append(
                f"자기상관 존재 (Ljung-Box p={lbResult['pValue']:.4f})."
            )

        # 다중공선성
        if len(vifValues) > 0 and np.any(vifValues > 10):
            highVifIdx = np.where(vifValues > 10)[0]
            issues.append(
                f"심각한 다중공선성 (VIF > 10: 변수 인덱스 {highVifIdx.tolist()}). "
                "변수 제거 또는 릿지 회귀를 고려하세요."
            )
        if condNum > 30:
            issues.append(
                f"높은 조건수 (CN={condNum:.1f}). 다중공선성 의심."
            )

        # 영향점
        if len(cooksD) > 0:
            threshold = 4.0 / n
            nInfluential = np.sum(cooksD > threshold)
            if nInfluential > 0:
                ratio = nInfluential / n
                if ratio > 0.05:
                    issues.append(
                        f"영향점 다수 발견 (Cook's D > 4/n: {nInfluential}개, {ratio:.1%}). "
                        "이상치 검토 또는 강건 회귀를 고려하세요."
                    )

        return issues
