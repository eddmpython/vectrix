"""
Time Series Regression

시계열 특화 회귀 기법:
- Newey-West HAC 표준오차
- Cochrane-Orcutt (AR 잔차 보정)
- Prais-Winsten (Cochrane-Orcutt + 첫 관측값 보정)
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
    """회귀 결과"""
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
        """결과 요약"""
        lines = []
        lines.append("=" * 65)
        lines.append(f"  회귀분석 결과 ({self.covarianceType} 표준오차)")
        lines.append("=" * 65)
        lines.append(f"  관측수: {self.nObs}, 파라미터 수: {self.nParams}")
        lines.append(f"  R^2: {self.r2:.4f}, Adj R^2: {self.adjR2:.4f}")
        lines.append(f"  F-stat: {self.fStatistic:.4f}, p-value: {self.fPValue:.6f}")
        lines.append(f"  잔차 표준편차 (sigma): {np.sqrt(self.sigma2):.4f}")
        lines.append("-" * 65)
        lines.append(f"  {'변수':>10} {'계수':>12} {'표준오차':>12} {'t-stat':>10} {'p-value':>10}")
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
    """Granger 인과성 검정 결과"""
    fStatistic: float = 0.0
    pValue: float = 0.0
    optimalLag: int = 1
    aicPerLag: Dict[int, float] = field(default_factory=dict)
    bicPerLag: Dict[int, float] = field(default_factory=dict)
    fStatPerLag: Dict[int, float] = field(default_factory=dict)
    pValuePerLag: Dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        """결과 요약"""
        lines = []
        lines.append("=" * 55)
        lines.append("  Granger 인과성 검정 결과")
        lines.append("=" * 55)
        lines.append(f"  최적 lag: {self.optimalLag}")
        lines.append(f"  F-statistic: {self.fStatistic:.4f}")
        lines.append(f"  p-value: {self.pValue:.6f}")

        if self.pValue < 0.01:
            lines.append("  결론: 강한 Granger 인과성 존재 (p < 0.01)")
        elif self.pValue < 0.05:
            lines.append("  결론: Granger 인과성 존재 (p < 0.05)")
        else:
            lines.append("  결론: Granger 인과성 없음")

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
    HAC (Heteroscedasticity and Autocorrelation Consistent) 표준오차

    Bartlett kernel:
        K(j, L) = 1 - j/(L+1) for j <= L, 0 otherwise

    HAC covariance:
        V_HAC = (X'X)^{-1} S (X'X)^{-1}
        where S = sum_{j=-L}^{L} K(j,L) * Gamma_j
        Gamma_j = (1/n) sum_t (x_t * e_t) * (x_{t-j} * e_{t-j})'

    OLS 계수는 동일하지만, 표준오차가 이분산과 자기상관에 강건하게 보정됨.

    Parameters
    ----------
    maxLags : int, optional
        최대 lag 수. None이면 floor(4*(n/100)^(2/9))로 자동 계산
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
    """

    def __init__(self, maxLags: Optional[int] = None, fitIntercept: bool = True):
        self.maxLags = maxLags
        self.fitIntercept = fitIntercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> TSRegressionResult:
        """
        OLS 계수 추정 + Newey-West HAC 표준오차로 t-stat/p-value 재계산

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수

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

        # OLS 추정
        try:
            XtXinv = np.linalg.inv(Xa.T @ Xa)
        except np.linalg.LinAlgError:
            XtXinv = np.linalg.pinv(Xa.T @ Xa)

        beta = XtXinv @ (Xa.T @ y)
        residuals = y - Xa @ beta

        # 잔차 분산
        sigma2 = np.sum(residuals ** 2) / max(n - k, 1)

        # Newey-West lag 수 결정
        L = self.maxLags
        if L is None:
            L = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        L = max(L, 1)

        # HAC S 행렬 계산
        S = self._computeHACMatrix(Xa, residuals, n, k, L)

        # HAC 공분산 행렬
        hacCov = XtXinv @ S @ XtXinv

        # 표준오차
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
        HAC S 행렬 계산 (Bartlett kernel)

        S = Gamma_0 + sum_{j=1}^{L} w_j * (Gamma_j + Gamma_j')
        w_j = 1 - j/(L+1) (Bartlett)
        Gamma_j = (1/n) sum_{t=j+1}^{n} (x_t * e_t) * (x_{t-j} * e_{t-j})'
        """
        # Score 벡터: g_t = x_t * e_t
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
        """이 클래스는 fit에서 결과를 반환하므로, 결과의 coef/intercept 사용을 권장"""
        raise NotImplementedError(
            "NeweyWestOLS는 fit() 결과의 TSRegressionResult를 사용하세요. "
            "result.coef, result.intercept로 직접 예측하세요."
        )


class CochraneOrcutt:
    """
    Cochrane-Orcutt 반복 추정

    AR(1) 자기상관이 있는 잔차를 보정하는 GLS 추정법.

    모형: y_t = X_t'b + u_t, u_t = rho*u_{t-1} + e_t

    반복 알고리즘:
        1. OLS로 beta 추정
        2. 잔차에서 rho 추정 (AR(1) 계수)
        3. y* = y_t - rho*y_{t-1}, X* = X_t - rho*X_{t-1}로 변환
        4. y*, X*에 OLS
        5. 수렴까지 반복

    주의: 첫 번째 관측값은 버림 (Prais-Winsten과의 차이)

    Parameters
    ----------
    maxIter : int
        최대 반복 횟수 (기본값: 100)
    tol : float
        수렴 허용 오차 (기본값: 1e-6)
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
    """

    def __init__(self, maxIter: int = 100, tol: float = 1e-6, fitIntercept: bool = True):
        self.maxIter = maxIter
        self.tol = tol
        self.fitIntercept = fitIntercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[TSRegressionResult, float]:
        """
        Cochrane-Orcutt 반복 추정 수행

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수

        Returns
        -------
        Tuple[TSRegressionResult, float]
            (회귀 결과, 추정된 rho)
        """
        n, p = X.shape

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]

        # Step 1: 초기 OLS
        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(k)

        rho = 0.0

        for iteration in range(self.maxIter):
            betaOld = beta.copy()
            rhoOld = rho

            # Step 2: 잔차에서 rho 추정
            residuals = y - Xa @ beta
            rho = self._estimateRho(residuals)

            # Step 3: 데이터 변환 (첫 관측값 제거)
            yTransformed = y[1:] - rho * y[:-1]
            XaTransformed = Xa[1:] - rho * Xa[:-1]

            # Step 4: 변환된 데이터에 OLS
            try:
                beta = np.linalg.lstsq(XaTransformed, yTransformed, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            # Step 5: 수렴 확인
            betaDiff = np.max(np.abs(beta - betaOld))
            rhoDiff = abs(rho - rhoOld)
            if betaDiff < self.tol and rhoDiff < self.tol:
                break

        # 최종 결과 계산
        residuals = y - Xa @ beta
        fittedValues = Xa @ beta
        sigma2 = np.sum((yTransformed - XaTransformed @ beta) ** 2) / max(n - 1 - k, 1)

        # 표준오차 (변환된 데이터 기반)
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
        """잔차의 AR(1) 계수 추정"""
        n = len(residuals)
        if n < 3:
            return 0.0
        denom = np.sum(residuals[:-1] ** 2)
        if denom < 1e-15:
            return 0.0
        rho = np.sum(residuals[1:] * residuals[:-1]) / denom
        # rho를 (-1, 1) 구간으로 제한
        return float(np.clip(rho, -0.999, 0.999))


class PraisWinsten:
    """
    Prais-Winsten 추정 (Cochrane-Orcutt + 첫 관측값 보정)

    Cochrane-Orcutt과 동일하지만 첫 번째 관측값을 보존하여 효율성을 높임.

    첫 관측값 변환:
        y*_1 = sqrt(1 - rho^2) * y_1
        X*_1 = sqrt(1 - rho^2) * X_1

    나머지:
        y*_t = y_t - rho * y_{t-1}
        X*_t = X_t - rho * X_{t-1}

    Parameters
    ----------
    maxIter : int
        최대 반복 횟수 (기본값: 100)
    tol : float
        수렴 허용 오차 (기본값: 1e-6)
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
    """

    def __init__(self, maxIter: int = 100, tol: float = 1e-6, fitIntercept: bool = True):
        self.maxIter = maxIter
        self.tol = tol
        self.fitIntercept = fitIntercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[TSRegressionResult, float]:
        """
        Prais-Winsten 반복 추정 수행

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수

        Returns
        -------
        Tuple[TSRegressionResult, float]
            (회귀 결과, 추정된 rho)
        """
        n, p = X.shape

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]

        # 초기 OLS
        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(k)

        rho = 0.0

        for iteration in range(self.maxIter):
            betaOld = beta.copy()
            rhoOld = rho

            # 잔차에서 rho 추정
            residuals = y - Xa @ beta
            rho = self._estimateRho(residuals)

            # Prais-Winsten 변환
            yTransformed, XaTransformed = self._transform(y, Xa, rho)

            # 변환된 데이터에 OLS
            try:
                beta = np.linalg.lstsq(XaTransformed, yTransformed, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            # 수렴 확인
            betaDiff = np.max(np.abs(beta - betaOld))
            rhoDiff = abs(rho - rhoOld)
            if betaDiff < self.tol and rhoDiff < self.tol:
                break

        # 최종 결과 계산
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
        Prais-Winsten 변환 수행

        첫 관측값: sqrt(1-rho^2) * 원래값
        나머지: x_t - rho * x_{t-1}
        """
        n = len(y)
        yT = np.zeros(n)
        XaT = np.zeros_like(Xa)

        # 첫 관측값 변환
        sqrtFactor = np.sqrt(max(1.0 - rho ** 2, 1e-15))
        yT[0] = sqrtFactor * y[0]
        XaT[0] = sqrtFactor * Xa[0]

        # 나머지 변환
        yT[1:] = y[1:] - rho * y[:-1]
        XaT[1:] = Xa[1:] - rho * Xa[:-1]

        return yT, XaT

    @staticmethod
    def _estimateRho(residuals: np.ndarray) -> float:
        """잔차의 AR(1) 계수 추정"""
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
    Granger 인과성 검정

    H0: X가 Y를 Granger-cause 하지 않음

    알고리즘:
        1. 제한 모델: Y_t = a + sum(b_i * Y_{t-i})
        2. 비제한 모델: Y_t = a + sum(b_i * Y_{t-i}) + sum(c_j * X_{t-j})
        3. F = ((SSR_r - SSR_u) / q) / (SSR_u / (n - k))

    AIC/BIC로 최적 lag 자동 선택.

    Parameters
    ----------
    maxLag : int
        최대 lag 수 (기본값: 4)
    """

    def __init__(self, maxLag: int = 4):
        if maxLag < 1:
            raise ValueError(f"maxLag는 1 이상이어야 합니다: {maxLag}")
        self.maxLag = maxLag

    def test(self, y: np.ndarray, x: np.ndarray) -> GrangerResult:
        """
        Granger 인과성 검정 수행

        H0: x가 y를 Granger-cause 하지 않음

        Parameters
        ----------
        y : np.ndarray, shape (T,)
            종속 시계열 (인과의 결과)
        x : np.ndarray, shape (T,)
            독립 시계열 (인과의 원인 후보)

        Returns
        -------
        GrangerResult
        """
        y = np.asarray(y, dtype=float).ravel()
        x = np.asarray(x, dtype=float).ravel()

        if len(y) != len(x):
            raise ValueError(f"y({len(y)})와 x({len(x)})의 길이가 다릅니다")

        T = len(y)
        if T <= 2 * self.maxLag + 2:
            raise ValueError(
                f"데이터 길이({T})가 최대 lag({self.maxLag})에 비해 너무 짧습니다"
            )

        aicPerLag = {}
        bicPerLag = {}
        fStatPerLag = {}
        pValuePerLag = {}

        bestAIC = np.inf
        optimalLag = 1

        for lag in range(1, self.maxLag + 1):
            # 유효 데이터 범위: t = lag, lag+1, ..., T-1
            nEff = T - lag

            if nEff <= 2 * lag + 2:
                continue

            # 종속변수
            yTarget = y[lag:]

            # 제한 모델 설계행렬: intercept + Y lags
            XrParts = [np.ones((nEff, 1))]
            for j in range(1, lag + 1):
                XrParts.append(y[lag - j: T - j].reshape(-1, 1))
            Xr = np.hstack(XrParts)

            # 비제한 모델 설계행렬: intercept + Y lags + X lags
            XuParts = [Xr]
            for j in range(1, lag + 1):
                XuParts.append(x[lag - j: T - j].reshape(-1, 1))
            Xu = np.hstack(XuParts)

            # 제한 모델 OLS
            try:
                betaR = np.linalg.lstsq(Xr, yTarget, rcond=None)[0]
                ssrR = np.sum((yTarget - Xr @ betaR) ** 2)
            except np.linalg.LinAlgError:
                continue

            # 비제한 모델 OLS
            try:
                betaU = np.linalg.lstsq(Xu, yTarget, rcond=None)[0]
                ssrU = np.sum((yTarget - Xu @ betaU) ** 2)
            except np.linalg.LinAlgError:
                continue

            # F 검정
            q = lag  # X lag 변수 수
            kU = Xu.shape[1]  # 비제한 모델 파라미터 수
            dfResid = nEff - kU

            if dfResid <= 0 or ssrU < 1e-15:
                continue

            fStat = ((ssrR - ssrU) / q) / (ssrU / dfResid)
            pValue = 1.0 - stats.f.cdf(fStat, q, dfResid)

            # AIC/BIC (비제한 모델 기준)
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

        # 최적 lag의 결과
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
    Distributed Lag Model (분포시차 모형)

    Y_t = a + sum_{j=0}^{q} beta_j * X_{t-j} + epsilon_t

    X의 현재값과 과거값(lags)이 Y에 미치는 영향을 동시에 추정.
    장기 승수(long-run multiplier)와 lag 구조를 분석할 수 있음.

    Parameters
    ----------
    maxLag : int
        최대 lag (기본값: 4)
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
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
        분포시차 모형 추정

        Parameters
        ----------
        y : np.ndarray, shape (T,)
            종속 시계열
        x : np.ndarray, shape (T,) or (T, p)
            독립 시계열 (1차원이면 단일 변수)

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
            raise ValueError(f"y({len(y)})와 x({T})의 길이가 다릅니다")

        nEff = T - self.maxLag
        if nEff <= self.maxLag * px + 2:
            raise ValueError("데이터 길이가 lag 수에 비해 너무 짧습니다")

        # 설계행렬 구성: X_{t}, X_{t-1}, ..., X_{t-q}
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

        # OLS 추정
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

        # lag 계수 분리
        if self.fitIntercept:
            self.intercept = beta[0]
            self.lagCoefficients = beta[1:]
        else:
            self.intercept = 0.0
            self.lagCoefficients = beta.copy()

        self.coef = self.lagCoefficients.copy()
        self.longRunMultiplier = float(np.sum(self.lagCoefficients))

        # 전체 잔차 (원래 길이)
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
        예측값 반환

        Parameters
        ----------
        x : np.ndarray, shape (T, p) or (T,)
            독립 시계열 (최소 maxLag+1 길이 필요)

        Returns
        -------
        np.ndarray
            예측값 (길이: T - maxLag)
        """
        if self.lagCoefficients is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")

        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        T = x.shape[0]
        nEff = T - self.maxLag

        if nEff <= 0:
            raise ValueError("데이터 길이가 lag 수보다 길어야 합니다")

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
