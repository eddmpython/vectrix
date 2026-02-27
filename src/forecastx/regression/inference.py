"""
Statistical Inference Engine for Regression

statsmodels 수준의 완전한 회귀분석 통계 추론:
- R-squared, Adjusted R-squared
- 표준오차, t-통계량, p-value
- F-통계량
- 신뢰구간 (계수 및 예측)
- AIC / BIC / Log-likelihood
- 잔차 (raw, standardized, studentized)
- Durbin-Watson 통계량
- Condition Number

순수 numpy/scipy만 사용. sklearn 의존성 없음.
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


@dataclass
class RegressionResult:
    """
    회귀분석 결과 객체 - statsmodels.OLSResults 수준

    OLS 추정으로부터 계산된 모든 통계량을 한 곳에서 접근 가능.
    summary() 메서드로 statsmodels 스타일의 포맷된 텍스트 출력 지원.
    """

    # ── 계수 ──
    coefficients: np.ndarray          # beta (절편 포함)
    standardErrors: np.ndarray        # SE(beta)
    tValues: np.ndarray               # t = beta / SE(beta)
    pValues: np.ndarray               # P(|t| > t_obs)
    confidenceIntervals: np.ndarray   # [lower, upper] per coef, shape (k, 2)

    # ── 적합도 ──
    rSquared: float
    adjustedRSquared: float
    fStatistic: float
    fPValue: float

    # ── 정보 기준 ──
    logLikelihood: float
    aic: float
    bic: float

    # ── 잔차 ──
    residuals: np.ndarray              # raw residuals
    standardizedResiduals: np.ndarray  # standardized
    studentizedResiduals: np.ndarray   # studentized (leave-one-out)
    fittedValues: np.ndarray

    # ── 행렬 정보 ──
    hatMatrix: np.ndarray              # diag(H) = leverage
    covarianceMatrix: np.ndarray       # Var(beta)
    conditionNumber: float

    # ── 메타 ──
    nObs: int
    nParams: int                       # 절편 포함
    degreesOfFreedom: int              # n - k
    featureNames: Optional[List[str]] = None
    sigma: float = 0.0                 # residual standard error
    ssRes: float = 0.0
    ssTot: float = 0.0
    ssReg: float = 0.0
    durbinWatson: float = 0.0

    def summary(self, title: str = "OLS Regression Results") -> str:
        """
        statsmodels 스타일 포맷된 텍스트 출력

        Returns:
            회귀분석 결과 요약 문자열 (고정폭 폰트 기준 78자 너비)
        """
        width = 78
        halfWidth = width // 2

        # 피처 이름 결정
        if self.featureNames is not None:
            names = list(self.featureNames)
        else:
            names = [f"x{i}" for i in range(self.nParams)]

        # ── 상단 헤더 ──
        lines: List[str] = []
        lines.append(title.center(width))
        lines.append("=" * width)

        # ── 모델 정보 블록 (좌우 2열) ──
        def _leftRight(leftLabel: str, leftVal: str,
                       rightLabel: str, rightVal: str) -> str:
            """좌측/우측 정보를 78자 너비로 포맷"""
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
        # BIC만 오른쪽에 표시
        leftPad = " " * 39
        rightPart = f"   {'BIC:':<22s}{self.bic:>14.1f}"
        lines.append(leftPad + rightPart)
        lines.append("=" * width)

        # ── 계수 테이블 헤더 ──
        alpha = 0.05  # 기본 신뢰구간
        halfAlpha = alpha / 2.0
        lowerLabel = f"[{halfAlpha:.3f}"
        upperLabel = f"{1 - halfAlpha:.3f}]"

        header = (f"{'':>15s} {'coef':>10s} {'std err':>10s} "
                  f"{'t':>10s} {'P>|t|':>10s} {lowerLabel:>10s} {upperLabel:>10s}")
        lines.append(header)
        lines.append("-" * width)

        # ── 각 계수 행 ──
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

        # ── 하단 진단 통계 ──
        # Omnibus 검정 (정규성)
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
        """간단한 문자열 표현"""
        return (
            f"<RegressionResult nObs={self.nObs} nParams={self.nParams} "
            f"R2={self.rSquared:.4f} adjR2={self.adjustedRSquared:.4f}>"
        )


class OLSInference:
    """
    OLS 통계적 추론 엔진

    순수 numpy/scipy 기반으로 statsmodels OLS와 동등한 수준의 통계적 추론을 수행.
    수치적 안정성을 위해 SVD 기반 pseudo-inverse를 fallback으로 사용.

    Usage:
        >>> engine = OLSInference()
        >>> result = engine.fit(X, y)
        >>> print(result.summary())
        >>> result.pValues       # 각 계수의 p-value
        >>> result.rSquared      # 결정계수
        >>> yHat, lower, upper = engine.predict(Xnew, interval='prediction')
    """

    def __init__(self, fitIntercept: bool = True, alpha: float = 0.05):
        """
        OLS 추론 엔진 초기화

        Args:
            fitIntercept: 절편 포함 여부 (True면 X에 1열 추가)
            alpha: 기본 신뢰수준 (1 - alpha), 예: 0.05 -> 95% 신뢰구간
        """
        self.fitIntercept = fitIntercept
        self.alpha = alpha

        # fit() 후 내부 상태
        self._Xa: Optional[np.ndarray] = None     # augmented design matrix
        self._beta: Optional[np.ndarray] = None
        self._XtXinv: Optional[np.ndarray] = None  # (X'X)^{-1}
        self._sigma: float = 0.0                    # residual std error
        self._result: Optional[RegressionResult] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            featureNames: Optional[List[str]] = None) -> RegressionResult:
        """
        OLS 회귀 적합 + 완전한 통계적 추론 계산

        계산 과정:
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
            X: 설계행렬, shape (n, p). 절편 열 미포함.
            y: 종속변수 벡터, shape (n,)
            featureNames: 피처 이름 리스트 (옵션). 절편 포함 가능.

        Returns:
            RegressionResult 객체

        Raises:
            ValueError: 입력 차원이 맞지 않거나 표본 수 부족 시
        """
        # ── 입력 검증 ──
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n, p = X.shape
        if n != y.shape[0]:
            raise ValueError(
                f"X와 y의 행 수 불일치: X.shape[0]={n}, y.shape[0]={y.shape[0]}"
            )

        # ── 절편 추가 ──
        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n, dtype=np.float64), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]  # 절편 포함 파라미터 수
        df = n - k        # 잔차 자유도

        if df <= 0:
            raise ValueError(
                f"자유도가 0 이하: n={n}, k={k}. "
                f"최소 {k + 1}개의 관측치가 필요합니다."
            )

        # ── 피처 이름 ──
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

        # ── Step 1: OLS 추정 (수치적 안정성을 위해 lstsq + SVD fallback) ──
        XtX = Xa.T @ Xa

        try:
            # Cholesky 분해 시도 (가장 빠르고 안정적)
            L = np.linalg.cholesky(XtX)
            XtXinv = np.linalg.inv(L.T) @ np.linalg.inv(L)
            beta = XtXinv @ (Xa.T @ y)
        except np.linalg.LinAlgError:
            # SVD pseudo-inverse fallback
            XtXinv = np.linalg.pinv(XtX)
            beta = XtXinv @ (Xa.T @ y)

        # ── Step 2: 잔차 ──
        yHat = Xa @ beta
        residuals = y - yHat

        # ── Step 3: 잔차 분산 (s^2) ──
        ssRes = float(residuals @ residuals)
        s2 = ssRes / df
        sigma = np.sqrt(s2)

        # ── Step 4: 공분산행렬 Var(beta) ──
        covBeta = s2 * XtXinv

        # ── Step 5: 표준오차 ──
        seRaw = np.diag(covBeta)
        # 수치적 안정성: 매우 작은 음수 방지
        seRaw = np.maximum(seRaw, 0.0)
        standardErrors = np.sqrt(seRaw)

        # ── Step 6: t-통계량 ──
        # 0으로 나누기 방지
        safeStdErrors = np.where(standardErrors > 1e-15, standardErrors, 1e-15)
        tValues = beta / safeStdErrors

        # ── Step 7: p-values (양측 검정) ──
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

        # ── Step 9: F-통계량 ──
        # dfModel = k - 1 (절편 제외한 회귀 파라미터 수)
        dfModel = k - 1 if self.fitIntercept else k
        if dfModel > 0 and df > 0 and ssRes > 1e-15:
            fStatistic = (ssReg / dfModel) / (ssRes / df)
            fPValue = float(stats.f.sf(fStatistic, dfModel, df))
        else:
            fStatistic = 0.0
            fPValue = 1.0

        # ── Step 10: 정보 기준 ──
        # Log-likelihood (정규 분포 가정)
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

        # ── 신뢰구간 ──
        tCrit = stats.t.ppf(1.0 - self.alpha / 2.0, df)
        ciLower = beta - tCrit * standardErrors
        ciUpper = beta + tCrit * standardErrors
        confidenceIntervals = np.column_stack([ciLower, ciUpper])

        # ── Hat Matrix (leverage) ──
        # H = X (X'X)^{-1} X'
        # leverage h_ii = diag(H)
        # 효율적으로 h_ii만 계산: h_ii = x_i' (X'X)^{-1} x_i
        hatDiag = np.sum((Xa @ XtXinv) * Xa, axis=1)

        # ── 잔차 종류 ──
        # Standardized residuals: e_i / (s * sqrt(1 - h_ii))
        denomStd = sigma * np.sqrt(np.maximum(1.0 - hatDiag, 1e-15))
        standardizedResiduals = residuals / denomStd

        # External (leave-one-out) studentized residuals
        # s_{(i)}^2 = ((n-k)*s^2 - e_i^2 / (1 - h_ii)) / (n-k-1)
        if df > 1:
            eLoo = residuals ** 2 / np.maximum(1.0 - hatDiag, 1e-15)
            s2Loo = (df * s2 - eLoo) / (df - 1)
            # 수치적 안정성: 음수 방지
            s2Loo = np.maximum(s2Loo, 1e-15)
            sLoo = np.sqrt(s2Loo)
            studentizedResiduals = residuals / (sLoo * np.sqrt(np.maximum(1.0 - hatDiag, 1e-15)))
        else:
            # 자유도가 1이면 leave-one-out 불가
            studentizedResiduals = standardizedResiduals.copy()

        # ── Condition Number ──
        sv = np.linalg.svd(Xa, compute_uv=False)
        if sv[-1] > 1e-10:
            conditionNumber = float(sv[0] / sv[-1])
        else:
            conditionNumber = float('inf')

        # ── Durbin-Watson 통계량 ──
        # DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        if ssRes > 1e-15 and n > 1:
            diffResiduals = np.diff(residuals)
            durbinWatson = float(np.sum(diffResiduals ** 2) / ssRes)
        else:
            durbinWatson = 0.0

        # ── 내부 상태 저장 (predict에서 사용) ──
        self._Xa = Xa
        self._beta = beta
        self._XtXinv = XtXinv
        self._sigma = sigma

        # ── 결과 객체 생성 ──
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
        새 데이터에 대한 예측 + 선택적 신뢰/예측 구간

        신뢰구간 (confidence): 평균 응답의 불확실성
            y_hat +/- t_{alpha/2, df} * s * sqrt(x' (X'X)^{-1} x)

        예측구간 (prediction): 개별 응답의 불확실성
            y_hat +/- t_{alpha/2, df} * s * sqrt(1 + x' (X'X)^{-1} x)

        Args:
            X: 새 설계행렬, shape (m, p). 절편 열 미포함.
            interval: 'none', 'confidence', 'prediction' 중 하나
            alpha: 구간의 유의수준 (기본 0.05 -> 95% 구간)

        Returns:
            (yPred, lower, upper) 튜플.
            interval='none'이면 lower, upper는 None.

        Raises:
            RuntimeError: fit()이 호출되지 않은 경우
            ValueError: interval 인자가 잘못된 경우
        """
        if self._beta is None or self._XtXinv is None:
            raise RuntimeError(
                "predict()를 호출하기 전에 fit()을 먼저 실행해야 합니다."
            )

        validIntervals = ('none', 'confidence', 'prediction')
        if interval not in validIntervals:
            raise ValueError(
                f"interval은 {validIntervals} 중 하나여야 합니다. "
                f"입력: '{interval}'"
            )

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # 절편 추가
        m = X.shape[0]
        if self.fitIntercept:
            Xa = np.column_stack([np.ones(m, dtype=np.float64), X])
        else:
            Xa = X.copy()

        # 예측값
        yPred = Xa @ self._beta

        if interval == 'none':
            return yPred, None, None

        # 구간 계산
        df = self._result.degreesOfFreedom
        tCrit = stats.t.ppf(1.0 - alpha / 2.0, df)

        # x' (X'X)^{-1} x for each row
        # 효율적 계산: sum((Xa @ XtXinv) * Xa, axis=1)
        leverage = np.sum((Xa @ self._XtXinv) * Xa, axis=1)

        if interval == 'confidence':
            # 평균 응답의 표준오차
            seInterval = self._sigma * np.sqrt(leverage)
        else:  # prediction
            # 개별 응답의 표준오차
            seInterval = self._sigma * np.sqrt(1.0 + leverage)

        lower = yPred - tCrit * seInterval
        upper = yPred + tCrit * seInterval

        return yPred, lower, upper
