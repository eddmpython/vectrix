"""
예측 모델 잔차 진단

Ljung-Box, Jarque-Bera, ARCH 검정 등
예측 모델 잔차가 백색잡음인지 검증

시계열 예측 모델의 잔차 품질을 평가하여
모델이 데이터의 구조를 충분히 포착했는지 판단.
잔차가 백색잡음이 아니면 모델 개선이 필요함을 시사.

참조:
- Ljung & Box (1978)
- Jarque & Bera (1987)
- Engle (1982) ARCH test
- Durbin & Watson (1950, 1951)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from scipy.stats import chi2, norm


@dataclass
class ForecastDiagnosticsResult:
    """예측 잔차 진단 결과"""

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
        """진단 요약"""
        lines = []
        lines.append("=" * 60)
        lines.append("       예측 모델 잔차 진단 결과 요약")
        lines.append("=" * 60)

        lines.append("\n[ Ljung-Box 자기상관 검정 ]")
        lbStatus = "PASS" if self.isWhiteNoise else "FAIL"
        sig = self._significance(self.ljungBoxPvalue)
        lines.append(f"  Q({self.ljungBoxLag}) = {self.ljungBoxStat:.4f}, "
                     f"p = {self.ljungBoxPvalue:.4f} {sig}")
        lines.append(f"  판정: {lbStatus}")
        if self.isWhiteNoise:
            lines.append("  해석: 잔차에 유의한 자기상관 없음 (백색잡음)")
        else:
            lines.append("  해석: 잔차에 자기상관 존재 → 모델이 시계열 구조를 충분히 포착하지 못함")

        lines.append("\n[ Jarque-Bera 정규성 검정 ]")
        jbStatus = "PASS" if self.isNormal else "FAIL"
        sig = self._significance(self.jarqueBeraPvalue)
        lines.append(f"  JB = {self.jarqueBeraStat:.4f}, "
                     f"p = {self.jarqueBeraPvalue:.4f} {sig}")
        lines.append(f"  왜도(Skewness) = {self.skewness:.4f}, "
                     f"첨도(Kurtosis) = {self.kurtosis:.4f}")
        lines.append(f"  판정: {jbStatus}")
        if self.isNormal:
            lines.append("  해석: 잔차가 정규분포를 따름 → 예측 구간 신뢰 가능")
        else:
            lines.append("  해석: 잔차 비정규 → 예측 구간의 커버리지가 부정확할 수 있음")

        lines.append("\n[ ARCH 이분산성 검정 ]")
        archStatus = "PASS" if not self.hasHeteroscedasticity else "FAIL"
        sig = self._significance(self.archPvalue)
        lines.append(f"  ARCH LM = {self.archStat:.4f}, "
                     f"p = {self.archPvalue:.4f} {sig}")
        lines.append(f"  판정: {archStatus}")
        if not self.hasHeteroscedasticity:
            lines.append("  해석: 잔차 분산이 일정 (등분산)")
        else:
            lines.append("  해석: 잔차에 조건부 이분산 존재 → GARCH 모델 고려")

        lines.append("\n[ Durbin-Watson 통계량 ]")
        lines.append(f"  DW = {self.durbinWatson:.4f}")
        if self.durbinWatson < 1.5:
            lines.append("  해석: 양의 자기상관 의심 (DW < 1.5)")
        elif self.durbinWatson > 2.5:
            lines.append("  해석: 음의 자기상관 의심 (DW > 2.5)")
        else:
            lines.append("  해석: 1차 자기상관 없음 (정상)")

        if self.issues:
            lines.append("\n[ 발견된 문제점 ]")
            for i, issue in enumerate(self.issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("\n[ 결론: 잔차가 백색잡음 조건을 만족 ]")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def _significance(pValue: float) -> str:
        """p-value 유의성 표시"""
        if pValue < 0.001:
            return "***"
        if pValue < 0.01:
            return "**"
        if pValue < 0.05:
            return "*"
        return ""


class ForecastDiagnostics:
    """
    예측 모델 잔차 진단

    시계열 예측 모델의 잔차가 백색잡음(white noise)인지
    종합적으로 검증. 잔차에 남아있는 패턴이 있으면
    모델 개선이 필요함을 의미.

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
        종합 잔차 진단

        Parameters
        ----------
        residuals : np.ndarray
            예측 모델의 잔차 (1차원)
        period : int
            계절 주기 (비계절이면 1)
        alpha : float
            유의수준 (기본 0.05)

        Returns
        -------
        ForecastDiagnosticsResult
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()
        n = len(residuals)

        if n < 4:
            return ForecastDiagnosticsResult(
                issues=["잔차 수가 너무 적어 진단 불가 (n < 4)"]
            )

        residualStd = np.std(residuals)
        if residualStd < 1e-15:
            return ForecastDiagnosticsResult(
                isWhiteNoise=True,
                isNormal=True,
                hasHeteroscedasticity=False,
                durbinWatson=2.0,
                issues=["잔차가 상수 (분산 = 0). 완벽 적합 또는 데이터 오류 확인 필요"]
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
        Ljung-Box Q 검정 (잔차 자기상관 검정)

        Parameters
        ----------
        residuals : np.ndarray
            잔차 배열
        maxLag : int, optional
            최대 시차. None이면 min(10, n//5) 사용

        Returns
        -------
        tuple
            (Q 통계량, p-value)
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
        Jarque-Bera 정규성 검정

        Parameters
        ----------
        residuals : np.ndarray
            잔차 배열

        Returns
        -------
        tuple
            (JB 통계량, p-value, 왜도, 첨도)
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
        ARCH 검정 (이분산성)

        Engle (1982)의 ARCH LM 검정.
        잔차 제곱을 자기 시차에 회귀하여
        조건부 이분산이 존재하는지 검정.

        Parameters
        ----------
        residuals : np.ndarray
            잔차 배열
        lags : int
            ARCH 시차 수 (기본 5)

        Returns
        -------
        tuple
            (LM 통계량, p-value)
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
        Durbin-Watson 검정 (1차 자기상관)

        Parameters
        ----------
        residuals : np.ndarray
            잔차 배열

        Returns
        -------
        float
            DW 통계량 (0~4 범위, 2에 가까울수록 자기상관 없음)
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
        잔차 ACF 계산

        Parameters
        ----------
        residuals : np.ndarray
            잔차 배열
        maxLag : int
            최대 시차 (기본 20)

        Returns
        -------
        np.ndarray
            시차 0부터 maxLag까지의 자기상관 값 (길이 maxLag+1)
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
        """Ljung-Box 검정의 기본 시차 결정"""
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
        """발견된 문제점 목록 생성"""
        issues = []

        if not isWhiteNoise:
            issues.append(
                f"잔차 자기상관 존재 (Ljung-Box Q({lbLag})={lbStat:.2f}, "
                f"p={lbPvalue:.4f}). "
                "ARIMA 차수를 높이거나 계절 성분을 추가하세요."
            )

        if period > 1 and len(acfVals) > period:
            seasonalACF = abs(acfVals[period])
            bartlettBound = 1.96 / np.sqrt(n)
            if seasonalACF > bartlettBound:
                issues.append(
                    f"계절 자기상관 잔존 (ACF[{period}]={acfVals[period]:.4f}, "
                    f"임계값=+/-{bartlettBound:.4f}). "
                    "계절 차분 또는 계절 ARIMA 항을 추가하세요."
                )

        if not isNormal:
            interpretation = []
            if abs(skew) > 1.0:
                direction = "양" if skew > 0 else "음"
                interpretation.append(f"강한 {direction}의 왜도({skew:.2f})")
            if kurt > 4.0:
                interpretation.append(f"두꺼운 꼬리(첨도={kurt:.2f})")
            elif kurt < 2.0:
                interpretation.append(f"얇은 꼬리(첨도={kurt:.2f})")
            detail = ", ".join(interpretation) if interpretation else "경미한 비정규"
            issues.append(
                f"잔차 비정규 (JB p={jbPvalue:.4f}, {detail}). "
                "예측 구간의 커버리지가 부정확할 수 있습니다. "
                "Box-Cox 변환 또는 비모수 예측 구간을 고려하세요."
            )

        if hasHeteroscedasticity:
            issues.append(
                f"조건부 이분산 존재 (ARCH p={archPvalue:.4f}). "
                "GARCH 모델 또는 변동성 조정 예측 구간을 고려하세요."
            )

        if dwStat < 1.5:
            issues.append(
                f"양의 1차 자기상관 의심 (DW={dwStat:.4f}). "
                "AR(1) 항 추가를 고려하세요."
            )
        elif dwStat > 2.5:
            issues.append(
                f"음의 1차 자기상관 의심 (DW={dwStat:.4f}). "
                "과대차분(over-differencing) 여부를 확인하세요."
            )

        if len(acfVals) > 1:
            bartlettBound = 1.96 / np.sqrt(n)
            nSignificant = np.sum(np.abs(acfVals[1:]) > bartlettBound)
            expectedFalse = max(1, int(0.05 * (len(acfVals) - 1)))
            if nSignificant > 2 * expectedFalse:
                issues.append(
                    f"유의한 ACF 시차 {nSignificant}개 (기대값 ~{expectedFalse}개). "
                    "잔차에 체계적 패턴이 남아있을 수 있습니다."
                )

        return issues
