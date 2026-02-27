"""
Regime-Aware Adaptive Forecasting

HMM(Hidden Markov Model) 기반 시계열 레짐 감지 및 레짐별 적응 예측.

레짐 유형:
- 'growth':   상승 추세 + 낮은 변동성
- 'decline':  하락 추세 + 낮은 변동성
- 'volatile': 높은 변동성 (추세 무관)
- 'stable':   낮은 변동성 + 추세 없음
- 'crisis':   급격한 하락 + 매우 높은 변동성

핵심 알고리즘:
- Baum-Welch (EM) 알고리즘으로 HMM 학습
- Forward-Backward 알고리즘 (log-space 수치 안정성)
- Viterbi 알고리즘으로 최적 상태 시퀀스 추정
- 전이 확률 가중 앙상블 예측

순수 numpy/scipy만 사용. 외부 HMM 라이브러리 없음.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """레짐 감지 결과"""

    states: np.ndarray = field(default_factory=lambda: np.array([]))
    """각 시점의 레짐 번호 (0-indexed)"""

    labels: List[str] = field(default_factory=list)
    """각 시점의 레짐 레이블 (예: 'growth', 'stable', ...)"""

    regimeHistory: List[Tuple[str, int, int]] = field(default_factory=list)
    """(레이블, 시작인덱스, 끝인덱스) 구간 리스트"""

    currentRegime: str = ""
    """마지막 시점의 레짐 레이블"""

    transitionMatrix: np.ndarray = field(default_factory=lambda: np.array([]))
    """K x K 전이 확률 행렬"""

    regimeStats: Dict[str, Dict] = field(default_factory=dict)
    """각 레짐의 통계 {'growth': {'mean': ..., 'std': ..., 'trend': ...}, ...}"""

    nRegimes: int = 0
    """감지된 레짐 수"""

    logLikelihood: float = 0.0
    """HMM 학습 후 로그 우도"""


@dataclass
class RegimeForecastResult:
    """레짐 인식 적응 예측 결과"""

    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    """예측값 배열"""

    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    """95% 하한 신뢰구간"""

    upper95: np.ndarray = field(default_factory=lambda: np.array([]))
    """95% 상한 신뢰구간"""

    currentRegime: str = ""
    """현재(마지막) 레짐"""

    regimeHistory: List[Tuple[str, int, int]] = field(default_factory=list)
    """레짐 전환 이력"""

    transitionMatrix: np.ndarray = field(default_factory=lambda: np.array([]))
    """전이 확률 행렬"""

    regimeStats: Dict[str, Dict] = field(default_factory=dict)
    """레짐별 통계"""

    modelPerRegime: Dict[str, str] = field(default_factory=dict)
    """{레짐 레이블: 사용된 모델 ID}"""

    regimeProbabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    """예측 기간 각 시점의 레짐 확률 [steps x K]"""

    scenarioForecasts: Dict[str, np.ndarray] = field(default_factory=dict)
    """각 레짐 시나리오별 예측 {레짐레이블: predictions}"""


# ---------------------------------------------------------------------------
# 수치 유틸리티
# ---------------------------------------------------------------------------

def _logSumExp(logA: np.ndarray) -> float:
    """
    log-sum-exp trick: log(sum(exp(logA)))

    수치 안정성을 위해 max를 빼고 계산한 후 다시 더한다.
    """
    maxVal = np.max(logA)
    if maxVal == -np.inf:
        return -np.inf
    return maxVal + np.log(np.sum(np.exp(logA - maxVal)))


def _logSumExpAxis(logA: np.ndarray, axis: int) -> np.ndarray:
    """다차원 배열에서 특정 axis에 대한 log-sum-exp"""
    maxVal = np.max(logA, axis=axis, keepdims=True)
    # -inf인 경우 처리
    mask = np.isfinite(maxVal)
    safe = np.where(mask, maxVal, 0.0)
    result = safe.squeeze(axis) + np.log(
        np.sum(np.exp(logA - np.where(mask, maxVal, 0.0)), axis=axis)
    )
    # 모두 -inf인 행/열은 -inf 유지
    allInf = ~np.any(np.isfinite(logA), axis=axis)
    result[allInf] = -np.inf
    return result


def _logGaussianPdf(x: float, mu: float, sigma2: float) -> float:
    """가우시안 PDF의 로그값: log N(x | mu, sigma2)"""
    if sigma2 <= 0:
        sigma2 = 1e-10
    return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2


# ---------------------------------------------------------------------------
# RegimeDetector: HMM 기반 레짐 감지
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    HMM 기반 시계열 레짐(국면) 감지기

    가우시안 관측 모델을 사용한 Hidden Markov Model로 시계열의
    숨겨진 상태(레짐)를 추정한다.

    관측값으로는 로그 수익률을 사용하며, 각 상태는 고유한
    평균과 분산을 가진 가우시안 분포로 모델링된다.

    Parameters
    ----------
    nRegimes : int
        감지할 레짐(상태) 수 (기본 3)
    maxIter : int
        Baum-Welch EM 알고리즘 최대 반복 횟수 (기본 100)

    Examples
    --------
    >>> detector = RegimeDetector(nRegimes=3)
    >>> result = detector.detect(y)
    >>> print(result.currentRegime)
    'growth'
    >>> print(result.transitionMatrix)
    [[0.9  0.05 0.05]
     [0.1  0.8  0.1 ]
     [0.05 0.15 0.8 ]]
    """

    def __init__(self, nRegimes: int = 3, maxIter: int = 100):
        if nRegimes < 2:
            raise ValueError("nRegimes는 2 이상이어야 합니다.")
        self.nRegimes = nRegimes
        self.maxIter = maxIter

        # HMM 파라미터 (학습 후 설정)
        self.pi: Optional[np.ndarray] = None          # 초기 상태 확률 [K]
        self.transitionMatrix: Optional[np.ndarray] = None  # 전이 행렬 [K x K]
        self.means: Optional[np.ndarray] = None        # 각 상태 평균 [K]
        self.variances: Optional[np.ndarray] = None    # 각 상태 분산 [K]

        self._fitted = False
        self._logLikelihood = -np.inf

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def detect(self, y: np.ndarray) -> RegimeResult:
        """
        레짐 감지 실행

        Parameters
        ----------
        y : np.ndarray
            시계열 원본 데이터 (레벨)

        Returns
        -------
        RegimeResult
            감지 결과 (상태, 레이블, 전이행렬, 통계 등)
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) < 10:
            raise ValueError("레짐 감지에 최소 10개 이상의 데이터가 필요합니다.")

        # 관측값 = 로그 수익률
        observations = self._computeReturns(y)

        # 적응적 레짐 수 결정: 데이터가 짧으면 레짐 수 줄이기
        effectiveRegimes = min(self.nRegimes, max(2, len(observations) // 20))

        # 원래 nRegimes 백업 후 적응
        origRegimes = self.nRegimes
        self.nRegimes = effectiveRegimes

        # HMM 학습 (Baum-Welch)
        self._fitHMM(observations)

        # 최적 상태 시퀀스 (Viterbi)
        states = self._viterbi(observations)

        # 상태 → 의미있는 레이블 변환
        labels = self._labelRegimes(y, states)

        # 레짐 이력 생성
        regimeHistory = self._buildRegimeHistory(labels)

        # 레짐별 통계
        regimeStats = self._computeRegimeStats(y, states, labels)

        # 전이 행렬 (학습된 것 사용)
        transitionMatrix = self.transitionMatrix.copy()

        currentRegime = labels[-1] if labels else "stable"

        # nRegimes 복원
        self.nRegimes = origRegimes

        return RegimeResult(
            states=states,
            labels=labels,
            regimeHistory=regimeHistory,
            currentRegime=currentRegime,
            transitionMatrix=transitionMatrix,
            regimeStats=regimeStats,
            nRegimes=effectiveRegimes,
            logLikelihood=self._logLikelihood,
        )

    # ------------------------------------------------------------------
    # 관측값 계산
    # ------------------------------------------------------------------

    def _computeReturns(self, y: np.ndarray) -> np.ndarray:
        """
        로그 수익률 계산: log(y[t] / y[t-1])

        0이나 음수 값이 있으면 안전하게 처리한다.
        """
        # 안전 처리: 0 이하인 값을 작은 양수로 대체
        safeY = y.copy()
        minPositive = np.min(safeY[safeY > 0]) if np.any(safeY > 0) else 1.0
        safeY[safeY <= 0] = minPositive * 0.01

        returns = np.diff(np.log(safeY))

        # NaN/Inf 처리
        mask = ~np.isfinite(returns)
        if np.any(mask):
            median = np.nanmedian(returns[~mask]) if np.any(~mask) else 0.0
            returns[mask] = median

        return returns

    # ------------------------------------------------------------------
    # HMM 학습: Baum-Welch (EM)
    # ------------------------------------------------------------------

    def _fitHMM(self, observations: np.ndarray) -> None:
        """
        Baum-Welch (EM) 알고리즘으로 HMM 학습

        모든 계산을 log-space에서 수행하여 수치 안정성을 보장한다.

        Parameters
        ----------
        observations : np.ndarray
            관측값 시퀀스 (로그 수익률)
        """
        T = len(observations)
        K = self.nRegimes

        # --- 파라미터 초기화 ---
        self._initializeParams(observations)

        prevLogLik = -np.inf
        tolerance = 1e-6

        for iteration in range(self.maxIter):
            # --- E-step: Forward-Backward ---
            logAlpha, logLik = self._forward(observations)
            logBeta = self._backward(observations)

            # 수렴 체크
            if abs(logLik - prevLogLik) < tolerance and iteration > 5:
                break
            prevLogLik = logLik

            # gamma[t][k] = P(state_t = k | Y): 사후 상태 확률
            logGamma = logAlpha + logBeta
            # 정규화: 각 시점에서 합이 1이 되도록
            logGammaNorm = _logSumExpAxis(logGamma, axis=1)
            logGamma = logGamma - logGammaNorm[:, np.newaxis]

            gamma = np.exp(logGamma)
            # 안전장치
            gamma = np.clip(gamma, 1e-300, None)
            gammaSum = gamma.sum(axis=0)
            gammaSum = np.where(gammaSum < 1e-300, 1e-300, gammaSum)

            # xi[t][j][k] = P(state_t=j, state_{t+1}=k | Y)
            logXi = np.full((T - 1, K, K), -np.inf)
            logA = np.log(np.clip(self.transitionMatrix, 1e-300, None))

            for t in range(T - 1):
                for j in range(K):
                    for k in range(K):
                        logEmission = _logGaussianPdf(
                            observations[t + 1], self.means[k], self.variances[k]
                        )
                        logXi[t, j, k] = (
                            logAlpha[t, j]
                            + logA[j, k]
                            + logEmission
                            + logBeta[t + 1, k]
                        )
                # 정규화
                norm = _logSumExp(logXi[t].ravel())
                if np.isfinite(norm):
                    logXi[t] -= norm

            xi = np.exp(logXi)
            xi = np.clip(xi, 1e-300, None)

            # --- M-step: 파라미터 업데이트 ---

            # 초기 상태 확률
            self.pi = gamma[0] / gamma[0].sum()
            self.pi = np.clip(self.pi, 1e-10, None)
            self.pi /= self.pi.sum()

            # 전이 행렬
            xiSumOverT = xi.sum(axis=0)  # [K x K]
            gammaSumForTrans = gamma[:-1].sum(axis=0)  # [K]
            gammaSumForTrans = np.where(
                gammaSumForTrans < 1e-300, 1e-300, gammaSumForTrans
            )
            self.transitionMatrix = xiSumOverT / gammaSumForTrans[:, np.newaxis]
            # 행 정규화
            rowSums = self.transitionMatrix.sum(axis=1, keepdims=True)
            rowSums = np.where(rowSums < 1e-300, 1e-300, rowSums)
            self.transitionMatrix /= rowSums
            # 수치 안정 클리핑
            self.transitionMatrix = np.clip(self.transitionMatrix, 1e-10, None)
            self.transitionMatrix /= self.transitionMatrix.sum(axis=1, keepdims=True)

            # 관측 모델 파라미터 (가우시안)
            for k in range(K):
                wk = gamma[:, k]
                wkSum = wk.sum()
                if wkSum < 1e-300:
                    continue

                # 평균
                self.means[k] = np.dot(wk, observations) / wkSum

                # 분산
                diff = observations - self.means[k]
                self.variances[k] = np.dot(wk, diff ** 2) / wkSum
                # 최소 분산 보장
                self.variances[k] = max(self.variances[k], 1e-10)

        self._logLikelihood = prevLogLik
        self._fitted = True

    def _initializeParams(self, observations: np.ndarray) -> None:
        """
        HMM 파라미터 K-means 스타일 초기화

        관측값을 분위수 기반으로 K개 클러스터로 나누어 초기 평균/분산을 설정한다.
        """
        K = self.nRegimes
        T = len(observations)

        # 초기 상태 확률: 균등
        self.pi = np.ones(K) / K

        # 전이 행렬: 대각 성분 우세 (자기 유지 확률 높음)
        self.transitionMatrix = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(self.transitionMatrix, 0.95)
        # 행 정규화
        self.transitionMatrix /= self.transitionMatrix.sum(axis=1, keepdims=True)

        # 관측 모델: 분위수 기반 초기화
        sortedObs = np.sort(observations)
        self.means = np.zeros(K)
        self.variances = np.zeros(K)

        for k in range(K):
            start = int(T * k / K)
            end = int(T * (k + 1) / K)
            segment = sortedObs[start:end]
            if len(segment) == 0:
                segment = sortedObs
            self.means[k] = np.mean(segment)
            self.variances[k] = max(np.var(segment), 1e-10)

        # 평균을 정렬하여 해석 용이하게
        sortIdx = np.argsort(self.means)
        self.means = self.means[sortIdx]
        self.variances = self.variances[sortIdx]

    # ------------------------------------------------------------------
    # Forward Algorithm (log-space)
    # ------------------------------------------------------------------

    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward 알고리즘 (log-space)

        alpha[t][k] = p(y_1,...,y_t, state_t=k)

        log-space:
            logAlpha[t][k] = log(p(y_t | state=k))
                            + logSumExp_j(logAlpha[t-1][j] + log(A[j][k]))

        Parameters
        ----------
        observations : np.ndarray
            관측값 시퀀스

        Returns
        -------
        logAlpha : np.ndarray
            [T x K] 전방 변수 (log)
        logLikelihood : float
            전체 시퀀스의 로그 우도
        """
        T = len(observations)
        K = self.nRegimes

        logAlpha = np.full((T, K), -np.inf)
        logA = np.log(np.clip(self.transitionMatrix, 1e-300, None))
        logPi = np.log(np.clip(self.pi, 1e-300, None))

        # t = 0
        for k in range(K):
            logAlpha[0, k] = logPi[k] + _logGaussianPdf(
                observations[0], self.means[k], self.variances[k]
            )

        # t = 1, ..., T-1
        for t in range(1, T):
            for k in range(K):
                logEmission = _logGaussianPdf(
                    observations[t], self.means[k], self.variances[k]
                )
                # logSumExp over previous states
                logTerms = logAlpha[t - 1, :] + logA[:, k]
                logAlpha[t, k] = logEmission + _logSumExp(logTerms)

        # 로그 우도
        logLikelihood = _logSumExp(logAlpha[T - 1, :])

        return logAlpha, logLikelihood

    # ------------------------------------------------------------------
    # Backward Algorithm (log-space)
    # ------------------------------------------------------------------

    def _backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward 알고리즘 (log-space)

        beta[t][k] = p(y_{t+1},...,y_T | state_t=k)

        log-space:
            logBeta[t][k] = logSumExp_j(log(A[k][j])
                            + log(p(y_{t+1} | state=j))
                            + logBeta[t+1][j])

        Parameters
        ----------
        observations : np.ndarray
            관측값 시퀀스

        Returns
        -------
        logBeta : np.ndarray
            [T x K] 후방 변수 (log)
        """
        T = len(observations)
        K = self.nRegimes

        logBeta = np.full((T, K), -np.inf)
        logA = np.log(np.clip(self.transitionMatrix, 1e-300, None))

        # t = T-1: log(1) = 0
        logBeta[T - 1, :] = 0.0

        # t = T-2, ..., 0
        for t in range(T - 2, -1, -1):
            for k in range(K):
                logTerms = np.zeros(K)
                for j in range(K):
                    logEmission = _logGaussianPdf(
                        observations[t + 1], self.means[j], self.variances[j]
                    )
                    logTerms[j] = logA[k, j] + logEmission + logBeta[t + 1, j]
                logBeta[t, k] = _logSumExp(logTerms)

        return logBeta

    # ------------------------------------------------------------------
    # Viterbi Algorithm (log-space)
    # ------------------------------------------------------------------

    def _viterbi(self, observations: np.ndarray) -> np.ndarray:
        """
        Viterbi 알고리즘으로 최적 상태 시퀀스 추정 (log-space)

        delta[t][k] = max_j(delta[t-1][j] * A[j][k]) * p(y_t | state=k)

        log-space:
            logDelta[t][k] = log(p(y_t | state=k))
                            + max_j(logDelta[t-1][j] + log(A[j][k]))

        Parameters
        ----------
        observations : np.ndarray
            관측값 시퀀스

        Returns
        -------
        states : np.ndarray
            최적 상태 시퀀스 [T] (0-indexed)
        """
        T = len(observations)
        K = self.nRegimes

        logDelta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)
        logA = np.log(np.clip(self.transitionMatrix, 1e-300, None))
        logPi = np.log(np.clip(self.pi, 1e-300, None))

        # t = 0
        for k in range(K):
            logDelta[0, k] = logPi[k] + _logGaussianPdf(
                observations[0], self.means[k], self.variances[k]
            )

        # t = 1, ..., T-1
        for t in range(1, T):
            for k in range(K):
                logEmission = _logGaussianPdf(
                    observations[t], self.means[k], self.variances[k]
                )
                candidates = logDelta[t - 1, :] + logA[:, k]
                bestPrev = np.argmax(candidates)
                logDelta[t, k] = logEmission + candidates[bestPrev]
                psi[t, k] = bestPrev

        # Backtrack
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(logDelta[T - 1, :])

        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        # 상태를 관측 시퀀스 길이(T)에서 원본 시계열 길이(T+1)로 확장
        # observations는 diff이므로 길이가 원본-1. 첫 시점은 첫 관측의 상태로 채운다.
        fullStates = np.empty(T + 1, dtype=int)
        fullStates[0] = states[0]
        fullStates[1:] = states

        return fullStates

    # ------------------------------------------------------------------
    # 상태 → 레이블 변환
    # ------------------------------------------------------------------

    def _labelRegimes(self, y: np.ndarray, states: np.ndarray) -> List[str]:
        """
        상태 번호를 의미있는 레이블로 변환

        각 HMM 상태의 통계를 분석하여 레짐 유형을 결정:
        - 평균 수익률 > 0.01  --> 'growth'
        - 평균 수익률 < -0.01 --> 'decline'
        - std > median_std * 1.5 --> 'volatile'
        - 평균 수익률 < -0.03 AND std > median_std * 2 --> 'crisis'
        - 나머지 --> 'stable'

        Parameters
        ----------
        y : np.ndarray
            원본 시계열
        states : np.ndarray
            상태 시퀀스 (길이 = len(y))

        Returns
        -------
        List[str]
            각 시점의 레이블
        """
        K = self.nRegimes
        returns = self._computeReturns(y)
        # states는 len(y) 길이이므로, 수익률(len(y)-1)에 맞춰 정렬
        # states[1:]가 returns에 대응
        stateForReturns = states[1:]  # len = len(returns)

        # 각 상태의 통계
        stateMeanReturn = np.zeros(K)
        stateStdReturn = np.zeros(K)

        for k in range(K):
            mask = stateForReturns == k
            if np.sum(mask) > 0:
                stateMeanReturn[k] = np.mean(returns[mask])
                stateStdReturn[k] = np.std(returns[mask])
            else:
                stateMeanReturn[k] = 0.0
                stateStdReturn[k] = 0.0

        medianStd = np.median(stateStdReturn[stateStdReturn > 0]) if np.any(stateStdReturn > 0) else 1e-6
        if medianStd < 1e-10:
            medianStd = 1e-6

        # 레이블 매핑: 상태 번호 -> 레이블
        stateToLabel: Dict[int, str] = {}

        for k in range(K):
            mr = stateMeanReturn[k]
            sd = stateStdReturn[k]

            # crisis: 큰 폭 하락 + 매우 높은 변동성
            if mr < -0.03 and sd > medianStd * 2:
                stateToLabel[k] = "crisis"
            # volatile: 높은 변동성
            elif sd > medianStd * 1.5:
                stateToLabel[k] = "volatile"
            # growth: 상승 추세
            elif mr > 0.01:
                stateToLabel[k] = "growth"
            # decline: 하락 추세
            elif mr < -0.01:
                stateToLabel[k] = "decline"
            # stable: 나머지
            else:
                stateToLabel[k] = "stable"

        # 중복 레이블 처리: 같은 레이블이 여러 상태에 배정되면
        # 구분을 위해 변동성 기준으로 재라벨
        usedLabels: Dict[str, List[int]] = {}
        for k, label in stateToLabel.items():
            usedLabels.setdefault(label, []).append(k)

        for label, stateList in usedLabels.items():
            if len(stateList) > 1:
                # 변동성 오름차순 정렬
                stateList.sort(key=lambda s: stateStdReturn[s])
                for idx, s in enumerate(stateList):
                    if idx == 0:
                        pass  # 변동성 가장 낮은 상태는 원래 레이블 유지
                    else:
                        # 변동성이 더 높은 상태에 다른 레이블 부여
                        if label == "stable":
                            stateToLabel[s] = "volatile"
                        elif label == "growth":
                            stateToLabel[s] = "volatile" if stateStdReturn[s] > medianStd else "stable"
                        elif label == "decline":
                            stateToLabel[s] = "crisis" if stateMeanReturn[s] < -0.02 else "volatile"
                        elif label == "volatile":
                            stateToLabel[s] = "crisis" if stateMeanReturn[s] < -0.01 else "growth"

        labels = [stateToLabel.get(s, "stable") for s in states]
        return labels

    # ------------------------------------------------------------------
    # 레짐 이력 구간 생성
    # ------------------------------------------------------------------

    def _buildRegimeHistory(self, labels: List[str]) -> List[Tuple[str, int, int]]:
        """연속된 동일 레짐을 구간으로 묶기"""
        if not labels:
            return []

        history: List[Tuple[str, int, int]] = []
        currentLabel = labels[0]
        start = 0

        for i in range(1, len(labels)):
            if labels[i] != currentLabel:
                history.append((currentLabel, start, i - 1))
                currentLabel = labels[i]
                start = i

        history.append((currentLabel, start, len(labels) - 1))
        return history

    # ------------------------------------------------------------------
    # 레짐별 통계 계산
    # ------------------------------------------------------------------

    def _computeRegimeStats(
        self, y: np.ndarray, states: np.ndarray, labels: List[str]
    ) -> Dict[str, Dict]:
        """
        각 레짐의 통계 계산

        Returns
        -------
        Dict[str, Dict]
            {레짐레이블: {'mean': float, 'std': float, 'trend': float,
                         'count': int, 'proportion': float}}
        """
        uniqueLabels = sorted(set(labels))
        stats: Dict[str, Dict] = {}
        totalLen = len(y)

        returns = self._computeReturns(y)

        for label in uniqueLabels:
            # 이 레짐에 속하는 인덱스들
            indices = [i for i, lb in enumerate(labels) if lb == label]
            if not indices:
                continue

            segmentValues = y[indices]
            # 수익률 인덱스 (labels[1:]에 대응)
            returnIndices = [i - 1 for i in indices if 0 < i <= len(returns)]
            segmentReturns = returns[returnIndices] if returnIndices else np.array([0.0])

            # 추세: 선형 회귀 기울기
            if len(segmentValues) > 1:
                x = np.arange(len(segmentValues))
                slope = np.polyfit(x, segmentValues, 1)[0]
            else:
                slope = 0.0

            stats[label] = {
                "mean": float(np.mean(segmentValues)),
                "std": float(np.std(segmentValues)),
                "meanReturn": float(np.mean(segmentReturns)),
                "stdReturn": float(np.std(segmentReturns)),
                "trend": float(slope),
                "count": len(indices),
                "proportion": len(indices) / totalLen,
            }

        return stats


# ---------------------------------------------------------------------------
# RegimeAwareForecaster: 레짐 인식 적응 예측기
# ---------------------------------------------------------------------------

class RegimeAwareForecaster:
    """
    레짐 인식 적응 예측기

    각 레짐에 최적화된 모델을 자동 선택하여 예측한다.
    전이 확률을 기반으로 여러 레짐 시나리오의 예측을 가중 결합한다.

    레짐별 모델 매핑:
    - growth   -> auto_ets 또는 theta (추세 추적에 강한 모델)
    - decline  -> rwd 또는 auto_ets (보수적 하락 반영)
    - volatile -> garch 또는 window_avg (변동성 모델링)
    - stable   -> mean 또는 naive (안정적 예측)
    - crisis   -> seasonal_naive (급변 시 보수적 접근)

    모델 ID 문자열만 반환하여 Vectrix 엔진이 실행하게 함.

    Parameters
    ----------
    nRegimes : int
        감지할 레짐 수 (기본 3)
    period : int
        계절 주기 (기본 7)

    Examples
    --------
    >>> raf = RegimeAwareForecaster()
    >>> result = raf.forecast(y, steps=30, period=7)
    >>> print(result.currentRegime)
    'growth'
    >>> print(result.regimeHistory)
    [('stable', 0, 50), ('growth', 51, 100)]
    >>> print(result.transitionMatrix)
    [[0.92 0.05 0.03]
     [0.08 0.87 0.05]
     [0.02 0.10 0.88]]
    """

    # 레짐별 기본 모델 매핑
    REGIME_MODEL_MAP: Dict[str, List[str]] = {
        "growth":   ["auto_ets", "theta", "dot"],
        "decline":  ["rwd", "auto_ets", "theta"],
        "volatile": ["garch", "window_avg", "auto_ets"],
        "stable":   ["mean", "naive", "auto_ets"],
        "crisis":   ["seasonal_naive", "mean", "rwd"],
    }

    def __init__(self, nRegimes: int = 3, period: int = 7):
        self.nRegimes = nRegimes
        self.period = period
        self.detector = RegimeDetector(nRegimes=nRegimes)
        self._regimeResult: Optional[RegimeResult] = None

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def forecast(
        self,
        y: np.ndarray,
        steps: int,
        period: int = 7,
    ) -> RegimeForecastResult:
        """
        레짐 인식 예측 실행

        절차:
        1. 레짐 감지 (HMM)
        2. 현재 레짐 파악
        3. 레짐별 최적 모델 선택
        4. 전이 확률 기반 앙상블 가중치 계산
        5. 각 레짐 시나리오별 예측 생성 (단순 통계 기반)
        6. 가중 결합 + 신뢰구간
        7. 레짐 전환 시나리오 반환

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        steps : int
            예측 스텝 수
        period : int
            계절 주기

        Returns
        -------
        RegimeForecastResult
            레짐 인식 예측 결과
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) < 10:
            raise ValueError("예측에 최소 10개 이상의 데이터가 필요합니다.")

        self.period = period

        # 1. 레짐 감지
        regimeResult = self.detector.detect(y)
        self._regimeResult = regimeResult

        currentRegime = regimeResult.currentRegime
        K = regimeResult.nRegimes
        transMatrix = regimeResult.transitionMatrix

        # 2. 레짐별 모델 선택
        uniqueLabels = sorted(set(regimeResult.labels))
        modelPerRegime: Dict[str, str] = {}
        for label in uniqueLabels:
            modelPerRegime[label] = self._selectModelForRegime(label, y, period)

        # 3. 레짐별 시나리오 예측 (통계 기반 단순 예측)
        scenarioForecasts: Dict[str, np.ndarray] = {}
        scenarioStds: Dict[str, float] = {}

        for label in uniqueLabels:
            pred, predStd = self._generateRegimeScenarioForecast(
                y, label, regimeResult, steps, period
            )
            scenarioForecasts[label] = pred
            scenarioStds[label] = predStd

        # 4. 전이 확률 기반 가중 예측
        predictions, regimeProbabilities = self._transitionWeightedForecast(
            y, steps, period, currentRegime, regimeResult
        )

        # 5. 신뢰구간 계산
        lower95, upper95 = self._computeConfidenceIntervals(
            predictions, y, steps, regimeProbabilities, regimeResult
        )

        return RegimeForecastResult(
            predictions=predictions,
            lower95=lower95,
            upper95=upper95,
            currentRegime=currentRegime,
            regimeHistory=regimeResult.regimeHistory,
            transitionMatrix=transMatrix,
            regimeStats=regimeResult.regimeStats,
            modelPerRegime=modelPerRegime,
            regimeProbabilities=regimeProbabilities,
            scenarioForecasts=scenarioForecasts,
        )

    # ------------------------------------------------------------------
    # 레짐별 모델 선택
    # ------------------------------------------------------------------

    def _selectModelForRegime(
        self, regimeLabel: str, y: np.ndarray, period: int
    ) -> str:
        """
        레짐별 최적 모델 ID 반환

        데이터 길이에 따라 후보 중 적합한 모델을 선택한다.

        Parameters
        ----------
        regimeLabel : str
            레짐 레이블
        y : np.ndarray
            시계열 데이터
        period : int
            계절 주기

        Returns
        -------
        str
            모델 ID (Vectrix 엔진용)
        """
        n = len(y)
        candidates = self.REGIME_MODEL_MAP.get(regimeLabel, ["auto_ets"])

        # 데이터 길이에 따른 필터링
        minDataRequirements = {
            "auto_ets": 20,
            "auto_arima": 30,
            "theta": 10,
            "dot": 10,
            "garch": 50,
            "seasonal_naive": max(period * 2, 14),
            "rwd": 5,
            "mean": 2,
            "naive": 2,
            "window_avg": 5,
        }

        for candidate in candidates:
            minRequired = minDataRequirements.get(candidate, 10)
            if n >= minRequired:
                return candidate

        # 모든 후보가 데이터 부족이면 가장 단순한 모델
        return "mean" if n >= 2 else "naive"

    # ------------------------------------------------------------------
    # 레짐 시나리오별 예측 생성
    # ------------------------------------------------------------------

    def _generateRegimeScenarioForecast(
        self,
        y: np.ndarray,
        regimeLabel: str,
        regimeResult: RegimeResult,
        steps: int,
        period: int,
    ) -> Tuple[np.ndarray, float]:
        """
        특정 레짐이 지속된다고 가정한 시나리오 예측

        해당 레짐 구간의 통계 특성(추세, 변동성)을 활용하여
        단순 통계 기반 예측을 생성한다.

        Parameters
        ----------
        y : np.ndarray
            원본 시계열
        regimeLabel : str
            시나리오 레짐
        regimeResult : RegimeResult
            레짐 감지 결과
        steps : int
            예측 스텝 수
        period : int
            계절 주기

        Returns
        -------
        Tuple[np.ndarray, float]
            (예측값, 예측 표준편차)
        """
        stats = regimeResult.regimeStats.get(regimeLabel, {})
        meanReturn = stats.get("meanReturn", 0.0)
        stdReturn = stats.get("stdReturn", 0.01)
        trend = stats.get("trend", 0.0)

        lastValue = y[-1]
        predictions = np.zeros(steps)

        if regimeLabel == "growth":
            # 상승 추세 반영: 마지막 값에서 평균 수익률만큼 성장
            for h in range(steps):
                if h == 0:
                    predictions[h] = lastValue * (1 + meanReturn)
                else:
                    predictions[h] = predictions[h - 1] * (1 + meanReturn)

        elif regimeLabel == "decline":
            # 하락 추세: damped decline (점점 완화)
            dampFactor = 0.95
            for h in range(steps):
                dampedReturn = meanReturn * (dampFactor ** h)
                if h == 0:
                    predictions[h] = lastValue * (1 + dampedReturn)
                else:
                    predictions[h] = predictions[h - 1] * (1 + dampedReturn)

        elif regimeLabel == "volatile":
            # 변동성: 평균 회귀 + 넓은 변동
            regimeMean = stats.get("mean", lastValue)
            for h in range(steps):
                # 평균으로의 점진적 회귀
                alpha = min(0.1 * (h + 1), 1.0)
                predictions[h] = lastValue * (1 - alpha) + regimeMean * alpha

        elif regimeLabel == "crisis":
            # 위기: 급락 후 안정화
            for h in range(steps):
                # 급격한 하락이 점점 완화
                dampedReturn = meanReturn * (0.8 ** h)
                if h == 0:
                    predictions[h] = lastValue * (1 + dampedReturn)
                else:
                    predictions[h] = predictions[h - 1] * (1 + dampedReturn)

        elif regimeLabel == "stable":
            # 안정: 거의 변화 없음
            predictions[:] = lastValue
            # 약간의 추세 반영
            if abs(trend) > 0:
                trendPerStep = trend / max(stats.get("count", 1), 1)
                for h in range(steps):
                    predictions[h] = lastValue + trendPerStep * (h + 1)

        else:
            predictions[:] = lastValue

        return predictions, float(stdReturn * lastValue) if lastValue != 0 else float(stdReturn)

    # ------------------------------------------------------------------
    # 전이 확률 가중 예측
    # ------------------------------------------------------------------

    def _transitionWeightedForecast(
        self,
        y: np.ndarray,
        steps: int,
        period: int,
        currentRegime: str,
        regimeResult: RegimeResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        전이 확률 가중 예측: 다른 레짐으로의 전환 확률도 반영

        현재 레짐이 growth이고 전이 확률이:
        - P(growth->growth) = 0.8
        - P(growth->stable) = 0.15
        - P(growth->decline) = 0.05
        이면 예측 가중치:
        - growth 모델 예측 * 0.8 + stable 모델 예측 * 0.15 + decline 모델 예측 * 0.05

        장기 예측에서는 행렬 거듭제곱으로 수렴:
        - step h에서의 가중치 = transitionMatrix^h 의 currentRegime 행

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        steps : int
            예측 스텝 수
        period : int
            계절 주기
        currentRegime : str
            현재 레짐
        regimeResult : RegimeResult
            레짐 감지 결과

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (가중 예측값 [steps], 레짐 확률 [steps x K])
        """
        K = regimeResult.nRegimes
        transMatrix = regimeResult.transitionMatrix  # [K x K]

        # 상태 번호와 레이블 매핑 구축
        # HMM 상태 -> 레이블 매핑 (labels의 마지막 K개 고유값과 상태 대응)
        uniqueLabels = sorted(set(regimeResult.labels))
        stateToLabel: Dict[int, str] = {}
        labelToState: Dict[str, int] = {}

        # 각 HMM 상태에서 가장 빈번한 레이블 결정
        for k in range(K):
            mask = regimeResult.states == k
            if np.sum(mask) > 0:
                labelsForState = [regimeResult.labels[i] for i in range(len(regimeResult.labels)) if mask[i]]
                # 최빈 레이블
                labelCounts: Dict[str, int] = {}
                for lb in labelsForState:
                    labelCounts[lb] = labelCounts.get(lb, 0) + 1
                bestLabel = max(labelCounts, key=labelCounts.get)  # type: ignore
                stateToLabel[k] = bestLabel
                labelToState[bestLabel] = k
            else:
                stateToLabel[k] = "stable"

        # 현재 레짐에 대응하는 상태 인덱스
        currentStateIdx = labelToState.get(currentRegime, 0)

        # 각 레짐별 시나리오 예측 생성
        scenarioPreds: Dict[int, np.ndarray] = {}
        for k in range(K):
            label = stateToLabel.get(k, "stable")
            pred, _ = self._generateRegimeScenarioForecast(
                y, label, regimeResult, steps, period
            )
            scenarioPreds[k] = pred

        # step h에서의 레짐 확률: transMatrix^h의 currentState 행
        regimeProbabilities = np.zeros((steps, K))
        weightedPredictions = np.zeros(steps)

        # 행렬 거듭제곱 누적
        matPower = np.eye(K)  # A^0 = I

        for h in range(steps):
            # h=0: A^1의 currentState 행
            matPower = matPower @ transMatrix

            regimeProbs = matPower[currentStateIdx, :]
            # 수치 안정화
            regimeProbs = np.clip(regimeProbs, 0, None)
            probSum = regimeProbs.sum()
            if probSum > 0:
                regimeProbs /= probSum

            regimeProbabilities[h, :] = regimeProbs

            # 가중 예측
            for k in range(K):
                weightedPredictions[h] += regimeProbs[k] * scenarioPreds[k][h]

        return weightedPredictions, regimeProbabilities

    # ------------------------------------------------------------------
    # 신뢰구간 계산
    # ------------------------------------------------------------------

    def _computeConfidenceIntervals(
        self,
        predictions: np.ndarray,
        y: np.ndarray,
        steps: int,
        regimeProbabilities: np.ndarray,
        regimeResult: RegimeResult,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        레짐 불확실성을 반영한 95% 신뢰구간 계산

        신뢰구간은 두 가지 불확실성을 결합:
        1. 각 레짐 내의 고유 변동성
        2. 레짐 전환 불확실성 (여러 레짐의 예측이 다를 때)

        Parameters
        ----------
        predictions : np.ndarray
            가중 예측값
        y : np.ndarray
            원본 시계열
        steps : int
            예측 스텝 수
        regimeProbabilities : np.ndarray
            [steps x K] 레짐 확률
        regimeResult : RegimeResult
            레짐 감지 결과

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (lower95, upper95)
        """
        K = regimeResult.nRegimes

        # 상태 -> 레이블 매핑
        stateToLabel: Dict[int, str] = {}
        for k in range(K):
            mask = regimeResult.states == k
            if np.sum(mask) > 0:
                labelsForState = [regimeResult.labels[i] for i in range(len(regimeResult.labels)) if mask[i]]
                labelCounts: Dict[str, int] = {}
                for lb in labelsForState:
                    labelCounts[lb] = labelCounts.get(lb, 0) + 1
                stateToLabel[k] = max(labelCounts, key=labelCounts.get)  # type: ignore
            else:
                stateToLabel[k] = "stable"

        # 각 레짐의 변동성 (표준편차)
        regimeStds = np.zeros(K)
        for k in range(K):
            label = stateToLabel.get(k, "stable")
            stats = regimeResult.regimeStats.get(label, {})
            regimeStds[k] = stats.get("std", np.std(y))

        # 기본 불확실성: 전체 데이터의 표준편차
        baseStd = np.std(y[-min(60, len(y)):])

        margin = np.zeros(steps)
        for h in range(steps):
            # 1. 레짐 내 변동성의 가중 합
            withinVar = 0.0
            for k in range(K):
                withinVar += regimeProbabilities[h, k] * regimeStds[k] ** 2

            # 2. 레짐 간 예측 분산 (시나리오 차이)
            betweenVar = 0.0
            for k in range(K):
                label = stateToLabel.get(k, "stable")
                scenarioPred, _ = self._generateRegimeScenarioForecast(
                    y, label, regimeResult, steps, self.period
                )
                betweenVar += regimeProbabilities[h, k] * (scenarioPred[h] - predictions[h]) ** 2

            # 총 분산 = 레짐 내 + 레짐 간 (Law of Total Variance)
            totalVar = withinVar + betweenVar + baseStd ** 2

            # 시간 경과에 따른 불확실성 증가 (sqrt(h+1))
            margin[h] = 1.96 * np.sqrt(totalVar) * np.sqrt(h + 1)

        lower95 = predictions - margin
        upper95 = predictions + margin

        return lower95, upper95
