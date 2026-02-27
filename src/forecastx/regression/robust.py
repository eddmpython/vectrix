"""
Robust Regression Models

이상치에 강건한 회귀 모델:
- WLS (Weighted Least Squares)
- HuberRegressor (IRLS)
- RANSACRegressor
- QuantileRegressor

Pure numpy/scipy implementation (no sklearn dependency).
"""

import numpy as np
from typing import Optional, Tuple
from scipy.optimize import linprog


class WLSRegressor:
    """
    가중 최소제곱법 (Weighted Least Squares)

    beta = (X'WX)^{-1} X'Wy

    가중치가 알려진 경우의 이분산성 보정.
    가중치 w_i는 분산의 역수에 비례해야 함.

    Parameters
    ----------
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
    """

    def __init__(self, fitIntercept: bool = True):
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0
        self._residuals = None
        self._fittedValues = None

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> 'WLSRegressor':
        """
        가중 최소제곱법으로 회귀계수 추정

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수
        weights : np.ndarray, shape (n,)
            양수 가중치. 분산의 역수에 비례.

        Returns
        -------
        self
        """
        n, p = X.shape

        # 가중치 검증
        weights = np.asarray(weights, dtype=float).ravel()
        if len(weights) != n:
            raise ValueError(f"가중치 길이({len(weights)})가 표본 수({n})와 불일치")
        if np.any(weights < 0):
            raise ValueError("가중치는 모두 양수여야 합니다")

        # 가중치 행렬 (대각)
        sqrtW = np.sqrt(weights)

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        # W^{1/2} X, W^{1/2} y로 변환 후 OLS
        Xw = Xa * sqrtW[:, np.newaxis]
        yw = y * sqrtW

        try:
            beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(Xa.shape[1])
            if self.fitIntercept:
                beta[0] = np.average(y, weights=weights)

        if self.fitIntercept:
            self.intercept = beta[0]
            self.coef = beta[1:]
        else:
            self.intercept = 0.0
            self.coef = beta

        self._fittedValues = Xa @ beta
        self._residuals = y - self._fittedValues

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측값 반환"""
        if self.coef is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        return X @ self.coef + self.intercept

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """학습 잔차"""
        return self._residuals

    @property
    def fittedValues(self) -> Optional[np.ndarray]:
        """학습 적합값"""
        return self._fittedValues


class HuberRegressor:
    """
    Huber M-estimation via IRLS (Iteratively Reweighted Least Squares)

    Huber 손실 함수를 사용하여 이상치에 강건한 회귀 추정.

    가중치 함수:
        w(e) = 1               if |e| <= epsilon * scale
        w(e) = epsilon / |e|   if |e| > epsilon * scale

    반복 알고리즘:
        1. OLS로 초기 beta 추정
        2. 잔차 및 scale(MAD) 계산
        3. Huber 가중치 계산
        4. WLS 수행
        5. 수렴까지 반복

    Parameters
    ----------
    epsilon : float
        Huber 함수의 경계값. 작을수록 이상치에 강건 (기본값: 1.35)
    maxIter : int
        최대 반복 횟수 (기본값: 100)
    tol : float
        수렴 허용 오차 (기본값: 1e-4)
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
    """

    def __init__(
        self,
        epsilon: float = 1.35,
        maxIter: int = 100,
        tol: float = 1e-4,
        fitIntercept: bool = True
    ):
        self.epsilon = epsilon
        self.maxIter = maxIter
        self.tol = tol
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0
        self.nIter = 0
        self.scale = 0.0
        self._residuals = None
        self._weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'HuberRegressor':
        """
        IRLS 알고리즘으로 Huber 회귀 추정

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수

        Returns
        -------
        self
        """
        n, p = X.shape

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]

        # Step 1: OLS 초기 추정
        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(k)
            if self.fitIntercept:
                beta[0] = np.median(y)

        for iteration in range(self.maxIter):
            betaOld = beta.copy()

            # Step 2: 잔차 계산
            residuals = y - Xa @ beta

            # Step 3: Scale 추정 (MAD)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = mad / 0.6745 if mad > 1e-15 else 1.0
            self.scale = scale

            # Step 4: Huber 가중치 계산
            scaledResid = np.abs(residuals) / scale
            weights = np.where(
                scaledResid <= self.epsilon,
                1.0,
                self.epsilon / np.maximum(scaledResid, 1e-15)
            )

            # Step 5: WLS
            sqrtW = np.sqrt(weights)
            Xw = Xa * sqrtW[:, np.newaxis]
            yw = y * sqrtW

            try:
                beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
            except np.linalg.LinAlgError:
                break

            # 수렴 확인
            if np.max(np.abs(beta - betaOld)) < self.tol:
                self.nIter = iteration + 1
                break
        else:
            self.nIter = self.maxIter

        if self.fitIntercept:
            self.intercept = beta[0]
            self.coef = beta[1:]
        else:
            self.intercept = 0.0
            self.coef = beta

        self._residuals = y - Xa @ beta
        self._weights = weights

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측값 반환"""
        if self.coef is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        return X @ self.coef + self.intercept

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """학습 잔차"""
        return self._residuals

    @property
    def weights(self) -> Optional[np.ndarray]:
        """최종 IRLS 가중치 (이상치일수록 작음)"""
        return self._weights


class RANSACRegressor:
    """
    Random Sample Consensus (RANSAC)

    랜덤 서브샘플링으로 이상치에 강건한 회귀 추정.

    알고리즘:
        1. 랜덤으로 최소 샘플(p+1) 선택
        2. 서브샘플로 OLS 학습
        3. 인라이어 판별 (|residual| < threshold)
        4. 인라이어 수가 최대인 모델 선택
        5. 최종 인라이어로 재학습

    Parameters
    ----------
    minSamples : int, optional
        최소 서브샘플 크기. None이면 p + 1
    residualThreshold : float, optional
        인라이어 판별 임계값. None이면 MAD 기반 자동 계산
    maxTrials : int
        최대 랜덤 시행 횟수 (기본값: 100)
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
    randomState : int, optional
        난수 시드
    """

    def __init__(
        self,
        minSamples: Optional[int] = None,
        residualThreshold: Optional[float] = None,
        maxTrials: int = 100,
        fitIntercept: bool = True,
        randomState: Optional[int] = None
    ):
        self.minSamples = minSamples
        self.residualThreshold = residualThreshold
        self.maxTrials = maxTrials
        self.fitIntercept = fitIntercept
        self.randomState = randomState
        self.coef = None
        self.intercept = 0.0
        self.inlierMask = None
        self.nTrials = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RANSACRegressor':
        """
        RANSAC 알고리즘으로 강건 회귀 추정

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수

        Returns
        -------
        self
        """
        n, p = X.shape
        rng = np.random.RandomState(self.randomState)

        # 최소 샘플 크기 결정
        minSamples = self.minSamples
        if minSamples is None:
            minSamples = p + 1 + (1 if self.fitIntercept else 0)
        minSamples = max(minSamples, p + 1)

        if minSamples >= n:
            # 표본이 너무 적으면 일반 OLS
            return self._fallbackOLS(X, y, n)

        # 잔차 임계값 결정
        threshold = self.residualThreshold
        if threshold is None:
            # MAD 기반: 전체 OLS 잔차의 MAD * 3
            try:
                Xa = np.column_stack([np.ones(n), X]) if self.fitIntercept else X
                betaInit = np.linalg.lstsq(Xa, y, rcond=None)[0]
                residInit = y - Xa @ betaInit
                mad = np.median(np.abs(residInit - np.median(residInit)))
                threshold = mad / 0.6745 * 3.0
            except np.linalg.LinAlgError:
                threshold = np.std(y) * 2.0
            if threshold < 1e-10:
                threshold = np.std(y) * 2.0
            if threshold < 1e-10:
                threshold = 1.0

        bestInlierCount = 0
        bestInlierMask = np.ones(n, dtype=bool)
        bestBeta = None

        for trial in range(self.maxTrials):
            # Step 1: 랜덤 서브샘플 선택
            sampleIdx = rng.choice(n, minSamples, replace=False)
            Xs = X[sampleIdx]
            ys = y[sampleIdx]

            # Step 2: 서브샘플로 OLS
            if self.fitIntercept:
                Xa = np.column_stack([np.ones(minSamples), Xs])
            else:
                Xa = Xs

            try:
                betaSample = np.linalg.lstsq(Xa, ys, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            # Step 3: 전체 데이터에 대한 잔차 계산
            if self.fitIntercept:
                XaFull = np.column_stack([np.ones(n), X])
            else:
                XaFull = X
            residuals = np.abs(y - XaFull @ betaSample)

            # Step 4: 인라이어 판별
            inlierMask = residuals < threshold
            inlierCount = np.sum(inlierMask)

            if inlierCount > bestInlierCount:
                bestInlierCount = inlierCount
                bestInlierMask = inlierMask.copy()
                bestBeta = betaSample.copy()

        self.nTrials = self.maxTrials

        # Step 5: 최종 인라이어로 재학습
        if bestInlierCount >= minSamples and bestBeta is not None:
            Xinlier = X[bestInlierMask]
            yInlier = y[bestInlierMask]
            nInlier = len(yInlier)

            if self.fitIntercept:
                Xa = np.column_stack([np.ones(nInlier), Xinlier])
            else:
                Xa = Xinlier

            try:
                betaFinal = np.linalg.lstsq(Xa, yInlier, rcond=None)[0]
            except np.linalg.LinAlgError:
                betaFinal = bestBeta
        elif bestBeta is not None:
            betaFinal = bestBeta
        else:
            # 모든 시행 실패 → OLS fallback
            return self._fallbackOLS(X, y, n)

        if self.fitIntercept:
            self.intercept = betaFinal[0]
            self.coef = betaFinal[1:]
        else:
            self.intercept = 0.0
            self.coef = betaFinal

        self.inlierMask = bestInlierMask
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측값 반환"""
        if self.coef is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        return X @ self.coef + self.intercept

    def _fallbackOLS(self, X: np.ndarray, y: np.ndarray, n: int) -> 'RANSACRegressor':
        """표본이 부족할 때 일반 OLS로 대체"""
        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X

        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(Xa.shape[1])

        if self.fitIntercept:
            self.intercept = beta[0]
            self.coef = beta[1:]
        else:
            self.intercept = 0.0
            self.coef = beta

        self.inlierMask = np.ones(n, dtype=bool)
        return self


class QuantileRegressor:
    """
    Quantile Regression

    조건부 분위수를 추정하는 회귀 모델.
    중위수 회귀(quantile=0.5)는 LAD(Least Absolute Deviations)와 동일.

    check function: rho_tau(u) = u * (tau - I(u < 0))

    선형 계획법(Linear Programming)으로 해결:
        min sum rho_tau(y_i - x_i'beta)
      = min tau * u_plus + (1-tau) * u_minus
        s.t. Xa @ beta + u_plus - u_minus = y
             u_plus, u_minus >= 0

    scipy.optimize.linprog 활용.

    Parameters
    ----------
    quantile : float
        목표 분위수, 0 < quantile < 1 (기본값: 0.5)
    fitIntercept : bool
        절편 포함 여부 (기본값: True)
    """

    def __init__(self, quantile: float = 0.5, fitIntercept: bool = True):
        if not 0 < quantile < 1:
            raise ValueError(f"quantile은 (0, 1) 구간이어야 합니다: {quantile}")
        self.quantile = quantile
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0
        self._residuals = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """
        선형 계획법으로 분위수 회귀 추정

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수

        Returns
        -------
        self
        """
        n, p = X.shape
        tau = self.quantile

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        k = Xa.shape[1]

        # LP 정식화:
        # 변수: [beta (k), u_plus (n), u_minus (n)]
        # 목적함수: 0'*beta + tau*1'*u_plus + (1-tau)*1'*u_minus
        # 등식 제약: Xa*beta + I*u_plus - I*u_minus = y
        # 비음 제약: u_plus >= 0, u_minus >= 0 (beta는 자유변수)

        # beta를 자유변수로 만들기 위해 beta = beta_plus - beta_minus로 분할
        # 변수: [beta_plus (k), beta_minus (k), u_plus (n), u_minus (n)]
        nVars = 2 * k + 2 * n

        # 목적함수 계수
        c = np.zeros(nVars)
        # beta_plus, beta_minus: 0
        # u_plus: tau
        c[2 * k: 2 * k + n] = tau
        # u_minus: 1 - tau
        c[2 * k + n: 2 * k + 2 * n] = 1.0 - tau

        # 등식 제약: Xa*(beta_plus - beta_minus) + I*u_plus - I*u_minus = y
        Aeq = np.zeros((n, nVars))
        Aeq[:, :k] = Xa           # beta_plus
        Aeq[:, k:2*k] = -Xa      # -beta_minus
        Aeq[:, 2*k:2*k+n] = np.eye(n)      # u_plus
        Aeq[:, 2*k+n:2*k+2*n] = -np.eye(n) # -u_minus
        beq = y

        # 모든 변수 >= 0
        bounds = [(0, None)] * nVars

        try:
            result = linprog(
                c, A_eq=Aeq, b_eq=beq, bounds=bounds,
                method='highs', options={'maxiter': 10000}
            )

            if result.success:
                betaPlus = result.x[:k]
                betaMinus = result.x[k:2*k]
                beta = betaPlus - betaMinus
            else:
                # LP 실패 시 OLS 대체
                beta = self._fallbackOLS(Xa, y)
        except Exception:
            beta = self._fallbackOLS(Xa, y)

        if self.fitIntercept:
            self.intercept = beta[0]
            self.coef = beta[1:]
        else:
            self.intercept = 0.0
            self.coef = beta

        self._residuals = y - Xa @ beta
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측값 반환"""
        if self.coef is None:
            raise ValueError("모델이 아직 학습되지 않았습니다.")
        return X @ self.coef + self.intercept

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """학습 잔차"""
        return self._residuals

    @staticmethod
    def _fallbackOLS(Xa: np.ndarray, y: np.ndarray) -> np.ndarray:
        """LP 실패 시 OLS 대체"""
        try:
            return np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(Xa.shape[1])
            beta[0] = np.median(y)
            return beta
