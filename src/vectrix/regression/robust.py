"""
Robust Regression Models

Outlier-robust regression models:
- WLS (Weighted Least Squares)
- HuberRegressor (IRLS)
- RANSACRegressor
- QuantileRegressor

Pure numpy/scipy implementation (no sklearn dependency).
"""

from typing import Optional

import numpy as np
from scipy.optimize import linprog


class WLSRegressor:
    """
    Weighted Least Squares (WLS)

    beta = (X'WX)^{-1} X'Wy

    Heteroscedasticity correction when weights are known.
    Weights w_i should be proportional to the inverse of variance.

    Parameters
    ----------
    fitIntercept : bool
        Whether to include intercept (default: True)
    """

    def __init__(self, fitIntercept: bool = True):
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0
        self._residuals = None
        self._fittedValues = None

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> 'WLSRegressor':
        """
        Estimate regression coefficients via weighted least squares

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable
        weights : np.ndarray, shape (n,)
            Positive weights. Proportional to inverse of variance.

        Returns
        -------
        self
        """
        n, p = X.shape

        # Validate weights
        weights = np.asarray(weights, dtype=float).ravel()
        if len(weights) != n:
            raise ValueError(f"Weight length({len(weights)}) mismatches sample count({n})")
        if np.any(weights < 0):
            raise ValueError("All weights must be positive")

        # Weight matrix (diagonal)
        sqrtW = np.sqrt(weights)

        if self.fitIntercept:
            Xa = np.column_stack([np.ones(n), X])
        else:
            Xa = X.copy()

        # Transform to W^{1/2} X, W^{1/2} y then OLS
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
        """Return predictions"""
        if self.coef is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.coef + self.intercept

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Training residuals"""
        return self._residuals

    @property
    def fittedValues(self) -> Optional[np.ndarray]:
        """Training fitted values"""
        return self._fittedValues


class HuberRegressor:
    """
    Huber M-estimation via IRLS (Iteratively Reweighted Least Squares)

    Outlier-robust regression estimation using the Huber loss function.

    Weight function:
        w(e) = 1               if |e| <= epsilon * scale
        w(e) = epsilon / |e|   if |e| > epsilon * scale

    Iterative algorithm:
        1. Initial beta estimation via OLS
        2. Compute residuals and scale (MAD)
        3. Compute Huber weights
        4. Perform WLS
        5. Repeat until convergence

    Parameters
    ----------
    epsilon : float
        Huber function threshold. Smaller values are more robust (default: 1.35)
    maxIter : int
        Maximum iterations (default: 100)
    tol : float
        Convergence tolerance (default: 1e-4)
    fitIntercept : bool
        Whether to include intercept (default: True)
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
        Huber regression estimation via IRLS algorithm

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable

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

        # Step 1: Initial OLS estimation
        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(k)
            if self.fitIntercept:
                beta[0] = np.median(y)

        for iteration in range(self.maxIter):
            betaOld = beta.copy()

            # Step 2: Compute residuals
            residuals = y - Xa @ beta

            # Step 3: Scale estimation (MAD)
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = mad / 0.6745 if mad > 1e-15 else 1.0
            self.scale = scale

            # Step 4: Compute Huber weights
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

            # Check convergence
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
        """Return predictions"""
        if self.coef is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.coef + self.intercept

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Training residuals"""
        return self._residuals

    @property
    def weights(self) -> Optional[np.ndarray]:
        """Final IRLS weights (smaller for outliers)"""
        return self._weights


class RANSACRegressor:
    """
    Random Sample Consensus (RANSAC)

    Outlier-robust regression via random subsampling.

    Algorithm:
        1. Randomly select minimum sample (p+1)
        2. OLS on subsample
        3. Determine inliers (|residual| < threshold)
        4. Select model with maximum inlier count
        5. Refit on final inliers

    Parameters
    ----------
    minSamples : int, optional
        Minimum subsample size. None defaults to p + 1
    residualThreshold : float, optional
        Inlier threshold. None uses MAD-based auto-computation
    maxTrials : int
        Maximum random trials (default: 100)
    fitIntercept : bool
        Whether to include intercept (default: True)
    randomState : int, optional
        Random seed
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
        Robust regression estimation via RANSAC algorithm

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable

        Returns
        -------
        self
        """
        n, p = X.shape
        rng = np.random.RandomState(self.randomState)

        # Determine minimum sample size
        minSamples = self.minSamples
        if minSamples is None:
            minSamples = p + 1 + (1 if self.fitIntercept else 0)
        minSamples = max(minSamples, p + 1)

        if minSamples >= n:
            # Too few samples, fall back to OLS
            return self._fallbackOLS(X, y, n)

        # Determine residual threshold
        threshold = self.residualThreshold
        if threshold is None:
            # MAD-based: full OLS residual MAD * 3
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
            # Step 1: Random subsample selection
            sampleIdx = rng.choice(n, minSamples, replace=False)
            Xs = X[sampleIdx]
            ys = y[sampleIdx]

            # Step 2: OLS on subsample
            if self.fitIntercept:
                Xa = np.column_stack([np.ones(minSamples), Xs])
            else:
                Xa = Xs

            try:
                betaSample = np.linalg.lstsq(Xa, ys, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            # Step 3: Compute residuals on full data
            if self.fitIntercept:
                XaFull = np.column_stack([np.ones(n), X])
            else:
                XaFull = X
            residuals = np.abs(y - XaFull @ betaSample)

            # Step 4: Determine inliers
            inlierMask = residuals < threshold
            inlierCount = np.sum(inlierMask)

            if inlierCount > bestInlierCount:
                bestInlierCount = inlierCount
                bestInlierMask = inlierMask.copy()
                bestBeta = betaSample.copy()

        self.nTrials = self.maxTrials

        # Step 5: Refit on final inliers
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
            # All trials failed -> OLS fallback
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
        """Return predictions"""
        if self.coef is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.coef + self.intercept

    def _fallbackOLS(self, X: np.ndarray, y: np.ndarray, n: int) -> 'RANSACRegressor':
        """Fall back to OLS when samples are insufficient"""
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

    Regression model that estimates conditional quantiles.
    Median regression (quantile=0.5) is equivalent to LAD (Least Absolute Deviations).

    check function: rho_tau(u) = u * (tau - I(u < 0))

    Solved via Linear Programming:
        min sum rho_tau(y_i - x_i'beta)
      = min tau * u_plus + (1-tau) * u_minus
        s.t. Xa @ beta + u_plus - u_minus = y
             u_plus, u_minus >= 0

    Uses scipy.optimize.linprog.

    Parameters
    ----------
    quantile : float
        Target quantile, 0 < quantile < 1 (default: 0.5)
    fitIntercept : bool
        Whether to include intercept (default: True)
    """

    def __init__(self, quantile: float = 0.5, fitIntercept: bool = True):
        if not 0 < quantile < 1:
            raise ValueError(f"quantile must be in (0, 1): {quantile}")
        self.quantile = quantile
        self.fitIntercept = fitIntercept
        self.coef = None
        self.intercept = 0.0
        self._residuals = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressor':
        """
        Quantile regression estimation via linear programming

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable

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

        # LP formulation:
        # Variables: [beta (k), u_plus (n), u_minus (n)]
        # Objective: 0'*beta + tau*1'*u_plus + (1-tau)*1'*u_minus
        # Equality constraint: Xa*beta + I*u_plus - I*u_minus = y
        # Non-negativity: u_plus >= 0, u_minus >= 0 (beta is free)

        # Split beta = beta_plus - beta_minus to make it free
        # Variables: [beta_plus (k), beta_minus (k), u_plus (n), u_minus (n)]
        nVars = 2 * k + 2 * n

        # Objective function coefficients
        c = np.zeros(nVars)
        # beta_plus, beta_minus: 0
        # u_plus: tau
        c[2 * k: 2 * k + n] = tau
        # u_minus: 1 - tau
        c[2 * k + n: 2 * k + 2 * n] = 1.0 - tau

        # Equality constraint: Xa*(beta_plus - beta_minus) + I*u_plus - I*u_minus = y
        Aeq = np.zeros((n, nVars))
        Aeq[:, :k] = Xa           # beta_plus
        Aeq[:, k:2*k] = -Xa      # -beta_minus
        Aeq[:, 2*k:2*k+n] = np.eye(n)      # u_plus
        Aeq[:, 2*k+n:2*k+2*n] = -np.eye(n) # -u_minus
        beq = y

        # All variables >= 0
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
                # Fall back to OLS on LP failure
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
        """Return predictions"""
        if self.coef is None:
            raise ValueError("Model has not been fitted yet.")
        return X @ self.coef + self.intercept

    @property
    def residuals(self) -> Optional[np.ndarray]:
        """Training residuals"""
        return self._residuals

    @staticmethod
    def _fallbackOLS(Xa: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fall back to OLS on LP failure"""
        try:
            return np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(Xa.shape[1])
            beta[0] = np.median(y)
            return beta
