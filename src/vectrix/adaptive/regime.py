"""
Regime-Aware Adaptive Forecasting

HMM (Hidden Markov Model) based time series regime detection and regime-adaptive forecasting.

Regime types:
- 'growth':   Upward trend + low volatility
- 'decline':  Downward trend + low volatility
- 'volatile': High volatility (trend-independent)
- 'stable':   Low volatility + no trend
- 'crisis':   Sharp decline + very high volatility

Core algorithms:
- Baum-Welch (EM) algorithm for HMM training
- Forward-Backward algorithm (log-space numerical stability)
- Viterbi algorithm for optimal state sequence estimation
- Transition probability weighted ensemble forecasting

Pure numpy/scipy only. No external HMM libraries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Regime detection result"""

    states: np.ndarray = field(default_factory=lambda: np.array([]))
    """Regime number for each time point (0-indexed)"""

    labels: List[str] = field(default_factory=list)
    """Regime label for each time point (e.g., 'growth', 'stable', ...)"""

    regimeHistory: List[Tuple[str, int, int]] = field(default_factory=list)
    """(label, start_index, end_index) interval list"""

    currentRegime: str = ""
    """Regime label at the last time point"""

    transitionMatrix: np.ndarray = field(default_factory=lambda: np.array([]))
    """K x K transition probability matrix"""

    regimeStats: Dict[str, Dict] = field(default_factory=dict)
    """Statistics per regime {'growth': {'mean': ..., 'std': ..., 'trend': ...}, ...}"""

    nRegimes: int = 0
    """Number of detected regimes"""

    logLikelihood: float = 0.0
    """Log-likelihood after HMM training"""


@dataclass
class RegimeForecastResult:
    """Regime-aware adaptive forecast result"""

    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    """Prediction array"""

    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    """95% lower confidence bound"""

    upper95: np.ndarray = field(default_factory=lambda: np.array([]))
    """95% upper confidence bound"""

    currentRegime: str = ""
    """Current (last) regime"""

    regimeHistory: List[Tuple[str, int, int]] = field(default_factory=list)
    """Regime transition history"""

    transitionMatrix: np.ndarray = field(default_factory=lambda: np.array([]))
    """Transition probability matrix"""

    regimeStats: Dict[str, Dict] = field(default_factory=dict)
    """Per-regime statistics"""

    modelPerRegime: Dict[str, str] = field(default_factory=dict)
    """{regime_label: model_id_used}"""

    regimeProbabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    """Regime probabilities for each forecast step [steps x K]"""

    scenarioForecasts: Dict[str, np.ndarray] = field(default_factory=dict)
    """Forecast per regime scenario {regime_label: predictions}"""


# ---------------------------------------------------------------------------
# Numerical Utilities
# ---------------------------------------------------------------------------

def _logSumExp(logA: np.ndarray) -> float:
    """
    log-sum-exp trick: log(sum(exp(logA)))

    Subtracts max for numerical stability, then adds it back.
    """
    maxVal = np.max(logA)
    if maxVal == -np.inf:
        return -np.inf
    return maxVal + np.log(np.sum(np.exp(logA - maxVal)))


def _logSumExpAxis(logA: np.ndarray, axis: int) -> np.ndarray:
    """log-sum-exp along a specific axis of a multi-dimensional array"""
    maxVal = np.max(logA, axis=axis, keepdims=True)
    # Handle -inf cases
    mask = np.isfinite(maxVal)
    safe = np.where(mask, maxVal, 0.0)
    result = safe.squeeze(axis) + np.log(
        np.sum(np.exp(logA - np.where(mask, maxVal, 0.0)), axis=axis)
    )
    # Rows/columns that are all -inf remain -inf
    allInf = ~np.any(np.isfinite(logA), axis=axis)
    result[allInf] = -np.inf
    return result


def _logGaussianPdf(x: float, mu: float, sigma2: float) -> float:
    """Log of Gaussian PDF: log N(x | mu, sigma2)"""
    if sigma2 <= 0:
        sigma2 = 1e-10
    return -0.5 * np.log(2 * np.pi * sigma2) - 0.5 * (x - mu) ** 2 / sigma2


# ---------------------------------------------------------------------------
# RegimeDetector: HMM-based Regime Detection
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    HMM-based time series regime detector

    Estimates hidden states (regimes) of a time series using a Hidden Markov Model
    with Gaussian observation model.

    Uses log returns as observations, where each state is modeled as a Gaussian
    distribution with unique mean and variance.

    Parameters
    ----------
    nRegimes : int
        Number of regimes (states) to detect (default 3)
    maxIter : int
        Maximum iterations for Baum-Welch EM algorithm (default 100)

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
            raise ValueError("nRegimes must be at least 2.")
        self.nRegimes = nRegimes
        self.maxIter = maxIter

        # HMM parameters (set after training)
        self.pi: Optional[np.ndarray] = None          # Initial state probabilities [K]
        self.transitionMatrix: Optional[np.ndarray] = None  # Transition matrix [K x K]
        self.means: Optional[np.ndarray] = None        # State means [K]
        self.variances: Optional[np.ndarray] = None    # State variances [K]

        self._fitted = False
        self._logLikelihood = -np.inf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, y: np.ndarray) -> RegimeResult:
        """
        Run regime detection

        Parameters
        ----------
        y : np.ndarray
            Original time series data (level)

        Returns
        -------
        RegimeResult
            Detection result (states, labels, transition matrix, statistics, etc.)
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) < 10:
            raise ValueError("Regime detection requires at least 10 data points.")

        # Observations = log returns
        observations = self._computeReturns(y)

        # Adaptive regime count: reduce if data is short
        effectiveRegimes = min(self.nRegimes, max(2, len(observations) // 20))

        # Backup original nRegimes then adapt
        origRegimes = self.nRegimes
        self.nRegimes = effectiveRegimes

        # HMM training (Baum-Welch)
        self._fitHMM(observations)

        # Optimal state sequence (Viterbi)
        states = self._viterbi(observations)

        # Convert states to meaningful labels
        labels = self._labelRegimes(y, states)

        # Build regime history
        regimeHistory = self._buildRegimeHistory(labels)

        # Per-regime statistics
        regimeStats = self._computeRegimeStats(y, states, labels)

        # Transition matrix (use trained one)
        transitionMatrix = self.transitionMatrix.copy()

        currentRegime = labels[-1] if labels else "stable"

        # Restore nRegimes
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
    # Observation Computation
    # ------------------------------------------------------------------

    def _computeReturns(self, y: np.ndarray) -> np.ndarray:
        """
        Compute log returns: log(y[t] / y[t-1])

        Safely handles zero and negative values.
        """
        # Safety: replace values <= 0 with small positive number
        safeY = y.copy()
        minPositive = np.min(safeY[safeY > 0]) if np.any(safeY > 0) else 1.0
        safeY[safeY <= 0] = minPositive * 0.01

        returns = np.diff(np.log(safeY))

        # Handle NaN/Inf
        mask = ~np.isfinite(returns)
        if np.any(mask):
            median = np.nanmedian(returns[~mask]) if np.any(~mask) else 0.0
            returns[mask] = median

        return returns

    # ------------------------------------------------------------------
    # HMM Training: Baum-Welch (EM)
    # ------------------------------------------------------------------

    def _fitHMM(self, observations: np.ndarray) -> None:
        """
        Train HMM using Baum-Welch (EM) algorithm

        All computations performed in log-space for numerical stability.

        Parameters
        ----------
        observations : np.ndarray
            Observation sequence (log returns)
        """
        T = len(observations)
        K = self.nRegimes

        # --- Parameter initialization ---
        self._initializeParams(observations)

        prevLogLik = -np.inf
        tolerance = 1e-6

        for iteration in range(self.maxIter):
            # --- E-step: Forward-Backward ---
            logAlpha, logLik = self._forward(observations)
            logBeta = self._backward(observations)

            # Convergence check
            if abs(logLik - prevLogLik) < tolerance and iteration > 5:
                break
            prevLogLik = logLik

            # gamma[t][k] = P(state_t = k | Y): posterior state probability
            logGamma = logAlpha + logBeta
            # Normalize: sum to 1 at each time step
            logGammaNorm = _logSumExpAxis(logGamma, axis=1)
            logGamma = logGamma - logGammaNorm[:, np.newaxis]

            gamma = np.exp(logGamma)
            # Safety guard
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
                # Normalize
                norm = _logSumExp(logXi[t].ravel())
                if np.isfinite(norm):
                    logXi[t] -= norm

            xi = np.exp(logXi)
            xi = np.clip(xi, 1e-300, None)

            # --- M-step: Parameter update ---

            # Initial state probabilities
            self.pi = gamma[0] / gamma[0].sum()
            self.pi = np.clip(self.pi, 1e-10, None)
            self.pi /= self.pi.sum()

            # Transition matrix
            xiSumOverT = xi.sum(axis=0)  # [K x K]
            gammaSumForTrans = gamma[:-1].sum(axis=0)  # [K]
            gammaSumForTrans = np.where(
                gammaSumForTrans < 1e-300, 1e-300, gammaSumForTrans
            )
            self.transitionMatrix = xiSumOverT / gammaSumForTrans[:, np.newaxis]
            # Row normalization
            rowSums = self.transitionMatrix.sum(axis=1, keepdims=True)
            rowSums = np.where(rowSums < 1e-300, 1e-300, rowSums)
            self.transitionMatrix /= rowSums
            # Numerical stability clipping
            self.transitionMatrix = np.clip(self.transitionMatrix, 1e-10, None)
            self.transitionMatrix /= self.transitionMatrix.sum(axis=1, keepdims=True)

            # Observation model parameters (Gaussian)
            for k in range(K):
                wk = gamma[:, k]
                wkSum = wk.sum()
                if wkSum < 1e-300:
                    continue

                # Mean
                self.means[k] = np.dot(wk, observations) / wkSum

                # Variance
                diff = observations - self.means[k]
                self.variances[k] = np.dot(wk, diff ** 2) / wkSum
                # Ensure minimum variance
                self.variances[k] = max(self.variances[k], 1e-10)

        self._logLikelihood = prevLogLik
        self._fitted = True

    def _initializeParams(self, observations: np.ndarray) -> None:
        """
        K-means style HMM parameter initialization

        Splits observations into K clusters based on quantiles to set initial means/variances.
        """
        K = self.nRegimes
        T = len(observations)

        # Initial state probabilities: uniform
        self.pi = np.ones(K) / K

        # Transition matrix: diagonal-dominant (high self-persistence probability)
        self.transitionMatrix = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(self.transitionMatrix, 0.95)
        # Row normalization
        self.transitionMatrix /= self.transitionMatrix.sum(axis=1, keepdims=True)

        # Observation model: quantile-based initialization
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

        # Sort means for interpretability
        sortIdx = np.argsort(self.means)
        self.means = self.means[sortIdx]
        self.variances = self.variances[sortIdx]

    # ------------------------------------------------------------------
    # Forward Algorithm (log-space)
    # ------------------------------------------------------------------

    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward algorithm (log-space)

        alpha[t][k] = p(y_1,...,y_t, state_t=k)

        log-space:
            logAlpha[t][k] = log(p(y_t | state=k))
                            + logSumExp_j(logAlpha[t-1][j] + log(A[j][k]))

        Parameters
        ----------
        observations : np.ndarray
            Observation sequence

        Returns
        -------
        logAlpha : np.ndarray
            [T x K] forward variables (log)
        logLikelihood : float
            Log-likelihood of the full sequence
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

        # Log-likelihood
        logLikelihood = _logSumExp(logAlpha[T - 1, :])

        return logAlpha, logLikelihood

    # ------------------------------------------------------------------
    # Backward Algorithm (log-space)
    # ------------------------------------------------------------------

    def _backward(self, observations: np.ndarray) -> np.ndarray:
        """
        Backward algorithm (log-space)

        beta[t][k] = p(y_{t+1},...,y_T | state_t=k)

        log-space:
            logBeta[t][k] = logSumExp_j(log(A[k][j])
                            + log(p(y_{t+1} | state=j))
                            + logBeta[t+1][j])

        Parameters
        ----------
        observations : np.ndarray
            Observation sequence

        Returns
        -------
        logBeta : np.ndarray
            [T x K] backward variables (log)
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
        Optimal state sequence estimation via Viterbi algorithm (log-space)

        delta[t][k] = max_j(delta[t-1][j] * A[j][k]) * p(y_t | state=k)

        log-space:
            logDelta[t][k] = log(p(y_t | state=k))
                            + max_j(logDelta[t-1][j] + log(A[j][k]))

        Parameters
        ----------
        observations : np.ndarray
            Observation sequence

        Returns
        -------
        states : np.ndarray
            Optimal state sequence [T] (0-indexed)
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

        # Extend states from observation sequence length (T) to original series length (T+1)
        # observations are diff so length is original-1. First point gets first observation's state.
        fullStates = np.empty(T + 1, dtype=int)
        fullStates[0] = states[0]
        fullStates[1:] = states

        return fullStates

    # ------------------------------------------------------------------
    # State to Label Conversion
    # ------------------------------------------------------------------

    def _labelRegimes(self, y: np.ndarray, states: np.ndarray) -> List[str]:
        """
        Convert state numbers to meaningful labels

        Analyzes statistics of each HMM state to determine regime type:
        - Mean return > 0.01  --> 'growth'
        - Mean return < -0.01 --> 'decline'
        - std > median_std * 1.5 --> 'volatile'
        - Mean return < -0.03 AND std > median_std * 2 --> 'crisis'
        - Otherwise --> 'stable'

        Parameters
        ----------
        y : np.ndarray
            Original time series
        states : np.ndarray
            State sequence (length = len(y))

        Returns
        -------
        List[str]
            Label for each time point
        """
        K = self.nRegimes
        returns = self._computeReturns(y)
        # states has length len(y), align with returns (len(y)-1)
        # states[1:] corresponds to returns
        stateForReturns = states[1:]  # len = len(returns)

        # Statistics per state
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

        # Label mapping: state number -> label
        stateToLabel: Dict[int, str] = {}

        for k in range(K):
            mr = stateMeanReturn[k]
            sd = stateStdReturn[k]

            # crisis: sharp decline + very high volatility
            if mr < -0.03 and sd > medianStd * 2:
                stateToLabel[k] = "crisis"
            # volatile: high volatility
            elif sd > medianStd * 1.5:
                stateToLabel[k] = "volatile"
            # growth: upward trend
            elif mr > 0.01:
                stateToLabel[k] = "growth"
            # decline: downward trend
            elif mr < -0.01:
                stateToLabel[k] = "decline"
            # stable: everything else
            else:
                stateToLabel[k] = "stable"

        # Handle duplicate labels: if same label assigned to multiple states,
        # re-label based on volatility for differentiation
        usedLabels: Dict[str, List[int]] = {}
        for k, label in stateToLabel.items():
            usedLabels.setdefault(label, []).append(k)

        for label, stateList in usedLabels.items():
            if len(stateList) > 1:
                # Sort by volatility ascending
                stateList.sort(key=lambda s: stateStdReturn[s])
                for idx, s in enumerate(stateList):
                    if idx == 0:
                        pass  # Keep original label for lowest volatility state
                    else:
                        # Assign different label to higher volatility states
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
    # Regime History Interval Construction
    # ------------------------------------------------------------------

    def _buildRegimeHistory(self, labels: List[str]) -> List[Tuple[str, int, int]]:
        """Group consecutive identical regimes into intervals"""
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
    # Per-Regime Statistics Computation
    # ------------------------------------------------------------------

    def _computeRegimeStats(
        self, y: np.ndarray, states: np.ndarray, labels: List[str]
    ) -> Dict[str, Dict]:
        """
        Compute statistics for each regime

        Returns
        -------
        Dict[str, Dict]
            {regime_label: {'mean': float, 'std': float, 'trend': float,
                           'count': int, 'proportion': float}}
        """
        uniqueLabels = sorted(set(labels))
        stats: Dict[str, Dict] = {}
        totalLen = len(y)

        returns = self._computeReturns(y)

        for label in uniqueLabels:
            # Indices belonging to this regime
            indices = [i for i, lb in enumerate(labels) if lb == label]
            if not indices:
                continue

            segmentValues = y[indices]
            # Return indices (corresponding to labels[1:])
            returnIndices = [i - 1 for i in indices if 0 < i <= len(returns)]
            segmentReturns = returns[returnIndices] if returnIndices else np.array([0.0])

            # Trend: linear regression slope
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
# RegimeAwareForecaster: Regime-Aware Adaptive Forecaster
# ---------------------------------------------------------------------------

class RegimeAwareForecaster:
    """
    Regime-aware adaptive forecaster

    Automatically selects optimal models for each regime and generates forecasts.
    Combines forecasts from multiple regime scenarios using transition probabilities.

    Per-regime model mapping:
    - growth   -> auto_ets or theta (models strong at trend tracking)
    - decline  -> rwd or auto_ets (conservative decline reflection)
    - volatile -> garch or window_avg (volatility modeling)
    - stable   -> mean or naive (stable prediction)
    - crisis   -> seasonal_naive (conservative approach during sharp changes)

    Returns only model ID strings for execution by the Vectrix engine.

    Parameters
    ----------
    nRegimes : int
        Number of regimes to detect (default 3)
    period : int
        Seasonal period (default 7)

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

    # Default model mapping per regime
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
    # Public API
    # ------------------------------------------------------------------

    def forecast(
        self,
        y: np.ndarray,
        steps: int,
        period: int = 7,
    ) -> RegimeForecastResult:
        """
        Run regime-aware forecasting

        Procedure:
        1. Regime detection (HMM)
        2. Identify current regime
        3. Select optimal model per regime
        4. Compute ensemble weights based on transition probabilities
        5. Generate forecasts per regime scenario (simple statistics-based)
        6. Weighted combination + confidence intervals
        7. Return regime transition scenarios

        Parameters
        ----------
        y : np.ndarray
            Time series data
        steps : int
            Number of forecast steps
        period : int
            Seasonal period

        Returns
        -------
        RegimeForecastResult
            Regime-aware forecast result
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        if len(y) < 10:
            raise ValueError("Forecasting requires at least 10 data points.")

        self.period = period

        # 1. Regime detection
        regimeResult = self.detector.detect(y)
        self._regimeResult = regimeResult

        currentRegime = regimeResult.currentRegime
        K = regimeResult.nRegimes
        transMatrix = regimeResult.transitionMatrix

        # 2. Model selection per regime
        uniqueLabels = sorted(set(regimeResult.labels))
        modelPerRegime: Dict[str, str] = {}
        for label in uniqueLabels:
            modelPerRegime[label] = self._selectModelForRegime(label, y, period)

        # 3. Scenario forecasts per regime (simple statistics-based)
        scenarioForecasts: Dict[str, np.ndarray] = {}
        scenarioStds: Dict[str, float] = {}

        for label in uniqueLabels:
            pred, predStd = self._generateRegimeScenarioForecast(
                y, label, regimeResult, steps, period
            )
            scenarioForecasts[label] = pred
            scenarioStds[label] = predStd

        # 4. Transition probability weighted forecast
        predictions, regimeProbabilities = self._transitionWeightedForecast(
            y, steps, period, currentRegime, regimeResult
        )

        # 5. Confidence interval computation
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
    # Per-Regime Model Selection
    # ------------------------------------------------------------------

    def _selectModelForRegime(
        self, regimeLabel: str, y: np.ndarray, period: int
    ) -> str:
        """
        Return optimal model ID for a regime

        Selects appropriate model from candidates based on data length.

        Parameters
        ----------
        regimeLabel : str
            Regime label
        y : np.ndarray
            Time series data
        period : int
            Seasonal period

        Returns
        -------
        str
            Model ID (for Vectrix engine)
        """
        n = len(y)
        candidates = self.REGIME_MODEL_MAP.get(regimeLabel, ["auto_ets"])

        # Filter based on data length requirements
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

        # If all candidates lack data, use simplest model
        return "mean" if n >= 2 else "naive"

    # ------------------------------------------------------------------
    # Per-Regime Scenario Forecast Generation
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
        Scenario forecast assuming a specific regime persists

        Generates simple statistics-based forecasts using the statistical properties
        (trend, volatility) of the corresponding regime segment.

        Parameters
        ----------
        y : np.ndarray
            Original time series
        regimeLabel : str
            Scenario regime
        regimeResult : RegimeResult
            Regime detection result
        steps : int
            Number of forecast steps
        period : int
            Seasonal period

        Returns
        -------
        Tuple[np.ndarray, float]
            (predictions, prediction standard deviation)
        """
        stats = regimeResult.regimeStats.get(regimeLabel, {})
        meanReturn = stats.get("meanReturn", 0.0)
        stdReturn = stats.get("stdReturn", 0.01)
        trend = stats.get("trend", 0.0)

        lastValue = y[-1]
        predictions = np.zeros(steps)

        if regimeLabel == "growth":
            # Reflect upward trend: grow from last value by mean return
            for h in range(steps):
                if h == 0:
                    predictions[h] = lastValue * (1 + meanReturn)
                else:
                    predictions[h] = predictions[h - 1] * (1 + meanReturn)

        elif regimeLabel == "decline":
            # Downward trend: damped decline (gradually attenuated)
            dampFactor = 0.95
            for h in range(steps):
                dampedReturn = meanReturn * (dampFactor ** h)
                if h == 0:
                    predictions[h] = lastValue * (1 + dampedReturn)
                else:
                    predictions[h] = predictions[h - 1] * (1 + dampedReturn)

        elif regimeLabel == "volatile":
            # Volatility: mean reversion + wide variation
            regimeMean = stats.get("mean", lastValue)
            for h in range(steps):
                # Gradual reversion to mean
                alpha = min(0.1 * (h + 1), 1.0)
                predictions[h] = lastValue * (1 - alpha) + regimeMean * alpha

        elif regimeLabel == "crisis":
            # Crisis: sharp drop then stabilization
            for h in range(steps):
                # Sharp decline gradually attenuated
                dampedReturn = meanReturn * (0.8 ** h)
                if h == 0:
                    predictions[h] = lastValue * (1 + dampedReturn)
                else:
                    predictions[h] = predictions[h - 1] * (1 + dampedReturn)

        elif regimeLabel == "stable":
            # Stable: minimal change
            predictions[:] = lastValue
            # Slight trend reflection
            if abs(trend) > 0:
                trendPerStep = trend / max(stats.get("count", 1), 1)
                for h in range(steps):
                    predictions[h] = lastValue + trendPerStep * (h + 1)

        else:
            predictions[:] = lastValue

        return predictions, float(stdReturn * lastValue) if lastValue != 0 else float(stdReturn)

    # ------------------------------------------------------------------
    # Transition Probability Weighted Forecast
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
        Transition probability weighted forecast: incorporates transition probabilities to other regimes

        If current regime is growth with transition probabilities:
        - P(growth->growth) = 0.8
        - P(growth->stable) = 0.15
        - P(growth->decline) = 0.05
        Then forecast weights:
        - growth forecast * 0.8 + stable forecast * 0.15 + decline forecast * 0.05

        For long-term forecasts, converges via matrix power:
        - Weights at step h = row of transitionMatrix^h for currentRegime

        Parameters
        ----------
        y : np.ndarray
            Time series data
        steps : int
            Number of forecast steps
        period : int
            Seasonal period
        currentRegime : str
            Current regime
        regimeResult : RegimeResult
            Regime detection result

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (weighted predictions [steps], regime probabilities [steps x K])
        """
        K = regimeResult.nRegimes
        transMatrix = regimeResult.transitionMatrix  # [K x K]

        # Build state number to label mapping
        uniqueLabels = sorted(set(regimeResult.labels))
        stateToLabel: Dict[int, str] = {}
        labelToState: Dict[str, int] = {}

        # Determine most frequent label for each HMM state
        for k in range(K):
            mask = regimeResult.states == k
            if np.sum(mask) > 0:
                labelsForState = [regimeResult.labels[i] for i in range(len(regimeResult.labels)) if mask[i]]
                # Most frequent label
                labelCounts: Dict[str, int] = {}
                for lb in labelsForState:
                    labelCounts[lb] = labelCounts.get(lb, 0) + 1
                bestLabel = max(labelCounts, key=labelCounts.get)  # type: ignore
                stateToLabel[k] = bestLabel
                labelToState[bestLabel] = k
            else:
                stateToLabel[k] = "stable"

        # State index corresponding to current regime
        currentStateIdx = labelToState.get(currentRegime, 0)

        # Generate scenario forecasts per regime
        scenarioPreds: Dict[int, np.ndarray] = {}
        for k in range(K):
            label = stateToLabel.get(k, "stable")
            pred, _ = self._generateRegimeScenarioForecast(
                y, label, regimeResult, steps, period
            )
            scenarioPreds[k] = pred

        # Regime probabilities at step h: row of transMatrix^h for currentState
        regimeProbabilities = np.zeros((steps, K))
        weightedPredictions = np.zeros(steps)

        # Cumulative matrix power
        matPower = np.eye(K)  # A^0 = I

        for h in range(steps):
            # h=0: row of A^1 for currentState
            matPower = matPower @ transMatrix

            regimeProbs = matPower[currentStateIdx, :]
            # Numerical stabilization
            regimeProbs = np.clip(regimeProbs, 0, None)
            probSum = regimeProbs.sum()
            if probSum > 0:
                regimeProbs /= probSum

            regimeProbabilities[h, :] = regimeProbs

            # Weighted forecast
            for k in range(K):
                weightedPredictions[h] += regimeProbs[k] * scenarioPreds[k][h]

        return weightedPredictions, regimeProbabilities

    # ------------------------------------------------------------------
    # Confidence Interval Computation
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
        Compute 95% confidence intervals reflecting regime uncertainty

        Confidence intervals combine two sources of uncertainty:
        1. Intrinsic volatility within each regime
        2. Regime transition uncertainty (when forecasts differ across regimes)

        Parameters
        ----------
        predictions : np.ndarray
            Weighted predictions
        y : np.ndarray
            Original time series
        steps : int
            Number of forecast steps
        regimeProbabilities : np.ndarray
            [steps x K] regime probabilities
        regimeResult : RegimeResult
            Regime detection result

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (lower95, upper95)
        """
        K = regimeResult.nRegimes

        # State -> label mapping
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

        # Volatility (std) per regime
        regimeStds = np.zeros(K)
        for k in range(K):
            label = stateToLabel.get(k, "stable")
            stats = regimeResult.regimeStats.get(label, {})
            regimeStds[k] = stats.get("std", np.std(y))

        # Base uncertainty: std of overall data
        baseStd = np.std(y[-min(60, len(y)):])

        margin = np.zeros(steps)
        for h in range(steps):
            # 1. Weighted sum of within-regime volatility
            withinVar = 0.0
            for k in range(K):
                withinVar += regimeProbabilities[h, k] * regimeStds[k] ** 2

            # 2. Between-regime forecast variance (scenario differences)
            betweenVar = 0.0
            for k in range(K):
                label = stateToLabel.get(k, "stable")
                scenarioPred, _ = self._generateRegimeScenarioForecast(
                    y, label, regimeResult, steps, self.period
                )
                betweenVar += regimeProbabilities[h, k] * (scenarioPred[h] - predictions[h]) ** 2

            # Total variance = within + between (Law of Total Variance)
            totalVar = withinVar + betweenVar + baseStd ** 2

            # Uncertainty increases over time (sqrt(h+1))
            margin[h] = 1.96 * np.sqrt(totalVar) * np.sqrt(h + 1)

        lower95 = predictions - margin
        upper95 = predictions + margin

        return lower95, upper95
