"""
Automatic Changepoint Detection

Automatically detects points where statistical properties change in time series data:
- PELT (Pruned Exact Linear Time): Gaussian log-likelihood based, BIC/custom penalty
- CUSUM (Cumulative Sum): Bidirectional cumulative sum based detection
- BOCPD (Bayesian Online Changepoint Detection): Simplified Bayesian online detection
- Auto: Consensus-based automatic detection from multiple methods

Uses pure numpy/scipy only
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ChangePointResult:
    """Changepoint detection result"""
    indices: np.ndarray          # Changepoint positions
    nChangepoints: int           # Number of changepoints
    confidence: np.ndarray       # Confidence of each changepoint (0~1)
    segments: List[Dict]         # Statistics for each segment (mean, std, trend)
    method: str                  # Method used


class ChangePointDetector:
    """
    Time Series Changepoint Detector

    Supports PELT, CUSUM, BOCPD, and Auto methods.
    In Auto mode, changepoints are determined through consensus of three methods.

    Examples
    --------
    >>> detector = ChangePointDetector()
    >>> y = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
    >>> result = detector.detect(y, method='pelt')
    >>> print(result.indices)
    """

    def detect(
        self,
        y: np.ndarray,
        method: str = 'auto',
        minSize: int = 10,
        penalty: str = 'bic'
    ) -> ChangePointResult:
        """
        Detect changepoints in a time series

        Parameters
        ----------
        y : np.ndarray
            Time series data (1-dimensional)
        method : str
            Detection method ('pelt', 'cusum', 'bocpd', 'auto')
        minSize : int
            Minimum segment size
        penalty : str or float
            Penalty type ('bic') or custom value

        Returns
        -------
        ChangePointResult
            Detected changepoint information
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        if n < 2 * minSize:
            return ChangePointResult(
                indices=np.array([], dtype=int),
                nChangepoints=0,
                confidence=np.array([], dtype=float),
                segments=[self._computeSegmentStats(y, 0, n)],
                method=method
            )

        try:
            if method == 'pelt':
                indices, confidence = self._detectPELT(y, minSize, penalty)
            elif method == 'cusum':
                indices, confidence = self._detectCUSUM(y, minSize)
            elif method == 'bocpd':
                indices, confidence = self._detectBOCPD(y, minSize)
            elif method == 'auto':
                indices, confidence = self._detectAuto(y, minSize, penalty)
            else:
                raise ValueError(f"Unknown method: {method}. Choose from 'pelt', 'cusum', 'bocpd', 'auto'")
        except Exception:
            # Graceful fallback: return no changepoints
            indices = np.array([], dtype=int)
            confidence = np.array([], dtype=float)

        # Compute segment statistics
        segments = self._computeAllSegmentStats(y, indices)

        return ChangePointResult(
            indices=indices,
            nChangepoints=len(indices),
            confidence=confidence,
            segments=segments,
            method=method
        )

    # ─── PELT (Pruned Exact Linear Time) ──────────────────────────────────

    def _detectPELT(
        self,
        y: np.ndarray,
        minSize: int,
        penalty: str = 'bic'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect changepoints using the PELT algorithm

        Cost function: Gaussian log-likelihood (mean + variance change)
        Penalty: BIC = log(n) or user-specified
        """
        n = len(y)

        # Compute penalty value
        if isinstance(penalty, str) and penalty.lower() == 'bic':
            pen = np.log(n)
        elif isinstance(penalty, (int, float)):
            pen = float(penalty)
        else:
            pen = np.log(n)

        # Precompute cumulative sums (for O(1) cost function computation)
        cumSum = np.zeros(n + 1)
        cumSumSq = np.zeros(n + 1)
        cumSum[1:] = np.cumsum(y)
        cumSumSq[1:] = np.cumsum(y ** 2)

        def cost(start: int, end: int) -> float:
            """Gaussian log-likelihood cost for segment [start, end)"""
            length = end - start
            if length <= 1:
                return 0.0
            s = cumSum[end] - cumSum[start]
            ss = cumSumSq[end] - cumSumSq[start]
            mean = s / length
            variance = ss / length - mean ** 2
            if variance <= 1e-12:
                return 0.0
            # -2 * log-likelihood (excluding constant terms)
            return length * (np.log(max(variance, 1e-20)) + 1.0)

        # DP array
        F = np.full(n + 1, np.inf)
        F[0] = -pen  # -pen at start (offsets pen of the first segment)
        lastChange = np.zeros(n + 1, dtype=int)

        # Candidate set for PELT pruning
        candidates = [0]

        for tStar in range(minSize, n + 1):
            bestCost = np.inf
            bestIdx = 0
            newCandidates = []

            for t in candidates:
                if tStar - t < minSize:
                    newCandidates.append(t)
                    continue
                c = F[t] + cost(t, tStar) + pen
                if c < bestCost:
                    bestCost = c
                    bestIdx = t
                # PELT pruning: keep if F[t] + cost(t, tStar) <= F[tStar]
                if F[t] + cost(t, tStar) <= bestCost:
                    newCandidates.append(t)

            F[tStar] = bestCost
            lastChange[tStar] = bestIdx
            newCandidates.append(tStar)
            candidates = newCandidates

        # Backtrack changepoints
        changepoints = []
        idx = n
        while idx > 0:
            cp = lastChange[idx]
            if cp > 0:
                changepoints.append(cp)
            idx = cp

        changepoints = sorted(changepoints)
        indices = np.array(changepoints, dtype=int)

        # Compute confidence (based on statistical difference between adjacent segments)
        confidence = self._computeConfidence(y, indices, minSize)

        return indices, confidence

    # ─── CUSUM (Cumulative Sum) ───────────────────────────────────────────

    def _detectCUSUM(
        self,
        y: np.ndarray,
        minSize: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bidirectional CUSUM-based changepoint detection

        Computes forward and backward CUSUM to detect changepoint candidates.
        threshold = 5 * sigma
        """
        n = len(y)
        mu = np.mean(y)
        sigma = max(np.std(y), 1e-10)
        threshold = 5.0 * sigma

        # Forward CUSUM (positive direction)
        cusumPos = np.zeros(n)
        cusumNeg = np.zeros(n)
        for t in range(1, n):
            cusumPos[t] = max(0, cusumPos[t - 1] + (y[t] - mu))
            cusumNeg[t] = min(0, cusumNeg[t - 1] + (y[t] - mu))

        # Backward CUSUM
        cusumPosRev = np.zeros(n)
        cusumNegRev = np.zeros(n)
        for t in range(n - 2, -1, -1):
            cusumPosRev[t] = max(0, cusumPosRev[t + 1] + (y[t] - mu))
            cusumNegRev[t] = min(0, cusumNegRev[t + 1] + (y[t] - mu))

        # Combined CUSUM statistic
        cusumStat = np.abs(cusumPos) + np.abs(cusumNeg)
        cusumStatRev = np.abs(cusumPosRev) + np.abs(cusumNegRev)
        combinedStat = cusumStat + cusumStatRev

        # Search for changepoint candidates at points exceeding threshold
        candidates = np.where(combinedStat > threshold)[0]
        if len(candidates) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # Cluster candidates to refine changepoints
        indices = self._clusterChangepoints(candidates, combinedStat, minSize)

        # Compute confidence
        confidence = self._computeConfidence(y, indices, minSize)

        return indices, confidence

    # ─── BOCPD (Bayesian Online Changepoint Detection) ────────────────────

    def _detectBOCPD(
        self,
        y: np.ndarray,
        minSize: int,
        hazardLambda: float = 250.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simplified Bayesian Online Changepoint Detection (BOCPD)

        Infers changepoints by computing the run-length distribution.
        hazard function: H(r) = 1/lambda (constant hazard rate)

        Parameters
        ----------
        y : np.ndarray
            Time series data
        minSize : int
            Minimum segment size
        hazardLambda : float
            Lambda parameter for the hazard function (expected run length)
        """
        n = len(y)
        hazard = 1.0 / hazardLambda

        # Run-length probability matrix (memory efficient: keep only current/previous)
        # R[t, r] = P(run length = r at time t)
        maxRunLen = n + 1

        # Sufficient statistics for Student-t predictive distribution
        # Maintained for each possible run length
        mu0 = np.mean(y)
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = np.var(y) if np.var(y) > 0 else 1.0

        # Sufficient statistics per current run length
        muN = np.full(maxRunLen, mu0)
        kappaN = np.full(maxRunLen, kappa0)
        alphaN = np.full(maxRunLen, alpha0)
        betaN = np.full(maxRunLen, beta0)

        # Run-length probabilities
        runLenProb = np.zeros(maxRunLen)
        runLenProb[0] = 1.0  # Initial: probability 1 at run length 0

        # Accumulated changepoint probabilities
        cpProb = np.zeros(n)

        for t in range(n):
            # Current data point
            xt = y[t]

            # Predictive probability for each run length (Student-t)
            predProb = np.zeros(t + 1)
            for r in range(t + 1):
                predProb[r] = self._studentTPdf(
                    xt,
                    muN[r],
                    betaN[r] * (kappaN[r] + 1) / (alphaN[r] * kappaN[r]),
                    2.0 * alphaN[r]
                )

            # Growth probability: P(r_{t+1} = r+1) = P(r_t = r) * pi(x_t | r_t = r) * (1-H)
            growthProb = runLenProb[:t + 1] * predProb * (1 - hazard)

            # Changepoint probability: P(r_{t+1} = 0) = sum P(r_t = r) * pi(x_t | r_t = r) * H
            cpMass = np.sum(runLenProb[:t + 1] * predProb * hazard)

            # New run-length probabilities
            newRunLenProb = np.zeros(maxRunLen)
            newRunLenProb[0] = cpMass
            newRunLenProb[1:t + 2] = growthProb

            # Normalize
            total = np.sum(newRunLenProb)
            if total > 0:
                newRunLenProb /= total

            runLenProb = newRunLenProb

            # Changepoint probability = P(r_t = 0)
            cpProb[t] = runLenProb[0]

            # Update sufficient statistics (for each run length)
            newMuN = np.full(maxRunLen, mu0)
            newKappaN = np.full(maxRunLen, kappa0)
            newAlphaN = np.full(maxRunLen, alpha0)
            newBetaN = np.full(maxRunLen, beta0)

            for r in range(min(t + 1, maxRunLen - 1)):
                k = kappaN[r]
                m = muN[r]
                a = alphaN[r]
                b = betaN[r]

                newKappaN[r + 1] = k + 1
                newMuN[r + 1] = (k * m + xt) / (k + 1)
                newAlphaN[r + 1] = a + 0.5
                newBetaN[r + 1] = b + 0.5 * k * (xt - m) ** 2 / (k + 1)

            muN = newMuN
            kappaN = newKappaN
            alphaN = newAlphaN
            betaN = newBetaN

        # Extract peaks from changepoint probabilities
        indices = self._extractPeaks(cpProb, minSize, threshold=0.1)

        # Confidence is directly extracted from changepoint probabilities
        if len(indices) > 0:
            confidence = np.clip(cpProb[indices], 0, 1)
        else:
            confidence = np.array([], dtype=float)

        return indices, confidence

    def _studentTPdf(
        self,
        x: float,
        mu: float,
        varScale: float,
        nu: float
    ) -> float:
        """Student-t probability density function"""
        try:
            from scipy.special import gammaln
            sigma2 = max(varScale, 1e-10)
            z = (x - mu) ** 2 / sigma2
            logp = (
                gammaln((nu + 1) / 2)
                - gammaln(nu / 2)
                - 0.5 * np.log(nu * np.pi * sigma2)
                - (nu + 1) / 2 * np.log(1 + z / nu)
            )
            return np.exp(logp)
        except Exception:
            # Fallback: Gaussian approximation
            sigma = np.sqrt(max(varScale, 1e-10))
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # ─── Auto (Consensus-based) ──────────────────────────────────────────

    def _detectAuto(
        self,
        y: np.ndarray,
        minSize: int,
        penalty: str = 'bic'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Automatic changepoint detection: multi-method consensus

        Runs all three methods and only accepts changepoints detected
        at nearby positions by at least 2 methods.
        """
        n = len(y)

        # Detect with each method
        results = {}
        methods = ['pelt', 'cusum', 'bocpd']

        for m in methods:
            try:
                if m == 'pelt':
                    idx, conf = self._detectPELT(y, minSize, penalty)
                elif m == 'cusum':
                    idx, conf = self._detectCUSUM(y, minSize)
                else:
                    idx, conf = self._detectBOCPD(y, minSize)
                results[m] = idx
            except Exception:
                results[m] = np.array([], dtype=int)

        # Consensus-based changepoint determination
        # Collect all detected changepoints and group nearby ones
        allCandidates = []
        for m, idx in results.items():
            for cp in idx:
                allCandidates.append((cp, m))

        if len(allCandidates) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        allCandidates.sort(key=lambda x: x[0])

        # Group nearby changepoints (tolerance = minSize // 2)
        tolerance = max(minSize // 2, 3)
        groups = []
        currentGroup = [allCandidates[0]]

        for i in range(1, len(allCandidates)):
            if allCandidates[i][0] - currentGroup[-1][0] <= tolerance:
                currentGroup.append(allCandidates[i])
            else:
                groups.append(currentGroup)
                currentGroup = [allCandidates[i]]
        groups.append(currentGroup)

        # Accept only groups detected by at least 2 methods
        consensusIndices = []
        consensusConfidence = []

        for group in groups:
            uniqueMethods = set(item[1] for item in group)
            if len(uniqueMethods) >= 2:
                # Use median of group as changepoint
                positions = [item[0] for item in group]
                cpIdx = int(np.median(positions))
                consensusIndices.append(cpIdx)
                # Confidence: number of agreeing methods / total methods
                consensusConfidence.append(len(uniqueMethods) / len(methods))

        indices = np.array(consensusIndices, dtype=int)
        confidence = np.array(consensusConfidence, dtype=float)

        # Re-verify minSize condition
        if len(indices) > 0:
            indices, confidence = self._enforceMinSize(indices, confidence, n, minSize)

        return indices, confidence

    # ─── Utilities ────────────────────────────────────────────────────────

    def _clusterChangepoints(
        self,
        candidates: np.ndarray,
        stat: np.ndarray,
        minSize: int
    ) -> np.ndarray:
        """
        Cluster and refine candidate changepoints

        Groups nearby candidates and selects the point with the highest statistic in each cluster.
        """
        if len(candidates) == 0:
            return np.array([], dtype=int)

        # Group consecutive candidates into clusters
        clusters = []
        currentCluster = [candidates[0]]

        for i in range(1, len(candidates)):
            if candidates[i] - candidates[i - 1] <= minSize // 2:
                currentCluster.append(candidates[i])
            else:
                clusters.append(currentCluster)
                currentCluster = [candidates[i]]
        clusters.append(currentCluster)

        # Select point with maximum statistic from each cluster
        changepoints = []
        for cluster in clusters:
            clusterArr = np.array(cluster)
            bestIdx = clusterArr[np.argmax(stat[clusterArr])]
            changepoints.append(bestIdx)

        # Ensure minSize spacing
        result = []
        for cp in sorted(changepoints):
            if len(result) == 0 or cp - result[-1] >= minSize:
                result.append(cp)

        return np.array(result, dtype=int)

    def _extractPeaks(
        self,
        prob: np.ndarray,
        minSize: int,
        threshold: float = 0.1
    ) -> np.ndarray:
        """Extract peaks (changepoint candidates) from probability array"""
        n = len(prob)
        peaks = []

        for i in range(1, n - 1):
            if prob[i] > threshold and prob[i] > prob[i - 1] and prob[i] >= prob[i + 1]:
                peaks.append(i)

        if len(peaks) == 0:
            return np.array([], dtype=int)

        # Ensure minSize spacing (highest probability first)
        peaks = sorted(peaks, key=lambda x: prob[x], reverse=True)
        selected = []

        for p in peaks:
            if all(abs(p - s) >= minSize for s in selected):
                selected.append(p)

        return np.array(sorted(selected), dtype=int)

    def _computeConfidence(
        self,
        y: np.ndarray,
        indices: np.ndarray,
        minSize: int
    ) -> np.ndarray:
        """
        Compute changepoint confidence

        Normalizes the mean difference between adjacent segments by the global standard deviation.
        Based on Welch t-test p-value.
        """
        if len(indices) == 0:
            return np.array([], dtype=float)

        n = len(y)
        confidence = np.zeros(len(indices))
        globalStd = max(np.std(y), 1e-10)

        for i, cp in enumerate(indices):
            # Before segment
            start = indices[i - 1] if i > 0 else 0
            before = y[start:cp]

            # After segment
            end = indices[i + 1] if i < len(indices) - 1 else n
            after = y[cp:end]

            if len(before) < 2 or len(after) < 2:
                confidence[i] = 0.0
                continue

            # Welch t-test approximation
            meanDiff = abs(np.mean(after) - np.mean(before))
            pooledSe = np.sqrt(
                np.var(before) / len(before) + np.var(after) / len(after)
            )

            if pooledSe < 1e-10:
                tStat = meanDiff / globalStd * np.sqrt(min(len(before), len(after)))
            else:
                tStat = meanDiff / pooledSe

            # Convert t-stat to 0~1 confidence (sigmoid approximation)
            confidence[i] = 1.0 - 2.0 / (1.0 + np.exp(0.5 * tStat))

        return np.clip(confidence, 0.0, 1.0)

    def _computeSegmentStats(
        self,
        y: np.ndarray,
        start: int,
        end: int
    ) -> Dict:
        """Compute statistics for a single segment"""
        segment = y[start:end]
        n = len(segment)

        if n == 0:
            return {'start': start, 'end': end, 'mean': 0, 'std': 0, 'trend': 0}

        mean = float(np.mean(segment))
        std = float(np.std(segment))

        # Trend (linear regression slope)
        if n >= 2:
            x = np.arange(n, dtype=np.float64)
            xMean = np.mean(x)
            yMean = mean
            num = np.sum((x - xMean) * (segment - yMean))
            den = np.sum((x - xMean) ** 2)
            trend = float(num / den) if den > 0 else 0.0
        else:
            trend = 0.0

        return {
            'start': int(start),
            'end': int(end),
            'length': int(n),
            'mean': mean,
            'std': std,
            'trend': trend
        }

    def _computeAllSegmentStats(
        self,
        y: np.ndarray,
        indices: np.ndarray
    ) -> List[Dict]:
        """Compute statistics for all segments"""
        n = len(y)
        segments = []

        # Include start point
        boundaries = [0] + list(indices) + [n]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segments.append(self._computeSegmentStats(y, start, end))

        return segments

    def _enforceMinSize(
        self,
        indices: np.ndarray,
        confidence: np.ndarray,
        n: int,
        minSize: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ensure minimum size between changepoints"""
        if len(indices) == 0:
            return indices, confidence

        # Sort by confidence, keeping highest first
        order = np.argsort(-confidence)
        selected = []
        selectedConf = []

        for idx in order:
            cp = indices[idx]
            # Check minSize spacing (from start/end/other changepoints)
            if cp < minSize or cp > n - minSize:
                continue
            if all(abs(cp - s) >= minSize for s in selected):
                selected.append(cp)
                selectedConf.append(confidence[idx])

        # Re-sort by position
        if len(selected) > 0:
            sortOrder = np.argsort(selected)
            selected = np.array(selected, dtype=int)[sortOrder]
            selectedConf = np.array(selectedConf, dtype=float)[sortOrder]
        else:
            selected = np.array([], dtype=int)
            selectedConf = np.array([], dtype=float)

        return selected, selectedConf
