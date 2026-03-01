"""
Entropic Confidence Scoring

Information entropy-based forecast confidence assessment.
Analyzes entropy of bootstrap forecast paths to measure the "quality" of uncertainty.
Examines distributional structure (multimodality, bias, concentration) rather than simple interval width.
"""

from typing import Dict, List, Optional

import numpy as np
from scipy.cluster.vq import kmeans2

DEFAULT_N_BOOTSTRAP = 200
DEFAULT_N_BINS = 30
DEFAULT_N_SCENARIOS = 3

GRADE_THRESHOLDS = {
    'A': 0.8,
    'B': 0.6,
    'C': 0.4,
    'D': 0.2,
}

MODE_MIN_HEIGHT_RATIO = 0.05
MULTIMODAL_PENALTY = 0.1
SMOOTHING_KERNEL_SIZE = 3


class EntropyResult:
    """Data container for entropy analysis results."""

    def __init__(self):
        self.stepEntropy: Optional[np.ndarray] = None
        self.normalizedEntropy: Optional[np.ndarray] = None
        self.confidenceScore: Optional[np.ndarray] = None
        self.nModes: Optional[np.ndarray] = None
        self.pathDiversity: float = 0.0
        self.scenarios: List[Dict] = []
        self.bootstrapPaths: Optional[np.ndarray] = None
        self.overallConfidence: float = 0.0

    def summary(self) -> Dict:
        """Summary of analysis results."""
        result = {
            'overallConfidence': float(self.overallConfidence),
            'grade': self.grade(),
            'pathDiversity': float(self.pathDiversity),
        }

        if self.confidenceScore is not None:
            result['meanStepConfidence'] = float(np.mean(self.confidenceScore))
            result['minStepConfidence'] = float(np.min(self.confidenceScore))
            result['maxStepConfidence'] = float(np.max(self.confidenceScore))

        if self.nModes is not None:
            result['maxModes'] = int(np.max(self.nModes))
            result['hasMultimodal'] = bool(np.any(self.nModes > 1))

        if self.normalizedEntropy is not None:
            result['meanNormalizedEntropy'] = float(np.mean(self.normalizedEntropy))

        result['nScenarios'] = len(self.scenarios)

        return result

    def grade(self) -> str:
        """A/B/C/D/F confidence grade."""
        score = self.overallConfidence
        for g, threshold in GRADE_THRESHOLDS.items():
            if score > threshold:
                return g
        return 'F'


class EntropicConfidenceScorer:
    """
    Information entropy-based forecast confidence assessment.
    Analyzes entropy of bootstrap forecast paths to measure
    the "quality" of uncertainty.
    """

    def __init__(
        self,
        model=None,
        nBootstrap: int = DEFAULT_N_BOOTSTRAP,
        nBins: int = DEFAULT_N_BINS,
    ):
        self.model = model
        self.nBootstrap = nBootstrap
        self.nBins = nBins
        self._result: Optional[EntropyResult] = None
        self._y: Optional[np.ndarray] = None

    def analyze(self, y: np.ndarray, steps: int = 12) -> EntropyResult:
        """Generate bootstrap forecast paths and analyze entropy-based confidence."""
        y = np.asarray(y, dtype=np.float64)
        self._y = y.copy()

        result = EntropyResult()

        bootstrapPaths = self._generateBootstrapPaths(y, steps)
        result.bootstrapPaths = bootstrapPaths

        stepEntropy = np.zeros(steps)
        normalizedEntropy = np.zeros(steps)
        confidenceScore = np.zeros(steps)
        nModes = np.zeros(steps, dtype=np.int32)

        maxEntropy = np.log(self.nBins) if self.nBins > 1 else 1.0

        for h in range(steps):
            values = bootstrapPaths[:, h]
            cleanValues = values[np.isfinite(values)]

            if len(cleanValues) < 2:
                stepEntropy[h] = 0.0
                normalizedEntropy[h] = 0.0
                confidenceScore[h] = 1.0
                nModes[h] = 1
                continue

            entropy = self._computeShannonEntropy(cleanValues)
            stepEntropy[h] = entropy

            normEnt = entropy / maxEntropy if maxEntropy > 0 else 0.0
            normEnt = np.clip(normEnt, 0.0, 1.0)
            normalizedEntropy[h] = normEnt

            modesAtStep = self._detectModes(cleanValues)
            nModes[h] = modesAtStep

            conf = 1.0 - normEnt
            if modesAtStep > 1:
                conf -= MULTIMODAL_PENALTY * (modesAtStep - 1)
            confidenceScore[h] = np.clip(conf, 0.0, 1.0)

        result.stepEntropy = stepEntropy
        result.normalizedEntropy = normalizedEntropy
        result.confidenceScore = confidenceScore
        result.nModes = nModes

        result.pathDiversity = self._computePathDiversity(bootstrapPaths)

        if np.any(confidenceScore > 0):
            weights = np.exp(-np.arange(steps) * 0.05)
            result.overallConfidence = float(
                np.average(confidenceScore, weights=weights)
            )
        else:
            result.overallConfidence = 0.0

        self._result = result
        return result

    def extractScenarios(self, nScenarios: int = DEFAULT_N_SCENARIOS) -> List[Dict]:
        """Extract representative scenarios by K-means clustering of bootstrap paths."""
        if self._result is None or self._result.bootstrapPaths is None:
            return []

        paths = self._result.bootstrapPaths
        nPaths = paths.shape[0]

        if nPaths < nScenarios:
            nScenarios = max(1, nPaths)

        try:
            centroids, labels = kmeans2(
                paths.astype(np.float64), nScenarios, minit='points'
            )
        except Exception:
            return []

        scenarios = []
        for k in range(nScenarios):
            mask = labels == k
            prob = float(mask.sum()) / float(len(labels))

            if prob <= 0.05:
                continue

            centroid = centroids[k]
            description = self._describeScenario(centroid, k)

            scenarios.append({
                'probability': prob,
                'path': centroid,
                'description': description,
            })

        scenarios = sorted(scenarios, key=lambda s: -s['probability'])

        self._result.scenarios = scenarios
        return scenarios

    def _generateBootstrapPaths(self, y: np.ndarray, steps: int) -> np.ndarray:
        n = len(y)
        model = self._resolveModel()

        model.fit(y)
        basePred, _, _ = model.predict(steps)
        basePred = np.asarray(basePred, dtype=np.float64)

        fittedValues = self._computeFittedValues(y, model)
        residuals = y - fittedValues
        cleanResiduals = residuals[np.isfinite(residuals)]

        if len(cleanResiduals) == 0:
            return np.tile(basePred, (self.nBootstrap, 1))

        blockSize = self._inferBlockSize(y)
        paths = np.zeros((self.nBootstrap, steps))
        rng = np.random.default_rng()

        for b in range(self.nBootstrap):
            bootResiduals = self._blockBootstrap(cleanResiduals, blockSize, rng)
            bootY = fittedValues + bootResiduals[:n]

            try:
                bootModel = self._resolveModel()
                bootModel.fit(bootY)
                pred, _, _ = bootModel.predict(steps)
                paths[b, :] = np.asarray(pred, dtype=np.float64)
            except Exception:
                paths[b, :] = basePred

        return paths

    def _resolveModel(self):
        if self.model is not None:
            if callable(self.model):
                return self.model()
            try:
                modelClass = self.model.__class__
                return modelClass()
            except Exception:
                pass

        from vectrix.engine.ets import ETSModel
        return ETSModel()

    def _computeFittedValues(self, y: np.ndarray, model) -> np.ndarray:
        n = len(y)
        if hasattr(model, 'residuals') and model.residuals is not None:
            residuals = np.asarray(model.residuals, dtype=np.float64)
            if len(residuals) == n:
                return y - residuals

        fittedValues = np.empty(n, dtype=np.float64)
        fittedValues[0] = y[0]

        for t in range(1, n):
            if t < 3:
                fittedValues[t] = np.mean(y[:t])
            else:
                try:
                    tmpModel = self._resolveModel()
                    tmpModel.fit(y[:t])
                    onePred, _, _ = tmpModel.predict(1)
                    fittedValues[t] = onePred[0]
                except Exception:
                    fittedValues[t] = np.mean(y[max(0, t - 5):t])

        return fittedValues

    def _inferBlockSize(self, y: np.ndarray) -> int:
        n = len(y)
        if hasattr(self.model, 'period') and self.model is not None:
            try:
                period = getattr(self.model, 'period', 1)
                if isinstance(period, int) and period > 0:
                    return max(1, period)
            except Exception:
                pass
        return max(1, int(np.sqrt(n)))

    def _blockBootstrap(
        self,
        residuals: np.ndarray,
        blockSize: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = len(residuals)
        blockSize = min(blockSize, n)
        blockSize = max(blockSize, 1)

        nBlocks = (n + blockSize - 1) // blockSize
        indices = []

        for _ in range(nBlocks):
            start = rng.integers(0, max(1, n - blockSize + 1))
            indices.extend(range(start, min(start + blockSize, n)))

        return residuals[np.array(indices[:n])]

    def _computeShannonEntropy(self, values: np.ndarray) -> float:
        hist, _ = np.histogram(values, bins=self.nBins, density=False)
        counts = hist.astype(np.float64)
        total = counts.sum()

        if total == 0:
            return 0.0

        probs = counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))

        return float(entropy)

    def _detectModes(self, values: np.ndarray) -> int:
        hist, _ = np.histogram(values, bins=self.nBins, density=True)
        kernel = np.ones(SMOOTHING_KERNEL_SIZE) / SMOOTHING_KERNEL_SIZE
        smoothed = np.convolve(hist, kernel, mode='same')

        maxVal = np.max(smoothed) if len(smoothed) > 0 else 0.0
        if maxVal <= 0:
            return 1

        minHeight = MODE_MIN_HEIGHT_RATIO * maxVal
        peaks = 0

        for i in range(1, len(smoothed) - 1):
            if (
                smoothed[i] > smoothed[i - 1]
                and smoothed[i] > smoothed[i + 1]
                and smoothed[i] > minHeight
            ):
                peaks += 1

        return max(1, peaks)

    def _computePathDiversity(self, paths: np.ndarray) -> float:
        nPaths = paths.shape[0]

        if nPaths < 2:
            return 0.0

        maxSample = min(nPaths, 100)
        rng = np.random.default_rng(42)
        indices = rng.choice(nPaths, size=maxSample, replace=False)
        sampledPaths = paths[indices]

        distances = []
        for i in range(maxSample):
            for j in range(i + 1, maxSample):
                dist = np.sqrt(np.mean((sampledPaths[i] - sampledPaths[j]) ** 2))
                distances.append(dist)

        if len(distances) == 0:
            return 0.0

        meanDist = float(np.mean(distances))
        scale = float(np.std(paths)) if np.std(paths) > 0 else 1.0
        diversity = meanDist / (scale + 1e-10)

        return float(diversity)

    def _describeScenario(self, centroid: np.ndarray, idx: int) -> str:
        if len(centroid) == 0:
            return f"Scenario {idx}"

        trend = centroid[-1] - centroid[0]
        meanLevel = float(np.mean(centroid))
        volatility = float(np.std(centroid))

        if abs(trend) < volatility * 0.1:
            trendDesc = "stable"
        elif trend > 0:
            trendDesc = "upward"
        else:
            trendDesc = "downward"

        if volatility < meanLevel * 0.05:
            volDesc = "low volatility"
        elif volatility < meanLevel * 0.2:
            volDesc = "moderate volatility"
        else:
            volDesc = "high volatility"

        return f"Scenario {idx}: {trendDesc}, {volDesc}"
