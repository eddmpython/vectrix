"""
Adversarial Stress Testing for Time Series Forecasting Models

Systematically discovers forecast model vulnerabilities by applying
five perturbation operators to the input series and measuring
how predictions degrade:

1. Level Shift   - sudden mean displacement at random time point
2. Volatility Burst - variance amplification over random segment
3. Trend Break   - slope change at random time point
4. Seasonal Corruption - phase shift or amplitude modulation
5. Tail Injection - extreme outlier insertion (4-6 sigma)

Fragility score aggregates sensitivity across all operators.
Resilience = 1 - fragility.
"""

from typing import Any, Dict, List

import numpy as np

OPERATOR_WEIGHTS = {
    "levelShift": 0.25,
    "volatilityBurst": 0.20,
    "trendBreak": 0.25,
    "seasonalCorruption": 0.15,
    "tailInjection": 0.15,
}

FRAGILITY_THRESHOLDS = [
    (0.15, "Robust"),
    (0.40, "Moderate"),
    (0.70, "Fragile"),
    (1.01, "Critical"),
]


def _safeMAPE(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.abs(actual) > 1e-10
    if np.sum(mask) == 0:
        diffs = np.abs(actual - predicted)
        return float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _isDivergent(predictions: np.ndarray, baselinePredictions: np.ndarray, threshold: float = 10.0) -> bool:
    if len(predictions) == 0 or len(baselinePredictions) == 0:
        return False
    baseRange = np.ptp(baselinePredictions)
    if baseRange < 1e-10:
        baseRange = np.std(baselinePredictions) + 1e-10
    maxDeviation = np.max(np.abs(predictions - baselinePredictions))
    return float(maxDeviation / baseRange) > threshold


class StressTestResult:
    """Container for adversarial stress test results."""

    def __init__(self):
        self.baselineError: float = 0.0
        self.perturbationResults: Dict[str, Dict[str, Any]] = {}
        self.fragilityScore: float = 0.0
        self.resilience: float = 0.0
        self.worstCase: Dict[str, Any] = {}
        self.recommendations: List[str] = []

    def summary(self) -> Dict[str, Any]:
        """Return concise summary of stress test."""
        return {
            "baselineError": self.baselineError,
            "fragilityScore": self.fragilityScore,
            "resilience": self.resilience,
            "grade": self._grade(),
            "worstOperator": self.worstCase.get("operator", "N/A"),
            "worstSensitivity": self.worstCase.get("sensitivity", 0.0),
            "recommendations": self.recommendations,
        }

    def vulnerabilityProfile(self) -> Dict[str, Any]:
        """Return per-operator vulnerability profile."""
        profile = {}
        for opName, result in self.perturbationResults.items():
            profile[opName] = {
                "medianMAPE": result.get("medianMAPE", 0.0),
                "sensitivity": result.get("sensitivity", 0.0),
                "divergenceRate": result.get("divergenceRate", 0.0),
                "maxMAPE": result.get("maxMAPE", 0.0),
                "weight": OPERATOR_WEIGHTS.get(opName, 0.0),
            }
        return profile

    def _grade(self) -> str:
        for threshold, label in FRAGILITY_THRESHOLDS:
            if self.fragilityScore < threshold:
                return label
        return "Critical"


class AdversarialStressTester:
    """
    Adversarial stress tester for forecasting models.
    Applies five perturbation operators to discover prediction vulnerabilities.
    """

    def __init__(
        self,
        model=None,
        nPerturbations: int = 50
    ):
        self._modelSpec = model
        self.nPerturbations = max(5, nPerturbations)
        self._rng = np.random.RandomState(42)

    def analyze(self, y: np.ndarray, steps: int = 12) -> StressTestResult:
        """Run full adversarial analysis with all perturbation operators."""
        result = StressTestResult()
        n = len(y)
        sigma = np.std(y) if n > 1 else 1.0

        baseModel = self._createModel()
        baseModel.fit(y)
        basePred, _, _ = baseModel.predict(steps)

        if n > steps:
            holdout = y[-steps:]
            trainY = y[:-steps]
            trainModel = self._createModel()
            trainModel.fit(trainY)
            trainPred, _, _ = trainModel.predict(steps)
            result.baselineError = _safeMAPE(holdout, trainPred)
        else:
            result.baselineError = _safeMAPE(y, basePred[:n])

        if result.baselineError < 1e-10:
            result.baselineError = 1.0

        operators = [
            ("levelShift", self._runLevelShiftTrials),
            ("volatilityBurst", self._runVolatilityBurstTrials),
            ("trendBreak", self._runTrendBreakTrials),
            ("seasonalCorruption", self._runSeasonalCorruptionTrials),
            ("tailInjection", self._runTailInjectionTrials),
        ]

        worstSensitivity = -1.0
        worstOp = ""

        for opName, opFunc in operators:
            opResult = opFunc(y, basePred, steps, sigma)
            result.perturbationResults[opName] = opResult
            if opResult["sensitivity"] > worstSensitivity:
                worstSensitivity = opResult["sensitivity"]
                worstOp = opName

        result.worstCase = {
            "operator": worstOp,
            "sensitivity": worstSensitivity,
        }

        weightedSum = 0.0
        for opName, opResult in result.perturbationResults.items():
            w = OPERATOR_WEIGHTS.get(opName, 0.0)
            weightedSum += w * opResult["sensitivity"]

        result.fragilityScore = float(np.clip(weightedSum, 0.0, 1.0))
        result.resilience = 1.0 - result.fragilityScore

        result.recommendations = self._generateRecommendations(result)

        return result

    def _createModel(self):
        if self._modelSpec is not None:
            if callable(self._modelSpec):
                return self._modelSpec()
            return self._modelSpec
        from .ets import ETSModel
        return ETSModel()

    def _runOperatorTrials(
        self,
        y: np.ndarray,
        basePred: np.ndarray,
        steps: int,
        perturbFunc,
    ) -> Dict[str, Any]:
        mapeList = []
        divergenceCount = 0
        n = len(y)

        for _ in range(self.nPerturbations):
            perturbedY = perturbFunc(y.copy())

            try:
                model = self._createModel()
                model.fit(perturbedY)
                pred, _, _ = model.predict(steps)

                if n > steps:
                    holdout = y[-steps:]
                    mape = _safeMAPE(holdout, pred[:steps])
                else:
                    mape = _safeMAPE(y, pred[:n])

                mapeList.append(mape)

                if _isDivergent(pred, basePred):
                    divergenceCount += 1
            except Exception:
                divergenceCount += 1
                mapeList.append(np.inf)

        validMapes = [m for m in mapeList if np.isfinite(m)]

        if len(validMapes) == 0:
            return {
                "medianMAPE": np.inf,
                "maxMAPE": np.inf,
                "sensitivity": 1.0,
                "divergenceRate": 1.0,
                "nTrials": self.nPerturbations,
            }

        medianMAPE = float(np.median(validMapes))
        maxMAPE = float(np.max(validMapes))
        baseMAPE = max(_safeMAPE(y[-steps:], basePred[:steps]) if len(y) > steps else 1.0, 1e-10)

        rawSensitivity = (medianMAPE / baseMAPE) - 1.0
        sensitivity = float(np.clip(rawSensitivity, 0.0, 1.0))

        return {
            "medianMAPE": medianMAPE,
            "maxMAPE": maxMAPE,
            "sensitivity": sensitivity,
            "divergenceRate": float(divergenceCount / self.nPerturbations),
            "nTrials": self.nPerturbations,
        }

    def _runLevelShiftTrials(
        self,
        y: np.ndarray,
        basePred: np.ndarray,
        steps: int,
        sigma: float
    ) -> Dict[str, Any]:
        def perturb(yy):
            return self._levelShift(yy, sigma * self._rng.uniform(1.0, 3.0))
        return self._runOperatorTrials(y, basePred, steps, perturb)

    def _runVolatilityBurstTrials(
        self,
        y: np.ndarray,
        basePred: np.ndarray,
        steps: int,
        sigma: float
    ) -> Dict[str, Any]:
        n = len(y)
        def perturb(yy):
            factor = self._rng.uniform(2.0, 5.0)
            duration = max(3, int(n * self._rng.uniform(0.05, 0.15)))
            return self._volatilityBurst(yy, factor, duration)
        return self._runOperatorTrials(y, basePred, steps, perturb)

    def _runTrendBreakTrials(
        self,
        y: np.ndarray,
        basePred: np.ndarray,
        steps: int,
        sigma: float
    ) -> Dict[str, Any]:
        def perturb(yy):
            slopeChange = sigma * self._rng.uniform(0.01, 0.1) * self._rng.choice([-1, 1])
            return self._trendBreak(yy, slopeChange)
        return self._runOperatorTrials(y, basePred, steps, perturb)

    def _runSeasonalCorruptionTrials(
        self,
        y: np.ndarray,
        basePred: np.ndarray,
        steps: int,
        sigma: float
    ) -> Dict[str, Any]:
        n = len(y)
        period = self._detectPeriod(y)
        def perturb(yy):
            return self._seasonalCorruption(yy, period)
        return self._runOperatorTrials(y, basePred, steps, perturb)

    def _runTailInjectionTrials(
        self,
        y: np.ndarray,
        basePred: np.ndarray,
        steps: int,
        sigma: float
    ) -> Dict[str, Any]:
        n = len(y)
        def perturb(yy):
            nOutliers = max(1, int(n * self._rng.uniform(0.01, 0.05)))
            sigmaMultiple = self._rng.uniform(4.0, 6.0)
            return self._tailInjection(yy, nOutliers, sigmaMultiple)
        return self._runOperatorTrials(y, basePred, steps, perturb)

    def _levelShift(self, y: np.ndarray, magnitude: float) -> np.ndarray:
        n = len(y)
        t = self._rng.randint(max(1, n // 4), max(2, 3 * n // 4))
        direction = self._rng.choice([-1, 1])
        y[t:] += direction * magnitude
        return y

    def _volatilityBurst(self, y: np.ndarray, factor: float, duration: int) -> np.ndarray:
        n = len(y)
        maxStart = max(1, n - duration - 1)
        start = self._rng.randint(0, maxStart)
        end = min(start + duration, n)

        segment = y[start:end]
        segmentMean = np.mean(segment)
        deviations = segment - segmentMean
        y[start:end] = segmentMean + deviations * factor
        return y

    def _trendBreak(self, y: np.ndarray, slopeChange: float) -> np.ndarray:
        n = len(y)
        t = self._rng.randint(max(1, n // 4), max(2, 3 * n // 4))
        ramp = np.arange(n - t, dtype=np.float64) * slopeChange
        y[t:] += ramp
        return y

    def _seasonalCorruption(self, y: np.ndarray, period: int) -> np.ndarray:
        n = len(y)
        if period < 2 or n < period * 2:
            amplitude = np.std(y) * self._rng.uniform(0.2, 0.5)
            corruptionIdx = self._rng.randint(0, n, size=max(1, n // 10))
            y[corruptionIdx] += self._rng.choice([-1, 1], size=len(corruptionIdx)) * amplitude
            return y

        mode = self._rng.choice(["phaseShift", "amplitudeModulation"])

        if mode == "phaseShift":
            shift = self._rng.randint(1, max(2, period // 2))
            breakPoint = self._rng.randint(max(1, n // 3), max(2, 2 * n // 3))
            segment = y[breakPoint:].copy()
            shiftedSegment = np.roll(segment, -shift)
            segMean = np.mean(segment)
            shiftedMean = np.mean(shiftedSegment)
            y[breakPoint:] = shiftedSegment + (segMean - shiftedMean)
        else:
            seasonalPattern = np.zeros(period)
            for i in range(period):
                vals = y[i::period]
                seasonalPattern[i] = np.mean(vals) - np.mean(y)

            modulationFactor = self._rng.uniform(0.3, 2.0)
            delta = seasonalPattern * (modulationFactor - 1.0)
            tiledDelta = np.tile(delta, n // period + 1)[:n]
            y += tiledDelta

        return y

    def _tailInjection(self, y: np.ndarray, nOutliers: int, sigmaMultiple: float) -> np.ndarray:
        n = len(y)
        sigma = np.std(y) if n > 1 else 1.0
        outlierPositions = self._rng.choice(n, size=min(nOutliers, n), replace=False)
        for pos in outlierPositions:
            direction = self._rng.choice([-1, 1])
            y[pos] += direction * sigmaMultiple * sigma
        return y

    def _detectPeriod(self, y: np.ndarray) -> int:
        n = len(y)
        if n < 10:
            return 1

        maxLag = min(n // 2, 52)
        if np.var(y) < 1e-10:
            return 1

        from .turbo import TurboCore
        acf = TurboCore.acf(y, maxLag)

        if len(acf) < 3:
            return 1

        bestLag = 1
        bestVal = -1.0
        for lag in range(2, len(acf)):
            if acf[lag] > bestVal and acf[lag] > 0.1:
                if lag > 1 and (lag == 2 or acf[lag] > acf[lag - 1]):
                    bestVal = acf[lag]
                    bestLag = lag

        return bestLag if bestVal > 0.1 else 1

    def _generateRecommendations(self, result: StressTestResult) -> List[str]:
        recs = []
        grade = result.summary()["grade"]

        if grade == "Robust":
            recs.append("Model shows strong robustness across all perturbation types.")
            return recs

        if grade == "Critical":
            recs.append("Model is critically fragile. Consider switching to a more robust method or using an ensemble.")

        for opName, opResult in result.perturbationResults.items():
            sensitivity = opResult["sensitivity"]
            divergenceRate = opResult["divergenceRate"]

            if sensitivity > 0.5 or divergenceRate > 0.2:
                if opName == "levelShift":
                    recs.append("High sensitivity to level shifts. Consider adaptive models or change-point detection preprocessing.")
                elif opName == "volatilityBurst":
                    recs.append("Vulnerable to volatility bursts. Consider GARCH-family models or robust estimation.")
                elif opName == "trendBreak":
                    recs.append("Sensitive to trend breaks. Consider piecewise-linear trend models or trend-dampening.")
                elif opName == "seasonalCorruption":
                    recs.append("Seasonal pattern corruption degrades forecasts. Consider robust seasonal decomposition (MSTL).")
                elif opName == "tailInjection":
                    recs.append("Extreme outliers significantly impact predictions. Consider outlier-robust preprocessing or trimmed estimation.")

        if result.fragilityScore > 0.4:
            recs.append("Consider an ensemble approach (e.g., LotkaVolterraEnsemble) to reduce single-model fragility.")

        return recs
