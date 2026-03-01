"""
Probabilistic Forecasting

Quantile forecasting, bootstrap prediction intervals, and
probabilistic scoring rules (CRPS, Pinball Loss, Winkler Score).
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.stats import norm


class ProbabilisticForecaster:
    """Probabilistic forecasting (quantiles, CRPS)"""

    QUANTILES = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

    def quantileForecast(
        self,
        predictions: np.ndarray,
        residuals: np.ndarray,
        steps: int,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """Quantile forecast based on residual distribution"""
        if quantiles is None:
            quantiles = self.QUANTILES

        predictions = np.asarray(predictions, dtype=np.float64)
        residuals = np.asarray(residuals, dtype=np.float64)

        cleanResiduals = residuals[np.isfinite(residuals)]
        if len(cleanResiduals) == 0:
            return {q: predictions[:steps].copy() for q in quantiles}

        sigma = np.std(cleanResiduals, ddof=1) if len(cleanResiduals) > 1 else 0.0
        if sigma == 0.0:
            return {q: predictions[:steps].copy() for q in quantiles}

        horizonScale = np.sqrt(np.arange(1, steps + 1, dtype=np.float64))
        pred = predictions[:steps]

        result: Dict[float, np.ndarray] = {}
        for q in quantiles:
            zScore = norm.ppf(q)
            result[q] = pred + zScore * sigma * horizonScale

        return result

    def bootstrapQuantiles(
        self,
        y: np.ndarray,
        modelFactory: Callable,
        steps: int,
        nBoot: int = 200,
        quantiles: Optional[List[float]] = None,
    ) -> Dict[float, np.ndarray]:
        """Bootstrap-based quantile forecast"""
        if quantiles is None:
            quantiles = self.QUANTILES

        y = np.asarray(y, dtype=np.float64)
        if len(y) < 3:
            fallback = np.full(steps, np.nanmean(y) if len(y) > 0 else 0.0)
            return {q: fallback.copy() for q in quantiles}

        baseModel = modelFactory()
        baseModel.fit(y)
        basePred, _, _ = baseModel.predict(steps)
        basePred = np.asarray(basePred, dtype=np.float64)

        fittedValues = np.empty(len(y), dtype=np.float64)
        for t in range(len(y)):
            if t < 2:
                fittedValues[t] = y[t]
            else:
                tmpModel = modelFactory()
                try:
                    tmpModel.fit(y[:t])
                    onePred, _, _ = tmpModel.predict(1)
                    fittedValues[t] = onePred[0]
                except Exception:
                    fittedValues[t] = y[t]

        residuals = y - fittedValues
        cleanResiduals = residuals[np.isfinite(residuals)]
        if len(cleanResiduals) == 0:
            return {q: basePred.copy() for q in quantiles}

        paths = np.empty((nBoot, steps), dtype=np.float64)
        rng = np.random.default_rng()

        for b in range(nBoot):
            sampledResiduals = rng.choice(cleanResiduals, size=steps, replace=True)
            paths[b, :] = basePred + sampledResiduals

        result: Dict[float, np.ndarray] = {}
        for q in quantiles:
            result[q] = np.quantile(paths, q, axis=0)

        return result

    @staticmethod
    def crps(
        actual: float,
        predictedMean: float,
        predictedStd: float,
    ) -> float:
        """CRPS (Continuous Ranked Probability Score) -- Gaussian closed-form"""
        if not np.isfinite(actual) or not np.isfinite(predictedMean):
            return np.nan

        predictedStd = float(predictedStd)
        if predictedStd <= 0.0:
            return float(np.abs(actual - predictedMean))

        z = (actual - predictedMean) / predictedStd
        phiZ = norm.pdf(z)
        bigPhiZ = norm.cdf(z)

        return predictedStd * (z * (2.0 * bigPhiZ - 1.0) + 2.0 * phiZ - 1.0 / np.sqrt(np.pi))

    @staticmethod
    def crpsArray(
        actuals: np.ndarray,
        predictedMeans: np.ndarray,
        predictedStds: np.ndarray,
    ) -> np.ndarray:
        """Array CRPS"""
        actuals = np.asarray(actuals, dtype=np.float64)
        predictedMeans = np.asarray(predictedMeans, dtype=np.float64)
        predictedStds = np.asarray(predictedStds, dtype=np.float64)

        n = len(actuals)
        if n == 0:
            return np.array([], dtype=np.float64)

        scores = np.empty(n, dtype=np.float64)
        for i in range(n):
            scores[i] = ProbabilisticForecaster.crps(actuals[i], predictedMeans[i], predictedStds[i])

        return scores

    @staticmethod
    def quantileLoss(
        actual: float,
        predicted: float,
        quantile: float,
    ) -> float:
        """Pinball Loss"""
        if not np.isfinite(actual) or not np.isfinite(predicted):
            return np.nan

        diff = actual - predicted
        return float(max(quantile * diff, (quantile - 1.0) * diff))

    @staticmethod
    def winklerScore(
        actual: float,
        lower: float,
        upper: float,
        alpha: float = 0.05,
    ) -> float:
        """Winkler Score"""
        if not np.isfinite(actual) or not np.isfinite(lower) or not np.isfinite(upper):
            return np.nan

        intervalWidth = upper - lower

        if actual < lower:
            return intervalWidth + (2.0 / alpha) * (lower - actual)
        elif actual > upper:
            return intervalWidth + (2.0 / alpha) * (actual - upper)
        else:
            return intervalWidth
