"""
Parametric Distribution Forecasting

Fit parametric distributions to forecast residuals and generate
full predictive distributions (not just point + interval).

Supported distributions: Gaussian, Student-t, Log-Normal, Negative Binomial.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class ForecastDistribution:
    """
    Parametric forecast distribution.

    Wraps a scipy distribution with forecast-specific methods.

    Parameters
    ----------
    distName : str
        Distribution name.
    params : dict
        Distribution parameters.
    pointForecast : np.ndarray
        Point forecast values.
    sigma : np.ndarray
        Per-step standard deviations.
    """

    def __init__(
        self,
        distName: str,
        params: Dict,
        pointForecast: np.ndarray,
        sigma: np.ndarray,
    ):
        self.distName = distName
        self.params = params
        self.pointForecast = pointForecast
        self.sigma = sigma
        self._dist = self._buildDist()

    def _buildDist(self):
        distMap = {
            "gaussian": stats.norm,
            "student_t": stats.t,
            "lognormal": stats.lognorm,
        }
        return distMap.get(self.distName, stats.norm)

    def quantile(self, q: float) -> np.ndarray:
        """
        Get quantile forecast.

        Parameters
        ----------
        q : float
            Quantile level (0-1).

        Returns
        -------
        np.ndarray
            Quantile values for each step.
        """
        if self.distName == "student_t":
            df = self.params.get("df", 5)
            return self.pointForecast + stats.t.ppf(q, df) * self.sigma
        elif self.distName == "lognormal":
            s = self.params.get("s", 0.5)
            return stats.lognorm.ppf(q, s, loc=self.pointForecast, scale=self.sigma)
        else:
            return self.pointForecast + stats.norm.ppf(q) * self.sigma

    def quantiles(self, levels: Optional[List[float]] = None) -> Dict[float, np.ndarray]:
        """
        Get multiple quantile forecasts.

        Parameters
        ----------
        levels : list of float, optional
            Quantile levels (default: [0.1, 0.25, 0.5, 0.75, 0.9]).

        Returns
        -------
        dict
            Mapping from quantile level to forecast array.
        """
        if levels is None:
            levels = [0.1, 0.25, 0.5, 0.75, 0.9]
        return {q: self.quantile(q) for q in levels}

    def interval(self, coverage: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get prediction interval.

        Parameters
        ----------
        coverage : float
            Coverage level (e.g. 0.95 for 95% interval).

        Returns
        -------
        lower, upper : np.ndarray
        """
        alpha = (1 - coverage) / 2
        return self.quantile(alpha), self.quantile(1 - alpha)

    def sample(self, nSamples: int = 1000) -> np.ndarray:
        """
        Generate random samples from the predictive distribution.

        Parameters
        ----------
        nSamples : int
            Number of samples per step.

        Returns
        -------
        np.ndarray
            Shape (nSamples, steps).
        """
        steps = len(self.pointForecast)
        samples = np.zeros((nSamples, steps))

        for h in range(steps):
            if self.distName == "student_t":
                df = self.params.get("df", 5)
                samples[:, h] = self.pointForecast[h] + stats.t.rvs(df, size=nSamples) * self.sigma[h]
            elif self.distName == "lognormal":
                s = self.params.get("s", 0.5)
                samples[:, h] = stats.lognorm.rvs(s, loc=self.pointForecast[h], scale=self.sigma[h], size=nSamples)
            else:
                samples[:, h] = stats.norm.rvs(loc=self.pointForecast[h], scale=self.sigma[h], size=nSamples)

        return samples

    def pdf(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        """
        Evaluate PDF at given values for a specific step.

        Parameters
        ----------
        x : np.ndarray
            Values to evaluate.
        step : int
            Forecast step index.

        Returns
        -------
        np.ndarray
            Density values.
        """
        mu = self.pointForecast[step]
        sig = self.sigma[step]

        if self.distName == "student_t":
            df = self.params.get("df", 5)
            return stats.t.pdf((x - mu) / sig, df) / sig
        elif self.distName == "lognormal":
            s = self.params.get("s", 0.5)
            return stats.lognorm.pdf(x, s, loc=mu, scale=sig)
        else:
            return stats.norm.pdf(x, loc=mu, scale=sig)

    def crps(self, actual: np.ndarray) -> np.ndarray:
        """
        Compute CRPS for each step.

        Parameters
        ----------
        actual : np.ndarray
            Actual observed values.

        Returns
        -------
        np.ndarray
            CRPS values per step.
        """
        return empiricalCRPS(actual, self.pointForecast, self.sigma, self.distName, self.params)


class DistributionFitter:
    """
    Automatic distribution fitting for forecast residuals.

    Fits multiple distributions to residuals and selects the best one
    based on AIC (Akaike Information Criterion).
    """

    CANDIDATES = ["gaussian", "student_t", "lognormal"]

    def fit(
        self,
        residuals: np.ndarray,
        pointForecast: np.ndarray,
        steps: int,
    ) -> ForecastDistribution:
        """
        Fit the best distribution to residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals (actual - predicted).
        pointForecast : np.ndarray
            Point forecasts for future steps.
        steps : int
            Number of forecast steps.

        Returns
        -------
        ForecastDistribution
            Best-fit distribution object.
        """
        clean = residuals[np.isfinite(residuals)]
        if len(clean) < 5:
            sigma = np.ones(steps) * (np.std(clean) if len(clean) > 0 else 1.0)
            horizonScale = np.sqrt(np.arange(1, steps + 1))
            return ForecastDistribution("gaussian", {}, pointForecast[:steps], sigma * horizonScale)

        bestDist = "gaussian"
        bestAIC = np.inf
        bestParams = {}

        gaussianSigma = np.std(clean, ddof=1)
        gaussianLL = np.sum(stats.norm.logpdf(clean, loc=0, scale=gaussianSigma))
        gaussianAIC = 2 * 1 - 2 * gaussianLL
        if gaussianAIC < bestAIC:
            bestAIC = gaussianAIC
            bestDist = "gaussian"
            bestParams = {"sigma": gaussianSigma}

        if len(clean) > 10:
            df, _, tScale = stats.t.fit(clean, floc=0)
            if df > 2:
                tLL = np.sum(stats.t.logpdf(clean, df, loc=0, scale=tScale))
                tAIC = 2 * 2 - 2 * tLL
                if tAIC < bestAIC:
                    bestAIC = tAIC
                    bestDist = "student_t"
                    bestParams = {"df": df, "scale": tScale}

        if np.all(clean > -np.mean(np.abs(clean)) * 5) and len(clean) > 10:
            shifted = clean - np.min(clean) + 1e-6
            if np.all(shifted > 0):
                try:
                    s, _, lnScale = stats.lognorm.fit(shifted, floc=0)
                    lnLL = np.sum(stats.lognorm.logpdf(shifted, s, loc=0, scale=lnScale))
                    lnAIC = 2 * 2 - 2 * lnLL
                    if lnAIC < bestAIC:
                        bestAIC = lnAIC
                        bestDist = "lognormal"
                        bestParams = {"s": s, "scale": lnScale}
                except Exception:
                    pass

        baseSigma = bestParams.get("scale", bestParams.get("sigma", gaussianSigma))
        horizonScale = np.sqrt(np.arange(1, steps + 1))
        sigma = baseSigma * horizonScale

        return ForecastDistribution(bestDist, bestParams, pointForecast[:steps], sigma)


def empiricalCRPS(
    actual: np.ndarray,
    predictedMean: np.ndarray,
    predictedStd: np.ndarray,
    distName: str = "gaussian",
    params: Optional[Dict] = None,
) -> np.ndarray:
    """
    Compute CRPS for forecast distribution.

    Supports Gaussian (closed-form) and empirical (Monte Carlo) for other distributions.

    Parameters
    ----------
    actual : np.ndarray
        Observed values.
    predictedMean : np.ndarray
        Point forecasts.
    predictedStd : np.ndarray
        Standard deviations per step.
    distName : str
        Distribution type.
    params : dict, optional
        Distribution parameters.

    Returns
    -------
    np.ndarray
        CRPS per step.
    """
    actual = np.asarray(actual, dtype=np.float64)
    predictedMean = np.asarray(predictedMean, dtype=np.float64)
    predictedStd = np.asarray(predictedStd, dtype=np.float64)
    n = min(len(actual), len(predictedMean))

    scores = np.zeros(n)

    for i in range(n):
        mu = predictedMean[i]
        sig = max(predictedStd[i], 1e-10)
        y = actual[i]

        if not np.isfinite(y) or not np.isfinite(mu):
            scores[i] = np.nan
            continue

        if distName == "gaussian":
            z = (y - mu) / sig
            phiZ = stats.norm.pdf(z)
            bigPhiZ = stats.norm.cdf(z)
            scores[i] = sig * (z * (2.0 * bigPhiZ - 1.0) + 2.0 * phiZ - 1.0 / np.sqrt(np.pi))
        else:
            nSamples = 200
            if distName == "student_t":
                df = params.get("df", 5) if params else 5
                samples = mu + stats.t.rvs(df, size=nSamples) * sig
            else:
                samples = stats.norm.rvs(loc=mu, scale=sig, size=nSamples)

            scores[i] = _mcCRPS(y, samples)

    return scores


def _mcCRPS(actual: float, samples: np.ndarray) -> float:
    """Monte Carlo CRPS estimation."""
    n = len(samples)
    term1 = np.mean(np.abs(samples - actual))
    sortedSamples = np.sort(samples)
    term2 = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            term2 += np.abs(sortedSamples[i] - sortedSamples[j])
    term2 /= (n * (n - 1)) if n > 1 else 1.0
    return term1 - 0.5 * term2
