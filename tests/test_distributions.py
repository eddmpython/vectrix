"""
Parametric distribution forecasting tests.
"""

import numpy as np
import pytest

from vectrix.intervals.distributions import DistributionFitter, ForecastDistribution, empiricalCRPS


class TestForecastDistribution:
    def test_gaussianQuantile(self):
        pred = np.array([10.0, 20.0, 30.0])
        sigma = np.array([1.0, 2.0, 3.0])
        dist = ForecastDistribution("gaussian", {}, pred, sigma)

        q50 = dist.quantile(0.5)
        np.testing.assert_allclose(q50, pred, atol=1e-10)

        q10 = dist.quantile(0.1)
        assert np.all(q10 < pred)
        q90 = dist.quantile(0.9)
        assert np.all(q90 > pred)

    def test_studentTQuantile(self):
        pred = np.array([10.0, 20.0])
        sigma = np.array([1.0, 2.0])
        dist = ForecastDistribution("student_t", {"df": 5}, pred, sigma)

        q50 = dist.quantile(0.5)
        np.testing.assert_allclose(q50, pred, atol=1e-10)

        q05 = dist.quantile(0.05)
        assert np.all(q05 < pred)

    def test_interval(self):
        pred = np.array([100.0, 200.0, 300.0])
        sigma = np.array([10.0, 20.0, 30.0])
        dist = ForecastDistribution("gaussian", {}, pred, sigma)

        lo, hi = dist.interval(0.95)
        assert np.all(lo < pred)
        assert np.all(hi > pred)
        assert np.all(hi - lo > 0)

    def test_quantilesDict(self):
        pred = np.array([10.0, 20.0])
        sigma = np.array([1.0, 2.0])
        dist = ForecastDistribution("gaussian", {}, pred, sigma)

        qs = dist.quantiles([0.1, 0.5, 0.9])
        assert 0.1 in qs
        assert 0.5 in qs
        assert 0.9 in qs
        assert len(qs[0.5]) == 2

    def test_sample(self):
        pred = np.array([100.0])
        sigma = np.array([10.0])
        dist = ForecastDistribution("gaussian", {}, pred, sigma)

        samples = dist.sample(nSamples=500)
        assert samples.shape == (500, 1)
        assert abs(np.mean(samples) - 100.0) < 5.0

    def test_pdf(self):
        pred = np.array([0.0])
        sigma = np.array([1.0])
        dist = ForecastDistribution("gaussian", {}, pred, sigma)

        x = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        densities = dist.pdf(x, step=0)
        assert densities[2] > densities[0]
        assert densities[2] > densities[4]

    def test_crps(self):
        pred = np.array([10.0, 20.0])
        sigma = np.array([1.0, 2.0])
        dist = ForecastDistribution("gaussian", {}, pred, sigma)

        actual = np.array([10.5, 19.0])
        scores = dist.crps(actual)
        assert len(scores) == 2
        assert np.all(scores >= 0)


class TestDistributionFitter:
    def test_fitGaussianResiduals(self):
        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 2, size=100)
        pointForecast = np.array([10.0, 20.0, 30.0])

        fitter = DistributionFitter()
        dist = fitter.fit(residuals, pointForecast, steps=3)

        assert isinstance(dist, ForecastDistribution)
        assert len(dist.pointForecast) == 3
        assert len(dist.sigma) == 3

    def test_fitStudentTResiduals(self):
        rng = np.random.default_rng(42)
        from scipy.stats import t as tdist
        residuals = tdist.rvs(df=3, size=200, random_state=rng)
        pointForecast = np.array([10.0, 20.0, 30.0])

        fitter = DistributionFitter()
        dist = fitter.fit(residuals, pointForecast, steps=3)

        assert isinstance(dist, ForecastDistribution)

    def test_fitSmallResiduals(self):
        residuals = np.array([0.1, -0.2])
        pointForecast = np.array([10.0])

        fitter = DistributionFitter()
        dist = fitter.fit(residuals, pointForecast, steps=1)
        assert dist.distName == "gaussian"


class TestEmpiricalCRPS:
    def test_perfectForecast(self):
        actual = np.array([10.0, 20.0])
        mean = np.array([10.0, 20.0])
        std = np.array([0.001, 0.001])

        scores = empiricalCRPS(actual, mean, std)
        assert np.all(scores < 0.01)

    def test_badForecast(self):
        actual = np.array([10.0])
        mean = np.array([100.0])
        std = np.array([1.0])

        scores = empiricalCRPS(actual, mean, std)
        assert scores[0] > 50

    def test_widerDistributionHigherCRPS(self):
        actual = np.array([10.0])
        mean = np.array([10.0])

        narrow = empiricalCRPS(actual, mean, np.array([0.5]))
        wide = empiricalCRPS(actual, mean, np.array([5.0]))
        assert wide[0] > narrow[0]
