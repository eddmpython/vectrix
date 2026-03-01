import numpy as np
import pytest

from vectrix.engine.dtsf import DynamicTimeScanForecaster
from vectrix.engine.esn import EchoStateForecaster
from vectrix.engine.fourTheta import AdaptiveThetaEnsemble


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def seasonalData(rng):
    n = 200
    t = np.arange(n, dtype=np.float64)
    return 100.0 + 0.3 * t + 15.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, n)


@pytest.fixture
def trendData(rng):
    return 100.0 + 0.5 * np.arange(200, dtype=np.float64) + rng.normal(0, 2, 200)


@pytest.fixture
def shortData(rng):
    return 50.0 + rng.normal(0, 5, 20)


@pytest.fixture
def repeatingPatternData():
    pattern = np.array([10.0, 20.0, 30.0, 25.0, 15.0, 5.0, 8.0])
    return np.tile(pattern, 40)


@pytest.fixture
def nonlinearData(rng):
    n = 300
    y = np.zeros(n)
    y[0] = 0.1
    for t in range(1, n):
        y[t] = 0.9 * np.sin(y[t - 1] * np.pi) + 0.1 * rng.normal()
    return y


@pytest.fixture
def m4StyleData(rng):
    n = 120
    t = np.arange(n, dtype=np.float64)
    return 50.0 + 0.2 * t + 10.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)


class TestDynamicTimeScanForecaster:

    def test_fitPredictSeasonal(self, seasonalData):
        model = DynamicTimeScanForecaster()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert len(pred) == 12
        assert np.all(np.isfinite(pred))

    def test_fitPredictTrending(self, trendData):
        model = DynamicTimeScanForecaster()
        model.fit(trendData)
        pred, lower, upper = model.predict(10)
        assert len(pred) == 10
        assert np.all(np.isfinite(pred))

    def test_fitPredictShort(self, shortData):
        model = DynamicTimeScanForecaster()
        model.fit(shortData)
        pred, lower, upper = model.predict(5)
        assert len(pred) == 5
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    @pytest.mark.parametrize("steps", [1, 5, 12, 24])
    def test_predictLengthMatchesSteps(self, seasonalData, steps):
        model = DynamicTimeScanForecaster()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(steps)
        assert len(pred) == steps
        assert len(lower) == steps
        assert len(upper) == steps

    def test_predictionsFinite(self, seasonalData):
        model = DynamicTimeScanForecaster()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    def test_predictionsReasonableRange(self, seasonalData):
        model = DynamicTimeScanForecaster()
        model.fit(seasonalData)
        pred, _, _ = model.predict(12)
        yMin = np.min(seasonalData)
        yMax = np.max(seasonalData)
        yRange = yMax - yMin
        assert np.all(pred > yMin - 3 * yRange)
        assert np.all(pred < yMax + 3 * yRange)

    def test_unfittedModelRaisesError(self):
        model = DynamicTimeScanForecaster()
        with pytest.raises(ValueError, match="fit"):
            model.predict(5)

    def test_repeatingPatternCapture(self, repeatingPatternData):
        model = DynamicTimeScanForecaster(windowSize=7, nNeighbors=10)
        model.fit(repeatingPatternData)
        pred, _, _ = model.predict(7)
        assert len(pred) == 7
        assert np.all(np.isfinite(pred))
        expectedPattern = repeatingPatternData[:7]
        mae = np.mean(np.abs(pred - expectedPattern))
        assert mae < 15.0

    def test_confidenceIntervalOrdering(self, seasonalData):
        model = DynamicTimeScanForecaster()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert np.all(lower <= upper + 1e-6)

    def test_residualsComputed(self, seasonalData):
        model = DynamicTimeScanForecaster(computeResiduals=True)
        model.fit(seasonalData)
        assert model.residuals is not None
        assert len(model.residuals) == len(seasonalData)
        assert np.all(np.isfinite(model.residuals))

    def test_customWindowSize(self, seasonalData):
        model = DynamicTimeScanForecaster(windowSize=14, nNeighbors=3)
        model.fit(seasonalData)
        pred, _, _ = model.predict(7)
        assert len(pred) == 7
        assert np.all(np.isfinite(pred))


class TestEchoStateForecaster:

    def test_fitPredictSeasonal(self, seasonalData):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert len(pred) == 12
        assert np.all(np.isfinite(pred))

    def test_fitPredictTrending(self, trendData):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(trendData)
        pred, lower, upper = model.predict(10)
        assert len(pred) == 10
        assert np.all(np.isfinite(pred))

    def test_fitPredictShort(self, shortData):
        model = EchoStateForecaster(reservoirSize=30, seed=42)
        model.fit(shortData)
        pred, lower, upper = model.predict(5)
        assert len(pred) == 5
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    @pytest.mark.parametrize("steps", [1, 5, 12, 24])
    def test_predictLengthMatchesSteps(self, seasonalData, steps):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(seasonalData)
        pred, lower, upper = model.predict(steps)
        assert len(pred) == steps
        assert len(lower) == steps
        assert len(upper) == steps

    def test_predictionsFinite(self, seasonalData):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    def test_predictionsReasonableRange(self, seasonalData):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(seasonalData)
        pred, _, _ = model.predict(12)
        yMean = np.mean(seasonalData)
        yStd = np.std(seasonalData)
        assert np.all(pred > yMean - 10 * yStd)
        assert np.all(pred < yMean + 10 * yStd)

    def test_unfittedModelRaisesError(self):
        model = EchoStateForecaster()
        with pytest.raises(ValueError, match="fit"):
            model.predict(5)

    def test_nonlinearDataCapture(self, nonlinearData):
        model = EchoStateForecaster(reservoirSize=100, spectralRadius=0.95, seed=42)
        model.fit(nonlinearData)
        pred, _, _ = model.predict(10)
        assert len(pred) == 10
        assert np.all(np.isfinite(pred))
        yRange = np.max(nonlinearData) - np.min(nonlinearData)
        assert np.all(np.abs(pred - np.mean(nonlinearData)) < 5 * yRange)

    def test_confidenceIntervalOrdering(self, seasonalData):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert np.all(lower <= upper + 1e-6)

    def test_residualsComputed(self, seasonalData):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(seasonalData)
        assert model.residuals is not None
        assert len(model.residuals) == len(seasonalData)
        assert np.all(np.isfinite(model.residuals))

    def test_seedReproducibility(self, seasonalData):
        model1 = EchoStateForecaster(reservoirSize=50, seed=123)
        model1.fit(seasonalData)
        pred1, _, _ = model1.predict(10)

        model2 = EchoStateForecaster(reservoirSize=50, seed=123)
        model2.fit(seasonalData)
        pred2, _, _ = model2.predict(10)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_confidenceWidthGrows(self, seasonalData):
        model = EchoStateForecaster(reservoirSize=50, seed=42)
        model.fit(seasonalData)
        _, lower, upper = model.predict(20)
        widths = upper - lower
        assert widths[-1] > widths[0]


class TestAdaptiveThetaEnsemble:

    def test_fitPredictSeasonal(self, seasonalData):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert len(pred) == 12
        assert np.all(np.isfinite(pred))

    def test_fitPredictTrending(self, trendData):
        model = AdaptiveThetaEnsemble()
        model.fit(trendData)
        pred, lower, upper = model.predict(10)
        assert len(pred) == 10
        assert np.all(np.isfinite(pred))

    def test_fitPredictShort(self, shortData):
        model = AdaptiveThetaEnsemble()
        model.fit(shortData)
        pred, lower, upper = model.predict(5)
        assert len(pred) == 5
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    @pytest.mark.parametrize("steps", [1, 5, 12, 24])
    def test_predictLengthMatchesSteps(self, seasonalData, steps):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(steps)
        assert len(pred) == steps
        assert len(lower) == steps
        assert len(upper) == steps

    def test_predictionsFinite(self, seasonalData):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lower))
        assert np.all(np.isfinite(upper))

    def test_predictionsReasonableRange(self, seasonalData):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        pred, _, _ = model.predict(12)
        yMin = np.min(seasonalData)
        yMax = np.max(seasonalData)
        yRange = yMax - yMin
        assert np.all(pred > yMin - 3 * yRange)
        assert np.all(pred < yMax + 3 * yRange)

    def test_unfittedModelRaisesError(self):
        model = AdaptiveThetaEnsemble()
        with pytest.raises(ValueError, match="fit"):
            model.predict(5)

    def test_m4StyleHoldoutValidation(self, m4StyleData):
        holdout = 12
        train = m4StyleData[:-holdout]
        actual = m4StyleData[-holdout:]

        model = AdaptiveThetaEnsemble(period=12)
        model.fit(train)
        pred, _, _ = model.predict(holdout)

        smape = np.mean(2.0 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-10))
        assert smape < 1.0

    def test_confidenceIntervalOrdering(self, seasonalData):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        pred, lower, upper = model.predict(12)
        assert np.all(lower <= upper + 1e-6)

    def test_residualsComputed(self, seasonalData):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        assert model.residuals is not None
        assert len(model.residuals) == len(seasonalData)
        assert np.all(np.isfinite(model.residuals))

    def test_weightsNormalized(self, seasonalData):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        assert model._weights is not None
        assert abs(np.sum(model._weights) - 1.0) < 1e-10
        assert np.all(model._weights >= 0)

    def test_customThetaValues(self, seasonalData):
        model = AdaptiveThetaEnsemble(thetaValues=[0, 1, 2])
        model.fit(seasonalData)
        pred, _, _ = model.predict(7)
        assert len(pred) == 7
        assert np.all(np.isfinite(pred))
        assert len(model._models) == 3
        assert len(model._weights) == 3

    def test_confidenceWidthGrows(self, seasonalData):
        model = AdaptiveThetaEnsemble()
        model.fit(seasonalData)
        _, lower, upper = model.predict(20)
        widths = upper - lower
        assert widths[-1] > widths[0]
