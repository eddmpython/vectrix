import numpy as np
import pytest

from tests.conftest import (
    SEASONAL_FACTORIES,
    INTERMITTENT_FACTORIES,
    GARCH_FACTORIES,
)


class TestSeasonalModelsFitPredict:

    @pytest.mark.parametrize("modelFactory", SEASONAL_FACTORIES)
    def test_fitPredictSeasonal(self, modelFactory, seasonalData7):
        model = modelFactory()
        model.fit(seasonalData7)
        pred, lower, upper = model.predict(12)
        assert len(pred) == 12
        assert np.all(np.isfinite(pred))

    @pytest.mark.parametrize("modelFactory", SEASONAL_FACTORIES)
    @pytest.mark.parametrize("steps", [1, 5, 24])
    def test_predictLengthMatchesSteps(self, modelFactory, seasonalData7, steps):
        model = modelFactory()
        model.fit(seasonalData7)
        pred, lo, hi = model.predict(steps)
        assert len(pred) == steps
        assert len(lo) == steps
        assert len(hi) == steps

    @pytest.mark.parametrize("modelFactory", SEASONAL_FACTORIES)
    def test_confidenceIntervalOrdering(self, modelFactory, seasonalData7):
        model = modelFactory()
        model.fit(seasonalData7)
        pred, lo, hi = model.predict(12)
        assert np.all(lo <= hi + 1e-6)

    @pytest.mark.parametrize("modelFactory", SEASONAL_FACTORIES)
    def test_predictionsFinite(self, modelFactory, seasonalData12):
        model = modelFactory()
        model.fit(seasonalData12)
        pred, lo, hi = model.predict(6)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(lo))
        assert np.all(np.isfinite(hi))


class TestIntermittentModelsFitPredict:

    @pytest.mark.parametrize("modelFactory", INTERMITTENT_FACTORIES)
    def test_fitPredictIntermittent(self, modelFactory, intermittentData):
        model = modelFactory()
        model.fit(intermittentData)
        pred, lower, upper = model.predict(12)
        assert len(pred) == 12
        assert np.all(np.isfinite(pred))
        assert np.all(pred >= 0)

    @pytest.mark.parametrize("modelFactory", INTERMITTENT_FACTORIES)
    @pytest.mark.parametrize("steps", [1, 5, 24])
    def test_predictLengthMatchesSteps(self, modelFactory, intermittentData, steps):
        model = modelFactory()
        model.fit(intermittentData)
        pred, lo, hi = model.predict(steps)
        assert len(pred) == steps
        assert len(lo) == steps
        assert len(hi) == steps

    @pytest.mark.parametrize("modelFactory", INTERMITTENT_FACTORIES)
    def test_confidenceIntervalOrdering(self, modelFactory, intermittentData):
        model = modelFactory()
        model.fit(intermittentData)
        pred, lo, hi = model.predict(12)
        assert np.all(lo <= hi + 1e-6)

    @pytest.mark.parametrize("modelFactory", INTERMITTENT_FACTORIES)
    def test_nonNegativePredictions(self, modelFactory, intermittentData):
        model = modelFactory()
        model.fit(intermittentData)
        pred, lo, hi = model.predict(12)
        assert np.all(pred >= 0)
        assert np.all(lo >= 0)


class TestGARCHModelsFitPredict:

    @pytest.mark.parametrize("modelFactory", GARCH_FACTORIES)
    def test_fitPredictVolatile(self, modelFactory, volatileData):
        model = modelFactory()
        model.fit(volatileData)
        pred, lower, upper = model.predict(12)
        assert len(pred) == 12
        assert np.all(np.isfinite(pred))

    @pytest.mark.parametrize("modelFactory", GARCH_FACTORIES)
    @pytest.mark.parametrize("steps", [1, 5, 24])
    def test_predictLengthMatchesSteps(self, modelFactory, volatileData, steps):
        model = modelFactory()
        model.fit(volatileData)
        pred, lo, hi = model.predict(steps)
        assert len(pred) == steps
        assert len(lo) == steps
        assert len(hi) == steps

    @pytest.mark.parametrize("modelFactory", GARCH_FACTORIES)
    def test_confidenceIntervalOrdering(self, modelFactory, volatileData):
        model = modelFactory()
        model.fit(volatileData)
        pred, lo, hi = model.predict(12)
        assert np.all(lo <= hi + 1e-6)

    @pytest.mark.parametrize("modelFactory", GARCH_FACTORIES)
    def test_variancePositive(self, modelFactory, volatileData):
        model = modelFactory()
        model.fit(volatileData)
        pred, lo, hi = model.predict(12)
        widths = hi - lo
        assert np.all(widths >= 0)
