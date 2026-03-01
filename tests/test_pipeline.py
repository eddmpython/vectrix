"""파이프라인 시스템 테스트"""

import numpy as np
import pytest

from vectrix.pipeline import (
    BaseTransformer,
    BoxCoxTransformer,
    Deseasonalizer,
    Detrend,
    Differencer,
    ForecastPipeline,
    LogTransformer,
    MissingValueImputer,
    OutlierClipper,
    Scaler,
)


class _DummyForecaster:
    def __init__(self):
        self.fitted = False
        self._y = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).ravel()
        self.fitted = True
        return self

    def predict(self, steps):
        last = self._y[-1]
        pred = np.full(steps, last)
        lo = pred - 1.0
        hi = pred + 1.0
        return pred, lo, hi


class TestScaler:
    def test_zscoreRoundTrip(self):
        rng = np.random.default_rng(42)
        y = rng.normal(50, 10, 100)
        s = Scaler(method='zscore')
        transformed = s.fitTransform(y)
        recovered = s.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, atol=1e-10)

    def test_minmaxRoundTrip(self):
        rng = np.random.default_rng(42)
        y = rng.uniform(10, 100, 100)
        s = Scaler(method='minmax')
        transformed = s.fitTransform(y)
        assert np.min(transformed) >= -1e-10
        assert np.max(transformed) <= 1.0 + 1e-10
        recovered = s.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, atol=1e-10)

    def test_zscoreMeanStd(self):
        rng = np.random.default_rng(42)
        y = rng.normal(100, 20, 200)
        s = Scaler(method='zscore')
        transformed = s.fitTransform(y)
        assert abs(np.mean(transformed)) < 0.1
        assert abs(np.std(transformed) - 1.0) < 0.1


class TestLogTransformer:
    def test_positiveRoundTrip(self):
        y = np.array([1.0, 10.0, 100.0, 1000.0])
        lt = LogTransformer()
        transformed = lt.fitTransform(y)
        recovered = lt.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, atol=1e-10)

    def test_negativeAutoShift(self):
        y = np.array([-5.0, -2.0, 0.0, 3.0, 10.0])
        lt = LogTransformer()
        transformed = lt.fitTransform(y)
        assert np.all(np.isfinite(transformed))
        recovered = lt.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, atol=1e-10)


class TestDifferencer:
    def test_order1RoundTrip(self):
        y = np.array([10.0, 12.0, 15.0, 20.0, 18.0, 22.0])
        d = Differencer(d=1)
        d.fit(y)
        transformed = d.transform(y)
        assert len(transformed) == len(y) - 1
        pred = np.array([3.0, -1.0])
        recovered = d.inverseTransform(pred)
        assert len(recovered) == 2
        assert abs(recovered[0] - (22.0 + 3.0)) < 1e-10

    def test_order2(self):
        y = np.arange(10, dtype=np.float64)
        d = Differencer(d=2)
        d.fit(y)
        transformed = d.transform(y)
        assert len(transformed) == len(y) - 2


class TestBoxCoxTransformer:
    def test_roundTrip(self):
        y = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
        bc = BoxCoxTransformer()
        transformed = bc.fitTransform(y)
        recovered = bc.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, rtol=1e-4)

    def test_fixedLambda(self):
        y = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        bc = BoxCoxTransformer(lmbda=0.5)
        transformed = bc.fitTransform(y)
        recovered = bc.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, rtol=1e-4)


class TestDeseasonalizer:
    def test_removeSeasonality(self):
        rng = np.random.default_rng(42)
        seasonal = np.tile([0, 5, 10, 5, 0, -5, -10], 10)
        y = 100 + seasonal + rng.normal(0, 0.1, 70)
        ds = Deseasonalizer(period=7)
        transformed = ds.fitTransform(y)
        assert np.std(transformed) < np.std(y)

    def test_roundTrip(self):
        seasonal = np.tile([0, 5, 10, 5, 0, -5, -10], 10)
        y = 100 + seasonal.astype(np.float64)
        ds = Deseasonalizer(period=7)
        transformed = ds.fitTransform(y)
        recovered = ds.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, atol=1e-10)


class TestDetrend:
    def test_removeTrend(self):
        t = np.arange(100, dtype=np.float64)
        y = 50.0 + 2.0 * t
        dt = Detrend()
        transformed = dt.fitTransform(y)
        assert np.std(transformed) < 1.0

    def test_inverseOnFuture(self):
        t = np.arange(50, dtype=np.float64)
        y = 10.0 + 3.0 * t
        dt = Detrend()
        dt.fit(y)
        futurePred = np.zeros(5)
        recovered = dt.inverseTransform(futurePred)
        for i in range(5):
            expected = 10.0 + 3.0 * (50 + i)
            assert abs(recovered[i] - expected) < 1e-8


class TestOutlierClipper:
    def test_clipping(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        y[50] = 100.0
        y[51] = -100.0
        oc = OutlierClipper(factor=3.0)
        transformed = oc.fitTransform(y)
        assert np.max(transformed) < 100.0
        assert np.min(transformed) > -100.0


class TestMissingValueImputer:
    def test_linearInterpolation(self):
        y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        imp = MissingValueImputer(method='linear')
        result = imp.fitTransform(y)
        assert np.all(np.isfinite(result))
        assert abs(result[1] - 2.0) < 1e-10
        assert abs(result[3] - 4.0) < 1e-10

    def test_meanImputation(self):
        y = np.array([10.0, np.nan, 20.0, np.nan, 30.0])
        imp = MissingValueImputer(method='mean')
        result = imp.fitTransform(y)
        assert np.all(np.isfinite(result))
        assert abs(result[1] - 20.0) < 1e-10

    def test_ffillImputation(self):
        y = np.array([5.0, np.nan, np.nan, 10.0, np.nan])
        imp = MissingValueImputer(method='ffill')
        result = imp.fitTransform(y)
        assert np.all(np.isfinite(result))
        assert abs(result[1] - 5.0) < 1e-10
        assert abs(result[2] - 5.0) < 1e-10


class TestForecastPipeline:
    def test_fitPredict(self):
        rng = np.random.default_rng(42)
        y = rng.normal(100, 10, 200)

        pipe = ForecastPipeline([
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        pipe.fit(y)
        pred, lo, hi = pipe.predict(10)
        assert pred.shape == (10,)
        assert lo.shape == (10,)
        assert hi.shape == (10,)
        assert np.all(lo < hi)

    def test_multiStepTransform(self):
        rng = np.random.default_rng(42)
        y = np.abs(rng.normal(50, 10, 200))

        pipe = ForecastPipeline([
            ('log', LogTransformer()),
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        pipe.fit(y)
        pred, lo, hi = pipe.predict(5)
        assert pred.shape == (5,)
        assert np.all(np.isfinite(pred))

    def test_transformOnly(self):
        rng = np.random.default_rng(42)
        y = rng.normal(100, 10, 50)
        pipe = ForecastPipeline([
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        pipe.fit(y)
        transformed = pipe.transform(y)
        assert len(transformed) == 50
        recovered = pipe.inverseTransform(transformed)
        np.testing.assert_allclose(recovered, y, atol=1e-10)

    def test_getStep(self):
        pipe = ForecastPipeline([
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        scaler = pipe.getStep('scale')
        assert isinstance(scaler, Scaler)

    def test_getStepNotFound(self):
        pipe = ForecastPipeline([
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        with pytest.raises(KeyError):
            pipe.getStep('missing')

    def test_duplicateNamesRaises(self):
        with pytest.raises(ValueError, match="unique"):
            ForecastPipeline([
                ('scale', Scaler()),
                ('scale', Scaler()),
            ])

    def test_emptyPipelineRaises(self):
        with pytest.raises(ValueError, match="at least"):
            ForecastPipeline([])

    def test_unfittedPredictRaises(self):
        pipe = ForecastPipeline([
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        with pytest.raises(ValueError, match="not been fitted"):
            pipe.predict(5)

    def test_listSteps(self):
        pipe = ForecastPipeline([
            ('impute', MissingValueImputer()),
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        assert pipe.listSteps() == ['impute', 'scale', 'forecast']

    def test_getParams(self):
        pipe = ForecastPipeline([
            ('scale', Scaler(method='minmax')),
            ('forecast', _DummyForecaster()),
        ])
        params = pipe.getParams()
        assert 'scale__method' in params
        assert params['scale__method'] == 'minmax'

    def test_repr(self):
        pipe = ForecastPipeline([
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        r = repr(pipe)
        assert 'Scaler' in r
        assert '_DummyForecaster' in r

    def test_chainedTransformersWithForecaster(self):
        rng = np.random.default_rng(42)
        y = np.abs(rng.normal(100, 20, 300)) + 1.0

        pipe = ForecastPipeline([
            ('clip', OutlierClipper()),
            ('log', LogTransformer()),
            ('scale', Scaler()),
            ('forecast', _DummyForecaster()),
        ])
        pipe.fit(y)
        pred, lo, hi = pipe.predict(10)
        assert pred.shape == (10,)
        assert np.all(np.isfinite(pred))
        assert np.all(pred > 0)
