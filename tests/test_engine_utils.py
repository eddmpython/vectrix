"""Engine utility modules tests (ARIMAX, CrossVal, Decomposition, Diagnostics, PeriodicDrop, Comparison, Impute)"""
import pytest
import numpy as np

from vectrix.engine.arimax import ARIMAXModel, AutoARIMAX
from vectrix.engine.crossval import TimeSeriesCrossValidator
from vectrix.engine.decomposition import SeasonalDecomposition, MSTLDecomposition
from vectrix.engine.diagnostics import ForecastDiagnostics, ForecastDiagnosticsResult
from vectrix.engine.periodic_drop import PeriodicDropDetector
from vectrix.engine.comparison import ModelComparison
from vectrix.engine.impute import TimeSeriesImputer


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def seasonalData(rng):
    n = 200
    t = np.arange(n, dtype=np.float64)
    trend = 0.05 * t
    seasonal = 5.0 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 0.5, n)
    return trend + seasonal + noise + 50.0


@pytest.fixture
def trendData(rng):
    n = 200
    t = np.arange(n, dtype=np.float64)
    return 0.5 * t + rng.normal(0, 1.0, n) + 100.0


class TestARIMAX:

    def test_fitAndPredictWithExog(self, rng):
        n = 150
        X = rng.normal(0, 1, (n, 2))
        trueCoef = np.array([3.0, -2.0])
        noise = rng.normal(0, 0.5, n)
        y = X @ trueCoef + noise

        model = ARIMAXModel(order=(1, 0, 0))
        model.fit(y, X)

        assert model.fitted is True
        assert model.regCoef is not None
        assert len(model.regCoef) == 2

        steps = 10
        XFuture = rng.normal(0, 1, (steps, 2))
        pred, lo, hi = model.predict(steps, XFuture)

        assert pred.shape == (steps,)
        assert lo.shape == (steps,)
        assert hi.shape == (steps,)
        assert np.all(lo <= pred)
        assert np.all(pred <= hi)

    def test_predictWithoutExog(self, rng):
        n = 100
        X = rng.normal(0, 1, (n, 1))
        y = 2.0 * X.ravel() + rng.normal(0, 0.3, n)

        model = ARIMAXModel(order=(1, 0, 0))
        model.fit(y, X)

        steps = 5
        pred, lo, hi = model.predict(steps, XFuture=None)
        assert pred.shape == (steps,)

    def test_predictBeforeFitRaises(self):
        model = ARIMAXModel(order=(1, 0, 0))
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(5)

    def test_oneDimensionalExog(self, rng):
        n = 100
        X = rng.normal(0, 1, n)
        y = 2.0 * X + rng.normal(0, 0.3, n)

        model = ARIMAXModel(order=(1, 0, 0))
        model.fit(y, X)
        assert model.fitted is True
        assert model.regCoef.shape == (1,)

    def test_autoArimaxFitPredict(self, rng):
        n = 120
        X = rng.normal(0, 1, (n, 1))
        y = 5.0 * X.ravel() + rng.normal(0, 0.5, n)

        autoModel = AutoARIMAX(maxP=2, maxD=1, maxQ=2)
        autoModel.fit(y, X)

        assert autoModel.fitted is True
        assert autoModel.bestOrder is not None

        steps = 5
        XFuture = rng.normal(0, 1, (steps, 1))
        pred, lo, hi = autoModel.predict(steps, XFuture)
        assert pred.shape == (steps,)


class TestCrossValidation:

    def test_expandingSplitCounts(self):
        y = np.arange(200, dtype=np.float64)
        cv = TimeSeriesCrossValidator(n_splits=5, horizon=10, strategy='expanding', min_train_size=50)
        splits = cv.split(y)
        assert len(splits) >= 1
        assert len(splits) <= 5

        for trainIdx, testIdx in splits:
            assert trainIdx[0] == 0
            assert len(testIdx) <= 10

    def test_slidingWindowSplits(self):
        y = np.arange(200, dtype=np.float64)
        cv = TimeSeriesCrossValidator(n_splits=5, horizon=10, strategy='sliding', min_train_size=50)
        splits = cv.split(y)
        assert len(splits) >= 1

        for trainIdx, testIdx in splits:
            assert len(trainIdx) <= 50

    def test_expandingTrainGrows(self):
        y = np.arange(300, dtype=np.float64)
        cv = TimeSeriesCrossValidator(n_splits=3, horizon=10, strategy='expanding', min_train_size=50)
        splits = cv.split(y)
        if len(splits) >= 2:
            assert len(splits[1][0]) >= len(splits[0][0])

    def test_tooShortDataReturnsEmpty(self):
        y = np.arange(10, dtype=np.float64)
        cv = TimeSeriesCrossValidator(n_splits=3, horizon=20, min_train_size=50)
        splits = cv.split(y)
        assert len(splits) == 0

    def test_evaluateReturnsMetrics(self, rng):
        n = 200
        t = np.arange(n, dtype=np.float64)
        y = 0.1 * t + rng.normal(0, 1, n) + 50.0

        class SimpleModel:
            def fit(self, y):
                self._lastVal = y[-1]
            def predict(self, steps):
                pred = np.full(steps, self._lastVal)
                return pred, pred - 5, pred + 5

        cv = TimeSeriesCrossValidator(n_splits=3, horizon=10, strategy='expanding', min_train_size=50)
        result = cv.evaluate(y, lambda: SimpleModel())

        assert 'mape' in result
        assert 'rmse' in result
        assert 'mae' in result
        assert 'smape' in result
        assert 'fold_results' in result
        assert result['n_folds'] >= 1

    def test_evaluateWithInsufficientData(self):
        y = np.arange(5, dtype=np.float64)
        cv = TimeSeriesCrossValidator(n_splits=3, horizon=10, min_train_size=50)
        result = cv.evaluate(y, lambda: None)
        assert result['mape'] == np.inf
        assert result['n_folds'] == 0


class TestDecomposition:

    def test_additiveDecompositionComponents(self, seasonalData):
        decomposer = SeasonalDecomposition(period=7, model='additive', method='classical')
        result = decomposer.decompose(seasonalData)

        assert result.trend.shape == seasonalData.shape
        assert result.seasonal.shape == seasonalData.shape
        assert result.residual.shape == seasonalData.shape
        assert np.array_equal(result.observed, seasonalData)

        reconstructed = result.trend + result.seasonal + result.residual
        np.testing.assert_allclose(reconstructed, seasonalData, atol=1e-10)

    def test_multiplicativeDecomposition(self, rng):
        n = 140
        t = np.arange(n, dtype=np.float64)
        trend = 100.0 + 0.5 * t
        seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * t / 7)
        y = trend * seasonal + rng.normal(0, 0.5, n)
        y = np.maximum(y, 1.0)

        decomposer = SeasonalDecomposition(period=7, model='multiplicative', method='classical')
        result = decomposer.decompose(y)

        assert result.trend.shape == y.shape
        assert result.seasonal.shape == y.shape
        assert result.residual.shape == y.shape

    def test_stlDecomposition(self, seasonalData):
        decomposer = SeasonalDecomposition(period=7, model='additive', method='stl')
        result = decomposer.decompose(seasonalData)

        assert result.trend.shape == seasonalData.shape
        assert result.seasonal.shape == seasonalData.shape
        assert result.residual.shape == seasonalData.shape

        residualStd = np.std(result.residual)
        dataStd = np.std(seasonalData)
        assert residualStd < dataStd

    def test_extractSeasonalAndTrend(self, seasonalData):
        decomposer = SeasonalDecomposition(period=7, model='additive', method='classical')
        seasonal = decomposer.extractSeasonal(seasonalData)
        trend = decomposer.extractTrend(seasonalData)

        assert seasonal.shape == seasonalData.shape
        assert trend.shape == seasonalData.shape
        assert np.std(seasonal) > 0.1

    def test_deseasonalize(self, seasonalData):
        decomposer = SeasonalDecomposition(period=7, model='additive', method='classical')
        adjusted = decomposer.deseasonalize(seasonalData)

        assert adjusted.shape == seasonalData.shape
        assert np.std(adjusted) < np.std(seasonalData)

    def test_mstlDecomposition(self, rng):
        n = 365
        t = np.arange(n, dtype=np.float64)
        weekly = 5.0 * np.sin(2 * np.pi * t / 7)
        monthly = 3.0 * np.sin(2 * np.pi * t / 30)
        trend = 0.02 * t
        noise = rng.normal(0, 0.5, n)
        y = trend + weekly + monthly + noise + 50.0

        mstl = MSTLDecomposition(periods=[7, 30], model='additive')
        result = mstl.decompose(y)

        assert 'trend' in result
        assert 'seasonals' in result
        assert 'residual' in result
        assert 7 in result['seasonals']

    def test_mstlPredict(self, rng):
        n = 200
        t = np.arange(n, dtype=np.float64)
        y = 0.05 * t + 3.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.5, n) + 50.0

        mstl = MSTLDecomposition(periods=[7], model='additive')
        pred = mstl.predict(y, steps=14)

        assert pred.shape == (14,)
        assert np.all(np.isfinite(pred))


class TestDiagnostics:

    def test_whiteNoiseResiduals(self, rng):
        residuals = rng.normal(0, 1, 200)
        diag = ForecastDiagnostics()
        result = diag.analyze(residuals)

        assert isinstance(result, ForecastDiagnosticsResult)
        assert result.isWhiteNoise is True

    def test_autocorrelatedResiduals(self):
        n = 200
        residuals = np.zeros(n)
        residuals[0] = 1.0
        for i in range(1, n):
            residuals[i] = 0.9 * residuals[i - 1] + np.random.default_rng(42).normal(0, 0.1)

        diag = ForecastDiagnostics()
        result = diag.analyze(residuals)
        assert result.isWhiteNoise is False

    def test_ljungBoxTest(self, rng):
        residuals = rng.normal(0, 1, 100)
        diag = ForecastDiagnostics()
        qStat, pValue = diag.ljungBoxTest(residuals, maxLag=10)

        assert isinstance(qStat, float)
        assert isinstance(pValue, float)
        assert 0.0 <= pValue <= 1.0
        assert qStat >= 0.0

    def test_jarqueBeraTest(self, rng):
        residuals = rng.normal(0, 1, 500)
        diag = ForecastDiagnostics()
        jbStat, pValue, skew, kurt = diag.jarqueBeraTest(residuals)

        assert isinstance(jbStat, float)
        assert 0.0 <= pValue <= 1.0
        assert abs(skew) < 1.0
        assert abs(kurt - 3.0) < 1.5

    def test_archTest(self, rng):
        residuals = rng.normal(0, 1, 200)
        diag = ForecastDiagnostics()
        lmStat, pValue = diag.archTest(residuals, lags=5)

        assert isinstance(lmStat, float)
        assert 0.0 <= pValue <= 1.0

    def test_durbinWatsonTest(self, rng):
        residuals = rng.normal(0, 1, 100)
        diag = ForecastDiagnostics()
        dwStat = diag.durbinWatsonTest(residuals)

        assert isinstance(dwStat, float)
        assert 0.0 <= dwStat <= 4.0
        assert abs(dwStat - 2.0) < 0.5

    def test_acfComputation(self, rng):
        residuals = rng.normal(0, 1, 100)
        diag = ForecastDiagnostics()
        acfVals = diag.acf(residuals, maxLag=20)

        assert len(acfVals) == 21
        assert acfVals[0] == pytest.approx(1.0)
        assert np.all(np.abs(acfVals) <= 1.0 + 1e-10)

    def test_issueIdentification(self):
        n = 200
        rng = np.random.default_rng(99)
        residuals = np.zeros(n)
        residuals[0] = rng.normal(0, 1)
        for i in range(1, n):
            residuals[i] = 0.85 * residuals[i - 1] + rng.normal(0, 0.2)

        diag = ForecastDiagnostics()
        result = diag.analyze(residuals)
        assert len(result.issues) >= 1

    def test_shortResidualsGraceful(self):
        diag = ForecastDiagnostics()
        result = diag.analyze(np.array([1.0, 2.0, 3.0]))
        assert len(result.issues) >= 1

    def test_summaryString(self, rng):
        residuals = rng.normal(0, 1, 100)
        diag = ForecastDiagnostics()
        result = diag.analyze(residuals)
        summary = result.summary()
        assert isinstance(summary, str)
        assert "Ljung-Box" in summary
        assert "Jarque-Bera" in summary
        assert "ARCH" in summary
        assert "Durbin-Watson" in summary


class TestPeriodicDrop:

    def test_detectPeriodicDrops(self, rng):
        n = 400
        y = np.ones(n) * 100.0 + rng.normal(0, 2, n)
        for cycle in range(4):
            start = 90 * cycle + 80
            end = start + 7
            if end < n:
                y[start:end] = 10.0

        detector = PeriodicDropDetector(minDropRatio=0.8, minDropDuration=3)
        found = detector.detect(y)

        assert found is True
        assert detector.dropPeriod is not None
        assert detector.dropRatio is not None
        assert len(detector.detectedDrops) >= 2

    def test_noDropInStableData(self, rng):
        y = rng.normal(100, 2, 200)
        detector = PeriodicDropDetector()
        found = detector.detect(y)
        assert found is False

    def test_removeDrops(self, rng):
        n = 400
        y = np.ones(n) * 100.0 + rng.normal(0, 1, n)
        for cycle in range(4):
            start = 90 * cycle + 80
            end = start + 7
            if end < n:
                y[start:end] = 5.0

        detector = PeriodicDropDetector(minDropRatio=0.8, minDropDuration=3)
        detector.detect(y)

        cleaned = detector.removeDrops(y)
        assert cleaned.shape == y.shape
        for start, end in detector.detectedDrops:
            assert np.mean(cleaned[start:end]) > np.mean(y[start:end])

    def test_applyDropPattern(self, rng):
        n = 400
        y = np.ones(n) * 100.0 + rng.normal(0, 1, n)
        for cycle in range(4):
            start = 90 * cycle + 80
            end = start + 7
            if end < n:
                y[start:end] = 10.0

        detector = PeriodicDropDetector(minDropRatio=0.8, minDropDuration=3)
        detector.detect(y)

        predictions = np.ones(100) * 100.0
        adjusted = detector.applyDropPattern(predictions, startIdx=n)
        assert adjusted.shape == (100,)

    def test_shortDataReturnsFalse(self):
        y = np.array([1.0, 2.0, 3.0])
        detector = PeriodicDropDetector()
        assert detector.detect(y) is False

    def test_hasPeriodicDrop(self, rng):
        detector = PeriodicDropDetector()
        assert detector.hasPeriodicDrop() is False

        n = 400
        y = np.ones(n) * 100.0 + rng.normal(0, 1, n)
        for cycle in range(4):
            start = 90 * cycle + 80
            end = start + 7
            if end < n:
                y[start:end] = 5.0
        detector.detect(y)
        assert detector.hasPeriodicDrop() is True


class TestComparison:

    def test_dieboldMarianoTwoSided(self, rng):
        errors1 = rng.normal(0, 2, 100)
        errors2 = rng.normal(0, 1, 100)

        result = ModelComparison.dieboldMariano(errors1, errors2)
        assert 'statistic' in result
        assert 'pValue' in result
        assert 'conclusion' in result
        assert 0.0 <= result['pValue'] <= 1.0

    def test_dieboldMarianoIdenticalErrors(self, rng):
        errors = rng.normal(0, 1, 100)
        result = ModelComparison.dieboldMariano(errors, errors)
        assert result['pValue'] >= 0.05

    def test_dieboldMarianoShortData(self):
        result = ModelComparison.dieboldMariano(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
        assert result['pValue'] == 1.0

    def test_dieboldMarianoLengthMismatch(self):
        with pytest.raises(ValueError):
            ModelComparison.dieboldMariano(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))

    def test_modelComparisonTable(self, rng):
        modelErrors = {
            'ModelA': rng.normal(0, 2, 50),
            'ModelB': rng.normal(0, 1, 50),
            'ModelC': rng.normal(0, 3, 50),
        }
        result = ModelComparison.modelComparisonTable(modelErrors)

        assert 'rankings' in result
        assert 'metrics' in result
        assert 'pairwiseTests' in result
        assert len(result['rankings']) == 3
        assert 'ModelA' in result['metrics']

    def test_modelComparisonTableEmpty(self):
        result = ModelComparison.modelComparisonTable({})
        assert result['rankings'] == []
        assert result['metrics'] == {}

    def test_giacominiWhite(self, rng):
        errors1 = rng.normal(0, 2, 50)
        errors2 = rng.normal(0, 1, 50)
        result = ModelComparison.giacominiWhite(errors1, errors2)

        assert 'statistic' in result
        assert 'pValue' in result
        assert 0.0 <= result['pValue'] <= 1.0

    def test_forecastEncompassingTest(self, rng):
        errors1 = rng.normal(0, 1, 50)
        errors2 = rng.normal(0, 1, 50) + 2.0
        result = ModelComparison.forecastEncompassingTest(errors1, errors2)

        assert 'statistic' in result
        assert 'pValue' in result
        assert 0.0 <= result['pValue'] <= 1.0


class TestImpute:

    def test_linearInterpolation(self):
        y = np.array([1.0, np.nan, np.nan, 4.0, 5.0])
        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='linear')

        assert not np.any(np.isnan(result))
        assert result[1] == pytest.approx(2.0, abs=0.01)
        assert result[2] == pytest.approx(3.0, abs=0.01)

    def test_seasonalInterpolation(self):
        period = 4
        base = np.array([10.0, 20.0, 30.0, 40.0])
        y = np.tile(base, 5).astype(np.float64)
        y[5] = np.nan

        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='seasonal', period=period)

        assert not np.any(np.isnan(result))
        assert result[5] == pytest.approx(20.0, abs=0.01)

    def test_locf(self):
        y = np.array([1.0, 2.0, np.nan, np.nan, 5.0])
        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='locf')

        assert not np.any(np.isnan(result))
        assert result[2] == 2.0
        assert result[3] == 2.0

    def test_nocb(self):
        y = np.array([np.nan, np.nan, 3.0, 4.0, 5.0])
        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='nocb')

        assert not np.any(np.isnan(result))
        assert result[0] == 3.0
        assert result[1] == 3.0

    def test_autoImpute(self, rng):
        y = rng.normal(50, 5, 100)
        y[10] = np.nan
        y[30] = np.nan
        y[50] = np.nan

        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='auto', period=7)
        assert not np.any(np.isnan(result))

    def test_allNanReturnsZeros(self):
        y = np.array([np.nan, np.nan, np.nan])
        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='linear')
        assert not np.any(np.isnan(result))
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_noNanReturnsOriginal(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='linear')
        np.testing.assert_array_equal(result, y)

    def test_detectMissingPattern(self):
        imputer = TimeSeriesImputer()

        complete = np.array([1.0, 2.0, 3.0])
        info = imputer.detectMissing(complete)
        assert info['nMissing'] == 0
        assert info['pattern'] == 'complete'

        scattered = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])
        info = imputer.detectMissing(scattered)
        assert info['nMissing'] == 2
        assert info['pattern'] == 'scattered'
        assert info['maxConsecutiveMissing'] == 1

    def test_detectMissingEmpty(self):
        imputer = TimeSeriesImputer()
        info = imputer.detectMissing(np.array([]))
        assert info['pattern'] == 'empty'

    def test_detectMissingAllNan(self):
        imputer = TimeSeriesImputer()
        info = imputer.detectMissing(np.array([np.nan, np.nan, np.nan]))
        assert info['pattern'] == 'all_missing'
        assert info['nMissing'] == 3

    def test_unknownMethodRaises(self):
        imputer = TimeSeriesImputer()
        with pytest.raises(ValueError, match="Unknown method"):
            imputer.impute(np.array([1.0, np.nan, 3.0]), method='unknown')
