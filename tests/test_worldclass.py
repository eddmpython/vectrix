"""Phase 2 Feature Validation: Probabilistic, Imputation, Logistic Growth, Model Comparison"""
import pytest
import numpy as np

from forecastx.engine.probabilistic import ProbabilisticForecaster
from forecastx.engine.impute import TimeSeriesImputer
from forecastx.engine.logistic import LogisticGrowthModel, SaturatingTrendModel
from forecastx.engine.comparison import ModelComparison


@pytest.fixture
def rng():
    np.random.seed(42)
    return np.random.RandomState(42)


@pytest.fixture
def logisticData():
    np.random.seed(42)
    return 100 / (1 + np.exp(-0.05 * (np.arange(200) - 100))) + np.random.randn(200) * 2


class TestProbabilisticForecasting:

    def testQuantileForecastBasic(self):
        np.random.seed(42)
        pf = ProbabilisticForecaster()
        predictions = np.random.randn(20) + 50.0
        residuals = np.random.randn(100)
        steps = 10
        quantiles = [0.1, 0.5, 0.9]
        result = pf.quantileForecast(predictions, residuals, steps, quantiles)
        assert 0.1 in result
        assert 0.5 in result
        assert 0.9 in result
        for h in range(steps):
            assert result[0.1][h] < result[0.5][h], \
                f"q0.1 should be < q0.5 at step {h}"
            assert result[0.5][h] < result[0.9][h], \
                f"q0.5 should be < q0.9 at step {h}"

    def testQuantileForecastWidening(self):
        np.random.seed(42)
        pf = ProbabilisticForecaster()
        predictions = np.full(20, 50.0)
        residuals = np.random.randn(100)
        steps = 10
        quantiles = [0.1, 0.9]
        result = pf.quantileForecast(predictions, residuals, steps, quantiles)
        widths = result[0.9] - result[0.1]
        for h in range(1, steps):
            assert widths[h] > widths[h - 1], \
                f"Band width should widen: step {h} ({widths[h]:.4f}) <= step {h-1} ({widths[h-1]:.4f})"

    def testCrpsPerfectForecast(self):
        actual = 5.0
        predictedMean = 5.0
        predictedStd = 1.0
        score = ProbabilisticForecaster.crps(actual, predictedMean, predictedStd)
        assert score < 0.6, \
            f"CRPS for perfect mean should be near 0, got {score:.4f}"

    def testCrpsBadForecast(self):
        actual = 5.0
        predictedMean = 50.0
        predictedStd = 1.0
        score = ProbabilisticForecaster.crps(actual, predictedMean, predictedStd)
        assert score > 1.0, \
            f"CRPS for bad forecast should be > 1.0, got {score:.4f}"

    def testQuantileLossSymmetric(self):
        actual = 10.0
        predicted = 7.0
        quantile = 0.5
        loss = ProbabilisticForecaster.quantileLoss(actual, predicted, quantile)
        expected = 0.5 * abs(actual - predicted)
        assert abs(loss - expected) < 1e-10, \
            f"Pinball loss at q=0.5 should equal 0.5*|actual-predicted|={expected}, got {loss}"

    def testWinklerScoreInside(self):
        actual = 5.0
        lower = 3.0
        upper = 7.0
        alpha = 0.05
        score = ProbabilisticForecaster.winklerScore(actual, lower, upper, alpha)
        expectedWidth = upper - lower
        assert abs(score - expectedWidth) < 1e-10, \
            f"When actual is inside, score should equal interval width {expectedWidth}, got {score}"

    def testWinklerScoreOutside(self):
        actual = 10.0
        lower = 3.0
        upper = 7.0
        alpha = 0.05
        score = ProbabilisticForecaster.winklerScore(actual, lower, upper, alpha)
        intervalWidth = upper - lower
        penalty = (2.0 / alpha) * (actual - upper)
        expected = intervalWidth + penalty
        assert abs(score - expected) < 1e-10, \
            f"When actual is outside, score should include penalty: expected {expected}, got {score}"
        assert score > intervalWidth, \
            f"Score ({score}) should exceed interval width ({intervalWidth}) when actual is outside"


class TestMissingValueHandling:

    def testLinearInterpolate(self):
        imputer = TimeSeriesImputer()
        y = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = imputer.linearInterpolate(y)
        assert not np.any(np.isnan(result)), "No NaN should remain after linear interpolation"
        assert abs(result[1] - 2.0) < 1e-10, f"Interpolated value at index 1 should be 2.0, got {result[1]}"
        assert abs(result[3] - 4.0) < 1e-10, f"Interpolated value at index 3 should be 4.0, got {result[3]}"

    def testSeasonalInterpolate(self):
        np.random.seed(42)
        period = 7
        nCycles = 10
        seasonal = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        y = np.tile(seasonal, nCycles).astype(np.float64)
        nanPositions = [7, 14, 21]
        y[nanPositions] = np.nan
        imputer = TimeSeriesImputer()
        result = imputer.seasonalInterpolate(y, period)
        assert not np.any(np.isnan(result)), "No NaN should remain after seasonal interpolation"
        for pos in nanPositions:
            expectedVal = seasonal[pos % period]
            assert abs(result[pos] - expectedVal) < 1e-10, \
                f"Seasonal interpolation at {pos} should be {expectedVal}, got {result[pos]}"

    def testLocf(self):
        imputer = TimeSeriesImputer()
        y = np.array([1.0, 2.0, np.nan, np.nan, 5.0, np.nan])
        result = imputer.locf(y)
        assert abs(result[2] - 2.0) < 1e-10, f"LOCF at index 2 should be 2.0, got {result[2]}"
        assert abs(result[3] - 2.0) < 1e-10, f"LOCF at index 3 should be 2.0, got {result[3]}"
        assert abs(result[5] - 5.0) < 1e-10, f"LOCF at index 5 should be 5.0, got {result[5]}"

    def testDetectMissingComplete(self):
        imputer = TimeSeriesImputer()
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        info = imputer.detectMissing(y)
        assert info['pattern'] == 'complete', \
            f"Pattern for complete data should be 'complete', got '{info['pattern']}'"
        assert info['nMissing'] == 0

    def testDetectMissingScattered(self):
        np.random.seed(42)
        y = np.arange(100, dtype=np.float64)
        nanIdx = np.random.choice(100, size=5, replace=False)
        y[nanIdx] = np.nan
        imputer = TimeSeriesImputer()
        info = imputer.detectMissing(y)
        assert info['pattern'] == 'scattered', \
            f"Pattern for random NaN should be 'scattered', got '{info['pattern']}'"
        assert info['nMissing'] == 5

    def testImputeAuto(self):
        np.random.seed(42)
        y = np.arange(50, dtype=np.float64)
        y[10] = np.nan
        y[25] = np.nan
        y[40] = np.nan
        imputer = TimeSeriesImputer()
        result = imputer.impute(y, method='auto')
        assert not np.any(np.isnan(result)), "Auto imputation should handle all NaN values"
        assert len(result) == len(y)


class TestLogisticGrowth:

    def testLogisticGrowthBasic(self, logisticData):
        model = LogisticGrowthModel(cap=100.0)
        model.fit(logisticData)
        predictions, lower95, upper95 = model.predict(50)
        assert len(predictions) == 50
        assert all(p <= 105.0 for p in predictions), \
            "Predictions should approach cap (100) and not drastically exceed it"
        assert predictions[-1] > predictions[0] or abs(predictions[-1] - predictions[0]) < 5.0, \
            "Predictions should be increasing or near saturation"

    def testLogisticGrowthAutoCap(self, logisticData):
        model = LogisticGrowthModel(cap=None)
        model.fit(logisticData)
        assert model.fittedCap is not None, "Auto cap should be estimated"
        assert model.fittedCap > np.max(logisticData), \
            f"Auto cap ({model.fittedCap:.2f}) should be > max(y) ({np.max(logisticData):.2f})"

    def testSaturatingTrendWithSeason(self, logisticData):
        period = 12
        seasonal = 5.0 * np.sin(2 * np.pi * np.arange(len(logisticData)) / period)
        yWithSeason = logisticData + seasonal
        model = SaturatingTrendModel(cap=110.0, period=period)
        model.fit(yWithSeason)
        predictions, lower95, upper95 = model.predict(24)
        assert len(predictions) == 24
        assert model.seasonal is not None, "Seasonal component should be extracted"
        assert len(model.seasonal) == period, \
            f"Seasonal array length should be {period}, got {len(model.seasonal)}"


class TestModelComparison:

    def testDieboldMarianoEqualModels(self):
        np.random.seed(42)
        errors = np.random.randn(200)
        result = ModelComparison.dieboldMariano(errors, errors)
        assert result['pValue'] > 0.05, \
            f"Same errors should yield p > 0.05, got {result['pValue']:.4f}"

    def testDieboldMarianoDifferentModels(self):
        np.random.seed(42)
        errors1 = np.random.randn(200) * 0.1
        errors2 = np.random.randn(200) * 5.0
        result = ModelComparison.dieboldMariano(errors1, errors2)
        assert result['pValue'] < 0.05, \
            f"Clearly different models should yield p < 0.05, got {result['pValue']:.4f}"

    def testModelComparisonTable(self):
        np.random.seed(42)
        modelErrors = {
            'modelA': np.random.randn(100) * 1.0,
            'modelB': np.random.randn(100) * 2.0,
            'modelC': np.random.randn(100) * 5.0,
        }
        result = ModelComparison.modelComparisonTable(modelErrors)
        assert 'rankings' in result
        assert 'metrics' in result
        assert 'pairwiseTests' in result
        assert len(result['rankings']) == 3
        for name in ['modelA', 'modelB', 'modelC']:
            assert name in result['metrics']
            assert 'rmse' in result['metrics'][name]
            assert 'mae' in result['metrics'][name]
        assert result['rankings'][0] == 'modelA', \
            f"modelA should rank first (smallest errors), got '{result['rankings'][0]}'"

    def testForecastEncompassing(self):
        np.random.seed(42)
        errors1 = np.random.randn(200) * 0.5
        errors2 = errors1 + np.random.randn(200) * 0.01
        result = ModelComparison.forecastEncompassingTest(errors1, errors2)
        assert 'statistic' in result
        assert 'pValue' in result
        assert 'conclusion' in result
        assert isinstance(result['pValue'], float)
        assert 0.0 <= result['pValue'] <= 1.0
