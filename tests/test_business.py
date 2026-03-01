"""Business Intelligence Module Tests"""
import os
import tempfile

import numpy as np
import pytest

from vectrix.business import (
    AnomalyDetector,
    Backtester,
    BusinessMetrics,
    ForecastExplainer,
    HTMLReportGenerator,
    ReportGenerator,
    WhatIfAnalyzer,
)
from vectrix.business.anomaly import AnomalyResult
from vectrix.business.backtest import BacktestFold, BacktestResult
from vectrix.business.whatif import Scenario, ScenarioResult


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def seasonalData(seed):
    rng = np.random.default_rng(seed)
    n = 200
    t = np.arange(n, dtype=np.float64)
    trend = 100.0 + 0.3 * t
    seasonal = 15.0 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 2.0, n)
    return trend + seasonal + noise


@pytest.fixture
def dataWithAnomalies(seed):
    rng = np.random.default_rng(seed)
    n = 200
    y = rng.normal(100, 5, n)
    y[30] = 200.0
    y[80] = 10.0
    y[150] = 250.0
    return y


@pytest.fixture
def predictions(seasonalData):
    rng = np.random.default_rng(99)
    lastVal = seasonalData[-1]
    steps = 14
    pred = lastVal + np.linspace(0, 5, steps) + rng.normal(0, 1, steps)
    return pred


@pytest.fixture
def actualVsPred(seed):
    rng = np.random.default_rng(seed)
    n = 50
    actual = 100.0 + rng.normal(0, 10, n)
    predicted = actual + rng.normal(0, 3, n)
    return actual, predicted


class TestAnomalyDetector:

    def test_zscoreDetectFindsOutliers(self, dataWithAnomalies):
        detector = AnomalyDetector()
        result = detector.detect(dataWithAnomalies, method='zscore', threshold=3.0)
        assert isinstance(result, AnomalyResult)
        assert result.method == 'zscore'
        assert result.nAnomalies > 0
        assert 30 in result.indices

    def test_iqrDetectFindsOutliers(self, dataWithAnomalies):
        detector = AnomalyDetector()
        result = detector.detect(dataWithAnomalies, method='iqr', threshold=1.5)
        assert isinstance(result, AnomalyResult)
        assert result.method == 'iqr'
        assert result.nAnomalies > 0
        assert len(result.details) == result.nAnomalies

    def test_seasonalDetect(self, seasonalData):
        detector = AnomalyDetector()
        result = detector.detect(seasonalData, method='seasonal', threshold=3.0, period=7)
        assert isinstance(result, AnomalyResult)
        assert result.method == 'seasonal'
        assert result.anomalyRatio >= 0.0
        assert result.anomalyRatio <= 1.0

    def test_rollingDetect(self, dataWithAnomalies):
        detector = AnomalyDetector()
        result = detector.detect(dataWithAnomalies, method='rolling', threshold=3.0)
        assert isinstance(result, AnomalyResult)
        assert result.method == 'rolling'
        assert len(result.scores) == len(dataWithAnomalies)

    def test_autoDetectConsensus(self, dataWithAnomalies):
        detector = AnomalyDetector()
        result = detector.detect(dataWithAnomalies, method='auto', threshold=3.0, period=1)
        assert isinstance(result, AnomalyResult)
        assert 'auto' in result.method
        assert result.nAnomalies == len(result.indices)
        for detail in result.details:
            assert 'index' in detail
            assert 'value' in detail
            assert 'score' in detail

    def test_constantDataNoAnomalies(self):
        detector = AnomalyDetector()
        y = np.ones(100) * 50.0
        result = detector.detect(y, method='zscore', threshold=3.0)
        assert result.nAnomalies == 0

    def test_anomalyRatioCalculation(self, dataWithAnomalies):
        detector = AnomalyDetector()
        result = detector.detect(dataWithAnomalies, method='zscore', threshold=3.0)
        expectedRatio = result.nAnomalies / len(dataWithAnomalies)
        assert abs(result.anomalyRatio - expectedRatio) < 1e-10


class TestBusinessMetrics:

    def test_calculateReturnsAllKeys(self, actualVsPred):
        actual, predicted = actualVsPred
        metrics = BusinessMetrics()
        result = metrics.calculate(actual, predicted)
        assert isinstance(result, dict)
        expectedKeys = [
            'bias', 'biasPercent', 'trackingSignal', 'wape',
            'mase', 'overForecastRatio', 'underForecastRatio',
            'fillRateImpact', 'forecastAccuracy',
        ]
        for key in expectedKeys:
            assert key in result, f"'{key}' missing from result"

    def test_calculateWithValues(self, actualVsPred):
        actual, predicted = actualVsPred
        values = np.abs(actual)
        metrics = BusinessMetrics()
        result = metrics.calculate(actual, predicted, values=values)
        assert 'valueWeightedAccuracy' in result

    def test_biasPositiveOverForecast(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([15.0, 25.0, 35.0])
        assert BusinessMetrics.bias(actual, predicted) == 5.0

    def test_biasNegativeUnderForecast(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([5.0, 15.0, 25.0])
        assert BusinessMetrics.bias(actual, predicted) == -5.0

    def test_wapeCalculation(self):
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([110.0, 190.0, 310.0])
        totalActual = 600.0
        totalAbsError = 10.0 + 10.0 + 10.0
        expectedWape = totalAbsError / totalActual * 100
        assert abs(BusinessMetrics.wape(actual, predicted) - expectedWape) < 1e-6

    def test_overUnderForecastRatio(self):
        actual = np.array([10.0, 20.0, 30.0, 40.0])
        predicted = np.array([15.0, 15.0, 35.0, 35.0])
        assert BusinessMetrics.overForecastRatio(actual, predicted) == 0.5
        assert BusinessMetrics.underForecastRatio(actual, predicted) == 0.5

    def test_fillRateImpact(self):
        actual = np.array([100.0, 100.0, 100.0])
        predicted = np.array([80.0, 100.0, 120.0])
        result = BusinessMetrics.fillRateImpact(actual, predicted)
        assert 'potentialStockout' in result
        assert 'fillRate' in result
        assert result['fillRate'] >= 0.0
        assert result['fillRate'] <= 100.0

    def test_forecastAccuracyBounded(self, actualVsPred):
        actual, predicted = actualVsPred
        metrics = BusinessMetrics()
        result = metrics.calculate(actual, predicted)
        assert result['forecastAccuracy'] >= 0.0

    def test_maseCalculation(self):
        actual = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        predicted = np.array([11.0, 13.0, 15.0, 17.0, 19.0, 21.0])
        mase = BusinessMetrics.mase(actual, predicted, seasonalPeriod=1)
        assert mase > 0.0
        assert np.isfinite(mase)


class TestBacktester:

    def test_runExpandingWindow(self, seasonalData):
        bt = Backtester(nFolds=3, horizon=10, strategy='expanding', minTrainSize=50)

        class SimpleModel:
            def fit(self, y):
                self.lastVal = y[-1]
            def predict(self, steps):
                pred = np.full(steps, self.lastVal)
                return pred, pred - 5, pred + 5

        result = bt.run(seasonalData, lambda: SimpleModel())
        assert isinstance(result, BacktestResult)
        assert result.nFolds > 0
        assert result.avgMAPE >= 0.0
        assert len(result.folds) > 0

    def test_runSlidingWindow(self, seasonalData):
        bt = Backtester(nFolds=3, horizon=10, strategy='sliding', minTrainSize=50)

        class SimpleModel:
            def fit(self, y):
                self.lastVal = y[-1]
            def predict(self, steps):
                pred = np.full(steps, self.lastVal)
                return pred, pred - 5, pred + 5

        result = bt.run(seasonalData, lambda: SimpleModel())
        assert isinstance(result, BacktestResult)
        assert result.nFolds > 0

    def test_runTooShortData(self):
        bt = Backtester(nFolds=3, horizon=30, minTrainSize=50)
        y = np.arange(20, dtype=np.float64)

        class SimpleModel:
            def fit(self, y): pass
            def predict(self, steps):
                return np.zeros(steps), np.zeros(steps), np.zeros(steps)

        result = bt.run(y, lambda: SimpleModel())
        assert result.nFolds == 0

    def test_foldMetricsPresent(self, seasonalData):
        bt = Backtester(nFolds=2, horizon=5, minTrainSize=50)

        class SimpleModel:
            def fit(self, y):
                self.lastVal = y[-1]
            def predict(self, steps):
                pred = np.full(steps, self.lastVal)
                return pred, pred - 5, pred + 5

        result = bt.run(seasonalData, lambda: SimpleModel())
        for fold in result.folds:
            assert isinstance(fold, BacktestFold)
            assert fold.trainSize > 0
            assert fold.testSize > 0

    def test_summaryText(self, seasonalData):
        bt = Backtester(nFolds=2, horizon=5, minTrainSize=50)

        class SimpleModel:
            def fit(self, y):
                self.lastVal = y[-1]
            def predict(self, steps):
                pred = np.full(steps, self.lastVal)
                return pred, pred - 5, pred + 5

        result = bt.run(seasonalData, lambda: SimpleModel())
        summary = bt.summary(result)
        assert isinstance(summary, str)
        assert 'MAPE' in summary


class TestWhatIfAnalyzer:

    def test_trendChangeScenario(self, predictions, seasonalData):
        analyzer = WhatIfAnalyzer()
        scenarios = [{'name': 'optimistic', 'trend_change': 0.1}]
        results = analyzer.analyze(predictions, seasonalData, scenarios, period=7)
        assert len(results) == 1
        assert isinstance(results[0], ScenarioResult)
        assert results[0].name == 'optimistic'
        assert len(results[0].predictions) == len(predictions)
        assert not np.array_equal(results[0].predictions, predictions)

    def test_shockScenario(self, predictions, seasonalData):
        analyzer = WhatIfAnalyzer()
        scenarios = [{'name': 'shock', 'shock_at': 5, 'shock_magnitude': -0.2, 'shock_duration': 3}]
        results = analyzer.analyze(predictions, seasonalData, scenarios, period=7)
        assert len(results) == 1
        assert results[0].predictions[5] < predictions[5]

    def test_levelShiftScenario(self, predictions, seasonalData):
        analyzer = WhatIfAnalyzer()
        scenarios = [{'name': 'level_up', 'level_shift': 0.05}]
        results = analyzer.analyze(predictions, seasonalData, scenarios, period=7)
        assert np.all(results[0].predictions > predictions - 1e-10)

    def test_seasonalMultiplierScenario(self, predictions, seasonalData):
        analyzer = WhatIfAnalyzer()
        scenarios = [{'name': 'no_seasonal', 'seasonal_multiplier': 0.0}]
        results = analyzer.analyze(predictions, seasonalData, scenarios, period=7)
        assert len(results) == 1

    def test_multipleScenarios(self, predictions, seasonalData):
        analyzer = WhatIfAnalyzer()
        scenarios = [
            {'name': 'optimistic', 'trend_change': 0.1},
            {'name': 'pessimistic', 'trend_change': -0.1},
            {'name': 'shock', 'shock_at': 3, 'shock_magnitude': -0.3},
        ]
        results = analyzer.analyze(predictions, seasonalData, scenarios, period=7)
        assert len(results) == 3
        names = [r.name for r in results]
        assert 'optimistic' in names
        assert 'pessimistic' in names
        assert 'shock' in names

    def test_scenarioResultDifference(self, predictions, seasonalData):
        analyzer = WhatIfAnalyzer()
        scenarios = [{'name': 'shift', 'level_shift': 0.1}]
        results = analyzer.analyze(predictions, seasonalData, scenarios, period=7)
        diff = results[0].difference
        assert len(diff) == len(predictions)
        assert np.all(np.abs(diff) > 0)

    def test_compareSummary(self, predictions, seasonalData):
        analyzer = WhatIfAnalyzer()
        scenarios = [
            {'name': 'up', 'trend_change': 0.1},
            {'name': 'down', 'trend_change': -0.1},
        ]
        results = analyzer.analyze(predictions, seasonalData, scenarios, period=7)
        summary = analyzer.compareSummary(results)
        assert isinstance(summary, str)
        assert 'Scenario Comparison' in summary

    def test_compareSummaryEmpty(self):
        analyzer = WhatIfAnalyzer()
        summary = analyzer.compareSummary([])
        assert summary == "No scenarios"


class TestForecastExplainer:

    def test_explainReturnsAllKeys(self, seasonalData, predictions):
        explainer = ForecastExplainer()
        result = explainer.explain(seasonalData, predictions, period=7)
        assert isinstance(result, dict)
        expectedKeys = ['drivers', 'narrative', 'decomposition', 'confidence', 'summary']
        for key in expectedKeys:
            assert key in result, f"'{key}' missing from explanation"

    def test_driversStructure(self, seasonalData, predictions):
        explainer = ForecastExplainer()
        result = explainer.explain(seasonalData, predictions, period=7)
        drivers = result['drivers']
        assert isinstance(drivers, list)
        assert len(drivers) >= 2
        for d in drivers:
            assert 'name' in d
            assert 'contribution' in d
            assert d['contribution'] >= 0

    def test_decompositionComponents(self, seasonalData, predictions):
        explainer = ForecastExplainer()
        result = explainer.explain(seasonalData, predictions, period=7)
        decomp = result['decomposition']
        assert 'trend' in decomp
        assert 'seasonal' in decomp
        assert 'residual' in decomp
        assert len(decomp['trend']) == len(seasonalData)
        assert len(decomp['seasonal']) == len(seasonalData)
        assert len(decomp['residual']) == len(seasonalData)

    def test_confidenceLevels(self, seasonalData, predictions):
        explainer = ForecastExplainer()
        result = explainer.explain(seasonalData, predictions, period=7)
        conf = result['confidence']
        assert conf['level'] in ('high', 'medium', 'low')
        assert conf['score'] in (85, 65, 40)
        assert conf['dataVariability'] >= 0.0

    def test_narrativeIsString(self, seasonalData, predictions):
        explainer = ForecastExplainer()
        result = explainer.explain(seasonalData, predictions, period=7)
        assert isinstance(result['narrative'], str)
        assert len(result['narrative']) > 0

    def test_summaryContainsDrivers(self, seasonalData, predictions):
        explainer = ForecastExplainer()
        result = explainer.explain(seasonalData, predictions, period=7)
        summary = result['summary']
        assert 'Key drivers' in summary

    def test_explainEnLocale(self, seasonalData, predictions):
        explainer = ForecastExplainer()
        result = explainer.explain(seasonalData, predictions, period=7, locale='en')
        assert isinstance(result['narrative'], str)
        assert len(result['narrative']) > 0


class TestReportGenerator:

    def test_generateReturnsAllSections(self, seasonalData, predictions):
        gen = ReportGenerator(locale='ko')
        report = gen.generate(seasonalData, predictions, period=7)
        assert isinstance(report, dict)
        expectedKeys = [
            'generatedAt', 'version', 'locale', 'overview',
            'dataAnalysis', 'forecast', 'anomalies',
            'explanation', 'recommendations',
        ]
        for key in expectedKeys:
            assert key in report, f"'{key}' missing from report"

    def test_overviewStatistics(self, seasonalData, predictions):
        gen = ReportGenerator()
        report = gen.generate(seasonalData, predictions, period=7)
        overview = report['overview']
        assert overview['dataPoints'] == len(seasonalData)
        assert overview['forecastHorizon'] == len(predictions)
        stats = overview['statistics']
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats

    def test_anomaliesSection(self, seasonalData, predictions):
        gen = ReportGenerator()
        report = gen.generate(seasonalData, predictions, period=7)
        anomalies = report['anomalies']
        assert 'count' in anomalies
        assert 'ratio' in anomalies
        assert anomalies['count'] >= 0

    def test_forecastSection(self, seasonalData, predictions):
        gen = ReportGenerator()
        report = gen.generate(seasonalData, predictions, period=7)
        forecast = report['forecast']
        assert 'values' in forecast
        assert 'mean' in forecast
        assert 'direction' in forecast
        assert forecast['direction'] in ('up', 'down', 'stable')

    def test_forecastWithConfidenceIntervals(self, seasonalData, predictions):
        lower = predictions - 5.0
        upper = predictions + 5.0
        gen = ReportGenerator()
        report = gen.generate(seasonalData, predictions, lower95=lower, upper95=upper, period=7)
        forecast = report['forecast']
        assert 'lower95' in forecast
        assert 'upper95' in forecast
        assert 'avgWidth' in forecast

    def test_toTextFormat(self, seasonalData, predictions):
        gen = ReportGenerator()
        report = gen.generate(seasonalData, predictions, period=7)
        text = gen.toText(report)
        assert isinstance(text, str)
        assert 'Vectrix Forecast Report' in text
        assert 'Forecast direction' in text

    def test_recommendationsPresent(self, seasonalData, predictions):
        gen = ReportGenerator()
        report = gen.generate(seasonalData, predictions, period=7)
        recs = report['recommendations']
        assert isinstance(recs, list)
        assert len(recs) > 0


class TestHTMLReportGenerator:

    def test_generateCreatesFile(self, seasonalData, predictions):
        lower = predictions - 5.0
        upper = predictions + 5.0
        gen = HTMLReportGenerator()
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            tmpPath = tmp.name
        try:
            result = gen.generate(
                seasonalData, predictions, lower, upper,
                modelName='TestModel', outputPath=tmpPath,
            )
            assert result == tmpPath
            assert os.path.exists(tmpPath)
            with open(tmpPath, 'r', encoding='utf-8') as f:
                html = f.read()
            assert '<!DOCTYPE html>' in html
            assert 'TestModel' in html
            assert 'Vectrix Forecast Report' in html
        finally:
            if os.path.exists(tmpPath):
                os.unlink(tmpPath)

    def test_generateWithDates(self, seasonalData, predictions):
        lower = predictions - 5.0
        upper = predictions + 5.0
        dates = [f"2026-01-{i+1:02d}" for i in range(len(predictions))]
        gen = HTMLReportGenerator()
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
            tmpPath = tmp.name
        try:
            gen.generate(
                seasonalData, predictions, lower, upper,
                dates=dates, outputPath=tmpPath,
            )
            with open(tmpPath, 'r', encoding='utf-8') as f:
                html = f.read()
            assert '2026-01-01' in html
        finally:
            if os.path.exists(tmpPath):
                os.unlink(tmpPath)
