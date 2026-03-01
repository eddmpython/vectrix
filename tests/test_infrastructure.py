"""Infrastructure module tests (FlatDefense, Hierarchy, Batch, Persistence, TSFrame, AutoAnalyzer)"""
import pytest
import numpy as np
import pandas as pd

from vectrix.flat_defense import FlatPredictionDetector, FlatPredictionCorrector, FlatRiskDiagnostic
from vectrix.hierarchy import BottomUp, TopDown, MinTrace
from vectrix.batch import BatchForecastResult, batchForecast
from vectrix.persistence import ModelPersistence
from vectrix.tsframe import TSFrame
from vectrix.analyzer.autoAnalyzer import AutoAnalyzer
from vectrix.types import FlatPredictionInfo, FlatPredictionType, RiskLevel, DataCharacteristics


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def seasonalData(rng):
    n = 200
    t = np.arange(n, dtype=np.float64)
    trend = 100.0 + 0.5 * t
    seasonal = 10.0 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 2, n)
    return trend + seasonal + noise


@pytest.fixture
def flatPredictions():
    return np.full(20, 150.0)


@pytest.fixture
def diagonalPredictions():
    return np.linspace(100, 200, 20)


@pytest.fixture
def normalPredictions(rng):
    t = np.arange(20, dtype=np.float64)
    return 100 + 0.5 * t + 8.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, 20)


@pytest.fixture
def summingMatrix():
    return np.array([
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def tsDf(rng):
    n = 200
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    t = np.arange(n, dtype=np.float64)
    values = 100.0 + 0.3 * t + 8.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 2, n)
    return pd.DataFrame({'date': dates, 'value': values})


class TestFlatPredictionDetector:

    def test_detectHorizontalFlat(self, flatPredictions, seasonalData):
        detector = FlatPredictionDetector()
        info = detector.detect(flatPredictions, seasonalData)
        assert info.isFlat is True
        assert info.flatType == FlatPredictionType.HORIZONTAL

    def test_detectNonFlat(self, normalPredictions, seasonalData):
        detector = FlatPredictionDetector()
        info = detector.detect(normalPredictions, seasonalData)
        assert info.isFlat is False
        assert info.flatType == FlatPredictionType.NONE

    def test_detectDiagonalFlat(self, diagonalPredictions, seasonalData):
        detector = FlatPredictionDetector(diagonalThreshold=1e-8)
        info = detector.detect(diagonalPredictions, seasonalData)
        assert info.isFlat is True
        assert info.flatType == FlatPredictionType.DIAGONAL

    def test_detectShortPredictions(self, seasonalData):
        detector = FlatPredictionDetector()
        info = detector.detect(np.array([1.0, 2.0]), seasonalData)
        assert info.isFlat is False
        assert 'too short' in info.message

    def test_detectMultiple(self, flatPredictions, normalPredictions, seasonalData):
        detector = FlatPredictionDetector()
        modelPreds = {
            'flat_model': flatPredictions,
            'good_model': normalPredictions,
        }
        results = detector.detectMultiple(modelPreds, seasonalData)
        assert len(results) == 2
        assert results['flat_model'].isFlat is True
        assert results['good_model'].isFlat is False

    def test_getFlatModels(self, flatPredictions, normalPredictions, seasonalData):
        detector = FlatPredictionDetector()
        modelPreds = {
            'flat_model': flatPredictions,
            'good_model': normalPredictions,
        }
        results = detector.detectMultiple(modelPreds, seasonalData)
        flatModels = detector.getFlatModels(results)
        validModels = detector.getValidModels(results)
        assert 'flat_model' in flatModels
        assert 'good_model' in validModels
        assert 'good_model' not in flatModels

    def test_detectMeanReversion(self, seasonalData):
        detector = FlatPredictionDetector()
        preds = np.concatenate([
            np.sin(np.linspace(0, 4 * np.pi, 10)) * 20 + 100,
            np.full(10, 100.0)
        ])
        info = detector.detect(preds, seasonalData)
        assert info.isFlat is True
        assert info.flatType == FlatPredictionType.MEAN_REVERSION


class TestFlatPredictionCorrector:

    def test_correctHorizontalFlat(self, flatPredictions, seasonalData):
        flatInfo = FlatPredictionInfo(
            isFlat=True,
            flatType=FlatPredictionType.HORIZONTAL,
            predictionStd=0.0,
            originalStd=float(np.std(seasonalData)),
        )
        corrector = FlatPredictionCorrector()
        corrected, updatedInfo = corrector.correct(flatPredictions, seasonalData, flatInfo, period=7)
        assert updatedInfo.correctionApplied is True
        assert np.std(corrected) > np.std(flatPredictions)

    def test_correctNonFlat(self, normalPredictions, seasonalData):
        nonFlatInfo = FlatPredictionInfo(isFlat=False, flatType=FlatPredictionType.NONE)
        corrector = FlatPredictionCorrector()
        corrected, updatedInfo = corrector.correct(normalPredictions, seasonalData, nonFlatInfo)
        np.testing.assert_array_equal(corrected, normalPredictions)

    def test_correctDiagonalFlat(self, diagonalPredictions, seasonalData):
        flatInfo = FlatPredictionInfo(
            isFlat=True,
            flatType=FlatPredictionType.DIAGONAL,
            predictionStd=float(np.std(diagonalPredictions)),
            originalStd=float(np.std(seasonalData)),
        )
        corrector = FlatPredictionCorrector()
        corrected, updatedInfo = corrector.correct(diagonalPredictions, seasonalData, flatInfo, period=7)
        assert updatedInfo.correctionApplied is True
        assert 'trend_plus_seasonal' in updatedInfo.correctionMethod

    def test_correctMeanReversion(self, seasonalData):
        preds = np.concatenate([
            np.sin(np.linspace(0, 4 * np.pi, 10)) * 20 + 100,
            np.full(10, 100.0)
        ])
        flatInfo = FlatPredictionInfo(
            isFlat=True,
            flatType=FlatPredictionType.MEAN_REVERSION,
            predictionStd=float(np.std(preds)),
            originalStd=float(np.std(seasonalData)),
        )
        corrector = FlatPredictionCorrector()
        corrected, updatedInfo = corrector.correct(preds, seasonalData, flatInfo, period=7)
        assert updatedInfo.correctionApplied is True


class TestFlatRiskDiagnostic:

    def test_diagnoseHighRisk(self):
        diagnostic = FlatRiskDiagnostic(period=7)
        values = np.full(50, 100.0) + np.random.default_rng(42).normal(0, 0.01, 50)
        result = diagnostic.diagnose(values)
        assert result.riskScore > 0.3
        assert result.riskLevel in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        assert len(result.warnings) > 0

    def test_diagnoseLowRisk(self, seasonalData):
        diagnostic = FlatRiskDiagnostic(period=7)
        result = diagnostic.diagnose(seasonalData)
        assert result.riskLevel in (RiskLevel.LOW, RiskLevel.MEDIUM)
        assert isinstance(result.recommendedModels, list)
        assert len(result.recommendedModels) > 0

    def test_diagnoseVeryShortData(self):
        diagnostic = FlatRiskDiagnostic()
        result = diagnostic.diagnose(np.array([1.0, 2.0, 3.0]))
        assert result.riskLevel == RiskLevel.CRITICAL
        assert result.riskScore == 1.0

    def test_diagnoseReturnsRiskFactors(self, seasonalData):
        diagnostic = FlatRiskDiagnostic(period=7)
        result = diagnostic.diagnose(seasonalData)
        expectedFactors = ['lowVariance', 'weakSeasonality', 'noTrend', 'shortData', 'highNoise', 'flatRecent']
        for factor in expectedFactors:
            assert factor in result.riskFactors


class TestBottomUp:

    def test_reconcileBasic(self, summingMatrix):
        bottomForecasts = np.array([
            [10.0, 12.0, 14.0],
            [20.0, 22.0, 24.0],
            [30.0, 32.0, 34.0],
        ])
        bu = BottomUp()
        reconciled = bu.reconcile(bottomForecasts, summingMatrix)
        assert reconciled.shape == (4, 3)
        np.testing.assert_array_almost_equal(reconciled[0], [60.0, 66.0, 72.0])
        np.testing.assert_array_almost_equal(reconciled[1], [10.0, 12.0, 14.0])
        np.testing.assert_array_almost_equal(reconciled[2], [20.0, 22.0, 24.0])
        np.testing.assert_array_almost_equal(reconciled[3], [30.0, 32.0, 34.0])


class TestTopDown:

    def test_reconcileBasic(self, summingMatrix):
        topForecast = np.array([100.0, 120.0, 140.0])
        proportions = np.array([0.2, 0.3, 0.5])
        td = TopDown()
        reconciled = td.reconcile(topForecast, proportions, summingMatrix)
        assert reconciled.shape == (4, 3)
        np.testing.assert_array_almost_equal(reconciled[0], topForecast)
        np.testing.assert_array_almost_equal(reconciled[1], topForecast * 0.2)

    def test_computeProportions(self):
        historicalBottom = np.array([
            [10, 20, 30],
            [40, 50, 60],
        ])
        props = TopDown.computeProportions(historicalBottom)
        assert len(props) == 2
        np.testing.assert_almost_equal(np.sum(props), 1.0)
        np.testing.assert_almost_equal(props[0], 60 / 210)
        np.testing.assert_almost_equal(props[1], 150 / 210)


class TestMinTrace:

    def test_reconcileOLS(self, summingMatrix):
        forecasts = np.array([
            [65.0, 70.0],
            [12.0, 15.0],
            [22.0, 25.0],
            [33.0, 35.0],
        ])
        mt = MinTrace(method='ols')
        reconciled = mt.reconcile(forecasts, summingMatrix)
        assert reconciled.shape == (4, 2)
        np.testing.assert_array_almost_equal(
            reconciled[0],
            reconciled[1] + reconciled[2] + reconciled[3]
        )

    def test_reconcileWLS(self, summingMatrix):
        rng = np.random.default_rng(42)
        forecasts = np.array([
            [65.0, 70.0],
            [12.0, 15.0],
            [22.0, 25.0],
            [33.0, 35.0],
        ])
        residuals = rng.normal(0, 1, (4, 50))
        mt = MinTrace(method='wls')
        reconciled = mt.reconcile(forecasts, summingMatrix, residuals=residuals)
        assert reconciled.shape == (4, 2)
        np.testing.assert_array_almost_equal(
            reconciled[0],
            reconciled[1] + reconciled[2] + reconciled[3]
        )

    def test_buildSummingMatrix(self):
        structure = {
            'total': ['A', 'B'],
            'A': ['A1', 'A2'],
            'B': ['B1', 'B2'],
        }
        S = MinTrace.buildSummingMatrix(structure)
        assert S.shape[1] == 4
        assert S.shape[0] == 7
        colSum = S.sum(axis=1)
        totalRow = None
        for i in range(S.shape[0]):
            if colSum[i] == 4:
                totalRow = i
                break
        assert totalRow is not None


class TestBatchForecastResult:

    def test_batchResultCounts(self):
        results = {'s1': 'mock1', 's2': 'mock2'}
        failures = {'s3': 'error'}
        batch = BatchForecastResult(results, failures)
        assert batch.successCount == 2
        assert batch.failureCount == 1
        assert batch.totalCount == 3

    def test_batchResultRepr(self):
        batch = BatchForecastResult({'a': 1}, {})
        reprStr = repr(batch)
        assert 'BatchForecastResult' in reprStr
        assert 'success=1' in reprStr

    def test_batchResultSummary(self):
        batch = BatchForecastResult({'a': 1, 'b': 2}, {'c': 'err'})
        summaryStr = str(batch)
        assert '2/3' in summaryStr


class TestModelPersistence:

    def test_saveAndLoad(self, tmp_path):
        from vectrix.vectrix import Vectrix
        fx = Vectrix(verbose=False)

        filePath = str(tmp_path / 'model.fxm')
        ModelPersistence.save(fx, filePath, metadata={'note': 'test'})

        loaded = ModelPersistence.load(filePath)
        assert hasattr(loaded, '_loadedMeta')
        assert loaded._loadedMeta['metadata']['note'] == 'test'
        assert loaded._loadedMeta['formatVersion'] == '1.0'

    def test_info(self, tmp_path):
        from vectrix.vectrix import Vectrix
        fx = Vectrix(verbose=False)

        filePath = str(tmp_path / 'model.fxm')
        ModelPersistence.save(fx, filePath, metadata={'author': 'testuser'})

        infoStr = ModelPersistence.info(filePath)
        assert 'Vectrix Model File' in infoStr
        assert 'testuser' in infoStr

    def test_loadInvalidFile(self, tmp_path):
        badPath = str(tmp_path / 'bad.fxm')
        with open(badPath, 'wb') as f:
            f.write(b'BAD_DATA')
        with pytest.raises(ValueError, match='Not a valid'):
            ModelPersistence.load(badPath)


class TestTSFrame:

    def test_createFromPandas(self, tsDf):
        ts = TSFrame.fromPandas(tsDf, dateCol='date', valueCol='value')
        assert len(ts) == 200
        assert ts.freq == 'D'
        assert ts.period == 7

    def test_createFromArrays(self):
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        values = np.arange(50, dtype=np.float64)
        ts = TSFrame.fromArrays(dates.values, values)
        assert len(ts) == 50
        assert ts.valueCol == 'value'

    def test_generate(self):
        ts = TSFrame.generate(n=100, freq='D', trend=0.2, seasonalAmplitude=5.0, seed=99)
        assert len(ts) == 100
        assert ts.freq == 'D'
        assert ts.period == 7
        vals = ts.values
        assert len(vals) == 100
        assert not np.any(np.isnan(vals))

    def test_info(self, tsDf):
        ts = TSFrame.fromPandas(tsDf, dateCol='date', valueCol='value')
        info = ts.info()
        assert info['length'] == 200
        assert info['frequency'] == 'D'
        assert 'mean' in info
        assert 'std' in info
        assert 'cv' in info

    def test_resample(self, tsDf):
        ts = TSFrame.fromPandas(tsDf, dateCol='date', valueCol='value')
        weekly = ts.resample('W', agg='mean')
        assert len(weekly) < len(ts)
        assert len(weekly) > 0

    def test_fillMissing(self, tsDf):
        dfCopy = tsDf.copy()
        dfCopy.loc[5, 'value'] = np.nan
        dfCopy.loc[10, 'value'] = np.nan
        dfCopy.loc[15, 'value'] = np.nan
        ts = TSFrame.fromPandas(dfCopy, dateCol='date', valueCol='value')
        filled = ts.fillMissing(method='interpolate')
        assert filled.df['value'].isna().sum() == 0

    def test_fillMissingFfill(self, tsDf):
        dfCopy = tsDf.copy()
        dfCopy.loc[3, 'value'] = np.nan
        ts = TSFrame.fromPandas(dfCopy, dateCol='date', valueCol='value')
        filled = ts.fillMissing(method='ffill')
        assert filled.df['value'].isna().sum() == 0

    def test_split(self, tsDf):
        ts = TSFrame.fromPandas(tsDf, dateCol='date', valueCol='value')
        train, test = ts.split(ratio=0.8)
        assert len(train) == 160
        assert len(test) == 40

    def test_diff(self, tsDf):
        ts = TSFrame.fromPandas(tsDf, dateCol='date', valueCol='value')
        diffed = ts.diff(periods=1)
        assert len(diffed) == len(ts) - 1

    def test_rollingMean(self, tsDf):
        ts = TSFrame.fromPandas(tsDf, dateCol='date', valueCol='value')
        rolled = ts.rollingMean(window=7)
        assert len(rolled) == len(ts)

    def test_repr(self, tsDf):
        ts = TSFrame.fromPandas(tsDf, dateCol='date', valueCol='value')
        reprStr = repr(ts)
        assert 'TSFrame' in reprStr
        assert 'n=200' in reprStr


class TestAutoAnalyzer:

    def test_analyzeBasic(self, tsDf):
        analyzer = AutoAnalyzer()
        result = analyzer.analyze(tsDf, dateCol='date', valueCol='value')
        assert isinstance(result, DataCharacteristics)
        assert result.length == 200
        assert result.period > 0
        assert 0 <= result.predictabilityScore <= 100

    def test_analyzeTrendData(self):
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        values = np.linspace(10, 100, n) + np.random.default_rng(42).normal(0, 1, n)
        df = pd.DataFrame({'date': dates, 'value': values})
        analyzer = AutoAnalyzer()
        result = analyzer.analyze(df, dateCol='date', valueCol='value')
        assert result.hasTrend == True
        assert result.trendDirection == 'up'
        assert result.trendStrength > 0.3

    def test_analyzeHighVolatility(self):
        rng = np.random.default_rng(42)
        n = 200
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        values = 100 + rng.normal(0, 50, n)
        df = pd.DataFrame({'date': dates, 'value': values})
        analyzer = AutoAnalyzer()
        result = analyzer.analyze(df, dateCol='date', valueCol='value')
        assert result.volatility > 0

    def test_quickAnalyze(self, seasonalData):
        analyzer = AutoAnalyzer()
        result = analyzer.quickAnalyze(seasonalData, period=7)
        assert isinstance(result, dict)
        assert 'hasSeasonality' in result
        assert 'hasTrend' in result
        assert 'volatility' in result
        assert result['length'] == len(seasonalData)

    def test_analyzeMissingValues(self):
        n = 100
        dates = pd.date_range('2023-01-01', periods=n, freq='D')
        values = np.arange(n, dtype=np.float64)
        values[10] = np.nan
        values[20] = np.nan
        df = pd.DataFrame({'date': dates, 'value': values})
        df = df.dropna().reset_index(drop=True)
        analyzer = AutoAnalyzer()
        result = analyzer.analyze(df, dateCol='date', valueCol='value')
        assert result.length == 98
