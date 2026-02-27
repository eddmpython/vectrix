import pytest
import numpy as np

from forecastx.engine.baselines import NaiveModel
from forecastx.engine.lotkaVolterra import LotkaVolterraEnsemble
from forecastx.engine.phaseTransition import PhaseTransitionForecaster
from forecastx.engine.adversarial import AdversarialStressTester, StressTestResult
from forecastx.engine.hawkes import HawkesIntermittentDemand
from forecastx.engine.entropic import EntropicConfidenceScorer, EntropyResult

np.random.seed(42)
STABLE_DATA = 100 + np.cumsum(np.random.randn(200) * 0.5)
SEASONAL_DATA = np.array([10 + 5*np.sin(2*np.pi*i/12) + np.random.randn()*0.5 for i in range(120)])
INTERMITTENT_DATA = np.zeros(200)
INTERMITTENT_DATA[np.random.choice(200, 30, replace=False)] = np.random.poisson(5, 30) + 1


def _makeNaiveFactory():
    return ("Naive", lambda: NaiveModel())


def _buildSimpleEnsembleModels():
    return [
        ("NaiveA", lambda: NaiveModel()),
        ("NaiveB", lambda: NaiveModel()),
        ("NaiveC", lambda: NaiveModel()),
    ]


class TestLotkaVolterraEnsemble:

    def test_fitAndPredictBasic(self):
        ensemble = LotkaVolterraEnsemble(models=_buildSimpleEnsembleModels())
        ensemble.fit(SEASONAL_DATA)
        predictions, lower95, upper95 = ensemble.predict(12)
        assert len(predictions) == 12
        assert len(lower95) == 12
        assert len(upper95) == 12
        assert all(np.isfinite(predictions))

    def test_getEcosystemStateKeys(self):
        ensemble = LotkaVolterraEnsemble(models=_buildSimpleEnsembleModels())
        ensemble.fit(STABLE_DATA)
        state = ensemble.getEcosystemState()
        expectedKeys = [
            "shannonDiversity", "dominantSpecies", "extinctSpecies",
            "symbiosisPairs", "competitionPairs", "speciesState", "weights",
            "competitionMatrix",
        ]
        for key in expectedKeys:
            assert key in state, f"Missing key: {key}"
        assert isinstance(state["dominantSpecies"], str)
        assert isinstance(state["extinctSpecies"], list)

    def test_weightsSumToOne(self):
        ensemble = LotkaVolterraEnsemble(models=_buildSimpleEnsembleModels())
        ensemble.fit(STABLE_DATA)
        state = ensemble.getEcosystemState()
        weightSum = float(np.sum(state["weights"]))
        assert abs(weightSum - 1.0) < 1e-6, f"Weights sum={weightSum}, expected ~1.0"

    def test_predictShapeMatchesSteps(self):
        ensemble = LotkaVolterraEnsemble(models=_buildSimpleEnsembleModels())
        ensemble.fit(SEASONAL_DATA)
        for horizon in [1, 6, 24]:
            predictions, lower95, upper95 = ensemble.predict(horizon)
            assert predictions.shape == (horizon,)
            assert lower95.shape == (horizon,)
            assert upper95.shape == (horizon,)


class TestPhaseTransitionForecaster:

    def test_stableDataNoCritical(self):
        baseModel = NaiveModel()
        forecaster = PhaseTransitionForecaster(baseModel=baseModel)
        forecaster.fit(STABLE_DATA)
        indicators = forecaster.getTransitionIndicators()
        assert indicators["criticalState"] is False

    def test_regimeShiftDetection(self):
        rng = np.random.default_rng(99)
        stablePart = 50.0 + rng.normal(0, 0.3, 150)
        shiftPart = 50.0 + np.cumsum(rng.normal(0.5, 2.0, 50))
        regimeData = np.concatenate([stablePart, shiftPart])
        baseModel = NaiveModel()
        forecaster = PhaseTransitionForecaster(baseModel=baseModel)
        forecaster.fit(regimeData)
        indicators = forecaster.getTransitionIndicators()
        assert indicators["compositeScore"] > 0.0

    def test_getTransitionIndicatorsKeys(self):
        baseModel = NaiveModel()
        forecaster = PhaseTransitionForecaster(baseModel=baseModel)
        forecaster.fit(SEASONAL_DATA)
        indicators = forecaster.getTransitionIndicators()
        expectedKeys = [
            "ar1Series", "varianceSeries", "skewnessSeries", "windowCenters",
            "kendallTauAr1", "kendallTauVariance", "kendallTauSkewness",
            "compositeScore", "criticalState", "threshold", "interpretation",
        ]
        for key in expectedKeys:
            assert key in indicators, f"Missing key: {key}"

    def test_predictShapeAndConfidenceInterval(self):
        baseModel = NaiveModel()
        forecaster = PhaseTransitionForecaster(baseModel=baseModel)
        forecaster.fit(STABLE_DATA)
        horizon = 10
        predictions, lower95, upper95 = forecaster.predict(horizon)
        assert predictions.shape == (horizon,)
        assert lower95.shape == (horizon,)
        assert upper95.shape == (horizon,)
        assert np.all(upper95 >= lower95)


class TestAdversarialStressTester:

    def test_analyzeBasic(self):
        modelFactory = lambda: NaiveModel()
        tester = AdversarialStressTester(model=modelFactory, nPerturbations=10)
        result = tester.analyze(STABLE_DATA, steps=5)
        assert isinstance(result, StressTestResult)
        assert result.fragilityScore >= 0.0
        assert result.resilience >= 0.0

    def test_fragilityScoreRange(self):
        modelFactory = lambda: NaiveModel()
        tester = AdversarialStressTester(model=modelFactory, nPerturbations=10)
        result = tester.analyze(SEASONAL_DATA, steps=5)
        assert 0.0 <= result.fragilityScore <= 1.0, \
            f"fragilityScore={result.fragilityScore} out of [0,1]"
        assert abs(result.fragilityScore + result.resilience - 1.0) < 1e-9

    def test_vulnerabilityProfileKeys(self):
        modelFactory = lambda: NaiveModel()
        tester = AdversarialStressTester(model=modelFactory, nPerturbations=10)
        result = tester.analyze(STABLE_DATA, steps=5)
        profile = result.vulnerabilityProfile()
        expectedOperators = [
            "levelShift", "volatilityBurst", "trendBreak",
            "seasonalCorruption", "tailInjection",
        ]
        for op in expectedOperators:
            assert op in profile, f"Missing operator: {op}"
            assert "sensitivity" in profile[op]
            assert "medianMAPE" in profile[op]

    def test_summaryContainsGrade(self):
        modelFactory = lambda: NaiveModel()
        tester = AdversarialStressTester(model=modelFactory, nPerturbations=10)
        result = tester.analyze(STABLE_DATA, steps=5)
        summaryDict = result.summary()
        assert "grade" in summaryDict
        assert summaryDict["grade"] in ["Robust", "Moderate", "Fragile", "Critical"]
        assert "recommendations" in summaryDict
        assert isinstance(summaryDict["recommendations"], list)


class TestHawkesIntermittentDemand:

    def test_fitAndPredictIntermittent(self):
        model = HawkesIntermittentDemand()
        model.fit(INTERMITTENT_DATA)
        predictions, lower95, upper95 = model.predict(10)
        assert predictions.shape == (10,)
        assert lower95.shape == (10,)
        assert upper95.shape == (10,)
        assert np.all(lower95 >= 0.0)

    def test_getIntensityProfileKeys(self):
        model = HawkesIntermittentDemand()
        model.fit(INTERMITTENT_DATA)
        profile = model.getIntensityProfile()
        expectedKeys = [
            "burstiness", "selfExcitationRatio", "meanInterArrival",
            "baseIntensity", "excitationDecay", "excitationMagnitude",
        ]
        for key in expectedKeys:
            assert key in profile, f"Missing key: {key}"

    def test_allPositiveData(self):
        positiveData = np.random.poisson(3, 100).astype(float) + 1.0
        model = HawkesIntermittentDemand()
        model.fit(positiveData)
        predictions, lower95, upper95 = model.predict(5)
        assert len(predictions) == 5
        assert all(np.isfinite(predictions))

    def test_allZeroDemandFitSucceeds(self):
        zeroData = np.zeros(100)
        model = HawkesIntermittentDemand()
        fittedModel = model.fit(zeroData)
        assert fittedModel is model
        profile = model.getIntensityProfile()
        assert profile["baseIntensity"] > 0


class TestEntropicConfidenceScorer:

    def test_analyzeBasicAndResultFields(self):
        modelFactory = lambda: NaiveModel()
        scorer = EntropicConfidenceScorer(model=modelFactory, nBootstrap=30, nBins=15)
        result = scorer.analyze(STABLE_DATA, steps=6)
        assert isinstance(result, EntropyResult)
        assert result.stepEntropy is not None
        assert result.normalizedEntropy is not None
        assert result.confidenceScore is not None
        assert result.nModes is not None
        assert len(result.stepEntropy) == 6
        assert len(result.confidenceScore) == 6

    def test_gradeInValidSet(self):
        modelFactory = lambda: NaiveModel()
        scorer = EntropicConfidenceScorer(model=modelFactory, nBootstrap=30, nBins=15)
        result = scorer.analyze(STABLE_DATA, steps=6)
        validGrades = {"A", "B", "C", "D", "F"}
        assert result.grade() in validGrades, f"grade()={result.grade()} not in {validGrades}"

    def test_extractScenariosReturnsListOfDicts(self):
        modelFactory = lambda: NaiveModel()
        scorer = EntropicConfidenceScorer(model=modelFactory, nBootstrap=30, nBins=15)
        scorer.analyze(STABLE_DATA, steps=6)
        scenarios = scorer.extractScenarios(nScenarios=3)
        assert isinstance(scenarios, list)
        for scenario in scenarios:
            assert "probability" in scenario
            assert "path" in scenario
            assert "description" in scenario

    def test_confidenceScoreRange(self):
        modelFactory = lambda: NaiveModel()
        scorer = EntropicConfidenceScorer(model=modelFactory, nBootstrap=30, nBins=15)
        result = scorer.analyze(SEASONAL_DATA, steps=6)
        assert 0.0 <= result.overallConfidence <= 1.0
        assert np.all(result.confidenceScore >= 0.0)
        assert np.all(result.confidenceScore <= 1.0)
