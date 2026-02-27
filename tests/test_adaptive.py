"""적응형 예측 모듈 테스트"""
import pytest
import numpy as np

from forecastx.adaptive.regime import RegimeDetector
from forecastx.adaptive.healing import SelfHealingForecast
from forecastx.adaptive.constraints import ConstraintAwareForecaster, Constraint
from forecastx.adaptive.dna import ForecastDNA


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def regime_data(seed):
    """레짐 전환이 있는 시계열: 안정 -> 상승 -> 하락"""
    rng = np.random.default_rng(seed)
    stable = 100.0 + rng.normal(0, 1, 100)
    growth = np.cumsum(rng.normal(0.5, 1, 100)) + stable[-1]
    decline = np.cumsum(rng.normal(-0.3, 1, 100)) + growth[-1]
    return np.concatenate([stable, growth, decline])


@pytest.fixture
def healing_data(seed):
    """자가치유 테스트용 데이터"""
    rng = np.random.default_rng(seed)
    n = 200
    historicalData = 100.0 + rng.normal(0, 5, n)
    steps = 30
    predictions = np.full(steps, np.mean(historicalData))
    lower95 = predictions - 10.0
    upper95 = predictions + 10.0
    return historicalData, predictions, lower95, upper95


class TestRegimeDetector:
    """HMM 레짐 감지 테스트"""

    def test_basic_detection(self, regime_data):
        """기본 레짐 감지: 결과 구조 확인"""
        detector = RegimeDetector(nRegimes=3)
        result = detector.detect(regime_data)
        assert len(result.states) == len(regime_data), \
            f"상태 시퀀스 길이({len(result.states)})가 데이터 길이({len(regime_data)})와 같아야 함"
        assert result.nRegimes >= 2, \
            f"2개 이상의 레짐이 감지되어야 함, 실제={result.nRegimes}"
        assert result.currentRegime != "", \
            "현재 레짐이 빈 문자열이면 안 됨"

    def test_regime_labels(self, regime_data):
        """레짐 레이블이 유효한 값"""
        detector = RegimeDetector(nRegimes=3)
        result = detector.detect(regime_data)
        validLabels = {'growth', 'decline', 'volatile', 'stable', 'crisis'}
        for label in result.labels:
            assert label in validLabels, \
                f"레이블 '{label}'이 유효한 레이블 목록에 없음: {validLabels}"
        assert len(result.labels) == len(regime_data), \
            f"레이블 길이({len(result.labels)})가 데이터 길이({len(regime_data)})와 같아야 함"

    def test_transition_matrix(self, regime_data):
        """전이행렬: 확률의 합 = 1, K x K"""
        detector = RegimeDetector(nRegimes=3)
        result = detector.detect(regime_data)
        K = result.nRegimes
        tm = result.transitionMatrix
        assert tm.shape == (K, K), \
            f"전이행렬 크기가 ({K}, {K})이어야 함, 실제={tm.shape}"
        rowSums = tm.sum(axis=1)
        for i, s in enumerate(rowSums):
            assert abs(s - 1.0) < 1e-6, \
                f"전이행렬 행 {i}의 합이 1.0이어야 함, 실제={s:.6f}"
        assert np.all(tm >= 0), "전이행렬의 모든 원소는 비음수여야 함"

    def test_minimum_data_requirement(self):
        """최소 데이터 요구: 10개 미만 시 에러"""
        detector = RegimeDetector(nRegimes=2)
        with pytest.raises(ValueError):
            detector.detect(np.array([1.0, 2.0, 3.0]))

    def test_regime_history(self, regime_data):
        """레짐 이력: 구간이 전체를 커버"""
        detector = RegimeDetector(nRegimes=3)
        result = detector.detect(regime_data)
        assert len(result.regimeHistory) >= 1, "최소 1개의 레짐 구간이 있어야 함"
        firstStart = result.regimeHistory[0][1]
        lastEnd = result.regimeHistory[-1][2]
        assert firstStart == 0, f"첫 구간 시작이 0이어야 함, 실제={firstStart}"
        assert lastEnd == len(regime_data) - 1, \
            f"마지막 구간 끝이 {len(regime_data) - 1}이어야 함, 실제={lastEnd}"


class TestSelfHealing:
    """자가치유 예측 테스트"""

    def test_observe_single(self, healing_data):
        """단일 관측 후 상태 반환"""
        historicalData, predictions, lower95, upper95 = healing_data
        healer = SelfHealingForecast(
            predictions, lower95, upper95, historicalData, period=7
        )
        status = healer.observe(np.array([predictions[0] + 3.0]))
        assert status.health in ('healthy', 'degrading', 'critical', 'healed'), \
            f"건강 상태가 유효한 값이어야 함, 실제='{status.health}'"
        assert 0 <= status.healthScore <= 100, \
            f"건강 점수가 0-100 범위여야 함, 실제={status.healthScore:.1f}"
        assert status.observedCount == 1, \
            f"1회 관측 후 observedCount == 1 기대, 실제={status.observedCount}"
        assert status.mape >= 0, \
            f"MAPE는 비음수여야 함, 실제={status.mape:.4f}"

    def test_drift_detection(self, healing_data):
        """체계적 편향 감지"""
        historicalData, predictions, lower95, upper95 = healing_data
        healer = SelfHealingForecast(
            predictions, lower95, upper95, historicalData,
            period=7, healingMode='adaptive'
        )
        biasedActuals = predictions[:15] + 20.0
        status = healer.observe(biasedActuals)
        assert status.driftDetected or status.healthScore < 80, \
            "큰 편향이 있으면 드리프트 감지 또는 건강점수 하락 기대"

    def test_healing_improvement(self, healing_data):
        """치유 후 보고서에 MAPE 존재"""
        historicalData, predictions, lower95, upper95 = healing_data
        healer = SelfHealingForecast(
            predictions, lower95, upper95, historicalData, period=7
        )
        nObs = 10
        actuals = predictions[:nObs] + np.random.default_rng(42).normal(0, 2, nObs)
        healer.observe(actuals)
        report = healer.getReport()
        assert report.originalMape >= 0, \
            f"originalMape는 비음수여야 함, 실제={report.originalMape:.4f}"
        assert report.totalObserved == nObs, \
            f"관측 수가 {nObs}이어야 함, 실제={report.totalObserved}"
        assert report.overallHealth in ('healthy', 'degrading', 'critical', 'healed'), \
            f"전체 건강상태가 유효해야 함, 실제='{report.overallHealth}'"

    def test_get_updated_forecast(self, healing_data):
        """교정된 예측 반환 확인"""
        historicalData, predictions, lower95, upper95 = healing_data
        healer = SelfHealingForecast(
            predictions, lower95, upper95, historicalData, period=7
        )
        actuals = predictions[:5] + 3.0
        healer.observe(actuals)
        preds, lo, hi = healer.getUpdatedForecast()
        assert len(preds) == len(predictions), \
            f"교정 예측 길이({len(preds)})가 원래 예측 길이({len(predictions)})와 같아야 함"
        for i in range(5):
            assert abs(preds[i] - actuals[i]) < 1e-10, \
                f"관측된 스텝 {i}은 실제값으로 대체되어야 함"
        assert np.all(lo <= preds), "하한은 예측값 이하여야 함"
        assert np.all(hi >= preds), "상한은 예측값 이상이어야 함"


class TestConstraintAware:
    """제약 조건 예측 테스트"""

    def test_non_negative(self, seed):
        """비음수 제약"""
        rng = np.random.default_rng(seed)
        predictions = rng.normal(0, 10, 30)
        lower95 = predictions - 15.0
        upper95 = predictions + 15.0
        caf = ConstraintAwareForecaster()
        result = caf.apply(
            predictions, lower95, upper95,
            constraints=[Constraint('non_negative', {})]
        )
        assert np.all(result.predictions >= 0), \
            "비음수 제약 후 모든 예측값 >= 0 기대"
        assert np.all(result.lower95 >= 0), \
            "비음수 제약 후 하한 >= 0 기대"
        assert result.violationsAfter == 0, \
            f"제약 적용 후 위반 수 0 기대, 실제={result.violationsAfter}"

    def test_range_constraint(self, seed):
        """범위 제약"""
        rng = np.random.default_rng(seed)
        predictions = rng.normal(500, 200, 30)
        lower95 = predictions - 100.0
        upper95 = predictions + 100.0
        caf = ConstraintAwareForecaster()
        result = caf.apply(
            predictions, lower95, upper95,
            constraints=[Constraint('range', {'min': 100, 'max': 800})]
        )
        assert np.all(result.predictions >= 100 - 1e-10), \
            "범위 제약 후 모든 예측값 >= 100 기대"
        assert np.all(result.predictions <= 800 + 1e-10), \
            "범위 제약 후 모든 예측값 <= 800 기대"
        assert result.violationsAfter == 0, \
            f"제약 적용 후 위반 수 0 기대, 실제={result.violationsAfter}"

    def test_monotone(self, seed):
        """단조 증가 제약 (PAVA) - smoothing=False로 정확한 단조 보장"""
        rng = np.random.default_rng(seed)
        predictions = np.array([10, 12, 11, 15, 13, 17, 14, 20], dtype=np.float64)
        lower95 = predictions - 2.0
        upper95 = predictions + 2.0
        caf = ConstraintAwareForecaster()
        result = caf.apply(
            predictions, lower95, upper95,
            constraints=[Constraint('monotone', {'direction': 'increasing'})],
            smoothing=False
        )
        diffs = np.diff(result.predictions)
        assert np.all(diffs >= -1e-10), \
            "단조 증가 제약 후 모든 연속 차이 >= 0 기대"
        assert result.violationsAfter == 0, \
            f"단조 제약 적용 후 위반 수 0 기대, 실제={result.violationsAfter}"

    def test_violations_count(self, seed):
        """제약 적용 전 위반 수 > 적용 후 위반 수"""
        predictions = np.array([-5, -3, 2, 7, -1, 10], dtype=np.float64)
        lower95 = predictions - 3.0
        upper95 = predictions + 3.0
        caf = ConstraintAwareForecaster()
        result = caf.apply(
            predictions, lower95, upper95,
            constraints=[Constraint('non_negative', {})]
        )
        assert result.violationsBefore > 0, \
            "음수 값이 있으므로 적용 전 위반 > 0 기대"
        assert result.violationsAfter <= result.violationsBefore, \
            "적용 후 위반 수는 적용 전 이하여야 함"


class TestForecastDNA:
    """시계열 DNA 분석 테스트"""

    def test_feature_count(self, seed):
        """65개 이상 특성 추출"""
        rng = np.random.default_rng(seed)
        y = 100.0 + np.cumsum(rng.normal(0, 1, 200))
        y += 10.0 * np.sin(2 * np.pi * np.arange(200) / 7)
        dna = ForecastDNA()
        profile = dna.analyze(y, period=7)
        assert len(profile.features) >= 50, \
            f"50개 이상의 특성 기대, 실제={len(profile.features)}"

    def test_fingerprint(self, seed):
        """8자 hex 지문"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        dna = ForecastDNA()
        profile = dna.analyze(y, period=1)
        assert len(profile.fingerprint) == 8, \
            f"지문 길이 8 기대, 실제={len(profile.fingerprint)}"
        assert all(c in '0123456789ABCDEF' for c in profile.fingerprint), \
            f"지문이 16진수 문자여야 함, 실제='{profile.fingerprint}'"

    def test_difficulty(self, seed):
        """난이도 평가: 유효 범위"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        dna = ForecastDNA()
        profile = dna.analyze(y, period=1)
        assert profile.difficulty in ('easy', 'medium', 'hard', 'very_hard'), \
            f"난이도가 유효한 값이어야 함, 실제='{profile.difficulty}'"
        assert 0 <= profile.difficultyScore <= 100, \
            f"난이도 점수 0-100 범위 기대, 실제={profile.difficultyScore:.1f}"

    def test_similarity(self, seed):
        """동일 시계열 유사도 = 1.0"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        dna = ForecastDNA()
        profile1 = dna.analyze(y, period=1)
        profile2 = dna.analyze(y, period=1)
        sim = dna.similarity(profile1, profile2)
        assert abs(sim - 1.0) < 1e-6, \
            f"동일 시계열의 유사도가 1.0이어야 함, 실제={sim:.6f}"

    def test_different_similarity(self, seed):
        """서로 다른 시계열: 유사도 < 1"""
        rng = np.random.default_rng(seed)
        y1 = rng.standard_normal(200)
        y2 = np.abs(rng.standard_normal(200)) * 100
        dna = ForecastDNA()
        p1 = dna.analyze(y1, period=1)
        p2 = dna.analyze(y2, period=1)
        sim = dna.similarity(p1, p2)
        assert 0.0 <= sim <= 1.0, \
            f"유사도가 0-1 범위여야 함, 실제={sim:.6f}"

    def test_category(self, seed):
        """카테고리 분류"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        dna = ForecastDNA()
        profile = dna.analyze(y, period=1)
        validCategories = {'trending', 'seasonal', 'stationary', 'intermittent', 'volatile', 'complex'}
        assert profile.category in validCategories, \
            f"카테고리가 유효해야 함, 실제='{profile.category}'"

    def test_recommended_models(self, seed):
        """모델 추천 리스트"""
        rng = np.random.default_rng(seed)
        y = 100.0 + np.cumsum(rng.normal(0, 1, 200))
        dna = ForecastDNA()
        profile = dna.analyze(y, period=7)
        assert len(profile.recommendedModels) >= 1, \
            "최소 1개 이상의 모델이 추천되어야 함"
        assert all(isinstance(m, str) for m in profile.recommendedModels), \
            "추천 모델은 문자열 리스트여야 함"
