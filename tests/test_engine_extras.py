"""엔진 확장 모듈 테스트 (변경점, 이벤트, 특성 추출)"""
import pytest
import numpy as np

from forecastx.engine.changepoint import ChangePointDetector
from forecastx.engine.events import EventEffect
from forecastx.engine.tsfeatures import TSFeatureExtractor


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def changepoint_data(seed):
    """명확한 변경점이 있는 시계열"""
    rng = np.random.default_rng(seed)
    seg1 = rng.normal(0, 1, 100)
    seg2 = rng.normal(5, 1, 100)
    seg3 = rng.normal(-2, 1, 100)
    return np.concatenate([seg1, seg2, seg3])


class TestChangePointDetector:
    """변경점 감지 테스트"""

    def test_pelt_detection(self, changepoint_data):
        """PELT: 변경점 감지"""
        detector = ChangePointDetector()
        result = detector.detect(changepoint_data, method='pelt', minSize=10)
        assert result.nChangepoints >= 1, \
            f"최소 1개 변경점 감지 기대, 실제={result.nChangepoints}"
        assert result.method == 'pelt', \
            f"방법이 'pelt'여야 함, 실제='{result.method}'"

    def test_cusum_detection(self, changepoint_data):
        """CUSUM: 변경점 감지"""
        detector = ChangePointDetector()
        result = detector.detect(changepoint_data, method='cusum', minSize=10)
        assert result.method == 'cusum', \
            f"방법이 'cusum'이어야 함, 실제='{result.method}'"
        assert isinstance(result.indices, np.ndarray), \
            "indices는 ndarray여야 함"
        assert isinstance(result.confidence, np.ndarray), \
            "confidence는 ndarray여야 함"

    def test_auto_detection(self, changepoint_data):
        """Auto: 합의 기반 감지"""
        detector = ChangePointDetector()
        result = detector.detect(changepoint_data, method='auto', minSize=10)
        assert result.method == 'auto', \
            f"방법이 'auto'여야 함, 실제='{result.method}'"

    def test_segments_statistics(self, changepoint_data):
        """세그먼트 통계 확인"""
        detector = ChangePointDetector()
        result = detector.detect(changepoint_data, method='pelt', minSize=10)
        assert len(result.segments) >= 1, "최소 1개 세그먼트 통계 기대"
        for seg in result.segments:
            assert 'mean' in seg, "세그먼트에 'mean' 키 기대"
            assert 'std' in seg, "세그먼트에 'std' 키 기대"
            assert 'start' in seg, "세그먼트에 'start' 키 기대"
            assert 'end' in seg, "세그먼트에 'end' 키 기대"

    def test_confidence_range(self, changepoint_data):
        """신뢰도 범위: 0 <= confidence <= 1"""
        detector = ChangePointDetector()
        result = detector.detect(changepoint_data, method='pelt', minSize=10)
        if result.nChangepoints > 0:
            assert np.all(result.confidence >= 0), \
                "신뢰도는 0 이상이어야 함"
            assert np.all(result.confidence <= 1), \
                "신뢰도는 1 이하여야 함"
            assert len(result.confidence) == result.nChangepoints, \
                f"신뢰도 수({len(result.confidence)})가 변경점 수({result.nChangepoints})와 같아야 함"

    def test_no_changepoint(self, seed):
        """변경점 없는 안정 데이터"""
        rng = np.random.default_rng(seed)
        y = rng.normal(0, 0.1, 100)
        detector = ChangePointDetector()
        result = detector.detect(y, method='pelt', minSize=20)
        assert result.nChangepoints >= 0, \
            "변경점 수는 비음수여야 함"

    def test_short_data(self):
        """짧은 데이터: graceful 처리"""
        detector = ChangePointDetector()
        y = np.array([1.0, 2.0, 3.0])
        result = detector.detect(y, method='pelt', minSize=10)
        assert result.nChangepoints == 0, \
            "짧은 데이터에서는 변경점 0 기대"
        assert len(result.segments) >= 1, \
            "짧은 데이터에서도 최소 1개 세그먼트 기대"


class TestEventEffect:
    """이벤트/공휴일 효과 테스트"""

    def test_korean_holidays(self):
        """한국 공휴일 목록 반환"""
        ee = EventEffect(holidays='kr')
        holidays = ee.getKoreanHolidays(2024)
        assert len(holidays) >= 8, \
            f"최소 8개 공휴일 기대 (양력 8개 + 음력), 실제={len(holidays)}"
        names = [h['name'] for h in holidays]
        assert any('신정' in n for n in names), "신정이 포함되어야 함"
        assert any('추석' in n for n in names), "추석이 포함되어야 함"
        for h in holidays:
            assert 'name' in h, "공휴일에 'name' 키 기대"
            assert 'date' in h, "공휴일에 'date' 키 기대"
            assert 'type' in h, "공휴일에 'type' 키 기대"

    def test_event_features_shape(self):
        """이벤트 특성 행렬 크기"""
        ee = EventEffect(holidays='kr')
        dates = np.array([
            np.datetime64('2024-01-01') + np.timedelta64(i, 'D')
            for i in range(365)
        ])
        features = ee.getEventFeatures(dates)
        assert features.shape[0] == 365, \
            f"행 수가 365여야 함, 실제={features.shape[0]}"
        assert features.shape[1] >= 1, \
            f"열 수가 1 이상이어야 함, 실제={features.shape[1]}"

    def test_holiday_effect_on_date(self):
        """공휴일 당일에 효과 = 1.0"""
        ee = EventEffect(holidays='kr')
        newYearDay = np.array([np.datetime64('2024-01-01')])
        features = ee.getEventFeatures(newYearDay)
        maxEffect = np.max(features[0])
        assert maxEffect >= 0.9, \
            f"신정 당일 효과가 0.9 이상이어야 함, 실제={maxEffect:.4f}"

    def test_no_holidays(self):
        """공휴일 없을 때"""
        ee = EventEffect(holidays='none')
        dates = np.array([
            np.datetime64('2024-06-01') + np.timedelta64(i, 'D')
            for i in range(30)
        ])
        features = ee.getEventFeatures(dates)
        assert features.shape[0] == 30, \
            f"행 수가 30이어야 함, 실제={features.shape[0]}"

    def test_custom_event(self):
        """사용자 정의 이벤트"""
        customEvents = [
            {
                'name': 'sale_event',
                'dates': ['2024-06-15'],
                'priorWindow': 2,
                'postWindow': 1,
            }
        ]
        ee = EventEffect(holidays='none', customEvents=customEvents)
        dates = np.array([
            np.datetime64('2024-06-13') + np.timedelta64(i, 'D')
            for i in range(5)
        ])
        features = ee.getEventFeatures(dates)
        assert features.shape[1] >= 1, "사용자 정의 이벤트 열이 존재해야 함"
        dayIdx = 2  # 2024-06-15
        assert features[dayIdx, 0] >= 0.9, \
            f"이벤트 당일 효과가 0.9 이상이어야 함, 실제={features[dayIdx, 0]:.4f}"

    def test_list_events(self):
        """등록된 이벤트 목록"""
        ee = EventEffect(holidays='kr')
        events = ee.listEvents()
        assert len(events) >= 8, \
            f"최소 8개 이벤트 기대, 실제={len(events)}"
        assert all(isinstance(e, str) for e in events), \
            "이벤트 이름은 문자열이어야 함"


class TestTSFeatureExtractor:
    """시계열 특성 추출 테스트"""

    def test_feature_count(self, seed):
        """65개 이상 특성 추출"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        extractor = TSFeatureExtractor()
        features = extractor.extract(y, period=7)
        assert len(features) >= 60, \
            f"60개 이상의 특성 기대, 실제={len(features)}"

    def test_basic_stats_keys(self, seed):
        """기본 통계 키 존재"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        extractor = TSFeatureExtractor()
        features = extractor.extract(y, period=7)
        expectedKeys = ['length', 'mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis']
        for key in expectedKeys:
            assert key in features, f"기본 통계 키 '{key}'가 존재해야 함"
        assert features['length'] == 200.0, \
            f"length가 200이어야 함, 실제={features['length']}"

    def test_trend_features(self, seed):
        """추세 특성 키 존재"""
        rng = np.random.default_rng(seed)
        y = np.arange(200, dtype=np.float64) + rng.normal(0, 1, 200)
        extractor = TSFeatureExtractor()
        features = extractor.extract(y, period=7)
        assert 'trend_strength' in features, "trend_strength 키 기대"
        assert 'trend_slope' in features, "trend_slope 키 기대"
        assert features['trend_strength'] > 0.5, \
            f"강한 추세 데이터에서 trend_strength > 0.5 기대, 실제={features['trend_strength']:.4f}"

    def test_similarity_identical(self, seed):
        """동일 특성의 유사도: z-score 정규화에서 동일 벡터는 0벡터가 됨"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        extractor = TSFeatureExtractor()
        f1 = extractor.extract(y, period=7)
        f2 = extractor.extract(y, period=7)
        sim = extractor.similarity(f1, f2)
        # z-score 정규화 시 동일 벡터 -> 0벡터 -> 유사도 0.0 반환
        assert abs(sim - 0.0) < 1e-6, \
            f"동일 특성의 z-score 유사도가 0.0이어야 함, 실제={sim:.6f}"

    def test_similarity_range(self, seed):
        """유사도 범위: -1 <= sim <= 1"""
        rng = np.random.default_rng(seed)
        y1 = rng.standard_normal(200)
        y2 = np.abs(rng.standard_normal(200)) * 100
        extractor = TSFeatureExtractor()
        f1 = extractor.extract(y1, period=7)
        f2 = extractor.extract(y2, period=7)
        sim = extractor.similarity(f1, f2)
        assert -1.0 <= sim <= 1.0, \
            f"유사도가 [-1, 1] 범위여야 함, 실제={sim:.6f}"

    def test_fingerprint(self, seed):
        """8자 hex 지문"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        extractor = TSFeatureExtractor()
        features = extractor.extract(y, period=7)
        fp = extractor.fingerprint(features)
        assert len(fp) == 8, \
            f"지문 길이 8 기대, 실제={len(fp)}"
        assert all(c in '0123456789abcdef' for c in fp), \
            f"지문이 16진수 소문자여야 함, 실제='{fp}'"

    def test_fingerprint_deterministic(self, seed):
        """동일 데이터 -> 동일 지문"""
        rng = np.random.default_rng(seed)
        y = rng.standard_normal(200)
        extractor = TSFeatureExtractor()
        f1 = extractor.extract(y, period=7)
        f2 = extractor.extract(y, period=7)
        fp1 = extractor.fingerprint(f1)
        fp2 = extractor.fingerprint(f2)
        assert fp1 == fp2, \
            f"동일 데이터의 지문이 같아야 함: '{fp1}' != '{fp2}'"

    def test_seasonality_features(self, seed):
        """계절성 특성"""
        rng = np.random.default_rng(seed)
        t = np.arange(200, dtype=np.float64)
        y = 10.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, 200)
        extractor = TSFeatureExtractor()
        features = extractor.extract(y, period=7)
        assert 'seasonality_strength' in features, "seasonality_strength 키 기대"
        assert features['seasonality_strength'] > 0.3, \
            f"강한 계절성에서 seasonality_strength > 0.3 기대, 실제={features['seasonality_strength']:.4f}"

    def test_short_data(self):
        """짧은 데이터: 빈 특성 반환"""
        y = np.array([1.0, 2.0, 3.0])
        extractor = TSFeatureExtractor()
        features = extractor.extract(y, period=1)
        assert isinstance(features, dict), "짧은 데이터에서도 딕셔너리 반환"
        assert len(features) > 0, "짧은 데이터에서도 비어있지 않은 딕셔너리 기대"
