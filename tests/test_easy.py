"""Easy API 테스트 (forecast, analyze, regress, quick_report)"""
import pytest
import numpy as np
import pandas as pd

from vectrix.easy import (
    forecast,
    analyze,
    regress,
    quick_report,
    EasyForecastResult,
    EasyAnalysisResult,
    EasyRegressionResult,
)


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def ts_list(seed):
    """예측용 리스트 데이터 (200개, 추세+노이즈)"""
    rng = np.random.default_rng(seed)
    trend = np.linspace(100, 200, 200)
    noise = rng.normal(0, 3, 200)
    return (trend + noise).tolist()


@pytest.fixture
def ts_ndarray(seed):
    """예측용 ndarray 데이터 (200개, 추세+계절성+노이즈)"""
    rng = np.random.default_rng(seed)
    t = np.arange(200, dtype=np.float64)
    trend = 100.0 + 0.5 * t
    seasonal = 10.0 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 2, 200)
    return trend + seasonal + noise


@pytest.fixture
def ts_dataframe(seed):
    """예측용 DataFrame 데이터 (date, sales 컬럼)"""
    rng = np.random.default_rng(seed)
    n = 200
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    values = 100.0 + np.linspace(0, 50, n) + rng.normal(0, 3, n)
    return pd.DataFrame({'date': dates, 'sales': values})


@pytest.fixture
def regression_data(seed):
    """회귀분석용 데이터 (y, X)"""
    rng = np.random.default_rng(seed)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + noise
    X = np.column_stack([x1, x2])
    return y, X


@pytest.fixture
def regression_dataframe(seed):
    """회귀분석용 DataFrame (formula 방식)"""
    rng = np.random.default_rng(seed)
    n = 100
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    y = 3.0 + 2.0 * x1 - 1.5 * x2 + noise
    return pd.DataFrame({'sales': y, 'ads': x1, 'price': x2})


# =========================================================================
# TestEasyForecast
# =========================================================================

class TestEasyForecast:
    """Easy API forecast() 테스트"""

    def test_forecast_from_list(self, ts_list):
        """리스트 입력 예측"""
        result = forecast(ts_list, steps=10)
        assert isinstance(result, EasyForecastResult), \
            "forecast(list) -> EasyForecastResult 타입이어야 함"
        assert len(result.predictions) == 10, \
            f"예측 길이가 10이어야 함, 실제={len(result.predictions)}"
        assert len(result.dates) == 10, \
            f"날짜 길이가 10이어야 함, 실제={len(result.dates)}"
        assert len(result.lower) == 10, \
            f"lower 길이가 10이어야 함, 실제={len(result.lower)}"
        assert len(result.upper) == 10, \
            f"upper 길이가 10이어야 함, 실제={len(result.upper)}"

    def test_forecast_from_ndarray(self, ts_ndarray):
        """ndarray 입력 예측"""
        result = forecast(ts_ndarray, steps=14)
        assert isinstance(result, EasyForecastResult), \
            "forecast(ndarray) -> EasyForecastResult 타입이어야 함"
        assert len(result.predictions) == 14, \
            f"예측 길이가 14여야 함, 실제={len(result.predictions)}"
        assert isinstance(result.model, str), \
            "model 이름은 문자열이어야 함"
        assert len(result.model) > 0, \
            "model 이름이 비어있으면 안 됨"

    def test_forecast_from_dataframe(self, ts_dataframe):
        """DataFrame 입력 예측 (자동 컬럼 감지)"""
        result = forecast(ts_dataframe, date='date', value='sales', steps=7)
        assert isinstance(result, EasyForecastResult), \
            "forecast(DataFrame) -> EasyForecastResult 타입이어야 함"
        assert len(result.predictions) == 7, \
            f"예측 길이가 7이어야 함, 실제={len(result.predictions)}"

    def test_forecast_confidence_interval(self, ts_list):
        """신뢰구간: lower <= predictions <= upper"""
        result = forecast(ts_list, steps=10)
        assert np.all(result.lower <= result.predictions + 1e-10), \
            "하한은 예측값 이하여야 함"
        assert np.all(result.upper >= result.predictions - 1e-10), \
            "상한은 예측값 이상이어야 함"

    def test_forecast_summary(self, ts_list):
        """summary() 문자열 반환"""
        result = forecast(ts_list, steps=5)
        summaryText = result.summary()
        assert isinstance(summaryText, str), \
            "summary()는 문자열을 반환해야 함"
        assert len(summaryText) > 0, \
            "summary() 문자열이 비어있으면 안 됨"
        assert "Vectrix" in summaryText, \
            "summary()에 'Vectrix' 문구가 포함되어야 함"

    def test_forecast_to_dataframe(self, ts_list):
        """to_dataframe() 결과 확인"""
        result = forecast(ts_list, steps=5)
        df = result.to_dataframe()
        assert isinstance(df, pd.DataFrame), \
            "to_dataframe()은 DataFrame을 반환해야 함"
        assert 'date' in df.columns, \
            "DataFrame에 'date' 컬럼이 있어야 함"
        assert 'prediction' in df.columns, \
            "DataFrame에 'prediction' 컬럼이 있어야 함"
        assert 'lower95' in df.columns, \
            "DataFrame에 'lower95' 컬럼이 있어야 함"
        assert 'upper95' in df.columns, \
            "DataFrame에 'upper95' 컬럼이 있어야 함"
        assert len(df) == 5, \
            f"DataFrame 행 수가 5여야 함, 실제={len(df)}"

    def test_forecast_short_data_error(self):
        """짧은 데이터: ValueError 발생"""
        with pytest.raises(ValueError):
            forecast([1, 2, 3], steps=5)

    def test_forecast_repr(self, ts_list):
        """__repr__ 형식 확인"""
        result = forecast(ts_list, steps=5)
        reprStr = repr(result)
        assert "ForecastResult" in reprStr, \
            f"repr에 'ForecastResult'가 포함되어야 함, 실제='{reprStr}'"


# =========================================================================
# TestEasyAnalyze
# =========================================================================

class TestEasyAnalyze:
    """Easy API analyze() 테스트"""

    def test_analyze_from_list(self, ts_list):
        """리스트 입력 분석"""
        result = analyze(ts_list)
        assert isinstance(result, EasyAnalysisResult), \
            "analyze(list) -> EasyAnalysisResult 타입이어야 함"

    def test_analyze_dna_profile(self, ts_ndarray):
        """DNA 프로파일 존재"""
        result = analyze(ts_ndarray)
        assert result.dna is not None, "DNA 프로파일이 None이면 안 됨"
        assert hasattr(result.dna, 'fingerprint'), \
            "DNA에 fingerprint 속성이 있어야 함"
        assert hasattr(result.dna, 'difficulty'), \
            "DNA에 difficulty 속성이 있어야 함"
        assert hasattr(result.dna, 'category'), \
            "DNA에 category 속성이 있어야 함"
        assert result.dna.difficulty in ('easy', 'medium', 'hard', 'very_hard'), \
            f"난이도가 유효해야 함, 실제='{result.dna.difficulty}'"

    def test_analyze_changepoints(self, ts_ndarray):
        """변경점 결과 타입 확인"""
        result = analyze(ts_ndarray)
        assert isinstance(result.changepoints, np.ndarray), \
            "changepoints는 ndarray여야 함"

    def test_analyze_anomalies(self, ts_ndarray):
        """이상치 결과 타입 확인"""
        result = analyze(ts_ndarray)
        assert isinstance(result.anomalies, np.ndarray), \
            "anomalies는 ndarray여야 함"

    def test_analyze_features(self, ts_ndarray):
        """특성 딕셔너리 크기"""
        result = analyze(ts_ndarray)
        assert isinstance(result.features, dict), \
            "features는 dict여야 함"
        assert len(result.features) >= 10, \
            f"최소 10개 이상의 특성 기대, 실제={len(result.features)}"

    def test_analyze_characteristics(self, ts_dataframe):
        """데이터 특성 객체 존재"""
        result = analyze(ts_dataframe, date='date', value='sales')
        assert result.characteristics is not None, \
            "characteristics가 None이면 안 됨"
        assert hasattr(result.characteristics, 'length'), \
            "characteristics에 length 속성이 있어야 함"

    def test_analyze_summary(self, ts_list):
        """summary() 문자열 반환"""
        result = analyze(ts_list)
        summaryText = result.summary()
        assert isinstance(summaryText, str), \
            "summary()는 문자열을 반환해야 함"
        assert len(summaryText) > 0, \
            "summary() 문자열이 비어있으면 안 됨"

    def test_analyze_from_dataframe(self, ts_dataframe):
        """DataFrame 입력 분석"""
        result = analyze(ts_dataframe, date='date', value='sales')
        assert isinstance(result, EasyAnalysisResult), \
            "analyze(DataFrame) -> EasyAnalysisResult 타입이어야 함"
        assert result.dna is not None, "DNA 프로파일이 None이면 안 됨"

    def test_analyze_repr(self, ts_list):
        """__repr__ 형식 확인"""
        result = analyze(ts_list)
        reprStr = repr(result)
        assert "AnalysisResult" in reprStr, \
            f"repr에 'AnalysisResult'가 포함되어야 함, 실제='{reprStr}'"


# =========================================================================
# TestEasyRegress
# =========================================================================

class TestEasyRegress:
    """Easy API regress() 테스트"""

    def test_regress_direct(self, regression_data):
        """직접 입력 방식: regress(y, X)"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        assert isinstance(result, EasyRegressionResult), \
            "regress(y, X) -> EasyRegressionResult 타입이어야 함"
        # 절편 포함이므로 계수 3개 (intercept, x1, x2)
        assert len(result.coefficients) == 3, \
            f"계수 3개 기대 (절편+2변수), 실제={len(result.coefficients)}"
        assert 0.0 <= result.r_squared <= 1.0, \
            f"R-squared가 0-1 범위여야 함, 실제={result.r_squared:.4f}"

    def test_regress_formula(self, regression_dataframe):
        """formula 방식: regress(data=df, formula='...')"""
        result = regress(
            data=regression_dataframe,
            formula='sales ~ ads + price',
            summary=False
        )
        assert isinstance(result, EasyRegressionResult), \
            "regress(formula) -> EasyRegressionResult 타입이어야 함"
        assert len(result.coefficients) == 3, \
            f"계수 3개 기대 (절편+ads+price), 실제={len(result.coefficients)}"
        assert result.r_squared > 0.5, \
            f"R-squared > 0.5 기대 (강한 선형 관계), 실제={result.r_squared:.4f}"

    def test_regress_pvalues(self, regression_data):
        """p-values 범위 확인"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        assert len(result.pvalues) == len(result.coefficients), \
            f"pvalues 수({len(result.pvalues)})가 계수 수({len(result.coefficients)})와 같아야 함"
        assert np.all(result.pvalues >= 0), \
            "모든 p-value는 비음수여야 함"
        assert np.all(result.pvalues <= 1), \
            "모든 p-value는 1 이하여야 함"

    def test_regress_adj_r_squared(self, regression_data):
        """수정 R-squared 확인"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        assert result.adj_r_squared <= result.r_squared + 1e-10, \
            f"adj_r_squared({result.adj_r_squared:.4f})는 r_squared({result.r_squared:.4f}) 이하여야 함"

    def test_regress_f_stat(self, regression_data):
        """F-통계량 양수 확인"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        assert result.f_stat > 0, \
            f"F-통계량은 양수여야 함, 실제={result.f_stat:.4f}"

    def test_regress_summary_output(self, regression_data):
        """summary() 문자열 반환"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        summaryText = result.summary()
        assert isinstance(summaryText, str), \
            "summary()는 문자열을 반환해야 함"
        assert len(summaryText) > 0, \
            "summary() 문자열이 비어있으면 안 됨"

    def test_regress_predict(self, regression_data):
        """새 데이터 예측: predict()"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        newX = np.array([[1.0, -1.0], [0.5, 0.5]])
        predDf = result.predict(newX)
        assert isinstance(predDf, pd.DataFrame), \
            "predict()는 DataFrame을 반환해야 함"
        assert 'prediction' in predDf.columns, \
            "predict 결과에 'prediction' 컬럼이 있어야 함"
        assert len(predDf) == 2, \
            f"predict 결과 행 수가 2여야 함, 실제={len(predDf)}"

    def test_regress_coefficients_accuracy(self, regression_data):
        """계수 정확도: 실제 계수와 가까움"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        # 참값: intercept=3.0, beta1=2.0, beta2=-1.5
        assert abs(result.coefficients[0] - 3.0) < 0.5, \
            f"절편이 3.0 근처여야 함, 실제={result.coefficients[0]:.4f}"
        assert abs(result.coefficients[1] - 2.0) < 0.5, \
            f"beta1이 2.0 근처여야 함, 실제={result.coefficients[1]:.4f}"
        assert abs(result.coefficients[2] - (-1.5)) < 0.5, \
            f"beta2가 -1.5 근처여야 함, 실제={result.coefficients[2]:.4f}"

    def test_regress_invalid_input(self):
        """잘못된 입력: ValueError 발생"""
        with pytest.raises(ValueError):
            regress(summary=False)

    def test_regress_repr(self, regression_data):
        """__repr__ 형식 확인"""
        y, X = regression_data
        result = regress(y, X, summary=False)
        reprStr = repr(result)
        assert "RegressionResult" in reprStr, \
            f"repr에 'RegressionResult'가 포함되어야 함, 실제='{reprStr}'"

    def test_regress_formula_dot(self, regression_dataframe):
        """formula '.' 방식: y ~ . (모든 독립변수)"""
        result = regress(
            data=regression_dataframe,
            formula='sales ~ .',
            summary=False
        )
        assert isinstance(result, EasyRegressionResult), \
            "regress(formula='.') -> EasyRegressionResult 타입이어야 함"
        # ads, price -> 2 변수 + 절편
        assert len(result.coefficients) == 3, \
            f"계수 3개 기대 (절편+2변수), 실제={len(result.coefficients)}"


# =========================================================================
# TestQuickReport
# =========================================================================

class TestQuickReport:
    """Easy API quick_report() 테스트"""

    def test_quick_report_keys(self, ts_list):
        """반환 딕셔너리 키 확인"""
        report = quick_report(ts_list, steps=5)
        assert isinstance(report, dict), \
            "quick_report()은 dict를 반환해야 함"
        assert 'forecast' in report, \
            "report에 'forecast' 키가 있어야 함"
        assert 'analysis' in report, \
            "report에 'analysis' 키가 있어야 함"
        assert 'summary' in report, \
            "report에 'summary' 키가 있어야 함"

    def test_quick_report_forecast_type(self, ts_list):
        """forecast 값 타입 확인"""
        report = quick_report(ts_list, steps=5)
        assert isinstance(report['forecast'], EasyForecastResult), \
            "report['forecast']는 EasyForecastResult 타입이어야 함"
        assert len(report['forecast'].predictions) == 5, \
            f"예측 길이가 5여야 함, 실제={len(report['forecast'].predictions)}"

    def test_quick_report_analysis_type(self, ts_list):
        """analysis 값 타입 확인"""
        report = quick_report(ts_list, steps=5)
        assert isinstance(report['analysis'], EasyAnalysisResult), \
            "report['analysis']는 EasyAnalysisResult 타입이어야 함"

    def test_quick_report_summary_text(self, ts_list):
        """summary 텍스트 확인"""
        report = quick_report(ts_list, steps=5)
        summaryText = report['summary']
        assert isinstance(summaryText, str), \
            "report['summary']는 문자열이어야 함"
        assert "Vectrix" in summaryText, \
            "summary에 'Vectrix' 문구가 포함되어야 함"

    def test_quick_report_from_dataframe(self, ts_dataframe):
        """DataFrame 입력 보고서"""
        report = quick_report(ts_dataframe, date='date', value='sales', steps=7)
        assert isinstance(report, dict), \
            "quick_report(DataFrame)은 dict를 반환해야 함"
        assert len(report['forecast'].predictions) == 7, \
            f"예측 길이가 7이어야 함, 실제={len(report['forecast'].predictions)}"
