"""회귀분석 모듈 테스트"""
import pytest
import numpy as np

from forecastx.regression import (
    OLSInference,
    RegressionResult,
    RegressionDiagnostics,
    DiagnosticResult,
    HuberRegressor,
    RANSACRegressor,
    QuantileRegressor,
    StepwiseSelector,
    RegularizationCV,
    NeweyWestOLS,
    CochraneOrcutt,
    GrangerCausality,
)


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def simple_data(seed):
    """y = 2 + 3*x1 + noise"""
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.standard_normal((n, 1))
    y = 2.0 + 3.0 * x[:, 0] + rng.normal(0, 0.5, n)
    return x, y


@pytest.fixture
def multi_data(seed):
    """y = 1 + 2*x1 - 0.5*x2 + noise"""
    rng = np.random.default_rng(seed)
    n = 200
    X = rng.standard_normal((n, 3))
    y = 1.0 + 2.0 * X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.3, n)
    return X, y


class TestOLSInference:
    """OLS 통계적 추론 테스트"""

    def test_basic_ols(self, simple_data):
        """기본 OLS: R2, coefficients, p-values"""
        X, y = simple_data
        engine = OLSInference()
        result = engine.fit(X, y)
        assert result.rSquared > 0.9, f"R2={result.rSquared:.4f}, 강한 선형 관계에서 R2 > 0.9 기대"
        assert abs(result.coefficients[0] - 2.0) < 0.5, f"절편 ≈ 2.0 기대, 실제={result.coefficients[0]:.4f}"
        assert abs(result.coefficients[1] - 3.0) < 0.5, f"기울기 ≈ 3.0 기대, 실제={result.coefficients[1]:.4f}"
        assert result.pValues[1] < 0.05, f"유의한 계수의 p-value < 0.05 기대, 실제={result.pValues[1]:.4f}"

    def test_perfect_fit(self):
        """완벽한 적합: R2 = 1.0"""
        n = 100
        X = np.arange(n, dtype=np.float64).reshape(-1, 1)
        y = 5.0 + 2.0 * X[:, 0]
        engine = OLSInference()
        result = engine.fit(X, y)
        assert abs(result.rSquared - 1.0) < 1e-10, f"완벽한 선형 관계에서 R2 = 1.0 기대, 실제={result.rSquared:.10f}"

    def test_no_relationship(self, seed):
        """무관계 데이터: R2 ≈ 0, p-values 큼"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 1))
        y = rng.standard_normal(n)
        engine = OLSInference()
        result = engine.fit(X, y)
        assert result.rSquared < 0.1, f"무관계 데이터에서 R2 < 0.1 기대, 실제={result.rSquared:.4f}"
        assert result.pValues[1] > 0.05, f"무관계 변수의 p-value > 0.05 기대, 실제={result.pValues[1]:.4f}"

    def test_single_feature(self, simple_data):
        """단일 변수 회귀"""
        X, y = simple_data
        engine = OLSInference()
        result = engine.fit(X, y)
        assert result.nParams == 2, f"절편 + 1변수 = 2 파라미터 기대, 실제={result.nParams}"
        assert result.nObs == 200, f"관측 수 200 기대, 실제={result.nObs}"
        assert result.degreesOfFreedom == 198, f"자유도 198 기대, 실제={result.degreesOfFreedom}"

    def test_no_intercept(self, seed):
        """절편 없는 회귀"""
        rng = np.random.default_rng(seed)
        n = 100
        X = rng.standard_normal((n, 1))
        y = 3.0 * X[:, 0] + rng.normal(0, 0.3, n)
        engine = OLSInference(fitIntercept=False)
        result = engine.fit(X, y)
        assert result.nParams == 1, f"절편 없이 1 파라미터 기대, 실제={result.nParams}"
        assert abs(result.coefficients[0] - 3.0) < 0.5, f"기울기 ≈ 3.0 기대, 실제={result.coefficients[0]:.4f}"

    def test_summary_output(self, simple_data):
        """summary() 형식 확인"""
        X, y = simple_data
        engine = OLSInference()
        result = engine.fit(X, y)
        summary = result.summary()
        assert isinstance(summary, str), "summary()는 문자열을 반환해야 함"
        assert "R-squared" in summary, "summary에 R-squared 포함 기대"
        assert "OLS" in summary, "summary에 OLS 포함 기대"
        assert "P>|t|" in summary, "summary에 P>|t| 포함 기대"

    def test_prediction_intervals(self, simple_data):
        """예측구간이 신뢰구간보다 넓음"""
        X, y = simple_data
        engine = OLSInference()
        engine.fit(X, y)
        Xnew = np.array([[0.5], [1.0], [1.5]])
        _, confLo, confHi = engine.predict(Xnew, interval='confidence')
        _, predLo, predHi = engine.predict(Xnew, interval='prediction')
        confWidths = confHi - confLo
        predWidths = predHi - predLo
        assert np.all(predWidths > confWidths), "예측구간은 신뢰구간보다 항상 넓어야 함"

    def test_f_statistic(self, simple_data):
        """F-통계량과 p-value 관계"""
        X, y = simple_data
        engine = OLSInference()
        result = engine.fit(X, y)
        assert result.fStatistic > 0, f"유의한 모델에서 F > 0 기대, 실제={result.fStatistic:.4f}"
        assert result.fPValue < 0.05, f"유의한 모델에서 F p-value < 0.05 기대, 실제={result.fPValue:.6f}"


class TestDiagnostics:
    """잔차 진단 테스트"""

    def test_vif_independent(self, seed):
        """독립 변수: VIF ≈ 1"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 3))
        y = 1.0 + X[:, 0] + X[:, 1] + X[:, 2] + rng.normal(0, 0.5, n)
        engine = OLSInference()
        result = engine.fit(X, y)
        diag = RegressionDiagnostics()
        diagResult = diag.diagnose(
            X, y, result.residuals, result.hatMatrix,
            result.coefficients, result.fittedValues
        )
        assert np.all(diagResult.vif < 5), f"독립 변수들의 VIF < 5 기대, 실제={diagResult.vif}"

    def test_vif_collinear(self, seed):
        """공선 변수: VIF > 10"""
        rng = np.random.default_rng(seed)
        n = 200
        x1 = rng.standard_normal(n)
        x2 = x1 + rng.normal(0, 0.01, n)
        X = np.column_stack([x1, x2])
        y = 1.0 + x1 + rng.normal(0, 0.5, n)
        engine = OLSInference()
        result = engine.fit(X, y)
        diag = RegressionDiagnostics()
        diagResult = diag.diagnose(
            X, y, result.residuals, result.hatMatrix,
            result.coefficients, result.fittedValues
        )
        assert np.any(diagResult.vif > 10), f"공선 변수에서 VIF > 10 기대, 실제={diagResult.vif}"

    def test_breusch_pagan(self, seed):
        """등분산 데이터: BP 유의하지 않음"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 2))
        y = 1.0 + X[:, 0] + rng.normal(0, 1.0, n)
        engine = OLSInference()
        result = engine.fit(X, y)
        diag = RegressionDiagnostics()
        diagResult = diag.diagnose(
            X, y, result.residuals, result.hatMatrix,
            result.coefficients, result.fittedValues
        )
        assert diagResult.breuschPagan['pValue'] > 0.01, \
            f"등분산 데이터에서 BP p-value > 0.01 기대, 실제={diagResult.breuschPagan['pValue']:.4f}"

    def test_durbin_watson_range(self, simple_data):
        """DW 범위: 0 < DW < 4"""
        X, y = simple_data
        engine = OLSInference()
        result = engine.fit(X, y)
        diag = RegressionDiagnostics()
        diagResult = diag.diagnose(
            X, y, result.residuals, result.hatMatrix,
            result.coefficients, result.fittedValues
        )
        dw = diagResult.durbinWatson
        assert 0 < dw < 4, f"DW 범위 (0, 4) 기대, 실제={dw:.4f}"

    def test_cooks_distance(self, seed):
        """이상치 포함: Cook's D > 4/n"""
        rng = np.random.default_rng(seed)
        n = 100
        X = rng.standard_normal((n, 1))
        y = 2.0 + 3.0 * X[:, 0] + rng.normal(0, 0.5, n)
        y[0] = 100.0
        engine = OLSInference()
        result = engine.fit(X, y)
        diag = RegressionDiagnostics()
        diagResult = diag.diagnose(
            X, y, result.residuals, result.hatMatrix,
            result.coefficients, result.fittedValues
        )
        threshold = 4.0 / n
        assert np.any(diagResult.cooksDistance > threshold), \
            f"이상치가 있을 때 Cook's D > {threshold:.4f}인 관측치 기대"

    def test_plot_data(self, simple_data):
        """플롯 데이터 구조 확인"""
        X, y = simple_data
        engine = OLSInference()
        result = engine.fit(X, y)
        diag = RegressionDiagnostics()
        diagResult = diag.diagnose(
            X, y, result.residuals, result.hatMatrix,
            result.coefficients, result.fittedValues
        )
        assert 'residualsVsFitted' in diagResult.plotData, "residualsVsFitted 플롯 데이터 기대"
        assert 'normalQQ' in diagResult.plotData, "normalQQ 플롯 데이터 기대"
        assert 'scaleLocation' in diagResult.plotData, "scaleLocation 플롯 데이터 기대"
        assert 'residualsVsLeverage' in diagResult.plotData, "residualsVsLeverage 플롯 데이터 기대"
        assert 'x' in diagResult.plotData['residualsVsFitted'], "플롯 데이터에 x 키 기대"
        assert 'y' in diagResult.plotData['residualsVsFitted'], "플롯 데이터에 y 키 기대"


class TestRobust:
    """강건 회귀 테스트"""

    def test_huber_outlier_resistance(self, seed):
        """Huber: 이상치에 강건"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 1))
        y = 2.0 + 3.0 * X[:, 0] + rng.normal(0, 0.5, n)
        y[:5] = 100.0

        ols = OLSInference()
        olsResult = ols.fit(X, y)

        huber = HuberRegressor()
        huber.fit(X, y)

        trueSlope = 3.0
        olsError = abs(olsResult.coefficients[1] - trueSlope)
        huberError = abs(huber.coef[0] - trueSlope)
        assert huberError < olsError, \
            f"Huber가 OLS보다 이상치에 강건해야 함: Huber 오차={huberError:.4f}, OLS 오차={olsError:.4f}"

    def test_ransac_outlier_resistance(self, seed):
        """RANSAC: 이상치 무시"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 1))
        y = 1.0 + 2.0 * X[:, 0] + rng.normal(0, 0.3, n)
        y[:20] = 50.0

        ransac = RANSACRegressor(randomState=seed)
        ransac.fit(X, y)

        assert ransac.coef is not None, "RANSAC 계수가 None이면 안 됨"
        assert abs(ransac.coef[0] - 2.0) < 1.0, \
            f"RANSAC 기울기 ≈ 2.0 기대, 실제={ransac.coef[0]:.4f}"
        assert ransac.inlierMask is not None, "inlierMask가 존재해야 함"
        assert np.sum(~ransac.inlierMask) > 0, "일부 이상치가 마스킹되어야 함"

    def test_quantile_median(self, seed):
        """Quantile(0.5) ≈ median regression"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 1))
        y = 1.0 + 2.0 * X[:, 0] + rng.normal(0, 0.5, n)

        qr = QuantileRegressor(quantile=0.5)
        qr.fit(X, y)
        assert qr.coef is not None, "QuantileRegressor 계수가 None이면 안 됨"
        assert abs(qr.coef[0] - 2.0) < 1.0, \
            f"Quantile(0.5) 기울기 ≈ 2.0 기대, 실제={qr.coef[0]:.4f}"


class TestSelection:
    """변수 선택 테스트"""

    def test_forward_selection(self, seed):
        """Forward: 유의한 변수만 선택"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 5))
        y = 1.0 + 2.0 * X[:, 0] + 1.5 * X[:, 1] + rng.normal(0, 0.5, n)

        selector = StepwiseSelector(direction='forward', criterion='bic')
        result = selector.select(X, y, featureNames=['x0', 'x1', 'x2', 'x3', 'x4'])
        assert 0 in result.selectedIndices, "x0(유의한 변수)가 선택되어야 함"
        assert 1 in result.selectedIndices, "x1(유의한 변수)가 선택되어야 함"
        assert len(result.selectedFeatures) >= 2, "최소 2개 유의한 변수가 선택되어야 함"
        assert len(result.selectionHistory) > 0, "선택 이력이 존재해야 함"

    def test_regularization_cv(self, seed):
        """RidgeCV: bestAlpha > 0"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 5))
        y = 1.0 + X[:, 0] + X[:, 1] + rng.normal(0, 0.5, n)

        rcv = RegularizationCV(model='ridge', nFolds=3, randomState=seed)
        result = rcv.fit(X, y)
        assert result.bestAlpha > 0, f"bestAlpha > 0 기대, 실제={result.bestAlpha}"
        assert len(result.coef) == 5, f"5개 계수 기대, 실제={len(result.coef)}"
        assert result.bestScore < 0, f"neg MSE이므로 bestScore < 0 기대, 실제={result.bestScore}"


class TestTimeSeriesRegression:
    """시계열 회귀 테스트"""

    def test_newey_west(self, seed):
        """Newey-West: SE 존재"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 2))
        y = 1.0 + X[:, 0] + rng.normal(0, 1.0, n)

        nw = NeweyWestOLS()
        result = nw.fit(X, y)
        assert len(result.stdErrors) == 3, f"3개 표준오차(절편+2변수) 기대, 실제={len(result.stdErrors)}"
        assert np.all(result.stdErrors >= 0), "표준오차는 비음수여야 함"
        assert result.covarianceType == 'HAC (Newey-West)', \
            f"공분산 유형은 HAC (Newey-West) 기대, 실제={result.covarianceType}"

    def test_cochrane_orcutt(self, seed):
        """CO: rho in [-1, 1]"""
        rng = np.random.default_rng(seed)
        n = 200
        X = rng.standard_normal((n, 1))
        noise = np.zeros(n)
        noise[0] = rng.normal()
        for t in range(1, n):
            noise[t] = 0.5 * noise[t - 1] + rng.normal()
        y = 1.0 + 2.0 * X[:, 0] + noise

        co = CochraneOrcutt()
        result, rho = co.fit(X, y)
        assert -1 <= rho <= 1, f"rho in [-1, 1] 기대, 실제={rho:.4f}"
        assert result.covarianceType == 'Cochrane-Orcutt GLS', \
            f"공분산 유형은 Cochrane-Orcutt GLS 기대, 실제={result.covarianceType}"
        assert len(result.coef) == 1, f"1개 계수 기대, 실제={len(result.coef)}"

    def test_granger_causality(self, seed):
        """Granger: 비관계 시 p > 0.05"""
        rng = np.random.default_rng(seed)
        n = 200
        y = rng.standard_normal(n)
        x = rng.standard_normal(n)

        gc = GrangerCausality(maxLag=4)
        result = gc.test(y, x)
        assert result.pValue > 0.01, \
            f"독립 시계열에서 Granger p-value > 0.01 기대, 실제={result.pValue:.4f}"
        assert result.optimalLag >= 1, f"optimalLag >= 1 기대, 실제={result.optimalLag}"
        assert len(result.fStatPerLag) > 0, "lag별 F-통계량이 존재해야 함"
