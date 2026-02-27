"""ForecastX core tests"""

import numpy as np
import pandas as pd
import pytest

from forecastx import ForecastX, ForecastResult


def _make_ts(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    trend = np.linspace(100, 150, n)
    weekly = 20 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = rng.normal(0, 5, n)
    return pd.DataFrame({"date": dates, "value": trend + weekly + noise})


class TestBasicForecast:
    def test_forecast_returns_success(self):
        df = _make_ts(200)
        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
        assert result.success is True
        assert len(result.predictions) == 30
        assert len(result.dates) == 30

    def test_forecast_with_short_data(self):
        df = _make_ts(30)
        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=14)
        assert result.success is True
        assert len(result.predictions) == 14

    def test_forecast_too_short_data(self):
        df = _make_ts(5)
        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=7)
        assert result.success is False

    def test_forecast_result_type(self):
        df = _make_ts(200)
        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
        assert isinstance(result, ForecastResult)
        assert result.bestModelName is not None
        assert result.characteristics is not None


class TestAnalyze:
    def test_analyze_only(self):
        df = _make_ts(150)
        fx = ForecastX()
        analysis = fx.analyze(df, dateCol="date", valueCol="value")
        assert "characteristics" in analysis
        assert "flatRisk" in analysis
        assert analysis["characteristics"].length == 150
        assert analysis["characteristics"].hasSeasonality == True


class TestVariability:
    def test_prediction_preserves_variability(self):
        df = _make_ts(200)
        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
        if result.success:
            origStd = np.std(df["value"].values[-30:])
            predStd = np.std(result.predictions)
            ratio = predStd / origStd
            assert ratio > 0.1, f"Prediction variability too low: {ratio:.3f}"


class TestRandomWalk:
    def test_random_walk_data(self):
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        values = 100 + np.cumsum(rng.normal(0, 1, n))
        df = pd.DataFrame({"date": dates, "value": values})

        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
        assert result.success is True
        if result.flatRisk:
            assert result.flatRisk.riskLevel is not None


class TestConfidenceInterval:
    def test_confidence_intervals_exist(self):
        df = _make_ts(200)
        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
        assert result.success is True
        assert len(result.lower95) == 30
        assert len(result.upper95) == 30
        assert np.all(result.lower95 <= result.upper95)

    def test_confidence_intervals_widen(self):
        df = _make_ts(200)
        fx = ForecastX()
        result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
        if result.success:
            widths = result.upper95 - result.lower95
            # 허용오차 1e-10 (앙상블의 경우 일정 폭 신뢰구간 가능)
            assert widths[-1] >= widths[0] - 1e-10, "Confidence intervals should widen over time"
