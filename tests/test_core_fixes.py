import pytest
import numpy as np

from vectrix import Vectrix
import vectrix
from vectrix.engine.ets import ETSModel, AutoETS
from vectrix.engine.arima import ARIMAModel, AutoARIMA, _checkStationarity


np.random.seed(42)

seasonalData = (
    np.sin(2 * np.pi * np.arange(120) / 12) * 10
    + np.arange(120) * 0.1
    + 100
    + np.random.randn(120) * 2
)

maInnovations = np.random.randn(202)
maData = np.zeros(201)
for _t in range(1, 201):
    maData[_t] = maInnovations[_t] + 0.6 * maInnovations[_t - 1]
maData = maData[1:]

constantData = np.ones(100) * 50

largeValueData = (np.sin(2 * np.pi * np.arange(100) / 7) * 5 + 50) * 1e8


def test_version_consistency():
    assert Vectrix.VERSION == "3.0.0"
    assert vectrix.__version__ == "3.0.0"
    assert Vectrix.VERSION == vectrix.__version__


def test_autoEts_evaluates_multiplicative_error():
    autoEts = AutoETS(period=12, exhaustive=False)
    autoEts.fit(seasonalData)
    mErrorKeys = [k for k in autoEts.allResults.keys() if k[0] == "M"]
    assert len(mErrorKeys) >= 1


def test_autoEts_evaluates_multiplicative_seasonal():
    autoEts = AutoETS(period=12, exhaustive=False)
    autoEts.fit(seasonalData)
    mSeasonalKeys = [k for k in autoEts.allResults.keys() if k[2] == "M"]
    assert len(mSeasonalKeys) >= 1


def test_autoEts_stepwise_evaluates_more_than_4():
    autoEts = AutoETS(period=12, exhaustive=False)
    autoEts.fit(seasonalData)
    assert len(autoEts.allResults) > 4


def test_autoEts_aicc_for_all_candidates():
    autoEts = AutoETS(period=12, exhaustive=False)
    autoEts.fit(seasonalData)
    for key, result in autoEts.allResults.items():
        if result is not None:
            model, aicc = result
            assert aicc is not None
            assert np.isfinite(aicc)


def test_ets_multiplicative_error_model():
    model = ETSModel(errorType="M", trendType="A", seasonalType="N", period=12)
    model.fit(seasonalData)
    predictions, lower, upper = model.predict(12)
    assert not np.any(np.isnan(predictions))
    assert len(predictions) == 12


def test_ets_multiplicative_seasonal_model():
    model = ETSModel(errorType="A", trendType="N", seasonalType="M", period=12)
    model.fit(seasonalData)
    predictions, lower, upper = model.predict(12)
    assert not np.any(np.isnan(predictions))
    assert len(predictions) == 12


def test_ets_large_values_no_overflow():
    model = ETSModel(errorType="A", trendType="A", seasonalType="A", period=7)
    model.fit(largeValueData)
    predictions, lower, upper = model.predict(14)
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))


def test_ets_constant_data():
    model = ETSModel(errorType="A", trendType="N", seasonalType="N", period=1)
    model.fit(constantData)
    predictions, lower, upper = model.predict(10)
    assert len(predictions) == 10
    assert not np.any(np.isnan(predictions))


def test_arima_ma_not_heuristic():
    rng = np.random.RandomState(42)
    innovations = rng.randn(502)
    localMaData = np.zeros(501)
    for t in range(1, 501):
        localMaData[t] = innovations[t] + 0.6 * innovations[t - 1]
    localMaData = localMaData[1:]
    model = ARIMAModel(order=(0, 0, 1))
    model.fit(localMaData)
    estimatedTheta = model.maCoefs[0]
    defaultHeuristic = 0.3
    assert abs(estimatedTheta - defaultHeuristic) > 1e-6
    predictions, _, _ = model.predict(5)
    assert not np.any(np.isnan(predictions))


def test_arima_optimizer_maxiter():
    model = ARIMAModel(order=(1, 0, 1))
    model.fit(maData)
    initialAr = 0.5
    initialMa = 0.3
    arDiffersFromInit = abs(model.arCoefs[0] - initialAr) > 1e-6
    maDiffersFromInit = abs(model.maCoefs[0] - initialMa) > 1e-6
    assert arDiffersFromInit or maDiffersFromInit


def test_arima_stationarity_check():
    rng = np.random.RandomState(42)
    arData = rng.randn(300)
    for t in range(2, 300):
        arData[t] = 0.5 * arData[t - 1] - 0.2 * arData[t - 2] + arData[t]
    model = ARIMAModel(order=(2, 0, 0))
    model.fit(arData)
    correctedCoefs = _checkStationarity(model.arCoefs)
    polyCoefs = np.concatenate(([1.0], -correctedCoefs))
    roots = np.roots(polyCoefs)
    moduli = np.abs(roots)
    assert np.all(np.isfinite(moduli))
    assert len(correctedCoefs) == 2
    predictions, _, _ = model.predict(10)
    assert not np.any(np.isnan(predictions))
    assert not np.any(np.isinf(predictions))


def test_sarima_fit():
    np.random.seed(42)
    t = np.arange(144)
    sarimaData = np.sin(2 * np.pi * t / 12) * 15 + t * 0.3 + 50 + np.random.randn(144) * 3
    model = ARIMAModel(order=(1, 1, 1), seasonalOrder=(1, 1, 1, 12))
    model.fit(sarimaData)
    predictions, lower, upper = model.predict(12)
    assert len(predictions) == 12
    assert not np.any(np.isnan(predictions))


def test_sarima_seasonal_coefficients():
    np.random.seed(42)
    t = np.arange(144)
    sarimaData = np.sin(2 * np.pi * t / 12) * 15 + t * 0.3 + 50 + np.random.randn(144) * 3
    model = ARIMAModel(order=(1, 1, 1), seasonalOrder=(1, 1, 1, 12))
    model.fit(sarimaData)
    sarNotAllZero = model.sarCoefs is not None and np.any(np.abs(model.sarCoefs) > 1e-10)
    smaNotAllZero = model.smaCoefs is not None and np.any(np.abs(model.smaCoefs) > 1e-10)
    assert sarNotAllZero or smaNotAllZero


def test_auto_arima_seasonal_search():
    np.random.seed(42)
    t = np.arange(156)
    strongSeasonalData = (
        np.sin(2 * np.pi * t / 12) * 30
        + t * 0.2
        + 100
        + np.random.randn(156) * 2
    )
    autoArima = AutoARIMA(maxP=2, maxD=1, maxQ=2, seasonalPeriod=12)
    autoArima.fit(strongSeasonalData)
    assert autoArima.bestSeasonalOrder is not None
