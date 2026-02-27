import numpy as np
import pandas as pd
import pytest

from forecastx.engine.baselines import (
    NaiveModel,
    SeasonalNaiveModel,
    MeanModel,
    RandomWalkDrift,
    WindowAverage,
)
from forecastx.engine.ets import ETSModel
from forecastx.engine.arima import ARIMAModel
from forecastx.engine.theta import ThetaModel, OptimizedTheta
from forecastx.engine.ces import CESModel
from forecastx.engine.dot import DynamicOptimizedTheta
from forecastx.engine.tbats import TBATS
from forecastx.engine.logistic import LogisticGrowthModel
from forecastx.engine.croston import CrostonClassic, CrostonSBA, CrostonTSB
from forecastx.engine.garch import GARCHModel, EGARCHModel, GJRGARCHModel


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def seasonalData7(rng):
    n = 200
    t = np.arange(n, dtype=np.float64)
    return 100.0 + 0.3 * t + 15.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, n)


@pytest.fixture
def seasonalData12(rng):
    n = 144
    t = np.arange(n, dtype=np.float64)
    return 100.0 + 0.2 * t + 20.0 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)


@pytest.fixture
def trendData(rng):
    return 100.0 + 0.5 * np.arange(200) + rng.normal(0, 2, 200)


@pytest.fixture
def constantData():
    return np.ones(100) * 50.0


@pytest.fixture
def intermittentData(rng):
    data = np.zeros(200)
    idx = rng.choice(200, size=30, replace=False)
    data[idx] = rng.poisson(5, 30).astype(np.float64) + 1.0
    return data


@pytest.fixture
def shortData(rng):
    return 50.0 + rng.normal(0, 5, 20)


@pytest.fixture
def volatileData(rng):
    n = 500
    returns = np.zeros(n)
    sigma2 = np.ones(n)
    for t in range(1, n):
        sigma2[t] = 0.05 + 0.1 * returns[t - 1] ** 2 + 0.85 * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return returns


@pytest.fixture
def tsDataframe(rng):
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    values = 100.0 + np.linspace(0, 50, n) + rng.normal(0, 3, n)
    return pd.DataFrame({"date": dates, "value": values})


BASELINE_FACTORIES = [
    pytest.param(lambda: NaiveModel(), id="NaiveModel"),
    pytest.param(lambda: SeasonalNaiveModel(period=7), id="SeasonalNaiveModel"),
    pytest.param(lambda: MeanModel(), id="MeanModel"),
    pytest.param(lambda: RandomWalkDrift(), id="RandomWalkDrift"),
    pytest.param(lambda: WindowAverage(window=7), id="WindowAverage"),
]

CORE_FACTORIES = [
    pytest.param(
        lambda: ETSModel(errorType="A", trendType="A", seasonalType="A", period=7),
        id="ETSModel_AAA",
    ),
    pytest.param(lambda: ARIMAModel(order=(1, 1, 1)), id="ARIMAModel_111"),
    pytest.param(lambda: ThetaModel(period=7), id="ThetaModel"),
    pytest.param(lambda: OptimizedTheta(period=7), id="OptimizedTheta"),
    pytest.param(lambda: CESModel(form="S", period=7), id="CESModel_S"),
    pytest.param(
        lambda: DynamicOptimizedTheta(period=7), id="DynamicOptimizedTheta"
    ),
    pytest.param(lambda: TBATS(periods=[7]), id="TBATS"),
    pytest.param(lambda: LogisticGrowthModel(cap=300.0), id="LogisticGrowthModel"),
]

SEASONAL_FACTORIES = BASELINE_FACTORIES + CORE_FACTORIES

INTERMITTENT_FACTORIES = [
    pytest.param(lambda: CrostonClassic(), id="CrostonClassic"),
    pytest.param(lambda: CrostonSBA(), id="CrostonSBA"),
    pytest.param(lambda: CrostonTSB(), id="CrostonTSB"),
]

GARCH_FACTORIES = [
    pytest.param(lambda: GARCHModel(), id="GARCHModel"),
    pytest.param(lambda: EGARCHModel(), id="EGARCHModel"),
    pytest.param(lambda: GJRGARCHModel(), id="GJRGARCHModel"),
]
