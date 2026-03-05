"""
Test refit() contract: refit must produce same output structure as fresh fit+predict.

Models with refit: AutoETS, AutoARIMA, OptimizedTheta, AutoMSTL, ETSModel
"""

import numpy as np
import pytest

from vectrix.engine.registry import createModel, getRegistry


REFIT_MODELS = [
    mid for mid, spec in getRegistry().items()
    if mid in ['auto_ets', 'auto_arima', 'theta', 'auto_mstl', 'ets_aan', 'ets_aaa']
]

STABLE_REFIT_MODELS = REFIT_MODELS

STEPS = 12
PERIOD = 12


def _makeSeasonal(n=120, period=12, seed=42):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = 100 + 0.5 * t
    season = 10 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, 2, n)
    return trend + season + noise


@pytest.fixture
def data():
    return _makeSeasonal()


@pytest.fixture
def extendedData():
    return _makeSeasonal(n=140)


@pytest.mark.parametrize("modelId", REFIT_MODELS)
def test_refit_returns_self(modelId, data):
    model = createModel(modelId, PERIOD)
    model.fit(data)
    result = model.refit(data)
    assert result is model


@pytest.mark.parametrize("modelId", REFIT_MODELS)
def test_refit_output_shape(modelId, data, extendedData):
    model = createModel(modelId, PERIOD)
    model.fit(data)
    pred1, lo1, hi1 = model.predict(STEPS)

    model.refit(extendedData)
    pred2, lo2, hi2 = model.predict(STEPS)

    assert pred2.shape == (STEPS,), f"{modelId}: pred shape mismatch"
    assert lo2.shape == (STEPS,), f"{modelId}: lower shape mismatch"
    assert hi2.shape == (STEPS,), f"{modelId}: upper shape mismatch"


@pytest.mark.parametrize("modelId", REFIT_MODELS)
def test_refit_output_finite(modelId, extendedData):
    model = createModel(modelId, PERIOD)
    model.fit(extendedData[:100])
    model.refit(extendedData)
    pred, lo, hi = model.predict(STEPS)

    assert np.all(np.isfinite(pred)), f"{modelId}: non-finite predictions"
    assert np.all(np.isfinite(lo)), f"{modelId}: non-finite lower"
    assert np.all(np.isfinite(hi)), f"{modelId}: non-finite upper"


@pytest.mark.parametrize("modelId", REFIT_MODELS)
def test_refit_ci_ordering(modelId, data):
    model = createModel(modelId, PERIOD)
    model.fit(data)
    model.refit(data)
    pred, lo, hi = model.predict(STEPS)

    assert np.all(lo <= pred + 1e-6), f"{modelId}: lower > pred"
    assert np.all(pred <= hi + 1e-6), f"{modelId}: pred > upper"


@pytest.mark.parametrize("modelId", STABLE_REFIT_MODELS)
def test_refit_same_data_similar_output(modelId, data):
    model = createModel(modelId, PERIOD)
    model.fit(data)
    pred1, _, _ = model.predict(STEPS)

    model.refit(data)
    pred2, _, _ = model.predict(STEPS)

    maxDiff = np.max(np.abs(pred1 - pred2))
    meanVal = np.mean(np.abs(pred1))
    relDiff = maxDiff / (meanVal + 1e-10)
    assert relDiff < 0.05, (
        f"{modelId}: refit on same data diverges by {relDiff:.2%} "
        f"(max abs diff={maxDiff:.4f})"
    )
