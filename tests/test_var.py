"""
VAR and VECM model tests.
"""

import numpy as np
import pytest

from vectrix.engine.var import VARModel, VECMModel


class TestVARModel:
    def test_fitPredict2d(self):
        rng = np.random.default_rng(42)
        T = 100
        Y = np.zeros((T, 2))
        Y[0] = [1.0, 2.0]
        for t in range(1, T):
            Y[t, 0] = 0.5 * Y[t - 1, 0] + 0.2 * Y[t - 1, 1] + rng.normal(0, 0.1)
            Y[t, 1] = 0.3 * Y[t - 1, 0] + 0.4 * Y[t - 1, 1] + rng.normal(0, 0.1)

        model = VARModel(maxLag=3)
        model.fit(Y)

        assert model.fitted
        assert model.order >= 1

        pred, lo, hi = model.predict(10)
        assert pred.shape == (10, 2)
        assert lo.shape == (10, 2)
        assert hi.shape == (10, 2)
        assert np.all(lo < hi)

    def test_fitPredict3d(self):
        rng = np.random.default_rng(42)
        T = 150
        Y = rng.normal(0, 1, size=(T, 3))
        for t in range(1, T):
            Y[t] = 0.3 * Y[t - 1] + rng.normal(0, 0.2, size=3)

        model = VARModel(maxLag=4)
        model.fit(Y)

        pred, lo, hi = model.predict(5)
        assert pred.shape == (5, 3)

    def test_unfittedRaises(self):
        model = VARModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(5)

    def test_lagSelection(self):
        rng = np.random.default_rng(42)
        T = 200
        Y = np.zeros((T, 2))
        for t in range(2, T):
            Y[t, 0] = 0.5 * Y[t - 1, 0] + 0.3 * Y[t - 2, 0] + rng.normal(0, 0.1)
            Y[t, 1] = 0.4 * Y[t - 1, 1] + rng.normal(0, 0.1)

        model = VARModel(maxLag=5, criterion="bic")
        model.fit(Y)
        assert model.order >= 1

    def test_grangerCausality(self):
        rng = np.random.default_rng(42)
        T = 200
        Y = np.zeros((T, 2))
        for t in range(1, T):
            Y[t, 0] = 0.8 * Y[t - 1, 0] + rng.normal(0, 0.1)
            Y[t, 1] = 0.5 * Y[t - 1, 1] + 0.4 * Y[t - 1, 0] + rng.normal(0, 0.1)

        model = VARModel()
        result = model.grangerCausality(Y, cause=0, effect=1, maxLag=3)
        assert "fStat" in result
        assert "pValue" in result
        assert result["pValue"] < 0.05


class TestVECMModel:
    def test_fitPredict(self):
        rng = np.random.default_rng(42)
        T = 150
        e1 = np.cumsum(rng.normal(0, 1, T))
        e2 = e1 + rng.normal(0, 0.5, T)
        Y = np.column_stack([e1, e2])

        model = VECMModel(maxLag=3)
        model.fit(Y)

        assert model.fitted
        pred, lo, hi = model.predict(10)
        assert pred.shape == (10, 2)
        assert np.all(lo < hi)

    def test_unfittedRaises(self):
        model = VECMModel()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(5)

    def test_shortData(self):
        Y = np.random.randn(5, 2)
        model = VECMModel(maxLag=2)
        model.fit(Y)
        assert model.fitted

        pred, _, _ = model.predict(3)
        assert pred.shape == (3, 2)
