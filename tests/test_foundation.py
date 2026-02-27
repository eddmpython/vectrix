"""
Foundation Model wrapper tests.

Tests import availability flags and basic interface contracts.
Actual model inference is skipped unless dependencies are installed.
"""

import numpy as np
import pytest

from vectrix.ml import CHRONOS_AVAILABLE, NEURALFORECAST_AVAILABLE, TIMESFM_AVAILABLE
from vectrix.ml.chronos_model import ChronosForecaster
from vectrix.ml.neuralforecast_model import NBEATSForecaster, NeuralForecaster, NHITSForecaster, TFTForecaster
from vectrix.ml.timesfm_model import TimesFMForecaster


class TestChronosForecaster:
    def test_importFlag(self):
        assert isinstance(CHRONOS_AVAILABLE, bool)

    def test_constructorWithoutDependency(self):
        if CHRONOS_AVAILABLE:
            pytest.skip("chronos is installed")
        with pytest.raises(ImportError):
            ChronosForecaster()

    @pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="chronos not installed")
    def test_fitStoresContext(self):
        model = ChronosForecaster()
        y = np.random.randn(100)
        model.fit(y)
        assert model.fitted
        assert model._y is not None
        assert len(model._y) == 100

    @pytest.mark.skipif(not CHRONOS_AVAILABLE, reason="chronos not installed")
    def test_predictWithoutFitRaises(self):
        model = ChronosForecaster()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(10)

    def test_interfaceConsistency(self):
        assert hasattr(ChronosForecaster, 'fit')
        assert hasattr(ChronosForecaster, 'predict')
        assert hasattr(ChronosForecaster, 'predictQuantiles')
        assert hasattr(ChronosForecaster, 'predictBatch')


class TestTimesFMForecaster:
    def test_importFlag(self):
        assert isinstance(TIMESFM_AVAILABLE, bool)

    def test_constructorWithoutDependency(self):
        if TIMESFM_AVAILABLE:
            pytest.skip("timesfm is installed")
        with pytest.raises(ImportError):
            TimesFMForecaster()

    @pytest.mark.skipif(not TIMESFM_AVAILABLE, reason="timesfm not installed")
    def test_fitStoresContext(self):
        model = TimesFMForecaster()
        y = np.random.randn(100)
        model.fit(y)
        assert model.fitted
        assert model._y is not None
        assert len(model._y) == 100

    @pytest.mark.skipif(not TIMESFM_AVAILABLE, reason="timesfm not installed")
    def test_predictWithoutFitRaises(self):
        model = TimesFMForecaster()
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(10)

    def test_interfaceConsistency(self):
        assert hasattr(TimesFMForecaster, 'fit')
        assert hasattr(TimesFMForecaster, 'predict')
        assert hasattr(TimesFMForecaster, 'predictQuantiles')
        assert hasattr(TimesFMForecaster, 'predictBatch')
        assert hasattr(TimesFMForecaster, 'predictWithCovariates')


class TestNeuralForecaster:
    def test_importFlag(self):
        assert isinstance(NEURALFORECAST_AVAILABLE, bool)

    def test_constructorWithoutDependency(self):
        if NEURALFORECAST_AVAILABLE:
            pytest.skip("neuralforecast is installed")
        with pytest.raises(ImportError):
            NeuralForecaster()

    def test_invalidModelRaises(self):
        if not NEURALFORECAST_AVAILABLE:
            pytest.skip("neuralforecast not installed")
        with pytest.raises(ValueError, match="Unsupported"):
            NeuralForecaster(model="invalid_model")

    def test_interfaceConsistency(self):
        assert hasattr(NeuralForecaster, 'fit')
        assert hasattr(NeuralForecaster, 'predict')
        assert hasattr(NeuralForecaster, 'predictQuantiles')

    def test_convenienceClasses(self):
        assert issubclass(NBEATSForecaster, NeuralForecaster)
        assert issubclass(NHITSForecaster, NeuralForecaster)
        assert issubclass(TFTForecaster, NeuralForecaster)


class TestTopLevelExports:
    def test_chronosExported(self):
        from vectrix import CHRONOS_AVAILABLE, ChronosForecaster
        assert ChronosForecaster is not None

    def test_timesfmExported(self):
        from vectrix import TIMESFM_AVAILABLE, TimesFMForecaster
        assert TimesFMForecaster is not None

    def test_neuralExported(self):
        from vectrix import NEURALFORECAST_AVAILABLE, NBEATSForecaster, NHITSForecaster, TFTForecaster
        assert NBEATSForecaster is not None
        assert NHITSForecaster is not None
        assert TFTForecaster is not None
