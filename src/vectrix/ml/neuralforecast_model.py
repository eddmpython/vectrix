"""
NeuralForecast Model Wrappers

Foundation deep learning models for time series forecasting.
Wraps Nixtla NeuralForecast: NBEATS, NHITS, TFT.

Optional dependency: neuralforecast, torch
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import NBEATS, NHITS, TFT
    NEURALFORECAST_AVAILABLE = True
except ImportError:
    NEURALFORECAST_AVAILABLE = False


class NeuralForecaster:
    """
    NeuralForecast-based time series forecaster.

    Wraps NBEATS, NHITS, or TFT with Vectrix's fit()/predict() interface.

    Requires: pip install neuralforecast

    Parameters
    ----------
    model : str
        Model type: 'nbeats', 'nhits', or 'tft'.
    horizon : int
        Forecast horizon.
    inputSize : int, optional
        Autoregressive input size (default: 2 * horizon).
    maxSteps : int
        Maximum training steps.
    level : list of int, optional
        Prediction interval levels (default: [90]).
    freq : str
        Pandas frequency string.
    modelParams : dict, optional
        Additional model parameters.
    """

    SUPPORTED_MODELS = {"nbeats", "nhits", "tft"}

    def __init__(
        self,
        model: str = "nhits",
        horizon: int = 12,
        inputSize: Optional[int] = None,
        maxSteps: int = 200,
        level: Optional[List[int]] = None,
        freq: str = "D",
        modelParams: Optional[Dict[str, Any]] = None,
    ):
        if not NEURALFORECAST_AVAILABLE:
            raise ImportError(
                "neuralforecast is required. Install: pip install neuralforecast"
            )

        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. Choose from {self.SUPPORTED_MODELS}")

        self.modelType = model
        self.horizon = horizon
        self.inputSize = inputSize or 2 * horizon
        self.maxSteps = maxSteps
        self.level = level or [90]
        self.freq = freq
        self.modelParams = modelParams or {}
        self._nf = None
        self._y = None
        self.fitted = False

    def _buildModel(self):
        baseParams = {
            "h": self.horizon,
            "input_size": self.inputSize,
            "max_steps": self.maxSteps,
            "loss": MQLoss(level=self.level),
        }
        baseParams.update(self.modelParams)

        modelClass = {"nbeats": NBEATS, "nhits": NHITS, "tft": TFT}[self.modelType]
        return modelClass(**baseParams)

    def _toDataFrame(self, y: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "unique_id": "s1",
            "ds": pd.date_range("2000-01-01", periods=len(y), freq=self.freq),
            "y": y.astype(np.float64),
        })

    def fit(self, y: np.ndarray) -> 'NeuralForecaster':
        """
        Train the neural network on time series data.

        Parameters
        ----------
        y : np.ndarray
            Historical time series values.

        Returns
        -------
        self
        """
        self._y = np.asarray(y, dtype=np.float64).copy()
        df = self._toDataFrame(self._y)

        model = self._buildModel()
        self._nf = NeuralForecast(models=[model], freq=self.freq)
        self._nf.fit(df=df)

        self.fitted = True
        return self

    def predict(self, steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts.

        Parameters
        ----------
        steps : int, optional
            Must equal horizon (neural models have fixed output size).

        Returns
        -------
        predictions : np.ndarray
            Point forecasts (median).
        lower : np.ndarray
            Lower prediction interval.
        upper : np.ndarray
            Upper prediction interval.
        """
        if not self.fitted or self._nf is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if steps is not None and steps != self.horizon:
            raise ValueError(
                f"Neural models have fixed horizon={self.horizon}. "
                f"Requested steps={steps} does not match."
            )

        preds = self._nf.predict()

        modelName = self.modelType.upper()
        if self.modelType == "nhits":
            modelName = "NHITS"
        elif self.modelType == "nbeats":
            modelName = "NBEATS"
        elif self.modelType == "tft":
            modelName = "TFT"

        medianCol = f"{modelName}-median"
        loCol = f"{modelName}-lo-{self.level[0]}"
        hiCol = f"{modelName}-hi-{self.level[0]}"

        predictions = preds[medianCol].to_numpy()
        lower = preds[loCol].to_numpy() if loCol in preds.columns else predictions
        upper = preds[hiCol].to_numpy() if hiCol in preds.columns else predictions

        return predictions, lower, upper

    def predictQuantiles(
        self,
        quantileLevels: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate quantile forecasts.

        Parameters
        ----------
        quantileLevels : list of float, optional
            Custom quantile levels.

        Returns
        -------
        quantiles : np.ndarray
            Shape (horizon, nQuantiles).
        """
        if not self.fitted or self._nf is None:
            raise ValueError("Model not fitted. Call fit() first.")

        levels = quantileLevels or [0.1, 0.25, 0.5, 0.75, 0.9]

        preds = self._nf.predict(quantiles=levels)

        modelName = {"nbeats": "NBEATS", "nhits": "NHITS", "tft": "TFT"}[self.modelType]

        result = np.zeros((self.horizon, len(levels)))
        for i, q in enumerate(levels):
            col = f"{modelName}_ql{q}"
            if col in preds.columns:
                result[:, i] = preds[col].to_numpy()

        return result


class NBEATSForecaster(NeuralForecaster):
    """NBEATS forecaster (convenience wrapper)."""

    def __init__(self, horizon: int = 12, **kwargs):
        super().__init__(model="nbeats", horizon=horizon, **kwargs)


class NHITSForecaster(NeuralForecaster):
    """NHITS forecaster (convenience wrapper)."""

    def __init__(self, horizon: int = 12, **kwargs):
        super().__init__(model="nhits", horizon=horizon, **kwargs)


class TFTForecaster(NeuralForecaster):
    """Temporal Fusion Transformer forecaster (convenience wrapper)."""

    def __init__(self, horizon: int = 12, **kwargs):
        super().__init__(model="tft", horizon=horizon, **kwargs)
