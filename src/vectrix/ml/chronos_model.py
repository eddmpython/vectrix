"""
Chronos Forecaster

Foundation model wrapper for Amazon Chronos / Chronos-Bolt / Chronos-2.
Zero-shot time series forecasting with pre-trained transformer models.

Optional dependency: chronos-forecasting (>= 2.0), torch
"""

from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    from chronos import BaseChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False


class ChronosForecaster:
    """
    Amazon Chronos foundation model forecaster.

    Pre-trained transformer for zero-shot time series forecasting.
    Supports Chronos-T5, Chronos-Bolt, and Chronos-2 architectures.

    Requires: pip install chronos-forecasting torch

    Available models:
        - amazon/chronos-t5-tiny (8M)
        - amazon/chronos-t5-small (46M)
        - amazon/chronos-t5-base (200M)
        - amazon/chronos-t5-large (710M)
        - amazon/chronos-bolt-tiny (9M)
        - amazon/chronos-bolt-small (48M)
        - amazon/chronos-bolt-base (205M)
        - amazon/chronos-2 (120M, multivariate + covariates)

    Parameters
    ----------
    modelId : str
        HuggingFace model identifier.
    device : str
        Device for inference ('cpu', 'cuda', 'mps').
    torchDtype : str
        Precision ('float32', 'bfloat16', 'float16').
    quantileLevels : list of float
        Quantile levels for prediction intervals.
    """

    def __init__(
        self,
        modelId: str = "amazon/chronos-bolt-small",
        device: str = "cpu",
        torchDtype: str = "float32",
        quantileLevels: Optional[List[float]] = None,
    ):
        if not CHRONOS_AVAILABLE:
            raise ImportError(
                "chronos-forecasting and torch are required. "
                "Install: pip install chronos-forecasting torch"
            )

        self.modelId = modelId
        self.device = device
        self.torchDtype = torchDtype
        self.quantileLevels = quantileLevels or [0.025, 0.5, 0.975]
        self._pipeline = None
        self._y = None
        self.fitted = False

    def _loadPipeline(self):
        if self._pipeline is not None:
            return

        dtypeMap = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        dtype = dtypeMap.get(self.torchDtype, torch.float32)

        self._pipeline = BaseChronosPipeline.from_pretrained(
            self.modelId,
            device_map=self.device,
            torch_dtype=dtype,
        )

    def fit(self, y: np.ndarray) -> 'ChronosForecaster':
        """
        Store context data for zero-shot prediction.

        Foundation models do not train; fit() only stores the context.

        Parameters
        ----------
        y : np.ndarray
            Historical time series values.

        Returns
        -------
        self
        """
        self._y = np.asarray(y, dtype=np.float64).copy()
        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate zero-shot forecasts.

        Parameters
        ----------
        steps : int
            Number of future steps to forecast.

        Returns
        -------
        predictions : np.ndarray
            Point forecasts (median).
        lower : np.ndarray
            Lower prediction interval (2.5th percentile).
        upper : np.ndarray
            Upper prediction interval (97.5th percentile).
        """
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self._loadPipeline()

        context = torch.tensor(self._y, dtype=torch.float32).unsqueeze(0)

        quantiles, mean = self._pipeline.predict_quantiles(
            context=context,
            prediction_length=steps,
            quantile_levels=self.quantileLevels,
        )

        quantiles = quantiles.numpy()
        mean = mean.numpy()

        lowerIdx = 0
        medianIdx = len(self.quantileLevels) // 2
        upperIdx = len(self.quantileLevels) - 1

        predictions = quantiles[0, :, medianIdx]
        lower = quantiles[0, :, lowerIdx]
        upper = quantiles[0, :, upperIdx]

        return predictions, lower, upper

    def predictQuantiles(
        self,
        steps: int,
        quantileLevels: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate quantile forecasts.

        Parameters
        ----------
        steps : int
            Number of future steps.
        quantileLevels : list of float, optional
            Custom quantile levels (default: [0.1, 0.25, 0.5, 0.75, 0.9]).

        Returns
        -------
        quantiles : np.ndarray
            Shape (steps, nQuantiles).
        """
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self._loadPipeline()

        levels = quantileLevels or [0.1, 0.25, 0.5, 0.75, 0.9]
        context = torch.tensor(self._y, dtype=torch.float32).unsqueeze(0)

        quantiles, _ = self._pipeline.predict_quantiles(
            context=context,
            prediction_length=steps,
            quantile_levels=levels,
        )

        return quantiles[0].numpy()

    def predictBatch(
        self,
        seriesList: List[np.ndarray],
        steps: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Batch prediction for multiple time series.

        Parameters
        ----------
        seriesList : list of np.ndarray
            Multiple time series to forecast.
        steps : int
            Number of future steps.

        Returns
        -------
        results : list of (predictions, lower, upper)
        """
        self._loadPipeline()

        contexts = [
            torch.tensor(s, dtype=torch.float32) for s in seriesList
        ]

        quantiles, mean = self._pipeline.predict_quantiles(
            context=contexts,
            prediction_length=steps,
            quantile_levels=self.quantileLevels,
        )

        quantiles = quantiles.numpy()
        results = []
        lowerIdx = 0
        medianIdx = len(self.quantileLevels) // 2
        upperIdx = len(self.quantileLevels) - 1

        for i in range(len(seriesList)):
            pred = quantiles[i, :, medianIdx]
            lo = quantiles[i, :, lowerIdx]
            hi = quantiles[i, :, upperIdx]
            results.append((pred, lo, hi))

        return results
