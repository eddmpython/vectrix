"""
TimesFM Forecaster

Foundation model wrapper for Google TimesFM 2.5.
Zero-shot time series forecasting with pre-trained transformer.

Optional dependency: timesfm, torch
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False


class TimesFMForecaster:
    """
    Google TimesFM 2.5 foundation model forecaster.

    Pre-trained 200M parameter transformer for zero-shot time series forecasting.
    Supports quantile forecasts and optional covariates.

    Requires: pip install timesfm torch

    Parameters
    ----------
    modelId : str
        HuggingFace model identifier.
    maxContext : int
        Maximum context length (up to 16384).
    maxHorizon : int
        Maximum prediction horizon (up to 1024).
    normalizeInputs : bool
        Whether to normalize input time series.
    """

    def __init__(
        self,
        modelId: str = "google/timesfm-2.5-200m-pytorch",
        maxContext: int = 1024,
        maxHorizon: int = 256,
        normalizeInputs: bool = True,
    ):
        if not TIMESFM_AVAILABLE:
            raise ImportError(
                "timesfm is required. "
                "Install: pip install git+https://github.com/google-research/timesfm.git"
            )

        self.modelId = modelId
        self.maxContext = maxContext
        self.maxHorizon = maxHorizon
        self.normalizeInputs = normalizeInputs
        self._model = None
        self._y = None
        self.fitted = False

    def _loadModel(self):
        if self._model is not None:
            return

        self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.modelId,
            torch_compile=False,
        )

        self._model.compile(
            timesfm.ForecastConfig(
                max_context=self.maxContext,
                max_horizon=self.maxHorizon,
                normalize_inputs=self.normalizeInputs,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                fix_quantile_crossing=True,
            )
        )

    def fit(self, y: np.ndarray) -> 'TimesFMForecaster':
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
            Point forecasts.
        lower : np.ndarray
            Lower prediction interval (10th percentile).
        upper : np.ndarray
            Upper prediction interval (90th percentile).
        """
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self._loadModel()

        pointForecast, quantileForecast = self._model.forecast(
            horizon=steps,
            inputs=[self._y],
        )

        predictions = pointForecast[0]

        if quantileForecast is not None and len(quantileForecast.shape) == 3:
            nQuantiles = quantileForecast.shape[2]
            lower = quantileForecast[0, :, 1] if nQuantiles > 1 else predictions
            upper = quantileForecast[0, :, -1] if nQuantiles > 1 else predictions
        else:
            sigma = np.std(np.diff(self._y[-min(30, len(self._y)):]))
            margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
            lower = predictions - margin
            upper = predictions + margin

        return predictions, lower, upper

    def predictQuantiles(self, steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate point and quantile forecasts.

        Parameters
        ----------
        steps : int
            Number of future steps.

        Returns
        -------
        pointForecast : np.ndarray
            Shape (steps,).
        quantileForecast : np.ndarray
            Shape (steps, nQuantiles). Quantiles include mean + 10th~90th percentiles.
        """
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self._loadModel()

        pointForecast, quantileForecast = self._model.forecast(
            horizon=steps,
            inputs=[self._y],
        )

        return pointForecast[0], quantileForecast[0]

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
        self._loadModel()

        inputs = [np.asarray(s, dtype=np.float64) for s in seriesList]

        pointForecast, quantileForecast = self._model.forecast(
            horizon=steps,
            inputs=inputs,
        )

        results = []
        for i in range(len(seriesList)):
            pred = pointForecast[i]
            if quantileForecast is not None and len(quantileForecast.shape) == 3:
                nQ = quantileForecast.shape[2]
                lo = quantileForecast[i, :, 1] if nQ > 1 else pred
                hi = quantileForecast[i, :, -1] if nQ > 1 else pred
            else:
                sigma = np.std(np.diff(seriesList[i][-30:]))
                margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
                lo = pred - margin
                hi = pred + margin
            results.append((pred, lo, hi))

        return results

    def predictWithCovariates(
        self,
        steps: int,
        dynamicNumerical: Optional[Dict[str, List[List[float]]]] = None,
        dynamicCategorical: Optional[Dict[str, List[List[str]]]] = None,
        staticNumerical: Optional[Dict[str, List[List[float]]]] = None,
        staticCategorical: Optional[Dict[str, List[List[str]]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts with exogenous covariates.

        Requires: pip install timesfm[xreg]

        Parameters
        ----------
        steps : int
            Number of future steps.
        dynamicNumerical : dict, optional
            Dynamic numerical covariates (context + horizon length).
        dynamicCategorical : dict, optional
            Dynamic categorical covariates (context + horizon length).
        staticNumerical : dict, optional
            Static numerical covariates (one value per series).
        staticCategorical : dict, optional
            Static categorical covariates (one value per series).

        Returns
        -------
        predictions, lower, upper : np.ndarray
        """
        if not self.fitted or self._y is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self._loadModel()

        freq = [0] * len(self._y)

        pointForecast, quantileForecast = self._model.forecast_with_covariates(
            inputs=[self._y.tolist()],
            dynamic_numerical_covariates=dynamicNumerical or {},
            dynamic_categorical_covariates=dynamicCategorical or {},
            static_numerical_covariates=staticNumerical or {},
            static_categorical_covariates=staticCategorical or {},
            freq=freq,
            xreg_mode="xreg + timesfm",
            normalize_xreg_target_per_input=True,
        )

        predictions = np.array(pointForecast[0][:steps])

        if quantileForecast is not None and len(np.array(quantileForecast).shape) >= 3:
            qArr = np.array(quantileForecast)
            lower = qArr[0, :steps, 1]
            upper = qArr[0, :steps, -1]
        else:
            sigma = np.std(np.diff(self._y[-min(30, len(self._y)):]))
            margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
            lower = predictions - margin
            upper = predictions + margin

        return predictions, lower, upper
