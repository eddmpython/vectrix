"""
Time Series Forecast Pipeline

A pipeline system that chains preprocessing -> forecasting -> postprocessing.
Extends the sklearn Pipeline pattern for time series forecasting.

Examples
--------
>>> from vectrix.pipeline import ForecastPipeline, Differencer, LogTransformer, Scaler
>>> pipe = ForecastPipeline([
...     ('log', LogTransformer()),
...     ('scale', Scaler()),
...     ('forecast', Vectrix()),
... ])
>>> result = pipe.fit(y).predict(12)
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseTransformer:
    """
    Base class for time series transformers

    Parent class for all pre/post-processing transformers.
    Provides fit -> transform -> inverseTransform interface.
    """

    def fit(self, y: np.ndarray) -> 'BaseTransformer':
        """
        Fit the transformer

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        self
        """
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Forward transform

        Parameters
        ----------
        y : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Transformed data
        """
        return y

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse transform

        Parameters
        ----------
        y : np.ndarray
            Transformed data

        Returns
        -------
        np.ndarray
            Data in original scale
        """
        return y

    def fitTransform(self, y: np.ndarray) -> np.ndarray:
        """
        Fit and transform in a single step

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        np.ndarray
            Transformed data
        """
        return self.fit(y).transform(y)


class Differencer(BaseTransformer):
    """
    Differencing Transformer

    Applies d-th order differencing to achieve stationarity.
    Restores original level on inverse transform.

    Parameters
    ----------
    d : int
        Differencing order (default: 1)
    """

    def __init__(self, d: int = 1):
        self.d = d
        self._lastValues: List[np.ndarray] = []

    def fit(self, y: np.ndarray) -> 'Differencer':
        y = np.asarray(y, dtype=np.float64).ravel()
        self._lastValues = []
        current = y.copy()
        for _ in range(self.d):
            self._lastValues.append(current[-1])
            current = np.diff(current)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        current = y.copy()
        for _ in range(self.d):
            current = np.diff(current)
        return current

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        current = y.copy()
        for i in range(self.d - 1, -1, -1):
            current = np.cumsum(np.concatenate([[self._lastValues[i]], current]))
            current = current[1:]
        return current


class LogTransformer(BaseTransformer):
    """
    Log Transformer

    Applies log(1 + y) transform to positive data.
    Automatically applies a shift if negative values are present.

    Parameters
    ----------
    shift : float, optional
        Constant to add to data. If None, auto-computed.
    """

    def __init__(self, shift: Optional[float] = None):
        self.shift = shift
        self._autoShift = 0.0

    def fit(self, y: np.ndarray) -> 'LogTransformer':
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.shift is not None:
            self._autoShift = self.shift
        else:
            minVal = np.nanmin(y)
            self._autoShift = max(0.0, 1.0 - minVal)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        return np.log1p(y + self._autoShift)

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        return np.expm1(y) - self._autoShift


class BoxCoxTransformer(BaseTransformer):
    """
    Box-Cox Transformer

    Automatically estimates optimal lambda to improve normality.

    Parameters
    ----------
    lmbda : float, optional
        Box-Cox parameter. If None, auto-estimated.
    """

    def __init__(self, lmbda: Optional[float] = None):
        self.lmbda = lmbda
        self._fittedLmbda: float = 1.0
        self._shift: float = 0.0

    def fit(self, y: np.ndarray) -> 'BoxCoxTransformer':
        from scipy import stats as spstats
        y = np.asarray(y, dtype=np.float64).ravel()
        minVal = np.nanmin(y)
        self._shift = max(0.0, 1.0 - minVal)
        shifted = y + self._shift
        shifted = shifted[np.isfinite(shifted)]
        shifted = shifted[shifted > 0]

        if self.lmbda is not None:
            self._fittedLmbda = self.lmbda
        elif len(shifted) >= 3:
            _, self._fittedLmbda = spstats.boxcox(shifted)
        else:
            self._fittedLmbda = 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        shifted = y + self._shift
        shifted = np.maximum(shifted, 1e-10)
        if abs(self._fittedLmbda) < 1e-10:
            return np.log(shifted)
        return (np.power(shifted, self._fittedLmbda) - 1.0) / self._fittedLmbda

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if abs(self._fittedLmbda) < 1e-10:
            result = np.exp(y)
        else:
            inner = y * self._fittedLmbda + 1.0
            inner = np.maximum(inner, 1e-10)
            result = np.power(inner, 1.0 / self._fittedLmbda)
        return result - self._shift


class Scaler(BaseTransformer):
    """
    Scaling Transformer

    Z-score (standardization) or MinMax normalization.

    Parameters
    ----------
    method : str
        Scaling method ('zscore' or 'minmax')
    """

    def __init__(self, method: str = 'zscore'):
        self.method = method
        self._mean: float = 0.0
        self._std: float = 1.0
        self._min: float = 0.0
        self._max: float = 1.0

    def fit(self, y: np.ndarray) -> 'Scaler':
        y = np.asarray(y, dtype=np.float64).ravel()
        clean = y[np.isfinite(y)]
        if len(clean) == 0:
            return self
        self._mean = float(np.mean(clean))
        self._std = max(float(np.std(clean, ddof=1)), 1e-10)
        self._min = float(np.min(clean))
        self._max = float(np.max(clean))
        if self._max - self._min < 1e-10:
            self._max = self._min + 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.method == 'minmax':
            return (y - self._min) / (self._max - self._min)
        return (y - self._mean) / self._std

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.method == 'minmax':
            return y * (self._max - self._min) + self._min
        return y * self._std + self._mean


class Deseasonalizer(BaseTransformer):
    """
    Deseasonalizer Transformer

    Separates and removes seasonal component from the time series.
    Adds seasonal pattern back on inverse transform.

    Parameters
    ----------
    period : int
        Seasonal period
    """

    def __init__(self, period: int = 7):
        self.period = period
        self._seasonalPattern: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> 'Deseasonalizer':
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        if n < self.period * 2:
            self._seasonalPattern = np.zeros(self.period)
            return self

        nComplete = (n // self.period) * self.period
        reshaped = y[:nComplete].reshape(-1, self.period)
        self._seasonalPattern = np.mean(reshaped, axis=0) - np.mean(y[:nComplete])
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if self._seasonalPattern is None:
            return y
        n = len(y)
        seasonal = np.tile(self._seasonalPattern, (n // self.period) + 1)[:n]
        return y - seasonal

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        if self._seasonalPattern is None:
            return y
        n = len(y)
        seasonal = np.tile(self._seasonalPattern, (n // self.period) + 1)[:n]
        return y + seasonal


class Detrend(BaseTransformer):
    """
    Detrend Transformer

    Removes linear trend and restores it on inverse transform.
    """

    def __init__(self):
        self._slope: float = 0.0
        self._intercept: float = 0.0
        self._n: int = 0

    def fit(self, y: np.ndarray) -> 'Detrend':
        y = np.asarray(y, dtype=np.float64).ravel()
        self._n = len(y)
        t = np.arange(self._n, dtype=np.float64)
        if self._n >= 2:
            self._slope, self._intercept = np.polyfit(t, y, 1)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        t = np.arange(len(y), dtype=np.float64)
        return y - (self._slope * t + self._intercept)

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)
        t = np.arange(self._n, self._n + n, dtype=np.float64)
        return y + (self._slope * t + self._intercept)


class OutlierClipper(BaseTransformer):
    """
    Outlier Clipping Transformer

    Clips outliers to boundary values based on IQR.

    Parameters
    ----------
    factor : float
        IQR multiplier (default: 3.0)
    """

    def __init__(self, factor: float = 3.0):
        self.factor = factor
        self._lower: float = -np.inf
        self._upper: float = np.inf

    def fit(self, y: np.ndarray) -> 'OutlierClipper':
        y = np.asarray(y, dtype=np.float64).ravel()
        clean = y[np.isfinite(y)]
        if len(clean) < 4:
            return self
        q1 = np.percentile(clean, 25)
        q3 = np.percentile(clean, 75)
        iqr = q3 - q1
        self._lower = q1 - self.factor * iqr
        self._upper = q3 + self.factor * iqr
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel()
        return np.clip(y, self._lower, self._upper)

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64).ravel()


class MissingValueImputer(BaseTransformer):
    """
    Missing Value Imputer Transformer

    Parameters
    ----------
    method : str
        Imputation method ('linear', 'mean', 'ffill')
    """

    def __init__(self, method: str = 'linear'):
        self.method = method
        self._mean: float = 0.0

    def fit(self, y: np.ndarray) -> 'MissingValueImputer':
        y = np.asarray(y, dtype=np.float64).ravel()
        clean = y[np.isfinite(y)]
        self._mean = float(np.mean(clean)) if len(clean) > 0 else 0.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64).ravel().copy()
        mask = ~np.isfinite(y)
        if not np.any(mask):
            return y

        if self.method == 'mean':
            y[mask] = self._mean
        elif self.method == 'ffill':
            for i in range(len(y)):
                if mask[i] and i > 0:
                    y[i] = y[i - 1]
            remaining = ~np.isfinite(y)
            if np.any(remaining):
                y[remaining] = self._mean
        else:
            indices = np.arange(len(y))
            valid = ~mask
            if np.sum(valid) >= 2:
                y[mask] = np.interp(indices[mask], indices[valid], y[valid])
            else:
                y[mask] = self._mean
        return y

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64).ravel()


class ForecastPipeline:
    """
    Time Series Forecast Pipeline

    Chains preprocessing transformers with a forecast model.
    Automatically applies transform/inverse-transform in the fit -> predict flow.

    Parameters
    ----------
    steps : List[Tuple[str, object]]
        List of (name, transformer or model) tuples.
        If the last step has a predict method, it is treated as a forecast model.
        All other steps must follow the BaseTransformer interface.

    Examples
    --------
    >>> pipe = ForecastPipeline([
    ...     ('impute', MissingValueImputer()),
    ...     ('scale', Scaler()),
    ...     ('forecast', Vectrix()),
    ... ])
    >>> pipe.fit(y)
    >>> pred, lo, hi = pipe.predict(12)
    """

    def __init__(self, steps: List[Tuple[str, Any]]):
        self._validateSteps(steps)
        self.steps = steps
        self.fitted = False
        self._transformedY: Optional[np.ndarray] = None

    def _validateSteps(self, steps: List[Tuple[str, Any]]):
        if len(steps) < 1:
            raise ValueError("Pipeline requires at least 1 step.")
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("Step names must be unique.")

    @property
    def _transformers(self) -> List[Tuple[str, BaseTransformer]]:
        if self._hasForecaster():
            return self.steps[:-1]
        return self.steps

    @property
    def _forecaster(self) -> Optional[Any]:
        if self._hasForecaster():
            return self.steps[-1][1]
        return None

    def _hasForecaster(self) -> bool:
        _, lastStep = self.steps[-1]
        return hasattr(lastStep, 'predict') and not isinstance(lastStep, BaseTransformer)

    def fit(self, y: np.ndarray, **kwargs) -> 'ForecastPipeline':
        """
        Fit the entire pipeline

        Parameters
        ----------
        y : np.ndarray
            Time series data
        **kwargs
            Additional arguments passed to the forecast model

        Returns
        -------
        self
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        current = y.copy()

        for name, transformer in self._transformers:
            current = transformer.fitTransform(current)

        self._transformedY = current

        if self._forecaster is not None:
            forecaster = self._forecaster
            if hasattr(forecaster, 'fit'):
                forecaster.fit(current, **kwargs)

        self.fitted = True
        return self

    def predict(
        self,
        steps: int,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions (with automatic inverse transform)

        Parameters
        ----------
        steps : int
            Number of forecast steps
        **kwargs
            Additional arguments passed to the forecast model

        Returns
        -------
        predictions, lower, upper : np.ndarray
            Inverse-transformed predictions, lower bounds, upper bounds
        """
        if not self.fitted:
            raise ValueError("Pipeline has not been fitted. Call fit() first.")

        forecaster = self._forecaster
        if forecaster is None:
            raise ValueError("Pipeline has no forecast model.")

        result = forecaster.predict(steps, **kwargs)

        if isinstance(result, tuple) and len(result) == 3:
            pred, lo, hi = result
        elif isinstance(result, tuple) and len(result) == 2:
            pred, lo = result
            hi = pred.copy()
        else:
            pred = np.asarray(result, dtype=np.float64).ravel()
            lo = pred.copy()
            hi = pred.copy()

        pred = np.asarray(pred, dtype=np.float64).ravel()
        lo = np.asarray(lo, dtype=np.float64).ravel()
        hi = np.asarray(hi, dtype=np.float64).ravel()

        for name, transformer in reversed(self._transformers):
            pred = transformer.inverseTransform(pred)
            lo = transformer.inverseTransform(lo)
            hi = transformer.inverseTransform(hi)

        return pred, lo, hi

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing only (without forecasting)

        Parameters
        ----------
        y : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Transformed data
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        current = y.copy()
        for name, transformer in self._transformers:
            current = transformer.transform(current)
        return current

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        """
        Apply inverse transform only

        Parameters
        ----------
        y : np.ndarray
            Transformed data

        Returns
        -------
        np.ndarray
            Data in original scale
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        current = y.copy()
        for name, transformer in reversed(self._transformers):
            current = transformer.inverseTransform(current)
        return current

    def getStep(self, name: str) -> Any:
        """
        Retrieve a step by name

        Parameters
        ----------
        name : str
            Step name

        Returns
        -------
        object
            The transformer or model
        """
        for stepName, step in self.steps:
            if stepName == name:
                return step
        raise KeyError(f"Step '{name}' not found.")

    def getParams(self) -> Dict[str, Any]:
        """
        Retrieve all pipeline parameters

        Returns
        -------
        Dict
            Dictionary in {stepName__paramName: value} format
        """
        params = {}
        for name, step in self.steps:
            stepParams = vars(step) if hasattr(step, '__dict__') else {}
            for key, val in stepParams.items():
                if not key.startswith('_'):
                    params[f"{name}__{key}"] = val
        return params

    def listSteps(self) -> List[str]:
        """
        List of registered step names

        Returns
        -------
        List[str]
        """
        return [name for name, _ in self.steps]

    def __repr__(self) -> str:
        stepStrs = []
        for name, step in self.steps:
            stepStrs.append(f"  ('{name}', {step.__class__.__name__})")
        inner = ",\n".join(stepStrs)
        return f"ForecastPipeline([\n{inner}\n])"
