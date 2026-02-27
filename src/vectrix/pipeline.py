"""
시계열 예측 파이프라인

전처리 → 예측 → 후처리를 체이닝하는 파이프라인 시스템.
sklearn의 Pipeline 패턴을 시계열 예측에 맞게 확장.

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
    시계열 변환기 기본 클래스

    모든 전처리/후처리 변환기의 부모 클래스.
    fit → transform → inverseTransform 인터페이스를 제공.
    """

    def fit(self, y: np.ndarray) -> 'BaseTransformer':
        """
        변환기 학습

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터

        Returns
        -------
        self
        """
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        순방향 변환

        Parameters
        ----------
        y : np.ndarray
            입력 데이터

        Returns
        -------
        np.ndarray
            변환된 데이터
        """
        return y

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        """
        역변환

        Parameters
        ----------
        y : np.ndarray
            변환된 데이터

        Returns
        -------
        np.ndarray
            원래 스케일의 데이터
        """
        return y

    def fitTransform(self, y: np.ndarray) -> np.ndarray:
        """
        학습 후 변환을 한 번에 수행

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터

        Returns
        -------
        np.ndarray
            변환된 데이터
        """
        return self.fit(y).transform(y)


class Differencer(BaseTransformer):
    """
    차분 변환기

    시계열을 d차 차분하여 정상성을 확보.
    역변환 시 원래 수준으로 복원.

    Parameters
    ----------
    d : int
        차분 차수 (기본: 1)
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
    로그 변환기

    양수 데이터에 log(1 + y) 변환 적용.
    음수가 포함된 경우 shift를 자동 적용.

    Parameters
    ----------
    shift : float, optional
        데이터에 더할 상수. None이면 자동 계산.
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
    Box-Cox 변환기

    최적 lambda를 자동 추정하여 정규성 향상.

    Parameters
    ----------
    lmbda : float, optional
        Box-Cox 파라미터. None이면 자동 추정.
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
    스케일링 변환기

    Z-score (표준화) 또는 MinMax 정규화.

    Parameters
    ----------
    method : str
        스케일링 방법 ('zscore' 또는 'minmax')
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
    계절성 제거 변환기

    시계열에서 계절 성분을 분리하여 제거.
    역변환 시 계절 패턴을 다시 추가.

    Parameters
    ----------
    period : int
        계절 주기
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
    추세 제거 변환기

    선형 추세를 제거하고 역변환 시 복원.
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
    이상치 클리핑 변환기

    IQR 기반으로 이상치를 경계값으로 클리핑.

    Parameters
    ----------
    factor : float
        IQR 배수 (기본: 3.0)
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
    결측치 보간 변환기

    Parameters
    ----------
    method : str
        보간 방법 ('linear', 'mean', 'ffill')
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
    시계열 예측 파이프라인

    전처리 변환기들과 예측 모델을 체이닝.
    fit → predict 흐름에서 자동으로 변환/역변환 수행.

    Parameters
    ----------
    steps : List[Tuple[str, object]]
        (이름, 변환기 또는 모델) 튜플의 리스트.
        마지막 스텝이 predict 메서드를 가지면 예측 모델로 취급.
        그 외 스텝은 BaseTransformer 인터페이스를 따라야 함.

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
            raise ValueError("파이프라인에 최소 1개 스텝이 필요합니다.")
        names = [name for name, _ in steps]
        if len(set(names)) != len(names):
            raise ValueError("스텝 이름이 중복되었습니다.")

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
        전체 파이프라인 학습

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        **kwargs
            예측 모델에 전달할 추가 인자

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
        예측 수행 (역변환 자동 적용)

        Parameters
        ----------
        steps : int
            예측 스텝 수
        **kwargs
            예측 모델에 전달할 추가 인자

        Returns
        -------
        predictions, lower, upper : np.ndarray
            역변환된 예측값, 하한, 상한
        """
        if not self.fitted:
            raise ValueError("파이프라인이 학습되지 않았습니다. fit()을 먼저 호출하세요.")

        forecaster = self._forecaster
        if forecaster is None:
            raise ValueError("파이프라인에 예측 모델이 없습니다.")

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
        전처리만 수행 (예측 없이)

        Parameters
        ----------
        y : np.ndarray
            입력 데이터

        Returns
        -------
        np.ndarray
            변환된 데이터
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        current = y.copy()
        for name, transformer in self._transformers:
            current = transformer.transform(current)
        return current

    def inverseTransform(self, y: np.ndarray) -> np.ndarray:
        """
        역변환만 수행

        Parameters
        ----------
        y : np.ndarray
            변환된 데이터

        Returns
        -------
        np.ndarray
            원래 스케일의 데이터
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        current = y.copy()
        for name, transformer in reversed(self._transformers):
            current = transformer.inverseTransform(current)
        return current

    def getStep(self, name: str) -> Any:
        """
        이름으로 스텝 조회

        Parameters
        ----------
        name : str
            스텝 이름

        Returns
        -------
        object
            해당 변환기 또는 모델
        """
        for stepName, step in self.steps:
            if stepName == name:
                return step
        raise KeyError(f"스텝 '{name}'을 찾을 수 없습니다.")

    def getParams(self) -> Dict[str, Any]:
        """
        전체 파이프라인 파라미터 조회

        Returns
        -------
        Dict
            {stepName__paramName: value} 형태의 딕셔너리
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
        등록된 스텝 이름 목록

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
