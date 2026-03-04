"""
Time Series Cross-Validation

Provides expanding window and sliding window cross-validation
for time series models. Respects temporal ordering.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..engine.turbo import TurboCore


class TimeSeriesCrossValidator:
    """
    Time series cross-validator

    Supports:
    - Expanding window: train grows each fold
    - Sliding window: fixed-size train window slides forward

    Usage:
        >>> cv = TimeSeriesCrossValidator(nSplits=5, horizon=30)
        >>> results = cv.evaluate(y, modelFactory, period=7)
    """

    def __init__(
        self,
        nSplits: int = 5,
        horizon: int = 30,
        strategy: str = 'expanding',
        minTrainSize: int = 50,
        stepSize: Optional[int] = None,
        n_splits: int = None,
        min_train_size: int = None,
        step_size: int = None
    ):
        """
        Parameters
        ----------
        nSplits : int
            Number of CV folds
        horizon : int
            Forecast horizon per fold
        strategy : str
            'expanding' or 'sliding'
        minTrainSize : int
            Minimum training set size
        stepSize : int, optional
            Step between folds. If None, auto-calculated.
        """
        if n_splits is not None:
            nSplits = n_splits
        if min_train_size is not None:
            minTrainSize = min_train_size
        if step_size is not None:
            stepSize = step_size

        self.nSplits = nSplits
        self.horizon = horizon
        self.strategy = strategy
        self.minTrainSize = minTrainSize
        self.stepSize = stepSize

        self.n_splits = self.nSplits
        self.min_train_size = self.minTrainSize
        self.step_size = self.stepSize

    def split(self, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test index splits.

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (trainIndices, testIndices) pairs
        """
        n = len(y)
        needed = self.minTrainSize + self.horizon
        if n < needed:
            return []

        available = n - self.minTrainSize - self.horizon
        if self.stepSize is not None:
            step = self.stepSize
        else:
            step = max(1, available // max(self.nSplits - 1, 1))

        splits = []
        for i in range(self.nSplits):
            if self.strategy == 'sliding':
                trainEnd = self.minTrainSize + i * step
                trainStart = max(0, trainEnd - self.minTrainSize)
            else:
                trainEnd = self.minTrainSize + i * step
                trainStart = 0

            testEnd = min(trainEnd + self.horizon, n)

            if trainEnd >= n or testEnd <= trainEnd:
                break

            trainIdx = np.arange(trainStart, trainEnd)
            testIdx = np.arange(trainEnd, testEnd)
            splits.append((trainIdx, testIdx))

        return splits

    def evaluate(
        self,
        y: np.ndarray,
        modelFactory: Callable = None,
        period: int = 1,
        model_factory: Callable = None
    ) -> Dict[str, Any]:
        """
        Run cross-validation on a model.

        Parameters
        ----------
        y : np.ndarray
            Time series data
        modelFactory : Callable
            Function that returns a new model instance.
            The model must have fit(y) and predict(steps) methods.
        period : int
            Seasonal period (passed to model if supported)

        Returns
        -------
        Dict[str, Any]
            {
                'mape': float, 'rmse': float, 'mae': float, 'smape': float,
                'foldResults': List[Dict], 'nFolds': int
            }
        """
        if model_factory is not None:
            modelFactory = model_factory

        splits = self.split(y)
        if not splits:
            return {'mape': np.inf, 'rmse': np.inf, 'mae': np.inf, 'smape': np.inf,
                    'foldResults': [], 'nFolds': 0}

        foldResults = []

        for foldIdx, (trainIdx, testIdx) in enumerate(splits):
            train = y[trainIdx]
            test = y[testIdx]
            steps = len(test)

            try:
                model = modelFactory()
                model.fit(train)
                pred, _, _ = model.predict(steps)

                pred = pred[:len(test)]

                mape = TurboCore.mape(test, pred)
                rmse = TurboCore.rmse(test, pred)
                mae = TurboCore.mae(test, pred)
                smape = TurboCore.smape(test, pred)

                foldResults.append({
                    'fold': foldIdx,
                    'trainSize': len(train),
                    'testSize': len(test),
                    'mape': mape,
                    'rmse': rmse,
                    'mae': mae,
                    'smape': smape
                })
            except Exception:
                foldResults.append({
                    'fold': foldIdx,
                    'trainSize': len(train),
                    'testSize': len(test),
                    'mape': np.inf,
                    'rmse': np.inf,
                    'mae': np.inf,
                    'smape': np.inf
                })

        validFolds = [f for f in foldResults if f['mape'] < np.inf]
        nValid = len(validFolds)

        if nValid == 0:
            return {'mape': np.inf, 'rmse': np.inf, 'mae': np.inf, 'smape': np.inf,
                    'foldResults': foldResults, 'nFolds': len(splits)}

        avgMape = np.mean([f['mape'] for f in validFolds])
        avgRmse = np.mean([f['rmse'] for f in validFolds])
        avgMae = np.mean([f['mae'] for f in validFolds])
        avgSmape = np.mean([f['smape'] for f in validFolds])

        return {
            'mape': avgMape,
            'rmse': avgRmse,
            'mae': avgMae,
            'smape': avgSmape,
            'foldResults': foldResults,
            'nFolds': len(splits)
        }
