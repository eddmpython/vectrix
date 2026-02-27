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
        >>> cv = TimeSeriesCrossValidator(n_splits=5, horizon=30)
        >>> results = cv.evaluate(y, model_factory, period=7)
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 30,
        strategy: str = 'expanding',
        min_train_size: int = 50,
        step_size: Optional[int] = None
    ):
        """
        Parameters
        ----------
        n_splits : int
            Number of CV folds
        horizon : int
            Forecast horizon per fold
        strategy : str
            'expanding' or 'sliding'
        min_train_size : int
            Minimum training set size
        step_size : int, optional
            Step between folds. If None, auto-calculated.
        """
        self.n_splits = n_splits
        self.horizon = horizon
        self.strategy = strategy
        self.min_train_size = min_train_size
        self.step_size = step_size

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
            List of (train_indices, test_indices) pairs
        """
        n = len(y)
        needed = self.min_train_size + self.horizon
        if n < needed:
            return []

        available = n - self.min_train_size - self.horizon
        if self.step_size is not None:
            step = self.step_size
        else:
            step = max(1, available // max(self.n_splits - 1, 1))

        splits = []
        for i in range(self.n_splits):
            if self.strategy == 'sliding':
                train_end = self.min_train_size + i * step
                train_start = max(0, train_end - self.min_train_size)
            else:
                train_end = self.min_train_size + i * step
                train_start = 0

            test_end = min(train_end + self.horizon, n)

            if train_end >= n or test_end <= train_end:
                break

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(train_end, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def evaluate(
        self,
        y: np.ndarray,
        model_factory: Callable,
        period: int = 1
    ) -> Dict[str, Any]:
        """
        Run cross-validation on a model.

        Parameters
        ----------
        y : np.ndarray
            Time series data
        model_factory : Callable
            Function that returns a new model instance.
            The model must have fit(y) and predict(steps) methods.
        period : int
            Seasonal period (passed to model if supported)

        Returns
        -------
        Dict[str, Any]
            {
                'mape': float, 'rmse': float, 'mae': float, 'smape': float,
                'fold_results': List[Dict], 'n_folds': int
            }
        """
        splits = self.split(y)
        if not splits:
            return {'mape': np.inf, 'rmse': np.inf, 'mae': np.inf, 'smape': np.inf,
                    'fold_results': [], 'n_folds': 0}

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train = y[train_idx]
            test = y[test_idx]
            steps = len(test)

            try:
                model = model_factory()
                model.fit(train)
                pred, _, _ = model.predict(steps)

                pred = pred[:len(test)]

                mape = TurboCore.mape(test, pred)
                rmse = TurboCore.rmse(test, pred)
                mae = TurboCore.mae(test, pred)
                smape = TurboCore.smape(test, pred)

                fold_results.append({
                    'fold': fold_idx,
                    'train_size': len(train),
                    'test_size': len(test),
                    'mape': mape,
                    'rmse': rmse,
                    'mae': mae,
                    'smape': smape
                })
            except Exception:
                fold_results.append({
                    'fold': fold_idx,
                    'train_size': len(train),
                    'test_size': len(test),
                    'mape': np.inf,
                    'rmse': np.inf,
                    'mae': np.inf,
                    'smape': np.inf
                })

        valid_folds = [f for f in fold_results if f['mape'] < np.inf]
        n_valid = len(valid_folds)

        if n_valid == 0:
            return {'mape': np.inf, 'rmse': np.inf, 'mae': np.inf, 'smape': np.inf,
                    'fold_results': fold_results, 'n_folds': len(splits)}

        avg_mape = np.mean([f['mape'] for f in valid_folds])
        avg_rmse = np.mean([f['rmse'] for f in valid_folds])
        avg_mae = np.mean([f['mae'] for f in valid_folds])
        avg_smape = np.mean([f['smape'] for f in valid_folds])

        return {
            'mape': avg_mape,
            'rmse': avg_rmse,
            'mae': avg_mae,
            'smape': avg_smape,
            'fold_results': fold_results,
            'n_folds': len(splits)
        }
