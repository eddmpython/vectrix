"""Batch Forecast API"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd


class BatchForecastResult:
    """Batch forecast result"""

    def __init__(self, results, failures):
        self.results = results
        self.failures = failures

    @property
    def successCount(self):
        return len(self.results)

    @property
    def failureCount(self):
        return len(self.failures)

    @property
    def totalCount(self):
        return self.successCount + self.failureCount

    def bestModels(self) -> pd.DataFrame:
        """Summary of best model per series"""
        rows = []
        for seriesId, result in self.results.items():
            rows.append({
                'id': seriesId,
                'model': result.bestModelName if hasattr(result, 'bestModelName') else 'unknown',
            })
        return pd.DataFrame(rows)

    def export(self, path):
        """Export all forecast results to CSV"""
        allRows = []
        for seriesId, result in self.results.items():
            n = len(result.predictions)
            for i in range(n):
                allRows.append({
                    'id': seriesId,
                    'step': i + 1,
                    'prediction': result.predictions[i],
                    'lower95': result.lower95[i],
                    'upper95': result.upper95[i],
                })
        pd.DataFrame(allRows).to_csv(path, index=False)
        return self

    def summary(self) -> str:
        """Batch summary"""
        lines = [
            f"BatchForecast: {self.successCount}/{self.totalCount} succeeded",
            f"  Failed: {self.failureCount}",
        ]
        if self.results:
            models = [r.bestModelName for r in self.results.values() if hasattr(r, 'bestModelName')]
            if models:
                from collections import Counter
                mc = Counter(models).most_common(3)
                lines.append(f"  Top models: {', '.join(f'{m}({c})' for m, c in mc)}")
        return '\n'.join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return f"BatchForecastResult(success={self.successCount}, failed={self.failureCount})"


def batchForecast(
    data: pd.DataFrame,
    idCol: str,
    dateCol: str = 'date',
    valueCol: str = 'value',
    steps: int = 12,
    nJobs: int = 1,
    onFailure: str = 'skip',
    progress: Optional[Callable] = None,
) -> BatchForecastResult:
    """
    Batch forecast for multiple time series.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format data. Series distinguished by idCol.
    idCol : str
        Series ID column.
    dateCol : str
        Date column.
    valueCol : str
        Value column.
    steps : int
        Forecast horizon.
    nJobs : int
        Parallelism level (1=sequential).
    onFailure : str
        Failure handling ('skip', 'raise').
    progress : callable
        Progress callback (current, total, seriesId).
    """
    from .vectrix import Vectrix

    groups = list(data.groupby(idCol))
    totalGroups = len(groups)
    results = {}
    failures = {}

    def forecastSingle(seriesId, groupDf):
        fx = Vectrix(verbose=False)
        result = fx.forecast(groupDf, dateCol=dateCol, valueCol=valueCol, steps=steps)
        return result

    if nJobs == 1:
        for i, (seriesId, groupDf) in enumerate(groups):
            if progress:
                progress(i + 1, totalGroups, str(seriesId))
            try:
                results[seriesId] = forecastSingle(seriesId, groupDf)
            except Exception as e:
                if onFailure == 'raise':
                    raise
                failures[seriesId] = str(e)
    else:
        actualJobs = nJobs if nJobs > 0 else 4
        with ThreadPoolExecutor(max_workers=actualJobs) as executor:
            futureMap = {}
            for seriesId, groupDf in groups:
                future = executor.submit(forecastSingle, seriesId, groupDf)
                futureMap[future] = seriesId

            for i, future in enumerate(as_completed(futureMap)):
                seriesId = futureMap[future]
                if progress:
                    progress(i + 1, totalGroups, str(seriesId))
                try:
                    results[seriesId] = future.result()
                except Exception as e:
                    if onFailure == 'raise':
                        raise
                    failures[seriesId] = str(e)

    return BatchForecastResult(results, failures)
