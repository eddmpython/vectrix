"""
Panel Data Handler

Multi-series DataFrame management:
- Long format (id, date, value) conversion
- Wide format (date x series) conversion
- Per-series access
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class PanelData:
    """
    Panel (multi-series) data handler

    Usage:
        >>> panel = PanelData.fromLongFormat(df, idCol='store', dateCol='date', valueCol='sales')
        >>> series_dict = panel.toDict()
        >>> panel.getSeries('store_1')
    """

    def __init__(self):
        self.seriesData: Dict[str, np.ndarray] = {}
        self.seriesDates: Dict[str, np.ndarray] = {}
        self.seriesNames: List[str] = []

    @classmethod
    def fromLongFormat(
        cls,
        df: pd.DataFrame,
        idCol: str,
        dateCol: str,
        valueCol: str
    ) -> 'PanelData':
        """
        Create PanelData from a long format DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            Long format (id, date, value per row)
        idCol : str
            Series ID column
        dateCol : str
            Date column
        valueCol : str
            Value column
        """
        panel = cls()
        df = df.copy()
        df[dateCol] = pd.to_datetime(df[dateCol])

        for seriesId in df[idCol].unique():
            subset = df[df[idCol] == seriesId].sort_values(dateCol)
            name = str(seriesId)
            panel.seriesData[name] = subset[valueCol].values.astype(np.float64)
            panel.seriesDates[name] = subset[dateCol].values
            panel.seriesNames.append(name)

        return panel

    @classmethod
    def fromWideFormat(
        cls,
        df: pd.DataFrame,
        dateCol: str,
        valueCols: Optional[List[str]] = None
    ) -> 'PanelData':
        """
        Create PanelData from a wide format DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            Wide format (date x series columns)
        dateCol : str
            Date column
        valueCols : List[str], optional
            Value columns. If None, all columns except dateCol.
        """
        panel = cls()
        df = df.copy()
        df[dateCol] = pd.to_datetime(df[dateCol])
        df = df.sort_values(dateCol)

        if valueCols is None:
            valueCols = [c for c in df.columns if c != dateCol]

        dates = df[dateCol].values
        for col in valueCols:
            name = str(col)
            panel.seriesData[name] = df[col].values.astype(np.float64)
            panel.seriesDates[name] = dates
            panel.seriesNames.append(name)

        return panel

    @classmethod
    def fromDict(cls, data: Dict[str, np.ndarray]) -> 'PanelData':
        """Create PanelData from a dict"""
        panel = cls()
        for name, values in data.items():
            panel.seriesData[name] = np.asarray(values, dtype=np.float64)
            panel.seriesNames.append(name)
        return panel

    def toDict(self) -> Dict[str, np.ndarray]:
        """Return all series as a dict"""
        return dict(self.seriesData)

    def getSeries(self, name: str) -> np.ndarray:
        """Return a specific series"""
        if name not in self.seriesData:
            raise KeyError(f"Series '{name}' not found. Available: {self.seriesNames}")
        return self.seriesData[name]

    def getDates(self, name: str) -> Optional[np.ndarray]:
        """Return dates for a specific series"""
        return self.seriesDates.get(name)

    @property
    def nSeries(self) -> int:
        return len(self.seriesNames)

    @property
    def lengths(self) -> Dict[str, int]:
        return {name: len(data) for name, data in self.seriesData.items()}

    def summary(self) -> Dict[str, Dict]:
        """Panel data summary"""
        result = {}
        for name, data in self.seriesData.items():
            result[name] = {
                'length': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
            }
        return result
