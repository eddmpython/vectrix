"""
Panel Data Handler

다중 시계열 DataFrame 관리:
- Long format (id, date, value) 변환
- Wide format (date x series) 변환
- 시계열별 접근
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


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
        Long format DataFrame에서 PanelData 생성

        Parameters
        ----------
        df : pd.DataFrame
            Long format (id, date, value per row)
        idCol : str
            시계열 ID 컬럼
        dateCol : str
            날짜 컬럼
        valueCol : str
            값 컬럼
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
        Wide format DataFrame에서 PanelData 생성

        Parameters
        ----------
        df : pd.DataFrame
            Wide format (date x series columns)
        dateCol : str
            날짜 컬럼
        valueCols : List[str], optional
            값 컬럼들. None이면 dateCol 외 모든 컬럼.
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
        """Dict에서 PanelData 생성"""
        panel = cls()
        for name, values in data.items():
            panel.seriesData[name] = np.asarray(values, dtype=np.float64)
            panel.seriesNames.append(name)
        return panel

    def toDict(self) -> Dict[str, np.ndarray]:
        """모든 시계열을 Dict로 반환"""
        return dict(self.seriesData)

    def getSeries(self, name: str) -> np.ndarray:
        """특정 시계열 반환"""
        if name not in self.seriesData:
            raise KeyError(f"Series '{name}' not found. Available: {self.seriesNames}")
        return self.seriesData[name]

    def getDates(self, name: str) -> Optional[np.ndarray]:
        """특정 시계열의 날짜 반환"""
        return self.seriesDates.get(name)

    @property
    def nSeries(self) -> int:
        return len(self.seriesNames)

    @property
    def lengths(self) -> Dict[str, int]:
        return {name: len(data) for name, data in self.seriesData.items()}

    def summary(self) -> Dict[str, Dict]:
        """패널 데이터 요약"""
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
