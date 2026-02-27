"""
TSFrame: Time-Series-Aware DataFrame

pandas DataFrame 래퍼로, 시계열 분석에 특화된 메서드 제공.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class TSFrame:
    """
    Time Series DataFrame

    Usage:
        >>> ts = TSFrame.fromPandas(df, dateCol='date', valueCol='sales')
        >>> ts.info()
        >>> result = ts.forecast(steps=30)
        >>> decomposition = ts.decompose()
        >>> train, test = ts.split(ratio=0.8)
        >>> weekly = ts.resample('W')
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dateCol: str,
        valueCol: str,
        freq: Optional[str] = None
    ):
        self._df = df.copy()
        self.dateCol = dateCol
        self.valueCol = valueCol

        self._df[dateCol] = pd.to_datetime(self._df[dateCol])
        self._df = self._df.sort_values(dateCol).reset_index(drop=True)

        self.freq = freq or self._detectFrequency()
        self.period = self._freqToPeriod(self.freq)

    @classmethod
    def fromPandas(cls, df: pd.DataFrame, dateCol: str, valueCol: str,
                    freq: Optional[str] = None) -> 'TSFrame':
        return cls(df, dateCol, valueCol, freq)

    @classmethod
    def fromCsv(cls, path: str, dateCol: str, valueCol: str,
                 freq: Optional[str] = None, **kwargs) -> 'TSFrame':
        df = pd.read_csv(path, **kwargs)
        return cls(df, dateCol, valueCol, freq)

    @classmethod
    def fromArrays(cls, dates: np.ndarray, values: np.ndarray,
                     dateCol: str = 'date', valueCol: str = 'value',
                     freq: Optional[str] = None) -> 'TSFrame':
        df = pd.DataFrame({dateCol: dates, valueCol: values})
        return cls(df, dateCol, valueCol, freq)

    @classmethod
    def generate(
        cls,
        n: int = 365,
        freq: str = 'D',
        startDate: str = '2023-01-01',
        trend: float = 0.1,
        seasonalAmplitude: float = 10.0,
        seasonalPeriod: int = 7,
        noiseStd: float = 3.0,
        seed: int = 42
    ) -> 'TSFrame':
        """
        테스트용 시계열 데이터 자동 생성

        Parameters
        ----------
        n : int
            데이터 개수
        freq : str
            주기 ('D', 'W', 'M', 'H')
        startDate : str
            시작일
        trend : float
            추세 기울기
        seasonalAmplitude : float
            계절 진폭
        seasonalPeriod : int
            계절 주기
        noiseStd : float
            노이즈 표준편차
        seed : int
            랜덤 시드
        """
        np.random.seed(seed)
        dates = pd.date_range(start=startDate, periods=n, freq=freq)
        t = np.arange(n, dtype=np.float64)

        values = (
            100
            + trend * t
            + seasonalAmplitude * np.sin(2 * np.pi * t / seasonalPeriod)
            + noiseStd * np.random.randn(n)
        )

        df = pd.DataFrame({'date': dates, 'value': values})
        return cls(df, 'date', 'value', freq)

    @property
    def values(self) -> np.ndarray:
        return self._df[self.valueCol].values.astype(np.float64)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._df[self.dateCol])

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return (
            f"TSFrame(n={len(self)}, freq='{self.freq}', period={self.period}, "
            f"range=[{self.dates[0].strftime('%Y-%m-%d')} ~ {self.dates[-1].strftime('%Y-%m-%d')}])"
        )

    def info(self) -> Dict[str, Any]:
        y = self.values
        return {
            'length': len(y),
            'frequency': self.freq,
            'period': self.period,
            'dateRange': (
                self.dates[0].strftime('%Y-%m-%d'),
                self.dates[-1].strftime('%Y-%m-%d')
            ),
            'mean': round(float(np.mean(y)), 4),
            'std': round(float(np.std(y)), 4),
            'min': round(float(np.min(y)), 4),
            'max': round(float(np.max(y)), 4),
            'cv': round(float(np.std(y) / abs(np.mean(y))) if abs(np.mean(y)) > 0 else 0, 4),
            'missingCount': int(self._df[self.valueCol].isna().sum()),
        }

    def forecast(self, steps: int = 30, **kwargs) -> Any:
        """
        Vectrix로 예측

        Returns
        -------
        ForecastResult
        """
        from ..vectrix import Vectrix
        fx = Vectrix(**kwargs)
        return fx.forecast(self._df, self.dateCol, self.valueCol, steps=steps)

    def decompose(self, period: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        시계열 분해 (추세 + 계절성 + 잔차)
        """
        from ..business.explain import ForecastExplainer
        explainer = ForecastExplainer()
        p = period or self.period
        return explainer._decompose(self.values, p)

    def detectAnomalies(self, method: str = 'auto', threshold: float = 3.0) -> Any:
        """이상치 탐지"""
        from ..business.anomaly import AnomalyDetector
        detector = AnomalyDetector()
        return detector.detect(self.values, method=method, threshold=threshold, period=self.period)

    def split(self, ratio: float = 0.8) -> Tuple['TSFrame', 'TSFrame']:
        """
        Train/Test split by date

        Returns
        -------
        Tuple[TSFrame, TSFrame]
            (train, test)
        """
        n = len(self._df)
        splitIdx = int(n * ratio)

        trainDf = self._df.iloc[:splitIdx].copy()
        testDf = self._df.iloc[splitIdx:].copy()

        train = TSFrame(trainDf, self.dateCol, self.valueCol, self.freq)
        test = TSFrame(testDf, self.dateCol, self.valueCol, self.freq)

        return train, test

    def resample(self, rule: str, agg: str = 'mean') -> 'TSFrame':
        """
        리샘플링

        Parameters
        ----------
        rule : str
            'W' (주간), 'M' (월간), 'Q' (분기), etc.
        agg : str
            'mean', 'sum', 'last', 'first'
        """
        indexed = self._df.set_index(self.dateCol)
        aggFunc = {
            'mean': 'mean',
            'sum': 'sum',
            'last': 'last',
            'first': 'first',
        }.get(agg, 'mean')

        resampled = indexed[self.valueCol].resample(rule).agg(aggFunc).dropna()
        newDf = resampled.reset_index()
        newDf.columns = [self.dateCol, self.valueCol]

        return TSFrame(newDf, self.dateCol, self.valueCol, rule)

    def fillMissing(self, method: str = 'interpolate') -> 'TSFrame':
        """
        결측치 처리

        Parameters
        ----------
        method : str
            'interpolate', 'ffill', 'bfill', 'mean'
        """
        df = self._df.copy()

        if method == 'interpolate':
            df[self.valueCol] = df[self.valueCol].interpolate(method='linear')
        elif method == 'ffill':
            df[self.valueCol] = df[self.valueCol].ffill()
        elif method == 'bfill':
            df[self.valueCol] = df[self.valueCol].bfill()
        elif method == 'mean':
            df[self.valueCol] = df[self.valueCol].fillna(df[self.valueCol].mean())

        df[self.valueCol] = df[self.valueCol].ffill().bfill()

        return TSFrame(df, self.dateCol, self.valueCol, self.freq)

    def diff(self, periods: int = 1) -> 'TSFrame':
        """차분"""
        df = self._df.copy()
        df[self.valueCol] = df[self.valueCol].diff(periods)
        df = df.dropna().reset_index(drop=True)
        return TSFrame(df, self.dateCol, self.valueCol, self.freq)

    def rollingMean(self, window: int = 7) -> 'TSFrame':
        """이동평균"""
        df = self._df.copy()
        df[self.valueCol] = df[self.valueCol].rolling(window=window, min_periods=1).mean()
        return TSFrame(df, self.dateCol, self.valueCol, self.freq)

    def tail(self, n: int = 10) -> pd.DataFrame:
        return self._df.tail(n)

    def head(self, n: int = 10) -> pd.DataFrame:
        return self._df.head(n)

    def _detectFrequency(self) -> str:
        if len(self._df) < 2:
            return 'D'

        diffs = self._df[self.dateCol].diff().dropna()
        medianDiff = diffs.median()

        if medianDiff <= pd.Timedelta(hours=2):
            return 'H'
        elif medianDiff <= pd.Timedelta(days=1.5):
            return 'D'
        elif medianDiff <= pd.Timedelta(days=8):
            return 'W'
        elif medianDiff <= pd.Timedelta(days=35):
            return 'ME'
        elif medianDiff <= pd.Timedelta(days=100):
            return 'QE'
        else:
            return 'YE'

    @staticmethod
    def _freqToPeriod(freq: str) -> int:
        mapping = {
            'H': 24,
            'D': 7,
            'W': 52,
            'ME': 12,
            'QE': 4,
            'YE': 1,
            'M': 12,
            'Q': 4,
            'Y': 1,
        }
        return mapping.get(freq, 7)
