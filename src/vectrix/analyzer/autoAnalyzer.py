"""
AutoAnalyzer: Automatic Time Series Data Analyzer

Automatically analyzes data characteristics (trend, seasonality, stationarity, volatility, etc.).
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..types import DataCharacteristics, Frequency


class AutoAnalyzer:
    """
    Automatic Time Series Data Analyzer

    Automatically analyzes frequency, trend, seasonality, stationarity, volatility, etc.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        dateCol: str,
        valueCol: str
    ) -> DataCharacteristics:
        """
        Comprehensive data characteristics analysis

        Parameters
        ----------
        df : pd.DataFrame
            Time series DataFrame
        dateCol : str
            Date column name
        valueCol : str
            Value column name

        Returns
        -------
        DataCharacteristics
            Analyzed data characteristics
        """
        dates = pd.to_datetime(df[dateCol])
        values = df[valueCol].values.astype(float)
        n = len(values)

        # Basic info
        freq, basePeriod = self._detectFrequency(dates)
        dateRange = (
            dates.iloc[0].strftime('%Y-%m-%d'),
            dates.iloc[-1].strftime('%Y-%m-%d')
        )

        # Seasonality analysis
        seasonalPeriods = self._detectSeasonalPeriods(values, freq, basePeriod)
        hasSeasonality, seasonalStrength = self._analyzeSeasonality(
            values, seasonalPeriods[0] if seasonalPeriods else basePeriod
        )

        # Trend analysis
        hasTrend, trendDirection, trendStrength = self._analyzeTrend(values)

        # Stationarity test
        isStationary = self._checkStationarity(values)

        # Volatility analysis
        volatility, volatilityLevel = self._analyzeVolatility(values)

        # Quality analysis
        missingRatio = df[valueCol].isna().sum() / n * 100
        outlierCount, outlierRatio = self._detectOutliers(values)

        # Predictability score
        predictabilityScore = self._calculatePredictability(
            hasSeasonality, seasonalStrength,
            hasTrend, trendStrength,
            volatility, n, missingRatio
        )

        return DataCharacteristics(
            length=n,
            frequency=freq,
            period=seasonalPeriods[0] if seasonalPeriods else basePeriod,
            dateRange=dateRange,
            hasTrend=hasTrend,
            trendDirection=trendDirection,
            trendStrength=trendStrength,
            hasSeasonality=hasSeasonality,
            seasonalStrength=seasonalStrength,
            seasonalPeriods=seasonalPeriods,
            hasMultipleSeasonality=len(seasonalPeriods) > 1,
            isStationary=isStationary,
            volatility=volatility,
            volatilityLevel=volatilityLevel,
            missingRatio=missingRatio,
            outlierCount=outlierCount,
            outlierRatio=outlierRatio,
            predictabilityScore=predictabilityScore
        )

    def _detectFrequency(self, dates: pd.Series) -> Tuple[Frequency, int]:
        """Detect data frequency from date intervals"""
        if len(dates) < 2:
            return Frequency.DAILY, 7

        diffs = dates.diff().dropna()
        medianDays = diffs.median().days

        if medianDays >= 350:
            return Frequency.YEARLY, 1
        elif medianDays >= 85:
            return Frequency.QUARTERLY, 4
        elif medianDays >= 28:
            return Frequency.MONTHLY, 12
        elif medianDays >= 6:
            return Frequency.WEEKLY, 52
        elif medianDays >= 0.9:
            return Frequency.DAILY, 7
        else:
            return Frequency.HOURLY, 24

    def _detectSeasonalPeriods(
        self,
        values: np.ndarray,
        freq: Frequency,
        basePeriod: int
    ) -> List[int]:
        """Frequency-based seasonal period detection.

        Uses the known data frequency to determine the correct seasonal period.
        FFT-based detection was removed after E053 showed it produces spurious
        periods (e.g. 53, 144, 168 for Daily data) that degrade forecast accuracy
        across all M4 frequency groups.
        """
        defaultPeriods = {
            Frequency.DAILY: [7],
            Frequency.WEEKLY: [52],
            Frequency.MONTHLY: [12],
            Frequency.QUARTERLY: [4],
            Frequency.YEARLY: [1],
            Frequency.HOURLY: [24],
        }
        return defaultPeriods.get(freq, [basePeriod])

    def _analyzeSeasonality(
        self,
        values: np.ndarray,
        period: int
    ) -> Tuple[bool, float]:
        """Seasonality analysis"""
        n = len(values)

        if n < period * 2:
            return False, 0.0

        try:
            # Compute per-period means
            seasonalMeans = []
            for i in range(period):
                indices = list(range(i, n, period))
                if indices:
                    seasonalMeans.append(np.mean(values[indices]))

            if not seasonalMeans:
                return False, 0.0

            seasonalVar = np.var(seasonalMeans)
            totalVar = np.var(values)

            if totalVar < 1e-10:
                return False, 0.0

            seasonalStrength = min(seasonalVar / totalVar, 1.0)
            hasSeasonality = seasonalStrength > 0.1

            return hasSeasonality, seasonalStrength

        except Exception:
            return False, 0.0

    def _analyzeTrend(self, values: np.ndarray) -> Tuple[bool, str, float]:
        """Trend analysis - O(n) linear regression + t-test (replaces Mann-Kendall O(n^2))"""
        n = len(values)

        if n < 10:
            return False, "none", 0.0

        try:
            # O(n) linear regression t-test
            x = np.arange(n, dtype=np.float64)
            xMean = (n - 1) / 2.0
            yMean = np.mean(values)

            ssXY = np.sum((x - xMean) * (values - yMean))
            ssXX = np.sum((x - xMean) ** 2)

            if ssXX < 1e-10:
                return False, "none", 0.0

            slope = ssXY / ssXX
            residuals = values - (yMean + slope * (x - xMean))
            sse = np.sum(residuals ** 2)
            mse = sse / (n - 2) if n > 2 else 1.0
            seSlope = np.sqrt(mse / ssXX) if ssXX > 0 else 1.0

            if seSlope < 1e-10:
                return False, "none", 0.0

            tStat = slope / seSlope

            # Trend direction
            if tStat > 1.96:
                direction = "up"
            elif tStat < -1.96:
                direction = "down"
            else:
                direction = "none"

            # Trend strength (0 ~ 1)
            strength = min(abs(tStat) / 3.0, 1.0)

            # Stationarity check
            hasTrend = abs(tStat) > 1.96 or not self._checkStationarity(values)

            return hasTrend, direction, strength

        except Exception:
            return False, "none", 0.0

    def _checkStationarity(self, values: np.ndarray) -> bool:
        """Stationarity test (simplified ADF without statsmodels)"""
        n = len(values)
        if n < 20:
            return False

        try:
            diff = np.diff(values)
            lagged = values[:-1]

            laggedMean = np.mean(lagged)
            diffMean = np.mean(diff)

            numerator = np.sum((lagged - laggedMean) * (diff - diffMean))
            denominator = np.sum((lagged - laggedMean) ** 2)

            if denominator < 1e-10:
                return False

            rho = numerator / denominator
            se = np.sqrt(np.sum((diff - diffMean - rho * (lagged - laggedMean)) ** 2) / ((n - 2) * denominator))

            if se < 1e-10:
                return False

            tStat = rho / se

            criticalValues = {
                25: -3.00, 50: -2.93, 100: -2.89,
                250: -2.87, 500: -2.86
            }
            threshold = -2.89
            for size, cv in sorted(criticalValues.items()):
                if n <= size:
                    threshold = cv
                    break

            return tStat < threshold
        except Exception:
            return False

    def _analyzeVolatility(self, values: np.ndarray) -> Tuple[float, str]:
        """Volatility analysis"""
        if len(values) < 2:
            return 0.0, "normal"

        # Return-based volatility
        returns = np.diff(values) / (np.abs(values[:-1]) + 1e-10)
        volatility = float(np.std(returns))

        # Volatility level classification
        if volatility > 0.3:
            level = "high"
        elif volatility > 0.1:
            level = "normal"
        else:
            level = "low"

        return volatility, level

    def _detectOutliers(self, values: np.ndarray) -> Tuple[int, float]:
        """Outlier detection (IQR method)"""
        if len(values) < 4:
            return 0, 0.0

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lowerBound = q1 - 1.5 * iqr
        upperBound = q3 + 1.5 * iqr

        outlierMask = (values < lowerBound) | (values > upperBound)
        outlierCount = int(np.sum(outlierMask))
        outlierRatio = outlierCount / len(values) * 100

        return outlierCount, outlierRatio

    def _calculatePredictability(
        self,
        hasSeasonality: bool,
        seasonalStrength: float,
        hasTrend: bool,
        trendStrength: float,
        volatility: float,
        dataLength: int,
        missingRatio: float
    ) -> float:
        """Calculate predictability score (0 ~ 100)"""
        score = 50.0

        # Seasonality bonus
        if hasSeasonality:
            score += seasonalStrength * 20

        # Trend bonus
        if hasTrend:
            score += trendStrength * 10

        # Data length bonus
        if dataLength >= 100:
            score += 15
        elif dataLength >= 50:
            score += 10
        elif dataLength >= 30:
            score += 5
        elif dataLength < 20:
            score -= 15

        # Volatility penalty
        if volatility > 0.3:
            score -= 15
        elif volatility > 0.2:
            score -= 10

        # Missing value penalty
        if missingRatio > 10:
            score -= 20
        elif missingRatio > 5:
            score -= 10

        return max(0, min(100, score))

    def quickAnalyze(self, values: np.ndarray, period: int = 7) -> dict:
        """
        Quick analysis (values only, without DataFrame)

        Parameters
        ----------
        values : np.ndarray
            Time series values
        period : int
            Expected period

        Returns
        -------
        dict
            Simple analysis results
        """
        n = len(values)

        hasSeasonality, seasonalStrength = self._analyzeSeasonality(values, period)
        hasTrend, trendDirection, trendStrength = self._analyzeTrend(values)
        volatility, volatilityLevel = self._analyzeVolatility(values)

        return {
            'length': n,
            'period': period,
            'hasSeasonality': hasSeasonality,
            'seasonalStrength': seasonalStrength,
            'hasTrend': hasTrend,
            'trendDirection': trendDirection,
            'trendStrength': trendStrength,
            'volatility': volatility,
            'volatilityLevel': volatilityLevel
        }
