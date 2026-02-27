"""
AutoAnalyzer: 시계열 데이터 자동 분석기

데이터의 특성(추세, 계절성, 정상성, 변동성 등)을 자동으로 분석합니다.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from ..types import DataCharacteristics, Frequency


class AutoAnalyzer:
    """
    시계열 데이터 자동 분석기

    데이터의 주기, 추세, 계절성, 정상성, 변동성 등을 자동으로 분석합니다.
    """

    def analyze(
        self,
        df: pd.DataFrame,
        dateCol: str,
        valueCol: str
    ) -> DataCharacteristics:
        """
        데이터 특성 종합 분석

        Parameters
        ----------
        df : pd.DataFrame
            시계열 데이터프레임
        dateCol : str
            날짜 컬럼명
        valueCol : str
            값 컬럼명

        Returns
        -------
        DataCharacteristics
            분석된 데이터 특성
        """
        dates = pd.to_datetime(df[dateCol])
        values = df[valueCol].values.astype(float)
        n = len(values)

        # 기본 정보
        freq, basePeriod = self._detectFrequency(dates)
        dateRange = (
            dates.iloc[0].strftime('%Y-%m-%d'),
            dates.iloc[-1].strftime('%Y-%m-%d')
        )

        # 계절성 분석
        seasonalPeriods = self._detectSeasonalPeriods(values, freq, basePeriod)
        hasSeasonality, seasonalStrength = self._analyzeSeasonality(
            values, seasonalPeriods[0] if seasonalPeriods else basePeriod
        )

        # 추세 분석
        hasTrend, trendDirection, trendStrength = self._analyzeTrend(values)

        # 정상성 검정
        isStationary = self._checkStationarity(values)

        # 변동성 분석
        volatility, volatilityLevel = self._analyzeVolatility(values)

        # 품질 분석
        missingRatio = df[valueCol].isna().sum() / n * 100
        outlierCount, outlierRatio = self._detectOutliers(values)

        # 예측 가능성 점수
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
        """날짜 간격으로 데이터 주기 감지"""
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
        """FFT 기반 계절 주기 감지"""
        n = len(values)

        if n < 20:
            return [basePeriod]

        periods = []

        try:
            # 추세 제거
            detrended = values - np.linspace(values[0], values[-1], n)

            # FFT
            fft = np.fft.fft(detrended)
            magnitudes = np.abs(fft)
            magnitudes[0] = 0
            magnitudes[n // 2:] = 0

            # 유의한 주파수 탐지
            threshold = np.mean(magnitudes) + 2 * np.std(magnitudes)

            for i in range(1, min(n // 2, 100)):
                if magnitudes[i] > threshold:
                    period = int(round(n / i))
                    if 2 <= period <= n // 2:
                        periods.append(period)

            periods = sorted(set(periods))[:3]

        except Exception:
            pass

        # 기본 주기 추가
        if not periods:
            defaultPeriods = {
                Frequency.DAILY: [7],
                Frequency.WEEKLY: [13, 52],
                Frequency.MONTHLY: [12],
                Frequency.QUARTERLY: [4],
                Frequency.YEARLY: [1],
                Frequency.HOURLY: [24]
            }
            periods = defaultPeriods.get(freq, [basePeriod])

        return periods

    def _analyzeSeasonality(
        self,
        values: np.ndarray,
        period: int
    ) -> Tuple[bool, float]:
        """계절성 분석"""
        n = len(values)

        if n < period * 2:
            return False, 0.0

        try:
            # 주기별 평균 계산
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
        """추세 분석 — O(n) 선형회귀 + t-검정 (Mann-Kendall O(n²) 대체)"""
        n = len(values)

        if n < 10:
            return False, "none", 0.0

        try:
            # O(n) 선형회귀 t-검정
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

            # 추세 방향
            if tStat > 1.96:
                direction = "up"
            elif tStat < -1.96:
                direction = "down"
            else:
                direction = "none"

            # 추세 강도 (0 ~ 1)
            strength = min(abs(tStat) / 3.0, 1.0)

            # 정상성 확인
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
        """변동성 분석"""
        if len(values) < 2:
            return 0.0, "normal"

        # 수익률 기반 변동성
        returns = np.diff(values) / (np.abs(values[:-1]) + 1e-10)
        volatility = float(np.std(returns))

        # 변동성 수준 판단
        if volatility > 0.3:
            level = "high"
        elif volatility > 0.1:
            level = "normal"
        else:
            level = "low"

        return volatility, level

    def _detectOutliers(self, values: np.ndarray) -> Tuple[int, float]:
        """이상치 탐지 (IQR 방법)"""
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
        """예측 가능성 점수 계산 (0 ~ 100)"""
        score = 50.0

        # 계절성 보너스
        if hasSeasonality:
            score += seasonalStrength * 20

        # 추세 보너스
        if hasTrend:
            score += trendStrength * 10

        # 데이터 길이 보너스
        if dataLength >= 100:
            score += 15
        elif dataLength >= 50:
            score += 10
        elif dataLength >= 30:
            score += 5
        elif dataLength < 20:
            score -= 15

        # 변동성 페널티
        if volatility > 0.3:
            score -= 15
        elif volatility > 0.2:
            score -= 10

        # 결측치 페널티
        if missingRatio > 10:
            score -= 20
        elif missingRatio > 5:
            score -= 10

        return max(0, min(100, score))

    def quickAnalyze(self, values: np.ndarray, period: int = 7) -> dict:
        """
        빠른 분석 (DataFrame 없이 값만으로)

        Parameters
        ----------
        values : np.ndarray
            시계열 값
        period : int
            예상 주기

        Returns
        -------
        dict
            간단한 분석 결과
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
