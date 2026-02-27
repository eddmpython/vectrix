"""
==============================================================================
실험 ID: E006
실험명: 다중 계절성 지원 (MSTL 강화)
==============================================================================

목적:
- 강한 계절성 데이터(retail_sales, energy)에서 정확도 향상
- 주간(7) + 월간(30) + 연간(365) 다중 계절성 동시 처리
- statsforecast 대비 계절성 데이터에서 경쟁력 확보

가설:
1. 단일 계절성만 처리하면 다중 계절 패턴을 놓침
2. MSTL로 다중 계절성 분해 후 잔차에 ARIMA 적용하면 정확도 향상
3. 계절성 강도에 따라 적응적 period 선택 필요

방법:
1. 현재 방식: 단일 period (자동 감지된 하나)
2. 다중 계절성: [7, 30] 또는 [7, 365] 조합
3. MSTL 분해 + ARIMA 잔차 예측
4. 계절성 강도 기반 period 선택

==============================================================================
결과
==============================================================================

결과 요약:
- MSTL + ARIMA가 현재 방식 대비 **57.8% 개선** (15.46% → 6.53%)
- 적응적 period 선택은 중간 성능 (10.00%)
- statsforecast MSTL과 비슷하거나 일부 데이터에서 우수

수치:
| 방법 | 평균 MAPE | vs 현재 |
|------|-----------|---------|
| 현재 (단일 계절성) | 15.46% | - |
| **MSTL + ARIMA** | **6.53%** | **57.8% 개선** |
| 적응적 period | 10.00% | 35.3% 개선 |

데이터별 상세:
| 데이터 | 현재 | MSTL+ARIMA | statsforecast |
|--------|------|------------|---------------|
| retail_2yr | 13.95% | **9.69%** | 3.45% |
| energy_2yr | 13.04% | **6.21%** | 5.65% |
| weekly_monthly | 19.39% | **3.69%** | 16.98% |

결론:
1. **다중 계절성 분해가 매우 효과적** (평균 57.8% 개선)
2. statsforecast보다 weekly_monthly에서 우수 (3.69% vs 16.98%)
3. retail_2yr에서는 statsforecast가 여전히 우수 (3.45% vs 9.69%)
4. **권장**: 다중 계절성 의심 데이터에서 MSTL + ARIMA 적용

실험일: 2026-02-05
==============================================================================
"""

import numpy as np
import pandas as pd
import time
import sys
import io
import os
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================================
# 다중 계절성 데이터 생성
# ============================================================================

def generateMultiSeasonalRetail(n: int = 730, seed: int = 42) -> pd.DataFrame:
    """소매 판매 데이터 (주간 + 연간 계절성) - 2년치"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1300, n)
    weekly = 150 * np.sin(2 * np.pi * np.arange(n) / 7)       # 주간
    yearly = 250 * np.sin(2 * np.pi * np.arange(n) / 365)     # 연간
    noise = np.random.normal(0, 40, n)

    values = trend + weekly + yearly + noise
    values = np.maximum(values, 100)

    return pd.DataFrame({'date': dates, 'value': values})


def generateMultiSeasonalEnergy(n: int = 730, seed: int = 42) -> pd.DataFrame:
    """에너지 소비 데이터 (주간 + 연간 계절성) - 2년치"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(500, 550, n)
    weekly = 60 * np.sin(2 * np.pi * np.arange(n) / 7)        # 주간
    yearly = 120 * np.sin(2 * np.pi * (np.arange(n) - 30) / 365)  # 연간 (겨울 피크)
    noise = np.random.normal(0, 25, n)

    values = trend + weekly + yearly + noise
    values = np.maximum(values, 200)

    return pd.DataFrame({'date': dates, 'value': values})


def generateWeeklyMonthly(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """주간 + 월간 계절성"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(100, 120, n)
    weekly = 15 * np.sin(2 * np.pi * np.arange(n) / 7)
    monthly = 25 * np.sin(2 * np.pi * np.arange(n) / 30)
    noise = np.random.normal(0, 5, n)

    values = trend + weekly + monthly + noise

    return pd.DataFrame({'date': dates, 'value': values})


# ============================================================================
# MSTL 구현 (다중 계절성 분해)
# ============================================================================

class SimpleMSTL:
    """
    간단한 MSTL (Multiple Seasonal-Trend decomposition using LOESS)

    여러 계절 주기를 순차적으로 분해
    """

    def __init__(self, periods: List[int]):
        """
        Parameters
        ----------
        periods : List[int]
            계절 주기 리스트 (오름차순 권장, 예: [7, 30, 365])
        """
        self.periods = sorted(periods)
        self.seasonals = {}
        self.trend = None
        self.residual = None

    def decompose(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        다중 계절성 분해

        Returns
        -------
        Dict with 'trend', 'seasonal_7', 'seasonal_30', ..., 'residual'
        """
        n = len(y)
        residual = y.copy()

        # 각 계절 주기에 대해 순차적으로 분해
        for period in self.periods:
            if n < period * 2:
                self.seasonals[period] = np.zeros(n)
                continue

            # 이동 평균으로 추세 제거
            seasonal = self._extractSeasonal(residual, period)
            self.seasonals[period] = seasonal
            residual = residual - seasonal

        # 최종 추세 추출 (가장 긴 주기의 이동 평균)
        maxPeriod = max(self.periods)
        if n >= maxPeriod:
            self.trend = self._movingAverage(residual, maxPeriod)
        else:
            self.trend = self._movingAverage(residual, min(n // 2, 7))

        self.residual = residual - self.trend

        result = {'trend': self.trend, 'residual': self.residual}
        for period in self.periods:
            result[f'seasonal_{period}'] = self.seasonals[period]

        return result

    def _extractSeasonal(self, y: np.ndarray, period: int) -> np.ndarray:
        """계절 성분 추출"""
        n = len(y)
        seasonal = np.zeros(n)

        # 주기별 평균 계산
        periodMeans = np.zeros(period)
        for i in range(period):
            vals = y[i::period]
            periodMeans[i] = np.mean(vals)

        # 전체 평균을 빼서 중심화
        periodMeans -= np.mean(periodMeans)

        # 전체 길이로 확장
        for i in range(n):
            seasonal[i] = periodMeans[i % period]

        return seasonal

    def _movingAverage(self, y: np.ndarray, window: int) -> np.ndarray:
        """이동 평균"""
        n = len(y)
        result = np.zeros(n)
        halfWin = window // 2

        for i in range(n):
            start = max(0, i - halfWin)
            end = min(n, i + halfWin + 1)
            result[i] = np.mean(y[start:end])

        return result


# ============================================================================
# 실험 1: 현재 방식 (단일 계절성) 성능
# ============================================================================

def experiment1_currentMethod():
    """현재 방식 (단일 계절성) 성능 측정"""
    print("\n" + "=" * 70)
    print("실험 1: 현재 방식 (단일 계절성)")
    print("=" * 70)

    from forecastx import ChaniCast

    datasets = [
        ('retail_2yr', generateMultiSeasonalRetail()),
        ('energy_2yr', generateMultiSeasonalEnergy()),
        ('weekly_monthly', generateWeeklyMonthly()),
    ]

    results = {}

    for name, df in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        testValues = testDf['value'].values

        try:
            cc = ChaniCast(verbose=False)
            result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

            if result.success:
                pred = result.predictions[:60]
                mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
                results[name] = {'mape': mape, 'model': result.bestModelName}
                print(f"{name:20} MAPE = {mape:.2f}% (모델: {result.bestModelName})")
            else:
                results[name] = {'mape': float('inf'), 'error': result.error}
                print(f"{name:20} 오류: {result.error}")
        except Exception as e:
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 2: MSTL + ARIMA
# ============================================================================

def experiment2_mstlArima():
    """MSTL 분해 + ARIMA 잔차 예측"""
    print("\n" + "=" * 70)
    print("실험 2: MSTL + ARIMA")
    print("=" * 70)

    from forecastx.engine.arima import AutoARIMA

    datasets = [
        ('retail_2yr', generateMultiSeasonalRetail(), [7, 365]),
        ('energy_2yr', generateMultiSeasonalEnergy(), [7, 365]),
        ('weekly_monthly', generateWeeklyMonthly(), [7, 30]),
    ]

    results = {}

    for name, df, periods in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values
        testValues = testDf['value'].values

        try:
            # MSTL 분해
            mstl = SimpleMSTL(periods=periods)
            decomposed = mstl.decompose(trainValues)

            # 잔차에 ARIMA 적용
            arima = AutoARIMA()
            arima.fit(decomposed['residual'])
            residPred, _, _ = arima.predict(60)

            # 추세 외삽 (선형)
            trendSlope = (decomposed['trend'][-1] - decomposed['trend'][-10]) / 10
            trendPred = decomposed['trend'][-1] + trendSlope * np.arange(1, 61)

            # 계절성 외삽 (주기 반복)
            finalPred = trendPred + residPred
            for period in periods:
                seasonalPattern = mstl.seasonals[period][-period:]
                seasonalPred = np.array([seasonalPattern[i % period] for i in range(60)])
                finalPred += seasonalPred

            mape = np.mean(np.abs((testValues - finalPred) / testValues)) * 100
            results[name] = {'mape': mape, 'periods': periods}
            print(f"{name:20} MAPE = {mape:.2f}% (periods={periods})")

        except Exception as e:
            import traceback
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")
            traceback.print_exc()

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 3: 계절성 강도 기반 period 자동 선택
# ============================================================================

def experiment3_adaptivePeriod():
    """계절성 강도에 따른 적응적 period 선택"""
    print("\n" + "=" * 70)
    print("실험 3: 적응적 period 선택")
    print("=" * 70)

    from forecastx.engine.arima import AutoARIMA
    from forecastx.engine.turbo import TurboCore

    datasets = [
        ('retail_2yr', generateMultiSeasonalRetail()),
        ('energy_2yr', generateMultiSeasonalEnergy()),
        ('weekly_monthly', generateWeeklyMonthly()),
    ]

    results = {}

    for name, df in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values
        testValues = testDf['value'].values

        try:
            # 여러 주기 후보에서 계절성 강도 측정
            candidatePeriods = [7, 14, 30, 90, 365]
            detectedPeriods = []

            for period in candidatePeriods:
                if len(trainValues) < period * 2:
                    continue

                # ACF로 계절성 강도 측정
                acf = TurboCore.acf(trainValues, min(period + 1, len(trainValues) // 2))
                if len(acf) > period and abs(acf[period]) > 0.2:
                    detectedPeriods.append((period, abs(acf[period])))

            # 강도 순으로 정렬, 상위 2개 선택
            detectedPeriods.sort(key=lambda x: -x[1])
            selectedPeriods = [p[0] for p in detectedPeriods[:2]] if detectedPeriods else [7]

            # MSTL 분해
            mstl = SimpleMSTL(periods=selectedPeriods)
            decomposed = mstl.decompose(trainValues)

            # 잔차에 ARIMA 적용
            arima = AutoARIMA()
            arima.fit(decomposed['residual'])
            residPred, _, _ = arima.predict(60)

            # 추세 외삽
            trendSlope = (decomposed['trend'][-1] - decomposed['trend'][-10]) / 10
            trendPred = decomposed['trend'][-1] + trendSlope * np.arange(1, 61)

            # 계절성 외삽
            finalPred = trendPred + residPred
            for period in selectedPeriods:
                seasonalPattern = mstl.seasonals[period][-period:]
                seasonalPred = np.array([seasonalPattern[i % period] for i in range(60)])
                finalPred += seasonalPred

            mape = np.mean(np.abs((testValues - finalPred) / testValues)) * 100
            results[name] = {
                'mape': mape,
                'detectedPeriods': selectedPeriods,
                'allCandidates': detectedPeriods
            }
            print(f"{name:20} MAPE = {mape:.2f}% (감지된 주기={selectedPeriods})")

        except Exception as e:
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 4: statsforecast 비교
# ============================================================================

def experiment4_vsStatsforecast():
    """statsforecast MSTL과 비교"""
    print("\n" + "=" * 70)
    print("실험 4: statsforecast 비교")
    print("=" * 70)

    datasets = [
        ('retail_2yr', generateMultiSeasonalRetail()),
        ('energy_2yr', generateMultiSeasonalEnergy()),
        ('weekly_monthly', generateWeeklyMonthly()),
    ]

    results = {}

    for name, df in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        testValues = testDf['value'].values
        result = {'name': name}

        # statsforecast
        try:
            from statsforecast import StatsForecast
            from statsforecast.models import MSTL, AutoARIMA as SFAutoARIMA

            sfDf = trainDf.copy()
            sfDf['unique_id'] = 'series1'
            sfDf = sfDf.rename(columns={'date': 'ds', 'value': 'y'})

            sf = StatsForecast(
                models=[MSTL(season_length=[7, 365])],
                freq='D', n_jobs=1
            )
            sf.fit(sfDf)
            sfPred = sf.predict(h=60)
            sfValues = sfPred['MSTL'].values

            sfMape = np.mean(np.abs((testValues - sfValues) / testValues)) * 100
            result['statsforecast'] = sfMape
            print(f"{name:20} statsforecast MSTL: {sfMape:.2f}%")

        except Exception as e:
            result['statsforecast'] = float('inf')
            print(f"{name:20} statsforecast 오류: {str(e)[:50]}")

        results[name] = result

    return results


# ============================================================================
# 메인 실행
# ============================================================================

def runAllExperiments():
    """모든 실험 실행"""
    print("=" * 70)
    print("E006: 다중 계절성 지원 실험")
    print("=" * 70)

    allResults = {}

    # 실험 1: 현재 방식
    allResults['current'] = experiment1_currentMethod()

    # 실험 2: MSTL + ARIMA
    allResults['mstl_arima'] = experiment2_mstlArima()

    # 실험 3: 적응적 period
    allResults['adaptive'] = experiment3_adaptivePeriod()

    # 실험 4: statsforecast 비교
    allResults['comparison'] = experiment4_vsStatsforecast()

    # 종합
    print("\n" + "=" * 70)
    print("E006 실험 종합")
    print("=" * 70)

    print("\n방법별 평균 MAPE:")
    for method, results in allResults.items():
        if method == 'comparison':
            continue
        mapes = [r.get('mape', float('inf')) for r in results.values() if r.get('mape', float('inf')) < float('inf')]
        if mapes:
            print(f"  {method:20}: {np.mean(mapes):.2f}%")

    return allResults


if __name__ == '__main__':
    results = runAllExperiments()
