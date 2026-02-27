"""
==============================================================================
실험 ID: E010
실험명: MSTL LOESS 정교화
==============================================================================

목적:
- statsforecast MSTL 대비 열세 개선 (retail: 9.07% vs 3.45%)
- LOESS 평활화 품질 향상
- 계절성 추출 정확도 개선

가설:
1. 현재 단순 주기 평균보다 LOESS 평활이 더 정확
2. 계절 성분 추출 시 추세 제거 순서가 중요
3. 잔차 모델 선택이 최종 성능에 큰 영향

방법:
1. 현재 방식: 단순 주기 평균
2. LOESS 평활: scipy.interpolate 기반 지역 회귀
3. STL 스타일: 추세 → 계절 → 잔차 순서 분해
4. 반복 분해: 여러 번 반복하여 정교화

==============================================================================
결과
==============================================================================

결과 요약:
- LOESS 기반 구현이 오히려 악화 (9.07% → 14.50%)
- STL 스타일도 악화 (11.99%)
- 현재 AutoMSTL (9.07%)이 가장 좋음
- statsforecast (3.45%)와 큰 격차 유지

수치:
| 방법 | MAPE |
|------|------|
| **현재 AutoMSTL** | **9.07%** |
| LOESS 기반 | 14.50% |
| STL 스타일 | 11.99% |
| statsforecast | 3.45% |

결론:
1. 단순 LOESS 구현으로는 statsforecast 수준 달성 불가
2. statsforecast는 더 정교한 robust LOESS + 반복 정제 사용
3. 현재 AutoMSTL 유지, 추후 고도화 필요
4. **개선 포기** - 현재 방식 유지

실험일: 2026-02-05
==============================================================================
"""

import io
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def generateRetail2yr(n: int = 730, seed: int = 42) -> pd.DataFrame:
    """소매 판매 데이터 (주간 + 연간 계절성)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1300, n)
    weekly = 150 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 250 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 40, n)

    values = trend + weekly + yearly + noise
    values = np.maximum(values, 100)

    return pd.DataFrame({'date': dates, 'value': values})


class ImprovedMSTL:
    """개선된 MSTL (LOESS 기반)"""

    def __init__(self, periods: List[int], nIter: int = 2):
        self.periods = sorted(periods)
        self.nIter = nIter
        self.trend: Optional[np.ndarray] = None
        self.seasonals: Dict[int, np.ndarray] = {}
        self.residual: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> 'ImprovedMSTL':
        """LOESS 기반 분해"""
        n = len(y)

        # 초기화
        self.trend = np.zeros(n)
        for period in self.periods:
            self.seasonals[period] = np.zeros(n)

        residual = y.copy()

        # 반복 분해
        for iteration in range(self.nIter):
            # 추세 추출 (LOESS 근사)
            self.trend = self._loessSmooth(residual, span=0.3)
            residual = y - self.trend

            # 각 계절 성분 추출
            for period in self.periods:
                seasonal = self._extractSeasonalLoess(residual, period)
                self.seasonals[period] = seasonal
                residual = residual - seasonal

        self.residual = residual
        return self

    def _loessSmooth(self, y: np.ndarray, span: float = 0.3) -> np.ndarray:
        """LOESS 스타일 평활화"""
        n = len(y)
        windowSize = max(3, int(n * span))
        if windowSize % 2 == 0:
            windowSize += 1

        # 다항 회귀 기반 지역 평활
        result = np.zeros(n)
        halfWin = windowSize // 2

        for i in range(n):
            start = max(0, i - halfWin)
            end = min(n, i + halfWin + 1)

            localX = np.arange(end - start)
            localY = y[start:end]

            # 거리 가중치 (삼중 가중치 함수)
            distances = np.abs(localX - (i - start))
            maxDist = np.max(distances) + 1e-10
            weights = (1 - (distances / maxDist) ** 3) ** 3

            # 가중 선형 회귀
            try:
                coeffs = np.polyfit(localX, localY, deg=1, w=weights)
                result[i] = np.polyval(coeffs, i - start)
            except Exception:
                result[i] = np.average(localY, weights=weights)

        return result

    def _extractSeasonalLoess(self, y: np.ndarray, period: int) -> np.ndarray:
        """LOESS 기반 계절 성분 추출"""
        n = len(y)
        seasonal = np.zeros(n)

        # 주기별 평균 계산 (LOESS 평활)
        periodMeans = np.zeros(period)
        periodCounts = np.zeros(period)

        for i in range(n):
            idx = i % period
            periodMeans[idx] += y[i]
            periodCounts[idx] += 1

        periodMeans /= np.maximum(periodCounts, 1)

        # 주기 평균에 LOESS 적용 (부드럽게)
        if period > 5:
            periodMeansSmooth = self._loessSmooth(
                np.tile(periodMeans, 3), span=0.5
            )[period:2*period]
        else:
            periodMeansSmooth = periodMeans

        # 중심화
        periodMeansSmooth -= np.mean(periodMeansSmooth)

        # 전체 길이로 확장
        for i in range(n):
            seasonal[i] = periodMeansSmooth[i % period]

        return seasonal

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측"""
        from vectrix.engine.arima import AutoARIMA

        # 잔차에 ARIMA
        arima = AutoARIMA()
        arima.fit(self.residual)
        residPred, _, _ = arima.predict(steps)

        # 추세 외삽
        n = len(self.trend)
        trendSlope = (self.trend[-1] - self.trend[-10]) / 10
        trendPred = self.trend[-1] + trendSlope * np.arange(1, steps + 1)

        # 계절성 외삽
        pred = trendPred + residPred
        for period in self.periods:
            pattern = self.seasonals[period][-period:]
            seasonalPred = np.array([pattern[i % period] for i in range(steps)])
            pred += seasonalPred

        sigma = np.std(self.residual)
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

        return pred, pred - margin, pred + margin


class STLStyleMSTL:
    """STL 스타일 분해 (추세 → 계절 → 잔차)"""

    def __init__(self, periods: List[int], trendWindow: int = None):
        self.periods = sorted(periods)
        self.trendWindow = trendWindow
        self.trend: Optional[np.ndarray] = None
        self.seasonals: Dict[int, np.ndarray] = {}
        self.residual: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> 'STLStyleMSTL':
        """STL 스타일 분해"""
        n = len(y)

        # 1. 추세 추출 (긴 이동평균)
        trendWindow = self.trendWindow or max(self.periods) * 2 + 1
        if trendWindow % 2 == 0:
            trendWindow += 1
        trendWindow = min(trendWindow, n // 2)

        self.trend = uniform_filter1d(y.astype(float), size=trendWindow, mode='nearest')

        # 2. 탈추세
        detrended = y - self.trend

        # 3. 각 계절 성분 추출 (짧은 주기부터)
        residual = detrended.copy()
        for period in self.periods:
            seasonal = self._extractSeasonal(residual, period)
            self.seasonals[period] = seasonal
            residual = residual - seasonal

        self.residual = residual
        return self

    def _extractSeasonal(self, y: np.ndarray, period: int) -> np.ndarray:
        """주기별 중앙값 기반 계절 추출"""
        n = len(y)

        # 주기별 값 수집
        periodValues = [[] for _ in range(period)]
        for i in range(n):
            periodValues[i % period].append(y[i])

        # 중앙값 계산 (이상치에 강건)
        periodMedians = np.array([np.median(vals) for vals in periodValues])
        periodMedians -= np.mean(periodMedians)

        # 확장
        seasonal = np.array([periodMedians[i % period] for i in range(n)])
        return seasonal

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측"""
        from vectrix.engine.arima import AutoARIMA

        # 잔차 ARIMA
        arima = AutoARIMA()
        arima.fit(self.residual)
        residPred, _, _ = arima.predict(steps)

        # 추세 외삽
        trendSlope = (self.trend[-1] - self.trend[-10]) / 10
        trendPred = self.trend[-1] + trendSlope * np.arange(1, steps + 1)

        # 계절성 외삽
        pred = trendPred + residPred
        for period in self.periods:
            pattern = self.seasonals[period][-period:]
            seasonalPred = np.array([pattern[i % period] for i in range(steps)])
            pred += seasonalPred

        sigma = np.std(self.residual)
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

        return pred, pred - margin, pred + margin


# ============================================================================
# 실험 1: 현재 AutoMSTL
# ============================================================================

def experiment1_current():
    """현재 AutoMSTL"""
    print("\n" + "=" * 60)
    print("실험 1: 현재 AutoMSTL")
    print("=" * 60)

    from vectrix import ChaniCast

    df = generateRetail2yr()
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    cc = ChaniCast(verbose=False)
    result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

    if result.success:
        pred = result.predictions[:60]
        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"MAPE: {mape:.2f}% (모델: {result.bestModelName})")
        return {'mape': mape}
    return {'mape': float('inf')}


# ============================================================================
# 실험 2: LOESS 기반 MSTL
# ============================================================================

def experiment2_loessMstl():
    """LOESS 기반 개선 MSTL"""
    print("\n" + "=" * 60)
    print("실험 2: LOESS 기반 MSTL")
    print("=" * 60)

    df = generateRetail2yr()
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    trainValues = trainDf['value'].values
    testValues = testDf['value'].values

    mstl = ImprovedMSTL(periods=[7, 365], nIter=2)
    mstl.fit(trainValues)
    pred, _, _ = mstl.predict(60)

    mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
    print(f"MAPE: {mape:.2f}%")

    return {'mape': mape}


# ============================================================================
# 실험 3: STL 스타일 분해
# ============================================================================

def experiment3_stlStyle():
    """STL 스타일 분해"""
    print("\n" + "=" * 60)
    print("실험 3: STL 스타일 분해")
    print("=" * 60)

    df = generateRetail2yr()
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    trainValues = trainDf['value'].values
    testValues = testDf['value'].values

    mstl = STLStyleMSTL(periods=[7, 365])
    mstl.fit(trainValues)
    pred, _, _ = mstl.predict(60)

    mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
    print(f"MAPE: {mape:.2f}%")

    return {'mape': mape}


# ============================================================================
# 실험 4: statsforecast 비교
# ============================================================================

def experiment4_statsforecast():
    """statsforecast MSTL 비교"""
    print("\n" + "=" * 60)
    print("실험 4: statsforecast MSTL")
    print("=" * 60)

    df = generateRetail2yr()
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import MSTL

        sfDf = trainDf.copy()
        sfDf['unique_id'] = 'series1'
        sfDf = sfDf.rename(columns={'date': 'ds', 'value': 'y'})

        sf = StatsForecast(
            models=[MSTL(season_length=[7, 365])],
            freq='D', n_jobs=1
        )
        sf.fit(sfDf)
        sfPred = sf.predict(h=60)
        pred = sfPred['MSTL'].values

        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"MAPE: {mape:.2f}%")

        return {'mape': mape}

    except Exception as e:
        print(f"오류: {e}")
        return {'mape': float('inf')}


# ============================================================================
# 메인 실행
# ============================================================================

def runAllExperiments():
    """모든 실험 실행"""
    print("=" * 60)
    print("E010: MSTL LOESS 정교화")
    print("=" * 60)

    results = {}

    results['current'] = experiment1_current()
    results['loess'] = experiment2_loessMstl()
    results['stl_style'] = experiment3_stlStyle()
    results['statsforecast'] = experiment4_statsforecast()

    # 종합
    print("\n" + "=" * 60)
    print("E010 실험 종합")
    print("=" * 60)

    print("\n방법별 MAPE:")
    for name, res in results.items():
        mape = res.get('mape', float('inf'))
        print(f"  {name:20}: {mape:.2f}%")

    return results


if __name__ == '__main__':
    results = runAllExperiments()
