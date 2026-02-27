"""
==============================================================================
실험 ID: E009
실험명: 정기 패턴 감지 (Manufacturing 개선)
==============================================================================

목적:
- manufacturing 데이터 (+14.4%p 열세) 개선
- 정기 점검 드롭 같은 반복 패턴 감지 및 예측 반영
- statsforecast 8.63% 수준 달성 목표

가설:
1. 90일 주기 드롭 패턴을 감지하면 예측에 반영 가능
2. 드롭 구간을 제외하고 학습 후, 드롭 패턴을 다시 적용하면 개선
3. Fourier 변환으로 장주기 패턴 감지 가능

방법:
1. 현재 방식: 패턴 무시
2. 드롭 감지: 급격한 하락 구간 감지 → 제외 후 학습 → 패턴 재적용
3. Fourier 분석: 장주기 패턴 감지
4. 주기적 더미 변수: 90일 주기 더미 추가

==============================================================================
결과
==============================================================================

결과 요약:
- 드롭 감지 + 패턴 반영 방식이 **61.3% 개선** (5.87% → 2.27%)
- statsforecast (1.76%)에 근접
- Fourier 분석은 실패 (87.36%)

수치:
| 방법 | MAPE | vs 현재 |
|------|------|---------|
| 현재 | 5.87% | - |
| **드롭 감지** | **2.27%** | **61.3% 개선** |
| Fourier | 87.36% | 실패 |
| statsforecast | 1.76% | 목표 |

결론:
1. 드롭 구간 감지 → 보간으로 대체 → 학습 → 패턴 재적용 방식이 효과적
2. statsforecast 수준에 근접 (2.27% vs 1.76%)
3. ChaniCast 통합 권장

실험일: 2026-02-05
==============================================================================
"""

import io
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def generateManufacturing(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """제조 생산량 (정기 점검 드롭)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1100, n)
    noise = np.random.normal(0, 30, n)
    values = trend + noise

    # 90일마다 1주일 점검 (30% 감소)
    dropPeriod = 90
    dropDuration = 7
    dropRatio = 0.7

    for i in range(0, n, dropPeriod):
        if i + dropDuration < n:
            values[i:i+dropDuration] *= dropRatio

    return pd.DataFrame({'date': dates, 'value': values})


class PeriodicDropDetector:
    """정기 드롭 패턴 감지"""

    def __init__(self, minDropRatio: float = 0.5, minDropDuration: int = 3):
        self.minDropRatio = minDropRatio
        self.minDropDuration = minDropDuration
        self.detectedDrops: List[Tuple[int, int]] = []
        self.dropPeriod: Optional[int] = None
        self.dropDuration: Optional[int] = None
        self.dropRatio: Optional[float] = None

    def detect(self, y: np.ndarray) -> bool:
        """드롭 패턴 감지"""
        n = len(y)

        # 이동 평균 계산
        window = min(14, n // 10)
        ma = np.convolve(y, np.ones(window)/window, mode='same')

        # 드롭 구간 찾기 (이동평균 대비 급락)
        dropMask = y < ma * self.minDropRatio

        # 연속 드롭 구간 찾기
        drops = []
        inDrop = False
        dropStart = 0

        for i in range(n):
            if dropMask[i] and not inDrop:
                inDrop = True
                dropStart = i
            elif not dropMask[i] and inDrop:
                inDrop = False
                if i - dropStart >= self.minDropDuration:
                    drops.append((dropStart, i))

        if inDrop and n - dropStart >= self.minDropDuration:
            drops.append((dropStart, n))

        self.detectedDrops = drops

        # 주기 분석
        if len(drops) >= 2:
            intervals = [drops[i+1][0] - drops[i][0] for i in range(len(drops)-1)]
            if intervals:
                self.dropPeriod = int(np.median(intervals))
                durations = [end - start for start, end in drops]
                self.dropDuration = int(np.median(durations))

                # 드롭 비율 계산
                dropValues = []
                normalValues = []
                for start, end in drops:
                    dropValues.extend(y[start:end])
                    # 드롭 전 정상 구간
                    normalStart = max(0, start - 7)
                    normalValues.extend(y[normalStart:start])

                if normalValues:
                    self.dropRatio = np.mean(dropValues) / np.mean(normalValues)

                return True

        return False

    def removeDrops(self, y: np.ndarray) -> np.ndarray:
        """드롭 구간을 보간으로 대체"""
        result = y.copy()

        for start, end in self.detectedDrops:
            # 드롭 전후 값으로 선형 보간
            beforeVal = y[max(0, start-1)]
            afterVal = y[min(len(y)-1, end)]

            for i in range(start, min(end, len(y))):
                ratio = (i - start) / max(1, end - start)
                result[i] = beforeVal + ratio * (afterVal - beforeVal)

        return result

    def applyDropPattern(self, predictions: np.ndarray, startIdx: int) -> np.ndarray:
        """예측에 드롭 패턴 적용"""
        if self.dropPeriod is None or self.dropRatio is None:
            return predictions

        result = predictions.copy()
        n = len(predictions)

        # 다음 드롭 시점 계산
        if self.detectedDrops:
            lastDropStart = self.detectedDrops[-1][0]
            nextDropOffset = self.dropPeriod - ((startIdx - lastDropStart) % self.dropPeriod)
        else:
            nextDropOffset = self.dropPeriod

        # 드롭 적용
        i = nextDropOffset
        while i < n:
            dropEnd = min(i + (self.dropDuration or 7), n)
            result[i:dropEnd] *= self.dropRatio
            i += self.dropPeriod

        return result


# ============================================================================
# 실험 1: 현재 방식
# ============================================================================

def experiment1_current():
    """현재 방식 (드롭 패턴 무시)"""
    print("\n" + "=" * 60)
    print("실험 1: 현재 방식")
    print("=" * 60)

    from vectrix import ChaniCast

    df = generateManufacturing(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    cc = ChaniCast(verbose=False)
    result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

    if result.success:
        pred = result.predictions[:60]
        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"MAPE: {mape:.2f}% (모델: {result.bestModelName})")
        return {'mape': mape, 'model': result.bestModelName}
    else:
        print(f"오류: {result.error}")
        return {'mape': float('inf')}


# ============================================================================
# 실험 2: 드롭 감지 + 제거 후 학습 + 패턴 재적용
# ============================================================================

def experiment2_dropAware():
    """드롭 감지 및 패턴 반영"""
    print("\n" + "=" * 60)
    print("실험 2: 드롭 감지 + 패턴 반영")
    print("=" * 60)

    from vectrix.engine.theta import OptimizedTheta

    df = generateManufacturing(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    trainValues = trainDf['value'].values
    testValues = testDf['value'].values

    # 드롭 감지
    detector = PeriodicDropDetector(minDropRatio=0.8, minDropDuration=5)
    hasPattern = detector.detect(trainValues)

    print(f"드롭 패턴 감지: {hasPattern}")
    if hasPattern:
        print(f"  - 감지된 드롭 수: {len(detector.detectedDrops)}")
        print(f"  - 추정 주기: {detector.dropPeriod}일")
        print(f"  - 추정 지속기간: {detector.dropDuration}일")
        print(f"  - 추정 비율: {detector.dropRatio:.2f}")

    # 드롭 제거 후 학습
    cleanedValues = detector.removeDrops(trainValues)

    # Theta로 예측
    model = OptimizedTheta(period=7)
    model.fit(cleanedValues)
    pred, _, _ = model.predict(60)

    # 드롭 패턴 재적용
    if hasPattern:
        pred = detector.applyDropPattern(pred, len(trainValues))

    mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
    print(f"MAPE: {mape:.2f}%")

    return {'mape': mape, 'hasPattern': hasPattern, 'dropPeriod': detector.dropPeriod}


# ============================================================================
# 실험 3: Fourier 분석으로 장주기 감지
# ============================================================================

def experiment3_fourier():
    """Fourier 분석으로 장주기 패턴 감지"""
    print("\n" + "=" * 60)
    print("실험 3: Fourier 분석")
    print("=" * 60)

    from vectrix.engine.theta import OptimizedTheta

    df = generateManufacturing(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    trainValues = trainDf['value'].values
    testValues = testDf['value'].values

    n = len(trainValues)

    # FFT
    fft = np.fft.fft(trainValues - np.mean(trainValues))
    freqs = np.fft.fftfreq(n)

    # 주요 주기 찾기 (7일, 30일, 90일 근처)
    targetPeriods = [7, 30, 90]
    detectedPeriods = []

    for period in targetPeriods:
        freq = 1.0 / period
        idx = np.argmin(np.abs(freqs[:n//2] - freq))
        power = np.abs(fft[idx])
        if power > np.mean(np.abs(fft[:n//2])) * 2:
            detectedPeriods.append(period)
            print(f"  주기 {period}일 감지 (power: {power:.1f})")

    # 장주기(90일) 성분 추출
    if 90 in detectedPeriods:
        # 90일 성분만 추출
        fftFiltered = fft.copy()
        for i in range(n):
            if freqs[i] != 0 and abs(1/freqs[i] - 90) > 10:
                fftFiltered[i] *= 0.5  # 90일 외 성분 감쇠

        longCycle = np.real(np.fft.ifft(fftFiltered))

        # 잔차 예측
        residual = trainValues - longCycle
        model = OptimizedTheta(period=7)
        model.fit(residual)
        residPred, _, _ = model.predict(60)

        # 장주기 외삽
        t = np.arange(n, n + 60)
        longCyclePred = np.mean(trainValues) + np.mean(longCycle[-90:]) * np.sin(2 * np.pi * t / 90)

        pred = residPred + longCyclePred
    else:
        # 장주기 없으면 기본 예측
        model = OptimizedTheta(period=7)
        model.fit(trainValues)
        pred, _, _ = model.predict(60)

    mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
    print(f"MAPE: {mape:.2f}%")

    return {'mape': mape, 'detectedPeriods': detectedPeriods}


# ============================================================================
# 실험 4: statsforecast 비교
# ============================================================================

def experiment4_statsforecast():
    """statsforecast 비교"""
    print("\n" + "=" * 60)
    print("실험 4: statsforecast 비교")
    print("=" * 60)

    df = generateManufacturing(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS

        sfDf = trainDf.copy()
        sfDf['unique_id'] = 'series1'
        sfDf = sfDf.rename(columns={'date': 'ds', 'value': 'y'})

        sf = StatsForecast(
            models=[AutoARIMA(season_length=7), AutoETS(season_length=7)],
            freq='D', n_jobs=1
        )
        sf.fit(sfDf)
        sfPred = sf.predict(h=60)

        # 최적 모델 선택
        bestMape = float('inf')
        bestModel = ''
        for col in sfPred.columns:
            if col not in ['ds', 'unique_id']:
                pred = sfPred[col].values
                mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
                if mape < bestMape:
                    bestMape = mape
                    bestModel = col

        print(f"statsforecast MAPE: {bestMape:.2f}% (모델: {bestModel})")
        return {'mape': bestMape, 'model': bestModel}

    except Exception as e:
        print(f"statsforecast 오류: {e}")
        return {'mape': float('inf')}


# ============================================================================
# 메인 실행
# ============================================================================

def runAllExperiments():
    """모든 실험 실행"""
    print("=" * 60)
    print("E009: 정기 패턴 감지 (Manufacturing 개선)")
    print("=" * 60)

    results = {}

    results['current'] = experiment1_current()
    results['dropAware'] = experiment2_dropAware()
    results['fourier'] = experiment3_fourier()
    results['statsforecast'] = experiment4_statsforecast()

    # 종합
    print("\n" + "=" * 60)
    print("E009 실험 종합")
    print("=" * 60)

    print("\n방법별 MAPE:")
    for name, res in results.items():
        mape = res.get('mape', float('inf'))
        print(f"  {name:20}: {mape:.2f}%")

    return results


if __name__ == '__main__':
    results = runAllExperiments()
