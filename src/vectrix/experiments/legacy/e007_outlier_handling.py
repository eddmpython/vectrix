"""
==============================================================================
실험 ID: E007
실험명: 이상치 감지 및 처리
==============================================================================

목적:
- manufacturing 같은 급변 데이터 대응
- 이상치로 인한 예측 오차 감소
- 점검 드롭, 스파이크 등 비정상 패턴 처리

가설:
1. 이상치를 미리 감지하고 보간하면 모델 학습 품질 향상
2. 예측 후 이상치 패턴을 다시 반영하면 현실성 증가
3. Z-score/IQR 기반 감지가 효과적

방법:
1. 현재 방식: 이상치 처리 없음
2. Z-score 기반: |z| > 2.5 인 점 감지 → 선형 보간
3. IQR 기반: Q1-1.5*IQR ~ Q3+1.5*IQR 벗어나면 감지
4. Rolling window 기반: 이동 평균/표준편차로 감지

==============================================================================
결과
==============================================================================

결과 요약:
- 이상치 처리가 항상 효과적인 것은 아님
- 센서 데이터(극단값)에서만 효과적, 주기적 드롭에서는 역효과
- Rolling 감지가 가장 안정적이나 개선폭 제한적

수치:
| 데이터 | none | zscore | iqr | rolling |
|--------|------|--------|-----|---------|
| manufacturing | **3.32%** | 과적합 | 과적합 | 3.55% |
| sales_spikes | **6.32%** | 9.98% | 10.10% | 10.03% |
| sensor_glitches | 9.64% | 8.65% | 8.65% | **8.60%** |
| **평균** | **6.43%** | - | - | 7.39% |

결론:
1. **정기적 드롭(manufacturing)**: 이상치 처리 불필요, 패턴 자체가 정보
2. **랜덤 스파이크(sales)**: 이상치 처리가 오히려 정보 손실
3. **센서 오류(glitches)**: Rolling 감지가 가장 효과적 (10.8% 개선)
4. **권장**: 데이터 유형 판별 후 조건부 적용
   - 센서 데이터/극단값 → Rolling 감지 + 중앙값 대체
   - 비즈니스 이벤트(점검, 프로모션) → 처리하지 않음

실험일: 2026-02-05
==============================================================================
"""

import io
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================================
# 이상치 있는 데이터 생성
# ============================================================================

def generateManufacturingWithDrops(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """제조 생산량 (정기 점검 드롭) - 학습 구간에만"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1100, n)
    noise = np.random.normal(0, 30, n)
    values = trend + noise

    # 90일마다 1주일 점검 (30% 감소) - 학습 구간에만
    trainEnd = n - 60
    dropIndices = []
    for i in range(0, trainEnd, 90):
        if i + 7 < trainEnd:
            values[i:i+7] *= 0.7
            dropIndices.extend(range(i, min(i+7, trainEnd)))

    return pd.DataFrame({'date': dates, 'value': values}), dropIndices


def generateSalesWithSpikes(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """판매 데이터 (프로모션 스파이크) - 학습 구간에만"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(500, 600, n)
    weekly = 50 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 20, n)
    values = trend + weekly + noise

    # 랜덤 프로모션 스파이크 - 학습 구간(0~305)에만
    trainEnd = n - 60
    spikeIndices = np.random.choice(trainEnd, size=10, replace=False)
    values[spikeIndices] *= 1.5

    return pd.DataFrame({'date': dates, 'value': values}), spikeIndices.tolist()


def generateSensorWithGlitches(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """센서 데이터 (간헐적 오류) - 학습 구간에만 이상치"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    base = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 30)
    noise = np.random.normal(0, 3, n)
    values = base + noise

    # 센서 오류 (극단값) - 학습 구간(0~305)에만
    trainEnd = n - 60
    glitchIndices = np.random.choice(trainEnd, size=5, replace=False)
    values[glitchIndices] = np.random.choice([50, 200], size=5)  # 덜 극단적으로

    return pd.DataFrame({'date': dates, 'value': values}), glitchIndices.tolist()


# ============================================================================
# 이상치 감지 방법들
# ============================================================================

class OutlierDetector:
    """이상치 감지 클래스"""

    @staticmethod
    def zscoreDetect(y: np.ndarray, threshold: float = 2.5) -> np.ndarray:
        """Z-score 기반 이상치 감지"""
        mean = np.mean(y)
        std = np.std(y)
        if std < 1e-10:
            return np.zeros(len(y), dtype=bool)

        zscores = np.abs((y - mean) / std)
        return zscores > threshold

    @staticmethod
    def iqrDetect(y: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
        """IQR 기반 이상치 감지"""
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        return (y < lower) | (y > upper)

    @staticmethod
    def rollingDetect(y: np.ndarray, window: int = 14, threshold: float = 2.5) -> np.ndarray:
        """Rolling window 기반 이상치 감지"""
        n = len(y)
        outliers = np.zeros(n, dtype=bool)
        halfWin = window // 2

        for i in range(n):
            start = max(0, i - halfWin)
            end = min(n, i + halfWin + 1)

            if end - start < 3:
                continue

            localData = np.concatenate([y[start:i], y[i+1:end]]) if i > start and i < end-1 else y[start:end]
            if len(localData) < 2:
                continue

            localMean = np.mean(localData)
            localStd = np.std(localData)

            if localStd > 1e-10:
                zscore = abs(y[i] - localMean) / localStd
                outliers[i] = zscore > threshold

        return outliers


class OutlierHandler:
    """이상치 처리 클래스"""

    @staticmethod
    def interpolate(y: np.ndarray, outlierMask: np.ndarray) -> np.ndarray:
        """선형 보간으로 이상치 대체"""
        result = y.copy()
        outlierIndices = np.where(outlierMask)[0]

        for idx in outlierIndices:
            # 앞뒤 정상값 찾기
            prevIdx = idx - 1
            while prevIdx >= 0 and outlierMask[prevIdx]:
                prevIdx -= 1

            nextIdx = idx + 1
            while nextIdx < len(y) and outlierMask[nextIdx]:
                nextIdx += 1

            # 보간
            if prevIdx >= 0 and nextIdx < len(y):
                # 선형 보간
                ratio = (idx - prevIdx) / (nextIdx - prevIdx)
                result[idx] = y[prevIdx] + ratio * (y[nextIdx] - y[prevIdx])
            elif prevIdx >= 0:
                result[idx] = y[prevIdx]
            elif nextIdx < len(y):
                result[idx] = y[nextIdx]
            # 둘 다 없으면 원래 값 유지

        return result

    @staticmethod
    def medianReplace(y: np.ndarray, outlierMask: np.ndarray, window: int = 7) -> np.ndarray:
        """지역 중앙값으로 대체"""
        result = y.copy()
        outlierIndices = np.where(outlierMask)[0]
        n = len(y)

        for idx in outlierIndices:
            start = max(0, idx - window)
            end = min(n, idx + window + 1)

            # 이상치 제외한 지역 데이터
            localMask = ~outlierMask[start:end]
            localData = y[start:end][localMask]

            if len(localData) > 0:
                result[idx] = np.median(localData)

        return result


# ============================================================================
# 실험 1: 현재 방식 (이상치 처리 없음)
# ============================================================================

def experiment1_noOutlierHandling():
    """현재 방식 - 이상치 처리 없음"""
    print("\n" + "=" * 70)
    print("실험 1: 현재 방식 (이상치 처리 없음)")
    print("=" * 70)

    from vectrix import ChaniCast

    datasets = [
        ('manufacturing', *generateManufacturingWithDrops()),
        ('sales_spikes', *generateSalesWithSpikes()),
        ('sensor_glitches', *generateSensorWithGlitches()),
    ]

    results = {}

    for name, df, outlierIndices in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        testValues = testDf['value'].values

        try:
            cc = ChaniCast(verbose=False)
            result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

            if result.success:
                pred = result.predictions[:60]
                mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
                results[name] = {
                    'mape': mape,
                    'model': result.bestModelName,
                    'outlierCount': len([i for i in outlierIndices if i < len(trainDf)])
                }
                print(f"{name:20} MAPE = {mape:.2f}% (이상치 {results[name]['outlierCount']}개)")
            else:
                results[name] = {'mape': float('inf'), 'error': result.error}
                print(f"{name:20} 오류")
        except Exception as e:
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 2: Z-score + 선형 보간
# ============================================================================

def experiment2_zscoreInterpolate():
    """Z-score 감지 + 선형 보간"""
    print("\n" + "=" * 70)
    print("실험 2: Z-score + 선형 보간")
    print("=" * 70)

    from vectrix import ChaniCast

    datasets = [
        ('manufacturing', *generateManufacturingWithDrops()),
        ('sales_spikes', *generateSalesWithSpikes()),
        ('sensor_glitches', *generateSensorWithGlitches()),
    ]

    results = {}

    for name, df, _ in datasets:
        trainDf = df.iloc[:-60].copy()
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values.copy()
        testValues = testDf['value'].values

        try:
            # 이상치 감지 및 처리
            outlierMask = OutlierDetector.zscoreDetect(trainValues, threshold=2.5)
            cleanedValues = OutlierHandler.interpolate(trainValues, outlierMask)
            detectedCount = np.sum(outlierMask)

            # 처리된 데이터로 학습
            trainDf['value'] = cleanedValues

            cc = ChaniCast(verbose=False)
            result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

            if result.success:
                pred = result.predictions[:60]
                mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
                results[name] = {'mape': mape, 'detected': detectedCount}
                print(f"{name:20} MAPE = {mape:.2f}% (감지된 이상치: {detectedCount}개)")
            else:
                results[name] = {'mape': float('inf')}
                print(f"{name:20} 오류")
        except Exception as e:
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 3: IQR + 선형 보간
# ============================================================================

def experiment3_iqrInterpolate():
    """IQR 감지 + 선형 보간"""
    print("\n" + "=" * 70)
    print("실험 3: IQR + 선형 보간")
    print("=" * 70)

    from vectrix import ChaniCast

    datasets = [
        ('manufacturing', *generateManufacturingWithDrops()),
        ('sales_spikes', *generateSalesWithSpikes()),
        ('sensor_glitches', *generateSensorWithGlitches()),
    ]

    results = {}

    for name, df, _ in datasets:
        trainDf = df.iloc[:-60].copy()
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values.copy()
        testValues = testDf['value'].values

        try:
            # 이상치 감지 및 처리
            outlierMask = OutlierDetector.iqrDetect(trainValues, multiplier=1.5)
            cleanedValues = OutlierHandler.interpolate(trainValues, outlierMask)
            detectedCount = np.sum(outlierMask)

            trainDf['value'] = cleanedValues

            cc = ChaniCast(verbose=False)
            result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

            if result.success:
                pred = result.predictions[:60]
                mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
                results[name] = {'mape': mape, 'detected': detectedCount}
                print(f"{name:20} MAPE = {mape:.2f}% (감지된 이상치: {detectedCount}개)")
            else:
                results[name] = {'mape': float('inf')}
                print(f"{name:20} 오류")
        except Exception as e:
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 4: Rolling 감지 + 중앙값 대체
# ============================================================================

def experiment4_rollingMedian():
    """Rolling window 감지 + 중앙값 대체"""
    print("\n" + "=" * 70)
    print("실험 4: Rolling 감지 + 중앙값 대체")
    print("=" * 70)

    from vectrix import ChaniCast

    datasets = [
        ('manufacturing', *generateManufacturingWithDrops()),
        ('sales_spikes', *generateSalesWithSpikes()),
        ('sensor_glitches', *generateSensorWithGlitches()),
    ]

    results = {}

    for name, df, _ in datasets:
        trainDf = df.iloc[:-60].copy()
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values.copy()
        testValues = testDf['value'].values

        try:
            # 이상치 감지 및 처리
            outlierMask = OutlierDetector.rollingDetect(trainValues, window=14, threshold=2.5)
            cleanedValues = OutlierHandler.medianReplace(trainValues, outlierMask, window=7)
            detectedCount = np.sum(outlierMask)

            trainDf['value'] = cleanedValues

            cc = ChaniCast(verbose=False)
            result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

            if result.success:
                pred = result.predictions[:60]
                mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
                results[name] = {'mape': mape, 'detected': detectedCount}
                print(f"{name:20} MAPE = {mape:.2f}% (감지된 이상치: {detectedCount}개)")
            else:
                results[name] = {'mape': float('inf')}
                print(f"{name:20} 오류")
        except Exception as e:
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 5: 방법 비교 요약
# ============================================================================

def experiment5_comparison():
    """모든 방법 비교"""
    print("\n" + "=" * 70)
    print("실험 5: 방법별 비교 요약")
    print("=" * 70)

    from vectrix import ChaniCast

    datasets = [
        ('manufacturing', *generateManufacturingWithDrops()),
        ('sales_spikes', *generateSalesWithSpikes()),
        ('sensor_glitches', *generateSensorWithGlitches()),
    ]

    methods = {
        'none': lambda y: y,
        'zscore': lambda y: OutlierHandler.interpolate(y, OutlierDetector.zscoreDetect(y, 2.5)),
        'iqr': lambda y: OutlierHandler.interpolate(y, OutlierDetector.iqrDetect(y, 1.5)),
        'rolling': lambda y: OutlierHandler.medianReplace(y, OutlierDetector.rollingDetect(y, 14, 2.5), 7),
    }

    allResults = {method: {} for method in methods}

    for name, df, _ in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values.copy()
        testValues = testDf['value'].values

        for methodName, cleanFunc in methods.items():
            try:
                cleanedDf = trainDf.copy()
                cleanedDf['value'] = cleanFunc(trainValues.copy())

                cc = ChaniCast(verbose=False)
                result = cc.forecast(cleanedDf, dateCol='date', valueCol='value', steps=60)

                if result.success:
                    pred = result.predictions[:60]
                    mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
                    allResults[methodName][name] = mape
                else:
                    allResults[methodName][name] = float('inf')
            except Exception:
                allResults[methodName][name] = float('inf')

    # 결과 출력
    print("\n" + "-" * 70)
    print(f"{'데이터':<20}", end="")
    for method in methods:
        print(f"{method:<12}", end="")
    print()
    print("-" * 70)

    for name in [d[0] for d in datasets]:
        print(f"{name:<20}", end="")
        for method in methods:
            mape = allResults[method].get(name, float('inf'))
            print(f"{mape:<12.2f}", end="")
        print()

    # 평균
    print("-" * 70)
    print(f"{'평균':<20}", end="")
    for method in methods:
        mapes = [v for v in allResults[method].values() if v < float('inf')]
        avg = np.mean(mapes) if mapes else float('inf')
        print(f"{avg:<12.2f}", end="")
    print()

    return allResults


# ============================================================================
# 메인 실행
# ============================================================================

def runAllExperiments():
    """모든 실험 실행"""
    print("=" * 70)
    print("E007: 이상치 감지 및 처리 실험")
    print("=" * 70)

    allResults = {}

    # 실험 1: 현재 방식
    allResults['none'] = experiment1_noOutlierHandling()

    # 실험 2: Z-score
    allResults['zscore'] = experiment2_zscoreInterpolate()

    # 실험 3: IQR
    allResults['iqr'] = experiment3_iqrInterpolate()

    # 실험 4: Rolling
    allResults['rolling'] = experiment4_rollingMedian()

    # 실험 5: 비교
    allResults['comparison'] = experiment5_comparison()

    # 종합
    print("\n" + "=" * 70)
    print("E007 실험 종합 결론")
    print("=" * 70)

    return allResults


if __name__ == '__main__':
    results = runAllExperiments()
