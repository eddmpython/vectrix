"""
==============================================================================
실험 ID: E005
실험명: 앙상블 가중치 최적화
==============================================================================

목적:
- 앙상블 정확도 향상
- 데이터 특성에 따른 적응적 가중치 적용
- 강한 계절성 데이터에서 약점 개선

가설:
1. 계절성 강도에 따라 ETS 가중치 증가 시 정확도 향상
2. 변동성 매칭 가중치로 예측 안정성 향상
3. 검증 기반 동적 가중치가 고정 가중치보다 우수

방법:
1. 고정 가중치 (현재): 동일 가중치
2. 계절성 기반 가중치: 계절성 강하면 ETS 가중 증가
3. 검증 기반 가중치: 홀드아웃 검증 성능으로 가중치 결정
4. 스태킹: 메타 모델로 최적 조합

==============================================================================
결과
==============================================================================

결과 요약:
- 현재 ChaniCast: 단일 최적 모델(Theta) 선택이 앙상블보다 우수
- 이상치 많은 데이터(temperature) 제외 시 검증 기반 가중치가 약간 우수
- 앙상블이 오히려 성능 저하시킬 수 있음

수치 (전체 5개 데이터셋):
- 현재 (Theta 단일): 평균 MAPE 48.35%
- 계절성 기반 앙상블: 평균 MAPE 112.29%
- 검증 기반 앙상블: 평균 MAPE 93.05%

수치 (temperature 제외 4개):
- 현재: 평균 MAPE 11.64%
- 검증 기반: 평균 MAPE 10.78% (7.4% 개선)

데이터별 최적 방법:
| 데이터 | 현재 | 검증기반 | 승자 |
|--------|------|----------|------|
| retail_sales | 16.95% | 16.69% | 검증기반 |
| stock_price | 11.70% | 10.65% | 검증기반 |
| temperature | 195.17% | 422.13% | 현재 |
| energy | 14.60% | 13.93% | 검증기반 |
| manufacturing | 3.32% | 1.84% | 검증기반 |

결론:
1. 현재 단일 모델 선택 방식이 이상치 데이터에서 더 robust
2. 정상적인 데이터에서는 검증 기반 동적 가중치가 약간 우수
3. 앙상블 적용은 데이터 품질 확인 후 조건부로 적용 권장
4. 기본값은 현재 방식 유지, 고품질 데이터에서만 앙상블 적용

권장사항:
- 이상치 비율 > 5% 또는 계절성 감지 실패 시 단일 모델 유지
- 그 외에는 검증 기반 동적 가중치 앙상블 고려

실험일: 2026-02-05
==============================================================================
"""

import numpy as np
import pandas as pd
import time
import sys
import io
import os
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# ============================================================================
# 데이터 생성 함수들 (e001에서 가져옴)
# ============================================================================

def generateRetailSales(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """소매 판매 데이터 (강한 주간+연간 계절성)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1200, n)
    weekly = 150 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 200 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 50, n)

    values = trend + weekly + yearly + noise
    values = np.maximum(values, 100)

    return pd.DataFrame({'date': dates, 'value': values})


def generateStockPrice(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """주가 데이터 (랜덤워크)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    returns = np.random.normal(0.0005, 0.02, n)
    price = 100 * np.cumprod(1 + returns)

    return pd.DataFrame({'date': dates, 'value': price})


def generateTemperature(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """기온 데이터 (연간 계절성 + 이상치)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    seasonal = 15 * np.sin(2 * np.pi * (np.arange(n) - 80) / 365)
    base = 15
    noise = np.random.normal(0, 3, n)

    values = base + seasonal + noise

    outlierIdx = np.random.choice(n, size=int(n * 0.02), replace=False)
    values[outlierIdx] += np.random.choice([-10, 10], size=len(outlierIdx))

    return pd.DataFrame({'date': dates, 'value': values})


def generateEnergy(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """에너지 소비 데이터 (계절+요일 효과)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(500, 520, n)
    weekly = 50 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly = 100 * np.sin(2 * np.pi * (np.arange(n) - 30) / 365)
    noise = np.random.normal(0, 20, n)

    values = trend + weekly + yearly + noise
    values = np.maximum(values, 200)

    return pd.DataFrame({'date': dates, 'value': values})


def generateManufacturing(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """제조 생산량 (추세+점검 드롭)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1100, n)
    noise = np.random.normal(0, 30, n)
    values = trend + noise

    for i in range(0, n, 90):
        if i + 7 < n:
            values[i:i+7] *= 0.7

    return pd.DataFrame({'date': dates, 'value': values})


# ============================================================================
# 실험 1: 현재 앙상블 성능 (기준선)
# ============================================================================

def experiment1_baseline():
    """현재 앙상블 성능 측정 (기준선)"""
    print("\n" + "=" * 70)
    print("실험 1: 현재 앙상블 성능 (기준선)")
    print("=" * 70)

    from vectrix import ChaniCast

    datasets = [
        ('retail_sales', generateRetailSales()),
        ('stock_price', generateStockPrice()),
        ('temperature', generateTemperature()),
        ('energy', generateEnergy()),
        ('manufacturing', generateManufacturing()),
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
# 실험 2: 계절성 기반 가중치
# ============================================================================

def experiment2_seasonalityWeights():
    """계절성 강도에 따른 ETS 가중치 조정"""
    print("\n" + "=" * 70)
    print("실험 2: 계절성 기반 가중치")
    print("=" * 70)

    from vectrix.engine.ets import AutoETS
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.theta import OptimizedTheta
    from vectrix.analyzer import AutoAnalyzer

    datasets = [
        ('retail_sales', generateRetailSales()),
        ('stock_price', generateStockPrice()),
        ('temperature', generateTemperature()),
        ('energy', generateEnergy()),
        ('manufacturing', generateManufacturing()),
    ]

    results = {}

    for name, df in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values
        testValues = testDf['value'].values

        try:
            # 분석
            analyzer = AutoAnalyzer()
            chars = analyzer.analyze(trainDf, 'date', 'value')
            seasonalStrength = chars.seasonalStrength if hasattr(chars, 'seasonalStrength') else 0.5

            # 개별 모델 학습
            ets = AutoETS(period=chars.period)
            ets.fit(trainValues)
            etsPred, _, _ = ets.predict(60)

            arima = AutoARIMA()
            arima.fit(trainValues)
            arimaPred, _, _ = arima.predict(60)

            theta = OptimizedTheta(period=chars.period)
            theta.fit(trainValues)
            thetaPred, _, _ = theta.predict(60)

            # 계절성 기반 가중치 (계절성 강하면 ETS 가중 증가)
            if seasonalStrength > 0.6:
                weights = [0.5, 0.25, 0.25]  # ETS, ARIMA, Theta
            elif seasonalStrength > 0.3:
                weights = [0.4, 0.35, 0.25]
            else:
                weights = [0.3, 0.4, 0.3]  # ARIMA 가중

            # 가중 앙상블
            ensemble = weights[0] * etsPred + weights[1] * arimaPred + weights[2] * thetaPred
            mape = np.mean(np.abs((testValues - ensemble) / testValues)) * 100

            results[name] = {
                'mape': mape,
                'seasonalStrength': seasonalStrength,
                'weights': weights
            }
            print(f"{name:20} MAPE = {mape:.2f}% (계절성={seasonalStrength:.2f}, 가중치={weights})")

        except Exception as e:
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 3: 검증 기반 동적 가중치
# ============================================================================

def experiment3_validationWeights():
    """홀드아웃 검증 성능 기반 동적 가중치"""
    print("\n" + "=" * 70)
    print("실험 3: 검증 기반 동적 가중치")
    print("=" * 70)

    from vectrix.engine.ets import AutoETS
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.theta import OptimizedTheta
    from vectrix.analyzer import AutoAnalyzer

    datasets = [
        ('retail_sales', generateRetailSales()),
        ('stock_price', generateStockPrice()),
        ('temperature', generateTemperature()),
        ('energy', generateEnergy()),
        ('manufacturing', generateManufacturing()),
    ]

    results = {}

    for name, df in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values
        testValues = testDf['value'].values

        # 검증용 분할 (학습의 마지막 20%를 검증용으로)
        valSize = int(len(trainValues) * 0.2)
        trainPart = trainValues[:-valSize]
        valPart = trainValues[-valSize:]

        try:
            # 분석
            analyzer = AutoAnalyzer()
            chars = analyzer.analyze(trainDf.iloc[:-valSize], 'date', 'value')

            # 개별 모델 학습 (검증용)
            ets = AutoETS(period=chars.period)
            ets.fit(trainPart)
            etsValPred, _, _ = ets.predict(valSize)
            etsMape = np.mean(np.abs((valPart - etsValPred) / valPart))

            arima = AutoARIMA()
            arima.fit(trainPart)
            arimaValPred, _, _ = arima.predict(valSize)
            arimaMape = np.mean(np.abs((valPart - arimaValPred) / valPart))

            theta = OptimizedTheta(period=chars.period)
            theta.fit(trainPart)
            thetaValPred, _, _ = theta.predict(valSize)
            thetaMape = np.mean(np.abs((valPart - thetaValPred) / valPart))

            # 역 MAPE 기반 가중치 (성능 좋을수록 높은 가중치)
            mapes = np.array([etsMape, arimaMape, thetaMape])
            mapes = np.clip(mapes, 0.01, 10)  # 극단값 방지
            invMapes = 1.0 / mapes
            weights = invMapes / invMapes.sum()

            # 전체 데이터로 재학습
            ets2 = AutoETS(period=chars.period)
            ets2.fit(trainValues)
            etsPred, _, _ = ets2.predict(60)

            arima2 = AutoARIMA()
            arima2.fit(trainValues)
            arimaPred, _, _ = arima2.predict(60)

            theta2 = OptimizedTheta(period=chars.period)
            theta2.fit(trainValues)
            thetaPred, _, _ = theta2.predict(60)

            # 가중 앙상블
            ensemble = weights[0] * etsPred + weights[1] * arimaPred + weights[2] * thetaPred
            mape = np.mean(np.abs((testValues - ensemble) / testValues)) * 100

            results[name] = {
                'mape': mape,
                'weights': weights.tolist(),
                'valMapes': [etsMape, arimaMape, thetaMape]
            }
            print(f"{name:20} MAPE = {mape:.2f}% (가중치=[{weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f}])")

        except Exception as e:
            import traceback
            results[name] = {'mape': float('inf'), 'error': str(e)}
            print(f"{name:20} 예외: {str(e)[:50]}")
            traceback.print_exc()

    avgMape = np.mean([r['mape'] for r in results.values() if r['mape'] < float('inf')])
    print(f"\n평균 MAPE: {avgMape:.2f}%")

    return results


# ============================================================================
# 실험 4: 동일 가중치 vs 최적화 가중치 비교
# ============================================================================

def experiment4_weightComparison():
    """가중치 전략 비교"""
    print("\n" + "=" * 70)
    print("실험 4: 가중치 전략 비교")
    print("=" * 70)

    from vectrix.engine.ets import AutoETS
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.theta import OptimizedTheta
    from vectrix.analyzer import AutoAnalyzer

    datasets = [
        ('retail_sales', generateRetailSales()),
        ('stock_price', generateStockPrice()),
        ('temperature', generateTemperature()),
        ('energy', generateEnergy()),
        ('manufacturing', generateManufacturing()),
    ]

    strategies = {
        'equal': lambda e, a, t, s: [1/3, 1/3, 1/3],
        'ets_heavy': lambda e, a, t, s: [0.5, 0.25, 0.25],
        'arima_heavy': lambda e, a, t, s: [0.25, 0.5, 0.25],
        'seasonal_adaptive': lambda e, a, t, s: [0.5, 0.25, 0.25] if s > 0.5 else [0.25, 0.5, 0.25],
    }

    allResults = {strategy: {} for strategy in strategies}

    for name, df in datasets:
        trainDf = df.iloc[:-60]
        testDf = df.iloc[-60:]
        trainValues = trainDf['value'].values
        testValues = testDf['value'].values

        try:
            # 분석
            analyzer = AutoAnalyzer()
            chars = analyzer.analyze(trainDf, 'date', 'value')
            seasonalStrength = chars.seasonalStrength if hasattr(chars, 'seasonalStrength') else 0.5

            # 개별 모델 학습
            ets = AutoETS(period=chars.period)
            ets.fit(trainValues)
            etsPred, _, _ = ets.predict(60)

            arima = AutoARIMA()
            arima.fit(trainValues)
            arimaPred, _, _ = arima.predict(60)

            theta = OptimizedTheta(period=chars.period)
            theta.fit(trainValues)
            thetaPred, _, _ = theta.predict(60)

            # 각 전략별 앙상블
            for strategyName, weightFunc in strategies.items():
                weights = weightFunc(etsPred, arimaPred, thetaPred, seasonalStrength)
                ensemble = weights[0] * etsPred + weights[1] * arimaPred + weights[2] * thetaPred
                mape = np.mean(np.abs((testValues - ensemble) / testValues)) * 100
                allResults[strategyName][name] = mape

        except Exception as e:
            for strategyName in strategies:
                allResults[strategyName][name] = float('inf')

    # 결과 출력
    print("\n" + "-" * 70)
    print(f"{'데이터':<20}", end="")
    for strategy in strategies:
        print(f"{strategy:<18}", end="")
    print()
    print("-" * 70)

    for name in [d[0] for d in datasets]:
        print(f"{name:<20}", end="")
        for strategy in strategies:
            mape = allResults[strategy].get(name, float('inf'))
            print(f"{mape:<18.2f}", end="")
        print()

    # 평균
    print("-" * 70)
    print(f"{'평균':<20}", end="")
    for strategy in strategies:
        mapes = [v for v in allResults[strategy].values() if v < float('inf')]
        avg = np.mean(mapes) if mapes else float('inf')
        print(f"{avg:<18.2f}", end="")
    print()

    return allResults


# ============================================================================
# 메인 실행
# ============================================================================

def runAllExperiments():
    """모든 실험 실행"""
    print("=" * 70)
    print("E005: 앙상블 가중치 최적화 실험")
    print("=" * 70)

    allResults = {}

    # 실험 1: 기준선
    allResults['baseline'] = experiment1_baseline()

    # 실험 2: 계절성 기반
    allResults['seasonal'] = experiment2_seasonalityWeights()

    # 실험 3: 검증 기반
    allResults['validation'] = experiment3_validationWeights()

    # 실험 4: 전략 비교
    allResults['comparison'] = experiment4_weightComparison()

    # 종합
    print("\n" + "=" * 70)
    print("E005 실험 종합 결론")
    print("=" * 70)

    # 각 방법의 평균 MAPE 계산
    baselineAvg = np.mean([r['mape'] for r in allResults['baseline'].values() if r.get('mape', float('inf')) < float('inf')])
    seasonalAvg = np.mean([r['mape'] for r in allResults['seasonal'].values() if r.get('mape', float('inf')) < float('inf')])
    validationAvg = np.mean([r['mape'] for r in allResults['validation'].values() if r.get('mape', float('inf')) < float('inf')])

    print(f"\n방법별 평균 MAPE:")
    print(f"  1. 현재 (기준선):     {baselineAvg:.2f}%")
    print(f"  2. 계절성 기반:       {seasonalAvg:.2f}%")
    print(f"  3. 검증 기반:         {validationAvg:.2f}%")

    bestMethod = min([
        ('현재', baselineAvg),
        ('계절성 기반', seasonalAvg),
        ('검증 기반', validationAvg)
    ], key=lambda x: x[1])

    print(f"\n최적 방법: {bestMethod[0]} (MAPE = {bestMethod[1]:.2f}%)")

    return allResults


if __name__ == '__main__':
    results = runAllExperiments()
