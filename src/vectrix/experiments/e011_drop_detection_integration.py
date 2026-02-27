"""
==============================================================================
실험 ID: E011
실험명: 드롭 감지 통합 검증
==============================================================================

목적:
- E009 결과를 ChaniCast에 통합 후 검증
- PeriodicDropDetector 자동 감지 기능 확인
- 다양한 드롭 패턴 데이터에서 효과 검증

방법:
1. PeriodicDropDetector를 engine 모듈로 이동
2. chanicast_native.py에 자동 드롭 감지/처리 로직 추가
3. 다양한 드롭 주기로 검증 (90일, 30일, 7일)

==============================================================================
결과
==============================================================================

결과 요약:
- 드롭 감지 기능이 정상 동작 (90일, 30일 주기 감지 성공)
- 학습 데이터에서 드롭 제거 후 모델 학습 → 성능 개선
- 예측 구간에 실제 드롭이 있을 때만 드롭 패턴 적용이 효과적

수치 (최종):
| 시나리오 | MAPE | 개선 |
|----------|------|------|
| 예측 구간에 드롭 있음 | **2.38%** | flatCorrector 충돌 해결 |
| 예측 구간에 드롭 없음 | **2.54%** | Smart 드롭 미적용 |
| 드롭 없는 일반 데이터 | 1.87% | 오감지 없음 |

핵심 개선 (한계 극복):
1. **willDropOccurInPrediction()**: 예측 구간에 드롭 발생 여부 자동 계산
2. **applyDropPatternSmart()**: 발생 시에만 드롭 패턴 적용
3. **flatCorrector 충돌 해결**: 드롭 적용 후 일직선 보정 건너뜀

결론:
1. **한계 극복**: 예측 구간 드롭 발생 여부 자동 판단 구현
2. **높은 정확도**: 드롭 있을 때 2.38%, 없을 때 2.54%
3. **오감지 없음**: 드롭 없는 데이터에서 드롭 감지 안됨
4. **통합 완료**: ChaniCast에 완전 통합

실험일: 2026-02-05
==============================================================================
"""

import numpy as np
import pandas as pd
import sys
import io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def generateManufacturing90Day(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """제조 생산량 (90일 주기 정기 점검) - 테스트 구간에도 드롭 포함"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1100, n)
    noise = np.random.normal(0, 30, n)
    values = trend + noise

    for i in range(0, n, 90):
        dropEnd = min(i + 7, n)
        if dropEnd > i:
            values[i:dropEnd] *= 0.7

    return pd.DataFrame({'date': dates, 'value': values})


def generateManufacturing90DayTrainOnly(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """제조 생산량 - 학습 구간에만 드롭 (E009와 동일 조건)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(1000, 1100, n)
    noise = np.random.normal(0, 30, n)
    values = trend + noise

    trainEnd = n - 60
    for i in range(0, trainEnd, 90):
        if i + 7 < trainEnd:
            values[i:i+7] *= 0.7

    return pd.DataFrame({'date': dates, 'value': values})


def generateManufacturing30Day(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """제조 생산량 (30일 주기 짧은 점검)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(500, 550, n)
    noise = np.random.normal(0, 15, n)
    values = trend + noise

    for i in range(0, n, 30):
        if i + 3 < n:
            values[i:i+3] *= 0.75

    return pd.DataFrame({'date': dates, 'value': values})


def generateServerMaintenance(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """서버 트래픽 (주간 유지보수 윈도우)"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    base = 10000
    weekly = 500 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 200, n)
    values = base + weekly + noise

    for i in range(0, n, 7):
        values[i] *= 0.6

    return pd.DataFrame({'date': dates, 'value': values})


def generateNoDropData(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """드롭 없는 일반 데이터"""
    np.random.seed(seed)
    dates = pd.date_range('2022-01-01', periods=n, freq='D')

    trend = np.linspace(100, 120, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    noise = np.random.normal(0, 3, n)
    values = trend + seasonal + noise

    return pd.DataFrame({'date': dates, 'value': values})


def experiment1a_drop_in_prediction():
    """예측 구간에 드롭이 있는 경우"""
    print("\n" + "=" * 60)
    print("실험 1a: 예측 구간에 드롭 있음")
    print("=" * 60)

    from vectrix import ChaniCast

    df = generateManufacturing90Day(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    cc = ChaniCast(verbose=True)
    result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

    if result.success:
        pred = result.predictions[:60]
        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"\nMAPE: {mape:.2f}% (모델: {result.bestModelName})")
        print(f"경고: {result.warnings}")
        return {'mape': mape, 'model': result.bestModelName, 'warnings': result.warnings}

    print(f"오류: {result.error}")
    return {'mape': float('inf')}


def experiment1b_no_drop_in_prediction():
    """예측 구간에 드롭이 없는 경우"""
    print("\n" + "=" * 60)
    print("실험 1b: 예측 구간에 드롭 없음 (Smart 미적용)")
    print("=" * 60)

    from vectrix import ChaniCast

    df = generateManufacturing90DayTrainOnly(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    cc = ChaniCast(verbose=True)
    result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

    if result.success:
        pred = result.predictions[:60]
        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"\nMAPE: {mape:.2f}% (모델: {result.bestModelName})")
        return {'mape': mape, 'model': result.bestModelName}

    return {'mape': float('inf')}


def experiment2_30day():
    """30일 주기 짧은 드롭"""
    print("\n" + "=" * 60)
    print("실험 2: 30일 주기 드롭")
    print("=" * 60)

    from vectrix import ChaniCast

    df = generateManufacturing30Day(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    cc = ChaniCast(verbose=True)
    result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

    if result.success:
        pred = result.predictions[:60]
        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"\nMAPE: {mape:.2f}% (모델: {result.bestModelName})")
        return {'mape': mape, 'model': result.bestModelName}

    return {'mape': float('inf')}


def experiment3_weekly():
    """주간 유지보수 윈도우"""
    print("\n" + "=" * 60)
    print("실험 3: 주간 유지보수 윈도우")
    print("=" * 60)

    from vectrix import ChaniCast

    df = generateServerMaintenance(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    cc = ChaniCast(verbose=True)
    result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

    if result.success:
        pred = result.predictions[:60]
        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"\nMAPE: {mape:.2f}% (모델: {result.bestModelName})")
        return {'mape': mape, 'model': result.bestModelName}

    return {'mape': float('inf')}


def experiment4_nodrop():
    """드롭 없는 데이터 (부작용 확인)"""
    print("\n" + "=" * 60)
    print("실험 4: 드롭 없는 일반 데이터")
    print("=" * 60)

    from vectrix import ChaniCast

    df = generateNoDropData(365)
    trainDf = df.iloc[:-60]
    testDf = df.iloc[-60:]
    testValues = testDf['value'].values

    cc = ChaniCast(verbose=True)
    result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=60)

    if result.success:
        pred = result.predictions[:60]
        mape = np.mean(np.abs((testValues - pred) / testValues)) * 100
        print(f"\nMAPE: {mape:.2f}% (모델: {result.bestModelName})")
        dropDetected = any('드롭' in w for w in result.warnings) if result.warnings else False
        print(f"드롭 오감지: {dropDetected}")
        return {'mape': mape, 'model': result.bestModelName, 'falsePositive': dropDetected}

    return {'mape': float('inf')}


def experiment5_statsforecast_comparison():
    """statsforecast 비교"""
    print("\n" + "=" * 60)
    print("실험 5: statsforecast 비교 (90일 드롭)")
    print("=" * 60)

    df = generateManufacturing90Day(365)
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


def runAllExperiments():
    """모든 실험 실행"""
    print("=" * 60)
    print("E011: 드롭 감지 통합 검증")
    print("=" * 60)

    results = {}

    results['drop_in_prediction'] = experiment1a_drop_in_prediction()
    results['no_drop_in_prediction'] = experiment1b_no_drop_in_prediction()
    results['30day_drop'] = experiment2_30day()
    results['weekly_maintenance'] = experiment3_weekly()
    results['no_drop'] = experiment4_nodrop()
    results['statsforecast'] = experiment5_statsforecast_comparison()

    print("\n" + "=" * 60)
    print("E011 실험 종합")
    print("=" * 60)

    print("\n시나리오별 MAPE:")
    for name, res in results.items():
        mape = res.get('mape', float('inf'))
        model = res.get('model', 'N/A')
        print(f"  {name:30}: {mape:.2f}% ({model})")

    return results


if __name__ == '__main__':
    results = runAllExperiments()
