"""
==============================================================================
실험 ID: E001
실험명: 실데이터 벤치마크
==============================================================================

목적:
- M4 Competition 데이터 등 실제 시계열로 ChaniCast Native 성능 검증
- statsforecast와의 정확한 비교
- 데이터 특성별 강점/약점 파악

가설:
- ChaniCast Native는 자체 구현이지만 statsforecast의 70% 이상 성능 달성 가능
- 특정 데이터 유형(랜덤워크, 짧은 데이터)에서는 경쟁력 있을 것

방법:
1. M4 Competition Daily 데이터 샘플 사용
2. 다양한 실제 패턴의 합성 데이터 생성
3. 각 데이터에 대해 ChaniCast vs statsforecast 비교
4. MAPE, RMSE, 실행시간 측정

==============================================================================
결과 (실험 후 작성)
==============================================================================

결과 요약:
- [실험 후 작성]

수치:
- ChaniCast 평균 MAPE: [실험 후]
- statsforecast 평균 MAPE: [실험 후]
- 성능 비율: [실험 후]

결론:
- [실험 후 작성]

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
# 실데이터 생성기 (M4 Competition 스타일)
# ============================================================================

class RealDataGenerator:
    """다양한 실제 패턴의 시계열 데이터 생성"""

    @staticmethod
    def retailSales(n: int = 365, seed: int = 42) -> pd.DataFrame:
        """소매 판매 데이터 (강한 주간 + 연간 계절성)"""
        np.random.seed(seed)
        dates = pd.date_range('2022-01-01', periods=n, freq='D')

        # 기본 수준
        base = 1000

        # 추세 (완만한 성장)
        trend = np.linspace(0, 200, n)

        # 주간 계절성 (주말 높음)
        weekly = np.zeros(n)
        for i in range(n):
            dow = dates[i].dayofweek
            if dow == 5:  # 토요일
                weekly[i] = 150
            elif dow == 6:  # 일요일
                weekly[i] = 100
            elif dow == 4:  # 금요일
                weekly[i] = 50

        # 월간 변동 (월초 높음)
        monthly = 50 * np.sin(2 * np.pi * np.arange(n) / 30)

        # 연간 계절성 (연말 높음)
        yearly = np.zeros(n)
        for i in range(n):
            month = dates[i].month
            if month == 12:
                yearly[i] = 300
            elif month in [11, 1]:
                yearly[i] = 100
            elif month in [7, 8]:
                yearly[i] = -50

        # 노이즈
        noise = np.random.normal(0, 50, n)

        values = base + trend + weekly + monthly + yearly + noise
        values = np.maximum(values, 100)  # 최소값 보장

        return pd.DataFrame({'date': dates, 'value': values})

    @staticmethod
    def stockPrice(n: int = 252, seed: int = 42) -> pd.DataFrame:
        """주가 데이터 (랜덤워크 + 변동성 클러스터링)"""
        np.random.seed(seed)
        dates = pd.date_range('2022-01-01', periods=n, freq='B')  # 영업일

        # 기하 브라운 운동 (GBM)
        mu = 0.0005  # 일일 평균 수익률
        sigma = 0.02  # 일일 변동성

        returns = np.random.normal(mu, sigma, n)

        # 변동성 클러스터링 (GARCH-like)
        for i in range(1, n):
            if abs(returns[i-1]) > 0.03:  # 큰 변동 후 변동성 증가
                returns[i] *= 1.5

        # 가격
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame({'date': dates, 'value': prices})

    @staticmethod
    def temperature(n: int = 365, seed: int = 42) -> pd.DataFrame:
        """기온 데이터 (강한 연간 계절성 + 일간 노이즈)"""
        np.random.seed(seed)
        dates = pd.date_range('2022-01-01', periods=n, freq='D')

        # 연간 계절성
        yearly = 15 * np.cos(2 * np.pi * (np.arange(n) - 180) / 365)

        # 기본 기온 (서울 기준 약 12도)
        base = 12

        # 랜덤 노이즈
        noise = np.random.normal(0, 3, n)

        # 이상 기후 이벤트
        values = base + yearly + noise

        # 한파/폭염 (랜덤 발생)
        for i in np.random.choice(n, 10, replace=False):
            values[i] += np.random.choice([-10, 10])

        return pd.DataFrame({'date': dates, 'value': values})

    @staticmethod
    def websiteTraffic(n: int = 180, seed: int = 42) -> pd.DataFrame:
        """웹사이트 트래픽 (주간 패턴 + 점진적 성장 + 스파이크)"""
        np.random.seed(seed)
        dates = pd.date_range('2022-01-01', periods=n, freq='D')

        # 기본 레벨
        base = 5000

        # 성장 추세 (지수적)
        trend = 1000 * (np.exp(np.arange(n) / n) - 1)

        # 주간 패턴 (평일 높음)
        weekly = np.zeros(n)
        for i in range(n):
            dow = dates[i].dayofweek
            if dow < 5:  # 평일
                weekly[i] = 500
            else:
                weekly[i] = -300

        # 마케팅 스파이크 (랜덤)
        spikes = np.zeros(n)
        spikeIdx = np.random.choice(n, 5, replace=False)
        for idx in spikeIdx:
            spikes[idx] = np.random.uniform(2000, 5000)

        # 노이즈
        noise = np.random.normal(0, 200, n)

        values = base + trend + weekly + spikes + noise
        values = np.maximum(values, 1000)

        return pd.DataFrame({'date': dates, 'value': values})

    @staticmethod
    def energyConsumption(n: int = 365, seed: int = 42) -> pd.DataFrame:
        """에너지 소비 (강한 계절성 + 요일 효과)"""
        np.random.seed(seed)
        dates = pd.date_range('2022-01-01', periods=n, freq='D')

        # 기본 소비량
        base = 100

        # 계절 효과 (여름/겨울 높음)
        seasonal = np.zeros(n)
        for i in range(n):
            month = dates[i].month
            if month in [7, 8]:  # 여름 (에어컨)
                seasonal[i] = 40
            elif month in [12, 1, 2]:  # 겨울 (난방)
                seasonal[i] = 35
            elif month in [4, 5, 10]:  # 봄/가을
                seasonal[i] = -10

        # 요일 효과
        weekly = np.zeros(n)
        for i in range(n):
            dow = dates[i].dayofweek
            if dow < 5:  # 평일
                weekly[i] = 10
            else:
                weekly[i] = -5

        # 추세 (에너지 효율 개선으로 감소)
        trend = np.linspace(0, -10, n)

        # 노이즈
        noise = np.random.normal(0, 8, n)

        values = base + seasonal + weekly + trend + noise
        values = np.maximum(values, 50)

        return pd.DataFrame({'date': dates, 'value': values})

    @staticmethod
    def manufacturing(n: int = 200, seed: int = 42) -> pd.DataFrame:
        """제조 생산량 (점진적 증가 + 설비 점검 드롭)"""
        np.random.seed(seed)
        dates = pd.date_range('2022-01-01', periods=n, freq='D')

        # 기본 생산량
        base = 500

        # 추세 (생산성 향상)
        trend = np.linspace(0, 100, n)

        # 주간 패턴 (주말 감소)
        weekly = np.zeros(n)
        for i in range(n):
            dow = dates[i].dayofweek
            if dow == 5:
                weekly[i] = -100
            elif dow == 6:
                weekly[i] = -200

        # 설비 점검 (월 1회, 급격한 하락)
        maintenance = np.zeros(n)
        for i in range(n):
            if dates[i].day == 15:  # 매월 15일 점검
                maintenance[i] = -300

        # 노이즈
        noise = np.random.normal(0, 30, n)

        values = base + trend + weekly + maintenance + noise
        values = np.maximum(values, 100)

        return pd.DataFrame({'date': dates, 'value': values})


# ============================================================================
# 벤치마크 함수
# ============================================================================

def runSingleBenchmark(
    df: pd.DataFrame,
    dataName: str,
    trainRatio: float = 0.8
) -> Dict[str, Any]:
    """단일 데이터셋 벤치마크"""
    from forecastx import ChaniCast

    n = len(df)
    splitIdx = int(n * trainRatio)
    trainDf = df.iloc[:splitIdx].reset_index(drop=True)
    testDf = df.iloc[splitIdx:].reset_index(drop=True)
    testSteps = len(testDf)

    results = {'data': dataName, 'n': n, 'testSteps': testSteps}

    # ChaniCast Native
    try:
        startTime = time.time()
        cc = ChaniCast(verbose=False)
        result = cc.forecast(trainDf, dateCol='date', valueCol='value', steps=testSteps)
        ccTime = time.time() - startTime

        if result.success:
            actual = testDf['value'].values
            pred = result.predictions[:len(actual)]
            ccMape = np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100
            ccRmse = np.sqrt(np.mean((actual - pred) ** 2))
            results['chanicast'] = {
                'mape': ccMape,
                'rmse': ccRmse,
                'time': ccTime,
                'model': result.bestModelName,
                'flatDetected': result.flatInfo.isFlat if result.flatInfo else False
            }
        else:
            results['chanicast'] = {'error': result.error}
    except Exception as e:
        import traceback
        results['chanicast'] = {'error': str(e), 'traceback': traceback.format_exc()}

    # statsforecast
    try:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA, AutoETS, AutoTheta

        sfDf = trainDf.copy()
        sfDf['unique_id'] = 'series1'
        sfDf = sfDf.rename(columns={'date': 'ds', 'value': 'y'})

        startTime = time.time()
        sf = StatsForecast(
            models=[AutoARIMA(season_length=7), AutoETS(season_length=7)],
            freq='D', n_jobs=1
        )
        sf.fit(sfDf)
        forecast = sf.predict(h=testSteps)
        sfTime = time.time() - startTime

        actual = testDf['value'].values
        pred = forecast['AutoARIMA'].values[:len(actual)]
        sfMape = np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100
        sfRmse = np.sqrt(np.mean((actual - pred) ** 2))
        results['statsforecast'] = {
            'mape': sfMape,
            'rmse': sfRmse,
            'time': sfTime,
            'model': 'AutoARIMA'
        }
    except Exception as e:
        results['statsforecast'] = {'error': str(e)}

    return results


def printResults(allResults: List[Dict]) -> None:
    """결과 출력"""
    print("\n" + "=" * 80)
    print("실데이터 벤치마크 결과")
    print("=" * 80)

    print(f"\n{'데이터':<20} {'ChaniCast MAPE':>15} {'statsforecast MAPE':>20} {'차이':>10}")
    print("-" * 70)

    ccMapes = []
    sfMapes = []

    for r in allResults:
        dataName = r['data']
        ccMape = r['chanicast'].get('mape', float('nan'))
        sfMape = r['statsforecast'].get('mape', float('nan'))
        diff = ccMape - sfMape if not (np.isnan(ccMape) or np.isnan(sfMape)) else 0

        if not np.isnan(ccMape):
            ccMapes.append(ccMape)
        if not np.isnan(sfMape):
            sfMapes.append(sfMape)

        ccStr = f"{ccMape:.2f}%" if not np.isnan(ccMape) else "FAIL"
        sfStr = f"{sfMape:.2f}%" if not np.isnan(sfMape) else "FAIL"
        diffStr = f"+{diff:.2f}%" if diff > 0 else f"{diff:.2f}%"

        print(f"{dataName:<20} {ccStr:>15} {sfStr:>20} {diffStr:>10}")

    print("-" * 70)
    avgCc = np.mean(ccMapes) if ccMapes else 0
    avgSf = np.mean(sfMapes) if sfMapes else 0
    print(f"{'평균':<20} {avgCc:>14.2f}% {avgSf:>19.2f}% {avgCc - avgSf:>+9.2f}%")

    # 성능 비율
    if avgSf > 0:
        ratio = avgCc / avgSf
        print(f"\nChaniCast / statsforecast 비율: {ratio:.2f}x")
        print(f"목표 달성률 (1.0x 기준): {100/ratio:.1f}%")


def runFullBenchmark():
    """전체 벤치마크 실행"""
    print("=" * 80)
    print("E001: 실데이터 벤치마크")
    print("=" * 80)

    generator = RealDataGenerator()

    datasets = [
        ('retail_sales', generator.retailSales()),
        ('stock_price', generator.stockPrice()),
        ('temperature', generator.temperature()),
        ('website_traffic', generator.websiteTraffic()),
        ('energy', generator.energyConsumption()),
        ('manufacturing', generator.manufacturing()),
    ]

    allResults = []

    for dataName, df in datasets:
        print(f"\n처리 중: {dataName} (n={len(df)})")
        result = runSingleBenchmark(df, dataName)
        allResults.append(result)

        # 즉시 출력
        ccMape = result['chanicast'].get('mape', float('nan'))
        sfMape = result['statsforecast'].get('mape', float('nan'))
        ccError = result['chanicast'].get('error', '')
        print(f"  ChaniCast: {ccMape:.2f}% | statsforecast: {sfMape:.2f}%")
        if ccError:
            print(f"  [ChaniCast 오류] {ccError[:100]}")

    printResults(allResults)

    return allResults


if __name__ == '__main__':
    results = runFullBenchmark()
