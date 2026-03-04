"""
실험 ID: modelCreation/048
실험명: DOT 8-Way Pre-Filtering — 데이터 특성으로 config 축소

목적:
- DOT-Hybrid는 8개 config(2 trend × 2 model × 2 season)를 모두 fit하여 최적 선택
- 데이터 특성(트렌드 강도, 계절성 유무, 양수/음수 비율)으로 사전에 2~3개만 시도하면
  DOT fit 시간을 50~75% 절감 가능
- 단, OWA 열화가 없어야 함 — 현재 기준선 OWA 0.877

가설:
1. 특정 config가 지배적으로 선택되는 패턴이 존재 (예: additive+linear가 80%+)
2. 데이터 특성(양수 비율, 트렌드 강도, 변동성)으로 선택 가능한 config를 2~3개로 축소 가능
3. 축소해도 OWA 열화 < 0.3%

방법:
1. M4 6개 그룹에서 그룹당 200개 시리즈 = 1200개 시리즈
2. DOT-Hybrid 8-way를 모두 fit하여 각 config의 MAE 기록
3. best config 분포 분석 → 지배적 패턴 확인
4. 데이터 특성별 config 승률 분석
5. 규칙 기반 사전 필터링 시뮬레이션 → OWA 비교

결과 (실험 후 작성):
- M4 1154개 시리즈(그룹당 ~200개, Yearly는 154개) 8+4 config 전부 fit

  Global config 승률 (12개 config, 계절성 있는 그룹은 8개, 없으면 4개)
    exponential_additive_additive            15.6%
    exponential_additive_multiplicative      14.9%
    linear_additive_additive                 11.9%
    linear_multiplicative_multiplicative     10.4%
    exponential_multiplicative_multiplicative 9.5%
    linear_additive_multiplicative            9.0%
    (나머지 6개: 1.7~7.9%)

  Top-K coverage (global 기준)
    Top-2: 30.5%
    Top-3: 42.4%
    Top-4: 52.8%

  그룹별 패턴
    Yearly    exponential_additive_none 41.6% (비계절, 4개 config만)
    Quarterly exponential_additive_additive 33.0% (가장 집중)
    Monthly   8개 config가 8.5~18.5%로 고르게 분포 (사전 필터링 불가)
    Weekly    8개 config가 9.5~17.5%로 고르게 분포
    Daily     8개 config가 9.5~18.0%로 고르게 분포
    Hourly    상위 5개가 15~18.5%, 하위 3개가 4~5%

  Smart filter (pos_ratio 기반) AVG coverage = 30.1% — 실패

결론:
- **기각** — 8개 config가 고르게 분산되어 있어 안전한 사전 축소 불가
- 가설 1 기각: 최다 config가 15.6%에 불과. "지배적 config" 패턴 없음
- 가설 2 기각: Top-4 config도 52.8% 커버리지. 나머지 47.2%에서 정확도 손실
- 가설 3 기각: 3개로 축소하면 57.6%가 차선 config 강제 사용 → OWA 열화 확실
- Quarterly만 예외적으로 exponential_additive 계열이 59%로 집중
- Monthly/Weekly/Daily/Hourly는 8개 config가 거의 균등 분포 — holdout validation이 필수
- **결론: DOT 8-way는 줄일 수 없음. holdout validation이 유일하게 올바른 방법**
- **속도 최적화 방향**: 8-way 순차 fit을 병렬화하거나, _fitVariant 자체의 Rust 이전이 정답

실험일: 2026-03-05
"""
import sys
import io
import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')

    from vectrix.engine.dot import DynamicOptimizedTheta

    GROUPS = {
        'Yearly':    {'period': 1,  'horizon': 6,  'prefix': 'Y'},
        'Quarterly': {'period': 4,  'horizon': 8,  'prefix': 'Q'},
        'Monthly':   {'period': 12, 'horizon': 18, 'prefix': 'M'},
        'Weekly':    {'period': 52, 'horizon': 13, 'prefix': 'W'},
        'Daily':     {'period': 7,  'horizon': 14, 'prefix': 'D'},
        'Hourly':    {'period': 24, 'horizon': 48, 'prefix': 'H'},
    }

    N_SERIES = 200

    def smape(actual, predicted):
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        a, p = actual[mask], predicted[mask]
        denom = np.abs(a) + np.abs(p)
        denom = np.where(denom == 0, 1, denom)
        return np.mean(2 * np.abs(a - p) / denom) * 100

    def mase(actual, predicted, trainY, period):
        if period >= len(trainY):
            period = 1
        naiveDiffs = np.abs(trainY[period:] - trainY[:-period])
        scale = np.mean(naiveDiffs) if len(naiveDiffs) > 0 else 1.0
        if scale < 1e-10:
            scale = 1.0
        return np.mean(np.abs(actual - predicted)) / scale

    M4_SMAPE = {
        'Yearly': 13.528, 'Quarterly': 9.733, 'Monthly': 12.126,
        'Weekly': 7.817, 'Daily': 3.045, 'Hourly': 9.328,
    }
    M4_MASE = {
        'Yearly': 2.980, 'Quarterly': 1.111, 'Monthly': 0.836,
        'Weekly': 2.108, 'Daily': 3.278, 'Hourly': 0.821,
    }

    allConfigWins = {}
    allSeriesData = []

    for groupName, gCfg in GROUPS.items():
        period = gCfg['period']
        horizon = gCfg['horizon']

        trainFile = os.path.join(DATA_DIR, f'{groupName}-train.csv')
        testFile = os.path.join(DATA_DIR, f'{groupName}-test.csv')
        if not os.path.exists(trainFile):
            print(f"[SKIP] {groupName} — data not found")
            continue

        trainDf = pd.read_csv(trainFile)
        testDf = pd.read_csv(testFile)

        nRows = min(N_SERIES, len(trainDf))

        configWins = {}
        configSmapes = {}

        print(f"\n=== {groupName} (period={period}, h={horizon}, n={nRows}) ===")

        for rowIdx in range(nRows):
            trainY = trainDf.iloc[rowIdx, 1:].dropna().values.astype(float)
            testY = testDf.iloc[rowIdx, 1:].dropna().values[:horizon].astype(float)
            sid = trainDf.iloc[rowIdx, 0]

            if len(trainY) < 20 or len(testY) < horizon:
                continue

            n = len(trainY)
            hasSeason = period > 1 and n >= period * 3
            seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']

            scaled = trainY.copy()
            base = np.mean(np.abs(scaled))
            if base > 0:
                scaled = scaled / base
            else:
                base = 1.0

            dot = DynamicOptimizedTheta(period=period)

            bestMae = np.inf
            bestConfig = None
            configResults = {}

            for seasonType in seasonTypes:
                if seasonType != 'none':
                    seasonal, deseasonalized = dot._deseasonalizeAdvanced(scaled, period, seasonType)
                else:
                    seasonal = None
                    deseasonalized = scaled

                for trendType in ['linear', 'exponential']:
                    thetaLine0 = dot._fitTrendLine(deseasonalized, trendType)
                    if thetaLine0 is None:
                        continue

                    for modelType in ['additive', 'multiplicative']:
                        if modelType == 'multiplicative' and np.any(thetaLine0 <= 0):
                            continue
                        if modelType == 'multiplicative' and np.any(deseasonalized <= 0):
                            continue

                        configKey = f"{trendType}_{modelType}_{seasonType}"

                        result = dot._fitVariant(deseasonalized, thetaLine0, trendType, modelType)
                        if result is None:
                            continue

                        pred = dot._predictVariantSteps(result, trendType, modelType, horizon)
                        if seasonal is not None:
                            for h in range(horizon):
                                idx = (n + h) % period
                                if seasonType == 'multiplicative':
                                    pred[h] *= seasonal[idx]
                                else:
                                    pred[h] += seasonal[idx]

                        pred = pred * base
                        pred = np.clip(pred, -1e15, 1e15)
                        pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)

                        s = smape(testY, pred)
                        m = mase(testY, pred, trainY, period)

                        configResults[configKey] = {'smape': s, 'mase': m}

                        if s < bestMae:
                            bestMae = s
                            bestConfig = configKey

            if bestConfig:
                configWins[bestConfig] = configWins.get(bestConfig, 0) + 1

                for ck, cv in configResults.items():
                    if ck not in configSmapes:
                        configSmapes[ck] = []
                    configSmapes[ck].append(cv['smape'])

                posRatio = np.mean(trainY > 0)
                trendSlope = np.polyfit(np.arange(n), trainY, 1)[0] / (np.std(trainY) + 1e-10)
                volatility = np.std(np.diff(trainY)) / (np.mean(np.abs(trainY)) + 1e-10)

                allSeriesData.append({
                    'group': groupName,
                    'sid': sid,
                    'bestConfig': bestConfig,
                    'posRatio': posRatio,
                    'trendSlope': trendSlope,
                    'volatility': volatility,
                    'n': n,
                    'period': period,
                })

        print(f"\n  Config win distribution:")
        totalWins = sum(configWins.values())
        for ck, cnt in sorted(configWins.items(), key=lambda x: -x[1]):
            print(f"    {ck:40s}: {cnt:4d} ({cnt/totalWins*100:.1f}%)")

        allConfigWins[groupName] = configWins

    print("\n\n" + "=" * 70)
    print("GLOBAL CONFIG WIN DISTRIBUTION")
    print("=" * 70)

    from collections import Counter
    globalWins = Counter()
    for gw in allConfigWins.values():
        globalWins.update(gw)

    totalGlobal = sum(globalWins.values())
    for ck, cnt in globalWins.most_common():
        print(f"  {ck:40s}: {cnt:4d} ({cnt/totalGlobal*100:.1f}%)")

    print("\n\n" + "=" * 70)
    print("TOP-K CONFIG COVERAGE ANALYSIS")
    print("=" * 70)

    for K in [2, 3, 4]:
        topK = [ck for ck, _ in globalWins.most_common(K)]
        covered = sum(globalWins[ck] for ck in topK)
        print(f"  Top-{K} configs cover {covered}/{totalGlobal} = {covered/totalGlobal*100:.1f}%")
        print(f"    Configs: {topK}")

    print("\n\n" + "=" * 70)
    print("DATA-DRIVEN FILTERING RULES")
    print("=" * 70)

    df = pd.DataFrame(allSeriesData)

    for groupName in GROUPS:
        gdf = df[df['group'] == groupName]
        if len(gdf) == 0:
            continue

        print(f"\n  {groupName}:")

        posOnly = gdf[gdf['posRatio'] >= 0.99]
        mixed = gdf[gdf['posRatio'] < 0.99]

        if len(posOnly) > 0:
            posCounts = Counter(posOnly['bestConfig'])
            topPosConfigs = posCounts.most_common(3)
            print(f"    All-positive (n={len(posOnly)}): {topPosConfigs}")

        if len(mixed) > 0:
            mixCounts = Counter(mixed['bestConfig'])
            topMixConfigs = mixCounts.most_common(3)
            print(f"    Has negatives (n={len(mixed)}): {topMixConfigs}")

        highTrend = gdf[gdf['trendSlope'].abs() > 0.01]
        lowTrend = gdf[gdf['trendSlope'].abs() <= 0.01]

        if len(highTrend) > 0:
            htCounts = Counter(highTrend['bestConfig'])
            print(f"    Strong trend (n={len(highTrend)}): {htCounts.most_common(3)}")

        if len(lowTrend) > 0:
            ltCounts = Counter(lowTrend['bestConfig'])
            print(f"    Weak trend (n={len(lowTrend)}): {ltCounts.most_common(3)}")

    print("\n\n" + "=" * 70)
    print("SIMULATION: Pre-filtering OWA Impact")
    print("=" * 70)

    def simulatePreFilter(seriesData, allConfigWins, filterFn, filterName):
        print(f"\n  Strategy: {filterName}")
        groupOwas = []

        for groupName, gCfg in GROUPS.items():
            gdf_list = [s for s in seriesData if s['group'] == groupName]
            if not gdf_list:
                continue

            hits = 0
            total = len(gdf_list)

            for s in gdf_list:
                allowed = filterFn(s)
                if s['bestConfig'] in allowed:
                    hits += 1

            coverage = hits / total * 100
            groupOwas.append(coverage)
            print(f"    {groupName:10s}: coverage={coverage:.1f}% ({hits}/{total})")

        if groupOwas:
            print(f"    {'AVG':10s}: coverage={np.mean(groupOwas):.1f}%")

    topGlobal2 = [ck for ck, _ in globalWins.most_common(2)]
    topGlobal3 = [ck for ck, _ in globalWins.most_common(3)]
    topGlobal4 = [ck for ck, _ in globalWins.most_common(4)]

    simulatePreFilter(
        allSeriesData, allConfigWins,
        lambda s: topGlobal2,
        f"Global Top-2: {topGlobal2}"
    )

    simulatePreFilter(
        allSeriesData, allConfigWins,
        lambda s: topGlobal3,
        f"Global Top-3: {topGlobal3}"
    )

    simulatePreFilter(
        allSeriesData, allConfigWins,
        lambda s: topGlobal4,
        f"Global Top-4: {topGlobal4}"
    )

    def smartFilter(s):
        posRatio = s['posRatio']
        configs = []

        if posRatio >= 0.99:
            configs = ['linear_additive_multiplicative', 'linear_multiplicative_multiplicative',
                       'linear_additive_additive', 'exponential_multiplicative_multiplicative']
        else:
            configs = ['linear_additive_none', 'linear_additive_additive',
                       'exponential_additive_none', 'exponential_additive_additive']

        return configs[:3]

    simulatePreFilter(
        allSeriesData, allConfigWins,
        smartFilter,
        "Smart: pos_ratio based (3 configs)"
    )

    print("\n\nDone.")
