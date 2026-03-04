"""
실험 ID: modelCreation/047
실험명: Adaptive Candidate Count — DNA 기반 모델 수 축소로 속도 향상

목적:
- 현재 forecast()는 4~5개 모델을 모두 fit+predict한 뒤 최선을 선택
- DNA 특성으로 사전에 최적 모델 1~3개만 선택하면 fit 시간을 50~75% 절감 가능
- 단, 정확도 손실이 없어야 함 — M4 OWA 0.877 기준선 대비 확인

가설:
1. DNA 특성 상위 5개(volatilityClustering, seasonalPeakPeriod, nonlinearAutocorr, demandDensity, hurstExponent)로
   최적 모델을 사전 예측하면, 4~5개 전체 fit 대비 OWA 열화 < 0.5%
2. 평균 fit 모델 수를 4.2 → 2.0 이하로 줄여 속도 2x 향상

방법:
1. M4 6개 그룹에서 그룹당 100개 시리즈 = 600개 시리즈 사용
2. 5개 후보 모델(dot, auto_ces, four_theta, auto_ets, dtsf)을 모두 fit하여 per-series 최적 모델 확인
3. DNA 65+ 특성 추출 → 어떤 모델이 1등인지 패턴 분석
4. 규칙 기반 / 간단한 분류기로 top-K(K=1,2,3) 선택 → OWA 비교
5. 기준선: 현재 엔진 (5개 모델 fit, OWA 0.877)

결과 (실험 후 작성):
- M4 600개 시리즈(그룹당 100개) 4개 모델 전부 fit

  Global 모델 승률
    auto_ces    35.8%
    dot         32.5%
    four_theta  17.1%
    auto_ets    14.6%

  그룹별 지배 모델
    Yearly     dot 43.3%
    Quarterly  four_theta 41.0%
    Monthly    dot 43.0%
    Weekly     auto_ces 49.0%
    Daily      dot 32.0% / auto_ces 28.0%
    Hourly     auto_ces 51.0% / dot 44.0%

  Oracle (per-series 최적) AVG OWA = 0.679
  4개 모델 전부 fit 후 최선 선택 = 현재 엔진 = OWA ~0.877

  가설 1 검증: 단일 모델이 40% 이하 승률 → 사전 예측으로 1개만 선택하면 60% 확률로 차선 모델 사용
  가설 2 검증: dot+auto_ces 2개만 fit하면 68.3% 커버리지, 나머지 31.7%에서 정확도 손실 발생

결론:
- **기각** — 모델 승률이 고르게 분산 (최다 35.8%)되어 있어 안전한 사전 축소 불가
- dot+auto_ces만으로 충분하지 않음 (Quarterly에서 four_theta가 41% 승률)
- 빈도별 2모델 규칙(Yearly→dot+4theta, Quarterly→4theta+ces, Hourly→ces+dot 등)도 고려 가능하나,
  이는 현재 _selectNativeModels()의 빈도별 하드코딩과 본질적으로 동일
- **속도 최적화 방향 전환 권장**: 모델 수를 줄이기보다 개별 모델 fit 속도 자체를 올리는 것이 안전

실험일: 2026-03-05
"""
import sys
import io
import os
import time

import numpy as np
import pandas as pd

if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')

    from vectrix.engine.registry import createModel
    from vectrix.engine.tsfeatures import TSFeatureExtractor

    GROUPS = {
        'Yearly':    {'period': 1,  'horizon': 6,  'prefix': 'Y'},
        'Quarterly': {'period': 4,  'horizon': 8,  'prefix': 'Q'},
        'Monthly':   {'period': 12, 'horizon': 18, 'prefix': 'M'},
        'Weekly':    {'period': 52, 'horizon': 13, 'prefix': 'W'},
        'Daily':     {'period': 7,  'horizon': 14, 'prefix': 'D'},
        'Hourly':    {'period': 24, 'horizon': 48, 'prefix': 'H'},
    }

    MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']
    N_SERIES = 100

    extractor = TSFeatureExtractor()

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

    allResults = {}
    allFeatures = []
    allBestModels = []

    for groupName, gCfg in GROUPS.items():
        period = gCfg['period']
        horizon = gCfg['horizon']
        prefix = gCfg['prefix']

        trainFile = os.path.join(DATA_DIR, f'{groupName}-train.csv')
        testFile = os.path.join(DATA_DIR, f'{groupName}-test.csv')
        if not os.path.exists(trainFile):
            print(f"[SKIP] {groupName} — data not found")
            continue

        trainDf = pd.read_csv(trainFile)
        testDf = pd.read_csv(testFile)

        nRows = min(N_SERIES, len(trainDf))

        groupSmape = {m: [] for m in MODELS}
        groupMase = {m: [] for m in MODELS}
        groupBest = []

        print(f"\n=== {groupName} (period={period}, h={horizon}, n={nRows}) ===")

        for rowIdx in range(nRows):
            trainY = trainDf.iloc[rowIdx, 1:].dropna().values.astype(float)
            testY = testDf.iloc[rowIdx, 1:].dropna().values[:horizon].astype(float)
            sid = trainDf.iloc[rowIdx, 0]

            if len(trainY) < 20 or len(testY) < horizon:
                continue

            features = extractor.extract(trainY, period=period)
            features['group'] = groupName
            features['sid'] = sid
            allFeatures.append(features)

            bestSmape = np.inf
            bestModel = MODELS[0]

            for modelId in MODELS:
                try:
                    model = createModel(modelId, period)
                    model.fit(trainY)
                    pred, _, _ = model.predict(horizon)
                    pred = np.clip(pred, -1e15, 1e15)
                    pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)

                    s = smape(testY, pred)
                    m = mase(testY, pred, trainY, period)

                    groupSmape[modelId].append(s)
                    groupMase[modelId].append(m)

                    if s < bestSmape:
                        bestSmape = s
                        bestModel = modelId
                except Exception:
                    groupSmape[modelId].append(np.nan)
                    groupMase[modelId].append(np.nan)

            groupBest.append(bestModel)
            allBestModels.append(bestModel)

        print(f"\n  Per-model sMAPE (mean):")
        for m in MODELS:
            vals = [v for v in groupSmape[m] if not np.isnan(v)]
            if vals:
                print(f"    {m:15s}: {np.mean(vals):.3f}")

        from collections import Counter
        bestCounts = Counter(groupBest)
        print(f"  Best model distribution:")
        for m, c in bestCounts.most_common():
            print(f"    {m:15s}: {c} ({c/len(groupBest)*100:.1f}%)")

        allResults[groupName] = {
            'smape': groupSmape,
            'mase': groupMase,
            'best': groupBest,
        }

    print("\n\n" + "=" * 70)
    print("GLOBAL ANALYSIS: Model Win Rates")
    print("=" * 70)

    from collections import Counter
    globalCounts = Counter(allBestModels)
    total = len(allBestModels)
    for m, c in globalCounts.most_common():
        print(f"  {m:15s}: {c:4d} ({c/total*100:.1f}%)")

    featureDf = pd.DataFrame(allFeatures)
    featureDf['best_model'] = allBestModels

    topFeatures = [
        'hurst_exponent', 'seasonal_strength', 'trend_strength',
        'nonlinearity', 'lempel_ziv_complexity', 'sample_entropy',
        'approximate_entropy', 'stability', 'lumpiness',
        'cv', 'acf1', 'seasonal_peak_period',
    ]
    available = [f for f in topFeatures if f in featureDf.columns]

    print(f"\n  DNA features available for analysis: {len(available)}")

    print("\n\n" + "=" * 70)
    print("SIMULATION: Top-K Selection OWA Impact")
    print("=" * 70)

    for K in [1, 2, 3, 4]:
        groupOwas = []

        for groupName, gCfg in GROUPS.items():
            if groupName not in allResults:
                continue

            res = allResults[groupName]
            nSeries = len(res['best'])
            if nSeries == 0:
                continue

            owaSmapes = []
            owaMases = []

            for i in range(nSeries):
                bestM = res['best'][i]

                if K >= len(MODELS):
                    selected = MODELS
                else:
                    modelRanks = []
                    for m in MODELS:
                        s = res['smape'][m][i] if i < len(res['smape'][m]) else np.nan
                        if not np.isnan(s):
                            modelRanks.append((s, m))
                    modelRanks.sort()

                    if K == 1:
                        selected = [bestM]
                    else:
                        selected = [mr[1] for mr in modelRanks[:K]]

                bestS = np.inf
                bestMaseVal = np.inf
                for m in selected:
                    s = res['smape'][m][i] if i < len(res['smape'][m]) else np.nan
                    ma = res['mase'][m][i] if i < len(res['mase'][m]) else np.nan
                    if not np.isnan(s) and s < bestS:
                        bestS = s
                        bestMaseVal = ma

                if not np.isnan(bestS) and not np.isnan(bestMaseVal):
                    owaSmapes.append(bestS)
                    owaMases.append(bestMaseVal)

            if owaSmapes:
                avgSmape = np.mean(owaSmapes)
                avgMase = np.mean(owaMases)
                relSmape = avgSmape / M4_SMAPE[groupName]
                relMase = avgMase / M4_MASE[groupName]
                owa = (relSmape + relMase) / 2
                groupOwas.append(owa)
                print(f"  K={K} {groupName:10s}: OWA={owa:.4f} (sMAPE={avgSmape:.3f}, MASE={avgMase:.3f})")

        if groupOwas:
            avgOwa = np.mean(groupOwas)
            print(f"  K={K} {'AVG':10s}: OWA={avgOwa:.4f}")
        print()

    print("\n\n" + "=" * 70)
    print("ORACLE SIMULATION: If DNA perfectly predicts best model")
    print("=" * 70)

    for K in [1, 2, 3]:
        print(f"\n  K={K} (Oracle selects top-{K} by true OOS sMAPE):")
        groupOwas = []
        for groupName in GROUPS:
            if groupName not in allResults:
                continue
            res = allResults[groupName]
            nSeries = len(res['best'])
            if nSeries == 0:
                continue

            owaSmapes = []
            owaMases = []
            for i in range(nSeries):
                modelRanks = []
                for m in MODELS:
                    s = res['smape'][m][i] if i < len(res['smape'][m]) else np.nan
                    ma = res['mase'][m][i] if i < len(res['mase'][m]) else np.nan
                    if not np.isnan(s):
                        modelRanks.append((s, ma, m))
                modelRanks.sort()

                if modelRanks:
                    bestS = modelRanks[0][0]
                    bestMa = modelRanks[0][1]
                    owaSmapes.append(bestS)
                    owaMases.append(bestMa)

            if owaSmapes:
                avgSmape = np.mean(owaSmapes)
                avgMase = np.mean(owaMases)
                relSmape = avgSmape / M4_SMAPE[groupName]
                relMase = avgMase / M4_MASE[groupName]
                owa = (relSmape + relMase) / 2
                groupOwas.append(owa)
                print(f"    {groupName:10s}: OWA={owa:.4f}")

        if groupOwas:
            print(f"    {'AVG':10s}: OWA={np.mean(groupOwas):.4f}")

    print("\n\nDone.")
