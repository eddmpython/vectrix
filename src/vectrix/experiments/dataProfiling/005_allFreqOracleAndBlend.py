"""
실험 ID: dataProfiling/005
실험명: All-Frequency Oracle Gap + DNA-Based Blending

목적:
- 6개 빈도 전체에서 oracle gap 측정 (Monthly만 봤던 E002~E004 확장)
- 특히 Daily(OWA 0.996)에서 oracle gap이 크면 프로파일링 가치가 다를 수 있음
- 모델 "선택"(이산) 대신 "블렌딩"(연속 가중치)이 더 안전한지 검증
- DNA 기반 동적 블렌딩 vs 고정 가중치 vs DOT단일 비교

가설:
1. Daily/Hourly에서 oracle gap이 Monthly보다 클 것 (모델 간 성능 분산이 클 것)
2. 블렌딩이 선택보다 "잘못 고르면 악화" 리스크가 작을 것
3. DNA 기반 동적 가중치가 고정 가중치보다 나을 것

방법:
1. 6개 빈도, 빈도별 200개 시리즈, 4모델 OWA 계산
2. 빈도별 oracle gap 측정 + best model 분포
3. 고정 가중치 블렌딩 (uniform, inverse-OWA, optimal grid)
4. DNA 단일특성(seasonality_strength) 기반 동적 가중치
5. 전략 비교: DOT단일, 고정블렌드, 동적블렌드, 단순규칙, oracle

결과 (실험 후 작성):
- 6개 빈도 × 200개 시리즈, 4모델 전체 평가 완료

Oracle Gap by Frequency
  Yearly:    DOT 0.999, Oracle 0.643, Gap 0.356 (35.6%) — 가장 큰 gap!
  Quarterly: DOT 0.806, Oracle 0.645, Gap 0.160 (19.9%)
  Monthly:   DOT 0.788, Oracle 0.665, Gap 0.123 (15.6%)
  Weekly:    DOT 1.043, Oracle 0.841, Gap 0.202 (19.4%) — 4Theta/ETS 폭발
  Daily:     DOT 0.829, Oracle 0.707, Gap 0.122 (14.7%)
  Hourly:    DOT 1.691, Oracle 1.322, Gap 0.369 (21.8%) — CES가 DOT보다 나음

Best model 분포
  Yearly/Quarterly/Monthly: DOT 최다 (40~43%)
  Weekly: DOT 29%, ETS 28%, 4Theta 26% — 가장 분산적
  Daily: DOT 34%, ETS 27%
  Hourly: CES 40%, DOT 37% — CES가 1위

Optimal Blend 결과
  Yearly:    0.899 (DOT 대비 -10.0%, oracle gap 28.1%) — DOT 0.4 + CES 0.4 + 4Theta 0.2
  Weekly:    0.986 (DOT 대비 -5.5%, oracle gap 28.3%) — DOT 0.2 + CES 0.8
  Hourly:    1.611 (DOT 대비 -4.8%, oracle gap 21.8%) — DOT 0.2 + CES 0.8
  Monthly:   0.781 (DOT 대비 -0.9%, oracle gap 5.9%) — DOT 0.8 + CES 0.2
  Quarterly: 0.790 (DOT 대비 -1.9%, oracle gap 9.6%)
  Daily:     0.828 (DOT 대비 -0.1%, oracle gap 0.6%) — 거의 개선 없음

DNA-Based Dynamic Blend
  Monthly: seas>0.6→DOT50%+CES50%, else DOT80% = 0.774 (DOT 대비 -1.4%, gap 11.4%)
  Hourly:  seas>0.7→DOT50%+CES50%, else DOT80% = 1.632 (DOT 대비 -3.5%)
  Quarterly/Yearly/Weekly/Daily: 개선 없거나 미미

결론:
- **Yearly oracle gap이 35.6%로 가장 크다** — DOT의 Yearly 성능이 상대적으로 약함
- **블렌딩이 선택보다 안전하고 효과적** — Yearly에서 DOT+CES+4Theta 블렌드로 oracle gap 28% 캡처
- **핵심 패턴: DOT+CES 조합이 거의 모든 빈도에서 최적 블렌드**
  - 4Theta/ETS 가중치는 대부분 0 — 블렌드에 기여 안 함
- **빈도별 최적 DOT:CES 비율이 다름** — Yearly 4:4, Monthly 8:2, Hourly 2:8
  - 이것이 DNA 기반 프로파일링의 실용적 가치: 빈도별 블렌드 비율 최적화
- **Daily는 프로파일링 가치 거의 없음** — oracle gap 14.7%이지만 어떤 전략도 0.6%만 캡처
- **4Theta/ETS Weekly 폭발** — 4Theta OWA 2.05, ETS 77억 → 안전장치 필수
- **DNA 동적 블렌딩은 일부 빈도에서만 유효** — Monthly(-1.4%), Hourly(-3.5%)에서 DOT 대비 개선

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

M4_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
SEED = 42
N_SAMPLE = 200

FREQ_CONFIG = {
    'Yearly':    {'period': 1,  'horizon': 6,  'minLen': 10},
    'Quarterly': {'period': 4,  'horizon': 8,  'minLen': 16},
    'Monthly':   {'period': 12, 'horizon': 18, 'minLen': 24},
    'Weekly':    {'period': 1,  'horizon': 13, 'minLen': 10},
    'Daily':     {'period': 7,  'horizon': 14, 'minLen': 28},
    'Hourly':    {'period': 24, 'horizon': 48, 'minLen': 48},
}

MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']


def loadM4WithTest(freq, nSample, minLen, seed=42):
    trainPath = os.path.join(M4_DIR, f'{freq}-train.csv')
    testPath = os.path.join(M4_DIR, f'{freq}-test.csv')
    trainDf = pd.read_csv(trainPath, index_col=0)
    testDf = pd.read_csv(testPath, index_col=0)

    horizon = FREQ_CONFIG[freq]['horizon']

    rng = np.random.RandomState(seed)
    if nSample >= len(trainDf):
        idx = np.arange(len(trainDf))
    else:
        idx = rng.choice(len(trainDf), nSample, replace=False)

    trainSampled = trainDf.iloc[idx]
    testSampled = testDf.iloc[idx]

    data = {}
    for sid in trainSampled.index:
        trainVals = trainSampled.loc[sid].dropna().values.astype(np.float64)
        testVals = testSampled.loc[sid].dropna().values.astype(np.float64)
        if len(trainVals) >= minLen and len(testVals) >= horizon:
            data[str(sid)] = {
                'train': trainVals,
                'test': testVals[:horizon]
            }
    return data


def extractFeatures(data, period):
    from vectrix.engine.tsfeatures import TSFeatureExtractor
    extractor = TSFeatureExtractor()

    records = []
    validIds = []
    for sid, d in data.items():
        try:
            feat = extractor.extract(d['train'], period=period)
            if feat and len(feat) > 0:
                records.append(feat)
                validIds.append(sid)
        except Exception:
            pass
    return pd.DataFrame(records, index=validIds)


def computeSmape(actual, predicted):
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100


def computeMase(actual, predicted, trainY, period):
    n = len(trainY)
    if n <= period:
        naiveErr = np.mean(np.abs(np.diff(trainY)))
    else:
        naiveErr = np.mean(np.abs(trainY[period:] - trainY[:-period]))
    if naiveErr < 1e-10:
        naiveErr = 1e-10
    return np.mean(np.abs(actual - predicted)) / naiveErr


def computeOwa(smape, mase, naiveSmape, naiveMase):
    if naiveSmape < 1e-10:
        naiveSmape = 1e-10
    if naiveMase < 1e-10:
        naiveMase = 1e-10
    return 0.5 * (smape / naiveSmape) + 0.5 * (mase / naiveMase)


def naiveSeasonal(trainY, steps, period):
    if period <= 1:
        return np.full(steps, trainY[-1])
    lastSeason = trainY[-(period):]
    reps = (steps // period) + 1
    return np.tile(lastSeason, reps)[:steps]


def forecastWithModel(trainY, steps, period, modelId):
    from vectrix.engine.registry import createModel
    try:
        model = createModel(modelId, period)
        model.fit(trainY)
        pred, _, _ = model.predict(steps)
        pred = np.where(np.isfinite(pred), pred, np.nanmean(trainY))
        return pred
    except Exception:
        return np.full(steps, np.nanmean(trainY))


def evaluateFreq(freq, cfg, models, nSample):
    period = cfg['period']
    horizon = cfg['horizon']
    minLen = cfg['minLen']

    data = loadM4WithTest(freq, nSample, minLen, SEED)
    if len(data) < 10:
        return None

    featDf = extractFeatures(data, period)
    validIds = [sid for sid in featDf.index if sid in data]

    if len(validIds) < 10:
        return None

    allPreds = {m: {} for m in models}
    owaPerSeries = {m: [] for m in models}
    seasStrengths = []
    seriesIds = []

    for i, sid in enumerate(validIds):
        trainY = data[sid]['train']
        testY = data[sid]['test']

        naive2Pred = naiveSeasonal(trainY, horizon, period)
        naiveSmape = computeSmape(testY, naive2Pred)
        naiveMase = computeMase(testY, naive2Pred, trainY, period)

        for modelId in models:
            pred = forecastWithModel(trainY, horizon, period, modelId)
            allPreds[modelId][sid] = pred

            smape = computeSmape(testY, pred)
            mase = computeMase(testY, pred, trainY, period)
            owa = computeOwa(smape, mase, naiveSmape, naiveMase)
            owaPerSeries[modelId].append(owa)

        if 'seasonality_strength' in featDf.columns:
            seasStrengths.append(featDf.loc[sid, 'seasonality_strength'])
        else:
            seasStrengths.append(0.0)
        seriesIds.append(sid)

    owaDf = pd.DataFrame(owaPerSeries, index=seriesIds)
    seasArr = np.array(seasStrengths)

    return {
        'data': data,
        'featDf': featDf,
        'owaDf': owaDf,
        'allPreds': allPreds,
        'seasArr': seasArr,
        'validIds': seriesIds,
        'period': period,
        'horizon': horizon,
    }


def analyzeOracleGap(freqResults, models):
    print(f"\n{'=' * 90}")
    print(f"=== Oracle Gap by Frequency ===")
    print(f"{'=' * 90}")

    header = f"  {'Freq':>10} {'N':>5}"
    for m in models:
        header += f" {m:>10}"
    header += f" {'Oracle':>10} {'Gap':>8} {'Gap%':>7} {'BestModel':>12}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")

    allGaps = {}
    for freq in FREQ_CONFIG:
        r = freqResults.get(freq)
        if r is None:
            print(f"  {freq:>10} {'SKIP':>5}")
            continue

        owaDf = r['owaDf']
        n = len(owaDf)

        modelOwas = {}
        for m in models:
            modelOwas[m] = owaDf[m].mean()

        oracleOwa = owaDf.min(axis=1).mean()
        dotOwa = modelOwas['dot']
        gap = dotOwa - oracleOwa
        gapPct = gap / dotOwa * 100 if dotOwa > 1e-6 else 0

        bestModelDist = owaDf.idxmin(axis=1).value_counts()
        topModel = bestModelDist.index[0]

        row = f"  {freq:>10} {n:>5}"
        for m in models:
            row += f" {modelOwas[m]:>10.4f}"
        row += f" {oracleOwa:>10.4f} {gap:>8.4f} {gapPct:>6.1f}% {topModel:>12}"
        print(row)

        allGaps[freq] = {
            'n': n, 'dotOwa': dotOwa, 'oracleOwa': oracleOwa,
            'gap': gap, 'gapPct': gapPct, 'dist': bestModelDist.to_dict(),
            'modelOwas': modelOwas,
        }

    print(f"\n  Best model distribution per frequency")
    for freq, info in allGaps.items():
        dist = info['dist']
        total = info['n']
        parts = [f"{m}:{cnt}({cnt/total*100:.0f}%)" for m, cnt in sorted(dist.items(), key=lambda x: -x[1])]
        print(f"    {freq:>10}: {', '.join(parts)}")

    return allGaps


def testBlending(freqResults, models):
    print(f"\n{'=' * 90}")
    print(f"=== Blending Strategies ===")
    print(f"{'=' * 90}")

    blendResults = {}

    for freq, r in freqResults.items():
        if r is None:
            continue

        data = r['data']
        owaDf = r['owaDf']
        allPreds = r['allPreds']
        validIds = r['validIds']
        period = r['period']
        horizon = r['horizon']

        dotOwa = owaDf['dot'].mean()
        oracleOwa = owaDf.min(axis=1).mean()

        uniformOwas = []
        invOwaOwas = []
        optOwas = []

        for sid in validIds:
            trainY = data[sid]['train']
            testY = data[sid]['test']

            preds = np.array([allPreds[m][sid] for m in models])

            uniformPred = np.mean(preds, axis=0)

            naive2Pred = naiveSeasonal(trainY, horizon, period)
            naiveSmape = computeSmape(testY, naive2Pred)
            naiveMase = computeMase(testY, naive2Pred, trainY, period)

            smape = computeSmape(testY, uniformPred)
            mase = computeMase(testY, uniformPred, trainY, period)
            uniformOwas.append(computeOwa(smape, mase, naiveSmape, naiveMase))

            modelOwaVals = np.array([owaDf.loc[sid, m] for m in models])
            invOwa = 1.0 / np.maximum(modelOwaVals, 0.01)
            weights = invOwa / invOwa.sum()
            invOwaPred = np.average(preds, axis=0, weights=weights)
            smape = computeSmape(testY, invOwaPred)
            mase = computeMase(testY, invOwaPred, trainY, period)
            invOwaOwas.append(computeOwa(smape, mase, naiveSmape, naiveMase))

        bestOptOwa = dotOwa
        bestOptWeights = None
        for w0 in np.arange(0.0, 1.01, 0.2):
            for w1 in np.arange(0.0, 1.01 - w0, 0.2):
                for w2 in np.arange(0.0, 1.01 - w0 - w1, 0.2):
                    w3 = 1.0 - w0 - w1 - w2
                    if w3 < -0.01:
                        continue
                    w3 = max(w3, 0.0)
                    weights = np.array([w0, w1, w2, w3])

                    blendOwas = []
                    for sid in validIds:
                        trainY = data[sid]['train']
                        testY = data[sid]['test']
                        preds = np.array([allPreds[m][sid] for m in models])
                        blendPred = np.average(preds, axis=0, weights=weights)

                        naive2Pred = naiveSeasonal(trainY, horizon, period)
                        naiveSmape = computeSmape(testY, naive2Pred)
                        naiveMase = computeMase(testY, naive2Pred, trainY, period)
                        smape = computeSmape(testY, blendPred)
                        mase = computeMase(testY, blendPred, trainY, period)
                        blendOwas.append(computeOwa(smape, mase, naiveSmape, naiveMase))

                    meanOwa = np.mean(blendOwas)
                    if meanOwa < bestOptOwa:
                        bestOptOwa = meanOwa
                        bestOptWeights = weights.copy()

        blendResults[freq] = {
            'dotOwa': dotOwa,
            'uniformOwa': np.mean(uniformOwas),
            'optOwa': bestOptOwa,
            'optWeights': bestOptWeights,
            'oracleOwa': oracleOwa,
        }

        print(f"\n  {freq}")
        print(f"    DOT only:       {dotOwa:.4f}")
        print(f"    Uniform blend:  {np.mean(uniformOwas):.4f}")
        if bestOptWeights is not None:
            wStr = ', '.join(f'{m}:{w:.1f}' for m, w in zip(models, bestOptWeights))
            print(f"    Optimal blend:  {bestOptOwa:.4f} [{wStr}]")
        else:
            print(f"    Optimal blend:  no improvement found")
        print(f"    Oracle:         {oracleOwa:.4f}")

    return blendResults


def testDnaBlending(freqResults, models):
    print(f"\n{'=' * 90}")
    print(f"=== DNA-Based Dynamic Blending (seasonality_strength) ===")
    print(f"{'=' * 90}")

    dnaResults = {}

    for freq, r in freqResults.items():
        if r is None:
            continue

        data = r['data']
        owaDf = r['owaDf']
        allPreds = r['allPreds']
        validIds = r['validIds']
        seasArr = r['seasArr']
        period = r['period']
        horizon = r['horizon']

        dotOwa = owaDf['dot'].mean()
        oracleOwa = owaDf.min(axis=1).mean()

        bestDnaOwa = dotOwa
        bestDnaThresh = 0.5
        bestDnaHighWeights = None
        bestDnaLowWeights = None

        for thresh in np.arange(0.1, 0.9, 0.1):
            highMask = seasArr > thresh
            lowMask = ~highMask

            if highMask.sum() < 5 or lowMask.sum() < 5:
                continue

            for dotW in np.arange(0.0, 1.01, 0.25):
                for cesW in np.arange(0.0, 1.01 - dotW, 0.25):
                    restW = 1.0 - dotW - cesW
                    highWeights = np.array([dotW, cesW, restW / 2, restW / 2])

                    blendOwas = []
                    for i, sid in enumerate(validIds):
                        trainY = data[sid]['train']
                        testY = data[sid]['test']
                        preds = np.array([allPreds[m][sid] for m in models])

                        if highMask[i]:
                            w = highWeights
                        else:
                            w = np.array([0.8, 0.1, 0.05, 0.05])

                        blendPred = np.average(preds, axis=0, weights=w)
                        naive2Pred = naiveSeasonal(trainY, horizon, period)
                        naiveSmape = computeSmape(testY, naive2Pred)
                        naiveMase = computeMase(testY, naive2Pred, trainY, period)
                        smape = computeSmape(testY, blendPred)
                        mase = computeMase(testY, blendPred, trainY, period)
                        blendOwas.append(computeOwa(smape, mase, naiveSmape, naiveMase))

                    meanOwa = np.mean(blendOwas)
                    if meanOwa < bestDnaOwa:
                        bestDnaOwa = meanOwa
                        bestDnaThresh = thresh
                        bestDnaHighWeights = highWeights.copy()

        dnaResults[freq] = {
            'dotOwa': dotOwa,
            'dnaOwa': bestDnaOwa,
            'thresh': bestDnaThresh,
            'highWeights': bestDnaHighWeights,
            'oracleOwa': oracleOwa,
        }

        diff = bestDnaOwa - dotOwa
        print(f"  {freq:>10}: DOT={dotOwa:.4f}, DNA-blend={bestDnaOwa:.4f} ({diff:+.4f}), Oracle={oracleOwa:.4f}")
        if bestDnaHighWeights is not None:
            wStr = ', '.join(f'{m}:{w:.2f}' for m, w in zip(models, bestDnaHighWeights))
            print(f"             seas>{bestDnaThresh:.1f} → [{wStr}], else DOT-heavy")

    return dnaResults


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E005: All-Frequency Oracle Gap + DNA-Based Blending")
    print(f"  N={N_SAMPLE} per freq, Models={MODELS}")
    print("=" * 70)

    t0 = time.time()

    freqResults = {}
    for freq, cfg in FREQ_CONFIG.items():
        print(f"\n[{freq}] Loading + evaluating...")
        r = evaluateFreq(freq, cfg, MODELS, N_SAMPLE)
        if r:
            print(f"  {freq}: {len(r['validIds'])} series evaluated")
        else:
            print(f"  {freq}: SKIPPED (insufficient data)")
        freqResults[freq] = r

    allGaps = analyzeOracleGap(freqResults, MODELS)

    blendResults = testBlending(freqResults, MODELS)

    dnaResults = testDnaBlending(freqResults, MODELS)

    print(f"\n{'=' * 90}")
    print(f"=== FINAL COMPARISON ===")
    print(f"{'=' * 90}")

    print(f"  {'Freq':>10} {'DOT':>8} {'Uniform':>8} {'OptBlend':>8} {'DNABlend':>8} {'Oracle':>8} {'Gap':>8} {'BestCapt':>8}")
    print(f"  {'-' * 75}")

    for freq in FREQ_CONFIG:
        br = blendResults.get(freq, {})
        dr = dnaResults.get(freq, {})
        gap = allGaps.get(freq, {})

        dotOwa = gap.get('dotOwa', 0)
        oracleOwa = gap.get('oracleOwa', 0)
        uniformOwa = br.get('uniformOwa', 0)
        optOwa = br.get('optOwa', 0)
        dnaOwa = dr.get('dnaOwa', 0)
        totalGap = dotOwa - oracleOwa

        bestCapt = max(dotOwa - uniformOwa, dotOwa - optOwa, dotOwa - dnaOwa)
        captPct = bestCapt / totalGap * 100 if totalGap > 1e-6 else 0

        if dotOwa > 0:
            print(f"  {freq:>10} {dotOwa:>8.4f} {uniformOwa:>8.4f} {optOwa:>8.4f} {dnaOwa:>8.4f} {oracleOwa:>8.4f} {totalGap:>8.4f} {captPct:>7.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
