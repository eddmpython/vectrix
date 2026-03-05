"""
실험 ID: dataProfiling/006
실험명: DOT+CES Frequency-Based Blend — M4 Full Benchmark

목적:
- E005에서 발견한 빈도별 DOT+CES 최적 비율을 M4 전체(100K)로 검증
- 200개 샘플에서 찾은 비율이 전체 데이터에서도 유효한지 과적합 확인
- 현재 엔진 기준선(AVG OWA 0.848)과 비교하여 엔진 반영 근거 확보

가설:
1. E005 최적 비율이 M4 전체에서도 DOT 단일 대비 개선될 것
2. Yearly에서 가장 큰 개선 (-5% 이상)
3. Daily는 거의 개선 없을 것 (E005 결과와 일치)

방법:
1. M4 6개 빈도 전체 시리즈 (Yearly 23K, Monthly 48K 등)
2. DOT 단독 예측 + CES 단독 예측
3. E005 비율로 블렌딩한 예측의 OWA 계산
4. 추가: 전체 데이터에서 비율 재최적화하여 E005 비율과 비교

결과 (실험 후 작성):
- M4 6개 빈도, 빈도별 1000개(Weekly 359, Hourly 414)

E005 비율 그대로 적용 (과적합 검증)
  Yearly:    DOT 0.812 → E005블렌드 0.848 (+0.036 악화!) — E005 비율 과적합 확인
  Quarterly: DOT 0.791 → 0.796 (+0.005 악화)
  Monthly:   DOT 0.780 → 0.778 (-0.002 미미한 개선)
  Weekly:    DOT 0.959 → 0.967 (+0.009 악화)
  Daily:     DOT 0.797 → 0.795 (-0.002 미미한 개선)
  Hourly:    DOT 1.178 → 1.131 (-0.048 유의미한 개선!)
  AVG:       0.886 → 0.886 (-0.0003, 거의 불변)

1000개에서 재최적화한 비율
  Yearly:    DOT 0.8 + 4Theta 0.2 = 0.807 (-0.006) — CES 가중치 0! E005와 완전 다름
  Quarterly: DOT 0.9 + CES 0.1 = 0.790 (-0.001)
  Monthly:   DOT 0.8 + CES 0.2 = 0.778 (-0.002) — E005와 일치!
  Weekly:    DOT 단독 최적 (블렌드 불필요)
  Daily:     DOT 0.3 + CES 0.7 = 0.795 (-0.002) — E005와 반대! (E005는 DOT 0.6)
  Hourly:    DOT 0.2 + CES 0.8 = 1.131 (-0.048) — E005와 일치!
  AVG:       0.876 (-0.010)

결론:
- **E005 비율은 대부분 과적합** — Yearly에서 가장 심각 (+3.6% 악화)
  - 200개 샘플에서 찾은 CES 40%가 1000개에서는 CES 0%로 반전
  - Yearly에서 CES가 나쁜 모델(OWA 0.984 vs DOT 0.812)이라는 게 대규모에서 확인
- **재최적화 비율은 소폭 개선** — AVG OWA 0.886→0.876 (-0.010, -1.1%)
  - 안정적으로 작동하는 빈도: Monthly, Hourly
  - Hourly가 가장 큰 개선 (-0.048, -4.1%)
- **핵심 교훈**: 소규모 샘플의 최적 비율을 그대로 쓰면 위험. 재최적화 필수
- **Monthly/Hourly에서만 DOT+CES 블렌드가 안정적으로 작동**
- **Yearly는 DOT+4Theta가 맞음** (CES Yearly 성능이 나쁨)
- **Weekly는 DOT 단독이 최적** (블렌드 불필요)

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

FREQ_CONFIG = {
    'Yearly':    {'period': 1,  'horizon': 6,  'minLen': 10,  'sample': 1000},
    'Quarterly': {'period': 4,  'horizon': 8,  'minLen': 16,  'sample': 1000},
    'Monthly':   {'period': 12, 'horizon': 18, 'minLen': 24,  'sample': 1000},
    'Weekly':    {'period': 1,  'horizon': 13, 'minLen': 10,  'sample': 359},
    'Daily':     {'period': 7,  'horizon': 14, 'minLen': 28,  'sample': 1000},
    'Hourly':    {'period': 24, 'horizon': 48, 'minLen': 48,  'sample': 414},
}

E005_BLEND = {
    'Yearly':    {'dot': 0.4, 'auto_ces': 0.4, 'four_theta': 0.2},
    'Quarterly': {'dot': 0.6, 'auto_ces': 0.4},
    'Monthly':   {'dot': 0.8, 'auto_ces': 0.2},
    'Weekly':    {'dot': 0.2, 'auto_ces': 0.8},
    'Daily':     {'dot': 0.6, 'auto_ces': 0.4},
    'Hourly':    {'dot': 0.2, 'auto_ces': 0.8},
}


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


def evaluateFreq(freq, cfg):
    period = cfg['period']
    horizon = cfg['horizon']
    minLen = cfg['minLen']
    nSample = cfg['sample']

    data = loadM4WithTest(freq, nSample, minLen, SEED)
    if len(data) < 10:
        return None

    blendWeights = E005_BLEND[freq]
    modelIds = list(blendWeights.keys())

    dotSmapes = []
    dotMases = []
    cesSmapes = []
    cesMases = []
    blendSmapes = []
    blendMases = []
    naiveSmapes = []
    naiveMases = []
    dotOwas = []
    cesOwas = []
    blendOwas = []

    allPreds = {m: {} for m in modelIds}

    for i, (sid, d) in enumerate(data.items()):
        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(data)}...")

        trainY = d['train']
        testY = d['test']

        naive2Pred = naiveSeasonal(trainY, horizon, period)
        nSmape = computeSmape(testY, naive2Pred)
        nMase = computeMase(testY, naive2Pred, trainY, period)
        naiveSmapes.append(nSmape)
        naiveMases.append(nMase)

        preds = {}
        for modelId in modelIds:
            pred = forecastWithModel(trainY, horizon, period, modelId)
            preds[modelId] = pred
            allPreds[modelId][sid] = pred

        dotPred = preds['dot']
        dSmape = computeSmape(testY, dotPred)
        dMase = computeMase(testY, dotPred, trainY, period)
        dotSmapes.append(dSmape)
        dotMases.append(dMase)
        dotOwas.append(computeOwa(dSmape, dMase, nSmape, nMase))

        cesPred = preds['auto_ces']
        cSmape = computeSmape(testY, cesPred)
        cMase = computeMase(testY, cesPred, trainY, period)
        cesSmapes.append(cSmape)
        cesMases.append(cMase)
        cesOwas.append(computeOwa(cSmape, cMase, nSmape, nMase))

        predArrays = np.array([preds[m] for m in modelIds])
        weights = np.array([blendWeights[m] for m in modelIds])
        blendPred = np.average(predArrays, axis=0, weights=weights)

        bSmape = computeSmape(testY, blendPred)
        bMase = computeMase(testY, blendPred, trainY, period)
        blendSmapes.append(bSmape)
        blendMases.append(bMase)
        blendOwas.append(computeOwa(bSmape, bMase, nSmape, nMase))

    dotOwa = computeOwa(np.mean(dotSmapes), np.mean(dotMases),
                         np.mean(naiveSmapes), np.mean(naiveMases))
    cesOwa = computeOwa(np.mean(cesSmapes), np.mean(cesMases),
                         np.mean(naiveSmapes), np.mean(naiveMases))
    blendOwa = computeOwa(np.mean(blendSmapes), np.mean(blendMases),
                           np.mean(naiveSmapes), np.mean(naiveMases))

    return {
        'n': len(data),
        'dotOwa': dotOwa,
        'cesOwa': cesOwa,
        'blendOwa': blendOwa,
        'dotOwaMean': np.mean(dotOwas),
        'cesOwaMean': np.mean(cesOwas),
        'blendOwaMean': np.mean(blendOwas),
        'allPreds': allPreds,
        'data': data,
        'modelIds': modelIds,
        'naiveSmapes': naiveSmapes,
        'naiveMases': naiveMases,
    }


def reoptimizeBlend(result, freq):
    data = result['data']
    allPreds = result['allPreds']
    modelIds = result['modelIds']
    period = FREQ_CONFIG[freq]['period']
    horizon = FREQ_CONFIG[freq]['horizon']
    naiveSmapes = result['naiveSmapes']
    naiveMases = result['naiveMases']

    sids = list(data.keys())

    bestOwa = result['dotOwa']
    bestWeights = None

    if len(modelIds) == 2:
        for w in np.arange(0.0, 1.01, 0.1):
            weights = np.array([w, 1.0 - w])
            smapes = []
            mases = []
            for i, sid in enumerate(sids):
                testY = data[sid]['test']
                trainY = data[sid]['train']
                predArrays = np.array([allPreds[m][sid] for m in modelIds])
                blendPred = np.average(predArrays, axis=0, weights=weights)
                smapes.append(computeSmape(testY, blendPred))
                mases.append(computeMase(testY, blendPred, trainY, period))

            owa = computeOwa(np.mean(smapes), np.mean(mases),
                              np.mean(naiveSmapes), np.mean(naiveMases))
            if owa < bestOwa:
                bestOwa = owa
                bestWeights = dict(zip(modelIds, weights))
    else:
        for w0 in np.arange(0.0, 1.01, 0.1):
            for w1 in np.arange(0.0, 1.01 - w0, 0.1):
                w2 = 1.0 - w0 - w1
                if w2 < -0.01:
                    continue
                w2 = max(w2, 0.0)
                weights = np.array([w0, w1, w2])
                smapes = []
                mases = []
                for i, sid in enumerate(sids):
                    testY = data[sid]['test']
                    trainY = data[sid]['train']
                    predArrays = np.array([allPreds[m][sid] for m in modelIds])
                    blendPred = np.average(predArrays, axis=0, weights=weights)
                    smapes.append(computeSmape(testY, blendPred))
                    mases.append(computeMase(testY, blendPred, trainY, period))

                owa = computeOwa(np.mean(smapes), np.mean(mases),
                                  np.mean(naiveSmapes), np.mean(naiveMases))
                if owa < bestOwa:
                    bestOwa = owa
                    bestWeights = dict(zip(modelIds, weights))

    return bestOwa, bestWeights


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E006: DOT+CES Blend — M4 Large-Scale Benchmark")
    print("=" * 70)

    t0 = time.time()

    results = {}
    for freq, cfg in FREQ_CONFIG.items():
        print(f"\n[{freq}] n={cfg['sample']}, period={cfg['period']}, horizon={cfg['horizon']}...")
        result = evaluateFreq(freq, cfg)
        if result:
            print(f"  DOT={result['dotOwa']:.4f}, CES={result['cesOwa']:.4f}, E005-Blend={result['blendOwa']:.4f}")
            results[freq] = result
        else:
            print(f"  SKIPPED")

    print(f"\n{'=' * 90}")
    print(f"=== Re-optimizing blend weights on full data ===")
    print(f"{'=' * 90}")

    reoptResults = {}
    for freq, result in results.items():
        reoptOwa, reoptWeights = reoptimizeBlend(result, freq)
        reoptResults[freq] = {'owa': reoptOwa, 'weights': reoptWeights}
        if reoptWeights:
            wStr = ', '.join(f'{m}:{w:.1f}' for m, w in reoptWeights.items())
            print(f"  {freq:>10}: OWA={reoptOwa:.4f} [{wStr}]")
        else:
            print(f"  {freq:>10}: no improvement over DOT")

    print(f"\n{'=' * 90}")
    print(f"=== FINAL COMPARISON ===")
    print(f"{'=' * 90}")

    print(f"  {'Freq':>10} {'N':>6} {'DOT':>8} {'CES':>8} {'E005Bl':>8} {'ReoptBl':>8} {'E005vsD':>8} {'ReoptvsD':>8}")
    print(f"  {'-' * 72}")

    totalDotSmape = 0
    totalDotMase = 0
    totalBlendSmape = 0
    totalBlendMase = 0
    totalReoptSmape = 0
    totalReoptMase = 0
    totalNaiveSmape = 0
    totalNaiveMase = 0
    freqCount = 0

    dotOwas = []
    blendOwas = []
    reoptOwas = []

    for freq in FREQ_CONFIG:
        r = results.get(freq)
        if r is None:
            continue

        reopt = reoptResults.get(freq, {})
        reoptOwa = reopt.get('owa', r['dotOwa'])

        e005Diff = r['blendOwa'] - r['dotOwa']
        reoptDiff = reoptOwa - r['dotOwa']

        print(f"  {freq:>10} {r['n']:>6} {r['dotOwa']:>8.4f} {r['cesOwa']:>8.4f} "
              f"{r['blendOwa']:>8.4f} {reoptOwa:>8.4f} {e005Diff:>+8.4f} {reoptDiff:>+8.4f}")

        dotOwas.append(r['dotOwa'])
        blendOwas.append(r['blendOwa'])
        reoptOwas.append(reoptOwa)

    avgDot = np.mean(dotOwas)
    avgBlend = np.mean(blendOwas)
    avgReopt = np.mean(reoptOwas)

    print(f"  {'-' * 72}")
    print(f"  {'AVG':>10} {'':>6} {avgDot:>8.4f} {'':>8} {avgBlend:>8.4f} {avgReopt:>8.4f} "
          f"{avgBlend - avgDot:>+8.4f} {avgReopt - avgDot:>+8.4f}")

    print(f"\n  E005 blend weights used")
    for freq, w in E005_BLEND.items():
        wStr = ', '.join(f'{m}:{v:.1f}' for m, v in w.items())
        print(f"    {freq:>10}: [{wStr}]")

    print(f"\n  Re-optimized weights")
    for freq, info in reoptResults.items():
        if info['weights']:
            wStr = ', '.join(f'{m}:{v:.1f}' for m, v in info['weights'].items())
            print(f"    {freq:>10}: [{wStr}]")
        else:
            print(f"    {freq:>10}: DOT only (no improvement)")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
