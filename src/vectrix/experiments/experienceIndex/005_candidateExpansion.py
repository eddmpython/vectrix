"""
실험 ID: experienceIndex/005
실험명: Candidate Expansion — 후보 모델 8→20개 확장 효과

========================================================================
배경
========================================================================

E002~E004에서 8후보(4모델×2전처리)로 kNN 경험 선택이 +2.8~7.2% 개선을 보였다.
Oracle gap은 22~38%인데, kNN은 그 중 10~19%만 캡처한다.

후보 모델을 확장하면:
1. Oracle gap이 커진다 (더 다양한 모델 중 정답이 있을 확률)
2. 하지만 선택 난이도도 올라간다 (20개 중 1개 고르기가 8개 중보다 어려움)

질문: 후보 확장이 순효과(Oracle gap 확대)인가, 역효과(선택 실패 증가)인가?

========================================================================

목적:
- 후보 8개(4모델×2전처리) vs 20개(10모델×2전처리) 비교
- Oracle gap 확대 → kNN 캡처율 변화 측정
- 최적 후보 풀 크기 결정

가설:
1. Oracle gap이 8후보 22% → 20후보 28%+로 확대
2. kNN 캡처율은 유지되거나 소폭 감소 (선택 난이도↑)
3. 순효과: kNN OWA는 20후보가 8후보보다 좋음 (+1% 이상)

방법:
1. M4 Monthly 10K 로드
2. 10모델 × 2전처리 = 20후보 전체 실행
3. 8후보 서브셋과 20후보 전체 비교
4. Oracle gap, kNN 선택, 캡처율 비교

결과 (실험 후 작성):
- 10000 시리즈, 유효 10000, DOT-raw OWA 0.8674

- Oracle gap 비교:
  | 후보 | Oracle OWA | Gap   |
  |------|-----------|-------|
  | 8후보  | 0.6760    | 22.1% |
  | 20후보 | 0.5848    | 32.6% |
  Oracle gap 10.5%p 확대

- kNN 선택 비교 (k=50, DB=9000):
  | 후보 | OWA    | DOT대비  | 캡처율 |
  |------|--------|----------|--------|
  | 8후보  | 0.8078 | +4.38%   | 20.9%  |
  | 20후보 | 0.8062 | +4.57%   | 14.3%  |

- 모델별 Oracle 최적 점유율 (20후보):
  dot 19.6%, esn 14.3%, auto_ces 12.3%, four_theta 10.3%,
  auto_mstl 9.7%, auto_ets 9.5%, theta 8.1%, tbats 6.7%,
  dtsf 6.3%, auto_arima 3.2%

결론:
- 가설 1 (Oracle gap 확대 28%+) → 초과 달성! 22.1% → 32.6% (+10.5%p)
- 가설 2 (캡처율 소폭 감소) → 확인. 20.9% → 14.3% (분모 증가 효과)
- 가설 3 (순효과 +1%+) → 기각. +4.38% → +4.57% (차이 0.19%p로 미미)
- 핵심: 후보 확장은 Oracle을 크게 키우지만, kNN 선택의 순효과는 작다 (+0.2%p)
- ESN이 14.3% Oracle 최적으로 높은 다양성 기여. 6개 Tier 2 모델 합산 52.3%
- 실무적 결론: 8후보로도 충분. 20후보의 추가 계산 비용 대비 이득(+0.2%p) 미미
- 그러나 ESN, auto_mstl은 검토 가치 있음 (Oracle 기여도 높음)

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
PERIOD = 12
HORIZON = 18
MIN_LEN = 48
N_SAMPLE = 10000

MODELS_SMALL = ['dot', 'auto_ces', 'four_theta', 'auto_ets']
MODELS_LARGE = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_mstl',
                'dtsf', 'esn', 'auto_arima', 'theta', 'tbats']
PREPROCESS = ['raw', 'log']


def loadM4Monthly(n=N_SAMPLE):
    trainDf = pd.read_csv(os.path.join(M4_DIR, 'Monthly-train.csv'))
    testDf = pd.read_csv(os.path.join(M4_DIR, 'Monthly-test.csv'))
    np.random.seed(SEED)
    ids = np.random.choice(trainDf['V1'].values, size=n, replace=False)
    series = []
    for sid in ids:
        trainRow = trainDf[trainDf['V1'] == sid].iloc[0, 1:].dropna().values.astype(float)
        testRow = testDf[testDf['V1'] == sid].iloc[0, 1:].dropna().values.astype(float)
        if len(trainRow) >= MIN_LEN:
            series.append((sid, trainRow, testRow))
    return series


def applyPreprocess(y, method):
    if method == 'raw':
        return y, lambda pred: pred
    if method == 'log':
        minVal = np.min(y)
        shift = abs(minVal) + 1.0 if minVal <= 0 else 0.0
        return np.log(y + shift), lambda pred: np.exp(pred) - shift
    return y, lambda pred: pred


def fitPredict(modelId, y, period, steps):
    from vectrix.engine.registry import createModel
    try:
        model = createModel(modelId, period)
        model.fit(y)
        pred, _, _ = model.predict(steps)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return None
        if np.max(np.abs(pred)) > np.max(np.abs(y)) * 100:
            return None
        return pred
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None


def computeSmape(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted))
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    return np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def computeMase(actual, predicted, trainY, period):
    naiveErrors = np.abs(trainY[period:] - trainY[:-period])
    naiveMae = np.mean(naiveErrors)
    if naiveMae == 0:
        return 0.0
    return np.mean(np.abs(actual - predicted)) / naiveMae


def computeOwa(smape, mase, naiveSmape, naiveMase):
    if naiveSmape == 0 or naiveMase == 0:
        return 1.0
    return 0.5 * (smape / naiveSmape + mase / naiveMase)


def naiveSeasonalPred(trainY, period, steps):
    return np.tile(trainY[-period:], (steps // period + 1))[:steps]


def knnSelect(targetDna, dbDnas, dbExperiences, k, candidateKeys):
    sims = dbDnas @ targetDna
    topIdx = np.argsort(sims)[-k:]
    topSims = sims[topIdx]
    topSims = np.maximum(topSims, 0)
    wSum = np.sum(topSims)
    if wSum == 0:
        topSims = np.ones(k)
        wSum = k

    candidateScores = {}
    for cKey in candidateKeys:
        weightedSum = 0
        weightTotal = 0
        for j, idx in enumerate(topIdx):
            nOwa = dbExperiences[idx]['owas'].get(cKey, 99.0)
            if nOwa < 99.0:
                weightedSum += topSims[j] * nOwa
                weightTotal += topSims[j]
        if weightTotal > 0:
            candidateScores[cKey] = weightedSum / weightTotal

    if not candidateScores:
        return None
    return min(candidateScores, key=candidateScores.get)


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    from vectrix.engine.tsfeatures import TSFeatureExtractor

    startTime = time.time()

    print("=" * 70)
    print("experienceIndex/005: Candidate Expansion")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series")

    extractor = TSFeatureExtractor()
    allDnas = []
    featureNames = None

    print("\n--- DNA 추출 ---")
    for idx, (sid, trainY, testY) in enumerate(series):
        features = extractor.extract(trainY, period=PERIOD)
        if featureNames is None:
            featureNames = sorted(features.keys())
        vec = np.array([features.get(f, 0.0) for f in featureNames])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        allDnas.append(vec)
        if (idx + 1) % 5000 == 0:
            print(f"  DNA: {idx + 1}/{len(series)}")

    dnaMatrix = np.array(allDnas)
    means = np.mean(dnaMatrix, axis=0)
    stds = np.std(dnaMatrix, axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    dnaMatrixNorm = (dnaMatrix - means) / stds
    norms = np.linalg.norm(dnaMatrixNorm, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    dnaMatrixNorm = dnaMatrixNorm / norms
    print(f"  DNA: {dnaMatrix.shape[1]}d")

    print(f"\n--- 경험 수집 (10모델 × 2전처리 = 20후보) ---")

    experiences = []
    for idx, (sid, trainY, testY) in enumerate(series):
        actual = testY[:HORIZON]
        naivePred = naiveSeasonalPred(trainY, PERIOD, len(actual))
        naiveSmape = computeSmape(actual, naivePred)
        naiveMase = computeMase(actual, naivePred, trainY, PERIOD)

        candidateOwas = {}
        for modelId in MODELS_LARGE:
            for preproc in PREPROCESS:
                key = f"{modelId}_{preproc}"
                procTrain, invFunc = applyPreprocess(trainY, preproc)
                pred = fitPredict(modelId, procTrain, PERIOD, HORIZON)
                if pred is None:
                    candidateOwas[key] = 99.0
                    continue
                predOrig = invFunc(pred[:len(actual)])
                sm = computeSmape(actual, predOrig)
                ms = computeMase(actual, predOrig, trainY, PERIOD)
                owa = computeOwa(sm, ms, naiveSmape, naiveMase)
                candidateOwas[key] = owa

        dotOwa = candidateOwas.get('dot_raw', 99.0)
        experiences.append({
            'sid': sid,
            'owas': candidateOwas,
            'dotOwa': dotOwa,
        })

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - startTime
            print(f"  경험: {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

    validMask = np.array([e['dotOwa'] < 99.0 for e in experiences])
    validIdx = np.where(validMask)[0]
    expArray = np.array(experiences)
    print(f"\n  유효: {len(validIdx)}/{len(series)}")

    smallKeys = [f"{m}_{p}" for m in MODELS_SMALL for p in PREPROCESS]
    largeKeys = [f"{m}_{p}" for m in MODELS_LARGE for p in PREPROCESS]

    smallOracles = []
    largeOracles = []
    dotOwas = []
    for i in validIdx:
        dotOwa = experiences[i]['dotOwa']
        dotOwas.append(dotOwa)

        smallBest = min(experiences[i]['owas'].get(k, 99.0) for k in smallKeys)
        largeBest = min(experiences[i]['owas'].get(k, 99.0) for k in largeKeys)
        smallOracles.append(min(smallBest, 99.0))
        largeOracles.append(min(largeBest, 99.0))

    avgDot = np.mean(dotOwas)
    avgSmallOracle = np.mean(smallOracles)
    avgLargeOracle = np.mean(largeOracles)

    print(f"\n  DOT-raw:        OWA {avgDot:.4f}")
    print(f"  Oracle (8후보):  OWA {avgSmallOracle:.4f} (gap {(avgDot-avgSmallOracle)/avgDot*100:.1f}%)")
    print(f"  Oracle (20후보): OWA {avgLargeOracle:.4f} (gap {(avgDot-avgLargeOracle)/avgDot*100:.1f}%)")
    oracleExpansion = (avgSmallOracle - avgLargeOracle) / avgDot * 100
    print(f"  Oracle 확대:     {oracleExpansion:+.2f}%p (20후보가 8후보보다)")

    K = 50
    testSize = 1000
    np.random.seed(SEED + 1)
    allValid = validIdx.copy()
    np.random.shuffle(allValid)
    testIdx = allValid[:testSize]
    trainPool = allValid[testSize:]

    dbDnas = dnaMatrixNorm[trainPool]
    dbExps = expArray[trainPool]

    print(f"\n--- kNN 선택 비교 (k={K}, DB={len(trainPool)}) ---")
    print(f"\n  {'후보':>12} {'OWA':>8} {'DOT대비':>10} {'승률':>12} {'캡처율':>8}")
    print("  " + "-" * 56)

    for poolName, candidateKeys in [('8후보', smallKeys), ('20후보', largeKeys)]:
        oracleList = smallOracles if poolName == '8후보' else largeOracles
        avgOracleLocal = np.mean([oracleList[np.where(validIdx == i)[0][0]] for i in testIdx])
        oracleGapLocal = np.mean([dotOwas[np.where(validIdx == i)[0][0]] for i in testIdx]) - avgOracleLocal

        knnOwas = []
        dotOwasT = []
        wins = 0

        for i in testIdx:
            bestKey = knnSelect(dnaMatrixNorm[i], dbDnas, dbExps, K, candidateKeys)
            if bestKey is None:
                continue
            knnOwa = experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa'])
            knnOwas.append(knnOwa)
            dotOwasT.append(experiences[i]['dotOwa'])
            if knnOwa < experiences[i]['dotOwa']:
                wins += 1

        avgKnn = np.mean(knnOwas)
        avgDotT = np.mean(dotOwasT)
        impr = (avgDotT - avgKnn) / avgDotT * 100
        capture = (avgDotT - avgKnn) / oracleGapLocal * 100 if oracleGapLocal > 0 else 0
        n = len(knnOwas)
        print(f"  {poolName:>12} {avgKnn:>8.4f} {impr:>+10.2f}% {wins}/{n} ({wins/n:.1%}) {capture:>+8.1f}%")

    print(f"\n--- 모델별 Oracle 최적 빈도 ---")
    bestModelCounts = {}
    for i in validIdx:
        bestKey = min(largeKeys, key=lambda k: experiences[i]['owas'].get(k, 99.0))
        modelId = bestKey.rsplit('_', 1)[0]
        bestModelCounts[modelId] = bestModelCounts.get(modelId, 0) + 1

    total = len(validIdx)
    print(f"\n  {'모델':>15} {'Oracle최적':>10} {'비율':>8}")
    print("  " + "-" * 38)
    for m, c in sorted(bestModelCounts.items(), key=lambda x: -x[1]):
        print(f"  {m:>15} {c:>10} {c/total*100:>7.1f}%")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  시리즈: {len(series)}, 유효: {len(validIdx)}")
    print(f"  DOT-raw:        OWA {avgDot:.4f}")
    print(f"  Oracle (8후보):  {avgSmallOracle:.4f} (gap {(avgDot-avgSmallOracle)/avgDot*100:.1f}%)")
    print(f"  Oracle (20후보): {avgLargeOracle:.4f} (gap {(avgDot-avgLargeOracle)/avgDot*100:.1f}%)")
    print(f"  Oracle 확대:     {oracleExpansion:+.2f}%p")
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)
