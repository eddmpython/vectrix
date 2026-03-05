"""
실험 ID: experienceIndex/007
실험명: Top 50 DNA at Scale — 50d DNA가 대규모/다빈도에서도 유효한가?

========================================================================
배경
========================================================================

E006에서 Top 50 특성이 전체 65개보다 +1.13% 좋음을 확인 (5K Monthly).
그러나 이것이 소규모(5K)에서만 유효한 건 아닌지 검증 필요.

E003 (20K Monthly)과 E004 (Yearly/Quarterly)를 65d vs 50d로 재실행하여
Top 50 DNA의 효과가 스케일과 빈도에 관계없이 유지되는지 확인한다.

========================================================================

목적:
- E006의 Top 50 DNA가 20K Monthly에서도 유효한지 확인
- Yearly, Quarterly에서도 50d > 65d인지 검증
- 빈도별로 최적 차원 수가 다른지 탐색

가설:
1. Top 50은 20K Monthly에서도 65d 대비 +0.5% 이상 개선
2. Yearly/Quarterly에서도 50d >= 65d (빈도 무관하게 하위 15개가 노이즈)
3. 빈도별 최적 차원 수는 유사 (50±10)

방법:
1. Phase A: 5K Monthly에서 permutation importance → Top 50 특성 목록 추출
2. Phase B: 20K Monthly에서 65d vs 50d kNN 비교 (학습 곡선 포함)
3. Phase C: Yearly에서 65d vs 50d kNN 비교
4. Phase D: Quarterly에서 65d vs 50d kNN 비교
5. 종합 비교 테이블

결과 (실험 후 작성):
- Phase A (5K Monthly): 65d OWA 0.8131 (+5.72%), 50d OWA 0.8088 (+6.22%), 50d가 +0.53%
  → 소규모에서는 E006 결과 재현

- Phase B (20K Monthly) 전체 DB:
  65d OWA 0.8194 (+5.30%), 50d OWA 0.8217 (+5.03%), 50d가 -0.28%
  → 전체 DB에서는 65d가 근소 우위

- Phase B 학습 곡선:
  | DB크기 | 65d OWA | 50d OWA | 50d-65d |
  |--------|---------|---------|---------|
  | 2K     | 0.8417  | 0.8460  | -0.51%  |
  | 5K     | 0.8483  | 0.8481  | +0.02%  |
  | 10K    | 0.8347  | 0.8309  | +0.45%  |
  | 15K    | 0.8308  | 0.8151  | +1.88%  |
  → DB 크기↑ → 50d 효과↑ 트렌드 존재. 15K에서 +1.88%

- Phase C (Yearly): 65d 0.8647 vs 50d 0.8653, 차이 -0.07% (동등)
- Phase D (Quarterly): 65d 0.7945 vs 50d 0.7992, 차이 -0.58% (65d 우위)

- Permutation importance 분산 문제:
  E006 Top 5 (seasonal_peak_position, forecastability, spectral_entropy)가
  이번 실행에서는 제거 대상 15개에 포함됨 → importance 순위 불안정

결론:
- 가설 1 (20K에서 +0.5%+) → 조건부. 15K에서 +1.88%이지만 전체 DB에서는 -0.28%
- 가설 2 (빈도 무관 50d>=65d) → 기각. Quarterly에서 -0.58%로 65d 우위
- 가설 3 (최적 차원 수 유사) → 기각. Monthly와 Quarterly의 중요 특성이 다를 가능성
- 핵심 발견 1: Permutation importance는 실행마다 변동 → 고정 Top 50은 위험
- 핵심 발견 2: Monthly 학습 곡선에서 DB↑→50d 효과↑ 트렌드 (10K +0.45%, 15K +1.88%)
- 핵심 발견 3: Quarterly에서는 65d가 일관되게 우위 → 빈도별 중요 특성이 다름
- 실무 결론: 65d를 그대로 사용하는 것이 안전. 빈도별 차원 선택은 복잡도 대비 이득 불확실

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

MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']
PREPROCESS = ['raw', 'log']
N_CANDIDATES = len(MODELS) * len(PREPROCESS)


def loadM4(freqName, nSample):
    configs = {
        'Monthly': {'prefix': 'M', 'period': 12, 'horizon': 18, 'minLen': 48,
                     'trainFile': 'Monthly-train.csv', 'testFile': 'Monthly-test.csv'},
        'Yearly': {'prefix': 'Y', 'period': 1, 'horizon': 6, 'minLen': 20,
                   'trainFile': 'Yearly-train.csv', 'testFile': 'Yearly-test.csv'},
        'Quarterly': {'prefix': 'Q', 'period': 4, 'horizon': 8, 'minLen': 20,
                      'trainFile': 'Quarterly-train.csv', 'testFile': 'Quarterly-test.csv'},
    }
    cfg = configs[freqName]
    trainDf = pd.read_csv(os.path.join(M4_DIR, cfg['trainFile']))
    testDf = pd.read_csv(os.path.join(M4_DIR, cfg['testFile']))

    np.random.seed(SEED)
    allIds = trainDf['V1'].values
    selectedIds = set(np.random.choice(allIds, size=min(nSample, len(allIds)), replace=False))

    series = []
    for _, row in trainDf.iterrows():
        sid = row['V1']
        if sid not in selectedIds:
            continue
        trainY = row.iloc[1:].dropna().values.astype(float)
        if len(trainY) < cfg['minLen']:
            continue
        testRow = testDf[testDf['V1'] == sid]
        if testRow.empty:
            continue
        testY = testRow.iloc[0, 1:].dropna().values.astype(float)
        series.append((sid, trainY, testY))

    return series, cfg


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
    if period >= len(trainY):
        naiveMae = np.mean(np.abs(np.diff(trainY)))
    else:
        naiveErrors = np.abs(trainY[period:] - trainY[:-period])
        naiveMae = np.mean(naiveErrors)
    if naiveMae == 0:
        return 0.0
    return np.mean(np.abs(actual - predicted)) / naiveMae


def computeOwa(smape, mase, naiveSmape, naiveMase):
    if naiveSmape == 0 or naiveMase == 0:
        return 1.0
    return 0.5 * (smape / naiveSmape + mase / naiveMase)


def naivePred(trainY, period, steps):
    if period <= 1:
        return np.full(steps, trainY[-1])
    return np.tile(trainY[-period:], (steps // period + 1))[:steps]


def normalizeDna(matrix):
    m = np.mean(matrix, axis=0)
    s = np.std(matrix, axis=0)
    s = np.where(s > 0, s, 1.0)
    normed = (matrix - m) / s
    n = np.linalg.norm(normed, axis=1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return normed / n


def knnSelect(targetDna, dbDnas, dbExperiences, k):
    sims = dbDnas @ targetDna
    topIdx = np.argsort(sims)[-k:]
    topSims = np.maximum(sims[topIdx], 0)
    wSum = np.sum(topSims)
    if wSum == 0:
        topSims = np.ones(k)
        wSum = k

    candidateScores = {}
    for cKey in dbExperiences[topIdx[0]]['owas']:
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


def extractDnas(series, extractor, period):
    allDnas = []
    featureNames = None
    for idx, (sid, trainY, testY) in enumerate(series):
        features = extractor.extract(trainY, period=max(period, 1))
        if featureNames is None:
            featureNames = sorted(features.keys())
        vec = np.array([features.get(f, 0.0) for f in featureNames])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        allDnas.append(vec)
        if (idx + 1) % 5000 == 0:
            print(f"    DNA: {idx + 1}/{len(series)}")
    return np.array(allDnas), featureNames


def collectExperiences(series, period, horizon):
    experiences = []
    for idx, (sid, trainY, testY) in enumerate(series):
        actual = testY[:horizon]
        nPred = naivePred(trainY, period, len(actual))
        nSmape = computeSmape(actual, nPred)
        nMase = computeMase(actual, nPred, trainY, period)

        candidateOwas = {}
        for modelId in MODELS:
            for preproc in PREPROCESS:
                key = f"{modelId}_{preproc}"
                procTrain, invFunc = applyPreprocess(trainY, preproc)
                pred = fitPredict(modelId, procTrain, period, horizon)
                if pred is None:
                    candidateOwas[key] = 99.0
                    continue
                predOrig = invFunc(pred[:len(actual)])
                sm = computeSmape(actual, predOrig)
                ms = computeMase(actual, predOrig, trainY, period)
                owa = computeOwa(sm, ms, nSmape, nMase)
                candidateOwas[key] = owa

        dotOwa = candidateOwas.get('dot_raw', 99.0)
        experiences.append({'owas': candidateOwas, 'dotOwa': dotOwa})

        if (idx + 1) % 2000 == 0:
            print(f"    경험: {idx + 1}/{len(series)}")

    return experiences


def evaluateKnn(dnaMatrixNorm, experiences, testIdx, trainPool, k=50):
    dbDnas = dnaMatrixNorm[trainPool]
    expArray = np.array(experiences)
    dbExps = expArray[trainPool]

    knnOwas = []
    dotOwas = []
    for i in testIdx:
        bestKey = knnSelect(dnaMatrixNorm[i], dbDnas, dbExps, min(k, len(trainPool)))
        if bestKey is None:
            continue
        knnOwa = experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa'])
        knnOwas.append(knnOwa)
        dotOwas.append(experiences[i]['dotOwa'])

    return np.mean(knnOwas), np.mean(dotOwas)


def runComparison(freqName, nSample, extractor, top50Indices, featureNamesRef):
    configs = {
        'Monthly': {'period': 12, 'horizon': 18},
        'Yearly': {'period': 1, 'horizon': 6},
        'Quarterly': {'period': 4, 'horizon': 8},
    }
    cfg = configs[freqName]
    period = cfg['period']
    horizon = cfg['horizon']

    print(f"\n{'='*70}")
    print(f"  {freqName} (n={nSample}, period={period})")
    print(f"{'='*70}")

    series, _ = loadM4(freqName, nSample)
    print(f"  Loaded: {len(series)} series")

    print(f"  DNA 추출...")
    dnaMatrix, featureNames = extractDnas(series, extractor, period)
    print(f"  DNA: {dnaMatrix.shape[1]}d")

    featureIdxMap = {f: i for i, f in enumerate(featureNames)}
    localTop50Indices = []
    for refIdx in top50Indices:
        refName = featureNamesRef[refIdx]
        if refName in featureIdxMap:
            localTop50Indices.append(featureIdxMap[refName])
    print(f"  Top 50 매칭: {len(localTop50Indices)}/{len(top50Indices)} features")

    print(f"  경험 수집...")
    experiences = collectExperiences(series, period, horizon)

    validMask = np.array([e['dotOwa'] < 99.0 for e in experiences])
    validIdx = np.where(validMask)[0]
    print(f"  유효: {len(validIdx)}")

    avgDot = np.mean([experiences[i]['dotOwa'] for i in validIdx])
    avgOracle = np.mean([min(e['owas'].values()) for i, e in enumerate(experiences) if i in set(validIdx)])

    testSize = min(2000, len(validIdx) // 5)
    np.random.seed(SEED + 1)
    allValid = validIdx.copy()
    np.random.shuffle(allValid)
    testIdx = allValid[:testSize]
    trainPool = allValid[testSize:]

    dna65Norm = normalizeDna(dnaMatrix)

    dna50Matrix = dnaMatrix[:, localTop50Indices]
    dna50Norm = normalizeDna(dna50Matrix)

    K = 50
    results = {}

    print(f"\n  --- 65d vs 50d (DB={len(trainPool)}, k={K}) ---")
    print(f"  {'차원':>6} {'OWA':>8} {'DOT대비':>10}")
    print("  " + "-" * 28)

    knn65, dot65 = evaluateKnn(dna65Norm, experiences, testIdx, trainPool, K)
    impr65 = (dot65 - knn65) / dot65 * 100
    print(f"  {'65d':>6} {knn65:>8.4f} {impr65:>+10.2f}%")
    results['65d'] = {'owa': knn65, 'dot': dot65, 'impr': impr65}

    knn50, dot50 = evaluateKnn(dna50Norm, experiences, testIdx, trainPool, K)
    impr50 = (dot50 - knn50) / dot50 * 100
    diff = (knn65 - knn50) / knn65 * 100
    print(f"  {'50d':>6} {knn50:>8.4f} {impr50:>+10.2f}%  (50d가 65d 대비 {diff:+.2f}%)")
    results['50d'] = {'owa': knn50, 'dot': dot50, 'impr': impr50, 'diff': diff}

    dbSizes = [2000, 5000]
    if freqName == 'Monthly':
        dbSizes.extend([10000, min(15000, len(trainPool))])
    else:
        dbSizes.extend([min(8000, len(trainPool))])
    dbSizes = sorted(set(s for s in dbSizes if s <= len(trainPool)))

    if len(dbSizes) > 1:
        print(f"\n  --- 학습 곡선 65d vs 50d ---")
        print(f"  {'DB크기':>8} {'65d OWA':>10} {'65d 개선':>10} {'50d OWA':>10} {'50d 개선':>10} {'50d-65d':>10}")
        print("  " + "-" * 64)

        lcResults = []
        for dbSize in dbSizes:
            dbIdx = trainPool[:dbSize]

            dbDnas65 = dna65Norm[dbIdx]
            dbDnas50 = dna50Norm[dbIdx]
            expArray = np.array(experiences)
            dbExps = expArray[dbIdx]

            knnOwas65 = []
            knnOwas50 = []
            dotOwasLc = []
            for i in testIdx:
                bestKey65 = knnSelect(dna65Norm[i], dbDnas65, dbExps, min(K, dbSize))
                bestKey50 = knnSelect(dna50Norm[i], dbDnas50, dbExps, min(K, dbSize))
                if bestKey65 is None or bestKey50 is None:
                    continue
                knnOwas65.append(experiences[i]['owas'].get(bestKey65, experiences[i]['dotOwa']))
                knnOwas50.append(experiences[i]['owas'].get(bestKey50, experiences[i]['dotOwa']))
                dotOwasLc.append(experiences[i]['dotOwa'])

            avg65 = np.mean(knnOwas65)
            avg50 = np.mean(knnOwas50)
            avgDotLc = np.mean(dotOwasLc)
            impr65lc = (avgDotLc - avg65) / avgDotLc * 100
            impr50lc = (avgDotLc - avg50) / avgDotLc * 100
            diffLc = (avg65 - avg50) / avg65 * 100
            print(f"  {dbSize:>8} {avg65:>10.4f} {impr65lc:>+10.2f}% {avg50:>10.4f} {impr50lc:>+10.2f}% {diffLc:>+10.2f}%")
            lcResults.append((dbSize, avg65, impr65lc, avg50, impr50lc, diffLc))

        results['lc'] = lcResults

    return {
        'freq': freqName,
        'nSeries': len(series),
        'nValid': len(validIdx),
        'dotOwa': avgDot,
        'results': results,
    }


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    from vectrix.engine.tsfeatures import TSFeatureExtractor

    startTime = time.time()

    print("=" * 70)
    print("experienceIndex/007: Top 50 DNA at Scale")
    print("=" * 70)

    extractor = TSFeatureExtractor()

    print("\n" + "=" * 70)
    print("  Phase A: Top 50 특성 추출 (5K Monthly)")
    print("=" * 70)

    phaseAStart = time.time()
    series5k, _ = loadM4('Monthly', 5000)
    print(f"  Loaded: {len(series5k)} series")

    print(f"  DNA 추출...")
    dnaMatrix5k, featureNames5k = extractDnas(series5k, extractor, 12)
    print(f"  DNA: {dnaMatrix5k.shape[1]}d, {len(featureNames5k)} features")

    print(f"  경험 수집...")
    experiences5k = collectExperiences(series5k, period=12, horizon=18)

    validMask5k = np.array([e['dotOwa'] < 99.0 for e in experiences5k])
    validIdx5k = np.where(validMask5k)[0]
    print(f"  유효: {len(validIdx5k)}")

    testSize5k = 500
    np.random.seed(SEED + 1)
    allValid5k = validIdx5k.copy()
    np.random.shuffle(allValid5k)
    testIdx5k = allValid5k[:testSize5k]
    trainPool5k = allValid5k[testSize5k:]

    dna5kNorm = normalizeDna(dnaMatrix5k)
    baseKnn5k, baseDot5k = evaluateKnn(dna5kNorm, experiences5k, testIdx5k, trainPool5k)

    print(f"\n  Permutation importance 계산 (65 features × 3 perms)...")
    featureImportance = {}
    for fIdx, fName in enumerate(featureNames5k):
        permKnns = []
        for perm in range(3):
            permMatrix = dnaMatrix5k.copy()
            rng = np.random.RandomState(SEED + 200 + perm)
            permMatrix[:, fIdx] = rng.permutation(permMatrix[:, fIdx])
            permNorm = normalizeDna(permMatrix)
            permKnn, _ = evaluateKnn(permNorm, experiences5k, testIdx5k, trainPool5k)
            permKnns.append(permKnn)
        avgPermKnn = np.mean(permKnns)
        degradation = (avgPermKnn - baseKnn5k) / baseKnn5k * 100
        featureImportance[fName] = max(0, degradation)

    sortedFeatures = sorted(featureImportance.items(), key=lambda x: -x[1])
    top50Names = [f for f, _ in sortedFeatures[:50]]
    top50Indices = [featureNames5k.index(f) for f in top50Names]

    print(f"\n  Top 50 추출 완료:")
    for rank, (f, imp) in enumerate(sortedFeatures[:10], 1):
        print(f"    #{rank:>2}: {f:>35} — {imp:.3f}")
    print(f"    ...")
    for rank, (f, imp) in enumerate(sortedFeatures[45:50], 46):
        print(f"    #{rank:>2}: {f:>35} — {imp:.3f}")

    bottom15 = [f for f, _ in sortedFeatures[50:]]
    print(f"\n  제거된 15개: {', '.join(bottom15)}")

    dna50Matrix5k = dnaMatrix5k[:, top50Indices]
    dna50Norm5k = normalizeDna(dna50Matrix5k)
    knn50_5k, dot50_5k = evaluateKnn(dna50Norm5k, experiences5k, testIdx5k, trainPool5k)
    impr65_5k = (baseDot5k - baseKnn5k) / baseDot5k * 100
    impr50_5k = (dot50_5k - knn50_5k) / dot50_5k * 100
    diff5k = (baseKnn5k - knn50_5k) / baseKnn5k * 100

    print(f"\n  5K Monthly 검증:")
    print(f"    65d: OWA {baseKnn5k:.4f} ({impr65_5k:+.2f}% vs DOT)")
    print(f"    50d: OWA {knn50_5k:.4f} ({impr50_5k:+.2f}% vs DOT)")
    print(f"    50d가 65d 대비: {diff5k:+.2f}%")

    phaseATime = time.time() - phaseAStart
    print(f"\n  Phase A 완료: {phaseATime:.0f}s")

    print("\n" + "=" * 70)
    print("  Phase B: 20K Monthly")
    print("=" * 70)

    phaseBStart = time.time()
    monthlyResult = runComparison('Monthly', 20000, extractor, top50Indices, featureNames5k)
    phaseBTime = time.time() - phaseBStart
    print(f"\n  Phase B 완료: {phaseBTime:.0f}s")

    print("\n" + "=" * 70)
    print("  Phase C: Yearly")
    print("=" * 70)

    phaseCStart = time.time()
    yearlyResult = runComparison('Yearly', 10000, extractor, top50Indices, featureNames5k)
    phaseCTime = time.time() - phaseCStart
    print(f"\n  Phase C 완료: {phaseCTime:.0f}s")

    print("\n" + "=" * 70)
    print("  Phase D: Quarterly")
    print("=" * 70)

    phaseDStart = time.time()
    quarterlyResult = runComparison('Quarterly', 10000, extractor, top50Indices, featureNames5k)
    phaseDTime = time.time() - phaseDStart
    print(f"\n  Phase D 완료: {phaseDTime:.0f}s")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)

    print(f"\n  Phase A (5K Monthly):")
    print(f"    65d: OWA {baseKnn5k:.4f} ({impr65_5k:+.2f}%)")
    print(f"    50d: OWA {knn50_5k:.4f} ({impr50_5k:+.2f}%)")
    print(f"    50d가 65d 대비: {diff5k:+.2f}%")

    print(f"\n  {'빈도':>12} {'n':>6} {'DOT':>8} {'65d OWA':>10} {'65d 개선':>10} {'50d OWA':>10} {'50d 개선':>10} {'50d-65d':>10}")
    print("  " + "-" * 82)

    for result in [monthlyResult, yearlyResult, quarterlyResult]:
        r = result['results']
        r65 = r['65d']
        r50 = r['50d']
        print(f"  {result['freq']:>12} {result['nValid']:>6} {r65['dot']:>8.4f} {r65['owa']:>10.4f} {r65['impr']:>+10.2f}% {r50['owa']:>10.4f} {r50['impr']:>+10.2f}% {r50['diff']:>+10.2f}%")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)
