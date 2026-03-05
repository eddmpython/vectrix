"""
실험 ID: experienceIndex/004
실험명: Cross-Frequency — Yearly/Quarterly 경험 검증

========================================================================
배경
========================================================================

E002~E003에서 Monthly(12) 경험 인덱스가 +3.7~4.0% 개선을 보였다.
그러나 Monthly만 검증했다. 다른 빈도에서도 작동하는지 확인해야 한다.

M4 빈도별 특성이 매우 다르다:
- Yearly(period=1): 계절성 없음, 트렌드 지배적
- Quarterly(period=4): 약한 계절성
- Monthly(period=12): 강한 계절성 ← 검증 완료

========================================================================

목적:
- Yearly, Quarterly에서 경험 인덱스 유효성 검증
- 빈도별 학습 곡선 비교
- 빈도별 최적 k값 탐색
- 계절성 유무가 경험 인덱스 성능에 미치는 영향

가설:
1. Quarterly는 Monthly와 유사한 개선 (+3%+)
2. Yearly는 계절성 없어서 개선폭이 작음 (+1~2%)
3. 학습 곡선은 모든 빈도에서 단조 증가

방법:
1. M4 Yearly 10K, Quarterly 10K 로드
2. 각각 TSFeatureExtractor로 65d DNA 추출
3. 8후보 실행 → 경험 DB 구축
4. kNN(k=10,20,50) 비교
5. 학습 곡선 (1K→3K→5K→8K)

결과 (실험 후 작성):
- Yearly: 7768개, DOT 0.9347, Oracle 0.5738 (gap 38.6%)
  kNN(k=50): OWA 0.8684 (+7.18%, 캡처 18.6%)
  학습곡선: 1K +5.73% → 5K +7.58% → 6.8K +7.18%

- Quarterly: 9999개, DOT 0.8386, Oracle 0.6206 (gap 26.0%)
  kNN(k=50): OWA 0.8067 (+2.82%, 캡처 10.7%)
  학습곡선: 1K +1.66% → 5K +1.99% → 8K +2.73%
  k=20이 +3.22%로 k=50보다 좋음

- 빈도별 비교 (kNN 선택, 최대 DB):
  | 빈도      | DOT    | kNN    | 개선   | 캡처  |
  |-----------|--------|--------|--------|-------|
  | Yearly    | 0.9347 | 0.8684 | +7.18% | 18.6% |
  | Monthly   | 0.8638 | 0.8308 | +3.98% | 18.1% |
  | Quarterly | 0.8386 | 0.8067 | +2.82% | 10.7% |

결론:
- 가설 1 (Quarterly +3%+) → 기각. +2.82~3.22%로 근접하지만 미달
- 가설 2 (Yearly 작은 개선) → 완전 기각! Yearly가 +7.18%로 가장 큰 개선
- 가설 3 (모든 빈도 단조 증가) → 대체로 확인. 일부 비선형 변동 있음
- 핵심: DOT 기준 OWA가 높을수록(나쁠수록) 경험 선택 개선이 크다
- Yearly는 계절성 없어서 모델 선택이 더 중요 (트렌드 유형에 민감)
- k값은 빈도별로 다를 수 있음: Monthly/Yearly는 k=50, Quarterly는 k=20~50
- 경험 인덱스는 3개 빈도 모두에서 유효

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
MIN_LEN_MULT = 4

MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']
PREPROCESS = ['raw', 'log']
CANDIDATES = [f"{m}_{p}" for m in MODELS for p in PREPROCESS]

FREQ_CONFIG = {
    'Yearly': {'prefix': 'Y', 'period': 1, 'horizon': 6, 'minLen': 20, 'nSample': 10000,
               'trainFile': 'Yearly-train.csv', 'testFile': 'Yearly-test.csv'},
    'Quarterly': {'prefix': 'Q', 'period': 4, 'horizon': 8, 'minLen': 20, 'nSample': 10000,
                  'trainFile': 'Quarterly-train.csv', 'testFile': 'Quarterly-test.csv'},
}


def loadM4(freqName):
    cfg = FREQ_CONFIG[freqName]
    trainDf = pd.read_csv(os.path.join(M4_DIR, cfg['trainFile']))
    testDf = pd.read_csv(os.path.join(M4_DIR, cfg['testFile']))

    np.random.seed(SEED)
    allIds = trainDf['V1'].values
    selectedIds = set(np.random.choice(allIds, size=min(cfg['nSample'], len(allIds)), replace=False))

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


def knnSelect(targetDna, dbDnas, dbExperiences, k, excludeIdx=-1):
    sims = dbDnas @ targetDna
    if excludeIdx >= 0:
        sims[excludeIdx] = -999

    topIdx = np.argsort(sims)[-k:]
    topSims = sims[topIdx]
    topSims = np.maximum(topSims, 0)
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


def runFrequency(freqName, extractor):
    cfg = FREQ_CONFIG[freqName]
    period = cfg['period']
    horizon = cfg['horizon']

    print(f"\n{'='*70}")
    print(f"  {freqName} (period={period}, horizon={horizon})")
    print(f"{'='*70}")

    series, _ = loadM4(freqName)
    print(f"  Loaded: {len(series)} series")

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

    dnaMatrix = np.array(allDnas)
    means = np.mean(dnaMatrix, axis=0)
    stds = np.std(dnaMatrix, axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    dnaMatrixNorm = (dnaMatrix - means) / stds
    norms = np.linalg.norm(dnaMatrixNorm, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    dnaMatrixNorm = dnaMatrixNorm / norms
    print(f"  DNA: {dnaMatrix.shape[1]}d, 정규화 완료")

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

        bestKey = min(candidateOwas, key=candidateOwas.get)
        dotOwa = candidateOwas.get('dot_raw', 99.0)
        experiences.append({
            'owas': candidateOwas,
            'bestKey': bestKey,
            'bestOwa': candidateOwas[bestKey],
            'dotOwa': dotOwa,
        })

        if (idx + 1) % 2000 == 0:
            print(f"    경험: {idx + 1}/{len(series)}")

    validMask = np.array([e['dotOwa'] < 99.0 for e in experiences])
    validIdx = np.where(validMask)[0]
    expArray = np.array(experiences)

    avgDot = np.mean([experiences[i]['dotOwa'] for i in validIdx])
    avgOracle = np.mean([experiences[i]['bestOwa'] for i in validIdx])
    oracleGap = avgDot - avgOracle
    print(f"\n  유효: {len(validIdx)}, DOT: {avgDot:.4f}, Oracle: {avgOracle:.4f} (gap {oracleGap/avgDot*100:.1f}%)")

    testSize = min(1000, len(validIdx) // 5)
    np.random.seed(SEED + 1)
    allValid = validIdx.copy()
    np.random.shuffle(allValid)
    testIdx = allValid[:testSize]
    trainPool = allValid[testSize:]

    print(f"\n  --- kNN k값 비교 (DB={len(trainPool)}) ---")
    print(f"  {'k':>4} {'OWA':>8} {'DOT대비':>10} {'승률':>12} {'캡처':>8}")
    print("  " + "-" * 46)

    kResults = {}
    for k in [10, 20, 50]:
        dbDnas = dnaMatrixNorm[trainPool]
        dbExps = expArray[trainPool]
        knnOwas = []
        dotOwas = []
        wins = 0
        for i in testIdx:
            bestKey = knnSelect(dnaMatrixNorm[i], dbDnas, dbExps, min(k, len(trainPool)))
            if bestKey is None:
                continue
            knnOwa = experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa'])
            knnOwas.append(knnOwa)
            dotOwas.append(experiences[i]['dotOwa'])
            if knnOwa < experiences[i]['dotOwa']:
                wins += 1

        avgKnn = np.mean(knnOwas)
        avgDotT = np.mean(dotOwas)
        impr = (avgDotT - avgKnn) / avgDotT * 100
        capture = (avgDotT - avgKnn) / oracleGap * 100 if oracleGap > 0 else 0
        n = len(knnOwas)
        print(f"  {k:>4} {avgKnn:>8.4f} {impr:>+10.2f}% {wins}/{n} ({wins/n:.1%}) {capture:>+8.1f}%")
        kResults[k] = {'owa': avgKnn, 'impr': impr, 'capture': capture}

    dbSizes = [1000, 3000, 5000, min(8000, len(trainPool))]
    dbSizes = [s for s in dbSizes if s <= len(trainPool)]
    bestK = 50

    print(f"\n  --- 학습 곡선 (k={bestK}) ---")
    print(f"  {'DB크기':>8} {'OWA':>8} {'DOT대비':>10} {'캡처':>8}")
    print("  " + "-" * 38)

    lcResults = []
    for dbSize in dbSizes:
        dbIdx = trainPool[:dbSize]
        dbDnas = dnaMatrixNorm[dbIdx]
        dbExps = expArray[dbIdx]
        knnOwas = []
        dotOwas = []
        for i in testIdx:
            bestKey = knnSelect(dnaMatrixNorm[i], dbDnas, dbExps, min(bestK, dbSize))
            if bestKey is None:
                continue
            knnOwa = experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa'])
            knnOwas.append(knnOwa)
            dotOwas.append(experiences[i]['dotOwa'])

        avgKnn = np.mean(knnOwas)
        avgDotT = np.mean(dotOwas)
        impr = (avgDotT - avgKnn) / avgDotT * 100
        capture = (avgDotT - avgKnn) / oracleGap * 100 if oracleGap > 0 else 0
        print(f"  {dbSize:>8} {avgKnn:>8.4f} {impr:>+10.2f}% {capture:>+8.1f}%")
        lcResults.append((dbSize, avgKnn, impr, capture))

    return {
        'freq': freqName,
        'nSeries': len(series),
        'nValid': len(validIdx),
        'dotOwa': avgDot,
        'oracleOwa': avgOracle,
        'kResults': kResults,
        'lcResults': lcResults,
    }


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    from vectrix.engine.tsfeatures import TSFeatureExtractor

    startTime = time.time()

    print("=" * 70)
    print("experienceIndex/004: Cross-Frequency Validation")
    print("=" * 70)

    extractor = TSFeatureExtractor()
    results = {}

    for freqName in ['Yearly', 'Quarterly']:
        freqStart = time.time()
        result = runFrequency(freqName, extractor)
        results[freqName] = result
        freqTime = time.time() - freqStart
        print(f"\n  {freqName} 완료: {freqTime:.0f}s")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== CROSS-FREQUENCY SUMMARY ===")
    print("=" * 70)

    print(f"\n  {'빈도':>12} {'n':>6} {'DOT':>8} {'Oracle':>8} {'kNN(50)':>10} {'개선':>8} {'캡처':>8}")
    print("  " + "-" * 65)

    for freqName in ['Yearly', 'Quarterly']:
        r = results[freqName]
        k50 = r['kResults'].get(50, {})
        print(f"  {freqName:>12} {r['nValid']:>6} {r['dotOwa']:>8.4f} {r['oracleOwa']:>8.4f} {k50.get('owa', 0):>10.4f} {k50.get('impr', 0):>+8.2f}% {k50.get('capture', 0):>+8.1f}%")

    print(f"  {'Monthly*':>12} {'~20K':>6} {'0.8638':>8} {'0.6736':>8} {'0.8308':>10} {'  +3.98':>8s}% {'  +18.1':>8s}%")
    print(f"\n  * Monthly는 E003 결과 참조")

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)
