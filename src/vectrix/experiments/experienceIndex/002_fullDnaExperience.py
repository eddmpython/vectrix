"""
실험 ID: experienceIndex/002
실험명: Full DNA Experience — 실제 DNA 65차원 + 대규모 경험 DB

========================================================================
배경
========================================================================

E001에서 간이 DNA 4차원으로 버킷/kNN을 시도했으나:
- 버킷 purity 0.348 (8후보 랜덤 0.125보다는 높지만 불충분)
- kNN 전부 악화 (-0.3% ~ -2.8%)
- 버킷 역인덱스 ±0.00% (해도 안해도 같음)

하지만 두 가지가 부실했다:
1. DNA가 직접 짠 4차원 간이 버전이었다 (실제 TSFeatureExtractor는 65차원)
2. 경험 DB가 1000개뿐이었다 (M4 Monthly만 해도 48K개)

이 실험은 제대로 한다:
- Vectrix 엔진의 TSFeatureExtractor로 실제 65차원 DNA 추출
- M4 Monthly 5000개 규모로 경험 DB 구축
- 경험 DNA = 통계 DNA(65d) + 모델별 OWA(8d) = 73차원 벡터
- Leave-One-Out으로 검증

핵심 질문:
- 65차원 DNA가 4차원보다 kNN 검색에서 나은가?
- 경험 DB가 커지면 정확도가 올라가는가? (학습 곡선)
- "통계 DNA만으로 검색" vs "경험 DNA로 검색" 차이

========================================================================

목적:
- 실제 DNA 65차원으로 kNN 경험 검색의 성능 확인
- 경험 DB 크기별 정확도 (500, 1000, 2000, 5000) 학습 곡선
- 통계 DNA 검색 → 경험 기반 블렌딩의 OWA 개선 여부

가설:
1. 65차원 DNA kNN이 4차원보다 OWA 개선폭이 큼
2. 경험 DB가 커질수록 단조 개선 (학습 곡선 우상향)
3. 경험 기반 블렌딩이 DOT 대비 +2% 이상 개선

방법:
1. M4 Monthly 5000개 로드
2. TSFeatureExtractor로 65차원 DNA 추출
3. 4모델 × 2전처리 = 8후보 전부 실행 → 경험 기록
4. Leave-One-Out kNN(k=10,20,50) 블렌딩
5. DB 크기별 학습 곡선 (500→1000→2000→5000)

결과 (실험 후 작성):
- DNA 차원: 65 (TSFeatureExtractor), 유효 경험: 5000
- DOT-raw 기준: OWA 0.8599, Oracle: 0.6728 (+21.8%)

- kNN Leave-One-Out (전체 DB):
  | k  | OWA    | DOT대비  | 승률  | 캡처율 |
  |----|--------|----------|-------|--------|
  | 5  | 0.8432 | +1.95%   | 39.7% | 9.0%   |
  | 10 | 0.8346 | +2.95%   | 39.2% | 13.5%  |
  | 20 | 0.8301 | +3.47%   | 38.0% | 16.0%  |
  | 50 | 0.8279 | +3.72%   | 36.6% | 17.1%  |

- 학습 곡선 (DB 크기별, 테스트 500개):
  | DB크기 | OWA    | DOT대비  | 캡처율 |
  |--------|--------|----------|--------|
  | 500    | 0.8549 | +3.07%   | 14.5%  |
  | 1000   | 0.8532 | +3.26%   | 15.4%  |
  | 2000   | 0.8458 | +4.09%   | 19.3%  |
  | 3000   | 0.8428 | +4.44%   | 20.9%  |
  | 4500   | 0.8410 | +4.64%   | 21.9%  |

- 경험 선택: +4.64%, 경험 블렌딩: -10.63% (블렌딩 실패, 선택만 유효)

결론:
- 가설 1 (65차원 > 4차원) → 확인! E001에서 전부 악화(-2.8%)했던 것이 전부 개선(+3.7%)
- 가설 2 (DB 커지면 개선) → 확인! 500개 +3.1% → 4500개 +4.6% 단조 증가
- 가설 3 (+2% 이상) → 초과 달성! kNN(k=50) +3.72%, DB학습곡선 +4.64%
- 핵심 발견: DNA 차원이 결정적. 4차원으로는 시리즈를 구분 못 하지만 65차원은 가능
- 경험 기반 "선택"은 작동하지만 "블렌딩"은 역효과 (나쁜 모델에 가중치가 분산됨)
- 학습 곡선 미포화 — M4 48K 전체로 하면 추가 개선 기대
- dataProfiling 15개 실험 최고(+0.96%)의 4배 이상 달성

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
N_SAMPLE = 5000

MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']
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


def knnBlend(targetDna, dbDnas, dbExperiences, k, excludeIdx=-1):
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
        return None, None

    bestKey = min(candidateScores, key=candidateScores.get)
    return bestKey, candidateScores


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from vectrix.engine.tsfeatures import TSFeatureExtractor

    startTime = time.time()

    print("=" * 70)
    print("experienceIndex/002: Full DNA Experience")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series")

    print("\n--- Phase 1: DNA 추출 (TSFeatureExtractor 65차원) ---")
    extractor = TSFeatureExtractor()
    allDnas = []
    featureNames = None

    for idx, (sid, trainY, testY) in enumerate(series):
        features = extractor.extract(trainY, period=PERIOD)
        if featureNames is None:
            featureNames = sorted(features.keys())
        vec = np.array([features.get(f, 0.0) for f in featureNames])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        allDnas.append(vec)

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - startTime
            print(f"  DNA: {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

    dnaMatrix = np.array(allDnas)
    print(f"  DNA 차원: {dnaMatrix.shape[1]}")

    means = np.mean(dnaMatrix, axis=0)
    stds = np.std(dnaMatrix, axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    dnaMatrixNorm = (dnaMatrix - means) / stds

    norms = np.linalg.norm(dnaMatrixNorm, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    dnaMatrixNorm = dnaMatrixNorm / norms

    print(f"  정규화 완료: z-score → L2 norm")

    print("\n--- Phase 2: 경험 수집 (8후보 × 5000시리즈) ---")

    experiences = []
    for idx, (sid, trainY, testY) in enumerate(series):
        actual = testY[:HORIZON]
        naivePred = naiveSeasonalPred(trainY, PERIOD, len(actual))
        naiveSmape = computeSmape(actual, naivePred)
        naiveMase = computeMase(actual, naivePred, trainY, PERIOD)

        candidateOwas = {}
        for modelId in MODELS:
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

        bestKey = min(candidateOwas, key=candidateOwas.get)
        dotOwa = candidateOwas.get('dot_raw', 99.0)

        experiences.append({
            'sid': sid,
            'owas': candidateOwas,
            'bestKey': bestKey,
            'bestOwa': candidateOwas[bestKey],
            'dotOwa': dotOwa,
        })

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - startTime
            print(f"  경험: {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

    validMask = np.array([e['dotOwa'] < 99.0 for e in experiences])
    validIdx = np.where(validMask)[0]
    print(f"\n  유효 경험: {len(validIdx)}")

    avgDotOwa = np.mean([experiences[i]['dotOwa'] for i in validIdx])
    avgOracleOwa = np.mean([experiences[i]['bestOwa'] for i in validIdx])
    oracleGap = avgDotOwa - avgOracleOwa
    print(f"  DOT-raw:  OWA {avgDotOwa:.4f}")
    print(f"  Oracle:   OWA {avgOracleOwa:.4f} ({oracleGap/avgDotOwa*100:+.1f}%)")

    print("\n--- Phase 3: kNN Leave-One-Out (전체 DB) ---")

    expArray = np.array(experiences)

    for k in [5, 10, 20, 50]:
        knnOwas = []
        dotOwas = []
        wins = 0

        for i in validIdx:
            result = knnBlend(dnaMatrixNorm[i], dnaMatrixNorm[validIdx], expArray[validIdx], k, excludeIdx=np.where(validIdx == i)[0][0])
            if result[0] is None:
                continue
            bestKey, _ = result
            knnOwa = experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa'])
            knnOwas.append(knnOwa)
            dotOwas.append(experiences[i]['dotOwa'])
            if knnOwa < experiences[i]['dotOwa']:
                wins += 1

        avgKnn = np.mean(knnOwas)
        avgDot = np.mean(dotOwas)
        impr = (avgDot - avgKnn) / avgDot * 100
        capture = (avgDot - avgKnn) / oracleGap * 100 if oracleGap > 0 else 0
        n = len(knnOwas)
        print(f"  kNN(k={k:>2}): OWA {avgKnn:.4f} ({impr:+.2f}%), 승률 {wins}/{n} ({wins/n:.1%}), 캡처 {capture:.1f}%")

    print("\n--- Phase 4: 학습 곡선 (DB 크기별) ---")

    bestK = 20
    dbSizes = [500, 1000, 2000, 3000, len(validIdx)]
    testSize = 500

    np.random.seed(SEED + 1)
    allValidIdx = validIdx.copy()
    np.random.shuffle(allValidIdx)
    testIdx = allValidIdx[:testSize]
    trainPool = allValidIdx[testSize:]

    print(f"\n  테스트셋: {testSize}개, 학습 풀: {len(trainPool)}개")
    print(f"\n  {'DB크기':>8} {'OWA':>8} {'DOT대비':>10} {'승률':>10} {'캡처율':>8}")
    print("  " + "-" * 50)

    for dbSize in dbSizes:
        if dbSize > len(trainPool):
            dbSize = len(trainPool)
        dbIdx = trainPool[:dbSize]

        dbDnas = dnaMatrixNorm[dbIdx]
        dbExps = expArray[dbIdx]

        knnOwas = []
        dotOwas = []
        wins = 0

        for i in testIdx:
            result = knnBlend(dnaMatrixNorm[i], dbDnas, dbExps, min(bestK, dbSize))
            if result[0] is None:
                continue
            bestKey, _ = result
            knnOwa = experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa'])
            knnOwas.append(knnOwa)
            dotOwas.append(experiences[i]['dotOwa'])
            if knnOwa < experiences[i]['dotOwa']:
                wins += 1

        avgKnn = np.mean(knnOwas)
        avgDot = np.mean(dotOwas)
        impr = (avgDot - avgKnn) / avgDot * 100
        capture = (avgDot - avgKnn) / oracleGap * 100 if oracleGap > 0 else 0
        n = len(knnOwas)
        print(f"  {dbSize:>8} {avgKnn:>8.4f} {impr:>+10.2f}% {wins}/{n} ({wins/n:.1%}) {capture:>+8.1f}%")

    print("\n--- Phase 5: 모델 선택 vs 가중 블렌딩 ---")

    dbIdx = trainPool
    dbDnas = dnaMatrixNorm[dbIdx]
    dbExps = expArray[dbIdx]

    selectOwas = []
    blendOwas = []
    dotOwasTest = []

    for i in testIdx:
        sims = dbDnas @ dnaMatrixNorm[i]
        topIdx = np.argsort(sims)[-bestK:]
        topSims = np.maximum(sims[topIdx], 0)
        wSum = np.sum(topSims)
        if wSum == 0:
            continue

        neighborExps = dbExps[topIdx]

        candidateOwas = {}
        for cKey in experiences[i]['owas']:
            if experiences[i]['owas'][cKey] >= 99.0:
                continue
            wOwa = 0
            wTotal = 0
            for j, ne in enumerate(neighborExps):
                nOwa = ne['owas'].get(cKey, 99.0)
                if nOwa < 99.0:
                    wOwa += topSims[j] * nOwa
                    wTotal += topSims[j]
            if wTotal > 0:
                candidateOwas[cKey] = wOwa / wTotal

        if not candidateOwas:
            continue

        selectKey = min(candidateOwas, key=candidateOwas.get)
        selectOwa = experiences[i]['owas'].get(selectKey, 99.0)

        modelWeights = {}
        for cKey, score in candidateOwas.items():
            invScore = 1.0 / (score + 0.01)
            modelWeights[cKey] = invScore
        wTotal = sum(modelWeights.values())
        for cKey in modelWeights:
            modelWeights[cKey] /= wTotal

        blendOwa = sum(experiences[i]['owas'].get(cKey, 99.0) * w for cKey, w in modelWeights.items() if experiences[i]['owas'].get(cKey, 99.0) < 99.0)

        selectOwas.append(selectOwa)
        blendOwas.append(blendOwa)
        dotOwasTest.append(experiences[i]['dotOwa'])

    avgSelect = np.mean(selectOwas)
    avgBlend = np.mean(blendOwas)
    avgDotT = np.mean(dotOwasTest)

    selectImpr = (avgDotT - avgSelect) / avgDotT * 100
    blendImpr = (avgDotT - avgBlend) / avgDotT * 100

    print(f"\n  DOT-raw:       OWA {avgDotT:.4f}")
    print(f"  경험 선택:     OWA {avgSelect:.4f} ({selectImpr:+.2f}%)")
    print(f"  경험 블렌딩:   OWA {avgBlend:.4f} ({blendImpr:+.2f}%)")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  DNA 차원:      {dnaMatrix.shape[1]}")
    print(f"  경험 DB:       {len(validIdx)}개")
    print(f"  DOT-raw:       OWA {avgDotOwa:.4f}")
    print(f"  Oracle:        OWA {avgOracleOwa:.4f} ({oracleGap/avgDotOwa*100:+.1f}%)")
    print(f"  경험 선택:     OWA {avgSelect:.4f} ({selectImpr:+.2f}%)")
    print(f"  경험 블렌딩:   OWA {avgBlend:.4f} ({blendImpr:+.2f}%)")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
