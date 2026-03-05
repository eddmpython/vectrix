"""
실험 ID: experienceIndex/001
실험명: Bucket Consistency — 역인덱스 버킷 내 전략 일관성 검증

========================================================================
배경
========================================================================

dataProfiling 15개 실험의 핵심 실패 원인은 "매핑의 비결정성"이었다.
같은 DNA라도 최적 전략이 다르다 — Ridge든 규칙이든 전역 매핑 함수가 실패했다.

이 실험은 접근을 바꾼다. 매핑 함수를 학습하지 않고,
DNA 특성을 이산화(버킷)하여 역인덱스로 경험을 저장한다.
같은 버킷 안의 경험들이 일관적이면 역인덱스 방식이 가능하다.

검색 엔진의 역인덱스에서 착안:
- 정방향: 시리즈 → DNA 특성 (분석)
- 역방향: DNA 버킷 → 경험 목록 (조회)

========================================================================

목적:
- DNA 특성 버킷 안에서 최적 모델/전처리의 일관성(purity) 측정
- 버킷 해상도(2~5 bins)별 purity 변화 관찰
- 어떤 특성 조합이 가장 높은 purity를 제공하는지 탐색
- Leave-One-Out으로 버킷 경험 기반 블렌딩 vs DOT 비교

가설:
1. 버킷 해상도 3~4에서 purity > 40% (랜덤 25%보다 유의미하게 높음)
2. seasonality + trend 조합이 가장 효과적인 버킷 키
3. 버킷 경험 기반 블렌딩이 DOT 대비 양의 개선 (+1% 이상)

방법:
1. M4 Monthly 1000개, 4모델(DOT, CES, 4Theta, ETS) × 2전처리(raw, log) = 8 후보
2. 각 시리즈에 대해 8후보 전부 실행 → oracle best 기록 = "경험"
3. DNA 특성 4개(seasonality, trend, acf1, series_length_ratio) 이산화
4. 버킷별 purity 계산 (최빈 전략의 비율)
5. Leave-One-Out: 자신 제외 같은 버킷 경험으로 블렌딩 → OWA 계산

결과 (실험 후 작성):
- 유효 경험 1000개, Oracle best 분포: 8후보가 10~16%로 균등 (DOT 16.1% 최다)
- DOT-raw OWA 0.8316, Oracle OWA 0.6604 (+20.6%)
- 최적 버킷: all4 bins=5, purity 0.348
- 버킷 역인덱스 LOO: OWA 0.8302 (-0.00%) — DOT와 동일
- kNN 전부 악화: k=5 -2.84%, k=10 -1.72%, k=20 -1.35%, k=50 -0.31%
- 유사도 가중 kNN: -1.35%, 캡처 -6.6%

결론:
- 가설 1 (purity > 40%) → 기각. 최대 0.348 (35%)
- 가설 2 (seas+trend 최적) → 기각. all4(전체 4특성)가 최적이었으나 여전히 부족
- 가설 3 (+1% 개선) → 기각. 모든 방법 0% 또는 악화
- 근본 원인: 4차원 간이 DNA로는 시리즈를 충분히 구분할 수 없다
- E002에서 65차원 실제 DNA로 반전 확인 → DNA 차원이 결정적

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
N_SAMPLE = 1000

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


def extractDna(y, period):
    n = len(y)
    trend = (y[-1] - y[0]) / (np.std(y) * n) if np.std(y) > 0 else 0

    if n > period:
        seasonal = np.mean([np.abs(y[i] - y[i - period]) for i in range(period, min(n, period * 3))]) / (np.std(y) + 1e-10)
        seasonal = min(seasonal, 2.0)
    else:
        seasonal = 0.0

    if n > 1:
        ym = y - np.mean(y)
        var = np.var(ym)
        acf1 = np.mean(ym[1:] * ym[:-1]) / var if var > 0 else 0
    else:
        acf1 = 0

    lengthRatio = min(n / (period * 10), 2.0)

    return {
        'seasonality': seasonal,
        'trend': abs(trend),
        'acf1': acf1,
        'lengthRatio': lengthRatio,
    }


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


def discretize(value, bins):
    edges = np.linspace(0, 1, bins + 1)
    clipped = np.clip(value, 0, 1)
    for i in range(bins):
        if clipped <= edges[i + 1]:
            return i
    return bins - 1


def makeBucketKey(dna, features, nBins):
    parts = []
    for feat in features:
        val = dna[feat]
        if feat == 'trend':
            val = min(val / 0.5, 1.0)
        elif feat == 'seasonality':
            val = min(val / 2.0, 1.0)
        elif feat == 'acf1':
            val = (val + 1.0) / 2.0
        elif feat == 'lengthRatio':
            val = min(val / 2.0, 1.0)
        parts.append(str(discretize(val, nBins)))
    return '_'.join(parts)


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    startTime = time.time()

    print("=" * 70)
    print("experienceIndex/001: Bucket Consistency")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series")

    print("\n--- Phase 1: 전체 경험 수집 (8후보 × 1000시리즈) ---")

    experiences = []
    for idx, (sid, trainY, testY) in enumerate(series):
        actual = testY[:HORIZON]
        naivePred = naiveSeasonalPred(trainY, PERIOD, len(actual))
        naiveSmape = computeSmape(actual, naivePred)
        naiveMase = computeMase(actual, naivePred, trainY, PERIOD)

        dna = extractDna(trainY, PERIOD)
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
            'dna': dna,
            'owas': candidateOwas,
            'bestKey': bestKey,
            'bestOwa': candidateOwas[bestKey],
            'dotOwa': dotOwa,
        })

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - startTime
            print(f"  {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

    validExp = [e for e in experiences if e['dotOwa'] < 99.0]
    print(f"\n  유효 경험: {len(validExp)}")

    bestDist = {}
    for e in validExp:
        model = e['bestKey']
        bestDist[model] = bestDist.get(model, 0) + 1
    print("\n  Oracle best 분포:")
    for k, v in sorted(bestDist.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v} ({v/len(validExp):.1%})")

    avgDotOwa = np.mean([e['dotOwa'] for e in validExp])
    avgOracleOwa = np.mean([e['bestOwa'] for e in validExp])
    print(f"\n  DOT-raw 평균 OWA: {avgDotOwa:.4f}")
    print(f"  Oracle 평균 OWA:  {avgOracleOwa:.4f} ({(avgDotOwa-avgOracleOwa)/avgDotOwa*100:+.1f}%)")

    print("\n--- Phase 2: 버킷 일관성 분석 ---")

    featureSets = [
        ('seas+trend', ['seasonality', 'trend']),
        ('seas+acf1', ['seasonality', 'acf1']),
        ('seas+trend+acf1', ['seasonality', 'trend', 'acf1']),
        ('all4', ['seasonality', 'trend', 'acf1', 'lengthRatio']),
    ]

    binOptions = [2, 3, 4, 5]

    print(f"\n  {'특성 조합':<20} {'bins':>4} {'버킷수':>6} {'평균크기':>8} {'purity':>8} {'empty':>6}")
    print("  " + "-" * 60)

    bestConfig = None
    bestPurity = 0

    for featName, features in featureSets:
        for nBins in binOptions:
            buckets = {}
            for e in validExp:
                key = makeBucketKey(e['dna'], features, nBins)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(e)

            nonEmpty = {k: v for k, v in buckets.items() if len(v) >= 2}
            if not nonEmpty:
                continue

            purities = []
            for bKey, bExps in nonEmpty.items():
                bestKeys = [e['bestKey'] for e in bExps]
                mostCommon = max(set(bestKeys), key=bestKeys.count)
                purity = bestKeys.count(mostCommon) / len(bestKeys)
                purities.append(purity)

            avgPurity = np.mean(purities)
            avgSize = np.mean([len(v) for v in nonEmpty.values()])
            totalBuckets = nBins ** len(features)
            emptyRate = 1 - len(nonEmpty) / totalBuckets

            print(f"  {featName:<20} {nBins:>4} {totalBuckets:>6} {avgSize:>8.1f} {avgPurity:>8.3f} {emptyRate:>6.1%}")

            if avgPurity > bestPurity and avgSize >= 5:
                bestPurity = avgPurity
                bestConfig = (featName, features, nBins)

    print(f"\n  최적 설정: {bestConfig[0]} bins={bestConfig[2]} (purity={bestPurity:.3f})")

    print("\n--- Phase 3: Leave-One-Out 블렌딩 ---")

    featName, features, nBins = bestConfig

    buckets = {}
    for e in validExp:
        key = makeBucketKey(e['dna'], features, nBins)
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(e)

    looOwas = []
    dotOwas = []
    wins = 0
    nSkipped = 0

    for i, e in enumerate(validExp):
        bKey = makeBucketKey(e['dna'], features, nBins)
        neighbors = [x for x in buckets.get(bKey, []) if x['sid'] != e['sid']]

        if len(neighbors) < 3:
            nSkipped += 1
            continue

        candidateScores = {}
        for cKey in e['owas']:
            if e['owas'][cKey] >= 99.0:
                continue
            neighborOwas = [n['owas'].get(cKey, 99.0) for n in neighbors if n['owas'].get(cKey, 99.0) < 99.0]
            if neighborOwas:
                candidateScores[cKey] = np.mean(neighborOwas)

        if not candidateScores:
            nSkipped += 1
            continue

        bestCandidate = min(candidateScores, key=candidateScores.get)
        looOwa = e['owas'].get(bestCandidate, e['dotOwa'])

        looOwas.append(looOwa)
        dotOwas.append(e['dotOwa'])
        if looOwa < e['dotOwa']:
            wins += 1

    n = len(looOwas)
    avgLoo = np.mean(looOwas)
    avgDot = np.mean(dotOwas)
    improvement = (avgDot - avgLoo) / avgDot * 100

    print(f"\n  평가 대상: {n}개 (스킵: {nSkipped})")
    print(f"  DOT-raw:     OWA {avgDot:.4f}")
    print(f"  버킷 선택:   OWA {avgLoo:.4f} ({improvement:+.2f}%)")
    print(f"  Oracle:      OWA {avgOracleOwa:.4f}")
    print(f"  승률:        {wins}/{n} ({wins/n:.1%})")

    oracleGap = avgDot - avgOracleOwa
    captured = (avgDot - avgLoo) / oracleGap * 100 if oracleGap > 0 else 0
    print(f"  Gap 캡처율:  {captured:.1f}%")

    print("\n--- Phase 4: kNN 비교 (코사인 유사도) ---")

    dnaVectors = np.array([[
        e['dna']['seasonality'] / 2.0,
        e['dna']['trend'] / 0.5,
        (e['dna']['acf1'] + 1) / 2.0,
        e['dna']['lengthRatio'] / 2.0,
    ] for e in validExp])

    norms = np.linalg.norm(dnaVectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    dnaVectorsNorm = dnaVectors / norms

    for k in [5, 10, 20, 50]:
        knnOwas = []
        knnDotOwas = []
        knnWins = 0

        for i, e in enumerate(validExp):
            sims = dnaVectorsNorm @ dnaVectorsNorm[i]
            sims[i] = -1
            topIdx = np.argsort(sims)[-k:]
            neighbors = [validExp[j] for j in topIdx]

            candidateScores = {}
            for cKey in e['owas']:
                if e['owas'][cKey] >= 99.0:
                    continue
                neighborOwas = [n['owas'].get(cKey, 99.0) for n in neighbors if n['owas'].get(cKey, 99.0) < 99.0]
                if neighborOwas:
                    candidateScores[cKey] = np.mean(neighborOwas)

            if not candidateScores:
                continue

            bestCandidate = min(candidateScores, key=candidateScores.get)
            knnOwa = e['owas'].get(bestCandidate, e['dotOwa'])

            knnOwas.append(knnOwa)
            knnDotOwas.append(e['dotOwa'])
            if knnOwa < e['dotOwa']:
                knnWins += 1

        nK = len(knnOwas)
        avgKnn = np.mean(knnOwas)
        avgKnnDot = np.mean(knnDotOwas)
        knnImpr = (avgKnnDot - avgKnn) / avgKnnDot * 100
        knnCapture = (avgKnnDot - avgKnn) / oracleGap * 100 if oracleGap > 0 else 0

        print(f"  kNN(k={k:>2}): OWA {avgKnn:.4f} ({knnImpr:+.2f}%), 승률 {knnWins}/{nK} ({knnWins/nK:.1%}), 캡처 {knnCapture:.1f}%")

    print("\n--- Phase 5: 블렌딩 vs 선택 (kNN 기반) ---")

    bestK = 20
    blendOwas = []
    selectOwas = []
    blendDotOwas = []

    for i, e in enumerate(validExp):
        sims = dnaVectorsNorm @ dnaVectorsNorm[i]
        sims[i] = -1
        topIdx = np.argsort(sims)[-bestK:]
        neighbors = [validExp[j] for j in topIdx]
        neighborSims = sims[topIdx]
        neighborWeights = np.maximum(neighborSims, 0)
        wSum = np.sum(neighborWeights)
        if wSum == 0:
            neighborWeights = np.ones(len(neighbors))
            wSum = len(neighbors)

        candidateBlendOwas = {}
        for cKey in e['owas']:
            if e['owas'][cKey] >= 99.0:
                continue
            weightedSum = 0
            weightTotal = 0
            for j, n in enumerate(neighbors):
                nOwa = n['owas'].get(cKey, 99.0)
                if nOwa < 99.0:
                    weightedSum += neighborWeights[j] * nOwa
                    weightTotal += neighborWeights[j]
            if weightTotal > 0:
                candidateBlendOwas[cKey] = weightedSum / weightTotal

        if not candidateBlendOwas:
            continue

        bestByBlend = min(candidateBlendOwas, key=candidateBlendOwas.get)
        blendOwa = e['owas'].get(bestByBlend, e['dotOwa'])

        modelOwaMeans = {}
        for cKey, score in candidateBlendOwas.items():
            modelId = cKey.split('_')[0] if '_raw' in cKey or '_log' in cKey else cKey
            if modelId not in modelOwaMeans:
                modelOwaMeans[modelId] = []
            modelOwaMeans[modelId].append(score)

        blendOwas.append(blendOwa)
        blendDotOwas.append(e['dotOwa'])

    avgBlend = np.mean(blendOwas)
    avgBlendDot = np.mean(blendDotOwas)
    blendImpr = (avgBlendDot - avgBlend) / avgBlendDot * 100
    blendCapture = (avgBlendDot - avgBlend) / oracleGap * 100 if oracleGap > 0 else 0
    blendWins = sum(1 for b, d in zip(blendOwas, blendDotOwas) if b < d)

    print(f"\n  유사도 가중 kNN(k={bestK}): OWA {avgBlend:.4f} ({blendImpr:+.2f}%)")
    print(f"  승률: {blendWins}/{len(blendOwas)} ({blendWins/len(blendOwas):.1%})")
    print(f"  Gap 캡처율: {blendCapture:.1f}%")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  DOT-raw 기준선:     OWA {avgDotOwa:.4f}")
    print(f"  Oracle:             OWA {avgOracleOwa:.4f} ({(avgDotOwa-avgOracleOwa)/avgDotOwa*100:+.1f}%)")
    print(f"  버킷 역인덱스:      OWA {avgLoo:.4f} ({improvement:+.2f}%)")
    print(f"  kNN 가중 선택:      OWA {avgBlend:.4f} ({blendImpr:+.2f}%)")
    print(f"  최적 버킷 purity:   {bestPurity:.3f}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
