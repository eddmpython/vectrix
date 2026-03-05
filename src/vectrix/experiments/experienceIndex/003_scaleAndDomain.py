"""
실험 ID: experienceIndex/003
실험명: Scale & Domain — 20K 스케일 + 도메인 분석

========================================================================
배경
========================================================================

E002에서 5000개 Monthly로 kNN +3.72~4.64% 달성, 학습 곡선 미포화.
두 가지 핵심 질문이 남아 있다:

1. E002의 4500개 이후 학습 곡선이 어디서 포화되는가?
2. 도메인(Micro/Finance/Macro/Industry/Demographic/Other)별로
   경험을 분리하면 혼합보다 나은가?

M4-info.csv에 6개 도메인 라벨이 존재한다.
20K 규모로 학습 곡선 포화 여부와 도메인 분석을 동시에 수행한다.

========================================================================

목적:
- M4 Monthly 20K로 학습 곡선 포화점 확인 (E002의 4500 이후)
- 도메인별 kNN 이웃 순도 측정 (같은 도메인 이웃 비율)
- 도메인 내 경험 vs 전체 경험 비교
- 도메인별 성능 차이 분석

가설:
1. 학습 곡선은 10K~15K에서 기울기가 감소하기 시작
2. kNN(k=50) 이웃 중 같은 도메인 비율이 50%+ (DNA가 도메인을 자연 분리)
3. 도메인 내 경험만 사용하면 전체보다 +1% 이상 개선 (도메인 노이즈 제거)

방법:
1. M4 Monthly 20K 로드 + M4-info.csv에서 도메인 라벨 매칭
2. TSFeatureExtractor로 65d DNA 추출
3. 8후보(4모델×2전처리) 전체 실행
4. Phase 1: 학습 곡선 (2K→5K→10K→15K)
5. Phase 2: 도메인 순도 분석 (kNN 이웃의 도메인 분포)
6. Phase 3: 도메인 내 경험 vs 전체 경험 비교
7. Phase 4: 도메인별 OWA 분석

결과 (실험 후 작성):
- 20K 시리즈, 유효 19993, DOT-raw OWA 0.8638, Oracle 0.6736 (gap 22.0%)

- 학습 곡선 (k=50, 테스트 2000개):
  | DB크기 | OWA    | DOT대비  | 캡처율 |
  |--------|--------|----------|--------|
  | 2,000  | 0.8417 | +2.73%   | 12.4%  |
  | 5,000  | 0.8483 | +1.96%   | 8.9%   |
  | 10,000 | 0.8347 | +3.54%   | 16.1%  |
  | 15,000 | 0.8308 | +3.98%   | 18.1%  |

- 도메인 순도: 40.6% (랜덤 22.8%, 1.78x). DNA가 자연 분리하지만 불완전

- 도메인 내 경험 vs 전체: 평균 -0.65% (도메인 내가 오히려 나쁨)
  Industry -2.97%, Finance -0.36%, Micro -0.16%, Macro +0.15%, Demographic +0.12%

- 도메인별 kNN 성능:
  Industry +8.0% (캡처 31.9%), Macro +5.7% (29.2%), Micro +5.3% (27.0%)
  Finance +3.8% (16.9%), Demographic +1.2% (8.0%)

결론:
- 가설 1 (포화) → 기각! 15K에서도 미포화, 2K→15K로 +2.7%→+4.0% 증가 지속
- 가설 2 (DNA 자연 분리 50%+) → 기각. 40.6%로 50% 미달. 랜덤 대비 1.78x
- 가설 3 (도메인 분리 +1%) → 기각! 오히려 -0.65%. DB 크기 > 도메인 순도
- 핵심 발견: 도메인 분리는 불필요. 전체 DB가 클수록 좋다
- DOT이 약한 도메인(Industry 0.93, Macro 0.86)에서 경험 선택 효과가 크다
- 학습 곡선 5K→10K에서 비선형 점프 — 임계 규모 존재 가능성

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

MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']
PREPROCESS = ['raw', 'log']
CANDIDATES = [f"{m}_{p}" for m in MODELS for p in PREPROCESS]
N_CANDIDATES = len(CANDIDATES)


N_SAMPLE = 20000


def loadM4Monthly(n=N_SAMPLE):
    trainDf = pd.read_csv(os.path.join(M4_DIR, 'Monthly-train.csv'))
    testDf = pd.read_csv(os.path.join(M4_DIR, 'Monthly-test.csv'))
    infoDf = pd.read_csv(os.path.join(M4_DIR, 'M4-info.csv'))

    monthlyInfo = infoDf[infoDf['M4id'].str.startswith('M')].set_index('M4id')

    np.random.seed(SEED)
    allIds = trainDf['V1'].values
    selectedIds = set(np.random.choice(allIds, size=min(n, len(allIds)), replace=False))

    series = []
    for _, row in trainDf.iterrows():
        sid = row['V1']
        if sid not in selectedIds:
            continue
        trainY = row.iloc[1:].dropna().values.astype(float)
        if len(trainY) < MIN_LEN:
            continue
        testRow = testDf[testDf['V1'] == sid]
        if testRow.empty:
            continue
        testY = testRow.iloc[0, 1:].dropna().values.astype(float)
        domain = monthlyInfo.loc[sid, 'category'] if sid in monthlyInfo.index else 'Other'
        series.append((sid, trainY, testY, domain))

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
        return None, None, None

    bestKey = min(candidateScores, key=candidateScores.get)
    return bestKey, candidateScores, topIdx


def analyzeDomainPurity(targetDomain, neighborDomains):
    if len(neighborDomains) == 0:
        return 0.0
    return np.mean([1.0 if d == targetDomain else 0.0 for d in neighborDomains])


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    from vectrix.engine.tsfeatures import TSFeatureExtractor

    startTime = time.time()

    print("=" * 70)
    print("experienceIndex/003: Scale & Domain")
    print("=" * 70)

    print(f"\n--- Loading M4 Monthly ({N_SAMPLE}개 샘플) ---")
    series = loadM4Monthly()
    print(f"  Loaded: {len(series)} series")

    domains = [s[3] for s in series]
    domainCounts = {}
    for d in domains:
        domainCounts[d] = domainCounts.get(d, 0) + 1
    print(f"  Domain distribution:")
    for d, c in sorted(domainCounts.items(), key=lambda x: -x[1]):
        print(f"    {d:15s}: {c:>6d} ({c/len(series)*100:.1f}%)")

    print(f"\n--- Phase 0: DNA 추출 ({len(series)}개) ---")
    extractor = TSFeatureExtractor()
    allDnas = []
    featureNames = None

    for idx, (sid, trainY, testY, dom) in enumerate(series):
        features = extractor.extract(trainY, period=PERIOD)
        if featureNames is None:
            featureNames = sorted(features.keys())
        vec = np.array([features.get(f, 0.0) for f in featureNames])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        allDnas.append(vec)

        if (idx + 1) % 5000 == 0:
            elapsed = time.time() - startTime
            print(f"  DNA: {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

    dnaMatrix = np.array(allDnas)
    print(f"  DNA 차원: {dnaMatrix.shape[1]}, 시리즈: {dnaMatrix.shape[0]}")

    means = np.mean(dnaMatrix, axis=0)
    stds = np.std(dnaMatrix, axis=0)
    stds = np.where(stds > 0, stds, 1.0)
    dnaMatrixNorm = (dnaMatrix - means) / stds

    norms = np.linalg.norm(dnaMatrixNorm, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    dnaMatrixNorm = dnaMatrixNorm / norms

    print(f"  정규화 완료: z-score → L2 norm")

    print(f"\n--- Phase 0.5: 경험 수집 ({len(series)}개 × {N_CANDIDATES}후보) ---")

    experiences = []
    domainList = []
    for idx, (sid, trainY, testY, dom) in enumerate(series):
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
            'domain': dom,
        })
        domainList.append(dom)

        if (idx + 1) % 2000 == 0:
            elapsed = time.time() - startTime
            validSoFar = sum(1 for e in experiences if e['dotOwa'] < 99.0)
            print(f"  경험: {idx + 1}/{len(series)}... 유효 {validSoFar} ({elapsed:.0f}s)")

    validMask = np.array([e['dotOwa'] < 99.0 for e in experiences])
    validIdx = np.where(validMask)[0]
    domainArr = np.array(domainList)
    expArray = np.array(experiences)

    avgDotOwa = np.mean([experiences[i]['dotOwa'] for i in validIdx])
    avgOracleOwa = np.mean([experiences[i]['bestOwa'] for i in validIdx])
    oracleGap = avgDotOwa - avgOracleOwa
    print(f"\n  유효 경험: {len(validIdx)}/{len(series)}")
    print(f"  DOT-raw:  OWA {avgDotOwa:.4f}")
    print(f"  Oracle:   OWA {avgOracleOwa:.4f} ({oracleGap/avgDotOwa*100:+.1f}%)")

    print(f"\n  도메인별 DOT OWA:")
    for dom in sorted(domainCounts.keys()):
        domIdx = [i for i in validIdx if experiences[i]['domain'] == dom]
        if len(domIdx) > 0:
            domDot = np.mean([experiences[i]['dotOwa'] for i in domIdx])
            domOracle = np.mean([experiences[i]['bestOwa'] for i in domIdx])
            print(f"    {dom:15s}: DOT {domDot:.4f}, Oracle {domOracle:.4f}, gap {(domDot-domOracle)/domDot*100:.1f}%, n={len(domIdx)}")

    print("\n" + "=" * 70)
    print("--- Phase 1: 대규모 학습 곡선 ---")
    print("=" * 70)

    K = 50
    testSize = 2000
    np.random.seed(SEED + 1)
    allValid = validIdx.copy()
    np.random.shuffle(allValid)
    testIdx = allValid[:testSize]
    trainPool = allValid[testSize:]

    dbSizes = [2000, 5000, 10000, min(15000, len(trainPool))]
    dbSizes = [s for s in dbSizes if s <= len(trainPool)]

    print(f"\n  테스트셋: {testSize}개, 학습 풀: {len(trainPool)}개")
    print(f"\n  {'DB크기':>8} {'OWA':>8} {'DOT대비':>10} {'승률':>10} {'캡처율':>8} {'시간':>6}")
    print("  " + "-" * 56)

    lcResults = []
    for dbSize in dbSizes:
        phaseStart = time.time()
        dbIdx = trainPool[:dbSize]
        dbDnas = dnaMatrixNorm[dbIdx]
        dbExps = expArray[dbIdx]

        knnOwas = []
        dotOwas = []
        wins = 0

        for i in testIdx:
            result = knnSelect(dnaMatrixNorm[i], dbDnas, dbExps, min(K, dbSize))
            if result[0] is None:
                continue
            bestKey, _, _ = result
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
        phaseTime = time.time() - phaseStart
        print(f"  {dbSize:>8} {avgKnn:>8.4f} {impr:>+10.2f}% {wins}/{n} ({wins/n:.1%}) {capture:>+8.1f}% {phaseTime:>5.0f}s")
        lcResults.append((dbSize, avgKnn, impr, capture))

    print("\n" + "=" * 70)
    print("--- Phase 2: 도메인 순도 분석 ---")
    print("=" * 70)

    dbIdx = trainPool
    dbDnas = dnaMatrixNorm[dbIdx]
    dbDomains = domainArr[dbIdx]

    domainPurities = {d: [] for d in domainCounts}
    overallPurities = []

    for i in testIdx[:1000]:
        sims = dbDnas @ dnaMatrixNorm[i]
        topIdx = np.argsort(sims)[-K:]
        neighborDomains = dbDomains[topIdx]
        targetDomain = domainArr[i]

        purity = analyzeDomainPurity(targetDomain, neighborDomains)
        domainPurities[targetDomain].append(purity)
        overallPurities.append(purity)

    print(f"\n  kNN(k={K}) 이웃 도메인 순도 (같은 도메인 비율):")
    print(f"  {'도메인':>15s} {'평균순도':>8} {'n':>6}")
    print("  " + "-" * 35)
    for dom in sorted(domainCounts.keys()):
        if domainPurities[dom]:
            avgP = np.mean(domainPurities[dom])
            print(f"  {dom:>15s} {avgP:>8.1%} {len(domainPurities[dom]):>6}")

    overallPurity = np.mean(overallPurities)
    print(f"  {'전체':>15s} {overallPurity:>8.1%} {len(overallPurities):>6}")

    randomPurity = max(domainCounts.values()) / sum(domainCounts.values())
    print(f"\n  랜덤 기대치: {randomPurity:.1%} (최대 도메인 비율)")
    print(f"  DNA 분리 효과: {overallPurity/randomPurity:.2f}x (1.0이면 분리 안 됨)")

    print("\n" + "=" * 70)
    print("--- Phase 3: 도메인 내 경험 vs 전체 경험 ---")
    print("=" * 70)

    print(f"\n  {'도메인':>15s} {'전체DB':>8} {'도메인DB':>8} {'차이':>10} {'n_test':>8} {'n_domDB':>8}")
    print("  " + "-" * 65)

    domainResults = {}
    for dom in sorted(domainCounts.keys()):
        domTestIdx = [i for i in testIdx if domainArr[i] == dom]
        if len(domTestIdx) < 50:
            continue
        domTrainIdx = [i for i in trainPool if domainArr[i] == dom]
        if len(domTrainIdx) < 100:
            continue

        domDbDnas = dnaMatrixNorm[domTrainIdx]
        domDbExps = expArray[domTrainIdx]

        fullKnnOwas = []
        domKnnOwas = []
        dotOwasD = []

        for i in domTestIdx:
            fullResult = knnSelect(dnaMatrixNorm[i], dbDnas, expArray[dbIdx], min(K, len(dbIdx)))
            if fullResult[0] is None:
                continue
            fullKey, _, _ = fullResult
            fullOwa = experiences[i]['owas'].get(fullKey, experiences[i]['dotOwa'])

            domResult = knnSelect(dnaMatrixNorm[i], domDbDnas, domDbExps, min(K, len(domTrainIdx)))
            if domResult[0] is None:
                continue
            domKey, _, _ = domResult
            domOwa = experiences[i]['owas'].get(domKey, experiences[i]['dotOwa'])

            fullKnnOwas.append(fullOwa)
            domKnnOwas.append(domOwa)
            dotOwasD.append(experiences[i]['dotOwa'])

        if len(fullKnnOwas) < 50:
            continue

        avgFull = np.mean(fullKnnOwas)
        avgDom = np.mean(domKnnOwas)
        avgDotD = np.mean(dotOwasD)
        fullImpr = (avgDotD - avgFull) / avgDotD * 100
        domImpr = (avgDotD - avgDom) / avgDotD * 100
        diff = domImpr - fullImpr

        print(f"  {dom:>15s} {fullImpr:>+8.2f}% {domImpr:>+8.2f}% {diff:>+10.2f}% {len(fullKnnOwas):>8} {len(domTrainIdx):>8}")
        domainResults[dom] = {'full': fullImpr, 'domain': domImpr, 'diff': diff, 'nTest': len(fullKnnOwas), 'nDb': len(domTrainIdx)}

    if domainResults:
        avgDiff = np.mean([v['diff'] for v in domainResults.values()])
        print(f"\n  도메인 내 경험의 평균 추가 개선: {avgDiff:+.2f}%")
        print(f"  결론: {'도메인 분리 유효' if avgDiff > 0.5 else '도메인 분리 불필요 또는 미미'}")

    print("\n" + "=" * 70)
    print("--- Phase 4: 도메인별 성능 상세 ---")
    print("=" * 70)

    print(f"\n  {'도메인':>15s} {'DOT':>8} {'kNN선택':>8} {'개선':>8} {'Oracle':>8} {'캡처':>8} {'n':>6}")
    print("  " + "-" * 65)

    for dom in sorted(domainCounts.keys()):
        domTestIdx = [i for i in testIdx if domainArr[i] == dom]
        if len(domTestIdx) < 50:
            continue

        knnOwasD = []
        dotOwasD = []
        oracleOwasD = []
        winsD = 0

        for i in domTestIdx:
            result = knnSelect(dnaMatrixNorm[i], dbDnas, expArray[dbIdx], min(K, len(dbIdx)))
            if result[0] is None:
                continue
            bestKey, _, _ = result
            knnOwa = experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa'])
            knnOwasD.append(knnOwa)
            dotOwasD.append(experiences[i]['dotOwa'])
            oracleOwasD.append(experiences[i]['bestOwa'])
            if knnOwa < experiences[i]['dotOwa']:
                winsD += 1

        if len(knnOwasD) < 50:
            continue

        avgKnnD = np.mean(knnOwasD)
        avgDotD = np.mean(dotOwasD)
        avgOracleD = np.mean(oracleOwasD)
        imprD = (avgDotD - avgKnnD) / avgDotD * 100
        gapD = avgDotD - avgOracleD
        captureD = (avgDotD - avgKnnD) / gapD * 100 if gapD > 0 else 0
        print(f"  {dom:>15s} {avgDotD:>8.4f} {avgKnnD:>8.4f} {imprD:>+8.2f}% {avgOracleD:>8.4f} {captureD:>+8.1f}% {len(knnOwasD):>6}")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  총 시리즈: {len(series)}, 유효: {len(validIdx)}")
    print(f"  DNA 차원: {dnaMatrix.shape[1]}")
    print(f"  DOT-raw 전체: OWA {avgDotOwa:.4f}")
    print(f"  Oracle 전체:  OWA {avgOracleOwa:.4f} ({oracleGap/avgDotOwa*100:+.1f}%)")
    print(f"\n  학습 곡선:")
    for dbSize, avgKnn, impr, capture in lcResults:
        print(f"    {dbSize:>6}개: OWA {avgKnn:.4f} ({impr:+.2f}%), 캡처 {capture:.1f}%")
    print(f"\n  도메인 순도: {overallPurity:.1%} (랜덤 {randomPurity:.1%}, {overallPurity/randomPurity:.2f}x)")
    if domainResults:
        avgDiff = np.mean([v['diff'] for v in domainResults.values()])
        print(f"  도메인 내 추가 개선: {avgDiff:+.2f}%")
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)
