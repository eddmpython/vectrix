"""
실험 ID: dataProfiling/008
실험명: Daily OWA 진단 + 다중 모델 블렌드 탐색

목적:
- Daily OWA 0.996(Naive2 수준)의 근본 원인 진단
- 어떤 시리즈에서 실패하는지, 실패 패턴은 무엇인지 파악
- DOT+CES+DTSF 블렌드가 Hourly처럼 Daily에서도 작동하는지 확인
- DNA 특성과 모델 실패의 관계 분석

가설:
1. Daily 실패는 특정 유형(노이즈/비정상/긴 시리즈)에 집중될 것
2. DTSF가 Daily에서도 잔차 직교성으로 블렌드 기여 가능
3. DNA 특성으로 "모델이 실패하는 시리즈"를 식별할 수 있을 것

방법:
1. M4 Daily 전체 4227개, seed=42 1000개 샘플
2. DOT, CES, DTSF, 4Theta, ETS 5개 모델 예측
3. per-series OWA 분석 → 실패 분포 파악
4. DNA 특성 추출 → 실패/성공 그룹 비교
5. 2-way, 3-way 블렌드 그리드 탐색

결과 (실험 후 작성):
- M4 Daily 500개 샘플 (seed=42), period=7

개별 모델
  DOT:  OWA 0.783  median 0.790  실패율 23.6%
  CES:  OWA 0.788  median 0.803  실패율 24.4%
  DTSF: OWA 1.784  median 1.203  실패율 65.2%

실패 분포 (DOT OWA > 1.0)
  실패 118/500 (23.6%)
  실패 시리즈 평균 길이 2221 (중앙값 1934)
  성공 시리즈 평균 길이 2428 (중앙값 3066)
  → 짧은 시리즈가 약간 더 실패하지만 차이 크지 않음

잔차 상관
  DOT vs CES:  0.987 (거의 동일하게 틀림)
  DOT vs DTSF: 0.599
  CES vs DTSF: 0.646

블렌드 탐색
  DOT+CES: DOT 단독이 최적 (블렌드 개선 0)
  DOT+CES+DTSF: DOT 단독이 최적 (DTSF 파괴적)
  → 블렌딩 효과 완전 제로

결론:
- **Daily는 블렌딩이 불가능** — DOT-CES 잔차 상관 0.987, 같은 방식으로 틀림
- **Hourly와 정반대** — Hourly는 블렌딩 -8.2%, Daily는 0%
- **DTSF Daily 완전 파괴** — OWA 1.784, 65.2% 실패. period=7에서 DTSF 부적합
- **Daily 실패 23.6%는 구조적** — 시리즈 길이와 약한 상관만 있고 DNA 특성으로 분리 불가
- **Daily 개선은 블렌딩/선택이 아닌 모델 자체 개선 필요**
  - 다중 계절성(일/주/연) 처리
  - 또는 DL 모델(Foundation Model) 도입

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
PERIOD = 7
HORIZON = 14
MIN_LEN = 28
N_SAMPLE = 500

MODELS = ['dot', 'auto_ces', 'dtsf']


def loadM4Daily(nSample):
    trainPath = os.path.join(M4_DIR, 'Daily-train.csv')
    testPath = os.path.join(M4_DIR, 'Daily-test.csv')
    trainDf = pd.read_csv(trainPath, index_col=0)
    testDf = pd.read_csv(testPath, index_col=0)

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(trainDf), min(nSample, len(trainDf)), replace=False)

    trainSampled = trainDf.iloc[idx]
    testSampled = testDf.iloc[idx]

    data = {}
    for sid in trainSampled.index:
        trainVals = trainSampled.loc[sid].dropna().values.astype(np.float64)
        testVals = testSampled.loc[sid].dropna().values.astype(np.float64)
        if len(trainVals) >= MIN_LEN and len(testVals) >= HORIZON:
            data[str(sid)] = {
                'train': trainVals,
                'test': testVals[:HORIZON]
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
    lastSeason = trainY[-period:]
    reps = (steps // period) + 1
    return np.tile(lastSeason, reps)[:steps]


def forecastModel(trainY, steps, period, modelId):
    from vectrix.engine.registry import createModel
    try:
        model = createModel(modelId, period)
        model.fit(trainY)
        pred, _, _ = model.predict(steps)
        pred = np.where(np.isfinite(pred), pred, np.nanmean(trainY))
        return pred
    except Exception:
        return np.full(steps, np.nanmean(trainY))


def extractDnaFeatures(trainY):
    try:
        from vectrix.adaptive.dna import TSFeatureExtractor
        extractor = TSFeatureExtractor(period=PERIOD)
        features = extractor.extract(trainY)
        return features
    except Exception:
        return {}


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E008: Daily OWA Diagnosis + Multi-Model Blend")
    print("=" * 70)

    t0 = time.time()

    data = loadM4Daily(N_SAMPLE)
    print(f"\nLoaded {len(data)} Daily series")

    naiveSmapes = []
    naiveMases = []
    modelSmapes = {m: [] for m in MODELS}
    modelMases = {m: [] for m in MODELS}
    modelOwas = {m: [] for m in MODELS}
    allPreds = {m: {} for m in MODELS}
    seriesLens = []
    dnaFeatures = []
    sids = []

    for i, (sid, d) in enumerate(data.items()):
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(data)}...")

        trainY = d['train']
        testY = d['test']
        sids.append(sid)
        seriesLens.append(len(trainY))

        naive2Pred = naiveSeasonal(trainY, HORIZON, PERIOD)
        nSmape = computeSmape(testY, naive2Pred)
        nMase = computeMase(testY, naive2Pred, trainY, PERIOD)
        naiveSmapes.append(nSmape)
        naiveMases.append(nMase)

        for m in MODELS:
            pred = forecastModel(trainY, HORIZON, PERIOD, m)
            allPreds[m][sid] = pred
            s = computeSmape(testY, pred)
            a = computeMase(testY, pred, trainY, PERIOD)
            modelSmapes[m].append(s)
            modelMases[m].append(a)
            modelOwas[m].append(computeOwa(s, a, nSmape, nMase))

    avgNaiveSmape = np.mean(naiveSmapes)
    avgNaiveMase = np.mean(naiveMases)

    print(f"\n{'=' * 70}")
    print(f"=== Phase 1: Individual Model Performance ===")
    print(f"{'=' * 70}")

    for m in MODELS:
        owa = computeOwa(np.mean(modelSmapes[m]), np.mean(modelMases[m]),
                          avgNaiveSmape, avgNaiveMase)
        perSeriesOwas = np.array(modelOwas[m])
        print(f"  {m:>12}: OWA={owa:.4f}  median={np.median(perSeriesOwas):.4f}  "
              f"Q25={np.percentile(perSeriesOwas, 25):.4f}  Q75={np.percentile(perSeriesOwas, 75):.4f}  "
              f">1.0={np.mean(perSeriesOwas > 1.0)*100:.1f}%")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 2: Failure Distribution (DOT) ===")
    print(f"{'=' * 70}")

    dotOwas = np.array(modelOwas['dot'])
    lenArr = np.array(seriesLens)

    failMask = dotOwas > 1.0
    successMask = dotOwas <= 1.0
    print(f"  Total: {len(dotOwas)}, Fail(OWA>1): {failMask.sum()} ({failMask.mean()*100:.1f}%)")
    print(f"  Fail mean OWA: {dotOwas[failMask].mean():.4f}")
    print(f"  Success mean OWA: {dotOwas[successMask].mean():.4f}")
    print(f"\n  Length distribution:")
    print(f"    Fail series  avg len: {lenArr[failMask].mean():.0f}  median: {np.median(lenArr[failMask]):.0f}")
    print(f"    Success series avg len: {lenArr[successMask].mean():.0f}  median: {np.median(lenArr[successMask]):.0f}")

    bins = [(0, 100), (100, 500), (500, 1000), (1000, 5000), (5000, 999999)]
    print(f"\n  OWA by series length:")
    for lo, hi in bins:
        mask = (lenArr >= lo) & (lenArr < hi)
        if mask.sum() > 0:
            print(f"    [{lo:>5},{hi:>6}): n={mask.sum():>4}, OWA={dotOwas[mask].mean():.4f}, >1.0={np.mean(dotOwas[mask]>1.0)*100:.1f}%")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 3: DNA Feature Extraction (subset) ===")
    print(f"{'=' * 70}")

    nDna = min(100, len(data))
    dnaSids = sids[:nDna]
    print(f"  Extracting DNA for {nDna} series...")

    dnaRows = []
    for j, sid in enumerate(dnaSids):
        if (j + 1) % 100 == 0:
            print(f"    {j+1}/{nDna}...")
        features = extractDnaFeatures(data[sid]['train'])
        features['sid'] = sid
        features['dotOwa'] = modelOwas['dot'][sids.index(sid)]
        dnaRows.append(features)

    dnaDf = pd.DataFrame(dnaRows)

    numCols = [c for c in dnaDf.columns if c not in ['sid', 'dotOwa'] and dnaDf[c].dtype in [np.float64, np.int64, float, int]]
    failDna = dnaDf[dnaDf['dotOwa'] > 1.0]
    successDna = dnaDf[dnaDf['dotOwa'] <= 1.0]

    print(f"\n  DNA: Fail={len(failDna)}, Success={len(successDna)}")
    print(f"\n  Top features differentiating fail vs success (abs mean diff):")

    diffs = {}
    for col in numCols:
        fMean = failDna[col].mean()
        sMean = successDna[col].mean()
        poolStd = dnaDf[col].std()
        if poolStd > 1e-10:
            diffs[col] = abs(fMean - sMean) / poolStd
        else:
            diffs[col] = 0

    sortedDiffs = sorted(diffs.items(), key=lambda x: -x[1])
    for col, d in sortedDiffs[:15]:
        fMean = failDna[col].mean()
        sMean = successDna[col].mean()
        print(f"    {col:>30}: fail={fMean:.3f}  success={sMean:.3f}  diff={d:.3f}")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 4: Residual Correlation ===")
    print(f"{'=' * 70}")

    pairs = [('dot', 'auto_ces'), ('dot', 'dtsf'), ('auto_ces', 'dtsf')]
    for m1, m2 in pairs:
        residuals1 = []
        residuals2 = []
        for sid in sids:
            testY = data[sid]['test']
            r1 = testY - allPreds[m1][sid]
            r2 = testY - allPreds[m2][sid]
            residuals1.extend(r1)
            residuals2.extend(r2)
        corr = np.corrcoef(residuals1, residuals2)[0, 1]
        print(f"  {m1:>12} vs {m2:<12}: residual corr = {corr:.3f}")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 5: Blend Search ===")
    print(f"{'=' * 70}")

    print("\n  --- DOT + CES ---")
    bestOwa_dc = 999
    bestW_dc = None
    for w10 in range(0, 11):
        wD = w10 / 10.0
        wC = 1.0 - wD
        smapes = []
        mases = []
        for sid in sids:
            pred = wD * allPreds['dot'][sid] + wC * allPreds['auto_ces'][sid]
            smapes.append(computeSmape(data[sid]['test'], pred))
            mases.append(computeMase(data[sid]['test'], pred, data[sid]['train'], PERIOD))
        owa = computeOwa(np.mean(smapes), np.mean(mases), avgNaiveSmape, avgNaiveMase)
        if owa < bestOwa_dc:
            bestOwa_dc = owa
            bestW_dc = (wD, wC)
    print(f"  Best DOT+CES: DOT={bestW_dc[0]:.1f} CES={bestW_dc[1]:.1f} → OWA={bestOwa_dc:.4f}")

    print("\n  --- DOT + CES + DTSF ---")
    bestOwa_dcd = 999
    bestW_dcd = None
    topN = []
    for wD10 in range(0, 11):
        for wC10 in range(0, 11 - wD10):
            wT10 = 10 - wD10 - wC10
            wD = wD10 / 10.0
            wC = wC10 / 10.0
            wT = wT10 / 10.0
            smapes = []
            mases = []
            for sid in sids:
                pred = wD * allPreds['dot'][sid] + wC * allPreds['auto_ces'][sid] + wT * allPreds['dtsf'][sid]
                smapes.append(computeSmape(data[sid]['test'], pred))
                mases.append(computeMase(data[sid]['test'], pred, data[sid]['train'], PERIOD))
            owa = computeOwa(np.mean(smapes), np.mean(mases), avgNaiveSmape, avgNaiveMase)
            topN.append((wD, wC, wT, owa))
            if owa < bestOwa_dcd:
                bestOwa_dcd = owa
                bestW_dcd = (wD, wC, wT)
    topN.sort(key=lambda x: x[3])
    print("  Top 5:")
    for wD, wC, wT, owa in topN[:5]:
        print(f"    DOT={wD:.1f} CES={wC:.1f} DTSF={wT:.1f} → OWA={owa:.4f}")

    dotOwa = computeOwa(np.mean(modelSmapes['dot']), np.mean(modelMases['dot']),
                         avgNaiveSmape, avgNaiveMase)

    print(f"\n{'=' * 70}")
    print(f"=== FINAL SUMMARY ===")
    print(f"{'=' * 70}")
    print(f"  DOT single:     OWA={dotOwa:.4f}")
    print(f"  Best DOT+CES:   OWA={bestOwa_dc:.4f} ({bestOwa_dc - dotOwa:+.4f})")
    print(f"  Best DOT+CES+D: OWA={bestOwa_dcd:.4f} ({bestOwa_dcd - dotOwa:+.4f})")
    print(f"  Naive2 ref:     OWA=1.000")
    print(f"\n  Daily series characteristics:")
    print(f"    Mean length: {np.mean(seriesLens):.0f}")
    print(f"    Median length: {np.median(seriesLens):.0f}")
    print(f"    Fail rate (OWA>1): {np.mean(dotOwas > 1.0)*100:.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
