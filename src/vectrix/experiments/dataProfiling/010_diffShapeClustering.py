"""
실험 ID: dataProfiling/010
실험명: 미분 형상 기반 클러스터링 + 전략 매핑

목적:
- DNA 통계량(4개 숫자)이 아닌 미분 형상(시계열 자체의 움직임)으로 분류
- 미분+정규화 형상 벡터로 클러스터링하면 같은 전략이 통하는 그룹이 나오는지
- E009의 oracle gap -17.4%에서 얼마를 캡처할 수 있는지
- DNA 기반 E003(36.7%) vs 형상 기반 전략 매핑 정확도 비교

가설:
1. 미분 형상이 비슷한 시리즈는 같은 최적 전략을 공유할 것
2. 형상 벡터 기반 클러스터가 DNA 기반보다 전략 일치율이 높을 것
3. LightGBM(형상→전략)이 Ridge(DNA→전략)보다 정확할 것

방법:
1. M4 Monthly 500개 (seed=42)
2. 각 시리즈: 미분 → 표준편차로 정규화 → 24포인트로 리샘플링 = 형상 벡터
3. 5가지 전처리 × DOT 예측 → per-series 최적 전략 라벨
4. K-means (k=5,8,12) 클러스터링 → 클러스터 내 전략 일치율
5. LightGBM: 형상 벡터 → 최적 전략 분류 (5-fold CV)
6. 비교: DNA Ridge(E003) vs 형상 LightGBM

결과 (실험 후 작성):
- Raw DOT OWA: 0.812, Oracle 전처리 OWA: 0.673 (-17.1%)
- Shape 클러스터 purity: k=5 0.284, k=8 0.290, k=12 0.320
- DNA 클러스터 purity: k=5 0.278, k=8 0.298 → shape와 거의 동일
- Cluster-best OWA: k=5 1.128(+39%), k=8 1.073(+32%), k=12 0.889(+9.5%) → 전부 악화
- Ridge(DNA 4d) 정확도: 26.2%, Ridge(Shape 24d): 24.4%, Ridge(Combined): 25.2%
- Ridge(Shape) OWA: 1.009(+24%), Ridge(Combined) OWA: 0.968(+19%) → 전부 악화
- LightGBM 미설치로 스킵

결론:
- 가설 1 기각: 미분 형상이 비슷해도 최적 전략이 다르다 (purity 0.32가 최대)
- 가설 2 기각: Shape 24d(24.4%) < DNA 4d(26.2%). 차원 증가가 정보 증가가 아님
- 가설 3 미검증: LightGBM 없었으나, Ridge 기준 표현력 차이 없음
- E009 결론 완전 재확인: DNA든 Shape든 전처리 전략 선택은 정적 규칙/분류기로 불가능
- 핵심 교훈: 문제는 "표현"이 아니라 "매핑 자체의 비결정성"
  → 같은 형태라도 내부 구조(노이즈 수준, 이상치 위치)에 따라 최적 전처리가 달라짐
  → holdout 기반 동적 선택만이 유일한 경로

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier

warnings.filterwarnings('ignore')

M4_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
SEED = 42
PERIOD = 12
HORIZON = 18
MIN_LEN = 36
N_SAMPLE = 500
SHAPE_DIM = 24

STRATEGIES = ['raw', 'log', 'diff', 'ma3', 'stl']


def loadM4Monthly(nSample):
    trainPath = os.path.join(M4_DIR, 'Monthly-train.csv')
    testPath = os.path.join(M4_DIR, 'Monthly-test.csv')
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
            data[str(sid)] = {'train': trainVals, 'test': testVals[:HORIZON]}
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
    lastSeason = trainY[-period:]
    reps = (steps // period) + 1
    return np.tile(lastSeason, reps)[:steps]


def makeShapeVector(trainY, dim=SHAPE_DIM):
    diffY = np.diff(trainY)
    std = np.std(diffY)
    if std < 1e-10:
        diffNorm = np.zeros_like(diffY)
    else:
        diffNorm = diffY / std

    n = len(diffNorm)
    if n <= 1:
        return np.zeros(dim)

    x = np.linspace(0, 1, n)
    xNew = np.linspace(0, 1, dim)
    f = interp1d(x, diffNorm, kind='linear')
    return f(xNew)


def makeDnaVector(trainY, period):
    cv = np.std(trainY) / (np.mean(np.abs(trainY)) + 1e-10)

    n = len(trainY)
    if n > period * 2:
        nCycles = n // period
        truncated = trainY[:nCycles * period]
        reshaped = truncated.reshape(nCycles, period)
        seasonalMean = np.mean(reshaped, axis=0)
        seasonalVar = np.var(seasonalMean)
        totalVar = np.var(trainY)
        seasStrength = seasonalVar / (totalVar + 1e-10)
    else:
        seasStrength = 0.0

    x = np.arange(n, dtype=np.float64)
    xNorm = x - x.mean()
    slope = np.sum(xNorm * (trainY - trainY.mean())) / (np.sum(xNorm ** 2) + 1e-10)
    trendStrength = abs(slope) * n / (np.std(trainY) + 1e-10)

    if n > 1:
        y = trainY - np.mean(trainY)
        acf1 = np.sum(y[:-1] * y[1:]) / (np.sum(y ** 2) + 1e-10)
    else:
        acf1 = 0.0

    return np.array([cv, seasStrength, trendStrength, acf1])


def forecastDot(trainY, steps, period):
    from vectrix.engine.registry import createModel
    try:
        model = createModel('dot', period)
        model.fit(trainY)
        pred, _, _ = model.predict(steps)
        pred = np.where(np.isfinite(pred), pred, np.nanmean(trainY))
        return pred
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return np.full(steps, np.nanmean(trainY))


def applyPreprocess(trainY, strategy, period):
    if strategy == 'raw':
        return trainY, lambda pred: pred

    if strategy == 'log':
        minVal = np.min(trainY)
        shift = abs(minVal) + 1.0 if minVal <= 0 else 0.0
        logY = np.log1p(trainY + shift)
        return logY, lambda pred: np.expm1(pred) - shift

    if strategy == 'diff':
        diffY = np.diff(trainY)
        lastVal = trainY[-1]
        return diffY, lambda pred: lastVal + np.cumsum(pred)

    if strategy == 'ma3':
        kernel = np.ones(3) / 3
        smoothed = np.convolve(trainY, kernel, mode='same')
        smoothed[:1] = trainY[:1]
        smoothed[-1:] = trainY[-1:]
        return smoothed, lambda pred: pred

    if strategy == 'stl':
        n = len(trainY)
        if n < period * 2:
            return trainY, lambda pred: pred
        nCycles = n // period
        truncated = trainY[:nCycles * period]
        reshaped = truncated.reshape(nCycles, period)
        seasonal = np.mean(reshaped, axis=0)
        seasonal = seasonal - np.mean(seasonal)
        seasonalFull = np.tile(seasonal, nCycles + 1)[:n]
        deseasonalized = trainY - seasonalFull

        from scipy.ndimage import uniform_filter1d
        trend = uniform_filter1d(deseasonalized, size=max(3, period // 2))
        remainder = deseasonalized - trend

        trendSlope = trend[-1] - trend[-2] if len(trend) >= 2 else 0
        def inverse(pred):
            steps = len(pred)
            seasExt = np.tile(seasonal, (steps // period) + 1)[:steps]
            trendExt = trend[-1] + trendSlope * np.arange(1, steps + 1)
            return pred + seasExt + trendExt

        return remainder, inverse

    return trainY, lambda pred: pred


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E010: Diff-Shape Clustering + Strategy Mapping")
    print("=" * 70)

    t0 = time.time()

    data = loadM4Monthly(N_SAMPLE)
    sids = list(data.keys())
    print(f"\nLoaded {len(data)} Monthly series")

    print(f"\n--- Computing shape vectors & strategy labels ---")

    shapeVectors = []
    dnaVectors = []
    stratOwas = {s: [] for s in STRATEGIES}
    naiveSmapes = []
    naiveMases = []

    for i, sid in enumerate(sids):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sids)}...", flush=True)

        trainY = data[sid]['train']
        testY = data[sid]['test']

        shapeVec = makeShapeVector(trainY, SHAPE_DIM)
        shapeVectors.append(shapeVec)

        dnaVec = makeDnaVector(trainY, PERIOD)
        dnaVectors.append(dnaVec)

        naive2Pred = naiveSeasonal(trainY, HORIZON, PERIOD)
        nSmape = computeSmape(testY, naive2Pred)
        nMase = computeMase(testY, naive2Pred, trainY, PERIOD)
        naiveSmapes.append(nSmape)
        naiveMases.append(nMase)

        for s in STRATEGIES:
            processed, inverseFn = applyPreprocess(trainY, s, PERIOD)
            if len(processed) < 10:
                stratOwas[s].append(999.0)
                continue
            prd = PERIOD if s != 'diff' else max(1, PERIOD)
            pred = forecastDot(processed, HORIZON, prd)
            predInv = inverseFn(pred)
            predInv = np.where(np.isfinite(predInv), predInv, np.nanmean(trainY))
            sm = computeSmape(testY, predInv)
            ma = computeMase(testY, predInv, trainY, PERIOD)
            stratOwas[s].append(computeOwa(sm, ma, nSmape, nMase))

    shapeMatrix = np.array(shapeVectors)
    dnaMatrix = np.array(dnaVectors)
    owaMatrix = np.array([stratOwas[s] for s in STRATEGIES])
    bestStratIdx = np.argmin(owaMatrix, axis=0)
    rawOwas = np.array(stratOwas['raw'])
    oracleBest = np.min(owaMatrix, axis=0)

    rawOwa = np.mean(rawOwas)
    oracleOwa = np.mean(oracleBest)
    print(f"\n  Raw OWA: {rawOwa:.4f}")
    print(f"  Oracle OWA: {oracleOwa:.4f} ({(oracleOwa-rawOwa)/rawOwa*100:+.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 1: Shape Clustering ===")
    print(f"{'=' * 70}")

    for k in [5, 8, 12]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(shapeMatrix)

        clusterPurities = []
        clusterSizes = []
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                continue
            clusterStrats = bestStratIdx[mask]
            mostCommon = np.bincount(clusterStrats, minlength=len(STRATEGIES)).argmax()
            purity = np.mean(clusterStrats == mostCommon)
            clusterPurities.append(purity)
            clusterSizes.append(mask.sum())

        avgPurity = np.average(clusterPurities, weights=clusterSizes)
        print(f"\n  k={k}: avg purity = {avgPurity:.3f}")
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                continue
            clusterStrats = bestStratIdx[mask]
            mostCommon = np.bincount(clusterStrats, minlength=len(STRATEGIES)).argmax()
            purity = np.mean(clusterStrats == mostCommon)
            clusterOwa = np.mean(rawOwas[mask])
            print(f"    cluster {c}: n={mask.sum():>3}, best={STRATEGIES[mostCommon]:>4}, "
                  f"purity={purity:.2f}, raw_owa={clusterOwa:.3f}")

    print(f"\n  --- DNA Clustering comparison ---")
    for k in [5, 8]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(dnaMatrix)

        clusterPurities = []
        clusterSizes = []
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                continue
            clusterStrats = bestStratIdx[mask]
            mostCommon = np.bincount(clusterStrats, minlength=len(STRATEGIES)).argmax()
            purity = np.mean(clusterStrats == mostCommon)
            clusterPurities.append(purity)
            clusterSizes.append(mask.sum())

        avgPurity = np.average(clusterPurities, weights=clusterSizes)
        print(f"  DNA k={k}: avg purity = {avgPurity:.3f}")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 2: Cluster-Best Strategy OWA ===")
    print(f"{'=' * 70}")

    for k in [5, 8, 12]:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(shapeMatrix)

        selectedOwas = np.zeros(len(sids))
        for c in range(k):
            mask = labels == c
            if mask.sum() == 0:
                continue
            clusterStrats = bestStratIdx[mask]
            mostCommon = np.bincount(clusterStrats, minlength=len(STRATEGIES)).argmax()
            selectedOwas[mask] = owaMatrix[mostCommon, mask]

        clusterOwa = np.mean(selectedOwas)
        gap = rawOwa - oracleOwa
        captured = (rawOwa - clusterOwa) / gap * 100 if gap > 0 else 0
        print(f"  Shape k={k}: OWA={clusterOwa:.4f} ({clusterOwa-rawOwa:+.4f}, gap captured {captured:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 3: LightGBM vs Ridge Classification ===")
    print(f"{'=' * 70}")

    print(f"\n  --- Ridge (DNA 4-dim) ---")
    ridge = RidgeClassifier(alpha=1.0)
    dnaScores = cross_val_score(ridge, dnaMatrix, bestStratIdx, cv=5, scoring='accuracy')
    print(f"  Ridge DNA accuracy: {dnaScores.mean():.3f} (+/- {dnaScores.std():.3f})")

    print(f"\n  --- Ridge (Shape 24-dim) ---")
    ridge2 = RidgeClassifier(alpha=1.0)
    shapeScores = cross_val_score(ridge2, shapeMatrix, bestStratIdx, cv=5, scoring='accuracy')
    print(f"  Ridge Shape accuracy: {shapeScores.mean():.3f} (+/- {shapeScores.std():.3f})")

    print(f"\n  --- Ridge (DNA + Shape combined) ---")
    combinedMatrix = np.hstack([dnaMatrix, shapeMatrix])
    ridge3 = RidgeClassifier(alpha=1.0)
    combScores = cross_val_score(ridge3, combinedMatrix, bestStratIdx, cv=5, scoring='accuracy')
    print(f"  Ridge Combined accuracy: {combScores.mean():.3f} (+/- {combScores.std():.3f})")

    try:
        from lightgbm import LGBMClassifier

        print(f"\n  --- LightGBM (Shape 24-dim) ---")
        lgb1 = LGBMClassifier(n_estimators=100, max_depth=5, random_state=SEED, verbose=-1)
        lgbShapeScores = cross_val_score(lgb1, shapeMatrix, bestStratIdx, cv=5, scoring='accuracy')
        print(f"  LGB Shape accuracy: {lgbShapeScores.mean():.3f} (+/- {lgbShapeScores.std():.3f})")

        print(f"\n  --- LightGBM (DNA + Shape combined) ---")
        lgb2 = LGBMClassifier(n_estimators=100, max_depth=5, random_state=SEED, verbose=-1)
        lgbCombScores = cross_val_score(lgb2, combinedMatrix, bestStratIdx, cv=5, scoring='accuracy')
        print(f"  LGB Combined accuracy: {lgbCombScores.mean():.3f} (+/- {lgbCombScores.std():.3f})")

    except ImportError:
        print(f"\n  LightGBM not available, skipping")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 4: Strategy OWA via Classification ===")
    print(f"{'=' * 70}")

    from sklearn.model_selection import cross_val_predict

    ridgePreds = cross_val_predict(RidgeClassifier(alpha=1.0), shapeMatrix, bestStratIdx, cv=5)
    ridgeSelectedOwas = np.array([owaMatrix[ridgePreds[j], j] for j in range(len(sids))])
    ridgeOwa = np.mean(ridgeSelectedOwas)
    gap = rawOwa - oracleOwa
    ridgeCaptured = (rawOwa - ridgeOwa) / gap * 100 if gap > 0 else 0
    print(f"  Ridge(Shape) selection: OWA={ridgeOwa:.4f} ({ridgeOwa-rawOwa:+.4f}, gap {ridgeCaptured:.1f}%)")

    ridgePreds2 = cross_val_predict(RidgeClassifier(alpha=1.0), combinedMatrix, bestStratIdx, cv=5)
    ridgeSelectedOwas2 = np.array([owaMatrix[ridgePreds2[j], j] for j in range(len(sids))])
    ridgeOwa2 = np.mean(ridgeSelectedOwas2)
    ridgeCaptured2 = (rawOwa - ridgeOwa2) / gap * 100 if gap > 0 else 0
    print(f"  Ridge(Combined) selection: OWA={ridgeOwa2:.4f} ({ridgeOwa2-rawOwa:+.4f}, gap {ridgeCaptured2:.1f}%)")

    try:
        from lightgbm import LGBMClassifier

        lgbPreds = cross_val_predict(
            LGBMClassifier(n_estimators=100, max_depth=5, random_state=SEED, verbose=-1),
            combinedMatrix, bestStratIdx, cv=5
        )
        lgbSelectedOwas = np.array([owaMatrix[lgbPreds[j], j] for j in range(len(sids))])
        lgbOwa = np.mean(lgbSelectedOwas)
        lgbCaptured = (rawOwa - lgbOwa) / gap * 100 if gap > 0 else 0
        print(f"  LGB(Combined) selection: OWA={lgbOwa:.4f} ({lgbOwa-rawOwa:+.4f}, gap {lgbCaptured:.1f}%)")
    except ImportError:
        pass

    print(f"\n{'=' * 70}")
    print(f"=== FINAL SUMMARY ===")
    print(f"{'=' * 70}")
    print(f"  Raw DOT:           OWA={rawOwa:.4f}")
    print(f"  Oracle preproc:     OWA={oracleOwa:.4f} ({(oracleOwa-rawOwa)/rawOwa*100:+.1f}%)")
    print(f"  E009 rule-based:    OWA=1.091 (+34.4%)")
    print(f"  Ridge(DNA) acc:     {dnaScores.mean():.3f}")
    print(f"  Ridge(Shape) acc:   {shapeScores.mean():.3f}")
    print(f"  Ridge(Combined) acc:{combScores.mean():.3f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
