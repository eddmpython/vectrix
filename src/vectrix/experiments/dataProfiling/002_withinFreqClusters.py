"""
실험 ID: dataProfiling/002
실험명: Within-Frequency Clustering — 빈도 내 하위 그룹과 최적 모델 매핑

목적:
- 같은 빈도(Monthly) 안에서 DNA 특성 기반 하위 클러스터가 존재하는지 확인
- 클러스터별로 최적 모델이 다른지 검증 (DOT, AutoCES, 4Theta, AutoETS)
- 클러스터별 최적 모델이 다르면 → 프로파일링 기반 모델 선택의 가치 입증

가설:
1. Monthly 시리즈는 DNA 공간에서 2~5개 하위 그룹으로 나뉠 것
2. 클러스터별 최적 모델이 다를 것 (예: 트렌드 클러스터→Theta, 계절성 클러스터→ETS)
3. 클러스터별 최적 모델을 쓰면 단일 모델(DOT) 대비 OWA 개선 가능

방법:
1. M4 Monthly 500개 샘플, DNA 추출 + 정규화
2. K-means (k=2~8) + silhouette score로 최적 k 탐색
3. 각 클러스터에서 4개 모델 fit+predict → OWA 계산
4. 클러스터별 최적 모델 확인 + oracle 선택 시 전체 OWA 계산

결과 (실험 후 작성):
- Monthly 200개, 최적 k=2 (silhouette=0.202, 낮음)
- Cluster 0 (61개, 30.5%): 계절성 지배(0.44), 약한 트렌드(0.24), 짧은 시리즈(185점), 높은 엔트로피(0.64)
  → 최적 모델: AutoCES (0.865), DOT(0.885)보다 우위
- Cluster 1 (139개, 69.5%): 트렌드 지배(0.67), 약한 계절성(0.33), 긴 시리즈(259점), 낮은 엔트로피(0.34)
  → 최적 모델: DOT (0.674), AutoCES(0.744)보다 대폭 우위
- 전략별 OWA: DOT단일=0.788, CES단일=0.845, cluster-best=0.787, per-series oracle=0.665
- 4Theta/AutoETS는 Monthly 두 클러스터 모두에서 DOT/CES에 밀림

결론:
- **가설 1 부분 확인**: 하위 그룹 존재하나 silhouette 0.202로 경계 불명확
- **가설 2 확인**: 클러스터별 최적 모델이 다름 (계절성→CES, 트렌드→DOT)
- **가설 3 기각**: cluster-best(0.787) ≈ DOT단일(0.788), k=2로는 실질 개선 없음
- 그러나 per-series oracle(0.665)은 15.6% 개선 여지 → 더 세밀한 분류가 핵심
- 원리는 입증됨: 프로파일링으로 모델을 선택하면 개선 가능. 문제는 클러스터 해상도

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

M4_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
FREQ = 'Monthly'
PERIOD = 12
HORIZON = 18
N_SAMPLE = 200
SEED = 42

MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']


def loadM4WithTest(freq, nSample, seed=42):
    trainPath = os.path.join(M4_DIR, f'{freq}-train.csv')
    testPath = os.path.join(M4_DIR, f'{freq}-test.csv')
    trainDf = pd.read_csv(trainPath, index_col=0)
    testDf = pd.read_csv(testPath, index_col=0)

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
        if len(trainVals) >= 24 and len(testVals) >= HORIZON:
            data[str(sid)] = {
                'train': trainVals,
                'test': testVals[:HORIZON]
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

    featDf = pd.DataFrame(records, index=validIds)
    return featDf


def normalizeFeatures(featDf):
    cleanDf = featDf.replace([np.inf, -np.inf], np.nan)
    nanRatio = cleanDf.isna().mean()
    goodCols = nanRatio[nanRatio < 0.3].index.tolist()
    cleanDf = cleanDf[goodCols]
    cleanDf = cleanDf.fillna(cleanDf.median())

    scaler = StandardScaler()
    normalized = scaler.fit_transform(cleanDf)
    normDf = pd.DataFrame(normalized, index=cleanDf.index, columns=cleanDf.columns)
    return normDf, goodCols


def findOptimalK(normDf, kRange=range(2, 9)):
    scores = {}
    for k in kRange:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(normDf.values)
        sil = silhouette_score(normDf.values, labels)
        scores[k] = sil
        print(f"  k={k}: silhouette={sil:.3f}")
    bestK = max(scores, key=scores.get)
    return bestK, scores


def clusterSeries(normDf, k):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(normDf.values)
    return labels, km


def describeCluster(normDf, labels, rawFeatDf):
    nClusters = len(set(labels))
    print(f"\n=== Cluster Descriptions (k={nClusters}) ===")

    for c in range(nClusters):
        mask = labels == c
        n = mask.sum()
        subRaw = rawFeatDf.loc[normDf.index[mask]]

        print(f"\n  Cluster {c} (n={n}, {n/len(labels)*100:.1f}%)")

        topFeats = ['trend_strength', 'seasonality_strength', 'acf_lag1',
                     'hurst_exponent', 'entropy' if 'entropy' in subRaw.columns else 'spectral_entropy',
                     'cv', 'length', 'mean']
        for feat in topFeats:
            if feat in subRaw.columns:
                val = subRaw[feat].mean()
                std = subRaw[feat].std()
                print(f"    {feat:>25}: {val:.3f} (std={std:.3f})")


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


def naiveSeasonal(trainY, steps, period):
    if period <= 1:
        return np.full(steps, trainY[-1])
    lastSeason = trainY[-(period):]
    reps = (steps // period) + 1
    return np.tile(lastSeason, reps)[:steps]


def evaluateModelsPerCluster(data, normDf, labels, models, period, horizon):
    nClusters = len(set(labels))
    validIds = normDf.index.tolist()

    clusterResults = {}
    seriesResults = {}

    for c in range(nClusters):
        mask = labels == c
        clusterIds = [validIds[i] for i in range(len(validIds)) if mask[i]]
        clusterResults[c] = {m: {'smapes': [], 'mases': []} for m in models}
        clusterResults[c]['naive2'] = {'smapes': [], 'mases': []}
        print(f"\n  Cluster {c}: {len(clusterIds)} series")

        for sid in clusterIds:
            if sid not in data:
                continue
            trainY = data[sid]['train']
            testY = data[sid]['test']

            naive2Pred = naiveSeasonal(trainY, horizon, period)
            naiveSmape = computeSmape(testY, naive2Pred)
            naiveMase = computeMase(testY, naive2Pred, trainY, period)
            clusterResults[c]['naive2']['smapes'].append(naiveSmape)
            clusterResults[c]['naive2']['mases'].append(naiveMase)

            seriesResults[sid] = {'cluster': c}

            for modelId in models:
                pred = forecastWithModel(trainY, horizon, period, modelId)
                smape = computeSmape(testY, pred)
                mase = computeMase(testY, pred, trainY, period)
                owa = computeOwa(smape, mase, naiveSmape, naiveMase)
                clusterResults[c][modelId]['smapes'].append(smape)
                clusterResults[c][modelId]['mases'].append(mase)
                seriesResults[sid][modelId] = owa

    return clusterResults, seriesResults


def printClusterModelTable(clusterResults, models):
    nClusters = len(clusterResults)
    print(f"\n{'=' * 80}")
    print(f"=== Cluster x Model OWA Table ===")
    print(f"{'=' * 80}")

    header = f"{'Cluster':>10} {'N':>6}"
    for m in models:
        header += f" {m:>12}"
    header += f" {'Best':>12} {'OWA':>8}"
    print(header)
    print("-" * len(header))

    clusterBest = {}
    for c in sorted(clusterResults.keys()):
        cr = clusterResults[c]
        n = len(cr['naive2']['smapes'])
        naiveSmape = np.mean(cr['naive2']['smapes'])
        naiveMase = np.mean(cr['naive2']['mases'])

        row = f"{'C' + str(c):>10} {n:>6}"
        bestModel = None
        bestOwa = 999

        for m in models:
            smape = np.mean(cr[m]['smapes'])
            mase = np.mean(cr[m]['mases'])
            owa = computeOwa(smape, mase, naiveSmape, naiveMase)
            row += f" {owa:>12.3f}"
            if owa < bestOwa:
                bestOwa = owa
                bestModel = m

        row += f" {bestModel:>12} {bestOwa:>8.3f}"
        print(row)
        clusterBest[c] = bestModel

    return clusterBest


def computeOverallOwa(seriesResults, clusterBest, models):
    print(f"\n{'=' * 80}")
    print(f"=== Overall OWA Comparison ===")
    print(f"{'=' * 80}")

    for m in models:
        owas = [sr[m] for sr in seriesResults.values() if m in sr]
        print(f"  {m:>15} (uniform): {np.mean(owas):.4f}")

    clusterOwas = []
    for sid, sr in seriesResults.items():
        c = sr['cluster']
        bestM = clusterBest[c]
        clusterOwas.append(sr[bestM])
    print(f"  {'cluster-best':>15} (oracle): {np.mean(clusterOwas):.4f}")

    oracleOwas = []
    for sid, sr in seriesResults.items():
        bestOwa = min(sr[m] for m in models if m in sr)
        oracleOwas.append(bestOwa)
    print(f"  {'per-series':>15} (oracle): {np.mean(oracleOwas):.4f}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E002: Within-Frequency Clustering + Model Selection")
    print(f"  Freq={FREQ}, Period={PERIOD}, Horizon={HORIZON}, N={N_SAMPLE}")
    print("=" * 70)

    t0 = time.time()

    print("\n[1] Loading M4 data with test sets...")
    data = loadM4WithTest(FREQ, N_SAMPLE, SEED)
    print(f"  Loaded: {len(data)} series with train+test")

    print("\n[2] Extracting DNA features...")
    rawFeatDf = extractFeatures(data, PERIOD)
    print(f"  Feature matrix: {rawFeatDf.shape}")

    print("\n[3] Normalizing...")
    normDf, goodCols = normalizeFeatures(rawFeatDf)
    print(f"  Normalized: {normDf.shape}")

    print("\n[4] Finding optimal k (silhouette)...")
    bestK, scores = findOptimalK(normDf)
    print(f"  Best k={bestK} (silhouette={scores[bestK]:.3f})")

    print(f"\n[5] Clustering with k={bestK}...")
    labels, km = clusterSeries(normDf, bestK)
    describeCluster(normDf, labels, rawFeatDf)

    print(f"\n[6] Evaluating models per cluster...")
    print(f"  Models: {MODELS}")
    clusterResults, seriesResults = evaluateModelsPerCluster(
        data, normDf, labels, MODELS, PERIOD, HORIZON
    )

    clusterBest = printClusterModelTable(clusterResults, MODELS)

    computeOverallOwa(seriesResults, clusterBest, MODELS)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
