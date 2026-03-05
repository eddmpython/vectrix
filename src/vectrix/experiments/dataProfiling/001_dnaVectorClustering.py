"""
실험 ID: dataProfiling/001
실험명: DNA Vector Clustering — M4 시계열 특성 공간 탐색

목적:
- M4 100K 시리즈의 DNA 특성 벡터를 추출하고 정규화
- 코사인 유사도(내적)로 시리즈 간 유사성 측정
- 차원 축소(PCA, t-SNE)로 시각화하여 자연스러운 클러스터 관찰
- 라벨 없이 시작 — 데이터가 스스로 그루핑을 보여주는지 확인

가설:
1. M4 시리즈는 DNA 특성 공간에서 자연스러운 클러스터를 형성할 것
2. 빈도(Yearly/Monthly/...)가 클러스터와 얼마나 겹치는지 불확실 — 관찰 대상
3. 같은 빈도 안에서도 하위 클러스터가 존재할 수 있음

방법:
1. M4 CSV에서 빈도별 500개 샘플 (Weekly/Hourly는 전체) → 총 ~3000개
2. TSFeatureExtractor로 65개 DNA 특성 추출
3. z-score 정규화 (각 특성별 mean=0, std=1)
4. 코사인 유사도 행렬 계산
5. PCA로 분산 설명률 확인, 상위 2~3 PC로 scatter
6. t-SNE 2D 시각화 (빈도별 색상)
7. HDBSCAN으로 참고용 클러스터링 (라벨 강제 아님)

결과 (실험 후 작성):
- 2773개 시리즈 (빈도별 500, Weekly 359, Hourly 414), 65개 DNA 특성 추출
- 코사인 유사도: 같은 빈도 내부 0.37~0.49, Hourly 가장 동질적(0.494), Daily 가장 이질적(0.369)
- 빈도 간: Daily-Yearly 가장 먼(-0.227), Monthly-Quarterly 가장 가까운(+0.082)
- PCA: PC1=자기상관축(20.8%), PC2=스케일축(18.7%), PC3=트렌드축(10.5%)
  - 95% 분산에 21개 성분 필요 → 고차원 구조
- 상위 판별 특성: seasonality_strength(2.50), length(2.04), diff_mean_ratio(1.96)
- t-SNE: Hourly/Yearly 명확히 분리, Monthly/Quarterly 겹침, Daily/Weekly 부분 겹침
- 빈도 내 분산: Daily는 max_kl_shift/seasonal_period, Hourly는 lumpiness/scale, Yearly는 trend_curvature/hurst

결론:
- 빈도 ≠ 유형이지만, 빈도는 강한 축. 빈도 효과를 제거해야 교차 빈도 유형 발견 가능
- 같은 빈도 안에서도 상당한 다양성 존재 (self-similarity 0.37~0.49) → 하위 클러스터 탐색 가치 있음
- Hourly가 가장 동질적 = 다중 계절성이라는 공통 특성이 지배적

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')

M4_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')

FREQ_CONFIG = {
    'Yearly': {'period': 1, 'sample': 500},
    'Quarterly': {'period': 4, 'sample': 500},
    'Monthly': {'period': 12, 'sample': 500},
    'Weekly': {'period': 1, 'sample': 359},
    'Daily': {'period': 7, 'sample': 500},
    'Hourly': {'period': 24, 'sample': 414},
}


def loadM4Series(freq, nSample, seed=42):
    path = os.path.join(M4_DIR, f'{freq}-train.csv')
    df = pd.read_csv(path, index_col=0)

    if nSample >= len(df):
        sampled = df
    else:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(df), nSample, replace=False)
        sampled = df.iloc[idx]

    series = {}
    for sid, row in sampled.iterrows():
        vals = row.dropna().values.astype(np.float64)
        if len(vals) >= 10:
            series[str(sid)] = vals
    return series


def extractDnaFeatures(allSeries, freqPeriods):
    from vectrix.engine.tsfeatures import TSFeatureExtractor

    extractor = TSFeatureExtractor()
    records = []
    seriesIds = []
    freqLabels = []

    for freq, seriesDict in allSeries.items():
        period = freqPeriods[freq]
        print(f"  {freq}: {len(seriesDict)} series, period={period}")

        for sid, y in seriesDict.items():
            try:
                feat = extractor.extract(y, period=period)
                if feat and len(feat) > 0:
                    records.append(feat)
                    seriesIds.append(sid)
                    freqLabels.append(freq)
            except Exception:
                pass

    featDf = pd.DataFrame(records, index=seriesIds)
    featDf['_freq'] = freqLabels
    return featDf


def cleanAndNormalize(featDf):
    freqCol = featDf['_freq']
    numDf = featDf.drop(columns=['_freq'])

    numDf = numDf.replace([np.inf, -np.inf], np.nan)

    nanRatio = numDf.isna().mean()
    goodCols = nanRatio[nanRatio < 0.3].index.tolist()
    numDf = numDf[goodCols]

    numDf = numDf.fillna(numDf.median())

    scaler = StandardScaler()
    normalized = scaler.fit_transform(numDf)
    normDf = pd.DataFrame(normalized, index=numDf.index, columns=numDf.columns)

    return normDf, freqCol, goodCols


def analyzeCosineSimilarity(normDf, freqCol):
    cosMat = cosine_similarity(normDf.values)

    freqs = freqCol.unique()
    print("\n=== Cosine Similarity (mean) ===")
    print(f"{'':>12}", end='')
    for f in sorted(freqs):
        print(f"{f:>12}", end='')
    print()

    for f1 in sorted(freqs):
        mask1 = (freqCol == f1).values
        print(f"{f1:>12}", end='')
        for f2 in sorted(freqs):
            mask2 = (freqCol == f2).values
            subMat = cosMat[np.ix_(mask1, mask2)]
            if f1 == f2:
                upperIdx = np.triu_indices(subMat.shape[0], k=1)
                if len(upperIdx[0]) > 0:
                    meanSim = subMat[upperIdx].mean()
                else:
                    meanSim = 0.0
            else:
                meanSim = subMat.mean()
            print(f"{meanSim:>12.3f}", end='')
        print()

    return cosMat


def runPCA(normDf, freqCol):
    pca = PCA(n_components=min(20, normDf.shape[1]))
    pcScores = pca.fit_transform(normDf.values)

    print("\n=== PCA Variance Explained ===")
    cumVar = np.cumsum(pca.explained_variance_ratio_)
    for i in range(min(10, len(cumVar))):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]:.3f} (cum: {cumVar[i]:.3f})")

    n90 = np.searchsorted(cumVar, 0.90) + 1
    n95 = np.searchsorted(cumVar, 0.95) + 1
    print(f"\n  Components for 90% variance: {n90}")
    print(f"  Components for 95% variance: {n95}")

    print("\n=== Top 5 features per PC (absolute loading) ===")
    for pc in range(min(3, len(pca.components_))):
        loadings = np.abs(pca.components_[pc])
        topIdx = np.argsort(loadings)[::-1][:5]
        feats = [(normDf.columns[i], pca.components_[pc][i]) for i in topIdx]
        print(f"  PC{pc+1}: {', '.join(f'{name}({val:+.3f})' for name, val in feats)}")

    freqs = sorted(freqCol.unique())
    print("\n=== PC1 vs PC2 centroids by frequency ===")
    for f in freqs:
        mask = (freqCol == f).values
        c1 = pcScores[mask, 0].mean()
        c2 = pcScores[mask, 1].mean()
        s1 = pcScores[mask, 0].std()
        s2 = pcScores[mask, 1].std()
        print(f"  {f:>12}: PC1={c1:+.2f} (std={s1:.2f}), PC2={c2:+.2f} (std={s2:.2f})")

    return pcScores, pca


def runTSNE(normDf, freqCol):
    print("\n=== t-SNE (perplexity=30) ===")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    embedded = tsne.fit_transform(normDf.values)

    freqs = sorted(freqCol.unique())
    print("  Frequency spread in t-SNE space:")
    for f in freqs:
        mask = (freqCol == f).values
        cx = embedded[mask, 0].mean()
        cy = embedded[mask, 1].mean()
        sx = embedded[mask, 0].std()
        sy = embedded[mask, 1].std()
        print(f"  {f:>12}: center=({cx:.1f}, {cy:.1f}), spread=({sx:.1f}, {sy:.1f})")

    return embedded


def analyzeWithinFreqVariation(normDf, freqCol):
    print("\n=== Within-Frequency Feature Variation ===")
    print("  (Which features vary most WITHIN each frequency?)")

    freqs = sorted(freqCol.unique())
    for f in freqs:
        mask = (freqCol == f).values
        subDf = normDf[mask]
        stds = subDf.std()
        topVar = stds.nlargest(5)
        print(f"  {f:>12}: {', '.join(f'{name}({val:.2f})' for name, val in topVar.items())}")


def featureDistributionByFreq(normDf, freqCol):
    print("\n=== Feature Means by Frequency (top discriminating) ===")

    freqs = sorted(freqCol.unique())
    means = {}
    for f in freqs:
        mask = (freqCol == f).values
        means[f] = normDf[mask].mean()

    meanDf = pd.DataFrame(means)
    rangePerFeat = meanDf.max(axis=1) - meanDf.min(axis=1)
    topDisc = rangePerFeat.nlargest(10)

    print(f"{'Feature':>30}", end='')
    for f in freqs:
        print(f"{f:>10}", end='')
    print(f"{'Range':>10}")

    for feat in topDisc.index:
        print(f"{feat:>30}", end='')
        for f in freqs:
            print(f"{means[f][feat]:>10.2f}", end='')
        print(f"{topDisc[feat]:>10.2f}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E001: DNA Vector Clustering — M4 Time Series Feature Space")
    print("=" * 70)

    t0 = time.time()

    print("\n[1] Loading M4 series...")
    allSeries = {}
    for freq, cfg in FREQ_CONFIG.items():
        series = loadM4Series(freq, cfg['sample'])
        allSeries[freq] = series
        print(f"  {freq}: {len(series)} series loaded")

    totalSeries = sum(len(s) for s in allSeries.values())
    print(f"  Total: {totalSeries} series")

    print(f"\n[2] Extracting DNA features ({totalSeries} series)...")
    featDf = extractDnaFeatures(allSeries, {f: c['period'] for f, c in FREQ_CONFIG.items()})
    print(f"  Feature matrix: {featDf.shape[0]} series x {featDf.shape[1]-1} features")

    print("\n[3] Cleaning and normalizing...")
    normDf, freqCol, goodCols = cleanAndNormalize(featDf)
    print(f"  After cleaning: {normDf.shape[0]} series x {normDf.shape[1]} features")
    print(f"  Dropped features (>30% NaN): {len(featDf.columns) - 1 - len(goodCols)}")

    print("\n[4] Cosine similarity analysis...")
    cosMat = analyzeCosineSimilarity(normDf, freqCol)

    print("\n[5] PCA analysis...")
    pcScores, pca = runPCA(normDf, freqCol)

    print("\n[6] Feature distribution by frequency...")
    featureDistributionByFreq(normDf, freqCol)

    print("\n[7] Within-frequency variation...")
    analyzeWithinFreqVariation(normDf, freqCol)

    print("\n[8] t-SNE visualization...")
    embedded = runTSNE(normDf, freqCol)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
