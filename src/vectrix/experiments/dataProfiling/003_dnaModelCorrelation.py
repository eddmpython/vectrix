"""
실험 ID: dataProfiling/003
실험명: DNA-Model Correlation — W벡터(DNA)와 R벡터(예측행동) 상관 분석

목적:
- DNA 특성(W)과 모델별 OWA(R) 사이의 직접 상관을 측정
- "어떤 DNA 특성이 모델 선택을 가장 잘 예측하는가" 확인
- W·R 내적이 의미 있는 매핑 함수를 형성하는지 검증

가설:
1. 특정 DNA 특성은 특정 모델의 OWA와 강한 상관이 있을 것
2. DNA 상위 5~10개 특성만으로도 최적 모델 예측이 가능할 것
3. 단순 선형 매핑(W→R)으로도 per-series oracle의 50% 이상 캡처 가능

방법:
1. M4 Monthly 300개 시리즈, DNA 추출 + 4개 모델 OWA 계산
2. DNA 65개 특성 × 4개 모델 OWA → 상관 행렬 (Pearson + Spearman)
3. 모델별 "가장 예측력 있는 DNA 특성" Top-5 확인
4. 단순 선형 모델 (Ridge)로 DNA → 최적모델 예측 시도
5. Ridge 예측 기반 OWA vs DOT단일 vs per-series oracle 비교

결과 (실험 후 작성):
- Monthly 300개, W(65 DNA) × R(4 model OWA) 상관 분석
- 모델별 핵심 상관: 4Theta↔seasonality_strength=0.40, AutoETS↔seasonality_strength=0.53 (p<0.001)
  - DOT/CES는 계절성에 덜 민감(0.14~0.20) → 범용성 높음
  - 4Theta/ETS는 계절성 강하면 확실히 나빠짐
- Best model 분포: DOT 39.3%, CES 26.3%, 4Theta 20.3%, ETS 14.0%
- Ridge 5-fold CV accuracy: 36.7% (랜덤 25% 대비 +12% 불과)
- Ridge OWA 0.833 > DOT단일 0.795 → DOT보다 악화 (Oracle gap -30.9%)
- Feature importance (Ridge coef): ljung_box_stat, max, length, mean_abs_change, diff_std_ratio

결론:
- **W·R 상관은 존재** — seasonality_strength가 4Theta/ETS 성능의 핵심 예측 변수
- **선형 매핑은 실패** — Ridge 36.7% 정확도, DOT보다 악화
- **원인**: 모델 간 성능 차이가 작음 (DOT이 비최적인 곳에서도 나쁘지 않음)
- W에서 R로 가는 매핑은 비선형이거나, 더 적은 축(계절성 유/무 하나만)으로 충분할 수 있음
- agipath 비유: W·R 내적 원리는 맞지만, "단어(시리즈) 수"와 "맥락 풍부함"이 부족

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings('ignore')

M4_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
FREQ = 'Monthly'
PERIOD = 12
HORIZON = 18
N_SAMPLE = 300
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
    lastSeason = trainY[-(period):]
    reps = (steps // period) + 1
    return np.tile(lastSeason, reps)[:steps]


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


def buildWR(data, featDf, models, period, horizon):
    validIds = [sid for sid in featDf.index if sid in data]

    wRecords = []
    rRecords = []
    bestModels = []

    for i, sid in enumerate(validIds):
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(validIds)}...")

        trainY = data[sid]['train']
        testY = data[sid]['test']

        naive2Pred = naiveSeasonal(trainY, horizon, period)
        naiveSmape = computeSmape(testY, naive2Pred)
        naiveMase = computeMase(testY, naive2Pred, trainY, period)

        rRow = {}
        bestOwa = 999
        bestM = models[0]

        for modelId in models:
            pred = forecastWithModel(trainY, horizon, period, modelId)
            smape = computeSmape(testY, pred)
            mase = computeMase(testY, pred, trainY, period)
            owa = computeOwa(smape, mase, naiveSmape, naiveMase)
            rRow[modelId] = owa
            if owa < bestOwa:
                bestOwa = owa
                bestM = modelId

        wRecords.append(featDf.loc[sid].to_dict())
        rRecords.append(rRow)
        bestModels.append(bestM)

    wDf = pd.DataFrame(wRecords, index=validIds)
    rDf = pd.DataFrame(rRecords, index=validIds)

    return wDf, rDf, bestModels


def analyzeCorrelation(wDf, rDf, models):
    wClean = wDf.replace([np.inf, -np.inf], np.nan)
    nanRatio = wClean.isna().mean()
    goodCols = nanRatio[nanRatio < 0.3].index.tolist()
    wClean = wClean[goodCols].fillna(wClean[goodCols].median())

    print(f"\n{'=' * 80}")
    print(f"=== DNA-Model Correlation (Spearman) ===")
    print(f"{'=' * 80}")

    corrResults = {}

    for modelId in models:
        modelOwa = rDf[modelId].values
        corrs = {}
        for feat in goodCols:
            rho, pval = spearmanr(wClean[feat].values, modelOwa)
            if np.isfinite(rho):
                corrs[feat] = (rho, pval)

        sortedCorrs = sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)
        corrResults[modelId] = sortedCorrs

        print(f"\n  {modelId} — Top 10 correlated DNA features")
        print(f"  {'Feature':>30} {'Spearman':>10} {'p-value':>10} {'Direction':>10}")
        print(f"  {'-' * 65}")
        for feat, (rho, pval) in sortedCorrs[:10]:
            direction = "낮을수록좋음" if rho > 0 else "높을수록좋음"
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {feat:>30} {rho:>+10.3f} {pval:>10.4f} {direction:>10} {sig}")

    return corrResults, wClean, goodCols


def analyzeModelDifferentiation(wClean, rDf, models):
    print(f"\n{'=' * 80}")
    print(f"=== Model Differentiation — Which DNA features distinguish model preference? ===")
    print(f"{'=' * 80}")

    bestModelPerSeries = rDf.idxmin(axis=1)
    modelCounts = bestModelPerSeries.value_counts()
    print(f"\n  Best model distribution")
    for m, cnt in modelCounts.items():
        print(f"    {m:>15}: {cnt} ({cnt/len(bestModelPerSeries)*100:.1f}%)")

    print(f"\n  Feature means by best-model group")
    topFeats = []
    for feat in wClean.columns:
        groups = {}
        for m in models:
            mask = bestModelPerSeries == m
            if mask.sum() >= 5:
                groups[m] = wClean.loc[mask, feat].mean()
        if len(groups) >= 2:
            vals = list(groups.values())
            spread = max(vals) - min(vals)
            topFeats.append((feat, spread, groups))

    topFeats.sort(key=lambda x: x[1], reverse=True)

    print(f"\n  {'Feature':>30}", end='')
    for m in models:
        print(f" {m:>12}", end='')
    print(f" {'Spread':>10}")
    print(f"  {'-' * (30 + 12 * len(models) + 10)}")

    for feat, spread, groups in topFeats[:15]:
        print(f"  {feat:>30}", end='')
        for m in models:
            val = groups.get(m, float('nan'))
            print(f" {val:>12.3f}", end='')
        print(f" {spread:>10.3f}")


def trainPredictor(wClean, rDf, models, bestModels):
    print(f"\n{'=' * 80}")
    print(f"=== Ridge Classifier: DNA → Best Model Prediction ===")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X = scaler.fit_transform(wClean.values)
    y = np.array(bestModels)

    clf = RidgeClassifier(alpha=1.0)
    yPred = cross_val_predict(clf, X, y, cv=5)

    accuracy = np.mean(yPred == y)
    print(f"\n  5-fold CV accuracy: {accuracy:.3f}")

    print(f"\n  Confusion matrix")
    uniqueModels = sorted(set(y))
    print(f"  {'Actual\\Pred':>15}", end='')
    for m in uniqueModels:
        print(f" {m:>12}", end='')
    print()
    for actual in uniqueModels:
        print(f"  {actual:>15}", end='')
        for pred in uniqueModels:
            cnt = np.sum((y == actual) & (yPred == pred))
            print(f" {cnt:>12}", end='')
        print()

    ridgeOwas = []
    dotOwas = []
    oracleOwas = []

    for i in range(len(rDf)):
        ridgeOwas.append(rDf.iloc[i][yPred[i]])
        dotOwas.append(rDf.iloc[i]['dot'])
        oracleOwas.append(rDf.iloc[i].min())

    print(f"\n  Strategy OWA comparison")
    print(f"    DOT (uniform):      {np.mean(dotOwas):.4f}")
    print(f"    Ridge (DNA→model):  {np.mean(ridgeOwas):.4f}")
    print(f"    Oracle (per-series): {np.mean(oracleOwas):.4f}")

    dotOwa = np.mean(dotOwas)
    ridgeOwa = np.mean(ridgeOwas)
    oracleOwa = np.mean(oracleOwas)

    totalGap = dotOwa - oracleOwa
    captured = dotOwa - ridgeOwa
    if totalGap > 1e-6:
        captureRate = captured / totalGap * 100
    else:
        captureRate = 0
    print(f"\n  Oracle gap: {totalGap:.4f}")
    print(f"  Ridge captured: {captured:.4f} ({captureRate:.1f}% of oracle gap)")

    clf.fit(X, y)
    print(f"\n  Feature importance (Ridge coef magnitude, top 10)")
    coefMag = np.mean(np.abs(clf.coef_), axis=0)
    topIdx = np.argsort(coefMag)[::-1][:10]
    for rank, idx in enumerate(topIdx):
        print(f"    {rank+1}. {wClean.columns[idx]:>30}: {coefMag[idx]:.4f}")

    return clf, scaler


def crossFreqValidation(models, period_map, horizon_map, nSample=100):
    print(f"\n{'=' * 80}")
    print(f"=== Cross-Frequency Validation ===")
    print(f"{'=' * 80}")

    for freq in ['Yearly', 'Quarterly', 'Daily']:
        period = period_map[freq]
        horizon = horizon_map[freq]
        print(f"\n  --- {freq} (period={period}, horizon={horizon}, n={nSample}) ---")

        data = loadM4WithTest(freq, nSample, SEED)
        if len(data) < 20:
            print(f"    Skipped (only {len(data)} series)")
            continue

        featDf = extractFeatures(data, period)
        print(f"    Features: {featDf.shape}")

        wDf, rDf, bestModels = buildWR(data, featDf, models, period, horizon)

        bestModelSeries = rDf.idxmin(axis=1)
        modelCounts = bestModelSeries.value_counts()
        print(f"    Best model distribution")
        for m, cnt in modelCounts.items():
            print(f"      {m:>15}: {cnt} ({cnt/len(bestModelSeries)*100:.1f}%)")

        dotOwa = rDf['dot'].mean()
        oracleOwa = rDf.min(axis=1).mean()
        print(f"    DOT OWA: {dotOwa:.4f}, Oracle OWA: {oracleOwa:.4f}, Gap: {dotOwa - oracleOwa:.4f}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E003: DNA-Model Correlation (W·R Analysis)")
    print(f"  Freq={FREQ}, Period={PERIOD}, Horizon={HORIZON}, N={N_SAMPLE}")
    print("=" * 70)

    t0 = time.time()

    print("\n[1] Loading M4 data...")
    data = loadM4WithTest(FREQ, N_SAMPLE, SEED)
    print(f"  Loaded: {len(data)} series")

    print("\n[2] Extracting DNA features (W vectors)...")
    featDf = extractFeatures(data, PERIOD)
    print(f"  W matrix: {featDf.shape}")

    print("\n[3] Computing model OWA per series (R vectors)...")
    wDf, rDf, bestModels = buildWR(data, featDf, MODELS, PERIOD, HORIZON)
    print(f"  R matrix: {rDf.shape}")

    print("\n[4] Analyzing W-R correlation...")
    corrResults, wClean, goodCols = analyzeCorrelation(wDf, rDf, MODELS)

    print("\n[5] Model differentiation by DNA...")
    analyzeModelDifferentiation(wClean, rDf, MODELS)

    print("\n[6] Ridge classifier: DNA → best model...")
    clf, scaler = trainPredictor(wClean, rDf, MODELS, bestModels)

    print("\n[7] Cross-frequency validation...")
    periodMap = {'Yearly': 1, 'Quarterly': 4, 'Daily': 7}
    horizonMap = {'Yearly': 6, 'Quarterly': 8, 'Daily': 14}
    crossFreqValidation(MODELS, periodMap, horizonMap, nSample=100)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
