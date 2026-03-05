"""
실험 ID: dataProfiling/004
실험명: Rich R + Simple Rule — R벡터 풍부화와 단순 분기 규칙 동시 검증

목적:
- R벡터를 horizon별/잔차별로 확장하여 W·R 매핑 재시도 (방향 A)
- seasonality_strength 단일 임계값으로 DOT vs CES 분기 규칙 검증 (방향 C)
- 두 접근의 실용적 가치를 per-series oracle 대비 정량 비교

가설:
1. Horizon별 R(단기/중기/장기 OWA)로 확장하면 W·R 매핑 정확도 향상
2. 잔차 DNA로 R을 확장하면 "모델이 못 잡는 패턴"이 보일 것
3. seasonality_strength > threshold → CES, 아니면 DOT 라는 단순 규칙이 Ridge보다 나을 것

방법:
1. M4 Monthly 500개, DNA(W) + 4모델 예측 + horizon별 OWA(R)
2. 잔차 DNA 추출: 각 모델의 잔차에서 ACF/트렌드/분산 특성
3. Rich R (horizon 3구간 × 4모델 = 12차원) 기반 Ridge 재시도
4. seasonality_strength 단일 임계값 최적화 (grid search)
5. 전략 비교: DOT단일, Ridge(기존R), Ridge(richR), 단순규칙, oracle

결과 (실험 후 작성):
- Monthly 500개, 4모델, horizon 3구간, 잔차 DNA 추출
- Ridge basic R: accuracy 39.4%, OWA 0.848 → DOT(0.812)보다 악화 (-28.5%)
- Ridge rich R (horizon): accuracy 38.2%, OWA 0.845 → 역시 악화 (-25.8%)
  - R 풍부화로 Ridge 약간 개선되었으나 여전히 DOT보다 나쁨
- Horizon 분석: short/mid/long 모두 auto_ces가 최다 best (233/223/199)
- 잔차 DNA: DOT best일 때 res_abs_mean 333 vs not-best 604 → DOT가 잘 맞는 시리즈는 잔차가 작음 (당연)
  - CES best → res_acf1 낮음(0.52 vs 0.61), res_trend 음수 → CES가 자기상관/트렌드를 더 잘 잡음
- 단순규칙 (seas>0.60→CES): OWA 0.807, DOT 대비 -0.005 (oracle gap 4.0%)
- 다중특성 (seas>0.5 & trend<=0.1→CES): OWA 0.803, DOT 대비 -0.009
- Exhaustive 탐색 최적: acf_lag2<=0.374→CES: OWA 0.801, DOT 대비 -0.011 (oracle gap 9.0%)
- Oracle gap 전체: DOT 0.812 → Oracle 0.685 = 0.127 차이

결론:
- **R 풍부화(방향A) 효과 미미**: horizon별 분해해도 Ridge 정확도 거의 불변. 문제는 R 차원이 아님
- **단순 규칙(방향C)이 Ridge보다 확실히 우월**: 최선 규칙 OWA 0.801 vs Ridge 0.845
- **최적 단순 규칙**: acf_lag2<=0.374→CES, 나머지→DOT (oracle gap 9.0% 캡처)
- **그러나 9%만 캡처 = 실용성 부족**: oracle gap 0.127 중 0.011만 회수
- **핵심 발견**: 모델 간 OWA 차이가 작아서 "잘못 선택"의 페널티 > "올바른 선택"의 보상
  - DOT이 비최적인 61%에서도 평균 OWA ~0.85 수준 → 대체 모델과 차이 작음
- **잔차 인사이트**: CES가 best인 시리즈는 DOT 잔차에 자기상관이 높음(res_acf1 차이)

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings('ignore')

M4_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
FREQ = 'Monthly'
PERIOD = 12
HORIZON = 18
N_SAMPLE = 500
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


def extractResidualFeatures(trainY, pred, testY):
    residual = testY - pred
    feats = {}
    feats['res_mean'] = np.mean(residual)
    feats['res_std'] = np.std(residual)
    feats['res_abs_mean'] = np.mean(np.abs(residual))

    if len(residual) >= 3:
        from numpy.fft import rfft
        spec = np.abs(rfft(residual - np.mean(residual)))
        if len(spec) > 1:
            feats['res_spectral_peak'] = np.argmax(spec[1:]) + 1
            feats['res_spectral_energy'] = np.sum(spec[1:] ** 2)
        else:
            feats['res_spectral_peak'] = 0
            feats['res_spectral_energy'] = 0
    else:
        feats['res_spectral_peak'] = 0
        feats['res_spectral_energy'] = 0

    if len(residual) >= 2:
        feats['res_acf1'] = np.corrcoef(residual[:-1], residual[1:])[0, 1]
        if not np.isfinite(feats['res_acf1']):
            feats['res_acf1'] = 0
    else:
        feats['res_acf1'] = 0

    feats['res_trend'] = np.polyfit(np.arange(len(residual)), residual, 1)[0] if len(residual) >= 2 else 0
    if not np.isfinite(feats['res_trend']):
        feats['res_trend'] = 0

    return feats


def buildFullData(data, featDf, models, period, horizon):
    validIds = [sid for sid in featDf.index if sid in data]

    wRecords = []
    rBasic = []
    rHorizon = []
    rResidual = {}
    bestModels = []

    horizonBins = [(0, 6, 'short'), (6, 12, 'mid'), (12, 18, 'long')]

    for modelId in models:
        rResidual[modelId] = []

    for i, sid in enumerate(validIds):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(validIds)}...")

        trainY = data[sid]['train']
        testY = data[sid]['test']

        naive2Pred = naiveSeasonal(trainY, horizon, period)
        naiveSmape = computeSmape(testY, naive2Pred)
        naiveMase = computeMase(testY, naive2Pred, trainY, period)

        basicRow = {}
        horizonRow = {}
        bestOwa = 999
        bestM = models[0]

        for modelId in models:
            pred = forecastWithModel(trainY, horizon, period, modelId)

            smape = computeSmape(testY, pred)
            mase = computeMase(testY, pred, trainY, period)
            owa = computeOwa(smape, mase, naiveSmape, naiveMase)
            basicRow[modelId] = owa

            for start, end, label in horizonBins:
                subTest = testY[start:end]
                subPred = pred[start:end]
                subNaive = naive2Pred[start:end]

                subSmape = computeSmape(subTest, subPred)
                subMase = computeMase(subTest, subPred, trainY, period)
                subNaiveSmape = computeSmape(subTest, subNaive)
                subNaiveMase = computeMase(subTest, subNaive, trainY, period)
                subOwa = computeOwa(subSmape, subMase, subNaiveSmape, subNaiveMase)
                horizonRow[f'{modelId}_{label}'] = subOwa

            resFeat = extractResidualFeatures(trainY, pred, testY)
            rResidual[modelId].append(resFeat)

            if owa < bestOwa:
                bestOwa = owa
                bestM = modelId

        wRecords.append(featDf.loc[sid].to_dict())
        rBasic.append(basicRow)
        rHorizon.append(horizonRow)
        bestModels.append(bestM)

    wDf = pd.DataFrame(wRecords, index=validIds)
    rBasicDf = pd.DataFrame(rBasic, index=validIds)
    rHorizonDf = pd.DataFrame(rHorizon, index=validIds)

    rResDfs = {}
    for modelId in models:
        rResDfs[modelId] = pd.DataFrame(rResidual[modelId], index=validIds)

    return wDf, rBasicDf, rHorizonDf, rResDfs, bestModels


def cleanW(wDf):
    wClean = wDf.replace([np.inf, -np.inf], np.nan)
    nanRatio = wClean.isna().mean()
    goodCols = nanRatio[nanRatio < 0.3].index.tolist()
    wClean = wClean[goodCols].fillna(wClean[goodCols].median())
    return wClean, goodCols


def testRidgeBasic(wClean, rBasicDf, bestModels, models):
    print(f"\n{'=' * 80}")
    print(f"=== Ridge with Basic R (4-dim) ===")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X = scaler.fit_transform(wClean.values)
    y = np.array(bestModels)

    clf = RidgeClassifier(alpha=1.0)
    yPred = cross_val_predict(clf, X, y, cv=5)
    accuracy = np.mean(yPred == y)

    ridgeOwas = [rBasicDf.iloc[i][yPred[i]] for i in range(len(rBasicDf))]
    dotOwas = rBasicDf['dot'].values
    oracleOwas = rBasicDf.min(axis=1).values

    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  DOT OWA:    {np.mean(dotOwas):.4f}")
    print(f"  Ridge OWA:  {np.mean(ridgeOwas):.4f}")
    print(f"  Oracle OWA: {np.mean(oracleOwas):.4f}")
    return accuracy, np.mean(ridgeOwas)


def testRidgeRichR(wClean, rBasicDf, rHorizonDf, bestModels, models):
    print(f"\n{'=' * 80}")
    print(f"=== Ridge with Rich R (horizon-aware) ===")
    print(f"{'=' * 80}")

    scaler = StandardScaler()
    X = scaler.fit_transform(wClean.values)
    y = np.array(bestModels)

    horizonBestModels = []
    for i in range(len(rHorizonDf)):
        modelScores = {}
        for m in models:
            score = sum(rHorizonDf.iloc[i].get(f'{m}_{h}', 999) for h in ['short', 'mid', 'long'])
            modelScores[m] = score
        horizonBestModels.append(min(modelScores, key=modelScores.get))

    yHorizon = np.array(horizonBestModels)

    clf = RidgeClassifier(alpha=1.0)
    yPred = cross_val_predict(clf, X, yHorizon, cv=5)
    accuracy = np.mean(yPred == yHorizon)

    ridgeOwas = [rBasicDf.iloc[i][yPred[i]] for i in range(len(rBasicDf))]

    print(f"  Horizon-aware accuracy: {accuracy:.3f}")
    print(f"  Ridge(rich) OWA: {np.mean(ridgeOwas):.4f}")

    print(f"\n  Horizon-level analysis")
    for h in ['short', 'mid', 'long']:
        hCols = [f'{m}_{h}' for m in models]
        hBest = rHorizonDf[hCols].idxmin(axis=1).apply(lambda x: x.split('_')[0])
        dist = hBest.value_counts()
        print(f"    {h}: {dict(dist)}")

    return accuracy, np.mean(ridgeOwas)


def testResidualInsight(rResDfs, rBasicDf, models):
    print(f"\n{'=' * 80}")
    print(f"=== Residual DNA Analysis ===")
    print(f"{'=' * 80}")

    bestModelPerSeries = rBasicDf.idxmin(axis=1)

    for modelId in models:
        resDf = rResDfs[modelId]
        mask = bestModelPerSeries == modelId
        notMask = bestModelPerSeries != modelId

        if mask.sum() < 5 or notMask.sum() < 5:
            continue

        print(f"\n  {modelId} — residual DNA when {modelId} IS best vs NOT best")
        print(f"    {'Feature':>25} {'IsBest':>10} {'NotBest':>10} {'Diff':>10}")

        for col in resDf.columns:
            cleanCol = resDf[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            isBestMean = cleanCol[mask].mean()
            notBestMean = cleanCol[notMask].mean()
            diff = isBestMean - notBestMean
            print(f"    {col:>25} {isBestMean:>10.4f} {notBestMean:>10.4f} {diff:>+10.4f}")


def testSimpleRule(wDf, rBasicDf, models):
    print(f"\n{'=' * 80}")
    print(f"=== Simple Rule: seasonality_strength threshold ===")
    print(f"{'=' * 80}")

    if 'seasonality_strength' not in wDf.columns:
        print("  seasonality_strength not found")
        return 0, 0

    seasStrength = wDf['seasonality_strength'].values
    dotOwas = rBasicDf['dot'].values
    cesOwas = rBasicDf['auto_ces'].values
    oracleOwas = rBasicDf.min(axis=1).values

    dotMean = np.mean(dotOwas)
    oracleMean = np.mean(oracleOwas)

    print(f"  DOT baseline: {dotMean:.4f}")
    print(f"  Oracle:       {oracleMean:.4f}")

    bestThresh = 0
    bestOwa = dotMean
    bestModel2 = 'auto_ces'

    print(f"\n  --- DOT vs CES switching ---")
    print(f"  {'Threshold':>10} {'N_DOT':>8} {'N_CES':>8} {'OWA':>10} {'vs DOT':>10}")

    for thresh in np.arange(0.0, 1.01, 0.05):
        selected = np.where(seasStrength > thresh, cesOwas, dotOwas)
        owa = np.mean(selected)
        nDot = np.sum(seasStrength <= thresh)
        nCes = np.sum(seasStrength > thresh)
        diff = owa - dotMean
        marker = " <-- best" if owa < bestOwa else ""
        print(f"  {thresh:>10.2f} {nDot:>8} {nCes:>8} {owa:>10.4f} {diff:>+10.4f}{marker}")
        if owa < bestOwa:
            bestOwa = owa
            bestThresh = thresh
            bestModel2 = 'auto_ces'

    print(f"\n  --- DOT vs 4Theta switching ---")
    thetaOwas = rBasicDf['four_theta'].values

    for thresh in np.arange(0.0, 1.01, 0.05):
        selected = np.where(seasStrength <= thresh, thetaOwas, dotOwas)
        owa = np.mean(selected)
        nTheta = np.sum(seasStrength <= thresh)
        nDot = np.sum(seasStrength > thresh)
        diff = owa - dotMean
        if owa < bestOwa:
            bestOwa = owa
            bestThresh = thresh
            bestModel2 = 'four_theta'

    print(f"\n  --- Multi-feature rule: seasonality + trend ---")
    if 'trend_strength' in wDf.columns:
        trendStrength = wDf['trend_strength'].values

        bestMultiOwa = dotMean
        bestMultiRule = "DOT only"

        for sTh in np.arange(0.1, 0.9, 0.1):
            for tTh in np.arange(0.1, 0.9, 0.1):
                selected = np.copy(dotOwas)
                highSeas = seasStrength > sTh
                lowTrend = trendStrength <= tTh
                selected[highSeas & lowTrend] = cesOwas[highSeas & lowTrend]

                owa = np.mean(selected)
                if owa < bestMultiOwa:
                    bestMultiOwa = owa
                    bestMultiRule = f"seas>{sTh:.1f} & trend<={tTh:.1f} → CES, else DOT"

        print(f"  Best multi-rule: {bestMultiRule}")
        print(f"  Multi-rule OWA: {bestMultiOwa:.4f} (vs DOT {dotMean:.4f}, diff {bestMultiOwa - dotMean:+.4f})")

    print(f"\n  === Best Simple Rule Summary ===")
    print(f"  Rule: seas_strength > {bestThresh:.2f} → {bestModel2}, else DOT")
    print(f"  Rule OWA: {bestOwa:.4f}")
    print(f"  DOT OWA:  {dotMean:.4f}")
    print(f"  Oracle:   {oracleMean:.4f}")

    totalGap = dotMean - oracleMean
    captured = dotMean - bestOwa
    if totalGap > 1e-6:
        captureRate = captured / totalGap * 100
    else:
        captureRate = 0
    print(f"  Gap captured: {captured:.4f} ({captureRate:.1f}% of oracle gap)")

    return bestThresh, bestOwa


def testAllModelsRule(wDf, rBasicDf, models):
    print(f"\n{'=' * 80}")
    print(f"=== Exhaustive Single-Feature Rules (all features × all model pairs) ===")
    print(f"{'=' * 80}")

    wClean, goodCols = cleanW(wDf)
    dotOwas = rBasicDf['dot'].values
    dotMean = np.mean(dotOwas)
    oracleOwas = rBasicDf.min(axis=1).values
    oracleMean = np.mean(oracleOwas)

    bestRule = None
    bestRuleOwa = dotMean

    for feat in goodCols:
        vals = wClean[feat].values
        percentiles = np.percentile(vals, np.arange(10, 91, 10))

        for altModel in models:
            if altModel == 'dot':
                continue
            altOwas = rBasicDf[altModel].values

            for thresh in percentiles:
                for direction in ['>', '<=']:
                    if direction == '>':
                        mask = vals > thresh
                    else:
                        mask = vals <= thresh

                    selected = np.copy(dotOwas)
                    selected[mask] = altOwas[mask]
                    owa = np.mean(selected)

                    if owa < bestRuleOwa:
                        bestRuleOwa = owa
                        bestRule = {
                            'feat': feat,
                            'thresh': thresh,
                            'direction': direction,
                            'altModel': altModel,
                            'nSwitch': mask.sum(),
                            'owa': owa
                        }

    if bestRule:
        captured = dotMean - bestRuleOwa
        totalGap = dotMean - oracleMean
        captureRate = captured / totalGap * 100 if totalGap > 1e-6 else 0

        print(f"\n  Best single-feature rule found")
        print(f"  Rule: {bestRule['feat']} {bestRule['direction']} {bestRule['thresh']:.3f} → {bestRule['altModel']}, else DOT")
        print(f"  N switched: {bestRule['nSwitch']}/{len(dotOwas)}")
        print(f"  Rule OWA:  {bestRuleOwa:.4f}")
        print(f"  DOT OWA:   {dotMean:.4f}")
        print(f"  Oracle:    {oracleMean:.4f}")
        print(f"  Captured:  {captured:.4f} ({captureRate:.1f}% of oracle gap)")
    else:
        print("  No rule beats DOT")

    return bestRule, bestRuleOwa


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E004: Rich R + Simple Rule")
    print(f"  Freq={FREQ}, Period={PERIOD}, Horizon={HORIZON}, N={N_SAMPLE}")
    print("=" * 70)

    t0 = time.time()

    print("\n[1] Loading M4 data...")
    data = loadM4WithTest(FREQ, N_SAMPLE, SEED)
    print(f"  Loaded: {len(data)} series")

    print("\n[2] Extracting DNA features...")
    featDf = extractFeatures(data, PERIOD)
    print(f"  W matrix: {featDf.shape}")

    print("\n[3] Computing full R vectors (basic + horizon + residual)...")
    wDf, rBasicDf, rHorizonDf, rResDfs, bestModels = buildFullData(
        data, featDf, MODELS, PERIOD, HORIZON
    )
    print(f"  R basic: {rBasicDf.shape}, R horizon: {rHorizonDf.shape}")

    wClean, goodCols = cleanW(wDf)

    print("\n[4] Ridge with basic R (E003 재현)...")
    accBasic, owaBasic = testRidgeBasic(wClean, rBasicDf, bestModels, MODELS)

    print("\n[5] Ridge with rich R (horizon-aware)...")
    accRich, owaRich = testRidgeRichR(wClean, rBasicDf, rHorizonDf, bestModels, MODELS)

    print("\n[6] Residual DNA analysis...")
    testResidualInsight(rResDfs, rBasicDf, MODELS)

    print("\n[7] Simple rule: seasonality threshold...")
    bestThresh, ruleOwa = testSimpleRule(wDf, rBasicDf, MODELS)

    print("\n[8] Exhaustive single-feature rules...")
    bestRule, bestRuleOwa = testAllModelsRule(wDf, rBasicDf, MODELS)

    dotOwa = np.mean(rBasicDf['dot'].values)
    oracleOwa = np.mean(rBasicDf.min(axis=1).values)

    print(f"\n{'=' * 70}")
    print(f"=== FINAL COMPARISON ===")
    print(f"{'=' * 70}")
    print(f"  {'Strategy':>30} {'OWA':>10} {'vs DOT':>10} {'Oracle%':>10}")
    print(f"  {'-' * 65}")
    for name, owa in [
        ('DOT (baseline)', dotOwa),
        ('Ridge basic R', owaBasic),
        ('Ridge rich R', owaRich),
        ('Seasonality rule', ruleOwa),
        ('Best single-feat rule', bestRuleOwa),
        ('Oracle (per-series)', oracleOwa),
    ]:
        diff = owa - dotOwa
        totalGap = dotOwa - oracleOwa
        captured = (dotOwa - owa) / totalGap * 100 if totalGap > 1e-6 else 0
        print(f"  {name:>30} {owa:>10.4f} {diff:>+10.4f} {captured:>9.1f}%")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
