"""
실험 ID: experienceIndex/006
실험명: DNA Importance — 65차원 특성 중요도 분석

========================================================================
배경
========================================================================

E002~E005에서 65d DNA + kNN이 모든 빈도에서 작동함을 확인.
그러나 65개 특성이 모두 동일하게 중요한 것은 아닐 것이다.
어떤 특성 그룹이 가장 기여하는지 알면:

1. DNA 차원 축소 가능성 (불필요한 특성 제거 → 더 깨끗한 검색)
2. 향후 DNA 확장 방향 결정 (중요한 카테고리를 더 풍부하게)
3. kNN 가중치 최적화 (중요 특성에 가중치)

========================================================================

목적:
- Permutation importance로 특성 그룹별 중요도 측정
- 카테고리별 중요도 순위: trend, seasonality, autocorrelation, ...
- 하위 50% 특성 제거 시 성능 변화

가설:
1. seasonality + autocorrelation이 가장 중요 (계절성이 모델 선택의 핵심)
2. basic statistics(mean, std 등)는 중요도 낮음
3. 상위 30개 특성만으로도 65개와 비슷한 성능

방법:
1. M4 Monthly 5000개, 8후보로 경험 DB 구축
2. 특성 그룹별 permutation importance (그룹 단위로 셔플)
3. 개별 특성 importance (상위/하위 순위)
4. 차원 축소 실험 (상위 N개만 사용)

결과 (실험 후 작성):
- 5000 시리즈, 기준선 DOT 0.8819, kNN(65d) 0.8323 (+5.63%)

- 그룹 중요도 (Permutation, 5회 평균):
  #1 forecastability 0.48, #2 seasonality 0.41, #3 misc 0.40
  #9 basic 0.00, #10 nonlinearity 0.00, #11 trend 0.00
  → trend와 basic statistics는 kNN 검색에 무용

- 개별 특성 Top 5:
  seasonal_peak_position 1.241, forecastability 0.874, spectral_entropy 0.874
  conditional_heteroscedasticity 0.799, longest_decreasing_run 0.785

- 차원 축소:
  | TopN | OWA    | DOT대비  | 기준대비  |
  |------|--------|----------|----------|
  | 10   | 0.8601 | +2.48%   | -3.34%   |
  | 20   | 0.8533 | +3.24%   | -2.53%   |
  | 30   | 0.8418 | +4.55%   | -1.14%   |
  | 40   | 0.8286 | +6.05%   | +0.44%   |
  | 50   | 0.8228 | +6.70%   | +1.13%   |
  | 65   | 0.8323 | +5.63%   | ±0.00%   |

결론:
- 가설 1 (seasonality+autocorrelation 최중요) → 부분 확인. seasonality #2, autocorrelation #7
- 가설 2 (basic statistics 불필요) → 확인! basic 0.00, trend 0.00
- 가설 3 (30개로 충분) → 기각. 30개는 -1.14%. 50개가 최적(+1.13%)
- 핵심 발견: Top 50 특성이 전체 65개보다 +1.13% 좋다! 하위 15개가 노이즈
- seasonal_peak_position이 압도적 1위 — 계절 피크 위치가 모델 선택의 핵심 단서
- forecastability, spectral_entropy가 2~3위 — "예측 가능성"이 모델 선택을 좌우
- trend 관련 특성은 전부 무용 — 트렌드 자체보다 "어떤 종류의 변동"인지가 중요
- 실무 적용: DNA에서 하위 15개 특성 제거 → 더 깨끗한 검색

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

FEATURE_GROUPS = {
    'basic': ['cv', 'iqr', 'kurtosis', 'length', 'max', 'mean', 'median', 'min', 'skewness', 'std'],
    'trend': ['trend_curvature', 'trend_direction', 'trend_linearity', 'trend_slope', 'trend_strength'],
    'seasonality': ['multi_seasonality', 'seasonal_peak', 'seasonal_period', 'seasonal_trough',
                    'seasonality_strength', 'seasonality_strength_2', 'seasonality_strength_3',
                    'seasonal_peak_2'],
    'autocorrelation': ['acf1', 'acf10_sum', 'acf_first_zero', 'arch_lm', 'ljung_box_stat',
                        'pacf1', 'pacf5_sum', 'seasonal_acf1'],
    'nonlinearity': ['approximate_entropy', 'hurst_exponent', 'lz_complexity',
                     'nonlinearity', 'sample_entropy'],
    'stability': ['crossing_points', 'flat_spots', 'lumpiness', 'max_kl_shift', 'stability'],
    'stationarity': ['diff_mean_ratio', 'diff_std_ratio', 'kpss_stat_approx', 'pp_stat_approx'],
    'forecastability': ['forecastability', 'mean_absolute_change', 'mean_change_rate',
                        'prediction_interval_width', 'snr'],
    'intermittency': ['adi', 'cv_squared', 'longest_zero_run', 'zero_count', 'zero_proportion'],
    'volatility': ['garch_alpha', 'garch_beta', 'max_drawdown', 'volatility_clustering',
                   'volatility_ratio'],
    'misc': ['longest_down_run', 'longest_up_run', 'n_peaks', 'n_troughs', 'peak_to_trough_time'],
}


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


def evaluateKnn(dnaMatrixNorm, experiences, validIdx, testIdx, trainPool, k=50):
    dbDnas = dnaMatrixNorm[trainPool]
    expArray = np.array(experiences)
    dbExps = expArray[trainPool]

    knnOwas = []
    dotOwas = []
    for i in testIdx:
        sims = dbDnas @ dnaMatrixNorm[i]
        topIdx = np.argsort(sims)[-k:]
        topSims = np.maximum(sims[topIdx], 0)
        wSum = np.sum(topSims)
        if wSum == 0:
            topSims = np.ones(k)
            wSum = k

        candidateScores = {}
        for cKey in dbExps[topIdx[0]]['owas']:
            wOwa = 0
            wTotal = 0
            for j, idx in enumerate(topIdx):
                nOwa = dbExps[idx]['owas'].get(cKey, 99.0)
                if nOwa < 99.0:
                    wOwa += topSims[j] * nOwa
                    wTotal += topSims[j]
            if wTotal > 0:
                candidateScores[cKey] = wOwa / wTotal

        if not candidateScores:
            continue
        bestKey = min(candidateScores, key=candidateScores.get)
        knnOwas.append(experiences[i]['owas'].get(bestKey, experiences[i]['dotOwa']))
        dotOwas.append(experiences[i]['dotOwa'])

    return np.mean(knnOwas), np.mean(dotOwas)


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

    from vectrix.engine.tsfeatures import TSFeatureExtractor

    startTime = time.time()

    print("=" * 70)
    print("experienceIndex/006: DNA Importance")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series")

    extractor = TSFeatureExtractor()
    allDnas = []
    featureNames = None

    print("\n--- DNA 추출 ---")
    for idx, (sid, trainY, testY) in enumerate(series):
        features = extractor.extract(trainY, period=PERIOD)
        if featureNames is None:
            featureNames = sorted(features.keys())
        vec = np.array([features.get(f, 0.0) for f in featureNames])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        allDnas.append(vec)

    dnaMatrix = np.array(allDnas)
    print(f"  DNA: {dnaMatrix.shape[1]}d, {len(featureNames)} features")

    featureGroupIdx = {}
    for groupName, groupFeatures in FEATURE_GROUPS.items():
        indices = [featureNames.index(f) for f in groupFeatures if f in featureNames]
        if indices:
            featureGroupIdx[groupName] = indices
    print(f"  Feature groups: {len(featureGroupIdx)}")
    for g, idx in featureGroupIdx.items():
        print(f"    {g:>20}: {len(idx)} features")

    print(f"\n--- 경험 수집 ---")
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

        dotOwa = candidateOwas.get('dot_raw', 99.0)
        experiences.append({'owas': candidateOwas, 'dotOwa': dotOwa})

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - startTime
            print(f"  경험: {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

    validMask = np.array([e['dotOwa'] < 99.0 for e in experiences])
    validIdx = np.where(validMask)[0]
    print(f"  유효: {len(validIdx)}")

    testSize = 500
    np.random.seed(SEED + 1)
    allValid = validIdx.copy()
    np.random.shuffle(allValid)
    testIdx = allValid[:testSize]
    trainPool = allValid[testSize:]

    def normalizeDna(matrix):
        m = np.mean(matrix, axis=0)
        s = np.std(matrix, axis=0)
        s = np.where(s > 0, s, 1.0)
        normed = (matrix - m) / s
        n = np.linalg.norm(normed, axis=1, keepdims=True)
        n = np.where(n > 0, n, 1.0)
        return normed / n

    dnaMatrixNorm = normalizeDna(dnaMatrix)

    baseKnn, baseDot = evaluateKnn(dnaMatrixNorm, experiences, validIdx, testIdx, trainPool)
    baseImpr = (baseDot - baseKnn) / baseDot * 100
    print(f"\n  기준선: DOT {baseDot:.4f}, kNN {baseKnn:.4f} ({baseImpr:+.2f}%)")

    print("\n" + "=" * 70)
    print("--- Phase 1: 그룹별 Permutation Importance ---")
    print("=" * 70)
    print(f"\n  {'그룹':>20} {'Perm OWA':>10} {'성능저하':>10} {'중요도':>10}")
    print("  " + "-" * 55)

    groupImportance = {}
    N_PERM = 5

    for groupName, indices in sorted(featureGroupIdx.items()):
        permKnns = []
        for perm in range(N_PERM):
            permMatrix = dnaMatrix.copy()
            rng = np.random.RandomState(SEED + 100 + perm)
            for fIdx in indices:
                permMatrix[:, fIdx] = rng.permutation(permMatrix[:, fIdx])
            permNorm = normalizeDna(permMatrix)
            permKnn, _ = evaluateKnn(permNorm, experiences, validIdx, testIdx, trainPool)
            permKnns.append(permKnn)

        avgPermKnn = np.mean(permKnns)
        degradation = (avgPermKnn - baseKnn) / baseKnn * 100
        importance = max(0, degradation)
        groupImportance[groupName] = importance
        print(f"  {groupName:>20} {avgPermKnn:>10.4f} {degradation:>+10.2f}% {importance:>10.2f}")

    print(f"\n  --- 중요도 순위 ---")
    sortedGroups = sorted(groupImportance.items(), key=lambda x: -x[1])
    for rank, (g, imp) in enumerate(sortedGroups, 1):
        nFeatures = len(featureGroupIdx[g])
        print(f"  #{rank}: {g:>20} ({nFeatures}개) — 중요도 {imp:.2f}")

    print("\n" + "=" * 70)
    print("--- Phase 2: 개별 특성 Top/Bottom ---")
    print("=" * 70)

    featureImportance = {}
    for fIdx, fName in enumerate(featureNames):
        permKnns = []
        for perm in range(3):
            permMatrix = dnaMatrix.copy()
            rng = np.random.RandomState(SEED + 200 + perm)
            permMatrix[:, fIdx] = rng.permutation(permMatrix[:, fIdx])
            permNorm = normalizeDna(permMatrix)
            permKnn, _ = evaluateKnn(permNorm, experiences, validIdx, testIdx, trainPool)
            permKnns.append(permKnn)
        avgPermKnn = np.mean(permKnns)
        degradation = (avgPermKnn - baseKnn) / baseKnn * 100
        featureImportance[fName] = max(0, degradation)

    sortedFeatures = sorted(featureImportance.items(), key=lambda x: -x[1])

    print(f"\n  --- Top 10 가장 중요한 특성 ---")
    for rank, (f, imp) in enumerate(sortedFeatures[:10], 1):
        print(f"  #{rank:>2}: {f:>30} — {imp:.3f}")

    print(f"\n  --- Bottom 10 가장 불필요한 특성 ---")
    for rank, (f, imp) in enumerate(sortedFeatures[-10:], len(sortedFeatures) - 9):
        print(f"  #{rank:>2}: {f:>30} — {imp:.3f}")

    print("\n" + "=" * 70)
    print("--- Phase 3: 차원 축소 실험 ---")
    print("=" * 70)

    topNs = [10, 20, 30, 40, 50, len(featureNames)]
    print(f"\n  {'TopN':>6} {'OWA':>8} {'DOT대비':>10} {'기준대비':>10}")
    print("  " + "-" * 38)

    for topN in topNs:
        topFeatureNames = [f for f, _ in sortedFeatures[:topN]]
        topIndices = [featureNames.index(f) for f in topFeatureNames]
        reducedMatrix = dnaMatrix[:, topIndices]
        reducedNorm = normalizeDna(reducedMatrix)
        redKnn, redDot = evaluateKnn(reducedNorm, experiences, validIdx, testIdx, trainPool)
        imprVsDot = (redDot - redKnn) / redDot * 100
        imprVsBase = (baseKnn - redKnn) / baseKnn * 100
        label = "(전체)" if topN == len(featureNames) else ""
        print(f"  {topN:>6} {redKnn:>8.4f} {imprVsDot:>+10.2f}% {imprVsBase:>+10.2f}% {label}")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  시리즈: {len(series)}, 유효: {len(validIdx)}")
    print(f"  기준선 DOT: {baseDot:.4f}, kNN(65d): {baseKnn:.4f} ({baseImpr:+.2f}%)")
    print(f"\n  그룹 중요도 Top 3:")
    for rank, (g, imp) in enumerate(sortedGroups[:3], 1):
        print(f"    #{rank}: {g} — {imp:.2f}")
    print(f"\n  개별 특성 Top 5:")
    for rank, (f, imp) in enumerate(sortedFeatures[:5], 1):
        print(f"    #{rank}: {f} — {imp:.3f}")
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 70)
