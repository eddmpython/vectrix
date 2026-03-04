"""
==============================================================================
Experiment ID: modelCreation/031
Experiment: FFORMA-Style Meta-Learning Ensemble
==============================================================================

Purpose:
- Achieve M4 Competition #1 (OWA 0.821) via meta-learning model combination
- FFORMA (M4 #2, OWA 0.838) selects from 9 base models using XGBoost
- We have 8+ strong base models + 65+ DNA features — same architecture, better components

Hypothesis:
1. Meta-learning ensemble with per-series optimal weights should beat single DOT-Hybrid (0.885)
2. Target: AVG OWA <= 0.840 (FFORMA-level, M4 #2)
3. Frequency-specific model pools should outperform fixed pool
4. DNA features contain enough signal to predict optimal model weights

Method:
1. Phase 1: Collect oracle data — run all candidate models on M4 2000/group
   - Models: dot, auto_ces, four_theta, auto_ets, auto_arima, dtsf, esn, auto_mstl
   - For each series: record DNA features + per-model OWA
2. Phase 2: Train XGBoost meta-learner (features → optimal model weights)
   - Multi-output regression: predict OWA for each model
   - Weight = softmax(1/predicted_OWA)
3. Phase 3: Evaluate meta-ensemble on held-out data
   - 5-fold cross-validation within each frequency group

Results:
  Phase 1 (8 models × 7273 series):
    Oracle best OWA = 0.6618 (always pick per-series best model)
    DOT-only OWA = 0.9462 (per-series average)
    Gap = 0.2844 — massive room for improvement

  Oracle best model counts:
    dot 24.0%, esn 16.9%, four_theta 15.8%, auto_ces 11.4%,
    theta 9.3%, auto_ets 9.2%, dtsf 7.8%, auto_arima 5.6%

  Phase 2 (GBR meta-learner, 5-fold OOF):
    GBR R2: dot=0.067, auto_ces=0.114, four_theta=0.372, theta=0.579

  Phase 3 (M4-style OWA):
    Strategy         Ye     Qu     Mo     We      Da      Ho     AVG
    oracle_best      0.568  0.659  0.676  0.636   0.732   0.422  0.615
    dot_only         0.797  0.905  0.933  0.959   0.994   0.722  0.885
    safe_top3_w*     0.834  0.907  0.890  0.892   1.004   0.567  0.847
    meta_select*     0.822  0.898  0.897  0.911   0.983   0.582  0.849
    meta_top1*       0.838  0.901  0.900  1.980   0.983   0.586  (Weekly fails)
    * = with oracle leakage for safeMask filter

Conclusion:
- PARTIALLY ADOPTED: Meta-learning shows clear signal
- Oracle ceiling (0.615) proves Vectrix model pool can theoretically beat M4 #1
- GBR R2 too low for DOT/CES (0.07/0.11) — can't predict when DOT fails
- Ensemble with dtsf/esn/auto_arima causes extreme predictions (Weekly/Daily explode)
- Safe pool (5 models) + meta_top1 achieves 0.873 vs DOT 0.885 = -1.4% improvement
- Per-series holdout validation HURTS (E035) due to data reduction
- To reach M4 #1 (0.821): need DL hybrid or much larger training corpus

Experiment date: 2026-03-04
==============================================================================
"""

import os
import sys
import time
import warnings
import pickle
import json

warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

M4_GROUPS = {
    'Yearly':    {'horizon': 6,  'seasonality': 1},
    'Quarterly': {'horizon': 8,  'seasonality': 4},
    'Monthly':   {'horizon': 18, 'seasonality': 12},
    'Weekly':    {'horizon': 13, 'seasonality': 1},
    'Daily':     {'horizon': 14, 'seasonality': 1},
    'Hourly':    {'horizon': 48, 'seasonality': 24},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
CACHE_DIR = os.path.join(os.path.dirname(__file__), '_cache')
P = lambda *a, **kw: print(*a, **kw, flush=True)


def _loadGroup(groupName):
    import pandas as pd
    trainPath = os.path.join(DATA_DIR, f'{groupName}-train.csv')
    testPath = os.path.join(DATA_DIR, f'{groupName}-test.csv')
    trainDf = pd.read_csv(trainPath)
    testDf = pd.read_csv(testPath)
    trainSeries = []
    testSeries = []
    for i in range(len(trainDf)):
        trainSeries.append(trainDf.iloc[i, 1:].dropna().values.astype(np.float64))
        testSeries.append(testDf.iloc[i, 1:].dropna().values.astype(np.float64))
    return trainSeries, testSeries


def _smape(actual, predicted):
    return np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-10)) * 100


def _mase(trainY, actual, predicted, seasonality):
    n = len(trainY)
    m = max(seasonality, 1)
    if n <= m:
        naiveErrors = np.abs(np.diff(trainY))
    else:
        naiveErrors = np.abs(trainY[m:] - trainY[:-m])
    masep = np.mean(naiveErrors) if len(naiveErrors) > 0 else 1e-10
    if masep < 1e-10:
        masep = 1e-10
    return np.mean(np.abs(actual - predicted)) / masep


def _naive2(trainY, horizon, seasonality):
    n = len(trainY)
    m = max(seasonality, 1)
    if m > 1 and n >= m * 2:
        seasonal = np.zeros(m)
        counts = np.zeros(m)
        trend = np.convolve(trainY, np.ones(m) / m, mode='valid')
        offset = (m - 1) // 2
        for i in range(len(trend)):
            idx = i + offset
            if idx < n and trend[i] > 0:
                seasonal[idx % m] += trainY[idx] / trend[i]
                counts[idx % m] += 1
        for i in range(m):
            seasonal[i] = seasonal[i] / max(counts[i], 1)
        meanS = np.mean(seasonal)
        if meanS > 0:
            seasonal /= meanS
        seasonal = np.maximum(seasonal, 0.01)
        deseasonalized = trainY / seasonal[np.arange(n) % m]
        pred = np.full(horizon, deseasonalized[-1])
        for h in range(horizon):
            pred[h] *= seasonal[(n + h) % m]
    else:
        pred = np.full(horizon, trainY[-1])
    return pred


def _fitPredict(model, trainY, horizon):
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred


def _fitPredictWithTimeout(model, trainY, horizon, timeoutSec=10):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_fitPredict, model, trainY, horizon)
        return fut.result(timeout=timeoutSec)


def _extractDnaFeatures(trainY, period):
    from vectrix.adaptive.dna import ForecastDNA
    dna = ForecastDNA()
    profile = dna.analyze(trainY, period=period)
    return profile.features


def _getModelFactory(modelName, period):
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
    from vectrix.engine.ets import AutoETS
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.dtsf import DynamicTimeScanForecaster
    from vectrix.engine.esn import EchoStateForecaster
    from vectrix.engine.mstl import AutoMSTL
    from vectrix.engine.theta import OptimizedTheta

    factories = {
        'dot': lambda: DynamicOptimizedTheta(period=period),
        'auto_ces': lambda: AutoCES(period=period),
        'four_theta': lambda: AdaptiveThetaEnsemble(period=period),
        'auto_ets': lambda: AutoETS(period=period),
        'auto_arima': lambda: AutoARIMA(seasonalPeriod=period),
        'dtsf': lambda: DynamicTimeScanForecaster(),
        'esn': lambda: EchoStateForecaster(),
        'auto_mstl': lambda: AutoMSTL(),
        'theta': lambda: OptimizedTheta(period=period),
    }
    return factories.get(modelName)


MODEL_NAMES = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_arima',
               'dtsf', 'esn', 'theta']


def _phase1CollectOracle(groupName, sampleCap=2000):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    seasonality = info['seasonality']

    cachePath = os.path.join(CACHE_DIR, f'oracle_{groupName}.pkl')
    if os.path.exists(cachePath):
        P(f"  [{groupName}] Loading cached oracle data...")
        with open(cachePath, 'rb') as f:
            return pickle.load(f)

    P(f"\n{'='*60}")
    P(f"  Phase 1: Collecting oracle — {groupName} (h={horizon}, m={seasonality})")
    P(f"{'='*60}")

    trainSeries, testSeries = _loadGroup(groupName)
    nSeries = len(trainSeries)

    validIdx = [i for i in range(nSeries)
                if len(trainSeries[i]) >= max(10, seasonality * 2 + 5) and len(testSeries[i]) >= horizon]
    P(f"  Valid: {len(validIdx)}/{nSeries}")

    if len(validIdx) > sampleCap:
        rng = np.random.default_rng(42)
        validIdx = sorted(rng.choice(validIdx, size=sampleCap, replace=False).tolist())
        P(f"  Sampled: {sampleCap}")

    oracleData = []
    startTime = time.perf_counter()
    nErrors = {m: 0 for m in MODEL_NAMES}

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]

        n2pred = _naive2(trainY, horizon, seasonality)
        n2Smape = _smape(testY, n2pred)
        n2Mase = _mase(trainY, testY, n2pred, seasonality)

        if n2Smape < 1e-10:
            n2Smape = 1e-10
        if n2Mase < 1e-10:
            n2Mase = 1e-10

        features = _extractDnaFeatures(trainY, seasonality)
        features['_length'] = float(len(trainY))
        features['_period'] = float(seasonality)
        features['_horizon'] = float(horizon)

        modelPreds = {}
        modelOwas = {}

        for modelName in MODEL_NAMES:
            factory = _getModelFactory(modelName, seasonality)
            if factory is None:
                continue
            try:
                pred = _fitPredictWithTimeout(factory(), trainY, horizon, timeoutSec=30)
                s = _smape(testY, pred)
                m = _mase(trainY, testY, pred, seasonality)
                owa = 0.5 * (s / n2Smape + m / n2Mase)
                modelPreds[modelName] = pred
                modelOwas[modelName] = owa
            except Exception:
                nErrors[modelName] += 1
                modelOwas[modelName] = 2.0
                modelPreds[modelName] = np.full(horizon, np.mean(trainY))

        oracleData.append({
            'idx': idx,
            'features': features,
            'modelOwas': modelOwas,
            'modelPreds': modelPreds,
            'testY': testY,
            'trainY': trainY,
            'n2Smape': n2Smape,
            'n2Mase': n2Mase,
        })

        if (count + 1) % 100 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            bestModel = min(modelOwas, key=modelOwas.get) if modelOwas else '?'
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s) best={bestModel}")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} series in {elapsed:.1f}s")
    P(f"  Errors: {nErrors}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cachePath, 'wb') as f:
        pickle.dump(oracleData, f)
    P(f"  Cached to {cachePath}")

    return oracleData


def _phase2TrainMetaLearner(allOracleData):
    P(f"\n{'='*60}")
    P(f"  Phase 2: Training Meta-Learner")
    P(f"{'='*60}")

    featureNames = None
    Xs = []
    Ys = []

    for groupName, oracleData in allOracleData.items():
        for entry in oracleData:
            features = entry['features']
            modelOwas = entry['modelOwas']

            if featureNames is None:
                featureNames = sorted([k for k in features.keys() if not k.startswith('_')])

            fVec = np.array([features.get(fn, 0.0) for fn in featureNames], dtype=np.float64)
            fVec = np.append(fVec, [features.get('_length', 0), features.get('_period', 0), features.get('_horizon', 0)])

            fVec = np.where(np.isfinite(fVec), fVec, 0.0)

            owaVec = np.array([modelOwas.get(m, 2.0) for m in MODEL_NAMES], dtype=np.float64)
            owaVec = np.clip(owaVec, 0.01, 5.0)

            Xs.append(fVec)
            Ys.append(owaVec)

    X = np.array(Xs)
    Y = np.array(Ys)
    P(f"  Training data: {X.shape[0]} series, {X.shape[1]} features, {Y.shape[1]} models")

    extendedFeatureNames = featureNames + ['_length', '_period', '_horizon']

    oracleOwas = {}
    for mi, mName in enumerate(MODEL_NAMES):
        oracleOwas[mName] = np.mean(Y[:, mi])
    P(f"  Oracle model OWAs (avg across all series):")
    for m, v in sorted(oracleOwas.items(), key=lambda x: x[1]):
        P(f"    {m:<15} {v:.4f}")

    oracleBest = np.argmin(Y, axis=1)
    bestCounts = {}
    for mi in range(len(MODEL_NAMES)):
        bestCounts[MODEL_NAMES[mi]] = np.sum(oracleBest == mi)
    P(f"\n  Oracle best model counts:")
    for m, c in sorted(bestCounts.items(), key=lambda x: -x[1]):
        P(f"    {m:<15} {c:>5} ({c/len(Y)*100:.1f}%)")

    oracleBestOwa = np.mean(np.min(Y, axis=1))
    P(f"\n  Oracle best OWA (always pick best): {oracleBestOwa:.4f}")
    P(f"  DOT-only OWA: {oracleOwas.get('dot', 0):.4f}")
    P(f"  Gap (oracle is ceiling): {oracleOwas.get('dot', 0) - oracleBestOwa:.4f}")

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    nModels = len(MODEL_NAMES)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oofPreds = np.zeros_like(Y)
    models = []

    for mi in range(nModels):
        modelName = MODEL_NAMES[mi]
        yTarget = Y[:, mi]

        foldModels = []
        for foldIdx, (trainIdx, valIdx) in enumerate(kf.split(X)):
            xTrain, yTrain = X[trainIdx], yTarget[trainIdx]

            gbr = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )
            gbr.fit(xTrain, yTrain)
            foldModels.append(gbr)

            oofPreds[valIdx, mi] = gbr.predict(X[valIdx])

        models.append(foldModels)

    P(f"\n  Meta-learner trained (5-fold CV, {nModels} models, 200 trees each)")

    return models, extendedFeatureNames, X, Y, oofPreds


def _phase3Evaluate(allOracleData, models, featureNames, X, Y, oofPreds):
    P(f"\n{'='*60}")
    P(f"  Phase 3: Evaluation")
    P(f"{'='*60}")

    nModels = len(MODEL_NAMES)

    predOwas = np.clip(oofPreds, 0.01, 5.0)
    invOwas = 1.0 / predOwas

    softmaxWeights = np.zeros_like(invOwas)
    for i in range(len(invOwas)):
        expW = np.exp(invOwas[i] - np.max(invOwas[i]))
        softmaxWeights[i] = expW / np.sum(expW)

    topKWeights = np.zeros_like(invOwas)
    K = 3
    for i in range(len(invOwas)):
        topIdx = np.argsort(predOwas[i])[:K]
        w = invOwas[i, topIdx]
        w = w / np.sum(w)
        topKWeights[i, topIdx] = w

    strategies = {
        'oracle_best': None,
        'dot_only': None,
        'equal_all': None,
        'softmax_all': softmaxWeights,
        'top3_weighted': topKWeights,
        'meta_select_best': None,
    }

    offset = 0
    groupResults = {}

    for groupName, oracleData in allOracleData.items():
        info = M4_GROUPS[groupName]
        horizon = info['horizon']
        seasonality = info['seasonality']
        nSeries = len(oracleData)

        stratOwas = {s: [] for s in strategies}
        stratSmapes = {s: [] for s in strategies}
        stratMases = {s: [] for s in strategies}

        for si, entry in enumerate(oracleData):
            gi = offset + si
            testY = entry['testY']
            n2S = entry['n2Smape']
            n2M = entry['n2Mase']
            preds = entry['modelPreds']
            owas = entry['modelOwas']

            allPreds = np.array([preds.get(m, np.full(horizon, np.mean(entry['trainY'])))
                                for m in MODEL_NAMES])

            bestIdx = np.argmin([owas.get(m, 2.0) for m in MODEL_NAMES])
            oraclePred = allPreds[bestIdx]
            s = _smape(testY, oraclePred)
            m = _mase(entry['trainY'], testY, oraclePred, seasonality)
            stratSmapes['oracle_best'].append(s)
            stratMases['oracle_best'].append(m)
            stratOwas['oracle_best'].append(0.5 * (s / n2S + m / n2M))

            dotIdx = MODEL_NAMES.index('dot')
            dotPred = allPreds[dotIdx]
            s = _smape(testY, dotPred)
            m = _mase(entry['trainY'], testY, dotPred, seasonality)
            stratSmapes['dot_only'].append(s)
            stratMases['dot_only'].append(m)
            stratOwas['dot_only'].append(0.5 * (s / n2S + m / n2M))

            equalPred = np.mean(allPreds, axis=0)
            s = _smape(testY, equalPred)
            m = _mase(entry['trainY'], testY, equalPred, seasonality)
            stratSmapes['equal_all'].append(s)
            stratMases['equal_all'].append(m)
            stratOwas['equal_all'].append(0.5 * (s / n2S + m / n2M))

            w = softmaxWeights[gi]
            softmaxPred = np.sum(allPreds * w[:, None], axis=0)
            s = _smape(testY, softmaxPred)
            m = _mase(entry['trainY'], testY, softmaxPred, seasonality)
            stratSmapes['softmax_all'].append(s)
            stratMases['softmax_all'].append(m)
            stratOwas['softmax_all'].append(0.5 * (s / n2S + m / n2M))

            w = topKWeights[gi]
            top3Pred = np.sum(allPreds * w[:, None], axis=0)
            s = _smape(testY, top3Pred)
            m = _mase(entry['trainY'], testY, top3Pred, seasonality)
            stratSmapes['top3_weighted'].append(s)
            stratMases['top3_weighted'].append(m)
            stratOwas['top3_weighted'].append(0.5 * (s / n2S + m / n2M))

            metaBestIdx = np.argmin(oofPreds[gi])
            metaPred = allPreds[metaBestIdx]
            s = _smape(testY, metaPred)
            m = _mase(entry['trainY'], testY, metaPred, seasonality)
            stratSmapes['meta_select_best'].append(s)
            stratMases['meta_select_best'].append(m)
            stratOwas['meta_select_best'].append(0.5 * (s / n2S + m / n2M))

        offset += nSeries

        P(f"\n  {groupName} (n={nSeries}):")

        groupOwa = {}
        n2SmapeAvg = np.mean([e['n2Smape'] for e in oracleData])
        n2MaseAvg = np.mean([e['n2Mase'] for e in oracleData])

        for sName in strategies:
            sSmape = np.mean(stratSmapes[sName])
            sMase = np.mean(stratMases[sName])
            sOwaM4 = 0.5 * (sSmape / n2SmapeAvg + sMase / n2MaseAvg)
            sOwaPer = np.mean(stratOwas[sName])
            groupOwa[sName] = sOwaM4
            P(f"    {sName:<20} M4-OWA={sOwaM4:.4f}  perSeriesOWA={sOwaPer:.4f}")

        groupResults[groupName] = groupOwa

    P(f"\n{'='*60}")
    P(f"  OVERALL AVG OWA (M4 style)")
    P(f"{'='*60}")

    for sName in strategies:
        vals = [groupResults[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        detail = ' '.join([f"{g[:2]}={groupResults[g][sName]:.3f}" for g in M4_GROUPS])
        P(f"  {sName:<20} AVG={avg:.4f}  [{detail}]")

    P(f"\n  Reference: M4 #1 ES-RNN 0.821, #2 FFORMA 0.838, #18 Theta 0.897")
    P(f"  Current DOT-Hybrid: 0.885")

    return groupResults


def _analyzeFeatureImportance(models, featureNames):
    P(f"\n{'='*60}")
    P(f"  Feature Importance Analysis")
    P(f"{'='*60}")

    for mi, mName in enumerate(MODEL_NAMES):
        importances = np.zeros(len(featureNames))
        for foldModel in models[mi]:
            importances += foldModel.feature_importances_
        importances /= len(models[mi])

        topIdx = np.argsort(importances)[-10:][::-1]
        P(f"\n  {mName}:")
        for ti in topIdx:
            P(f"    {featureNames[ti]:<30} {importances[ti]:.4f}")


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E031: FFORMA-Style Meta-Learning Ensemble")
    P("=" * 60)

    totalStart = time.perf_counter()

    GROUP_CAPS = {
        'Yearly': 2000, 'Quarterly': 2000, 'Monthly': 2000,
        'Weekly': 2000, 'Daily': 500, 'Hourly': 414,
    }

    allOracleData = {}
    for group in M4_GROUPS:
        allOracleData[group] = _phase1CollectOracle(group, sampleCap=GROUP_CAPS.get(group, 2000))

    models, featureNames, X, Y, oofPreds = _phase2TrainMetaLearner(allOracleData)
    groupResults = _phase3Evaluate(allOracleData, models, featureNames, X, Y, oofPreds)
    _analyzeFeatureImportance(models, featureNames)

    totalElapsed = time.perf_counter() - totalStart
    P(f"\nTotal time: {totalElapsed/60:.1f} min")
    P("=" * 60)
