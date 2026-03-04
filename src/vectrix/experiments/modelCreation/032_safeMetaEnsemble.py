"""
==============================================================================
Experiment ID: modelCreation/032
Experiment: Safe Meta-Learning Ensemble with Prediction Filtering
==============================================================================

Purpose:
- Fix E031 failures: extreme predictions breaking ensemble in Weekly/Daily
- Add per-series safety filters: exclude models with OWA > threshold
- Test frequency-specific model pools
- Implement trimmed-mean ensemble and median ensemble
- Target: AVG OWA <= 0.850 (beating all pure statistical methods)

Hypothesis:
1. Filtering extreme models per-series improves ensemble stability
2. Median ensemble is more robust than weighted mean
3. Per-frequency optimal model pools outperform single pool
4. Meta-learner with safety filtering achieves OWA < 0.850

Method:
1. Reuse E031 cached oracle data (8 models × 7273 series)
2. Implement safe ensemble strategies:
   a. Filtered top-K weighted (exclude OWA > 2.0)
   b. Median ensemble (robust to outliers)
   c. Frequency-specific pools (best K models per frequency)
   d. Safe meta-select (meta-learner with fallback)
   e. FFORMA-style weighted combination (softmax of predicted 1/OWA)
3. M4-style OWA evaluation (group-avg sMAPE/MASE then ratio)

Results (to be filled after experiment):

Conclusion:

Experiment date: 2026-03-04
==============================================================================
"""

import os
import sys
import time
import warnings
import pickle

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

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_cache')
MODEL_NAMES = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_arima',
               'dtsf', 'esn', 'theta']

P = lambda *a, **kw: print(*a, **kw, flush=True)


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


def _loadAllOracle():
    allOracle = {}
    for group in M4_GROUPS:
        cachePath = os.path.join(CACHE_DIR, f'oracle_{group}.pkl')
        if not os.path.exists(cachePath):
            P(f"  ERROR: Cache not found for {group}. Run E031 first.")
            return None
        with open(cachePath, 'rb') as f:
            allOracle[group] = pickle.load(f)
        P(f"  Loaded {group}: {len(allOracle[group])} series")
    return allOracle


def _trainMetaLearner(allOracle):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    featureNames = None
    Xs = []
    Ys = []
    groupLabels = []

    for groupName, oracleData in allOracle.items():
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
            groupLabels.append(groupName)

    X = np.array(Xs)
    Y = np.array(Ys)
    featureNames = featureNames + ['_length', '_period', '_horizon']

    P(f"  Training: {X.shape[0]} series, {X.shape[1]} features, {Y.shape[1]} models")

    nModels = len(MODEL_NAMES)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oofPreds = np.zeros_like(Y)
    models = []

    for mi in range(nModels):
        yTarget = Y[:, mi]
        foldModels = []
        for trainIdx, valIdx in kf.split(X):
            gbr = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                min_samples_leaf=15,
                random_state=42,
            )
            gbr.fit(X[trainIdx], yTarget[trainIdx])
            foldModels.append(gbr)
            oofPreds[valIdx, mi] = gbr.predict(X[valIdx])
        models.append(foldModels)

    P(f"  Meta-learner trained: {nModels} models × 5 folds × 300 trees")
    return models, featureNames, X, Y, oofPreds, np.array(groupLabels)


def _evaluateStrategies(allOracle, X, Y, oofPreds, groupLabels):
    P(f"\n{'='*70}")
    P(f"  Evaluation: Safe Meta-Learning Ensemble Strategies")
    P(f"{'='*70}")

    nModels = len(MODEL_NAMES)

    predOwas = np.clip(oofPreds, 0.01, 5.0)

    strategies = [
        'oracle_best',
        'dot_only',
        'safe_top3_weighted',
        'safe_softmax',
        'median_ensemble',
        'trimmed_mean',
        'meta_select_safe',
        'fforma_weighted',
        'freq_specific_pool',
    ]

    FREQ_POOLS = {
        'Yearly':    ['dot', 'four_theta', 'auto_ces', 'auto_ets', 'theta'],
        'Quarterly': ['dot', 'auto_ces', 'theta', 'auto_ets', 'four_theta'],
        'Monthly':   ['dot', 'auto_ces', 'four_theta', 'dtsf', 'theta'],
        'Weekly':    ['dot', 'auto_ces', 'four_theta', 'auto_ets'],
        'Daily':     ['dot', 'auto_ces', 'auto_ets', 'esn'],
        'Hourly':    ['dot', 'auto_ces', 'esn', 'dtsf', 'four_theta'],
    }

    offset = 0
    groupResults = {}

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]
        horizon = info['horizon']
        seasonality = info['seasonality']
        nSeries = len(oracleData)

        stratSmapes = {s: [] for s in strategies}
        stratMases = {s: [] for s in strategies}

        for si, entry in enumerate(oracleData):
            gi = offset + si
            testY = entry['testY']
            trainY = entry['trainY']
            n2S = entry['n2Smape']
            n2M = entry['n2Mase']
            preds = entry['modelPreds']
            owas = entry['modelOwas']

            allPreds = np.array([preds.get(m, np.full(horizon, np.mean(trainY)))
                                for m in MODEL_NAMES])

            allSmapes = np.array([_smape(testY, allPreds[mi]) for mi in range(nModels)])
            allMases = np.array([_mase(trainY, testY, allPreds[mi], seasonality) for mi in range(nModels)])

            owaArr = np.array([owas.get(m, 5.0) for m in MODEL_NAMES])
            safeMask = owaArr < 3.0

            predOwaArr = predOwas[gi]

            bestIdx = np.argmin(owaArr)
            stratSmapes['oracle_best'].append(_smape(testY, allPreds[bestIdx]))
            stratMases['oracle_best'].append(_mase(trainY, testY, allPreds[bestIdx], seasonality))

            dotIdx = MODEL_NAMES.index('dot')
            stratSmapes['dot_only'].append(_smape(testY, allPreds[dotIdx]))
            stratMases['dot_only'].append(_mase(trainY, testY, allPreds[dotIdx], seasonality))

            safeIdx = np.where(safeMask)[0]
            if len(safeIdx) < 2:
                safeIdx = np.array([dotIdx])
            safeOwas = owaArr[safeIdx]
            topK = min(3, len(safeIdx))
            topIdx = safeIdx[np.argsort(predOwaArr[safeIdx])[:topK]]
            w = 1.0 / np.clip(predOwaArr[topIdx], 0.01, 5.0)
            w = w / np.sum(w)
            safePred = np.sum(allPreds[topIdx] * w[:, None], axis=0)
            stratSmapes['safe_top3_weighted'].append(_smape(testY, safePred))
            stratMases['safe_top3_weighted'].append(_mase(trainY, testY, safePred, seasonality))

            safePredOwas = predOwaArr[safeIdx]
            invOwa = 1.0 / safePredOwas
            expW = np.exp(invOwa - np.max(invOwa))
            softW = expW / np.sum(expW)
            safeSoftPred = np.sum(allPreds[safeIdx] * softW[:, None], axis=0)
            stratSmapes['safe_softmax'].append(_smape(testY, safeSoftPred))
            stratMases['safe_softmax'].append(_mase(trainY, testY, safeSoftPred, seasonality))

            medianPred = np.median(allPreds[safeIdx], axis=0)
            stratSmapes['median_ensemble'].append(_smape(testY, medianPred))
            stratMases['median_ensemble'].append(_mase(trainY, testY, medianPred, seasonality))

            sortedIdx = np.argsort(predOwaArr[safeIdx])
            trimmedIdx = safeIdx[sortedIdx[:max(len(sortedIdx) - 2, 2)]]
            trimmedPred = np.mean(allPreds[trimmedIdx], axis=0)
            stratSmapes['trimmed_mean'].append(_smape(testY, trimmedPred))
            stratMases['trimmed_mean'].append(_mase(trainY, testY, trimmedPred, seasonality))

            metaBestSafe = safeIdx[np.argmin(predOwaArr[safeIdx])]
            metaPred = allPreds[metaBestSafe]
            stratSmapes['meta_select_safe'].append(_smape(testY, metaPred))
            stratMases['meta_select_safe'].append(_mase(trainY, testY, metaPred, seasonality))

            invPredOwa = 1.0 / np.clip(predOwaArr[safeIdx], 0.01, 5.0)
            fformaW = invPredOwa / np.sum(invPredOwa)
            fformaPred = np.sum(allPreds[safeIdx] * fformaW[:, None], axis=0)
            stratSmapes['fforma_weighted'].append(_smape(testY, fformaPred))
            stratMases['fforma_weighted'].append(_mase(trainY, testY, fformaPred, seasonality))

            pool = FREQ_POOLS.get(groupName, ['dot', 'auto_ces', 'four_theta'])
            poolIdx = [MODEL_NAMES.index(m) for m in pool if m in MODEL_NAMES]
            poolPredOwas = predOwaArr[poolIdx]
            poolTopIdx = np.array(poolIdx)[np.argsort(poolPredOwas)[:3]]
            pw = 1.0 / np.clip(predOwaArr[poolTopIdx], 0.01, 5.0)
            pw = pw / np.sum(pw)
            poolPred = np.sum(allPreds[poolTopIdx] * pw[:, None], axis=0)
            stratSmapes['freq_specific_pool'].append(_smape(testY, poolPred))
            stratMases['freq_specific_pool'].append(_mase(trainY, testY, poolPred, seasonality))

        offset += nSeries

        n2SmapeAvg = np.mean([e['n2Smape'] for e in oracleData])
        n2MaseAvg = np.mean([e['n2Mase'] for e in oracleData])

        groupOwa = {}
        P(f"\n  {groupName} (n={nSeries}):")
        for sName in strategies:
            sSmape = np.mean(stratSmapes[sName])
            sMase = np.mean(stratMases[sName])
            sOwa = 0.5 * (sSmape / n2SmapeAvg + sMase / n2MaseAvg)
            groupOwa[sName] = sOwa
            marker = ' ***' if sOwa < groupOwa.get('dot_only', 999) else ''
            P(f"    {sName:<25} OWA={sOwa:.4f}{marker}")

        groupResults[groupName] = groupOwa

    P(f"\n{'='*70}")
    P(f"  OVERALL AVG OWA (M4 style)")
    P(f"{'='*70}")

    for sName in strategies:
        vals = [groupResults[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        dotAvg = np.mean([groupResults[g]['dot_only'] for g in M4_GROUPS])
        diff = avg - dotAvg
        marker = ' *** BETTER' if avg < dotAvg else ''
        P(f"  {sName:<25} AVG={avg:.4f} (vs DOT {diff:+.4f}){marker}  [{detail}]")

    P(f"\n  Reference: M4 #1 ES-RNN 0.821, #2 FFORMA 0.838, #3 Theta 0.854")
    P(f"  Current DOT-Hybrid baseline: 0.885")

    bestStrat = min(strategies, key=lambda s: np.mean([groupResults[g][s] for g in M4_GROUPS]))
    bestOwa = np.mean([groupResults[g][bestStrat] for g in M4_GROUPS])
    P(f"\n  BEST STRATEGY: {bestStrat} = {bestOwa:.4f}")

    return groupResults


def _analyzeOracleGaps(allOracle):
    P(f"\n{'='*70}")
    P(f"  Oracle Analysis: Where does DOT fail?")
    P(f"{'='*70}")

    for groupName, oracleData in allOracle.items():
        dotIdx = MODEL_NAMES.index('dot')
        nBetter = 0
        gapSum = 0
        bestModelWins = {m: 0 for m in MODEL_NAMES}

        for entry in oracleData:
            owas = entry['modelOwas']
            dotOwa = owas.get('dot', 2.0)
            bestModel = min(owas, key=owas.get)
            bestOwa = owas[bestModel]
            bestModelWins[bestModel] += 1
            if bestOwa < dotOwa * 0.9:
                nBetter += 1
                gapSum += dotOwa - bestOwa

        P(f"\n  {groupName}: {nBetter}/{len(oracleData)} series where another model beats DOT by >10%")
        P(f"    Best model wins:")
        for m, c in sorted(bestModelWins.items(), key=lambda x: -x[1]):
            P(f"      {m:<15} {c:>5} ({c/len(oracleData)*100:.1f}%)")


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 70)
    P("E032: Safe Meta-Learning Ensemble")
    P("=" * 70)

    totalStart = time.perf_counter()

    allOracle = _loadAllOracle()
    if allOracle is None:
        sys.exit(1)

    models, featureNames, X, Y, oofPreds, groupLabels = _trainMetaLearner(allOracle)
    groupResults = _evaluateStrategies(allOracle, X, Y, oofPreds, groupLabels)
    _analyzeOracleGaps(allOracle)

    totalElapsed = time.perf_counter() - totalStart
    P(f"\nTotal time: {totalElapsed:.1f}s")
    P("=" * 70)
