"""
==============================================================================
Experiment ID: modelCreation/034
Experiment: Safe Pool Meta-Ensemble (Stable Models Only)
==============================================================================

Purpose:
- E033 showed extreme predictions from dtsf/esn/auto_arima destroy Weekly/Daily
- Use only stable statistical models: dot, auto_ces, four_theta, auto_ets, theta
- These 5 models never produce extreme predictions (bounded, well-behaved)
- Also test: DOT for Yearly (where DOT dominates) + meta for rest

Hypothesis:
1. Safe pool meta-ensemble should beat DOT across all frequencies
2. Target: AVG OWA <= 0.860
3. dot_yearly_meta_rest should be optimal hybrid

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
ALL_MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_arima',
              'dtsf', 'esn', 'theta']
SAFE_MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'theta']
SAFE_IDX = [ALL_MODELS.index(m) for m in SAFE_MODELS]

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
        with open(cachePath, 'rb') as f:
            allOracle[group] = pickle.load(f)
        P(f"  Loaded {group}: {len(allOracle[group])} series")
    return allOracle


def _trainSafeMetaLearner(allOracle):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import KFold

    featureNames = None
    Xs = []
    Ys = []

    for groupName, oracleData in allOracle.items():
        for entry in oracleData:
            features = entry['features']
            modelOwas = entry['modelOwas']

            if featureNames is None:
                featureNames = sorted([k for k in features.keys() if not k.startswith('_')])

            fVec = np.array([features.get(fn, 0.0) for fn in featureNames], dtype=np.float64)
            fVec = np.append(fVec, [features.get('_length', 0), features.get('_period', 0), features.get('_horizon', 0)])
            fVec = np.where(np.isfinite(fVec), fVec, 0.0)

            owaVec = np.array([modelOwas.get(m, 2.0) for m in SAFE_MODELS], dtype=np.float64)
            owaVec = np.clip(owaVec, 0.01, 5.0)

            Xs.append(fVec)
            Ys.append(owaVec)

    X = np.array(Xs)
    Y = np.array(Ys)
    featureNames = featureNames + ['_length', '_period', '_horizon']

    nModels = len(SAFE_MODELS)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oofPreds = np.zeros_like(Y)

    for mi in range(nModels):
        yTarget = Y[:, mi]
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
            oofPreds[valIdx, mi] = gbr.predict(X[valIdx])

    P(f"  Safe meta-learner: {X.shape[0]} series, {nModels} models")

    for mi in range(nModels):
        ssRes = np.sum((Y[:, mi] - oofPreds[:, mi]) ** 2)
        ssTot = np.sum((Y[:, mi] - np.mean(Y[:, mi])) ** 2)
        r2 = 1 - ssRes / max(ssTot, 1e-10)
        P(f"    {SAFE_MODELS[mi]:<15} R2={r2:.4f}")

    oraclePerSeries = np.min(Y, axis=1)
    P(f"\n  Safe Oracle avg OWA: {np.mean(oraclePerSeries):.4f}")
    P(f"  DOT avg OWA: {np.mean(Y[:, SAFE_MODELS.index('dot')]):.4f}")

    return featureNames, X, Y, oofPreds


def _evaluate(allOracle, X, Y, oofPreds):
    P(f"\n{'='*70}")
    P(f"  Evaluation: Safe Pool Meta-Ensemble")
    P(f"{'='*70}")

    nSafe = len(SAFE_MODELS)
    predOwas = np.clip(oofPreds, 0.01, 5.0)

    strategies = [
        'oracle_best_safe',
        'oracle_best_all',
        'dot_only',
        'meta_top1',
        'meta_top2_weighted',
        'meta_top3_weighted',
        'meta_inv_owa_all5',
        'dot_yearly_meta_rest',
        'meta_median',
    ]

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
            preds = entry['modelPreds']
            owas = entry['modelOwas']

            safePreds = np.array([preds.get(m, np.full(horizon, np.mean(trainY))) for m in SAFE_MODELS])
            allPreds8 = np.array([preds.get(m, np.full(horizon, np.mean(trainY))) for m in ALL_MODELS])

            safeOwas = np.array([owas.get(m, 5.0) for m in SAFE_MODELS])
            allOwas8 = np.array([owas.get(m, 5.0) for m in ALL_MODELS])
            predArr = predOwas[gi]

            bestSafeIdx = np.argmin(safeOwas)
            stratSmapes['oracle_best_safe'].append(_smape(testY, safePreds[bestSafeIdx]))
            stratMases['oracle_best_safe'].append(_mase(trainY, testY, safePreds[bestSafeIdx], seasonality))

            bestAllIdx = np.argmin(allOwas8)
            stratSmapes['oracle_best_all'].append(_smape(testY, allPreds8[bestAllIdx]))
            stratMases['oracle_best_all'].append(_mase(trainY, testY, allPreds8[bestAllIdx], seasonality))

            dotSafeIdx = SAFE_MODELS.index('dot')
            stratSmapes['dot_only'].append(_smape(testY, safePreds[dotSafeIdx]))
            stratMases['dot_only'].append(_mase(trainY, testY, safePreds[dotSafeIdx], seasonality))

            metaTop1 = np.argmin(predArr)
            stratSmapes['meta_top1'].append(_smape(testY, safePreds[metaTop1]))
            stratMases['meta_top1'].append(_mase(trainY, testY, safePreds[metaTop1], seasonality))

            top2Idx = np.argsort(predArr)[:2]
            w2 = 1.0 / predArr[top2Idx]
            w2 = w2 / np.sum(w2)
            pred2 = np.sum(safePreds[top2Idx] * w2[:, None], axis=0)
            stratSmapes['meta_top2_weighted'].append(_smape(testY, pred2))
            stratMases['meta_top2_weighted'].append(_mase(trainY, testY, pred2, seasonality))

            top3Idx = np.argsort(predArr)[:3]
            w3 = 1.0 / predArr[top3Idx]
            w3 = w3 / np.sum(w3)
            pred3 = np.sum(safePreds[top3Idx] * w3[:, None], axis=0)
            stratSmapes['meta_top3_weighted'].append(_smape(testY, pred3))
            stratMases['meta_top3_weighted'].append(_mase(trainY, testY, pred3, seasonality))

            invAll = 1.0 / predArr
            invAll = invAll / np.sum(invAll)
            predAll = np.sum(safePreds * invAll[:, None], axis=0)
            stratSmapes['meta_inv_owa_all5'].append(_smape(testY, predAll))
            stratMases['meta_inv_owa_all5'].append(_mase(trainY, testY, predAll, seasonality))

            if groupName == 'Yearly':
                stratSmapes['dot_yearly_meta_rest'].append(_smape(testY, safePreds[dotSafeIdx]))
                stratMases['dot_yearly_meta_rest'].append(_mase(trainY, testY, safePreds[dotSafeIdx], seasonality))
            else:
                stratSmapes['dot_yearly_meta_rest'].append(_smape(testY, pred3))
                stratMases['dot_yearly_meta_rest'].append(_mase(trainY, testY, pred3, seasonality))

            medPred = np.median(safePreds, axis=0)
            stratSmapes['meta_median'].append(_smape(testY, medPred))
            stratMases['meta_median'].append(_mase(trainY, testY, medPred, seasonality))

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
            dotOwa = groupOwa.get('dot_only', 999)
            diff = sOwa - dotOwa
            marker = ' ***' if diff < -0.001 else ''
            P(f"    {sName:<25} OWA={sOwa:.4f} ({diff:+.4f}){marker}")

        groupResults[groupName] = groupOwa

    P(f"\n{'='*70}")
    P(f"  OVERALL AVG OWA")
    P(f"{'='*70}")

    dotAvg = np.mean([groupResults[g]['dot_only'] for g in M4_GROUPS])
    for sName in strategies:
        vals = [groupResults[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if avg < dotAvg else ''
        P(f"  {sName:<25} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    P(f"\n  M4 #1 ES-RNN=0.821  #2 FFORMA=0.838  #3 Theta=0.854")
    P(f"  DOT baseline={dotAvg:.4f}")

    bestStrat = min([s for s in strategies if s not in ('oracle_best_safe', 'oracle_best_all')],
                    key=lambda s: np.mean([groupResults[g][s] for g in M4_GROUPS]))
    bestOwa = np.mean([groupResults[g][bestStrat] for g in M4_GROUPS])
    P(f"\n  BEST REALISTIC STRATEGY: {bestStrat} = {bestOwa:.4f}")

    return groupResults


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 70)
    P("E034: Safe Pool Meta-Ensemble")
    P("=" * 70)

    totalStart = time.perf_counter()

    allOracle = _loadAllOracle()
    featureNames, X, Y, oofPreds = _trainSafeMetaLearner(allOracle)
    groupResults = _evaluate(allOracle, X, Y, oofPreds)

    totalElapsed = time.perf_counter() - totalStart
    P(f"\nTotal time: {totalElapsed:.1f}s")
    P("=" * 70)
