"""
==============================================================================
Experiment ID: modelCreation/033
Experiment: Realistic Meta-Learning Ensemble (No Oracle Leakage)
==============================================================================

Purpose:
- E032 had oracle leakage: safeMask used actual OWA to filter models
- This experiment uses ONLY meta-learner predictions (no actual OWA)
- Simulate real-world deployment: DNA features → GBR → weights → ensemble
- Also test: what if we use DOT as fallback for Yearly (where DOT dominates)?

Hypothesis:
1. Realistic meta-ensemble should still beat DOT (0.885)
2. DOT fallback for Yearly + meta for rest should be best hybrid
3. Target: AVG OWA <= 0.860

Method:
1. Reuse E031 cached oracle data
2. Train GBR meta-learner (5-fold OOF, no leakage)
3. All filtering and selection based on GBR predictions only
4. Evaluate multiple strategies

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

    X = np.array(Xs)
    Y = np.array(Ys)
    featureNames = featureNames + ['_length', '_period', '_horizon']

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

    P(f"  Meta-learner: {X.shape[0]} series, {nModels} models, 5-fold OOF")

    r2Scores = []
    for mi in range(nModels):
        ssRes = np.sum((Y[:, mi] - oofPreds[:, mi]) ** 2)
        ssTot = np.sum((Y[:, mi] - np.mean(Y[:, mi])) ** 2)
        r2 = 1 - ssRes / max(ssTot, 1e-10)
        r2Scores.append(r2)
        P(f"    {MODEL_NAMES[mi]:<15} R2={r2:.4f}  mean_actual={np.mean(Y[:, mi]):.3f}  mean_pred={np.mean(oofPreds[:, mi]):.3f}")

    return models, featureNames, X, Y, oofPreds


def _evaluate(allOracle, X, Y, oofPreds):
    P(f"\n{'='*70}")
    P(f"  Evaluation (REALISTIC — no oracle leakage)")
    P(f"{'='*70}")

    nModels = len(MODEL_NAMES)
    predOwas = np.clip(oofPreds, 0.01, 5.0)

    strategies = [
        'oracle_best',
        'dot_only',
        'meta_top1',
        'meta_top3_weighted',
        'meta_top3_equal',
        'meta_softmax_all',
        'meta_inv_owa_all',
        'dot_yearly_meta_rest',
        'meta_top5_weighted',
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

            allPreds = np.array([preds.get(m, np.full(horizon, np.mean(trainY)))
                                for m in MODEL_NAMES])

            owaArr = np.array([owas.get(m, 5.0) for m in MODEL_NAMES])
            predArr = predOwas[gi]

            bestIdx = np.argmin(owaArr)
            stratSmapes['oracle_best'].append(_smape(testY, allPreds[bestIdx]))
            stratMases['oracle_best'].append(_mase(trainY, testY, allPreds[bestIdx], seasonality))

            dotIdx = MODEL_NAMES.index('dot')
            stratSmapes['dot_only'].append(_smape(testY, allPreds[dotIdx]))
            stratMases['dot_only'].append(_mase(trainY, testY, allPreds[dotIdx], seasonality))

            metaTop1Idx = np.argmin(predArr)
            stratSmapes['meta_top1'].append(_smape(testY, allPreds[metaTop1Idx]))
            stratMases['meta_top1'].append(_mase(trainY, testY, allPreds[metaTop1Idx], seasonality))

            top3Idx = np.argsort(predArr)[:3]
            w3 = 1.0 / predArr[top3Idx]
            w3 = w3 / np.sum(w3)
            pred3 = np.sum(allPreds[top3Idx] * w3[:, None], axis=0)
            stratSmapes['meta_top3_weighted'].append(_smape(testY, pred3))
            stratMases['meta_top3_weighted'].append(_mase(trainY, testY, pred3, seasonality))

            pred3eq = np.mean(allPreds[top3Idx], axis=0)
            stratSmapes['meta_top3_equal'].append(_smape(testY, pred3eq))
            stratMases['meta_top3_equal'].append(_mase(trainY, testY, pred3eq, seasonality))

            invAll = 1.0 / predArr
            expW = np.exp(invAll - np.max(invAll))
            softW = expW / np.sum(softW if 'softW' in dir() else expW)
            softW = expW / np.sum(expW)
            predSoft = np.sum(allPreds * softW[:, None], axis=0)
            stratSmapes['meta_softmax_all'].append(_smape(testY, predSoft))
            stratMases['meta_softmax_all'].append(_mase(trainY, testY, predSoft, seasonality))

            invW = 1.0 / predArr
            invW = invW / np.sum(invW)
            predInv = np.sum(allPreds * invW[:, None], axis=0)
            stratSmapes['meta_inv_owa_all'].append(_smape(testY, predInv))
            stratMases['meta_inv_owa_all'].append(_mase(trainY, testY, predInv, seasonality))

            if groupName == 'Yearly':
                dotPred = allPreds[dotIdx]
                stratSmapes['dot_yearly_meta_rest'].append(_smape(testY, dotPred))
                stratMases['dot_yearly_meta_rest'].append(_mase(trainY, testY, dotPred, seasonality))
            else:
                stratSmapes['dot_yearly_meta_rest'].append(_smape(testY, pred3))
                stratMases['dot_yearly_meta_rest'].append(_mase(trainY, testY, pred3, seasonality))

            top5Idx = np.argsort(predArr)[:5]
            w5 = 1.0 / predArr[top5Idx]
            w5 = w5 / np.sum(w5)
            pred5 = np.sum(allPreds[top5Idx] * w5[:, None], axis=0)
            stratSmapes['meta_top5_weighted'].append(_smape(testY, pred5))
            stratMases['meta_top5_weighted'].append(_mase(trainY, testY, pred5, seasonality))

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
            marker = f' ({diff:+.3f})' if sName != 'oracle_best' and sName != 'dot_only' else ''
            P(f"    {sName:<25} OWA={sOwa:.4f}{marker}")

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
    P(f"  Current DOT-Hybrid={dotAvg:.4f}")

    return groupResults


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 70)
    P("E033: Realistic Meta-Learning Ensemble (No Oracle Leakage)")
    P("=" * 70)

    totalStart = time.perf_counter()

    allOracle = _loadAllOracle()
    models, featureNames, X, Y, oofPreds = _trainMetaLearner(allOracle)
    groupResults = _evaluate(allOracle, X, Y, oofPreds)

    totalElapsed = time.perf_counter() - totalStart
    P(f"\nTotal time: {totalElapsed:.1f}s")
    P("=" * 70)
