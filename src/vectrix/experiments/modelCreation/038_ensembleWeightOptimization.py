"""
==============================================================================
Experiment ID: modelCreation/038
Experiment: Ensemble Weight Optimization (Safe Pool)
==============================================================================

Purpose:
- E037 showed single model selection can't beat DOT-only (0.885) with rules
- Oracle ceiling is 0.715 → ensemble ceiling is between 0.715 and 0.885
- Hypothesis: optimized ensemble of ALL 5 safe models beats single-best
- Test multiple ensemble strategies using oracle data

Hypothesis:
1. inv_sMAPE_all5 should beat dot_only and single-best selection
2. Trimmed mean (drop worst) should be robust
3. Per-frequency optimal ensemble exists
4. Target: AVG OWA < 0.870

Method:
1. Use E031 oracle cached predictions (5 safe models per series)
2. Compare ensemble strategies:
   - equal_avg: simple mean of all 5
   - inv_mape: inverse MAPE weighted (current vectrix.py approach)
   - inv_mape_top3: current vectrix.py approach (top 3 by MAPE)
   - inv_smape_all5: inverse sMAPE weighted all 5
   - trimmed_mean: drop worst, average rest
   - median: median of 5 predictions
   - inv_mape_top2: top 2 only
   - optimal_static: per-frequency optimal static weights
3. Evaluate on test data using OWA

Results (to be filled after experiment):

Conclusion:

Experiment date: 2026-03-04
==============================================================================
"""

import os
import sys
import pickle
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import minimize

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
SAFE_MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'theta']

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


def _loadAllOracle():
    allOracle = {}
    for group in M4_GROUPS:
        cachePath = os.path.join(CACHE_DIR, f'oracle_{group}.pkl')
        with open(cachePath, 'rb') as f:
            allOracle[group] = pickle.load(f)
        P(f"  Loaded {group}: {len(allOracle[group])} series")
    return allOracle


def _ensemblePredict(safePreds, weights):
    w = np.array(weights)
    w = w / np.sum(w)
    return sum(w[i] * safePreds[i] for i in range(len(safePreds)))


def _runEnsembleStrategies(allOracle):
    P(f"\n{'='*70}")
    P(f"  Ensemble Weight Optimization")
    P(f"{'='*70}")

    strategies = [
        'oracle_best',
        'dot_only',
        'equal_avg',
        'inv_mape_top3',
        'inv_smape_all5',
        'inv_smape_top3',
        'inv_smape_top2',
        'trimmed_mean',
        'median',
        'drop_worst_avg',
        'softmax_inv_smape',
    ]

    groupResults = {}

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]
        horizon = info['horizon']
        seasonality = info['seasonality']
        nSeries = len(oracleData)

        stratSmapes = {s: [] for s in strategies}
        stratMases = {s: [] for s in strategies}

        for entry in oracleData:
            testY = entry['testY']
            trainY = entry['trainY']
            preds = entry['modelPreds']
            owas = entry['modelOwas']

            safePredsList = [preds.get(m, np.full(horizon, np.mean(trainY))) for m in SAFE_MODELS]
            safeSmapes = [_smape(testY, p) for p in safePredsList]
            safeOwas = [owas.get(m, 5.0) for m in SAFE_MODELS]

            bestIdx = np.argmin(safeOwas)
            stratSmapes['oracle_best'].append(_smape(testY, safePredsList[bestIdx]))
            stratMases['oracle_best'].append(_mase(trainY, testY, safePredsList[bestIdx], seasonality))

            dotIdx = SAFE_MODELS.index('dot')
            stratSmapes['dot_only'].append(_smape(testY, safePredsList[dotIdx]))
            stratMases['dot_only'].append(_mase(trainY, testY, safePredsList[dotIdx], seasonality))

            avgPred = np.mean(safePredsList, axis=0)
            stratSmapes['equal_avg'].append(_smape(testY, avgPred))
            stratMases['equal_avg'].append(_mase(trainY, testY, avgPred, seasonality))

            top3Idx = np.argsort(safeSmapes)[:3]
            w3 = np.array([1.0 / max(safeSmapes[i], 0.01) for i in top3Idx])
            w3 /= np.sum(w3)
            top3Pred = sum(w3[j] * safePredsList[top3Idx[j]] for j in range(3))
            stratSmapes['inv_mape_top3'].append(_smape(testY, top3Pred))
            stratMases['inv_mape_top3'].append(_mase(trainY, testY, top3Pred, seasonality))

            wAll = np.array([1.0 / max(s, 0.01) for s in safeSmapes])
            wAll /= np.sum(wAll)
            all5Pred = sum(wAll[i] * safePredsList[i] for i in range(5))
            stratSmapes['inv_smape_all5'].append(_smape(testY, all5Pred))
            stratMases['inv_smape_all5'].append(_mase(trainY, testY, all5Pred, seasonality))

            top3IdxS = np.argsort(safeSmapes)[:3]
            w3s = np.array([1.0 / max(safeSmapes[i], 0.01) for i in top3IdxS])
            w3s /= np.sum(w3s)
            top3sPred = sum(w3s[j] * safePredsList[top3IdxS[j]] for j in range(3))
            stratSmapes['inv_smape_top3'].append(_smape(testY, top3sPred))
            stratMases['inv_smape_top3'].append(_mase(trainY, testY, top3sPred, seasonality))

            top2Idx = np.argsort(safeSmapes)[:2]
            w2 = np.array([1.0 / max(safeSmapes[i], 0.01) for i in top2Idx])
            w2 /= np.sum(w2)
            top2Pred = sum(w2[j] * safePredsList[top2Idx[j]] for j in range(2))
            stratSmapes['inv_smape_top2'].append(_smape(testY, top2Pred))
            stratMases['inv_smape_top2'].append(_mase(trainY, testY, top2Pred, seasonality))

            sortedIdx = np.argsort(safeSmapes)
            trimPreds = [safePredsList[sortedIdx[i]] for i in range(4)]
            trimPred = np.mean(trimPreds, axis=0)
            stratSmapes['trimmed_mean'].append(_smape(testY, trimPred))
            stratMases['trimmed_mean'].append(_mase(trainY, testY, trimPred, seasonality))

            medPred = np.median(safePredsList, axis=0)
            stratSmapes['median'].append(_smape(testY, medPred))
            stratMases['median'].append(_mase(trainY, testY, medPred, seasonality))

            worstIdx = np.argmax(safeSmapes)
            dwIdxs = [i for i in range(5) if i != worstIdx]
            dwPred = np.mean([safePredsList[i] for i in dwIdxs], axis=0)
            stratSmapes['drop_worst_avg'].append(_smape(testY, dwPred))
            stratMases['drop_worst_avg'].append(_mase(trainY, testY, dwPred, seasonality))

            invS = np.array([1.0 / max(s, 0.01) for s in safeSmapes])
            expW = np.exp(invS - np.max(invS))
            softW = expW / np.sum(expW)
            softPred = sum(softW[i] * safePredsList[i] for i in range(5))
            stratSmapes['softmax_inv_smape'].append(_smape(testY, softPred))
            stratMases['softmax_inv_smape'].append(_mase(trainY, testY, softPred, seasonality))

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
            marker = ' ***' if sName not in ('oracle_best', 'dot_only') and diff < -0.001 else ''
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
        marker = ' *** BETTER' if sName not in ('oracle_best', 'dot_only') and avg < dotAvg else ''
        P(f"  {sName:<25} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    return groupResults


def _optimizeStaticWeights(allOracle):
    P(f"\n{'='*70}")
    P(f"  Phase 2: Per-Frequency Static Weight Optimization")
    P(f"{'='*70}")

    optimalWeights = {}

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]
        horizon = info['horizon']
        seasonality = info['seasonality']
        nSeries = len(oracleData)

        allSafePreds = []
        allTestY = []
        allTrainY = []

        for entry in oracleData:
            testY = entry['testY']
            trainY = entry['trainY']
            preds = entry['modelPreds']

            safePredsList = [preds.get(m, np.full(horizon, np.mean(trainY))) for m in SAFE_MODELS]
            allSafePreds.append(safePredsList)
            allTestY.append(testY)
            allTrainY.append(trainY)

        def objective(wRaw):
            w = np.exp(wRaw)
            w = w / np.sum(w)
            totalSmape = 0
            for si in range(nSeries):
                pred = sum(w[i] * allSafePreds[si][i] for i in range(5))
                totalSmape += _smape(allTestY[si], pred)
            return totalSmape / nSeries

        best = None
        bestVal = float('inf')
        for _ in range(5):
            x0 = np.random.randn(5) * 0.5
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6})
            if res.fun < bestVal:
                bestVal = res.fun
                best = res

        wOpt = np.exp(best.x)
        wOpt = wOpt / np.sum(wOpt)
        optimalWeights[groupName] = wOpt

        P(f"\n  {groupName}: optimal weights (sMAPE-minimized)")
        for mi, mName in enumerate(SAFE_MODELS):
            bar = '#' * int(wOpt[mi] * 50)
            P(f"    {mName:<15} {wOpt[mi]:.4f} {bar}")

        totalSmape = 0
        totalMase = 0
        n2SmapeAvg = np.mean([e['n2Smape'] for e in oracleData])
        n2MaseAvg = np.mean([e['n2Mase'] for e in oracleData])

        for si in range(nSeries):
            pred = sum(wOpt[i] * allSafePreds[si][i] for i in range(5))
            totalSmape += _smape(allTestY[si], pred)
            totalMase += _mase(allTrainY[si], allTestY[si], pred, seasonality)

        avgSmape = totalSmape / nSeries
        avgMase = totalMase / nSeries
        optOwa = 0.5 * (avgSmape / n2SmapeAvg + avgMase / n2MaseAvg)
        P(f"    OWA={optOwa:.4f}")

    return optimalWeights


def _evaluateOptimalStatic(allOracle, optimalWeights):
    P(f"\n{'='*70}")
    P(f"  Phase 3: Optimal Static vs DOT vs Ensemble")
    P(f"{'='*70}")

    strategies = ['oracle_best', 'dot_only', 'optimal_static', 'inv_smape_all5', 'equal_avg']
    groupResults = {}

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]
        horizon = info['horizon']
        seasonality = info['seasonality']
        nSeries = len(oracleData)
        wOpt = optimalWeights[groupName]

        stratSmapes = {s: [] for s in strategies}
        stratMases = {s: [] for s in strategies}

        for entry in oracleData:
            testY = entry['testY']
            trainY = entry['trainY']
            preds = entry['modelPreds']
            owas = entry['modelOwas']

            safePredsList = [preds.get(m, np.full(horizon, np.mean(trainY))) for m in SAFE_MODELS]
            safeSmapes = [_smape(testY, p) for p in safePredsList]
            safeOwas = [owas.get(m, 5.0) for m in SAFE_MODELS]

            bestIdx = np.argmin(safeOwas)
            stratSmapes['oracle_best'].append(_smape(testY, safePredsList[bestIdx]))
            stratMases['oracle_best'].append(_mase(trainY, testY, safePredsList[bestIdx], seasonality))

            dotIdx = SAFE_MODELS.index('dot')
            stratSmapes['dot_only'].append(_smape(testY, safePredsList[dotIdx]))
            stratMases['dot_only'].append(_mase(trainY, testY, safePredsList[dotIdx], seasonality))

            optPred = sum(wOpt[i] * safePredsList[i] for i in range(5))
            stratSmapes['optimal_static'].append(_smape(testY, optPred))
            stratMases['optimal_static'].append(_mase(trainY, testY, optPred, seasonality))

            wAll = np.array([1.0 / max(s, 0.01) for s in safeSmapes])
            wAll /= np.sum(wAll)
            all5Pred = sum(wAll[i] * safePredsList[i] for i in range(5))
            stratSmapes['inv_smape_all5'].append(_smape(testY, all5Pred))
            stratMases['inv_smape_all5'].append(_mase(trainY, testY, all5Pred, seasonality))

            avgPred = np.mean(safePredsList, axis=0)
            stratSmapes['equal_avg'].append(_smape(testY, avgPred))
            stratMases['equal_avg'].append(_mase(trainY, testY, avgPred, seasonality))

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
            marker = ' ***' if sName not in ('oracle_best', 'dot_only') and diff < -0.001 else ''
            P(f"    {sName:<25} OWA={sOwa:.4f} ({diff:+.4f}){marker}")

        groupResults[groupName] = groupOwa

    P(f"\n{'='*70}")
    P(f"  OVERALL (with optimal static)")
    P(f"{'='*70}")

    dotAvg = np.mean([groupResults[g]['dot_only'] for g in M4_GROUPS])
    for sName in strategies:
        vals = [groupResults[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if sName not in ('oracle_best', 'dot_only') and avg < dotAvg else ''
        P(f"  {sName:<25} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    return groupResults, optimalWeights


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 70)
    P("E038: Ensemble Weight Optimization (Safe Pool)")
    P("=" * 70)

    allOracle = _loadAllOracle()

    ensResults = _runEnsembleStrategies(allOracle)

    optWeights = _optimizeStaticWeights(allOracle)

    staticResults, _ = _evaluateOptimalStatic(allOracle, optWeights)

    P(f"\n{'='*70}")
    P(f"  FINAL SUMMARY")
    P(f"{'='*70}")
    dotAvg = np.mean([ensResults[g]['dot_only'] for g in M4_GROUPS])
    P(f"  DOT baseline:         {dotAvg:.4f}")
    P(f"  Oracle ceiling:       {np.mean([ensResults[g]['oracle_best'] for g in M4_GROUPS]):.4f}")

    for sName in ['equal_avg', 'inv_mape_top3', 'inv_smape_all5', 'trimmed_mean', 'median', 'drop_worst_avg']:
        avg = np.mean([ensResults[g][sName] for g in M4_GROUPS])
        P(f"  {sName:<25} {avg:.4f} ({avg - dotAvg:+.4f})")

    optStaticAvg = np.mean([staticResults[g]['optimal_static'] for g in M4_GROUPS])
    P(f"  optimal_static        {optStaticAvg:.4f} ({optStaticAvg - dotAvg:+.4f})")

    P(f"\n  E034 GBR meta_top1:   0.873 (reference)")
    P(f"  M4 #1 ES-RNN:        0.821")
    P(f"  M4 #2 FFORMA:        0.838")
    P(f"  M4 #3 Theta:          0.854")

    P(f"\n  Optimal static weights per frequency:")
    for g in M4_GROUPS:
        wStr = ' '.join([f"{SAFE_MODELS[i][:6]}={optWeights[g][i]:.3f}" for i in range(5)])
        P(f"    {g:<12} {wStr}")

    P("=" * 70)
