"""
==============================================================================
Experiment ID: modelCreation/041
Experiment: Conditional Ensemble/Selection Strategy Verification
==============================================================================

Purpose:
- E040 showed partial improvements: Monthly safe3 -0.016, Hourly cv_best -0.030
- E034 showed meta_top1 Hourly 0.670 (DOT 0.722, -7.2%)
- Verify whether frequency-conditional strategies improve overall OWA
- Test realistic pipeline: actual vectrix forecast() with modified ensemble logic

Hypothesis:
1. Monthly: safe3_core (dot+ces+4theta inv-MAPE) > DOT-only by ~0.016
2. Hourly: cv_best or safe3_core > DOT-only by ~0.006-0.030
3. Overall: conditional strategy should yield ~0.880 (DOT baseline 0.885)
4. No group should regress vs DOT baseline

Method:
1. Full M4 benchmark (6 groups, 2000 cap per group)
2. Compare strategies:
   - baseline: DOT-only everywhere
   - conditional_v1: DOT for Y/Q/W/D, safe3_core for M/H
   - conditional_v2: DOT for Y/Q/W/D, safe3_core for M, cv_best for H
   - always_safe3: safe3_core everywhere (control)
   - smart_safe3: safe3_core only if ensemble std closer to data std (variability check)
3. Key metric: per-group OWA and overall average OWA

Results (to be filled after experiment):

Conclusion:

Experiment date: 2026-03-04
==============================================================================
"""

import os
import sys
import time
import warnings

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
P = lambda *a, **kw: print(*a, **kw, flush=True)

CORE3 = ['dot', 'auto_ces', 'four_theta']


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


def _getModel(modelName, period):
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.fourTheta import AdaptiveThetaEnsemble

    factories = {
        'dot': lambda: DynamicOptimizedTheta(period=period),
        'auto_ces': lambda: AutoCES(period=period),
        'four_theta': lambda: AdaptiveThetaEnsemble(period=period),
    }
    return factories[modelName]()


def _fitPredict(model, trainY, horizon):
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred


def _fitPredictSafe(model, trainY, horizon, timeoutSec=15):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_fitPredict, model, trainY, horizon)
        return fut.result(timeout=timeoutSec)


def _runSeries(trainY, testY, horizon, seasonality, groupName):
    n = len(trainY)

    splitIdx = int(n * 0.8)
    if splitIdx < 5:
        splitIdx = max(5, n - horizon)
    cvTrainY = trainY[:splitIdx]
    cvTestY = trainY[splitIdx:]
    cvH = len(cvTestY)

    fullPreds = {}
    cvSmapes = {}

    for mName in CORE3:
        try:
            cvModel = _getModel(mName, seasonality)
            cvPred = _fitPredictSafe(cvModel, cvTrainY, cvH, timeoutSec=15)
            cvSmapes[mName] = _smape(cvTestY, cvPred[:len(cvTestY)])

            fullModel = _getModel(mName, seasonality)
            fullPreds[mName] = _fitPredictSafe(fullModel, trainY, horizon, timeoutSec=15)
        except Exception:
            cvSmapes[mName] = 999.0
            fullPreds[mName] = np.full(horizon, np.mean(trainY))

    dotPred = fullPreds.get('dot', np.full(horizon, np.mean(trainY)))

    preds = [fullPreds[m] for m in CORE3 if m in fullPreds]
    mapes = [cvSmapes[m] for m in CORE3 if m in cvSmapes]
    valid = [(p, m) for p, m in zip(preds, mapes) if np.all(np.isfinite(p)) and m < 200]
    if valid:
        weights = np.array([1.0 / (m + 1e-6) for _, m in valid])
        weights /= np.sum(weights)
        safe3Pred = sum(weights[i] * valid[i][0] for i in range(len(valid)))
    else:
        safe3Pred = dotPred

    bestModel = min(cvSmapes, key=cvSmapes.get)
    cvBestPred = fullPreds[bestModel]

    origStd = np.std(trainY[-min(30, len(trainY)):])
    safe3Std = np.std(safe3Pred)
    dotStd = np.std(dotPred)
    if abs(safe3Std - origStd) < abs(dotStd - origStd):
        smartPred = safe3Pred
    else:
        smartPred = dotPred

    return {
        'dot_only': dotPred,
        'safe3_core': safe3Pred,
        'cv_best': cvBestPred,
        'smart_safe3': smartPred,
    }


def _runGroup(groupName, sampleCap=2000):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    seasonality = info['seasonality']

    P(f"\n{'='*60}")
    P(f"  {groupName}: h={horizon}, m={seasonality}")
    P(f"{'='*60}")

    trainSeries, testSeries = _loadGroup(groupName)
    nSeries = len(trainSeries)

    validIdx = [i for i in range(nSeries)
                if len(trainSeries[i]) >= 10 and len(testSeries[i]) >= horizon]

    if len(validIdx) > sampleCap:
        rng = np.random.default_rng(42)
        validIdx = sorted(rng.choice(validIdx, size=sampleCap, replace=False).tolist())
    P(f"  Using {len(validIdx)} series")

    strategies = ['dot_only', 'safe3_core', 'cv_best', 'smart_safe3']
    stratSmapes = {s: [] for s in strategies}
    stratMases = {s: [] for s in strategies}
    n2Smapes = []
    n2Mases = []

    startTime = time.perf_counter()

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]

        n2pred = _naive2(trainY, horizon, seasonality)
        n2Smapes.append(_smape(testY, n2pred))
        n2Mases.append(_mase(trainY, testY, n2pred, seasonality))

        result = _runSeries(trainY, testY, horizon, seasonality, groupName)

        for sName in strategies:
            pred = result[sName]
            stratSmapes[sName].append(_smape(testY, pred))
            stratMases[sName].append(_mase(trainY, testY, pred, seasonality))

        if (count + 1) % 200 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} in {elapsed:.1f}s")

    n2SmapeAvg = np.mean(n2Smapes)
    n2MaseAvg = np.mean(n2Mases)

    groupOwa = {}
    P(f"  Results:")
    for sName in strategies:
        sSmape = np.mean(stratSmapes[sName])
        sMase = np.mean(stratMases[sName])
        sOwa = 0.5 * (sSmape / n2SmapeAvg + sMase / n2MaseAvg)
        groupOwa[sName] = sOwa
        diff = sOwa - groupOwa.get('dot_only', sOwa)
        marker = ' ***' if diff < -0.001 else ''
        P(f"    {sName:<20} OWA={sOwa:.4f} ({diff:+.4f}){marker}")

    return groupOwa, elapsed


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E041: Conditional Ensemble/Selection Strategy Verification")
    P("=" * 60)

    GROUP_CAPS = {
        'Yearly': 2000, 'Quarterly': 2000, 'Monthly': 2000,
        'Weekly': 359, 'Daily': 500, 'Hourly': 414,
    }

    allOwas = {}
    totalTime = 0
    for group in M4_GROUPS:
        owas, elapsed = _runGroup(group, sampleCap=GROUP_CAPS.get(group, 2000))
        allOwas[group] = owas
        totalTime += elapsed

    P(f"\n{'='*60}")
    P(f"  OVERALL ({totalTime/60:.1f} min)")
    P(f"{'='*60}")

    strategies = ['dot_only', 'safe3_core', 'cv_best', 'smart_safe3']
    dotAvg = np.mean([allOwas[g]['dot_only'] for g in M4_GROUPS])

    for sName in strategies:
        vals = [allOwas[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if avg < dotAvg - 0.001 else ''
        P(f"  {sName:<20} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    P(f"\n  === CONDITIONAL STRATEGIES ===")

    cond_v1 = {}
    cond_v2 = {}
    cond_v3 = {}
    for g in M4_GROUPS:
        cond_v1[g] = allOwas[g]['dot_only']
        cond_v2[g] = allOwas[g]['dot_only']
        cond_v3[g] = allOwas[g]['dot_only']

    cond_v1['Monthly'] = allOwas['Monthly']['safe3_core']
    cond_v1['Hourly'] = allOwas['Hourly']['safe3_core']

    cond_v2['Monthly'] = allOwas['Monthly']['safe3_core']
    cond_v2['Hourly'] = allOwas['Hourly']['cv_best']

    cond_v3['Monthly'] = allOwas['Monthly']['smart_safe3']
    cond_v3['Hourly'] = allOwas['Hourly']['smart_safe3']

    for name, cond in [('cond_v1 (M:safe3,H:safe3)', cond_v1),
                        ('cond_v2 (M:safe3,H:cvbest)', cond_v2),
                        ('cond_v3 (M:smart,H:smart)', cond_v3)]:
        vals = [cond[g] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if avg < dotAvg - 0.001 else ''
        P(f"  {name:<35} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    P(f"\n  M4 #1 ES-RNN=0.821  #2 FFORMA=0.838  #18 Theta=0.897")
    P(f"  DOT-Hybrid baseline={dotAvg:.4f}")
    P("=" * 60)
