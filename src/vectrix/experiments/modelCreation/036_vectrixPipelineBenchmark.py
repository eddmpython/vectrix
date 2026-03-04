"""
==============================================================================
Experiment ID: modelCreation/036
Experiment: Vectrix Full Pipeline M4 Benchmark
==============================================================================

Purpose:
- Measure actual vectrix.forecast() pipeline performance on M4
- Compare against DOT-only (E019 = 0.885)
- Identify if the current pipeline model selection already improves on DOT
- Test improved model pools based on E031 oracle analysis

Hypothesis:
1. Current vectrix.forecast() should match or beat DOT-only (0.885)
2. Improved model pools (oracle-informed) should do better

Method:
1. Run DOT-only on M4 2000/group (E019 reference)
2. For each series, run 5 safe models with holdout validation (simulating forecast())
3. Compare: DOT-only vs holdout-best vs holdout-top2-weighted
4. Use safe models only: dot, auto_ces, four_theta, auto_ets, theta

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

SAFE_MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'theta']


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
    from vectrix.engine.ets import AutoETS
    from vectrix.engine.theta import OptimizedTheta

    factories = {
        'dot': lambda: DynamicOptimizedTheta(period=period),
        'auto_ces': lambda: AutoCES(period=period),
        'four_theta': lambda: AdaptiveThetaEnsemble(period=period),
        'auto_ets': lambda: AutoETS(period=period),
        'theta': lambda: OptimizedTheta(period=period),
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
                if len(trainSeries[i]) >= 5 and len(testSeries[i]) >= horizon]

    if len(validIdx) > sampleCap:
        rng = np.random.default_rng(42)
        validIdx = sorted(rng.choice(validIdx, size=sampleCap, replace=False).tolist())
    P(f"  Using {len(validIdx)} series")

    strategies = ['dot_only', 'cv_best', 'cv_top2']
    stratSmapes = {s: [] for s in strategies}
    stratMases = {s: [] for s in strategies}
    n2Smapes = []
    n2Mases = []
    selCounts = {m: 0 for m in SAFE_MODELS}

    startTime = time.perf_counter()

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]

        n2pred = _naive2(trainY, horizon, seasonality)
        n2Smapes.append(_smape(testY, n2pred))
        n2Mases.append(_mase(trainY, testY, n2pred, seasonality))

        splitIdx = int(len(trainY) * 0.8)
        if splitIdx < 5 or len(trainY) - splitIdx < 1:
            splitIdx = max(5, len(trainY) - horizon)
        cvTrainY = trainY[:splitIdx]
        cvTestY = trainY[splitIdx:]
        cvH = len(cvTestY)

        if cvH < 1 or len(cvTrainY) < 5:
            dotModel = _getModel('dot', seasonality)
            dotPred = _fitPredictSafe(dotModel, trainY, horizon)
            for s in strategies:
                stratSmapes[s].append(_smape(testY, dotPred))
                stratMases[s].append(_mase(trainY, testY, dotPred, seasonality))
            selCounts['dot'] += 1
            continue

        cvScores = {}
        fullPreds = {}

        for mName in SAFE_MODELS:
            try:
                cvModel = _getModel(mName, seasonality)
                cvPred = _fitPredictSafe(cvModel, cvTrainY, cvH, timeoutSec=10)
                cvScores[mName] = _smape(cvTestY, cvPred[:len(cvTestY)])

                fullModel = _getModel(mName, seasonality)
                fullPreds[mName] = _fitPredictSafe(fullModel, trainY, horizon, timeoutSec=10)
            except Exception:
                cvScores[mName] = 999.0
                fullPreds[mName] = np.full(horizon, np.mean(trainY))

        dotPred = fullPreds['dot']
        stratSmapes['dot_only'].append(_smape(testY, dotPred))
        stratMases['dot_only'].append(_mase(trainY, testY, dotPred, seasonality))

        bestModel = min(cvScores, key=cvScores.get)
        selCounts[bestModel] += 1
        bestPred = fullPreds[bestModel]
        stratSmapes['cv_best'].append(_smape(testY, bestPred))
        stratMases['cv_best'].append(_mase(trainY, testY, bestPred, seasonality))

        sortedModels = sorted(cvScores, key=cvScores.get)
        top2 = sortedModels[:2]
        w = np.array([1.0 / max(cvScores[m], 0.01) for m in top2])
        w = w / np.sum(w)
        top2Pred = sum(w[i] * fullPreds[top2[i]] for i in range(2))
        stratSmapes['cv_top2'].append(_smape(testY, top2Pred))
        stratMases['cv_top2'].append(_mase(trainY, testY, top2Pred, seasonality))

        if (count + 1) % 200 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} in {elapsed:.1f}s")
    P(f"  Selections: {selCounts}")

    n2SmapeAvg = np.mean(n2Smapes)
    n2MaseAvg = np.mean(n2Mases)

    groupOwa = {}
    for sName in strategies:
        sSmape = np.mean(stratSmapes[sName])
        sMase = np.mean(stratMases[sName])
        sOwa = 0.5 * (sSmape / n2SmapeAvg + sMase / n2MaseAvg)
        groupOwa[sName] = sOwa

    P(f"  Results:")
    for sName in strategies:
        diff = groupOwa[sName] - groupOwa['dot_only']
        marker = ' ***' if diff < -0.001 else ''
        P(f"    {sName:<20} OWA={groupOwa[sName]:.4f} ({diff:+.4f}){marker}")

    return groupOwa, elapsed


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E036: Vectrix Pipeline Benchmark (CV-based Model Selection)")
    P("=" * 60)

    GROUP_CAPS = {
        'Yearly': 2000, 'Quarterly': 2000, 'Monthly': 2000,
        'Weekly': 2000, 'Daily': 500, 'Hourly': 414,
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

    strategies = ['dot_only', 'cv_best', 'cv_top2']
    dotAvg = np.mean([allOwas[g]['dot_only'] for g in M4_GROUPS])

    for sName in strategies:
        vals = [allOwas[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if avg < dotAvg else ''
        P(f"  {sName:<20} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    P(f"\n  M4 #1 ES-RNN=0.821  #2 FFORMA=0.838  #3 Theta=0.854")
    P(f"  E019 DOT-Hybrid={dotAvg:.4f}")

    P(f"\n  Hybrid: DOT for Yearly + cv_best for rest:")
    hybridVals = [allOwas['Yearly']['dot_only']] + [allOwas[g]['cv_best'] for g in list(M4_GROUPS.keys())[1:]]
    P(f"    AVG={np.mean(hybridVals):.4f}")

    P(f"\n  Hybrid: DOT for Yearly + cv_top2 for rest:")
    hybridVals2 = [allOwas['Yearly']['dot_only']] + [allOwas[g]['cv_top2'] for g in list(M4_GROUPS.keys())[1:]]
    P(f"    AVG={np.mean(hybridVals2):.4f}")
