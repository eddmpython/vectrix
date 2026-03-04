"""
==============================================================================
Experiment ID: modelCreation/040
Experiment: Safe Ensemble Pipeline M4 Benchmark (No ARIMA)
==============================================================================

Purpose:
- E039 showed auto_arima destroys ensemble in Quarterly (OWA 1.6)
- E038 showed optimal weights: dot + auto_ces + four_theta (auto_ets/theta ~0)
- Test pipeline with 4 safe models: dot, auto_ces, four_theta, auto_ets
- Compare: DOT-only vs safe-4 ensemble vs safe-3 core ensemble

Hypothesis:
1. Safe-4 ensemble should not explode (no auto_arima)
2. DOT-only may still be best for Yearly
3. Target: AVG OWA <= 0.885 (at least match DOT-only)

Method:
1. For each M4 series: 80/20 split, train 4 safe models
2. Inverse-MAPE weighted ensemble of all valid models
3. Compare: dot_only, safe4_all, safe3_core (dot+ces+4theta only)

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

SAFE4 = ['dot', 'auto_ces', 'four_theta', 'auto_ets']
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
    from vectrix.engine.ets import AutoETS

    factories = {
        'dot': lambda: DynamicOptimizedTheta(period=period),
        'auto_ces': lambda: AutoCES(period=period),
        'four_theta': lambda: AdaptiveThetaEnsemble(period=period),
        'auto_ets': lambda: AutoETS(period=period),
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


def _invMapeEnsemble(preds, mapes):
    valid = [(p, m) for p, m in zip(preds, mapes) if np.all(np.isfinite(p)) and m < 200]
    if not valid:
        return preds[0] if preds else np.zeros(1)
    weights = np.array([1.0 / (m + 1e-6) for _, m in valid])
    weights /= np.sum(weights)
    return sum(weights[i] * valid[i][0] for i in range(len(valid)))


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

    strategies = ['dot_only', 'safe4_all', 'safe3_core', 'cv_best']
    stratSmapes = {s: [] for s in strategies}
    stratMases = {s: [] for s in strategies}
    n2Smapes = []
    n2Mases = []

    startTime = time.perf_counter()

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]
        n = len(trainY)

        n2pred = _naive2(trainY, horizon, seasonality)
        n2Smapes.append(_smape(testY, n2pred))
        n2Mases.append(_mase(trainY, testY, n2pred, seasonality))

        splitIdx = int(n * 0.8)
        if splitIdx < 5:
            splitIdx = max(5, n - horizon)
        cvTrainY = trainY[:splitIdx]
        cvTestY = trainY[splitIdx:]
        cvH = len(cvTestY)

        fullPreds = {}
        cvMapes = {}

        for mName in SAFE4:
            try:
                cvModel = _getModel(mName, seasonality)
                cvPred = _fitPredictSafe(cvModel, cvTrainY, cvH, timeoutSec=15)
                cvMapes[mName] = _smape(cvTestY, cvPred[:len(cvTestY)])

                fullModel = _getModel(mName, seasonality)
                fullPreds[mName] = _fitPredictSafe(fullModel, trainY, horizon, timeoutSec=15)
            except Exception:
                cvMapes[mName] = 999.0
                fullPreds[mName] = np.full(horizon, np.mean(trainY))

        dotPred = fullPreds.get('dot', np.full(horizon, np.mean(trainY)))
        stratSmapes['dot_only'].append(_smape(testY, dotPred))
        stratMases['dot_only'].append(_mase(trainY, testY, dotPred, seasonality))

        safe4Preds = [fullPreds[m] for m in SAFE4 if m in fullPreds]
        safe4Mapes = [cvMapes[m] for m in SAFE4 if m in cvMapes]
        safe4Ens = _invMapeEnsemble(safe4Preds, safe4Mapes)
        stratSmapes['safe4_all'].append(_smape(testY, safe4Ens))
        stratMases['safe4_all'].append(_mase(trainY, testY, safe4Ens, seasonality))

        core3Preds = [fullPreds[m] for m in CORE3 if m in fullPreds]
        core3Mapes = [cvMapes[m] for m in CORE3 if m in cvMapes]
        core3Ens = _invMapeEnsemble(core3Preds, core3Mapes)
        stratSmapes['safe3_core'].append(_smape(testY, core3Ens))
        stratMases['safe3_core'].append(_mase(trainY, testY, core3Ens, seasonality))

        bestModel = min(cvMapes, key=cvMapes.get)
        bestPred = fullPreds[bestModel]
        stratSmapes['cv_best'].append(_smape(testY, bestPred))
        stratMases['cv_best'].append(_mase(trainY, testY, bestPred, seasonality))

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
    P("E040: Safe Ensemble Pipeline M4 Benchmark")
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

    strategies = ['dot_only', 'safe4_all', 'safe3_core', 'cv_best']
    dotAvg = np.mean([allOwas[g]['dot_only'] for g in M4_GROUPS])

    for sName in strategies:
        vals = [allOwas[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if avg < dotAvg - 0.001 else ''
        P(f"  {sName:<20} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    P(f"\n  M4 #1 ES-RNN=0.821  #2 FFORMA=0.838  #3 Theta=0.854")
    P(f"  DOT-Hybrid baseline={dotAvg:.4f}")

    P(f"\n  Hybrid strategies:")
    for restStrat in ['safe4_all', 'safe3_core', 'cv_best']:
        hybVals = [allOwas['Yearly']['dot_only']] + [allOwas[g][restStrat] for g in list(M4_GROUPS.keys())[1:]]
        hybAvg = np.mean(hybVals)
        P(f"    dot_yearly + {restStrat}_rest: AVG={hybAvg:.4f} ({hybAvg - dotAvg:+.4f})")

    P("=" * 60)
