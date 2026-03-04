"""
==============================================================================
Experiment ID: modelCreation/039
Experiment: Improved Pipeline M4 Benchmark
==============================================================================

Purpose:
- Validate vectrix.py changes from E037/E038 findings
- Changes: improved model pool selection + always-ensemble + up-to-5 models
- Compare against DOT-only baseline (E019 = 0.885)
- Measure actual OWA improvement

Hypothesis:
1. New model selection + always-ensemble should beat DOT-only (0.885)
2. Improved Hourly (auto_ces in pool), keep Yearly stable
3. Target: AVG OWA <= 0.880

Method:
1. For each M4 series: simulate updated forecast() pipeline
   - Use new _selectNativeModels() logic
   - 80/20 split for holdout evaluation
   - Train 4-5 safe models, inverse-MAPE ensemble ALL valid
2. Compare: dot_only vs new_pipeline vs old_pipeline(top3)

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
    'Yearly':    {'horizon': 6,  'seasonality': 1, 'freq': 'Y'},
    'Quarterly': {'horizon': 8,  'seasonality': 4, 'freq': 'Q'},
    'Monthly':   {'horizon': 18, 'seasonality': 12, 'freq': 'M'},
    'Weekly':    {'horizon': 13, 'seasonality': 1, 'freq': 'W'},
    'Daily':     {'horizon': 14, 'seasonality': 1, 'freq': 'D'},
    'Hourly':    {'horizon': 48, 'seasonality': 24, 'freq': 'H'},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
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


def _getModel(modelName, period):
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
    from vectrix.engine.ets import AutoETS
    from vectrix.engine.theta import OptimizedTheta
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.dtsf import DynamicTimeScanForecaster
    from vectrix.engine.mstl import AutoMSTL

    factories = {
        'dot': lambda: DynamicOptimizedTheta(period=period),
        'auto_ces': lambda: AutoCES(period=period),
        'four_theta': lambda: AdaptiveThetaEnsemble(period=period),
        'auto_ets': lambda: AutoETS(period=period),
        'theta': lambda: OptimizedTheta(period=period),
        'auto_arima': lambda: AutoARIMA(seasonalPeriod=period),
        'dtsf': lambda: DynamicTimeScanForecaster(),
        'auto_mstl': lambda: AutoMSTL(),
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


def _selectModelsNew(freq, n, period, seasonalStrength=0.0, hasMultiSeason=False):
    if freq == 'H' and n >= 100:
        models = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'dtsf']
    elif (hasMultiSeason or seasonalStrength > 0.4) and n >= 60:
        models = ['dot', 'auto_ces', 'four_theta', 'auto_mstl', 'auto_ets']
    else:
        models = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_arima']

    if n < 30:
        models = [m for m in models if m not in ['auto_arima', 'dtsf']]
    if n < period * 2:
        models = [m for m in models if m not in ['auto_mstl']]

    if not models:
        models = ['dot', 'auto_ces']

    return models


def _selectModelsOld(freq, n, period, seasonalStrength=0.0, hasMultiSeason=False):
    if freq == 'H' and n >= 100:
        models = ['dot', 'auto_ces', 'dtsf', 'auto_mstl']
    elif (hasMultiSeason or seasonalStrength > 0.4) and n >= 60:
        models = ['dot', 'auto_ces', 'four_theta', 'auto_mstl', 'dtsf']
    else:
        models = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_arima']

    if n < 30:
        models = [m for m in models if m not in ['auto_arima', 'dtsf']]
    if n < period * 2:
        models = [m for m in models if m not in ['auto_mstl']]

    if not models:
        models = ['dot', 'auto_ces']

    return models


def _ensembleAllValid(modelPreds, modelMapes):
    validModels = [(mid, pred, mape) for mid, pred, mape in
                   zip(modelPreds.keys(), modelPreds.values(), modelMapes.values())
                   if np.all(np.isfinite(pred)) and mape < 200]

    if not validModels:
        return list(modelPreds.values())[0]

    sortedModels = sorted(validModels, key=lambda x: x[2])
    topModels = sortedModels[:min(len(sortedModels), 5)]

    weights = np.array([1.0 / (m[2] + 1e-6) for m in topModels])
    weights = weights / np.sum(weights)

    pred = sum(weights[i] * topModels[i][1] for i in range(len(topModels)))
    return pred


def _ensembleTop3(modelPreds, modelMapes):
    validModels = [(mid, pred, mape) for mid, pred, mape in
                   zip(modelPreds.keys(), modelPreds.values(), modelMapes.values())
                   if np.all(np.isfinite(pred)) and mape < 200]

    if not validModels:
        return list(modelPreds.values())[0]

    sortedModels = sorted(validModels, key=lambda x: x[2])
    topModels = sortedModels[:min(len(sortedModels), 3)]

    weights = np.array([1.0 / (m[2] + 1e-6) for m in topModels])
    weights = weights / np.sum(weights)

    pred = sum(weights[i] * topModels[i][1] for i in range(len(topModels)))
    return pred


def _runGroup(groupName, sampleCap=2000):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    seasonality = info['seasonality']
    freq = info['freq']

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

    strategies = ['dot_only', 'new_ensemble_all', 'new_ensemble_top3', 'old_pool_top3']
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

        newModels = _selectModelsNew(freq, n, seasonality)
        oldModels = _selectModelsOld(freq, n, seasonality)

        newPreds = {}
        newMapes = {}
        for mName in newModels:
            try:
                model = _getModel(mName, seasonality)
                cvPred = _fitPredictSafe(model, cvTrainY, cvH, timeoutSec=15)
                mape = _smape(cvTestY, cvPred[:len(cvTestY)])

                fullModel = _getModel(mName, seasonality)
                fullPred = _fitPredictSafe(fullModel, trainY, horizon, timeoutSec=15)
                newPreds[mName] = fullPred
                newMapes[mName] = mape
            except Exception:
                pass

        oldPreds = {}
        oldMapes = {}
        for mName in oldModels:
            if mName in newPreds:
                oldPreds[mName] = newPreds[mName]
                oldMapes[mName] = newMapes[mName]
            else:
                try:
                    model = _getModel(mName, seasonality)
                    cvPred = _fitPredictSafe(model, cvTrainY, cvH, timeoutSec=15)
                    mape = _smape(cvTestY, cvPred[:len(cvTestY)])

                    fullModel = _getModel(mName, seasonality)
                    fullPred = _fitPredictSafe(fullModel, trainY, horizon, timeoutSec=15)
                    oldPreds[mName] = fullPred
                    oldMapes[mName] = mape
                except Exception:
                    pass

        if 'dot' in newPreds:
            dotPred = newPreds['dot']
        elif 'dot' in oldPreds:
            dotPred = oldPreds['dot']
        else:
            dotModel = _getModel('dot', seasonality)
            dotPred = _fitPredictSafe(dotModel, trainY, horizon)

        stratSmapes['dot_only'].append(_smape(testY, dotPred))
        stratMases['dot_only'].append(_mase(trainY, testY, dotPred, seasonality))

        if newPreds:
            newAllPred = _ensembleAllValid(newPreds, newMapes)
            stratSmapes['new_ensemble_all'].append(_smape(testY, newAllPred))
            stratMases['new_ensemble_all'].append(_mase(trainY, testY, newAllPred, seasonality))

            newTop3Pred = _ensembleTop3(newPreds, newMapes)
            stratSmapes['new_ensemble_top3'].append(_smape(testY, newTop3Pred))
            stratMases['new_ensemble_top3'].append(_mase(trainY, testY, newTop3Pred, seasonality))
        else:
            stratSmapes['new_ensemble_all'].append(_smape(testY, dotPred))
            stratMases['new_ensemble_all'].append(_mase(trainY, testY, dotPred, seasonality))
            stratSmapes['new_ensemble_top3'].append(_smape(testY, dotPred))
            stratMases['new_ensemble_top3'].append(_mase(trainY, testY, dotPred, seasonality))

        if oldPreds:
            oldTop3Pred = _ensembleTop3(oldPreds, oldMapes)
            stratSmapes['old_pool_top3'].append(_smape(testY, oldTop3Pred))
            stratMases['old_pool_top3'].append(_mase(trainY, testY, oldTop3Pred, seasonality))
        else:
            stratSmapes['old_pool_top3'].append(_smape(testY, dotPred))
            stratMases['old_pool_top3'].append(_mase(trainY, testY, dotPred, seasonality))

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
    for sName in strategies:
        sSmape = np.mean(stratSmapes[sName])
        sMase = np.mean(stratMases[sName])
        sOwa = 0.5 * (sSmape / n2SmapeAvg + sMase / n2MaseAvg)
        groupOwa[sName] = sOwa

    P(f"  Results:")
    for sName in strategies:
        diff = groupOwa[sName] - groupOwa['dot_only']
        marker = ' ***' if diff < -0.001 else ''
        P(f"    {sName:<25} OWA={groupOwa[sName]:.4f} ({diff:+.4f}){marker}")

    return groupOwa, elapsed


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E039: Improved Pipeline M4 Benchmark")
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

    strategies = ['dot_only', 'new_ensemble_all', 'new_ensemble_top3', 'old_pool_top3']
    dotAvg = np.mean([allOwas[g]['dot_only'] for g in M4_GROUPS])

    for sName in strategies:
        vals = [allOwas[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if avg < dotAvg - 0.001 else ''
        P(f"  {sName:<25} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    P(f"\n  M4 #1 ES-RNN=0.821  #2 FFORMA=0.838  #3 Theta=0.854")
    P(f"  DOT-Hybrid baseline={dotAvg:.4f}")
    P("=" * 60)
