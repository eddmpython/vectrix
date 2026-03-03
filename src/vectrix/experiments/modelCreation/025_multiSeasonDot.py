"""
==============================================================================
Experiment ID: modelCreation/025
Experiment: Multi-Season DOT-Hybrid (Auto Period Detection for Weekly/Daily)
==============================================================================

Purpose:
- Weekly(period=1) and Daily(period=1) are the biggest bottlenecks: OWA 0.957, 0.996
- These groups have NO seasonality declared, so DOT uses classic mode
- But real data often has hidden periodicity (Weekly: ~52 annual, Daily: 7-day week)
- Auto-detect dominant period via ACF peak, re-fit DOT-Hybrid with detected period

Hypothesis:
1. Auto period detection will find meaningful periods in 30%+ of Weekly/Daily series
2. Re-fitting with detected period improves Weekly OWA: 0.957 -> <0.93
3. Re-fitting with detected period improves Daily OWA: 0.996 -> <0.97
4. No regression on already-strong groups (Yearly, Quarterly, Monthly, Hourly)

Method:
1. For each series: compute ACF up to lag=min(n//2, 365)
2. Find dominant period = argmax(ACF[2:]) + 2, require ACF > 0.1
3. If detected period differs from declared and n >= detected*3:
   fit DOT with detected period, compare OOS vs original
4. Pick whichever has lower OOS MAE on last-h holdout
5. M4 6 groups x 300 series, compare vs baseline DOT-Hybrid

Results (M4 6 groups x 300 series, 2.3 min):
            Baseline  MultiSeason  Delta   Multi Used
Yearly       0.9865     0.9897    +0.003    51/236 (21.6%)
Quarterly    0.9851     0.9838    -0.001   106/300 (35.3%)
Monthly      0.9831     0.9849    +0.002   123/300 (41.0%)
Weekly       1.0114     1.0185    +0.007    49/300 (16.3%)
Daily        0.9984     0.9982    -0.000    16/300 (5.3%)
Hourly       0.8580     0.8343    -0.024    25/300 (8.3%)
AVG          0.9704     0.9682    -0.002

Detected periods: Yearly[2], Quarterly[2,3], Monthly[2,3,6,7,11,13,14,21],
  Weekly[2,3,4,5,13,52], Daily[2], Hourly[2,16,17,23,72,168]

Conclusion:
- Hypothesis 1 PARTIALLY CONFIRMED: 5-41% of series had detected periods
- Hypothesis 2 REJECTED: Weekly got WORSE (+0.007), detected periods mostly noise
- Hypothesis 3 REJECTED: Daily nearly unchanged (-0.000)
- Hypothesis 4 PARTIALLY: Hourly improved nicely (-0.024) by detecting 168h period
- Overall AVG improvement marginal (-0.002), not worth the 3x computation cost
- Root cause: ACF peak detection finds spurious short periods (2,3) that hurt
  rather than help. Only Hourly with clear multi-seasonal structure benefits.
- REJECTED as standalone improvement. Hourly multi-season insight could be
  combined with other approaches (E026 DOT+CES).

Experiment date: 2026-03-04
==============================================================================
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from vectrix.engine.dot import DynamicOptimizedTheta

P = lambda *a, **kw: print(*a, **kw, flush=True)

M4_GROUPS = {
    'Yearly':    {'horizon': 6,  'seasonality': 1},
    'Quarterly': {'horizon': 8,  'seasonality': 4},
    'Monthly':   {'horizon': 18, 'seasonality': 12},
    'Weekly':    {'horizon': 13, 'seasonality': 1},
    'Daily':     {'horizon': 14, 'seasonality': 1},
    'Hourly':    {'horizon': 48, 'seasonality': 24},
}

SAMPLE_PER_GROUP = 300
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')


def _loadGroup(groupName, maxSeries=None):
    trainPath = os.path.join(DATA_DIR, f'{groupName}-train.csv')
    testPath = os.path.join(DATA_DIR, f'{groupName}-test.csv')
    trainDf = pd.read_csv(trainPath)
    testDf = pd.read_csv(testPath)
    nTotal = len(trainDf)
    if maxSeries and maxSeries < nTotal:
        rng = np.random.default_rng(42)
        indices = rng.choice(nTotal, maxSeries, replace=False)
        indices.sort()
    else:
        indices = np.arange(nTotal)
    trainSeries = []
    testSeries = []
    for i in indices:
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


def _detectPeriod(y, maxLag=None):
    n = len(y)
    if maxLag is None:
        maxLag = min(n // 2, 365)
    if maxLag < 4:
        return None

    yMean = np.mean(y)
    yDemeaned = y - yMean
    var = np.sum(yDemeaned ** 2)
    if var < 1e-10:
        return None

    acf = np.correlate(yDemeaned, yDemeaned, mode='full')
    acf = acf[n - 1:n - 1 + maxLag + 1] / var

    if len(acf) < 4:
        return None

    searchAcf = acf[2:]
    if len(searchAcf) == 0:
        return None

    bestLag = np.argmax(searchAcf) + 2
    if acf[bestLag] < 0.1:
        return None

    if bestLag >= n // 3:
        return None

    return bestLag


def _fitPredict(trainY, horizon, period):
    model = DynamicOptimizedTheta(period=period)
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred


def _processOneSeries(trainY, testY, horizon, declaredPeriod):
    n = len(trainY)
    if n < 20:
        return None

    naivePred = _naive2(trainY, horizon, declaredPeriod)
    smapeN = _smape(testY[:horizon], naivePred)
    maseN = _mase(trainY, testY[:horizon], naivePred, declaredPeriod)

    basePred = _fitPredict(trainY, horizon, declaredPeriod)
    baseSmape = _smape(testY[:horizon], basePred)
    baseMase = _mase(trainY, testY[:horizon], basePred, declaredPeriod)
    baseOwa = 0.5 * (baseSmape / max(smapeN, 1e-10) + baseMase / max(maseN, 1e-10))

    detectedPeriod = _detectPeriod(trainY)

    if detectedPeriod is not None and detectedPeriod != declaredPeriod and n >= detectedPeriod * 3:
        multiPred = _fitPredict(trainY, horizon, detectedPeriod)
        multiSmape = _smape(testY[:horizon], multiPred)
        multiMase = _mase(trainY, testY[:horizon], multiPred, declaredPeriod)
        multiOwa = 0.5 * (multiSmape / max(smapeN, 1e-10) + multiMase / max(maseN, 1e-10))

        holdoutH = min(horizon, n // 5)
        if holdoutH >= 2:
            trainHold = trainY[:-holdoutH]
            valY = trainY[-holdoutH:]

            basePredH = _fitPredict(trainHold, holdoutH, declaredPeriod)
            multiPredH = _fitPredict(trainHold, holdoutH, detectedPeriod)

            baseMaeH = np.mean(np.abs(valY - basePredH[:holdoutH]))
            multiMaeH = np.mean(np.abs(valY - multiPredH[:holdoutH]))

            if multiMaeH < baseMaeH * 0.98:
                bestOwa = multiOwa
                bestMethod = 'multi'
            else:
                bestOwa = baseOwa
                bestMethod = 'base'
        else:
            bestOwa = baseOwa
            bestMethod = 'base'
    else:
        bestOwa = baseOwa
        bestMethod = 'base'
        multiOwa = baseOwa
        detectedPeriod = declaredPeriod

    return {
        'baseOwa': baseOwa,
        'multiOwa': multiOwa,
        'bestOwa': bestOwa,
        'bestMethod': bestMethod,
        'detectedPeriod': detectedPeriod,
        'usedMulti': bestMethod == 'multi',
    }


def _runGroup(groupName, maxSeries=SAMPLE_PER_GROUP):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    period = info['seasonality']

    P(f'\n{"="*60}')
    P(f'{groupName} (h={horizon}, declared_m={period}, sample={maxSeries})')
    P(f'{"="*60}')

    trainSeries, testSeries = _loadGroup(groupName, maxSeries)
    nSeries = len(trainSeries)

    baseOwas = []
    multiOwas = []
    bestOwas = []
    detectedPeriods = []
    multiUsed = 0
    t0 = time.time()

    for idx in range(nSeries):
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period)
        if result is None:
            continue
        baseOwas.append(result['baseOwa'])
        multiOwas.append(result['multiOwa'])
        bestOwas.append(result['bestOwa'])
        if result['usedMulti']:
            multiUsed += 1
            detectedPeriods.append(result['detectedPeriod'])

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            P(f'  [{idx+1}/{nSeries}] {(idx+1)/elapsed:.1f} series/s')

    elapsed = time.time() - t0
    avgBase = np.mean(baseOwas) if baseOwas else float('nan')
    avgBest = np.mean(bestOwas) if bestOwas else float('nan')

    P(f'Completed in {elapsed:.1f}s')
    P(f'  Baseline OWA: {avgBase:.4f}')
    P(f'  Multi-Season OWA: {avgBest:.4f}')
    P(f'  Multi used: {multiUsed}/{len(baseOwas)} ({100*multiUsed/max(len(baseOwas),1):.1f}%)')
    if detectedPeriods:
        P(f'  Detected periods: {sorted(set(detectedPeriods))}')

    return {
        'baseOwa': avgBase,
        'bestOwa': avgBest,
        'multiUsed': multiUsed,
        'total': len(baseOwas),
        'periods': detectedPeriods,
    }


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('=' * 80)
    P('Multi-Season DOT-Hybrid — modelCreation/025')
    P('=' * 80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        groupResults[groupName] = _runGroup(groupName, SAMPLE_PER_GROUP)

    P('\n' + '=' * 80)
    P('FINAL RESULTS')
    P('=' * 80)
    P(f'\n{"Group":<12} {"Baseline":>10} {"MultiSeason":>12} {"Delta":>8} {"Used":>8}')
    P('-' * 52)
    allBase = []
    allBest = []
    for g in M4_GROUPS:
        r = groupResults[g]
        delta = r['bestOwa'] - r['baseOwa']
        P(f'{g:<12} {r["baseOwa"]:>10.4f} {r["bestOwa"]:>12.4f} {delta:>+8.4f} {r["multiUsed"]:>4}/{r["total"]}')
        allBase.append(r['baseOwa'])
        allBest.append(r['bestOwa'])
    avgBase = np.mean(allBase)
    avgBest = np.mean(allBest)
    P(f'{"AVG":<12} {avgBase:>10.4f} {avgBest:>12.4f} {avgBest-avgBase:>+8.4f}')

    P(f'\nTotal time: {(time.time()-totalStart)/60:.1f} min')
