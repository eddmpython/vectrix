"""
==============================================================================
Experiment ID: modelCreation/029
Experiment: Residual Seasonal Correction for DOT-Hybrid
==============================================================================

Purpose:
- DOT-Hybrid may leave systematic residual patterns (uncaptured seasonality)
- If residuals show significant autocorrelation at seasonal lags,
  a simple correction (add mean residual at that lag) can improve forecasts
- This is a lightweight post-processing step, not a new model

Hypothesis:
1. Monthly residuals show ACF peaks at lag 12 -> correction improves OWA
2. Quarterly residuals show ACF peaks at lag 4 -> correction improves OWA
3. Weekly/Daily may benefit from lag-7 residual correction
4. Overall improvement: AVG OWA 0.885 -> <0.88

Method:
1. Fit DOT-Hybrid, get in-sample residuals
2. Compute ACF of residuals up to 2*period lags
3. If ACF at any lag > 0.2: compute mean residual at that lag offset
4. Add correction: pred[h] += mean_residual[(n+h) % significant_lag]
5. Also test: exponentially weighted residual correction (recent residuals matter more)
6. M4 6 groups x 300 series

Results (M4 6 groups x 300 series, 0.8 min):
             Yearly  Quarterly  Monthly  Weekly   Daily  Hourly   AVG
base         0.9865   0.9851    0.9831   1.0114  0.9984  0.8580  0.9704
corrected    0.9878   1.0410    1.1849   1.0241  0.9984  0.8603  1.0161
weighted     0.9870   1.0510    1.1895   1.0005  0.9983  0.8921  1.0197
autoCorr     0.9866   1.0444    1.1724   0.9942  0.9984  0.8901  1.0144

Conclusion:
- ALL HYPOTHESES REJECTED: Residual correction HURTS performance
- Quarterly: +0.056 regression (corrected), +0.066 (weighted)
- Monthly: +0.202 regression (corrected) = CATASTROPHIC
- Hourly: +0.002 to +0.034 regression depending on method
- Only Weekly autoCorr showed slight improvement (-0.017)
- ROOT CAUSE: DOT-Hybrid residuals are already near white noise for
  well-fitted series. Adding mean residual correction introduces
  systematic bias from in-sample noise, especially for seasonal series
  where the residual pattern doesn't extrapolate stably.
- Monthly/Quarterly catastrophe: residual ACF peaks are sample artifacts,
  not genuine uncaptured patterns. Correcting for them amplifies noise.
- REJECTED: Residual correction is harmful. DOT-Hybrid already captures
  the signal; residuals are genuinely unpredictable.

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


def _computeAcf(residuals, maxLag):
    n = len(residuals)
    if n < maxLag + 1:
        maxLag = n - 1
    if maxLag < 1:
        return np.array([])

    rMean = np.mean(residuals)
    rDemeaned = residuals - rMean
    var = np.sum(rDemeaned ** 2)
    if var < 1e-10:
        return np.zeros(maxLag + 1)

    acf = np.zeros(maxLag + 1)
    for lag in range(maxLag + 1):
        acf[lag] = np.sum(rDemeaned[:n - lag] * rDemeaned[lag:]) / var
    return acf


def _residualCorrection(residuals, horizon, n, corrLag, weighted=False):
    if corrLag < 1:
        return np.zeros(horizon)

    nResid = len(residuals)
    correction = np.zeros(horizon)

    if weighted:
        for h in range(horizon):
            offset = (n + h) % corrLag
            relevantResids = []
            relevantWeights = []
            for i in range(nResid):
                if i % corrLag == offset:
                    age = nResid - i
                    w = np.exp(-0.05 * age)
                    relevantResids.append(residuals[i])
                    relevantWeights.append(w)
            if relevantResids:
                wArr = np.array(relevantWeights)
                rArr = np.array(relevantResids)
                correction[h] = np.sum(wArr * rArr) / np.sum(wArr)
    else:
        meanResid = np.zeros(corrLag)
        counts = np.zeros(corrLag)
        for i in range(nResid):
            idx = i % corrLag
            meanResid[idx] += residuals[i]
            counts[idx] += 1
        for i in range(corrLag):
            if counts[i] > 0:
                meanResid[i] /= counts[i]

        for h in range(horizon):
            offset = (n + h) % corrLag
            correction[h] = meanResid[offset]

    return correction


def _processOneSeries(trainY, testY, horizon, period):
    n = len(trainY)
    if n < 20:
        return None

    naivePred = _naive2(trainY, horizon, period)
    smapeN = _smape(testY[:horizon], naivePred)
    maseN = _mase(trainY, testY[:horizon], naivePred, period)

    def computeOwa(pred):
        s = _smape(testY[:horizon], pred[:horizon])
        m = _mase(trainY, testY[:horizon], pred[:horizon], period)
        return 0.5 * (s / max(smapeN, 1e-10) + m / max(maseN, 1e-10))

    model = DynamicOptimizedTheta(period=period)
    model.fit(trainY)
    basePred, _, _ = model.predict(horizon)
    basePred = np.asarray(basePred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(basePred)):
        basePred = np.where(np.isfinite(basePred), basePred, np.mean(trainY))
    baseOwa = computeOwa(basePred)

    residuals = model.residuals
    if residuals is None or len(residuals) < 5:
        return {'base': baseOwa, 'corrected': baseOwa, 'weighted': baseOwa, 'autoCorr': baseOwa}

    candidateLags = []
    if period > 1:
        candidateLags.extend([period, period * 2])
    candidateLags.extend([7, 12, 4, 24, 52])
    candidateLags = sorted(set([l for l in candidateLags if 2 <= l < len(residuals) // 2]))

    if not candidateLags:
        return {'base': baseOwa, 'corrected': baseOwa, 'weighted': baseOwa, 'autoCorr': baseOwa}

    maxLag = max(candidateLags)
    acf = _computeAcf(residuals, maxLag)

    bestLag = None
    bestAcfVal = 0.0
    for lag in candidateLags:
        if lag < len(acf) and abs(acf[lag]) > 0.15 and abs(acf[lag]) > bestAcfVal:
            bestAcfVal = abs(acf[lag])
            bestLag = lag

    if bestLag is None:
        return {'base': baseOwa, 'corrected': baseOwa, 'weighted': baseOwa, 'autoCorr': baseOwa}

    corrSimple = _residualCorrection(residuals, horizon, n, bestLag, weighted=False)
    correctedPred = basePred + corrSimple
    correctedPred = np.asarray(correctedPred, dtype=np.float64)
    if not np.all(np.isfinite(correctedPred)):
        correctedPred = basePred.copy()
    correctedOwa = computeOwa(correctedPred)

    corrWeighted = _residualCorrection(residuals, horizon, n, bestLag, weighted=True)
    weightedPred = basePred + corrWeighted
    weightedPred = np.asarray(weightedPred, dtype=np.float64)
    if not np.all(np.isfinite(weightedPred)):
        weightedPred = basePred.copy()
    weightedOwa = computeOwa(weightedPred)

    bestAutoLag = None
    bestAutoAcf = 0.0
    searchLags = range(2, min(len(acf), n // 3))
    for lag in searchLags:
        if abs(acf[lag]) > 0.2 and abs(acf[lag]) > bestAutoAcf:
            bestAutoAcf = abs(acf[lag])
            bestAutoLag = lag

    if bestAutoLag is not None:
        corrAuto = _residualCorrection(residuals, horizon, n, bestAutoLag, weighted=True)
        autoPred = basePred + corrAuto
        autoPred = np.asarray(autoPred, dtype=np.float64)
        if not np.all(np.isfinite(autoPred)):
            autoPred = basePred.copy()
        autoOwa = computeOwa(autoPred)
    else:
        autoOwa = baseOwa

    return {'base': baseOwa, 'corrected': correctedOwa, 'weighted': weightedOwa, 'autoCorr': autoOwa}


def _runGroup(groupName, maxSeries=SAMPLE_PER_GROUP):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    period = info['seasonality']

    P(f'\n{"="*60}')
    P(f'{groupName} (h={horizon}, m={period}, sample={maxSeries})')
    P(f'{"="*60}')

    trainSeries, testSeries = _loadGroup(groupName, maxSeries)
    nSeries = len(trainSeries)

    methods = ['base', 'corrected', 'weighted', 'autoCorr']
    allOwas = {m: [] for m in methods}
    t0 = time.time()

    for idx in range(nSeries):
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period)
        if result is None:
            continue
        for m in methods:
            allOwas[m].append(result[m])

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            P(f'  [{idx+1}/{nSeries}] {(idx+1)/elapsed:.1f} series/s')

    elapsed = time.time() - t0
    avgOwas = {m: np.mean(allOwas[m]) for m in methods}

    P(f'Completed in {elapsed:.1f}s')
    for m in methods:
        P(f'  {m}: {avgOwas[m]:.4f}')

    return avgOwas


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('=' * 80)
    P('Residual Seasonal Correction — modelCreation/029')
    P('=' * 80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        groupResults[groupName] = _runGroup(groupName, SAMPLE_PER_GROUP)

    P('\n' + '=' * 80)
    P('FINAL RESULTS')
    P('=' * 80)

    methods = ['base', 'corrected', 'weighted', 'autoCorr']
    groups = list(M4_GROUPS.keys())

    P(f'\n{"Method":<12}', end='')
    for g in groups:
        P(f'{g:>12}', end='')
    P(f'{"AVG":>12}')
    P('-' * (12 + 12 * (len(groups) + 1)))

    for method in methods:
        P(f'{method:<12}', end='')
        vals = []
        for g in groups:
            v = groupResults[g][method]
            vals.append(v)
            P(f'{v:>12.4f}', end='')
        P(f'{np.mean(vals):>12.4f}')

    P(f'\nTotal time: {(time.time()-totalStart)/60:.1f} min')
