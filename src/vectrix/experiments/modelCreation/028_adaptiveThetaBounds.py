"""
==============================================================================
Experiment ID: modelCreation/028
Experiment: Adaptive Theta Bounds (Data-Driven Optimization Range)
==============================================================================

Purpose:
- Current DOT-Hybrid uses fixed theta bounds [1.0, 50.0] for all series
- Short series (n<50) with wide bounds may select extreme theta -> overfitting
- Long stable series can benefit from wider bounds
- Adapt theta upper bound based on series length and volatility

Hypothesis:
1. Adaptive bounds improve Weekly OWA (short series): 0.957 -> <0.94
2. Adaptive bounds improve Daily OWA: 0.996 -> <0.98
3. No regression on Yearly/Quarterly where current bounds work well
4. Overall AVG OWA: 0.885 -> <0.88

Method:
1. Compute series characteristics: length n, CV (coefficient of variation)
2. Theta upper bound rules:
   - Short (n < 50): theta_max = 5 (conservative)
   - Medium (50 <= n < 200): theta_max = 15
   - Long (n >= 200): theta_max = 50 (current default)
   - High CV (> 0.5): reduce theta_max by 50%
3. Also test alpha bounds adaptation:
   - High volatility: alpha_min = 0.1 (more smoothing)
   - Low volatility: alpha_min = 0.01 (current default)
4. Compare: fixed bounds vs adaptive bounds vs tight bounds [1, 5]
5. M4 6 groups x 300 series

Results (M4 6 groups x 300 series, 5.5 min):
             Yearly  Quarterly  Monthly  Weekly   Daily  Hourly   AVG
base         0.9865   0.9851    0.9831   1.0114  0.9984  0.8580  0.9704
adaptive     0.9880   0.9775    0.9812   1.0091  0.9986  0.8580  0.9687
tight        0.9867   0.9775    0.9824   1.0093  0.9989  0.8580  0.9688

Conclusion:
- Hypothesis 1 REJECTED: Weekly improved only -0.002 (1.011 -> 1.009)
- Hypothesis 2 REJECTED: Daily unchanged (0.998)
- Hypothesis 3 CONFIRMED: Yearly no regression (+0.002 negligible)
- Hypothesis 4 REJECTED: AVG only -0.002, far from 0.88
- Quarterly got best improvement (-0.008) with both adaptive and tight bounds
- Monthly slight improvement (-0.002)
- Weekly/Daily/Hourly: nearly unchanged, period>=24 uses classic DOT
- Adaptive and tight bounds produce virtually identical results
  -> theta bounds are NOT the bottleneck; optimization converges to similar
  values regardless of upper bound (theta typically lands at 1-5 anyway)
- REJECTED: Marginal improvement not worth the added complexity.
  Theta bound adaptation is a solution to a non-problem.

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
from scipy.optimize import minimize, minimize_scalar

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


def _sesFilter(y, alpha):
    n = len(y)
    result = np.zeros(n)
    result[0] = y[0]
    for t in range(1, n):
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1]
    return result


def _sesSSE(y, alpha):
    n = len(y)
    level = y[0]
    sse = 0.0
    for t in range(1, n):
        error = y[t] - level
        sse += error * error
        level = alpha * y[t] + (1.0 - alpha) * level
    return sse


def _optimizeAlpha(y, alphaMin=0.001):
    if len(y) < 3:
        return 0.3
    result = minimize_scalar(lambda a: _sesSSE(y, a), bounds=(alphaMin, 0.999), method='bounded')
    return result.x if result.success else 0.3


def _linearReg(x, y):
    xMean = np.mean(x)
    yMean = np.mean(y)
    num = np.sum((x - xMean) * (y - yMean))
    den = np.sum((x - xMean) ** 2)
    slope = num / max(den, 1e-10)
    intercept = yMean - slope * xMean
    return slope, intercept


def _deseasonalizeAdv(y, period, seasonType):
    n = len(y)
    seasonal = np.zeros(period)
    counts = np.zeros(period)
    trend = np.convolve(y, np.ones(period) / period, mode='valid')
    offset = (period - 1) // 2

    if seasonType == 'multiplicative':
        for i in range(len(trend)):
            idx = i + offset
            if idx < n and trend[i] > 0:
                seasonal[idx % period] += y[idx] / trend[i]
                counts[idx % period] += 1
        for i in range(period):
            seasonal[i] = seasonal[i] / max(counts[i], 1)
        meanS = np.mean(seasonal)
        if meanS > 0:
            seasonal /= meanS
        seasonal = np.maximum(seasonal, 0.01)
        deseasonalized = y / seasonal[np.arange(n) % period]
    else:
        for i in range(len(trend)):
            idx = i + offset
            if idx < n:
                seasonal[idx % period] += y[idx] - trend[i]
                counts[idx % period] += 1
        for i in range(period):
            seasonal[i] = seasonal[i] / max(counts[i], 1)
        seasonal -= np.mean(seasonal)
        deseasonalized = y - seasonal[np.arange(n) % period]

    return seasonal, deseasonalized


def _fitHybridWithBounds(y, period, thetaMax=50.0, alphaMin=0.001):
    n = len(y)
    if n < 5:
        return np.full(1, y[-1] if n > 0 else 0)

    scaled = y.copy()
    base = np.mean(np.abs(scaled))
    if base > 0:
        scaled = scaled / base
    else:
        base = 1.0

    hasSeason = period > 1 and n >= period * 3
    seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']

    bestMae = np.inf
    bestResult = None

    for seasonType in seasonTypes:
        if seasonType != 'none':
            seasonal, deseasonalized = _deseasonalizeAdv(scaled, period, seasonType)
        else:
            seasonal = None
            deseasonalized = scaled

        ySafe = np.maximum(deseasonalized, 1e-10)
        x = np.arange(n, dtype=np.float64)

        for trendType in ['linear', 'exponential']:
            if trendType == 'exponential' and np.any(deseasonalized <= 0):
                continue
            if trendType == 'exponential':
                logY = np.log(ySafe)
                slope, intercept = _linearReg(x, logY)
                thetaLine0 = np.exp(intercept + slope * x)
            else:
                slope, intercept = _linearReg(x, deseasonalized)
                thetaLine0 = intercept + slope * x

            t0Safe = np.maximum(thetaLine0, 1e-10)

            for modelType in ['additive', 'multiplicative']:
                if modelType == 'multiplicative' and (np.any(thetaLine0 <= 0) or np.any(deseasonalized <= 0)):
                    continue

                isAdd = modelType == 'additive'

                def buildThetaLine(theta):
                    if isAdd:
                        return theta * deseasonalized + (1.0 - theta) * thetaLine0
                    return np.power(ySafe, theta) * np.power(t0Safe, 1.0 - theta)

                def objective(params, _isAdd=isAdd, _thetaLine0=thetaLine0, _deseasonalized=deseasonalized, _t0Safe=t0Safe, _ySafe=ySafe, _alphaMin=alphaMin):
                    theta = params[0]
                    if _isAdd:
                        tl = theta * _deseasonalized + (1.0 - theta) * _thetaLine0
                    else:
                        tl = np.power(_ySafe, theta) * np.power(_t0Safe, 1.0 - theta)
                    alpha = _optimizeAlpha(tl, _alphaMin)
                    filtered = _sesFilter(tl, alpha)
                    if _isAdd:
                        w = 1.0 / max(theta, 1.0)
                        fv = w * filtered + (1.0 - w) * _thetaLine0
                    else:
                        inv = 1.0 / max(theta, 1.0)
                        fv = np.power(np.maximum(filtered, 1e-10), inv) * np.power(_t0Safe, 1.0 - inv)
                    return np.mean(np.abs(_deseasonalized - fv))

                result = minimize(objective, x0=[2.0], bounds=[(1.0, thetaMax)],
                                  method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
                theta = result.x[0]

                thetaLine = buildThetaLine(theta)
                alpha = _optimizeAlpha(thetaLine, alphaMin)
                filtered = _sesFilter(thetaLine, alpha)
                lastLevel = filtered[-1]

                if isAdd:
                    w = 1.0 / max(theta, 1.0)
                    fittedVals = w * filtered + (1.0 - w) * thetaLine0
                else:
                    inv = 1.0 / max(theta, 1.0)
                    fittedVals = np.power(np.maximum(filtered, 1e-10), inv) * np.power(t0Safe, 1.0 - inv)

                if seasonal is not None:
                    fittedFull = fittedVals.copy()
                    for t in range(n):
                        idx = t % period
                        if seasonType == 'multiplicative':
                            fittedFull[t] *= seasonal[idx]
                        else:
                            fittedFull[t] += seasonal[idx]
                    fittedVals = fittedFull

                mae = np.mean(np.abs(scaled - fittedVals))
                if mae < bestMae:
                    bestMae = mae
                    bestResult = {
                        'theta': theta, 'alpha': alpha, 'intercept': intercept,
                        'slope': slope, 'lastLevel': lastLevel, 'base': base,
                        'seasonal': seasonal, 'trendType': trendType,
                        'modelType': modelType, 'seasonType': seasonType, 'n': n,
                    }

    return bestResult


def _predictFromResult(result, steps, period):
    if result is None:
        return None
    n = result['n']
    base = result['base']
    futureX = np.arange(n, n + steps, dtype=np.float64)

    if result['trendType'] == 'exponential':
        forecastTrend = np.exp(result['intercept'] + result['slope'] * futureX)
    else:
        forecastTrend = result['intercept'] + result['slope'] * futureX

    forecastSES = np.full(steps, result['lastLevel'])

    if result['modelType'] == 'additive':
        wses = 1.0 / max(result['theta'], 1.0)
        combined = wses * forecastSES + (1.0 - wses) * forecastTrend
    else:
        invTheta = 1.0 / max(result['theta'], 1.0)
        combined = np.power(np.maximum(forecastSES, 1e-10), invTheta) * \
                   np.power(np.maximum(forecastTrend, 1e-10), 1.0 - invTheta)

    seasonal = result['seasonal']
    seasonType = result['seasonType']
    if seasonal is not None:
        for h in range(steps):
            idx = (n + h) % period
            if seasonType == 'multiplicative':
                combined[h] *= seasonal[idx]
            else:
                combined[h] += seasonal[idx]

    return combined * base


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

    baseDot = DynamicOptimizedTheta(period=period)
    baseDot.fit(trainY)
    basePred, _, _ = baseDot.predict(horizon)
    basePred = np.asarray(basePred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(basePred)):
        basePred = np.where(np.isfinite(basePred), basePred, np.mean(trainY))
    baseOwa = computeOwa(basePred)

    if period >= 24:
        return {'base': baseOwa, 'adaptive': baseOwa, 'tight': baseOwa}

    cv = np.std(trainY) / max(np.mean(np.abs(trainY)), 1e-10)

    if n < 50:
        adaptiveThetaMax = 5.0
    elif n < 200:
        adaptiveThetaMax = 15.0
    else:
        adaptiveThetaMax = 50.0

    if cv > 0.5:
        adaptiveThetaMax = max(adaptiveThetaMax * 0.5, 2.0)

    adaptiveAlphaMin = 0.1 if cv > 0.5 else 0.001

    adaptiveResult = _fitHybridWithBounds(trainY, period, adaptiveThetaMax, adaptiveAlphaMin)
    if adaptiveResult is not None:
        adaptivePred = _predictFromResult(adaptiveResult, horizon, period)
        if adaptivePred is not None and np.all(np.isfinite(adaptivePred)):
            adaptiveOwa = computeOwa(adaptivePred)
        else:
            adaptiveOwa = baseOwa
    else:
        adaptiveOwa = baseOwa

    tightResult = _fitHybridWithBounds(trainY, period, 5.0, 0.01)
    if tightResult is not None:
        tightPred = _predictFromResult(tightResult, horizon, period)
        if tightPred is not None and np.all(np.isfinite(tightPred)):
            tightOwa = computeOwa(tightPred)
        else:
            tightOwa = baseOwa
    else:
        tightOwa = baseOwa

    return {'base': baseOwa, 'adaptive': adaptiveOwa, 'tight': tightOwa}


def _runGroup(groupName, maxSeries=SAMPLE_PER_GROUP):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    period = info['seasonality']

    P(f'\n{"="*60}')
    P(f'{groupName} (h={horizon}, m={period}, sample={maxSeries})')
    P(f'{"="*60}')

    trainSeries, testSeries = _loadGroup(groupName, maxSeries)
    nSeries = len(trainSeries)

    methods = ['base', 'adaptive', 'tight']
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
    P('Adaptive Theta Bounds — modelCreation/028')
    P('=' * 80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        groupResults[groupName] = _runGroup(groupName, SAMPLE_PER_GROUP)

    P('\n' + '=' * 80)
    P('FINAL RESULTS')
    P('=' * 80)

    methods = ['base', 'adaptive', 'tight']
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
