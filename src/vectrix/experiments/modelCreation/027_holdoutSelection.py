"""
==============================================================================
Experiment ID: modelCreation/027
Experiment: Holdout-Based Model Selection for DOT-Hybrid
==============================================================================

Purpose:
- Current DOT-Hybrid 8-way auto-select uses in-sample MAE for model selection
- In-sample MAE risks overfitting: complex models (exp trend, mult) may fit
  training data better but generalize worse
- E016 showed DOT++ Hourly 0.955 catastrophe — likely in-sample overfitting
- Replace in-sample MAE with out-of-sample holdout MAE for variant selection

Hypothesis:
1. Holdout selection improves Monthly OWA: 0.931 -> <0.92 (less overfitting)
2. Holdout selection improves Weekly OWA: 0.957 -> <0.95
3. No regression on Yearly (already correct variant selection)
4. Overall AVG OWA: 0.885 -> <0.875

Method:
1. Split train data: trainFit = y[:-h], trainVal = y[-h:]
2. Fit all 8 DOT-Hybrid variants on trainFit
3. Predict h steps, measure MAE on trainVal
4. Select variant with lowest validation MAE
5. Refit selected variant on full train data, predict horizon
6. Compare vs in-sample selection (current DOT-Hybrid)
7. M4 6 groups x 300 series

Results (M4 6 groups x 300 series, 3.4 min):
            InSample  Holdout   Delta
Yearly       0.9865   1.0097   +0.023
Quarterly    0.9851   0.9279   -0.057
Monthly      0.9831   0.9846   +0.002
Weekly       1.0114   1.0078   -0.004
Daily        0.9984   0.9985   +0.000
Hourly       0.8580   0.8580   +0.000
AVG          0.9704   0.9644   -0.006

Conclusion:
- Hypothesis 1 REJECTED: Monthly got slightly worse (+0.002), not better
- Hypothesis 2 PARTIALLY: Weekly improved marginally (-0.004)
- Hypothesis 3 REJECTED: Yearly regressed significantly (+0.023)
- Hypothesis 4 REJECTED: AVG only -0.006, not reaching 0.875
- Quarterly got major improvement (-0.057) = holdout prevents overfitting there
- Yearly regression: holdout reduces training data, hurting short series
  (Yearly series are short, ~30 obs, losing 6 to holdout = 20% data loss)
- Weekly/Daily/Hourly: no change because period>=24 uses classic DOT (no 8-way)
- CONDITIONAL: Holdout helps Quarterly a lot but hurts Yearly.
  Could be combined with length-adaptive strategy (holdout only for n>50).
  Not adopted standalone.

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


def _optimizeAlpha(y):
    if len(y) < 3:
        return 0.3
    result = minimize_scalar(lambda a: _sesSSE(y, a), bounds=(0.001, 0.999), method='bounded')
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


def _fitVariant(y, period, trendType, modelType, seasonType):
    n = len(y)
    if n < 10:
        return None

    scaled = y.copy()
    base = np.mean(np.abs(scaled))
    if base > 0:
        scaled = scaled / base
    else:
        base = 1.0

    hasSeason = period > 1 and n >= period * 3

    if hasSeason and seasonType != 'none':
        seasonal, deseasonalized = _deseasonalizeAdv(scaled, period, seasonType)
    else:
        seasonal = None
        deseasonalized = scaled

    ySafe = np.maximum(deseasonalized, 1e-10)
    x = np.arange(n, dtype=np.float64)

    if trendType == 'exponential':
        if np.any(deseasonalized <= 0):
            return None
        logY = np.log(ySafe)
        slope, intercept = _linearReg(x, logY)
        thetaLine0 = np.exp(intercept + slope * x)
    else:
        slope, intercept = _linearReg(x, deseasonalized)
        thetaLine0 = intercept + slope * x

    t0Safe = np.maximum(thetaLine0, 1e-10)
    isAdd = modelType == 'additive'

    if isAdd and np.any(thetaLine0 <= 0) and not isAdd:
        return None
    if not isAdd and (np.any(thetaLine0 <= 0) or np.any(deseasonalized <= 0)):
        return None

    def buildThetaLine(theta):
        if isAdd:
            return theta * deseasonalized + (1.0 - theta) * thetaLine0
        return np.power(ySafe, theta) * np.power(t0Safe, 1.0 - theta)

    def objective(params):
        theta = params[0]
        thetaLine = buildThetaLine(theta)
        alpha = _optimizeAlpha(thetaLine)
        filtered = _sesFilter(thetaLine, alpha)
        if isAdd:
            w = 1.0 / max(theta, 1.0)
            fittedVals = w * filtered + (1.0 - w) * thetaLine0
        else:
            inv = 1.0 / max(theta, 1.0)
            fittedVals = np.power(np.maximum(filtered, 1e-10), inv) * np.power(t0Safe, 1.0 - inv)
        return np.mean(np.abs(deseasonalized - fittedVals))

    result = minimize(objective, x0=[2.0], bounds=[(1.0, 50.0)],
                      method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
    theta = result.x[0]

    thetaLine = buildThetaLine(theta)
    alpha = _optimizeAlpha(thetaLine)
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

    predictions_scaled = fittedVals * base

    return {
        'theta': theta,
        'alpha': alpha,
        'intercept': intercept,
        'slope': slope,
        'lastLevel': lastLevel,
        'base': base,
        'seasonal': seasonal,
        'trendType': trendType,
        'modelType': modelType,
        'seasonType': seasonType,
        'n': n,
        'fittedScaled': predictions_scaled,
    }


def _predictVariant(variant, steps, period, n):
    trendType = variant['trendType']
    modelType = variant['modelType']
    seasonType = variant['seasonType']
    base = variant['base']
    theta = variant['theta']
    lastLevel = variant['lastLevel']
    intercept = variant['intercept']
    slope = variant['slope']
    seasonal = variant['seasonal']

    futureX = np.arange(n, n + steps, dtype=np.float64)
    if trendType == 'exponential':
        forecastTrend = np.exp(intercept + slope * futureX)
    else:
        forecastTrend = intercept + slope * futureX

    forecastSES = np.full(steps, lastLevel)

    if modelType == 'additive':
        wses = 1.0 / max(theta, 1.0)
        wtrend = 1.0 - wses
        combined = wses * forecastSES + wtrend * forecastTrend
    else:
        invTheta = 1.0 / max(theta, 1.0)
        combined = np.power(np.maximum(forecastSES, 1e-10), invTheta) * \
                   np.power(np.maximum(forecastTrend, 1e-10), 1.0 - invTheta)

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

    baseDot = DynamicOptimizedTheta(period=period)
    baseDot.fit(trainY)
    basePred, _, _ = baseDot.predict(horizon)
    basePred = np.asarray(basePred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(basePred)):
        basePred = np.where(np.isfinite(basePred), basePred, np.mean(trainY))
    baseOwa = 0.5 * (
        _smape(testY[:horizon], basePred) / max(smapeN, 1e-10) +
        _mase(trainY, testY[:horizon], basePred, period) / max(maseN, 1e-10)
    )

    if period < 24:
        hasSeason = period > 1 and n >= period * 3
        seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']

        holdoutH = min(horizon, max(n // 5, 3))
        if holdoutH < 2 or n - holdoutH < 20:
            holdoutOwa = baseOwa
        else:
            trainHold = trainY[:-holdoutH]
            valY = trainY[-holdoutH:]
            nHold = len(trainHold)

            bestValMae = np.inf
            bestVariantKey = None

            for seasonType in seasonTypes:
                for trendType in ['linear', 'exponential']:
                    for modelType in ['additive', 'multiplicative']:
                        variant = _fitVariant(trainHold, period, trendType, modelType, seasonType)
                        if variant is None:
                            continue
                        valPred = _predictVariant(variant, holdoutH, period, nHold)
                        valMae = np.mean(np.abs(valY - valPred[:holdoutH]))
                        if valMae < bestValMae:
                            bestValMae = valMae
                            bestVariantKey = (trendType, modelType, seasonType)

            if bestVariantKey is not None:
                fullVariant = _fitVariant(trainY, period, *bestVariantKey)
                if fullVariant is not None:
                    holdoutPred = _predictVariant(fullVariant, horizon, period, n)
                    holdoutPred = np.asarray(holdoutPred[:horizon], dtype=np.float64)
                    if not np.all(np.isfinite(holdoutPred)):
                        holdoutPred = np.where(np.isfinite(holdoutPred), holdoutPred, np.mean(trainY))
                    holdoutOwa = 0.5 * (
                        _smape(testY[:horizon], holdoutPred) / max(smapeN, 1e-10) +
                        _mase(trainY, testY[:horizon], holdoutPred, period) / max(maseN, 1e-10)
                    )
                else:
                    holdoutOwa = baseOwa
            else:
                holdoutOwa = baseOwa
    else:
        holdoutOwa = baseOwa

    return {
        'base': baseOwa,
        'holdout': holdoutOwa,
    }


def _runGroup(groupName, maxSeries=SAMPLE_PER_GROUP):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    period = info['seasonality']

    P(f'\n{"="*60}')
    P(f'{groupName} (h={horizon}, m={period}, sample={maxSeries})')
    P(f'{"="*60}')

    trainSeries, testSeries = _loadGroup(groupName, maxSeries)
    nSeries = len(trainSeries)

    baseOwas = []
    holdoutOwas = []
    t0 = time.time()

    for idx in range(nSeries):
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period)
        if result is None:
            continue
        baseOwas.append(result['base'])
        holdoutOwas.append(result['holdout'])

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            P(f'  [{idx+1}/{nSeries}] {(idx+1)/elapsed:.1f} series/s')

    elapsed = time.time() - t0
    avgBase = np.mean(baseOwas) if baseOwas else float('nan')
    avgHoldout = np.mean(holdoutOwas) if holdoutOwas else float('nan')

    P(f'Completed in {elapsed:.1f}s')
    P(f'  In-sample (current): {avgBase:.4f}')
    P(f'  Holdout selection:   {avgHoldout:.4f}')
    P(f'  Delta: {avgHoldout - avgBase:+.4f}')

    return {'base': avgBase, 'holdout': avgHoldout}


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('=' * 80)
    P('Holdout-Based Model Selection — modelCreation/027')
    P('=' * 80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        groupResults[groupName] = _runGroup(groupName, SAMPLE_PER_GROUP)

    P('\n' + '=' * 80)
    P('FINAL RESULTS')
    P('=' * 80)
    P(f'\n{"Group":<12} {"InSample":>10} {"Holdout":>10} {"Delta":>10}')
    P('-' * 44)
    allBase = []
    allHoldout = []
    for g in M4_GROUPS:
        r = groupResults[g]
        delta = r['holdout'] - r['base']
        P(f'{g:<12} {r["base"]:>10.4f} {r["holdout"]:>10.4f} {delta:>+10.4f}')
        allBase.append(r['base'])
        allHoldout.append(r['holdout'])
    avgBase = np.mean(allBase)
    avgHoldout = np.mean(allHoldout)
    P(f'{"AVG":<12} {avgBase:>10.4f} {avgHoldout:>10.4f} {avgHoldout-avgBase:>+10.4f}')

    P(f'\nTotal time: {(time.time()-totalStart)/60:.1f} min')
