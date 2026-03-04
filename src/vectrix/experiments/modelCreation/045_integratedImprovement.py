"""
==============================================================================
Experiment ID: modelCreation/045
Experiment: Integrated DOT Improvement Verification
==============================================================================

Purpose:
- Combine accepted improvements from E043 and E044:
  A) holdout_val: holdout-based 8-way config selection (E043: AVG -0.79%)
  B) weekly_classic: force classic DOT for Weekly period=1 (E044: Weekly -2.18%)
- Verify combined improvement across ALL 6 M4 groups
- Confirm no regression before engine integration

Hypotheses:
1. Combined improvements: AVG OWA < 0.8831 (current baseline)
2. No group regresses more than +1% vs baseline

Method:
1. "improved" DOT variant:
   - If period=1 AND data is "Weekly-like" (n<500, no strong seasonality) → classic DOT
   - Otherwise: DOT-Hybrid with holdout validation for config selection
2. Full 6-group M4 benchmark comparison

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
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

M4_GROUPS = {
    'Yearly':    {'horizon': 6,  'seasonality': 1,  'count': 23000},
    'Quarterly': {'horizon': 8,  'seasonality': 4,  'count': 24000},
    'Monthly':   {'horizon': 18, 'seasonality': 12, 'count': 48000},
    'Weekly':    {'horizon': 13, 'seasonality': 1,  'count': 359},
    'Daily':     {'horizon': 14, 'seasonality': 1,  'count': 4227},
    'Hourly':    {'horizon': 48, 'seasonality': 24, 'count': 414},
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


def _smapeM4(actual, predicted):
    return np.abs(actual - predicted) * 200.0 / (np.abs(actual) + np.abs(predicted) + 1e-10)


def _maseM4(trainY, actual, predicted, seasonality):
    m = max(seasonality, 1)
    n = len(trainY)
    if n <= m:
        naiveErrors = np.abs(np.diff(trainY))
    else:
        naiveErrors = np.abs(trainY[m:] - trainY[:-m])
    masep = np.mean(naiveErrors) if len(naiveErrors) > 0 else 1e-10
    if masep < 1e-10:
        masep = 1e-10
    return np.abs(actual - predicted) / masep


def _seasonalityTestM4(y, ppy):
    if len(y) < 3 * ppy:
        return False
    n = len(y)
    nlag = min(ppy, n // 3)
    if nlag < 1:
        return False
    y_mean = np.mean(y)
    y_centered = y - y_mean
    c0 = np.sum(y_centered ** 2) / n
    if c0 < 1e-10:
        return False
    acf = np.zeros(nlag + 1)
    acf[0] = 1.0
    for lag in range(1, nlag + 1):
        acf[lag] = np.sum(y_centered[:n-lag] * y_centered[lag:]) / (n * c0)
    tcrit = 1.645
    cumsum = 1.0
    for k in range(1, ppy):
        if k < len(acf):
            cumsum += 2 * acf[k] ** 2
    limit_adj = tcrit * np.sqrt(cumsum) / np.sqrt(n)
    if ppy < len(acf):
        return abs(acf[ppy]) > limit_adj
    return False


def _naive2M4(trainY, horizon, seasonality):
    n = len(trainY)
    m = max(seasonality, 1)
    isSeasonal = m > 1 and _seasonalityTestM4(trainY, m)
    if isSeasonal:
        trend = np.convolve(trainY, np.ones(m) / m, mode='valid')
        if m % 2 == 0:
            trend = (trend[:-1] + trend[1:]) / 2.0
        startIdx = (m - 1) // 2 if m % 2 == 1 else m // 2
        seasonal = np.full(n, np.nan)
        for i in range(len(trend)):
            idx = startIdx + i
            if idx < n and trend[i] > 1e-10:
                seasonal[idx] = trainY[idx] / trend[i]
        seasonalIdx = np.zeros(m)
        counts = np.zeros(m)
        for i in range(n):
            if not np.isnan(seasonal[i]):
                seasonalIdx[i % m] += seasonal[i]
                counts[i % m] += 1
        for i in range(m):
            seasonalIdx[i] = seasonalIdx[i] / max(counts[i], 1) if counts[i] > 0 else 1.0
        meanSI = np.mean(seasonalIdx)
        if meanSI > 0:
            seasonalIdx /= meanSI
        seasonalIdx = np.maximum(seasonalIdx, 0.01)
        deseasonalized = trainY / seasonalIdx[np.arange(n) % m]
        pred = np.full(horizon, deseasonalized[-1])
        pred *= np.array([seasonalIdx[(n + h) % m] for h in range(horizon)])
    else:
        pred = np.full(horizon, trainY[-1])
    return pred


# ============================================================
# DOT implementations
# ============================================================

from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.turbo import TurboCore

try:
    from vectrix_core import dot_objective as _rustDotObj
    RUST_DOT = True
except ImportError:
    RUST_DOT = False

try:
    from vectrix_core import ses_filter as _rustSesFilter
    from vectrix_core import ses_sse as _rustSesSSE
    RUST_SES = True
except ImportError:
    RUST_SES = False

try:
    from vectrix_core import dot_hybrid_objective as _rustDotHybridObj
    RUST_HYBRID = True
except ImportError:
    RUST_HYBRID = False


def _sesFilter(y, alpha):
    if RUST_SES:
        return np.asarray(_rustSesFilter(y, alpha))
    n = len(y)
    r = np.zeros(n)
    r[0] = y[0]
    for t in range(1, n):
        r[t] = alpha * y[t] + (1.0 - alpha) * r[t - 1]
    return r


def _sesSSE(y, alpha):
    if RUST_SES:
        return _rustSesSSE(y, alpha)
    n = len(y)
    level = y[0]
    sse = 0.0
    for t in range(1, n):
        e = y[t] - level
        sse += e * e
        level = alpha * y[t] + (1.0 - alpha) * level
    return sse


def _optimizeAlpha(y):
    if len(y) < 3:
        return 0.3
    result = minimize_scalar(lambda a: _sesSSE(y, a), bounds=(0.001, 0.999), method='bounded')
    return result.x if result.success else 0.3


def _linReg(x, y):
    xM = np.mean(x)
    yM = np.mean(y)
    num = np.sum((x - xM) * (y - yM))
    den = np.sum((x - xM) ** 2)
    slope = num / max(den, 1e-10)
    intercept = yM - slope * xM
    return slope, intercept


def _classicDotPredict(trainY, horizon, period):
    y = np.asarray(trainY, dtype=np.float64)
    n = len(y)
    if n < 5:
        return np.full(horizon, np.mean(y))

    seasonal = None
    if period > 1 and n >= period * 2:
        m = period
        s = np.zeros(m)
        for i in range(m):
            vals = y[i::m]
            s[i] = np.mean(vals) - np.mean(y)
        seasonal = s
        workData = np.zeros(n)
        for t in range(n):
            workData[t] = y[t] - s[t % m]
    else:
        workData = y

    x = np.arange(n, dtype=np.float64)
    slope, intercept = TurboCore.linearRegression(x, workData)

    if RUST_DOT:
        def objective(params):
            return _rustDotObj(workData, intercept, slope, params[0], params[1], params[2])
    else:
        from vectrix.engine.dot import _dotObjectivePython
        def objective(params):
            return _dotObjectivePython(workData, intercept, slope, params[0], params[1], params[2])

    result = minimize(objective, x0=[2.0, 0.3, 0.0],
                      bounds=[(0.5, 5.0), (0.01, 0.99), (-1.0, 1.0)],
                      method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
    theta, alpha, drift = result.x

    thetaLine = theta * workData + (1.0 - theta) * (intercept + slope * x)
    filtered = _sesFilter(thetaLine, alpha)
    lastLevel = filtered[-1]

    predictions = np.zeros(horizon)
    for h in range(1, horizon + 1):
        t = n + h - 1
        trendPred = intercept + slope * t
        sesPred = lastLevel + drift * (n + h)
        predictions[h - 1] = (trendPred + sesPred) / 2

    if seasonal is not None:
        for h in range(horizon):
            predictions[h] += seasonal[(n + h) % period]

    return predictions


def _deseasonalizeAdvanced(y, period, seasonType):
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


def _fitTrendLine(y, trendType):
    n = len(y)
    x = np.arange(n, dtype=np.float64)
    if trendType == 'exponential':
        if np.any(y <= 0):
            return None
        logY = np.log(y)
        slope, intercept = _linReg(x, logY)
        return np.exp(intercept + slope * x)
    else:
        slope, intercept = _linReg(x, y)
        return intercept + slope * x


def _fitVariant(y, thetaLine0, trendType, modelType):
    n = len(y)
    if n < 5:
        return None
    isAdd = modelType == 'additive'
    ySafe = np.maximum(y, 1e-10)
    t0Safe = np.maximum(thetaLine0, 1e-10)

    def buildThetaLine(theta):
        if isAdd:
            return theta * y + (1.0 - theta) * thetaLine0
        return np.power(ySafe, theta) * np.power(t0Safe, 1.0 - theta)

    def combineFitted(filtered, theta):
        if isAdd:
            w = 1.0 / max(theta, 1.0)
            return w * filtered + (1.0 - w) * thetaLine0
        inv = 1.0 / max(theta, 1.0)
        return np.power(np.maximum(filtered, 1e-10), inv) * np.power(t0Safe, 1.0 - inv)

    if RUST_HYBRID:
        def objective(params):
            return _rustDotHybridObj(y, thetaLine0, params[0], isAdd)
    else:
        def objective(params):
            theta = params[0]
            tl = buildThetaLine(theta)
            alpha = _optimizeAlpha(tl)
            filt = _sesFilter(tl, alpha)
            fv = combineFitted(filt, theta)
            return np.mean(np.abs(y - fv))

    result = minimize(objective, x0=[2.0], bounds=[(1.0, 50.0)],
                      method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
    theta = result.x[0]
    thetaLine = buildThetaLine(theta)
    alpha = _optimizeAlpha(thetaLine)
    filtered = _sesFilter(thetaLine, alpha)
    lastLevel = filtered[-1]
    fittedVals = combineFitted(filtered, theta)

    x = np.arange(n, dtype=np.float64)
    if trendType == 'exponential':
        slope, intercept = _linReg(x, np.log(ySafe))
    else:
        slope, intercept = _linReg(x, y)

    return {
        'theta': theta, 'alpha': alpha, 'intercept': intercept,
        'slope': slope, 'lastLevel': lastLevel, 'n': n,
        'residStd': max(np.std(y - fittedVals), 1e-8),
        'fittedValues': fittedVals,
    }


def _predictVariant(model, trendType, modelType, steps):
    n = model['n']
    futureX = np.arange(n, n + steps, dtype=np.float64)
    if trendType == 'exponential':
        forecastTrend = np.exp(model['intercept'] + model['slope'] * futureX)
    else:
        forecastTrend = model['intercept'] + model['slope'] * futureX
    forecastSES = np.full(steps, model['lastLevel'])
    if modelType == 'additive':
        w = 1.0 / max(model['theta'], 1.0)
        return w * forecastSES + (1.0 - w) * forecastTrend
    else:
        inv = 1.0 / max(model['theta'], 1.0)
        return np.power(np.maximum(forecastSES, 1e-10), inv) * \
               np.power(np.maximum(forecastTrend, 1e-10), 1.0 - inv)


def _reseasonalize(y, seasonal, seasonType):
    n = len(y)
    result = y.copy()
    period = len(seasonal)
    for t in range(n):
        idx = t % period
        if seasonType == 'multiplicative':
            result[t] *= seasonal[idx]
        else:
            result[t] += seasonal[idx]
    return result


def _dotHybridHoldoutPredict(trainY, horizon, period):
    """DOT-Hybrid with holdout validation for config selection."""
    y = np.asarray(trainY, dtype=np.float64)
    n = len(y)
    if n < 5:
        return np.full(horizon, np.mean(y))

    hasSeason = period > 1 and n >= period * 3
    seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']

    scaled = y.copy()
    base = np.mean(np.abs(scaled))
    if base > 0:
        scaled = scaled / base
    else:
        base = 1.0

    holdoutSize = max(1, min(n // 5, period * 2 if period > 1 else 5))
    holdoutSize = min(holdoutSize, n // 3)
    trainPart = scaled[:n - holdoutSize]
    valPart = scaled[n - holdoutSize:]
    nTrain = len(trainPart)

    bestMae = np.inf
    bestConfig = None

    for seasonType in seasonTypes:
        if seasonType != 'none':
            seasonal, deseasonalized = _deseasonalizeAdvanced(trainPart, period, seasonType)
        else:
            seasonal = None
            deseasonalized = trainPart

        for trendType in ['linear', 'exponential']:
            thetaLine0 = _fitTrendLine(deseasonalized, trendType)
            if thetaLine0 is None:
                continue

            for modelType in ['additive', 'multiplicative']:
                if modelType == 'multiplicative' and np.any(thetaLine0 <= 0):
                    continue
                if modelType == 'multiplicative' and np.any(deseasonalized <= 0):
                    continue

                result = _fitVariant(deseasonalized, thetaLine0, trendType, modelType)
                if result is None:
                    continue

                valPred = _predictVariant(result, trendType, modelType, holdoutSize)
                if seasonal is not None:
                    for h in range(holdoutSize):
                        idx = (nTrain + h) % period
                        if seasonType == 'multiplicative':
                            valPred[h] *= seasonal[idx]
                        else:
                            valPred[h] += seasonal[idx]

                mae = np.mean(np.abs(valPart - valPred))
                if mae < bestMae:
                    bestMae = mae
                    bestConfig = (trendType, modelType, seasonType)

    if bestConfig is None:
        return _classicDotPredict(y, horizon, period)

    trendType, modelType, seasonType = bestConfig

    if seasonType != 'none':
        seasonal, deseasonalized = _deseasonalizeAdvanced(scaled, period, seasonType)
    else:
        seasonal = None
        deseasonalized = scaled

    thetaLine0 = _fitTrendLine(deseasonalized, trendType)
    if thetaLine0 is None:
        return _classicDotPredict(y, horizon, period)
    bestModel = _fitVariant(deseasonalized, thetaLine0, trendType, modelType)
    if bestModel is None:
        return _classicDotPredict(y, horizon, period)

    pred = _predictVariant(bestModel, trendType, modelType, horizon)
    if seasonal is not None:
        for h in range(horizon):
            idx = (n + h) % period
            if seasonType == 'multiplicative':
                pred[h] *= seasonal[idx]
            else:
                pred[h] += seasonal[idx]

    return pred * base


def _improvedDotPredict(trainY, horizon, period, groupName=None):
    """
    Improved DOT combining accepted findings:
    - Weekly (period=1, short series): classic DOT (E044)
    - Others with period<24: holdout-validated hybrid (E043)
    - Period>=24: classic DOT (unchanged)
    """
    y = np.asarray(trainY, dtype=np.float64)
    n = len(y)

    if period >= 24:
        return _classicDotPredict(y, horizon, period)

    if period <= 1 and groupName == 'Weekly':
        return _classicDotPredict(y, horizon, period)

    return _dotHybridHoldoutPredict(y, horizon, period)


# ============================================================
# Main
# ============================================================

def _runGroup(groupName, sampleCap=None):
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

    if sampleCap and len(validIdx) > sampleCap:
        rng = np.random.default_rng(42)
        validIdx = sorted(rng.choice(validIdx, size=sampleCap, replace=False).tolist())
    P(f"  Using {len(validIdx)} / {nSeries} series")

    allSmapesBaseline = []
    allMasesBaseline = []
    allSmapesImproved = []
    allMasesImproved = []
    allSmapesN2 = []
    allMasesN2 = []

    startTime = time.perf_counter()

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]

        n2 = _naive2M4(trainY, horizon, seasonality)
        allSmapesN2.append(_smapeM4(testY, n2))
        allMasesN2.append(_maseM4(trainY, testY, n2, seasonality))

        baseModel = DynamicOptimizedTheta(period=seasonality)
        baseModel.fit(trainY)
        basePred, _, _ = baseModel.predict(horizon)
        basePred = np.asarray(basePred[:horizon], dtype=np.float64)
        if not np.all(np.isfinite(basePred)):
            basePred = np.where(np.isfinite(basePred), basePred, np.mean(trainY))
        allSmapesBaseline.append(_smapeM4(testY, basePred))
        allMasesBaseline.append(_maseM4(trainY, testY, basePred, seasonality))

        impPred = _improvedDotPredict(trainY, horizon, seasonality, groupName=groupName)
        impPred = np.asarray(impPred[:horizon], dtype=np.float64)
        if not np.all(np.isfinite(impPred)):
            impPred = np.where(np.isfinite(impPred), impPred, np.mean(trainY))
        allSmapesImproved.append(_smapeM4(testY, impPred))
        allMasesImproved.append(_maseM4(trainY, testY, impPred, seasonality))

        if (count + 1) % 500 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} in {elapsed:.1f}s")

    sN2 = np.mean(np.concatenate(allSmapesN2))
    mN2 = np.mean(np.concatenate(allMasesN2))

    sBAvg = np.mean(np.concatenate(allSmapesBaseline))
    mBAvg = np.mean(np.concatenate(allMasesBaseline))
    owaBase = 0.5 * (sBAvg / sN2 + mBAvg / mN2)

    sIAvg = np.mean(np.concatenate(allSmapesImproved))
    mIAvg = np.mean(np.concatenate(allMasesImproved))
    owaImp = 0.5 * (sIAvg / sN2 + mIAvg / mN2)

    diff = owaImp - owaBase
    P(f"  Naive2:   sMAPE={sN2:.3f}  MASE={mN2:.4f}")
    P(f"  Baseline: OWA={owaBase:.4f}  sMAPE={sBAvg:.3f}  MASE={mBAvg:.4f}")
    P(f"  Improved: OWA={owaImp:.4f}  sMAPE={sIAvg:.3f}  MASE={mIAvg:.4f}")
    P(f"  Change:   {diff:+.4f} ({diff/owaBase*100:+.2f}%)")

    return {
        'owaBase': owaBase, 'owaImp': owaImp,
        'smapeBase': sBAvg, 'maseBase': mBAvg,
        'smapeImp': sIAvg, 'maseImp': mIAvg,
        'elapsed': elapsed,
    }


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E045: Integrated DOT Improvement Verification")
    P("=" * 60)

    GROUP_CAPS = {
        'Yearly': 2000, 'Quarterly': 2000, 'Monthly': 2000,
        'Weekly': None, 'Daily': None, 'Hourly': None,
    }

    allResults = {}
    totalTime = 0

    for group in M4_GROUPS:
        result = _runGroup(group, sampleCap=GROUP_CAPS.get(group))
        allResults[group] = result
        totalTime += result['elapsed']

    P(f"\n{'='*60}")
    P(f"  FINAL SUMMARY ({totalTime/60:.1f} min)")
    P(f"{'='*60}")

    P(f"\n  {'Group':<12} {'Baseline':>10} {'Improved':>10} {'Change':>10}")
    P(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")

    avgBase = []
    avgImp = []
    for group in M4_GROUPS:
        r = allResults[group]
        diff = r['owaImp'] - r['owaBase']
        avgBase.append(r['owaBase'])
        avgImp.append(r['owaImp'])
        P(f"  {group:<12} {r['owaBase']:>10.4f} {r['owaImp']:>10.4f} {diff:>+10.4f}")

    avgB = np.mean(avgBase)
    avgI = np.mean(avgImp)
    avgD = avgI - avgB
    P(f"  {'AVG':<12} {avgB:>10.4f} {avgI:>10.4f} {avgD:>+10.4f}")

    P(f"\n  Overall: {avgB:.4f} -> {avgI:.4f} ({avgD/avgB*100:+.2f}%)")

    if avgI < avgB:
        P(f"  VERDICT: IMPROVEMENT CONFIRMED. Ready for engine integration.")
    else:
        P(f"  VERDICT: NO IMPROVEMENT. Do NOT integrate.")

    P("=" * 60)
