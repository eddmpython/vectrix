"""
==============================================================================
Experiment ID: modelCreation/043
Experiment: DOT Auto Period Detection + Holdout Validation
==============================================================================

Purpose:
- Daily OWA 1.007 is worst group — Naive2 level. Weekly 0.959 also weak.
- M4 official seasonality: Daily=1, Weekly=1 (no seasonality declared)
- Real data has period=7 (daily) and period=52 (weekly) patterns
- Two independent improvements to test:
  A) Auto period detection via ACF when period=1
  B) Holdout validation for 8-way config selection (instead of in-sample MAE)

Hypotheses:
1. ACF-based auto period detection recovers real seasonality → Daily OWA < 0.95
2. Holdout validation reduces overfitting in config selection → lower OWA across groups
3. Combined A+B gives best result

Method:
1. Run M4 Daily (4227 series) and Weekly (359 series) with 4 DOT variants:
   - baseline: original DOT (period=M4 official)
   - auto_period: ACF auto-detect period when declared=1
   - holdout_val: holdout-based 8-way selection (not in-sample MAE)
   - combined: auto_period + holdout_val
2. Measure OWA vs M4 official Naive2
3. Also run on Monthly/Yearly/Quarterly/Hourly to check no regression

Results:
                baseline   auto_period  holdout_val  combined
  Yearly         0.7971      0.8019      0.8064      0.8084
  Quarterly      0.9053      0.9053      0.8940      0.8940
  Monthly        0.9200      0.9200      0.8965      0.8965
  Weekly         0.9587      0.9952      0.9457      0.9831
  Daily          0.9949      1.0220      0.9918      1.0187
  Hourly         0.7223      0.7223      0.7223      0.7223
  AVG            0.8831      0.8944      0.8761      0.8872

Conclusion:
- auto_period: REJECTED. ACF detects spurious short periods (2,3) from noise.
  Daily +2.7%, Weekly +3.8% worse. Auto period detection is harmful.
- holdout_val: CONDITIONALLY ACCEPTED. AVG -0.79% improvement.
  Quarterly -1.25%, Monthly -2.55% significant gains.
  Daily only -0.3%, Weekly -1.4% — modest.
  Yearly +1.2% slight regression (less data for fitting).
- combined: REJECTED. auto_period dominates and cancels holdout gains.
- Daily OWA 0.9918 still near Naive2 level — need different approach for Daily.
- Hourly unaffected (uses classic DOT with period=24).

Next: holdout_val alone is worth integrating. Daily needs separate experiment
  (e.g. multi-model ensemble, frequency-aware period forcing).

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
    isSeasonal = False
    if m > 1:
        isSeasonal = _seasonalityTestM4(trainY, m)
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
            if counts[i] > 0:
                seasonalIdx[i] = seasonalIdx[i] / counts[i]
            else:
                seasonalIdx[i] = 1.0
        meanSI = np.mean(seasonalIdx)
        if meanSI > 0:
            seasonalIdx = seasonalIdx / meanSI
        seasonalIdx = np.maximum(seasonalIdx, 0.01)
        deseasonalized = trainY / seasonalIdx[np.arange(n) % m]
        pred = np.full(horizon, deseasonalized[-1])
        siOut = np.array([seasonalIdx[(n + h) % m] for h in range(horizon)])
        pred = pred * siOut
    else:
        pred = np.full(horizon, trainY[-1])
    return pred


# ============================================================
# DOT Variants (self-contained, no engine import dependency)
# ============================================================

from vectrix.engine.turbo import TurboCore

try:
    from vectrix_core import dot_objective as _rustDotObjective
    from vectrix_core import dot_residuals as _rustDotResiduals
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
    result = np.zeros(n)
    result[0] = y[0]
    for t in range(1, n):
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1]
    return result


def _sesSSE(y, alpha):
    if RUST_SES:
        return _rustSesSSE(y, alpha)
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


def _linReg(x, y):
    xM = np.mean(x)
    yM = np.mean(y)
    num = np.sum((x - xM) * (y - yM))
    den = np.sum((x - xM) ** 2)
    slope = num / max(den, 1e-10)
    intercept = yM - slope * xM
    return slope, intercept


def _detectPeriodACF(y):
    """ACF-based period detection. Returns detected period or 1."""
    n = len(y)
    if n < 14:
        return 1

    detrended = y - np.linspace(y[0], y[-1], n)
    maxLag = min(n // 2, 200)
    yMean = np.mean(detrended)
    yC = detrended - yMean
    c0 = np.sum(yC ** 2) / n
    if c0 < 1e-10:
        return 1

    acf = np.zeros(maxLag + 1)
    acf[0] = 1.0
    for lag in range(1, maxLag + 1):
        acf[lag] = np.sum(yC[:n - lag] * yC[lag:]) / (n * c0)

    candidates = []
    for lag in range(2, maxLag):
        if acf[lag - 1] < acf[lag] > acf[lag + 1] and acf[lag] > 0.05:
            candidates.append((lag, acf[lag]))

    if not candidates:
        return 1

    candidates.sort(key=lambda x: x[1], reverse=True)
    bestPeriod = candidates[0][0]
    if bestPeriod <= n // 4:
        return bestPeriod
    return 1


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


_HYBRID_THRESHOLD = 24


def _dotPredict(trainY, horizon, period, useAutoPeriod=False, useHoldout=False):
    """
    DOT-Hybrid prediction with optional improvements.

    useAutoPeriod: detect period via ACF when declared period<=1
    useHoldout: use holdout validation for config selection (not in-sample MAE)
    """
    y = np.asarray(trainY, dtype=np.float64)
    n = len(y)

    if n < 5:
        return np.full(horizon, np.mean(y))

    effectivePeriod = period
    if useAutoPeriod and period <= 1 and n >= 14:
        detected = _detectPeriodACF(y)
        if detected > 1:
            effectivePeriod = detected

    if effectivePeriod >= _HYBRID_THRESHOLD:
        return _dotClassicPredict(y, horizon, effectivePeriod)

    return _dotHybridPredict(y, horizon, effectivePeriod, useHoldout)


def _dotClassicPredict(y, horizon, period):
    """Classic DOT (period >= 24)."""
    n = len(y)
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
            return _rustDotObjective(workData, intercept, slope, params[0], params[1], params[2])
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


def _dotHybridPredict(y, horizon, period, useHoldout=False):
    """DOT-Hybrid (period < 24) with optional holdout validation."""
    n = len(y)
    hasSeason = period > 1 and n >= period * 3
    seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']

    scaled = y.copy()
    base = np.mean(np.abs(scaled))
    if base > 0:
        scaled = scaled / base
    else:
        base = 1.0

    if useHoldout:
        holdoutSize = max(1, min(n // 5, period * 2 if period > 1 else 5))
        holdoutSize = min(holdoutSize, n // 3)
        trainPart = scaled[:n - holdoutSize]
        valPart = scaled[n - holdoutSize:]
        nTrain = len(trainPart)
    else:
        trainPart = scaled
        valPart = None
        nTrain = n
        holdoutSize = 0

    fitData = trainPart if useHoldout else scaled

    bestMae = np.inf
    bestConfig = None

    for seasonType in seasonTypes:
        if seasonType != 'none':
            seasonal, deseasonalized = _deseasonalizeAdvanced(fitData, period, seasonType)
        else:
            seasonal = None
            deseasonalized = fitData

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

                if useHoldout:
                    valPred = _predictVariant(result, trendType, modelType, holdoutSize)
                    if seasonal is not None:
                        for h in range(holdoutSize):
                            idx = (nTrain + h) % period
                            if seasonType == 'multiplicative':
                                valPred[h] *= seasonal[idx]
                            else:
                                valPred[h] += seasonal[idx]
                    mae = np.mean(np.abs(valPart - valPred))
                else:
                    fv = result['fittedValues']
                    if seasonal is not None:
                        fv = _reseasonalize(fv, seasonal, seasonType)
                    mae = np.mean(np.abs(fitData - fv))

                if mae < bestMae:
                    bestMae = mae
                    bestConfig = (trendType, modelType, seasonType)

    if bestConfig is None:
        return _dotClassicPredict(y, horizon, period)

    trendType, modelType, seasonType = bestConfig

    if seasonType != 'none':
        seasonal, deseasonalized = _deseasonalizeAdvanced(scaled, period, seasonType)
    else:
        seasonal = None
        deseasonalized = scaled

    thetaLine0 = _fitTrendLine(deseasonalized, trendType)
    if thetaLine0 is None:
        return _dotClassicPredict(y, horizon, period)
    bestModel = _fitVariant(deseasonalized, thetaLine0, trendType, modelType)
    if bestModel is None:
        return _dotClassicPredict(y, horizon, period)

    pred = _predictVariant(bestModel, trendType, modelType, horizon)
    if seasonal is not None:
        for h in range(horizon):
            idx = (n + h) % period
            if seasonType == 'multiplicative':
                pred[h] *= seasonal[idx]
            else:
                pred[h] += seasonal[idx]

    return pred * base


# ============================================================
# Main experiment
# ============================================================

def _runGroup(groupName, variants, sampleCap=None):
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

    allSmapes = {v: [] for v in variants}
    allMases = {v: [] for v in variants}
    allSmapesN2 = []
    allMasesN2 = []

    startTime = time.perf_counter()

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]

        n2 = _naive2M4(trainY, horizon, seasonality)
        allSmapesN2.append(_smapeM4(testY, n2))
        allMasesN2.append(_maseM4(trainY, testY, n2, seasonality))

        for vName, vConfig in variants.items():
            pred = _dotPredict(trainY, horizon, seasonality,
                               useAutoPeriod=vConfig['autoPeriod'],
                               useHoldout=vConfig['holdout'])
            pred = np.asarray(pred[:horizon], dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
            allSmapes[vName].append(_smapeM4(testY, pred))
            allMases[vName].append(_maseM4(trainY, testY, pred, seasonality))

        if (count + 1) % 500 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} in {elapsed:.1f}s")

    smapeN2Avg = np.mean(np.concatenate(allSmapesN2))
    maseN2Avg = np.mean(np.concatenate(allMasesN2))

    P(f"\n  Naive2: sMAPE={smapeN2Avg:.3f}, MASE={maseN2Avg:.4f}")

    results = {}
    for vName in variants:
        smapeFlat = np.concatenate(allSmapes[vName])
        maseFlat = np.concatenate(allMases[vName])
        smapeAvg = np.mean(smapeFlat)
        maseAvg = np.mean(maseFlat)
        owa = 0.5 * (smapeAvg / smapeN2Avg + maseAvg / maseN2Avg)
        results[vName] = {
            'owa': owa, 'smape': smapeAvg, 'mase': maseAvg,
            'smapeFlat': smapeFlat, 'maseFlat': maseFlat,
        }
        P(f"  {vName:20s}: OWA={owa:.4f}  sMAPE={smapeAvg:.3f}  MASE={maseAvg:.4f}")

    return results, smapeN2Avg, maseN2Avg, elapsed


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E043: DOT Auto Period Detection + Holdout Validation")
    P("=" * 60)

    VARIANTS = {
        'baseline': {'autoPeriod': False, 'holdout': False},
        'auto_period': {'autoPeriod': True, 'holdout': False},
        'holdout_val': {'autoPeriod': False, 'holdout': True},
        'combined': {'autoPeriod': True, 'holdout': True},
    }

    GROUP_CAPS = {
        'Yearly': 2000, 'Quarterly': 2000, 'Monthly': 2000,
        'Weekly': None, 'Daily': None, 'Hourly': None,
    }

    allGroupResults = {}
    totalTime = 0

    for group in M4_GROUPS:
        groupResults, n2Smape, n2Mase, elapsed = _runGroup(
            group, VARIANTS, sampleCap=GROUP_CAPS.get(group)
        )
        allGroupResults[group] = groupResults
        totalTime += elapsed

    P(f"\n{'='*60}")
    P(f"  SUMMARY ({totalTime/60:.1f} min)")
    P(f"{'='*60}")

    P(f"\n  {'Group':<12} {'baseline':>10} {'auto_period':>12} {'holdout_val':>12} {'combined':>10}")
    P(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    avgOwas = {v: [] for v in VARIANTS}
    for group in M4_GROUPS:
        row = f"  {group:<12}"
        for vName in VARIANTS:
            owa = allGroupResults[group][vName]['owa']
            avgOwas[vName].append(owa)
            row += f" {owa:>10.4f}  "
        P(row)

    P(f"\n  {'AVG':<12}", end="")
    for vName in VARIANTS:
        avg = np.mean(avgOwas[vName])
        P(f" {avg:>10.4f}  ", end="")
    P()

    P(f"\n  Improvement over baseline (negative = better):")
    baselineAvg = np.mean(avgOwas['baseline'])
    for vName in VARIANTS:
        if vName == 'baseline':
            continue
        avg = np.mean(avgOwas[vName])
        diff = avg - baselineAvg
        P(f"    {vName:20s}: {diff:+.4f} ({diff/baselineAvg*100:+.2f}%)")

    P(f"\n  Per-group improvement (combined vs baseline):")
    for group in M4_GROUPS:
        bOwa = allGroupResults[group]['baseline']['owa']
        cOwa = allGroupResults[group]['combined']['owa']
        diff = cOwa - bOwa
        P(f"    {group:<12}: {bOwa:.4f} -> {cOwa:.4f}  ({diff:+.4f}, {diff/bOwa*100:+.2f}%)")

    P("=" * 60)
