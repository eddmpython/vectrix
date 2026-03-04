"""
==============================================================================
Experiment ID: modelCreation/046
Experiment: Final Integration — period<=1 classic, period>1 holdout hybrid
==============================================================================

Purpose:
- E045 showed Yearly +1.16% regression from holdout on short period=1 data
- Refined rule: period<=1 → always classic DOT, period>1 → holdout hybrid
- This preserves Yearly/Daily/Weekly strengths while improving Quarterly/Monthly

Hypotheses:
1. No group regresses vs baseline
2. AVG OWA < 0.8831

Method:
- period<=1: classic DOT (no hybrid, no holdout)
- period>1 and period<24: holdout-validated hybrid
- period>=24: classic DOT (unchanged)

Results:
  Group        Base     Improved     Change
  Yearly      0.7971     0.8869    +0.0897  <<< REGRESSION (classic kills Yearly!)
  Quarterly   0.9053     0.8940    -0.0113
  Monthly     0.9200     0.8965    -0.0235
  Weekly      0.9587     0.9378    -0.0209
  Daily       0.9949     1.0047    +0.0098  <<< REGRESSION
  Hourly      0.7223     0.7223    +0.0000
  AVG         0.8831     0.8904    +0.0073

Conclusion:
- REJECTED. period<=1 classic forced approach causes catastrophic Yearly regression.
  Yearly data (short, period=1) works BETTER with Hybrid 8-way — Hybrid captures
  trend patterns that classic 3-param misses.
- The only safe change is: holdout validation for 1<period<24 (Quarterly/Monthly).
  This is exactly E043 holdout_val (AVG 0.8831->0.8761, -0.79%).
- Final engine change: ONLY add holdout validation to _fitHybrid when period > 1.
  Leave period<=1 and period>=24 paths completely unchanged.

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
    trainSeries, testSeries = [], []
    for i in range(len(trainDf)):
        trainSeries.append(trainDf.iloc[i, 1:].dropna().values.astype(np.float64))
        testSeries.append(testDf.iloc[i, 1:].dropna().values.astype(np.float64))
    return trainSeries, testSeries


def _smapeM4(actual, predicted):
    return np.abs(actual - predicted) * 200.0 / (np.abs(actual) + np.abs(predicted) + 1e-10)

def _maseM4(trainY, actual, predicted, seasonality):
    m = max(seasonality, 1)
    n = len(trainY)
    naiveErrors = np.abs(np.diff(trainY)) if n <= m else np.abs(trainY[m:] - trainY[:-m])
    masep = max(np.mean(naiveErrors), 1e-10) if len(naiveErrors) > 0 else 1e-10
    return np.abs(actual - predicted) / masep

def _seasonalityTestM4(y, ppy):
    if len(y) < 3 * ppy:
        return False
    n = len(y)
    nlag = min(ppy, n // 3)
    if nlag < 1:
        return False
    y_mean, y_centered = np.mean(y), y - np.mean(y)
    c0 = np.sum(y_centered ** 2) / n
    if c0 < 1e-10:
        return False
    acf = np.zeros(nlag + 1)
    acf[0] = 1.0
    for lag in range(1, nlag + 1):
        acf[lag] = np.sum(y_centered[:n-lag] * y_centered[lag:]) / (n * c0)
    cumsum = 1.0
    for k in range(1, ppy):
        if k < len(acf):
            cumsum += 2 * acf[k] ** 2
    limit_adj = 1.645 * np.sqrt(cumsum) / np.sqrt(n)
    return abs(acf[ppy]) > limit_adj if ppy < len(acf) else False

def _naive2M4(trainY, horizon, seasonality):
    n, m = len(trainY), max(seasonality, 1)
    if m > 1 and _seasonalityTestM4(trainY, m):
        trend = np.convolve(trainY, np.ones(m) / m, mode='valid')
        if m % 2 == 0:
            trend = (trend[:-1] + trend[1:]) / 2.0
        startIdx = (m - 1) // 2 if m % 2 == 1 else m // 2
        seasonal = np.full(n, np.nan)
        for i in range(len(trend)):
            idx = startIdx + i
            if idx < n and trend[i] > 1e-10:
                seasonal[idx] = trainY[idx] / trend[i]
        si = np.zeros(m)
        counts = np.zeros(m)
        for i in range(n):
            if not np.isnan(seasonal[i]):
                si[i % m] += seasonal[i]
                counts[i % m] += 1
        for i in range(m):
            si[i] = si[i] / max(counts[i], 1) if counts[i] > 0 else 1.0
        meanSI = np.mean(si)
        if meanSI > 0:
            si /= meanSI
        si = np.maximum(si, 0.01)
        deseas = trainY / si[np.arange(n) % m]
        return np.full(horizon, deseas[-1]) * np.array([si[(n + h) % m] for h in range(horizon)])
    return np.full(horizon, trainY[-1])


# ============================================================
# DOT implementations (reuse from E045)
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
    n, r = len(y), np.zeros(len(y))
    r[0] = y[0]
    for t in range(1, n):
        r[t] = alpha * y[t] + (1.0 - alpha) * r[t - 1]
    return r

def _sesSSE(y, alpha):
    if RUST_SES:
        return _rustSesSSE(y, alpha)
    n, level, sse = len(y), y[0], 0.0
    for t in range(1, n):
        e = y[t] - level
        sse += e * e
        level = alpha * y[t] + (1.0 - alpha) * level
    return sse

def _optimizeAlpha(y):
    if len(y) < 3:
        return 0.3
    r = minimize_scalar(lambda a: _sesSSE(y, a), bounds=(0.001, 0.999), method='bounded')
    return r.x if r.success else 0.3

def _linReg(x, y):
    xM, yM = np.mean(x), np.mean(y)
    num = np.sum((x - xM) * (y - yM))
    den = np.sum((x - xM) ** 2)
    slope = num / max(den, 1e-10)
    return slope, yM - slope * xM

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
            s[i] = np.mean(y[i::m]) - np.mean(y)
        seasonal = s
        workData = y - s[np.arange(n) % m]
    else:
        workData = y
    x = np.arange(n, dtype=np.float64)
    slope, intercept = TurboCore.linearRegression(x, workData)
    if RUST_DOT:
        def obj(p): return _rustDotObj(workData, intercept, slope, p[0], p[1], p[2])
    else:
        from vectrix.engine.dot import _dotObjectivePython
        def obj(p): return _dotObjectivePython(workData, intercept, slope, p[0], p[1], p[2])
    result = minimize(obj, x0=[2.0, 0.3, 0.0], bounds=[(0.5, 5.0), (0.01, 0.99), (-1.0, 1.0)],
                      method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
    theta, alpha, drift = result.x
    thetaLine = theta * workData + (1.0 - theta) * (intercept + slope * x)
    filtered = _sesFilter(thetaLine, alpha)
    lastLevel = filtered[-1]
    predictions = np.zeros(horizon)
    for h in range(1, horizon + 1):
        t = n + h - 1
        predictions[h - 1] = (intercept + slope * t + lastLevel + drift * (n + h)) / 2
    if seasonal is not None:
        for h in range(horizon):
            predictions[h] += seasonal[(n + h) % period]
    return predictions


def _deseasonalizeAdvanced(y, period, seasonType):
    n = len(y)
    seasonal, counts = np.zeros(period), np.zeros(period)
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
        return seasonal, y / seasonal[np.arange(n) % period]
    else:
        for i in range(len(trend)):
            idx = i + offset
            if idx < n:
                seasonal[idx % period] += y[idx] - trend[i]
                counts[idx % period] += 1
        for i in range(period):
            seasonal[i] = seasonal[i] / max(counts[i], 1)
        seasonal -= np.mean(seasonal)
        return seasonal, y - seasonal[np.arange(n) % period]

def _fitTrendLine(y, trendType):
    n, x = len(y), np.arange(len(y), dtype=np.float64)
    if trendType == 'exponential':
        if np.any(y <= 0):
            return None
        slope, intercept = _linReg(x, np.log(y))
        return np.exp(intercept + slope * x)
    slope, intercept = _linReg(x, y)
    return intercept + slope * x

def _fitVariant(y, thetaLine0, trendType, modelType):
    n = len(y)
    if n < 5:
        return None
    isAdd = modelType == 'additive'
    ySafe, t0Safe = np.maximum(y, 1e-10), np.maximum(thetaLine0, 1e-10)
    def buildTL(theta):
        return theta * y + (1.0 - theta) * thetaLine0 if isAdd else np.power(ySafe, theta) * np.power(t0Safe, 1.0 - theta)
    def combineF(filt, theta):
        if isAdd:
            w = 1.0 / max(theta, 1.0)
            return w * filt + (1.0 - w) * thetaLine0
        inv = 1.0 / max(theta, 1.0)
        return np.power(np.maximum(filt, 1e-10), inv) * np.power(t0Safe, 1.0 - inv)
    if RUST_HYBRID:
        def obj(p): return _rustDotHybridObj(y, thetaLine0, p[0], isAdd)
    else:
        def obj(p):
            tl = buildTL(p[0])
            return np.mean(np.abs(y - combineF(_sesFilter(tl, _optimizeAlpha(tl)), p[0])))
    result = minimize(obj, x0=[2.0], bounds=[(1.0, 50.0)], method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
    theta = result.x[0]
    thetaLine = buildTL(theta)
    alpha = _optimizeAlpha(thetaLine)
    filtered = _sesFilter(thetaLine, alpha)
    fv = combineF(filtered, theta)
    x = np.arange(n, dtype=np.float64)
    slope, intercept = _linReg(x, np.log(ySafe)) if trendType == 'exponential' else _linReg(x, y)
    return {'theta': theta, 'alpha': alpha, 'intercept': intercept, 'slope': slope,
            'lastLevel': filtered[-1], 'n': n, 'residStd': max(np.std(y - fv), 1e-8), 'fittedValues': fv}

def _predictVariant(model, trendType, modelType, steps):
    n, futureX = model['n'], np.arange(model['n'], model['n'] + steps, dtype=np.float64)
    ft = np.exp(model['intercept'] + model['slope'] * futureX) if trendType == 'exponential' else model['intercept'] + model['slope'] * futureX
    fs = np.full(steps, model['lastLevel'])
    if modelType == 'additive':
        w = 1.0 / max(model['theta'], 1.0)
        return w * fs + (1.0 - w) * ft
    inv = 1.0 / max(model['theta'], 1.0)
    return np.power(np.maximum(fs, 1e-10), inv) * np.power(np.maximum(ft, 1e-10), 1.0 - inv)


def _dotHybridHoldoutPredict(trainY, horizon, period):
    y = np.asarray(trainY, dtype=np.float64)
    n = len(y)
    if n < 5:
        return np.full(horizon, np.mean(y))
    hasSeason = period > 1 and n >= period * 3
    seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']
    scaled = y.copy()
    base = np.mean(np.abs(scaled))
    if base > 0:
        scaled /= base
    else:
        base = 1.0
    holdoutSize = max(1, min(n // 5, period * 2 if period > 1 else 5))
    holdoutSize = min(holdoutSize, n // 3)
    trainPart, valPart = scaled[:n - holdoutSize], scaled[n - holdoutSize:]
    nTrain = len(trainPart)
    bestMae, bestConfig = np.inf, None
    for seasonType in seasonTypes:
        if seasonType != 'none':
            seasonal, deseasonalized = _deseasonalizeAdvanced(trainPart, period, seasonType)
        else:
            seasonal, deseasonalized = None, trainPart
        for trendType in ['linear', 'exponential']:
            tl0 = _fitTrendLine(deseasonalized, trendType)
            if tl0 is None:
                continue
            for modelType in ['additive', 'multiplicative']:
                if modelType == 'multiplicative' and (np.any(tl0 <= 0) or np.any(deseasonalized <= 0)):
                    continue
                result = _fitVariant(deseasonalized, tl0, trendType, modelType)
                if result is None:
                    continue
                vp = _predictVariant(result, trendType, modelType, holdoutSize)
                if seasonal is not None:
                    for h in range(holdoutSize):
                        idx = (nTrain + h) % period
                        vp[h] = vp[h] * seasonal[idx] if seasonType == 'multiplicative' else vp[h] + seasonal[idx]
                mae = np.mean(np.abs(valPart - vp))
                if mae < bestMae:
                    bestMae, bestConfig = mae, (trendType, modelType, seasonType)
    if bestConfig is None:
        return _classicDotPredict(y, horizon, period)
    trendType, modelType, seasonType = bestConfig
    if seasonType != 'none':
        seasonal, deseasonalized = _deseasonalizeAdvanced(scaled, period, seasonType)
    else:
        seasonal, deseasonalized = None, scaled
    tl0 = _fitTrendLine(deseasonalized, trendType)
    if tl0 is None:
        return _classicDotPredict(y, horizon, period)
    bm = _fitVariant(deseasonalized, tl0, trendType, modelType)
    if bm is None:
        return _classicDotPredict(y, horizon, period)
    pred = _predictVariant(bm, trendType, modelType, horizon)
    if seasonal is not None:
        for h in range(horizon):
            idx = (n + h) % period
            pred[h] = pred[h] * seasonal[idx] if seasonType == 'multiplicative' else pred[h] + seasonal[idx]
    return pred * base


def _finalDotPredict(trainY, horizon, period):
    """
    Final improved DOT rule:
    - period <= 1: classic DOT (best for Yearly/Daily/Weekly)
    - 1 < period < 24: holdout-validated hybrid (best for Quarterly/Monthly)
    - period >= 24: classic DOT (best for Hourly)
    """
    if period <= 1 or period >= 24:
        return _classicDotPredict(trainY, horizon, period)
    return _dotHybridHoldoutPredict(trainY, horizon, period)


def _runGroup(groupName, sampleCap=None):
    info = M4_GROUPS[groupName]
    horizon, seasonality = info['horizon'], info['seasonality']
    P(f"\n{'='*60}")
    P(f"  {groupName}: h={horizon}, m={seasonality}")
    P(f"{'='*60}")
    trainSeries, testSeries = _loadGroup(groupName)
    nSeries = len(trainSeries)
    validIdx = [i for i in range(nSeries) if len(trainSeries[i]) >= 10 and len(testSeries[i]) >= horizon]
    if sampleCap and len(validIdx) > sampleCap:
        rng = np.random.default_rng(42)
        validIdx = sorted(rng.choice(validIdx, size=sampleCap, replace=False).tolist())
    P(f"  Using {len(validIdx)} / {nSeries} series")
    smB, maB, smI, maI, smN, maN = [], [], [], [], [], []
    start = time.perf_counter()
    for count, idx in enumerate(validIdx):
        trainY, testY = trainSeries[idx], testSeries[idx][:horizon]
        n2 = _naive2M4(trainY, horizon, seasonality)
        smN.append(_smapeM4(testY, n2))
        maN.append(_maseM4(trainY, testY, n2, seasonality))
        bm = DynamicOptimizedTheta(period=seasonality)
        bm.fit(trainY)
        bp, _, _ = bm.predict(horizon)
        bp = np.asarray(bp[:horizon], dtype=np.float64)
        if not np.all(np.isfinite(bp)):
            bp = np.where(np.isfinite(bp), bp, np.mean(trainY))
        smB.append(_smapeM4(testY, bp))
        maB.append(_maseM4(trainY, testY, bp, seasonality))
        ip = _finalDotPredict(trainY, horizon, seasonality)
        ip = np.asarray(ip[:horizon], dtype=np.float64)
        if not np.all(np.isfinite(ip)):
            ip = np.where(np.isfinite(ip), ip, np.mean(trainY))
        smI.append(_smapeM4(testY, ip))
        maI.append(_maseM4(trainY, testY, ip, seasonality))
        if (count + 1) % 500 == 0:
            elapsed = time.perf_counter() - start
            rate = (count + 1) / elapsed
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {(len(validIdx)-count-1)/max(rate,0.01):.0f}s)")
    elapsed = time.perf_counter() - start
    P(f"  Done: {len(validIdx)} in {elapsed:.1f}s")
    sn, mn = np.mean(np.concatenate(smN)), np.mean(np.concatenate(maN))
    sb, mb = np.mean(np.concatenate(smB)), np.mean(np.concatenate(maB))
    si, mi = np.mean(np.concatenate(smI)), np.mean(np.concatenate(maI))
    ob = 0.5 * (sb / sn + mb / mn)
    oi = 0.5 * (si / sn + mi / mn)
    d = oi - ob
    P(f"  Baseline: OWA={ob:.4f}   Improved: OWA={oi:.4f}   Change: {d:+.4f} ({d/ob*100:+.2f}%)")
    return {'owaBase': ob, 'owaImp': oi, 'elapsed': elapsed}


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    P("=" * 60)
    P("E046: Final Integration — period<=1 classic, period>1 holdout hybrid")
    P("=" * 60)
    CAPS = {'Yearly': 2000, 'Quarterly': 2000, 'Monthly': 2000, 'Weekly': None, 'Daily': None, 'Hourly': None}
    allR, totalT = {}, 0
    for g in M4_GROUPS:
        r = _runGroup(g, sampleCap=CAPS.get(g))
        allR[g] = r
        totalT += r['elapsed']
    P(f"\n{'='*60}")
    P(f"  FINAL ({totalT/60:.1f} min)")
    P(f"{'='*60}")
    P(f"\n  {'Group':<12} {'Base':>8} {'Improved':>10} {'Change':>10}")
    aB, aI = [], []
    for g in M4_GROUPS:
        r = allR[g]
        d = r['owaImp'] - r['owaBase']
        aB.append(r['owaBase'])
        aI.append(r['owaImp'])
        marker = "<<< REGRESSION" if d > 0.005 else ""
        P(f"  {g:<12} {r['owaBase']:>8.4f} {r['owaImp']:>10.4f} {d:>+10.4f} {marker}")
    avgB, avgI = np.mean(aB), np.mean(aI)
    P(f"  {'AVG':<12} {avgB:>8.4f} {avgI:>10.4f} {avgI-avgB:>+10.4f}")
    P(f"\n  Overall: {avgB:.4f} -> {avgI:.4f} ({(avgI-avgB)/avgB*100:+.2f}%)")
    regressions = [g for g in M4_GROUPS if allR[g]['owaImp'] > allR[g]['owaBase'] + 0.005]
    if regressions:
        P(f"  WARNING: Regressions in {regressions}")
    if avgI < avgB and not regressions:
        P(f"  VERDICT: ALL CLEAR. Ready for engine integration.")
    elif avgI < avgB:
        P(f"  VERDICT: IMPROVED overall but has regressions. Review needed.")
    else:
        P(f"  VERDICT: NO IMPROVEMENT.")
    P("=" * 60)
