"""
==============================================================================
Experiment ID: modelCreation/044
Experiment: Daily/Weekly Specialist — Classic DOT vs Hybrid + Multi-Model
==============================================================================

Purpose:
- E043 showed auto_period detection is harmful for Daily/Weekly
- Daily OWA 0.9949 (near Naive2) is the biggest accuracy gap
- Hypothesis: for period=1 data, Classic DOT (3-param) may outperform
  Hybrid (8-way) because non-seasonal data doesn't need trend/season combos
- Also test Core3 ensemble (dot+ces+4theta) effect on Daily/Weekly

Hypotheses:
1. Classic DOT (force period=1, skip hybrid) beats Hybrid for Daily
2. Core3 ensemble improves over DOT-only for Daily/Weekly
3. holdout_val + classic DOT gives best Daily result

Method:
1. M4 Daily (4227) and Weekly (359) with variants:
   - baseline: current DOT-Hybrid (period=1, hybrid path)
   - classic_only: force classic DOT path even for period<24
   - holdout_hybrid: holdout validation + hybrid (from E043)
   - core3_hybrid: ensemble dot+ces+4theta (hybrid mode)
   - core3_classic: ensemble dot+ces+4theta (classic DOT)
   - holdout_classic: holdout + classic DOT
2. Also quick-check on Quarterly/Monthly to verify no regression

Results:
              baseline  classic_only  core3_hybrid  core3_classic
  Daily        0.9949      1.0047        1.2042        1.2078
  Weekly       0.9587      0.9378        1.0341        1.0326
  Quarterly    0.9029      0.9389        0.9185        0.9347
  Monthly      0.9100      0.9192        0.8966        0.9091

Conclusion:
- classic_only for Weekly: ACCEPTED (-2.18%). Hybrid 8-way overfits on period=1 data.
  Classic 3-param DOT is better when no real seasonality exists.
- classic_only for Daily: REJECTED (+0.98%). Hybrid still slightly better.
- Core3 ensemble for Daily/Weekly: REJECTED (+21%/+8%). CES and 4Theta perform
  poorly on period=1 non-seasonal data. Mixing weak models with strong DOT is harmful.
- Core3 for Monthly: modest benefit (-1.5%). E043 holdout_val was better (-2.55%).
- Daily OWA 0.9949 remains stubbornly near Naive2.
  All approaches tried so far cannot improve Daily. The data is mostly
  financial/economic with no seasonality — last-value (Naive) is hard to beat.
- Next steps: try CES-only for Daily (CES handles non-seasonal trend well),
  or accept Daily as structural limitation of statistical methods.

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
    'Daily':     {'horizon': 14, 'seasonality': 1,  'count': 4227},
    'Weekly':    {'horizon': 13, 'seasonality': 1,  'count': 359},
    'Quarterly': {'horizon': 8,  'seasonality': 4,  'count': 24000},
    'Monthly':   {'horizon': 18, 'seasonality': 12, 'count': 48000},
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
# Model implementations
# ============================================================

from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ces import AutoCES
from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
from vectrix.engine.turbo import TurboCore

try:
    from vectrix_core import dot_objective as _rustDotObj
    from vectrix_core import dot_residuals as _rustDotRes
    RUST_DOT = True
except ImportError:
    RUST_DOT = False

try:
    from vectrix_core import ses_filter as _rustSesFilter
    from vectrix_core import ses_sse as _rustSesSSE
    RUST_SES = True
except ImportError:
    RUST_SES = False


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


def _classicDotPredict(trainY, horizon, period):
    """Force classic DOT path regardless of period value."""
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


def _modelPredict(trainY, horizon, period, modelClass):
    """Generic model predict."""
    y = np.asarray(trainY, dtype=np.float64)
    model = modelClass(period=period)
    model.fit(y)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(y))
    return pred


def _core3Predict(trainY, horizon, period, useDotClassic=False):
    """Core3 ensemble: dot + auto_ces + four_theta with inverse-sMAPE weights."""
    y = np.asarray(trainY, dtype=np.float64)
    n = len(y)

    splitIdx = int(n * 0.8)
    if splitIdx < 5:
        splitIdx = max(5, n - horizon)
    cvTrain = y[:splitIdx]
    cvTest = y[splitIdx:]
    cvH = len(cvTest)

    models = {
        'dot': {'class': DynamicOptimizedTheta, 'classic': useDotClassic},
        'ces': {'class': AutoCES, 'classic': False},
        '4theta': {'class': AdaptiveThetaEnsemble, 'classic': False},
    }

    preds = {}
    cvSmapes = {}

    for mName, mInfo in models.items():
        if mName == 'dot' and mInfo['classic']:
            cvPred = _classicDotPredict(cvTrain, cvH, period)
            fullPred = _classicDotPredict(y, horizon, period)
        else:
            cvPred = _modelPredict(cvTrain, cvH, period, mInfo['class'])
            fullPred = _modelPredict(y, horizon, period, mInfo['class'])

        cvPred = np.asarray(cvPred[:cvH], dtype=np.float64)
        if not np.all(np.isfinite(cvPred)):
            cvPred = np.where(np.isfinite(cvPred), cvPred, np.mean(cvTrain))
        cvSmapes[mName] = np.mean(_smapeM4(cvTest, cvPred[:len(cvTest)]))

        fullPred = np.asarray(fullPred[:horizon], dtype=np.float64)
        if not np.all(np.isfinite(fullPred)):
            fullPred = np.where(np.isfinite(fullPred), fullPred, np.mean(y))
        preds[mName] = fullPred

    valid = [(preds[m], cvSmapes[m]) for m in models
             if np.all(np.isfinite(preds[m])) and cvSmapes[m] < 200]
    if valid:
        weights = np.array([1.0 / (s + 1e-6) for _, s in valid])
        weights /= np.sum(weights)
        ensemblePred = sum(weights[i] * valid[i][0] for i in range(len(valid)))
    else:
        ensemblePred = preds.get('dot', np.full(horizon, np.mean(y)))

    return ensemblePred


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

    variants = ['baseline', 'classic_only', 'core3_hybrid', 'core3_classic']
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

        dotHybrid = _modelPredict(trainY, horizon, seasonality, DynamicOptimizedTheta)
        allSmapes['baseline'].append(_smapeM4(testY, dotHybrid))
        allMases['baseline'].append(_maseM4(trainY, testY, dotHybrid, seasonality))

        dotClassic = _classicDotPredict(trainY, horizon, seasonality)
        dotClassic = np.asarray(dotClassic[:horizon], dtype=np.float64)
        if not np.all(np.isfinite(dotClassic)):
            dotClassic = np.where(np.isfinite(dotClassic), dotClassic, np.mean(trainY))
        allSmapes['classic_only'].append(_smapeM4(testY, dotClassic))
        allMases['classic_only'].append(_maseM4(trainY, testY, dotClassic, seasonality))

        c3h = _core3Predict(trainY, horizon, seasonality, useDotClassic=False)
        allSmapes['core3_hybrid'].append(_smapeM4(testY, c3h))
        allMases['core3_hybrid'].append(_maseM4(trainY, testY, c3h, seasonality))

        c3c = _core3Predict(trainY, horizon, seasonality, useDotClassic=True)
        allSmapes['core3_classic'].append(_smapeM4(testY, c3c))
        allMases['core3_classic'].append(_maseM4(trainY, testY, c3c, seasonality))

        if (count + 1) % 500 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} in {elapsed:.1f}s")

    smapeN2Avg = np.mean(np.concatenate(allSmapesN2))
    maseN2Avg = np.mean(np.concatenate(allMasesN2))
    P(f"  Naive2: sMAPE={smapeN2Avg:.3f}, MASE={maseN2Avg:.4f}")

    results = {}
    for vName in variants:
        sFlat = np.concatenate(allSmapes[vName])
        mFlat = np.concatenate(allMases[vName])
        sAvg = np.mean(sFlat)
        mAvg = np.mean(mFlat)
        owa = 0.5 * (sAvg / smapeN2Avg + mAvg / maseN2Avg)
        results[vName] = owa
        P(f"  {vName:20s}: OWA={owa:.4f}  sMAPE={sAvg:.3f}  MASE={mAvg:.4f}")

    return results, elapsed


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E044: Daily/Weekly Specialist — Classic DOT vs Hybrid + Core3")
    P("=" * 60)

    GROUP_CAPS = {
        'Daily': None, 'Weekly': None,
        'Quarterly': 1000, 'Monthly': 1000,
    }

    allResults = {}
    totalTime = 0

    for group in M4_GROUPS:
        results, elapsed = _runGroup(group, sampleCap=GROUP_CAPS.get(group))
        allResults[group] = results
        totalTime += elapsed

    P(f"\n{'='*60}")
    P(f"  SUMMARY ({totalTime/60:.1f} min)")
    P(f"{'='*60}")

    variants = ['baseline', 'classic_only', 'core3_hybrid', 'core3_classic']
    P(f"\n  {'Group':<12}", end="")
    for v in variants:
        P(f" {v:>15}", end="")
    P()

    for group in M4_GROUPS:
        P(f"  {group:<12}", end="")
        for v in variants:
            P(f" {allResults[group][v]:>15.4f}", end="")
        P()

    P(f"\n  Daily improvement over baseline:")
    bDaily = allResults['Daily']['baseline']
    for v in variants:
        if v == 'baseline':
            continue
        diff = allResults['Daily'][v] - bDaily
        P(f"    {v:20s}: {diff:+.4f} ({diff/bDaily*100:+.2f}%)")

    P(f"\n  Weekly improvement over baseline:")
    bWeekly = allResults['Weekly']['baseline']
    for v in variants:
        if v == 'baseline':
            continue
        diff = allResults['Weekly'][v] - bWeekly
        P(f"    {v:20s}: {diff:+.4f} ({diff/bWeekly*100:+.2f}%)")

    P("=" * 60)
