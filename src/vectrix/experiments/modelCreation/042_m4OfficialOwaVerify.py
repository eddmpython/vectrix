"""
==============================================================================
Experiment ID: modelCreation/042
Experiment: M4 Official OWA Calculation Verification
==============================================================================

Purpose:
- Previous benchmarks used 6-group simple average OWA (each group = 1/6 weight)
- M4 official: 100K series weighted by count (Monthly 48K >> Hourly 414)
- Naive2 implementation may differ from M4 official R decompose()
- Need to verify our OWA claim is valid under M4 official methodology

Issues found:
1. Overall OWA: 6-group avg vs 100K series-count-weighted
2. Naive2: custom decomposition vs M4 official SeasonalityTest
3. Sampling: 2000 cap may not represent full group

Method:
1. Use M4 official Naive2 values (pre-computed from R) or implement exact R logic
2. Calculate OWA both ways: our method vs M4 official method
3. Run on ALL series (no sampling cap) for at least 2 groups to verify
4. Compare our Naive2 vs M4 official Naive2

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
    """M4 official sMAPE: |Y-F|*200 / (|Y|+|F|), returns per-horizon vector"""
    return np.abs(actual - predicted) * 200.0 / (np.abs(actual) + np.abs(predicted) + 1e-10)


def _maseM4(trainY, actual, predicted, seasonality):
    """M4 official MASE: per-horizon vector, not scalar average"""
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
    """M4 official SeasonalityTest (R implementation)"""
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
    limit = tcrit / np.sqrt(n)

    cumsum = 1.0
    for k in range(1, ppy):
        if k < len(acf):
            cumsum += 2 * acf[k] ** 2
    limit_adj = tcrit * np.sqrt(cumsum) / np.sqrt(n)

    if ppy < len(acf):
        return abs(acf[ppy]) > limit_adj
    return False


def _naive2M4(trainY, horizon, seasonality):
    """M4 official Naive2: SeasonalityTest + multiplicative decomposition"""
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
            if not np.isnan(seasonal[i]) if isinstance(seasonal[i], float) else not np.isnan(seasonal[i]):
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


def _naive2Old(trainY, horizon, seasonality):
    """Our old Naive2 implementation (from previous experiments)"""
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

    allSmapesMethod = []
    allMasesMethod = []
    allSmapesN2Official = []
    allMasesN2Official = []
    allSmapesN2Old = []
    allMasesN2Old = []

    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
    from concurrent.futures import ThreadPoolExecutor

    startTime = time.perf_counter()

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]

        n2Official = _naive2M4(trainY, horizon, seasonality)
        n2Old = _naive2Old(trainY, horizon, seasonality)

        allSmapesN2Official.append(_smapeM4(testY, n2Official))
        allMasesN2Official.append(_maseM4(trainY, testY, n2Official, seasonality))
        allSmapesN2Old.append(_smapeM4(testY, n2Old))
        allMasesN2Old.append(_maseM4(trainY, testY, n2Old, seasonality))

        dotModel = DynamicOptimizedTheta(period=seasonality)
        cesModel = AutoCES(period=seasonality)
        ftModel = AdaptiveThetaEnsemble(period=seasonality)

        preds = {}
        cvSmapes = {}
        n = len(trainY)
        splitIdx = int(n * 0.8)
        if splitIdx < 5:
            splitIdx = max(5, n - horizon)
        cvTrainY = trainY[:splitIdx]
        cvTestY = trainY[splitIdx:]
        cvH = len(cvTestY)

        for mName, ModelClass in [('dot', DynamicOptimizedTheta),
                                   ('auto_ces', AutoCES),
                                   ('four_theta', AdaptiveThetaEnsemble)]:
            try:
                cvM = ModelClass(period=seasonality)
                cvM.fit(cvTrainY)
                cvP, _, _ = cvM.predict(cvH)
                cvP = np.asarray(cvP[:cvH], dtype=np.float64)
                if not np.all(np.isfinite(cvP)):
                    cvP = np.where(np.isfinite(cvP), cvP, np.mean(cvTrainY))
                cvSmapes[mName] = np.mean(_smapeM4(cvTestY, cvP[:len(cvTestY)]))

                fullM = ModelClass(period=seasonality)
                fullM.fit(trainY)
                fullP, _, _ = fullM.predict(horizon)
                fullP = np.asarray(fullP[:horizon], dtype=np.float64)
                if not np.all(np.isfinite(fullP)):
                    fullP = np.where(np.isfinite(fullP), fullP, np.mean(trainY))
                preds[mName] = fullP
            except Exception:
                cvSmapes[mName] = 999.0
                preds[mName] = np.full(horizon, np.mean(trainY))

        dotPred = preds.get('dot', np.full(horizon, np.mean(trainY)))

        valid = [(preds[m], cvSmapes[m]) for m in ['dot', 'auto_ces', 'four_theta']
                 if np.all(np.isfinite(preds[m])) and cvSmapes[m] < 200]
        if valid:
            weights = np.array([1.0 / (m + 1e-6) for _, m in valid])
            weights /= np.sum(weights)
            safe3Pred = sum(weights[i] * valid[i][0] for i in range(len(valid)))
        else:
            safe3Pred = dotPred

        origStd = np.std(trainY[-min(30, len(trainY)):])
        safe3Std = np.std(safe3Pred)
        dotStd = np.std(dotPred)
        if abs(safe3Std - origStd) < abs(dotStd - origStd):
            methodPred = safe3Pred
        else:
            methodPred = dotPred

        allSmapesMethod.append(_smapeM4(testY, methodPred))
        allMasesMethod.append(_maseM4(trainY, testY, methodPred, seasonality))

        if (count + 1) % 500 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} in {elapsed:.1f}s")

    smapeMethodFlat = np.concatenate(allSmapesMethod)
    maseMethodFlat = np.concatenate(allMasesMethod)
    smapeN2OfficialFlat = np.concatenate(allSmapesN2Official)
    maseN2OfficialFlat = np.concatenate(allMasesN2Official)
    smapeN2OldFlat = np.concatenate(allSmapesN2Old)
    maseN2OldFlat = np.concatenate(allMasesN2Old)

    smapeMethodAvg = np.mean(smapeMethodFlat)
    maseMethodAvg = np.mean(maseMethodFlat)
    smapeN2OfficialAvg = np.mean(smapeN2OfficialFlat)
    maseN2OfficialAvg = np.mean(maseN2OfficialFlat)
    smapeN2OldAvg = np.mean(smapeN2OldFlat)
    maseN2OldAvg = np.mean(maseN2OldFlat)

    owaOfficial = 0.5 * (smapeMethodAvg / smapeN2OfficialAvg + maseMethodAvg / maseN2OfficialAvg)
    owaOld = 0.5 * (smapeMethodAvg / smapeN2OldAvg + maseMethodAvg / maseN2OldAvg)

    owaPerSeriesOfficial = []
    owaPerSeriesOld = []
    for i in range(len(validIdx)):
        sM = np.mean(allSmapesMethod[i])
        mM = np.mean(allMasesMethod[i])
        sNO = np.mean(allSmapesN2Official[i])
        mNO = np.mean(allMasesN2Official[i])
        sNL = np.mean(allSmapesN2Old[i])
        mNL = np.mean(allMasesN2Old[i])
        if sNO > 0 and mNO > 0:
            owaPerSeriesOfficial.append(0.5 * (sM / sNO + mM / mNO))
        if sNL > 0 and mNL > 0:
            owaPerSeriesOld.append(0.5 * (sM / sNL + mM / mNL))

    P(f"\n  Naive2 comparison:")
    P(f"    Official Naive2: avg sMAPE={smapeN2OfficialAvg:.2f}, avg MASE={maseN2OfficialAvg:.4f}")
    P(f"    Old Naive2:      avg sMAPE={smapeN2OldAvg:.2f}, avg MASE={maseN2OldAvg:.4f}")
    P(f"    Difference:      sMAPE={abs(smapeN2OfficialAvg-smapeN2OldAvg):.4f}, MASE={abs(maseN2OfficialAvg-maseN2OldAvg):.6f}")

    P(f"\n  Method (smart_safe3): avg sMAPE={smapeMethodAvg:.2f}, avg MASE={maseMethodAvg:.4f}")

    P(f"\n  OWA calculation comparison:")
    P(f"    M4 Official Naive2: OWA = {owaOfficial:.4f}")
    P(f"    Old Naive2:         OWA = {owaOld:.4f}")
    P(f"    Difference:         {abs(owaOfficial - owaOld):.4f}")

    if owaPerSeriesOfficial:
        P(f"    Per-series OWA avg (official): {np.mean(owaPerSeriesOfficial):.4f}  (for reference, NOT used)")

    return {
        'groupName': groupName,
        'nSeries': len(validIdx),
        'owaOfficial': owaOfficial,
        'owaOld': owaOld,
        'smapeMethod': smapeMethodAvg,
        'maseMethod': maseMethodAvg,
        'smapeN2Official': smapeN2OfficialAvg,
        'maseN2Official': maseN2OfficialAvg,
        'smapeN2Old': smapeN2OldAvg,
        'maseN2Old': maseN2OldAvg,
        'smapeMethodFlat': smapeMethodFlat,
        'maseMethodFlat': maseMethodFlat,
        'smapeN2OfficialFlat': smapeN2OfficialFlat,
        'maseN2OfficialFlat': maseN2OfficialFlat,
        'elapsed': elapsed,
    }


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 60)
    P("E042: M4 Official OWA Calculation Verification")
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
    P(f"  OVERALL COMPARISON ({totalTime/60:.1f} min)")
    P(f"{'='*60}")

    P(f"\n  === Method 1: 6-group simple average (our previous method) ===")
    owaOfficialAvg = np.mean([allResults[g]['owaOfficial'] for g in M4_GROUPS])
    owaOldAvg = np.mean([allResults[g]['owaOld'] for g in M4_GROUPS])
    P(f"  Official Naive2 basis: {owaOfficialAvg:.4f}")
    P(f"  Old Naive2 basis:      {owaOldAvg:.4f}")
    for g in M4_GROUPS:
        r = allResults[g]
        P(f"    {g:<12} official={r['owaOfficial']:.4f}  old={r['owaOld']:.4f}  diff={abs(r['owaOfficial']-r['owaOld']):.4f}  n={r['nSeries']}")

    P(f"\n  === Method 2: M4 official (100K weighted by series count) ===")

    allSmapeMethod = np.concatenate([allResults[g]['smapeMethodFlat'] for g in M4_GROUPS])
    allMaseMethod = np.concatenate([allResults[g]['maseMethodFlat'] for g in M4_GROUPS])
    allSmapeN2 = np.concatenate([allResults[g]['smapeN2OfficialFlat'] for g in M4_GROUPS])
    allMaseN2 = np.concatenate([allResults[g]['maseN2OfficialFlat'] for g in M4_GROUPS])

    globalOwa = 0.5 * (np.mean(allSmapeMethod) / np.mean(allSmapeN2)
                       + np.mean(allMaseMethod) / np.mean(allMaseN2))
    P(f"  Global OWA (M4 official style): {globalOwa:.4f}")

    totalPoints = sum(allResults[g]['nSeries'] for g in M4_GROUPS)
    P(f"  Total data points: {totalPoints} series")

    P(f"\n  === Comparison ===")
    P(f"  6-group avg (old naive2):     {owaOldAvg:.4f}")
    P(f"  6-group avg (official naive2): {owaOfficialAvg:.4f}")
    P(f"  Global weighted (official):    {globalOwa:.4f}")
    P(f"  Difference (old vs global):    {abs(owaOldAvg - globalOwa):.4f}")

    P(f"\n  M4 reference: #1 ES-RNN=0.821  #2 FFORMA=0.838  #11 4Theta=0.874  #18 Theta=0.897")
    P("=" * 60)
