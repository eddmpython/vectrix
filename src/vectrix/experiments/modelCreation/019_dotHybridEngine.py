"""
==============================================================================
Experiment ID: modelCreation/019
Experiment: DOT-Hybrid Engine Verification (Integrated engine/dot.py)
==============================================================================

Purpose:
- Verify that the integrated DOT-Hybrid in engine/dot.py matches E018 results
- Confirm Rust acceleration (26 functions) works correctly
- Measure speed improvement from vectorized + Rust dot_hybrid_objective

Hypothesis:
1. Integrated DOT-Hybrid should match E018 results: AVG OWA ~0.884
2. Rust dot_hybrid_objective should be faster than pure Python
3. All 6 M4 groups should show consistent results with E018

Method:
1. Use engine/dot.py DynamicOptimizedTheta directly (not experiment class)
2. Run M4 100K benchmark (2000 sample/group, same seed=42)
3. Compare vs E018 results and vs current DOT baseline (0.905)

Results (M4 100K, 2000 sample/group):
             Yearly  Quarterly  Monthly  Weekly  Daily   Hourly  AVG
dot_engine   0.797   0.905      0.933    0.959   0.996   0.722   0.885

Speed (series/s): Yearly 331, Quarterly 125, Monthly 82, Weekly 58, Daily 43, Hourly 232
Total: 1.7 min (vs E016 16.6 min = 9.8x faster)

vs E018 reference:
dot_hybrid   0.796   0.904      0.931    0.957   0.996   0.722   0.884  (diff: +0.001)

Conclusion:
- VERIFIED: Integrated engine matches E018 within 0.001 OWA (Rust golden section vs scipy)
- SPEED: 9.8x faster than E016 thanks to Rust dot_hybrid_objective + vectorized NumPy
- Rust 26 functions all active: DOT=True, SES=True, Hybrid=True
- ADOPTED: DOT-Hybrid engine is production-ready in engine/dot.py

Experiment date: 2026-03-03
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
    'Yearly':    {'horizon': 6,  'seasonality': 1},
    'Quarterly': {'horizon': 8,  'seasonality': 4},
    'Monthly':   {'horizon': 18, 'seasonality': 12},
    'Weekly':    {'horizon': 13, 'seasonality': 1},
    'Daily':     {'horizon': 14, 'seasonality': 1},
    'Hourly':    {'horizon': 48, 'seasonality': 24},
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


def _fitPredict(model, trainY, horizon):
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred


def _runGroup(groupName):
    from vectrix.engine.dot import DynamicOptimizedTheta, RUST_AVAILABLE, SES_RUST_AVAILABLE, HYBRID_RUST_AVAILABLE

    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    seasonality = info['seasonality']

    P(f"\n{'='*60}")
    P(f"  {groupName}: h={horizon}, m={seasonality}")
    P(f"  Rust: DOT={RUST_AVAILABLE}, SES={SES_RUST_AVAILABLE}, Hybrid={HYBRID_RUST_AVAILABLE}")
    P(f"{'='*60}")

    trainSeries, testSeries = _loadGroup(groupName)
    nSeries = len(trainSeries)

    validIdx = [i for i in range(nSeries)
                if len(trainSeries[i]) >= 5 and len(testSeries[i]) >= horizon]
    P(f"  Loaded {len(validIdx)}/{nSeries} valid series")

    SAMPLE_CAP = 2000
    if len(validIdx) > SAMPLE_CAP:
        rng = np.random.default_rng(42)
        validIdx = sorted(rng.choice(validIdx, size=SAMPLE_CAP, replace=False).tolist())
        P(f"  Sampled {SAMPLE_CAP} series")

    models = {
        'dot_engine': lambda: DynamicOptimizedTheta(period=seasonality),
    }

    results = {name: {'smapes': [], 'mases': []} for name in models}
    results['naive2'] = {'smapes': [], 'mases': []}
    errors = {name: 0 for name in models}

    startTime = time.perf_counter()

    for count, idx in enumerate(validIdx):
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]

        n2pred = _naive2(trainY, horizon, seasonality)
        results['naive2']['smapes'].append(_smape(testY, n2pred))
        results['naive2']['mases'].append(_mase(trainY, testY, n2pred, seasonality))

        for modelName, modelFn in models.items():
            try:
                pred = _fitPredict(modelFn(), trainY, horizon)
                results[modelName]['smapes'].append(_smape(testY, pred))
                results[modelName]['mases'].append(_mase(trainY, testY, pred, seasonality))
            except Exception:
                errors[modelName] += 1

        if (count + 1) % 200 == 0:
            elapsed = time.perf_counter() - startTime
            rate = (count + 1) / elapsed
            remaining = (len(validIdx) - count - 1) / max(rate, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({rate:.1f}/s, ETA {remaining:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} series in {elapsed:.1f}s")

    n2Smape = np.mean(results['naive2']['smapes'])
    n2Mase = np.mean(results['naive2']['mases'])

    P(f"\n  {'Model':<20} {'sMAPE':>8} {'MASE':>8} {'OWA':>8}   Err")
    P(f"  {'-'*52}")

    owas = {}
    for modelName in models:
        if len(results[modelName]['smapes']) > 0:
            smape = np.mean(results[modelName]['smapes'])
            mase = np.mean(results[modelName]['mases'])
            owa = 0.5 * (smape / max(n2Smape, 1e-10) + mase / max(n2Mase, 1e-10))
            owas[modelName] = owa
            P(f"  {modelName:<20} {smape:>8.2f} {mase:>8.3f} {owa:>8.3f}   {errors[modelName]:>3}")
        else:
            owas[modelName] = np.nan

    P(f"  {'naive2':<20} {n2Smape:>8.2f} {n2Mase:>8.3f} {'1.000':>8}")

    return owas, elapsed


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from vectrix.engine.dot import RUST_AVAILABLE, SES_RUST_AVAILABLE, HYBRID_RUST_AVAILABLE

    P("=" * 60)
    P("E019: DOT-Hybrid Engine Verification (Integrated)")
    P(f"  Rust: DOT={RUST_AVAILABLE}, SES={SES_RUST_AVAILABLE}, Hybrid={HYBRID_RUST_AVAILABLE}")
    P("=" * 60)

    allOwas = {}
    totalTime = 0
    for group in M4_GROUPS:
        owas, elapsed = _runGroup(group)
        allOwas[group] = owas
        totalTime += elapsed

    P(f"\n{'='*60}")
    P(f"  OVERALL RESULTS ({totalTime/60:.1f} min)")
    P(f"{'='*60}")

    modelNames = ['dot_engine']
    P(f"\n  {'Model':<20} {'Yearl':>6} {'Quart':>6} {'Month':>6} {'Weekl':>6} {'Daily':>6} {'Hourl':>6}   {'AVG':>6}")
    P(f"  {'-'*74}")

    for modelName in modelNames:
        vals = [allOwas[g].get(modelName, np.nan) for g in M4_GROUPS]
        avg = np.nanmean(vals)
        line = f"  {modelName:<20}"
        for v in vals:
            line += f" {v:>6.3f}"
        line += f"   {avg:>6.3f}"
        P(line)

    P(f"\n  E018 Reference: dot_hybrid AVG=0.884, combined AVG=0.888")
    P(f"  M4 Reference: #1 ES-RNN 0.821, #2 FFORMA 0.838, #11 4Theta 0.874")
    P(f"  Current DOT baseline: 0.905")
    P(f"\n{'='*60}")
