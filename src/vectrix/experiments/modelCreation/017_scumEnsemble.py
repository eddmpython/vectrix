"""
==============================================================================
Experiment ID: modelCreation/017
Experiment: SCUM Ensemble (Simple Combination of Univariate Models)
==============================================================================

Purpose:
- Implement M4 Competition 6th place method (OWA 0.848)
- Per-horizon median of DOT + AutoCES + AutoETS + AutoARIMA
- All 4 models already exist in Vectrix engine
- Target: OWA 0.85~0.87 (between M4 #6 SCUM and #11 4Theta)

Hypothesis:
1. Per-horizon median eliminates outlier predictions from any single model
2. DOT+CES+ETS+ARIMA cover different failure modes (trend vs seasonal vs noise)
3. Median > mean for robustness (Petropoulos & Svetunkov 2020)
4. Confidence intervals via median of individual model intervals

Method:
1. Fit DOT, AutoCES, AutoETS, AutoARIMA independently
2. For each horizon h: prediction = median(DOT_h, CES_h, ETS_h, ARIMA_h)
3. Lower/Upper = median of individual model intervals
4. If any model fails, fallback to remaining models' median
5. Compare vs DOT alone (0.905) and vs M4 SCUM (0.848)
6. Also test: mean combination, trimmed mean, weighted median

Results (M4 100K, 2000 sample/group):
              Yearly  Quarterly  Monthly  Weekly  Daily   Hourly  AVG
dot_solo      0.887   0.942      0.937    0.938   1.004   0.722   0.905
scum_median   0.978   0.963      0.937    0.966   1.000   0.704   0.925
scum_mean     0.992   1.136      0.959    1.8B    7.1B    0.790   EXPLODED

- SCUM Median: generally WORSE than DOT solo (AVG 0.925 vs 0.905)
- SCUM Mean: CATASTROPHIC failure on Weekly/Daily (ETS/ARIMA extreme predictions)
- Hourly: scum_median 0.704 > dot_solo 0.722 — only group where combination helps
- Monthly: scum_median 0.937 = dot_solo 0.937 — exact tie
- All other groups: scum_median < dot_solo (0.978>0.887, 0.963>0.942, etc.)

Conclusion:
- REJECTED: Full SCUM with 4 models does NOT work in Vectrix
- ROOT CAUSE: Vectrix ETS/ARIMA are weaker than R forecast package equivalents
- ETS/ARIMA drag down DOT/CES quality via median inclusion
- SCUM Mean is dangerous — ETS/ARIMA produce extreme outlier predictions
- KEY INSIGHT: Only DOT+CES pair is strong enough for combination
- ADOPTION: SCUM-Lite (DOT+CES only) tested in E018 — SCUM2 achieves Hourly 0.702
- Median is robust but only valuable when ALL constituent models are strong

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

from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ces import AutoCES
from vectrix.engine.ets import AutoETS
from vectrix.engine.arima import AutoARIMA


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


class SCUMEnsemble:
    """
    Simple Combination of Univariate Models (SCUM).
    M4 Competition 6th place (OWA 0.848).

    Per-horizon median of DOT + AutoCES + AutoETS + AutoARIMA.
    """

    def __init__(self, period=1):
        self.period = period
        self._models = {}
        self._fitted = {}
        self.fitted = False
        self.residuals = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        self._y = y.copy()

        factories = {
            'dot': lambda: DynamicOptimizedTheta(period=self.period),
            'ces': lambda: AutoCES(period=self.period),
            'ets': lambda: AutoETS(period=self.period),
            'arima': lambda: AutoARIMA(period=self.period),
        }

        for name, factory in factories.items():
            try:
                model = factory()
                model.fit(y)
                self._fitted[name] = model
            except Exception:
                pass

        if not self._fitted:
            fallback = DynamicOptimizedTheta(period=1)
            fallback.fit(y)
            self._fitted['dot'] = fallback

        allResiduals = []
        for name, model in self._fitted.items():
            if hasattr(model, 'residuals') and model.residuals is not None:
                r = np.asarray(model.residuals, dtype=np.float64)
                if len(r) > 0 and np.all(np.isfinite(r)):
                    allResiduals.append(r)

        if allResiduals:
            minLen = min(len(r) for r in allResiduals)
            stacked = np.column_stack([r[:minLen] for r in allResiduals])
            self.residuals = np.median(stacked, axis=1)
        else:
            self.residuals = np.zeros(len(y))

        self.fitted = True
        return self

    def predict(self, steps):
        if not self.fitted:
            raise ValueError("fit() must be called before predict()")

        allPreds = []
        allLowers = []
        allUppers = []

        for name, model in self._fitted.items():
            try:
                pred, lower, upper = model.predict(steps)
                pred = np.asarray(pred[:steps], dtype=np.float64)
                lower = np.asarray(lower[:steps], dtype=np.float64)
                upper = np.asarray(upper[:steps], dtype=np.float64)
                if np.all(np.isfinite(pred)):
                    allPreds.append(pred)
                    if np.all(np.isfinite(lower)):
                        allLowers.append(lower)
                    if np.all(np.isfinite(upper)):
                        allUppers.append(upper)
            except Exception:
                pass

        if not allPreds:
            meanVal = np.mean(self._y)
            return np.full(steps, meanVal), np.full(steps, meanVal * 0.8), np.full(steps, meanVal * 1.2)

        predictions = np.median(np.array(allPreds), axis=0)

        if allLowers:
            lower = np.median(np.array(allLowers), axis=0)
        else:
            sigma = np.std(self.residuals) if len(self.residuals) > 1 else 1.0
            lower = predictions - 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

        if allUppers:
            upper = np.median(np.array(allUppers), axis=0)
        else:
            sigma = np.std(self.residuals) if len(self.residuals) > 1 else 1.0
            upper = predictions + 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

        return predictions, lower, upper


class SCUMMean:
    """Mean combination variant (for comparison with median)."""

    def __init__(self, period=1):
        self.period = period
        self.fitted = False
        self.residuals = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        self._y = y.copy()

        self._models = {}
        factories = {
            'dot': lambda: DynamicOptimizedTheta(period=self.period),
            'ces': lambda: AutoCES(period=self.period),
            'ets': lambda: AutoETS(period=self.period),
            'arima': lambda: AutoARIMA(period=self.period),
        }
        for name, factory in factories.items():
            try:
                model = factory()
                model.fit(y)
                self._models[name] = model
            except Exception:
                pass

        if not self._models:
            fallback = DynamicOptimizedTheta(period=1)
            fallback.fit(y)
            self._models['dot'] = fallback

        self.residuals = np.zeros(len(y))
        self.fitted = True
        return self

    def predict(self, steps):
        allPreds = []
        for name, model in self._models.items():
            try:
                pred, _, _ = model.predict(steps)
                pred = np.asarray(pred[:steps], dtype=np.float64)
                if np.all(np.isfinite(pred)):
                    allPreds.append(pred)
            except Exception:
                pass

        if not allPreds:
            return np.full(steps, np.mean(self._y)), np.full(steps, 0), np.full(steps, 0)

        predictions = np.mean(np.array(allPreds), axis=0)
        sigma = np.std(np.array(allPreds), axis=0)
        lower = predictions - 1.96 * sigma
        upper = predictions + 1.96 * sigma
        return predictions, lower, upper


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
    P(f"  Loaded {len(validIdx)}/{nSeries} valid series")

    SAMPLE_CAP = 2000
    if len(validIdx) > SAMPLE_CAP:
        rng = np.random.default_rng(42)
        validIdx = sorted(rng.choice(validIdx, size=SAMPLE_CAP, replace=False).tolist())
        P(f"  Sampled {SAMPLE_CAP} series")

    models = {
        'dot_solo': lambda: DynamicOptimizedTheta(period=seasonality),
        'scum_median': lambda: SCUMEnsemble(period=seasonality),
        'scum_mean': lambda: SCUMMean(period=seasonality),
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

        for mName, factory in models.items():
            try:
                pred = _fitPredict(factory(), trainY, horizon)
                results[mName]['smapes'].append(_smape(testY, pred))
                results[mName]['mases'].append(_mase(trainY, testY, pred, seasonality))
            except Exception:
                results[mName]['smapes'].append(results['naive2']['smapes'][-1])
                results[mName]['mases'].append(results['naive2']['mases'][-1])
                errors[mName] += 1

        if (count + 1) % 100 == 0:
            elapsed = time.perf_counter() - startTime
            speed = (count + 1) / elapsed
            eta = (len(validIdx) - count - 1) / max(speed, 0.01)
            P(f"    {count+1}/{len(validIdx)} ({speed:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {len(validIdx)} series in {elapsed:.1f}s")

    n2Smape = np.mean(results['naive2']['smapes'])
    n2Mase = np.mean(results['naive2']['mases'])

    P(f"\n  {'Model':<18} {'sMAPE':>8} {'MASE':>8} {'OWA':>8} {'Err':>5}")
    P(f"  {'-'*52}")

    groupOwas = {}
    for mName in models:
        avgSmape = np.mean(results[mName]['smapes'])
        avgMase = np.mean(results[mName]['mases'])
        owa = 0.5 * (avgSmape / max(n2Smape, 1e-10) + avgMase / max(n2Mase, 1e-10))
        groupOwas[mName] = owa
        P(f"  {mName:<18} {avgSmape:>8.2f} {avgMase:>8.3f} {owa:>8.3f} {errors[mName]:>5}")
    P(f"  {'naive2':<18} {n2Smape:>8.2f} {n2Mase:>8.3f} {'1.000':>8}")

    return groupName, groupOwas


def _runExperiment():
    P("=" * 60)
    P("E017: SCUM Ensemble (M4 Competition 6th Place Method)")
    P("  Per-horizon median of DOT + AutoCES + AutoETS + AutoARIMA")
    P("=" * 60)

    allOwas = {}
    totalStart = time.perf_counter()

    for groupName in M4_GROUPS:
        gName, gOwas = _runGroup(groupName)
        allOwas[gName] = gOwas

    totalElapsed = time.perf_counter() - totalStart

    P(f"\n{'='*60}")
    P(f"  OVERALL RESULTS ({totalElapsed/60:.1f} min)")
    P(f"{'='*60}")

    modelNames = ['dot_solo', 'scum_median', 'scum_mean']
    P(f"\n  {'Model':<18}", end='')
    for g in M4_GROUPS:
        P(f" {g[:5]:>7}", end='')
    P(f" {'AVG':>7}")
    P(f"  {'-' * (18 + 8 * (len(M4_GROUPS) + 1))}")

    for mName in modelNames:
        row = f"  {mName:<18}"
        owas = []
        for g in M4_GROUPS:
            if g in allOwas and mName in allOwas[g]:
                owa = allOwas[g][mName]
                row += f" {owa:>7.3f}"
                owas.append(owa)
            else:
                row += f" {'N/A':>7}"
        avgOwa = np.mean(owas) if owas else float('nan')
        row += f" {avgOwa:>7.3f}"
        P(row)

    P(f"\n  M4 Reference: SCUM(#6) 0.848, 4Theta(#11) 0.874, Theta(#18) 0.897")
    P(f"  Current DOT baseline: OWA 0.905")
    P(f"\n{'='*60}")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
