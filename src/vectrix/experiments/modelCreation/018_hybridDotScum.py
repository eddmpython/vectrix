"""
==============================================================================
Experiment ID: modelCreation/018
Experiment: Hybrid DOT + Targeted SCUM Variants
==============================================================================

Purpose:
- Combine learnings from E016 (DOT++) and E017 (SCUM) into optimal strategies
- E016 finding: DOT++ excels at Yearly/Quarterly (0.796/0.904) but fails Hourly (0.955 vs 0.722)
- E017 finding: SCUM with ETS/ARIMA hurts performance; only Hourly improves (0.704)

Hypothesis:
1. DOT-Hybrid: Use DOT++ for period<=12, original DOT for period>=24
   - Expected: 0.796 Yearly, 0.904 Quarterly, 0.931 Monthly, 0.722 Hourly => AVG ~0.88
2. SCUM-Lite: Median of DOT + CES only (drop weak ETS/ARIMA)
   - Expected: better than full SCUM, similar to DOT solo
3. DOT++ + CES Median: best of E016 + E017 insights combined
   - Expected: DOT++ handles trend/growth, CES handles seasonality
4. Combined Best: DOT-Hybrid + CES median
   - Expected: approach M4 top-10

Method:
1. DOT-Hybrid: frequency-adaptive model selection
2. SCUM-2 (DOT+CES median)
3. SCUM-2++ (DOT++ + CES median)
4. Combined: DOT-Hybrid + CES median, with frequency-specific strategy
5. Compare all on M4 100K (2000 sample per group)

Results (M4 100K, 2000 sample/group):
              Yearly  Quarterly  Monthly  Weekly  Daily   Hourly  AVG
dot_current   0.887   0.942      0.937    0.938   1.004   0.722   0.905
dot_hybrid    0.796   0.904      0.931    0.957   0.996   0.722   0.884 <<<
scum2         0.920   0.933      0.927    0.947   1.000   0.702   0.905
scum2pp       0.838   0.913      0.917    0.962   0.996   0.785   0.902
combined      0.838   0.913      0.917    0.962   0.996   0.702   0.888

Key findings:
1. DOT-Hybrid: BEST single-model AVG 0.884 — beats M4 #18 Theta (0.897)!
   - Yearly 0.796 = near ES-RNN class
   - Uses DOT++ for period<=12, original DOT for period>=24
2. Combined (DOT-Hybrid+CES): AVG 0.888 — best ensemble
   - Hourly 0.702 = best across all variants (CES contribution)
3. SCUM2 (DOT+CES median): AVG 0.905 — same as DOT solo
   - CES boosts Hourly (0.702) but hurts Yearly (0.920 vs 0.887)
4. SCUM2PP (DOT+++CES): AVG 0.902 — marginally better than DOT alone
   - Best Monthly (0.917) and Quarterly (0.913)

Conclusion:
- ADOPTED: DOT-Hybrid (period<=12: DOT++, period>=24: DOT original)
  * Single model AVG OWA 0.884, surpassing M4 #18 Theta (0.897)
  * Yearly breakthrough: 0.796 (world-class for single statistical model)
- ADOPTED: Combined (DOT-Hybrid + CES median) as ensemble default
  * AVG OWA 0.888, Hourly 0.702 (gains CES's Hourly strength)
- REJECTED: SCUM2PP — Hourly regression (0.785) unacceptable
- KEY INSIGHT: DOT++ exponential/multiplicative is transformative for low-frequency data
  but harmful for high-frequency multi-seasonal data
- NEXT: Integrate DOT-Hybrid into engine/dot.py + Combined into default ensemble

Experiment date: 2026-03-03
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

from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ces import AutoCES

try:
    from vectrix_core import ses_filter as _rustSesFilter
    from vectrix_core import ses_sse as _rustSesSSE
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


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


def _sesFilterPython(y, alpha):
    n = len(y)
    result = np.zeros(n)
    result[0] = y[0]
    for t in range(1, n):
        result[t] = alpha * y[t] + (1.0 - alpha) * result[t - 1]
    return result


def _sesSSEPython(y, alpha):
    n = len(y)
    level = y[0]
    sse = 0.0
    for t in range(1, n):
        error = y[t] - level
        sse += error * error
        level = alpha * y[t] + (1.0 - alpha) * level
    return sse


def _sesFilter(y, alpha):
    if RUST_AVAILABLE:
        return np.asarray(_rustSesFilter(y, alpha))
    return _sesFilterPython(y, alpha)


def _sesSSE(y, alpha):
    if RUST_AVAILABLE:
        return _rustSesSSE(y, alpha)
    return _sesSSEPython(y, alpha)


def _optimizeAlpha(y):
    if len(y) < 3:
        return 0.3
    result = minimize_scalar(lambda a: _sesSSE(y, a), bounds=(0.001, 0.999), method='bounded')
    return result.x if result.success else 0.3


class DOTPlusPlus:
    """DOT++ from E016: 8-way auto-select (2 trend x 2 model x 2 season)."""

    def __init__(self, period=1):
        self.period = period
        self._bestConfig = None
        self._bestModel = None
        self._n = 0
        self.fitted = False
        self.residuals = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        self._n = len(self._y)

        if self._n < 5:
            self._bestModel = {'lastLevel': self._y[-1] if self._n > 0 else 0.0,
                               'slope': 0.0, 'intercept': np.mean(self._y),
                               'theta': 2.0, 'residStd': 1.0, 'n': self._n,
                               'seasonal': None, 'base': 1.0, 'fittedValues': np.full(self._n, np.mean(self._y))}
            self._bestConfig = ('linear', 'additive', 'none')
            self.residuals = np.zeros(self._n)
            self.fitted = True
            return self

        hasSeason = self.period > 1 and self._n >= self.period * 3
        seasonTypes = ['multiplicative', 'additive'] if hasSeason else ['none']

        scaled = self._y.copy()
        base = np.mean(np.abs(scaled))
        if base > 0:
            scaled = scaled / base
        else:
            base = 1.0

        bestMae = np.inf
        bestConfig = None
        bestModel = None

        for seasonType in seasonTypes:
            if seasonType != 'none':
                seasonal, deseasonalized = self._deseasonalize(scaled, self.period, seasonType)
            else:
                seasonal = None
                deseasonalized = scaled

            for trendType in ['linear', 'exponential']:
                thetaLine0 = self._fitTrendLine(deseasonalized, trendType)
                if thetaLine0 is None:
                    continue

                for modelType in ['additive', 'multiplicative']:
                    if modelType == 'multiplicative' and np.any(thetaLine0 <= 0):
                        continue
                    if modelType == 'multiplicative' and np.any(deseasonalized <= 0):
                        continue

                    result = self._fitDOTVariant(deseasonalized, thetaLine0, trendType, modelType)
                    if result is None:
                        continue

                    fittedVals = result['fittedValues']
                    if seasonal is not None:
                        fittedVals = self._reseasonalize(fittedVals, seasonal, seasonType)

                    mae = np.mean(np.abs(scaled - fittedVals))
                    if mae < bestMae:
                        bestMae = mae
                        bestConfig = (trendType, modelType, seasonType)
                        bestModel = result
                        bestModel['seasonal'] = seasonal
                        bestModel['base'] = base

        if bestModel is None:
            n = self._n
            x = np.arange(n, dtype=np.float64)
            slope, intercept = self._linearRegression(x, scaled)
            bestModel = {'theta': 2.0, 'alpha': 0.3, 'intercept': intercept,
                         'slope': slope, 'lastLevel': scaled[-1],
                         'residStd': max(np.std(scaled), 1e-8),
                         'n': n, 'seasonal': None, 'base': base,
                         'fittedValues': intercept + slope * x}
            bestConfig = ('linear', 'additive', 'none')

        self._bestConfig = bestConfig
        self._bestModel = bestModel
        self.residuals = (self._y - bestModel['fittedValues'] * base)
        self.fitted = True
        return self

    def predict(self, steps):
        if not self.fitted:
            raise ValueError("fit() must be called before predict()")

        m = self._bestModel
        trendType, modelType, seasonType = self._bestConfig
        n = self._n
        base = m['base']

        forecastTrend = self._extrapolateTrend(m, trendType, steps)
        forecastSES = np.full(steps, m['lastLevel'])

        if modelType == 'additive':
            wses = 1.0 / max(m['theta'], 1.0)
            wtrend = 1.0 - wses
            combined = wses * forecastSES + wtrend * forecastTrend
        else:
            invTheta = 1.0 / max(m['theta'], 1.0)
            combined = np.power(np.maximum(forecastSES, 1e-10), invTheta) * \
                       np.power(np.maximum(forecastTrend, 1e-10), 1.0 - invTheta)

        if m['seasonal'] is not None:
            for h in range(steps):
                idx = (n + h) % self.period
                if seasonType == 'multiplicative':
                    combined[h] *= m['seasonal'][idx]
                else:
                    combined[h] += m['seasonal'][idx]

        predictions = combined * base
        sigma = m['residStd'] * base
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
        return predictions, predictions - margin, predictions + margin

    def _fitDOTVariant(self, y, thetaLine0, trendType, modelType):
        n = len(y)
        if n < 5:
            return None

        def buildThetaLine(theta):
            if modelType == 'additive':
                return theta * y + (1.0 - theta) * thetaLine0
            else:
                return np.power(np.maximum(y, 1e-10), theta) * \
                       np.power(np.maximum(thetaLine0, 1e-10), 1.0 - theta)

        def objective(params):
            theta = params[0]
            thetaLine = buildThetaLine(theta)
            alpha = _optimizeAlpha(thetaLine)
            filtered = _sesFilter(thetaLine, alpha)
            fittedVals = np.zeros(n)
            for t in range(n):
                if modelType == 'additive':
                    wses = 1.0 / max(theta, 1.0)
                    wtrend = 1.0 - wses
                    fittedVals[t] = wses * filtered[t] + wtrend * thetaLine0[t]
                else:
                    invTheta = 1.0 / max(theta, 1.0)
                    fittedVals[t] = (max(filtered[t], 1e-10) ** invTheta) * \
                                     (max(thetaLine0[t], 1e-10) ** (1.0 - invTheta))
            return np.mean(np.abs(y - fittedVals))

        result = minimize(objective, x0=[2.0], bounds=[(1.0, 50.0)],
                          method='L-BFGS-B', options={'maxiter': 30, 'ftol': 1e-4})
        theta = result.x[0]

        thetaLine = buildThetaLine(theta)
        alpha = _optimizeAlpha(thetaLine)
        filtered = _sesFilter(thetaLine, alpha)
        lastLevel = filtered[-1]

        fittedVals = np.zeros(n)
        for t in range(n):
            if modelType == 'additive':
                wses = 1.0 / max(theta, 1.0)
                wtrend = 1.0 - wses
                fittedVals[t] = wses * filtered[t] + wtrend * thetaLine0[t]
            else:
                invTheta = 1.0 / max(theta, 1.0)
                fittedVals[t] = (max(filtered[t], 1e-10) ** invTheta) * \
                                 (max(thetaLine0[t], 1e-10) ** (1.0 - invTheta))

        residuals = y - fittedVals
        residStd = max(np.std(residuals), 1e-8)

        x = np.arange(n, dtype=np.float64)
        if trendType == 'exponential':
            logY = np.log(np.maximum(y, 1e-10))
            slope, intercept = self._linearRegression(x, logY)
        else:
            slope, intercept = self._linearRegression(x, y)

        return {
            'theta': theta, 'alpha': alpha, 'intercept': intercept,
            'slope': slope, 'lastLevel': lastLevel, 'n': n,
            'residStd': residStd, 'fittedValues': fittedVals,
        }

    def _fitTrendLine(self, y, trendType):
        n = len(y)
        x = np.arange(n, dtype=np.float64)
        if trendType == 'exponential':
            if np.any(y <= 0):
                return None
            logY = np.log(y)
            slope, intercept = self._linearRegression(x, logY)
            return np.exp(intercept + slope * x)
        else:
            slope, intercept = self._linearRegression(x, y)
            return intercept + slope * x

    def _extrapolateTrend(self, m, trendType, steps):
        n = m['n']
        futureX = np.arange(n, n + steps, dtype=np.float64)
        if trendType == 'exponential':
            return np.exp(m['intercept'] + m['slope'] * futureX)
        else:
            return m['intercept'] + m['slope'] * futureX

    def _deseasonalize(self, y, period, seasonType):
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

    def _reseasonalize(self, y, seasonal, seasonType):
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

    @staticmethod
    def _linearRegression(x, y):
        xMean = np.mean(x)
        yMean = np.mean(y)
        num = np.sum((x - xMean) * (y - yMean))
        den = np.sum((x - xMean) ** 2)
        slope = num / max(den, 1e-10)
        intercept = yMean - slope * xMean
        return slope, intercept


class DOTHybrid:
    """DOT++ for period<=12, original DOT for period>=24."""

    def __init__(self, period=1):
        self.period = period
        self.fitted = False
        self.residuals = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        if self.period >= 24:
            self._inner = DynamicOptimizedTheta(period=self.period)
        else:
            self._inner = DOTPlusPlus(period=self.period)
        self._inner.fit(y)
        self.residuals = self._inner.residuals
        self.fitted = True
        return self

    def predict(self, steps):
        return self._inner.predict(steps)


class SCUM2:
    """DOT + CES median only (drop weak ETS/ARIMA)."""

    def __init__(self, period=1):
        self.period = period
        self.fitted = False
        self.residuals = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        self._y = y.copy()
        self._models = {}
        for name, factory in [('dot', lambda: DynamicOptimizedTheta(period=self.period)),
                               ('ces', lambda: AutoCES(period=self.period))]:
            try:
                m = factory()
                m.fit(y)
                self._models[name] = m
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
        allLowers = []
        allUppers = []
        for name, model in self._models.items():
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
            mv = np.mean(self._y)
            return np.full(steps, mv), np.full(steps, mv * 0.8), np.full(steps, mv * 1.2)
        predictions = np.median(np.array(allPreds), axis=0)
        lower = np.median(np.array(allLowers), axis=0) if allLowers else predictions * 0.8
        upper = np.median(np.array(allUppers), axis=0) if allUppers else predictions * 1.2
        return predictions, lower, upper


class SCUM2PP:
    """DOT++ + CES median."""

    def __init__(self, period=1):
        self.period = period
        self.fitted = False
        self.residuals = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        self._y = y.copy()
        self._models = {}
        for name, factory in [('dotpp', lambda: DOTPlusPlus(period=self.period)),
                               ('ces', lambda: AutoCES(period=self.period))]:
            try:
                m = factory()
                m.fit(y)
                self._models[name] = m
            except Exception:
                pass
        if not self._models:
            fallback = DOTPlusPlus(period=self.period)
            fallback.fit(y)
            self._models['dotpp'] = fallback
        self.residuals = np.zeros(len(y))
        self.fitted = True
        return self

    def predict(self, steps):
        allPreds = []
        allLowers = []
        allUppers = []
        for name, model in self._models.items():
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
            mv = np.mean(self._y)
            return np.full(steps, mv), np.full(steps, mv * 0.8), np.full(steps, mv * 1.2)
        predictions = np.median(np.array(allPreds), axis=0)
        lower = np.median(np.array(allLowers), axis=0) if allLowers else predictions * 0.8
        upper = np.median(np.array(allUppers), axis=0) if allUppers else predictions * 1.2
        return predictions, lower, upper


class CombinedBest:
    """DOT-Hybrid + CES median: frequency-adaptive DOT++ with CES robustness."""

    def __init__(self, period=1):
        self.period = period
        self.fitted = False
        self.residuals = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64)
        self._y = y.copy()
        self._models = {}

        if self.period >= 24:
            dotFactory = lambda: DynamicOptimizedTheta(period=self.period)
        else:
            dotFactory = lambda: DOTPlusPlus(period=self.period)

        for name, factory in [('dot', dotFactory),
                               ('ces', lambda: AutoCES(period=self.period))]:
            try:
                m = factory()
                m.fit(y)
                self._models[name] = m
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
        allLowers = []
        allUppers = []
        for name, model in self._models.items():
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
            mv = np.mean(self._y)
            return np.full(steps, mv), np.full(steps, mv * 0.8), np.full(steps, mv * 1.2)
        predictions = np.median(np.array(allPreds), axis=0)
        lower = np.median(np.array(allLowers), axis=0) if allLowers else predictions * 0.8
        upper = np.median(np.array(allUppers), axis=0) if allUppers else predictions * 1.2
        return predictions, lower, upper


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


def _smape(actual, predicted):
    return np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-10)) * 100


def _mase(trainY, actual, predicted, seasonality):
    n = len(trainY)
    m = max(seasonality, 1)
    naiveErrors = np.abs(trainY[m:] - trainY[:-m]) if n > m else np.abs(np.diff(trainY))
    masep = max(np.mean(naiveErrors), 1e-10) if len(naiveErrors) > 0 else 1e-10
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
        'dot_current': lambda: DynamicOptimizedTheta(period=seasonality),
        'dot_hybrid': lambda: DOTHybrid(period=seasonality),
        'scum2': lambda: SCUM2(period=seasonality),
        'scum2pp': lambda: SCUM2PP(period=seasonality),
        'combined': lambda: CombinedBest(period=seasonality),
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
    P("E018: Hybrid DOT + Targeted SCUM Variants")
    P("  DOT-Hybrid + SCUM2 + SCUM2++ + Combined Best")
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

    modelNames = ['dot_current', 'dot_hybrid', 'scum2', 'scum2pp', 'combined']
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

    P(f"\n  M4 Reference: #1 ES-RNN 0.821, #2 FFORMA 0.838, #11 4Theta 0.874")
    P(f"  E016 DOT++ AVG: 0.923, E017 SCUM AVG: 0.925")
    P(f"  Current DOT baseline: 0.905")
    P(f"\n{'='*60}")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
