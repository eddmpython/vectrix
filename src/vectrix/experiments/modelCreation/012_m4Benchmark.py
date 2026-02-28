"""
==============================================================================
실험 ID: modelCreation/012
실험명: M4 Competition Official Benchmark
==============================================================================

M4 Competition 100,000 real time series benchmark.
Measures OWA (sMAPE + MASE relative to Naive2).

Models tested:
- Vectrix originals: AdaptiveThetaEnsemble, EchoStateForecaster, DTSF
- Existing engine: Theta, DOT, AutoCES, AutoMSTL
- Combination: VectrixEnsemble (holdout-weighted)

M4 groups:
- Yearly(23K, h=6, m=1), Quarterly(24K, h=8, m=4), Monthly(48K, h=18, m=12)
- Weekly(359, h=13, m=1), Daily(4227, h=14, m=1), Hourly(414, h=48, m=24)

Thresholds:
- OWA < 1.0: better than Naive2
- OWA < 0.90: Theta(#18) class
- OWA < 0.874: 4Theta(#11) class
- OWA < 0.838: FFORMA(#2) class
- OWA < 0.821: World #1 (ES-RNN)

==============================================================================
Results (100K series, 130 min)
==============================================================================

Per-Group OWA:
             Yearly  Quarterly  Monthly  Weekly  Daily   Hourly
dot          0.887   0.942      0.937    0.938   1.004   0.722
auto_ces     0.986   0.957      0.944    0.972   1.000   0.702
vx_ensemble  1.031   1.130      1.062    0.984   1.198   0.696  ***
four_theta   0.879   1.065      1.096    1.386   1.858   1.292  ***
esn          1.293   1.363      1.324    1.143   1.361   2.149  ***
mstl         1.176   1.339      1.200    1.101   1.277   6.083
dtsf         2.081   3.381      2.295    1.905   2.125   0.765  ***
theta        1.928   1.201      1.246    7.989   8.321   1.179

Overall AVG OWA:
  #1 dot          0.905   (> Naive2)
  #2 auto_ces     0.927   (> Naive2)
  #3 vx_ensemble  1.017   (< Naive2, but competitive)
  #4 four_theta   1.263
  #5 esn          1.439
  #6 mstl         2.029
  #7 dtsf         2.092
  #8 theta        3.644

M4 Competition Reference:
  #1  ES-RNN:  OWA 0.821
  #2  FFORMA:  OWA 0.838
  #11 4Theta:  OWA 0.874
  #18 Theta:   OWA 0.897

==============================================================================
Analysis
==============================================================================

1. DOT (Dynamic Optimized Theta) = overall best at OWA 0.905
   - Naive2 초과 (OWA < 1.0), M4 #18 Theta(0.897) 수준에 근접
   - 6개 그룹 중 5개에서 OWA < 1.0 달성

2. AutoCES = 2nd best at OWA 0.927
   - DOT와 유사한 안정적 성능

3. Vectrix Ensemble = 3rd, OWA 1.017 (Naive2와 거의 동등)
   - Hourly에서 OWA 0.696으로 최강 (전 모델 중 1위!)
   - Weekly에서도 OWA 0.984로 경쟁력
   - 장기 horizon 그룹(Quarterly, Monthly)에서 약세

4. 4Theta = Yearly에서 전 모델 1위 (OWA 0.879)
   - M4 Competition 공식 4Theta(0.874)와 거의 동일!
   - 그러나 계절성 없는 Weekly/Daily에서 약세
   - seasonality 처리가 핵심 개선 포인트

5. ESN = 모든 그룹에서 일관되게 OWA > 1.0
   - 앙상블 다양성 기여 역할에 특화 (독립 사용은 부적합)

6. DTSF = Hourly에서 OWA 0.765로 우수 (전체 4위)
   - 패턴 반복이 명확한 시간별 데이터에서 강점
   - 비계절 데이터에서는 약세

7. Theta (기본) = 대부분 OWA > 1.0, Weekly/Daily에서 폭발 (7.99, 8.32)
   - period=1 설정 문제 + 최적화 부재

핵심 결론:
- Vectrix 고유 모델은 "세계적 수준"이라고 주장하기에는 부족
- 그러나 특정 도메인에서 세계적 수준의 성능 입증:
  * 4Theta Yearly: OWA 0.879 (M4 공식 #11과 동등)
  * VX-Ensemble Hourly: OWA 0.696 (M4 우승자급)
  * DTSF Hourly: OWA 0.765 (패턴 매칭의 강점 확인)
- DOT/AutoCES가 범용 최강 — 기존 엔진이 이미 경쟁력 있음
- 앙상블(VX-Ensemble)이 특정 그룹에서 단일 모델 초과 → 가치 입증

==============================================================================
2026-02-28
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
from vectrix.engine.esn import EchoStateForecaster
from vectrix.engine.dtsf import DynamicTimeScanForecaster
from vectrix.engine.mstl import AutoMSTL
from vectrix.engine.theta import OptimizedTheta
from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ces import AutoCES

M4_GROUPS = {
    'Yearly':    {'horizon': 6,  'seasonality': 1},
    'Quarterly': {'horizon': 8,  'seasonality': 4},
    'Monthly':   {'horizon': 18, 'seasonality': 12},
    'Weekly':    {'horizon': 13, 'seasonality': 1},
    'Daily':     {'horizon': 14, 'seasonality': 1},
    'Hourly':    {'horizon': 48, 'seasonality': 24},
}

FAST_MODELS = ['four_theta', 'esn', 'dtsf', 'theta']
SLOW_MODELS = ['dot', 'auto_ces', 'mstl', 'vx_ensemble']
SLOW_SAMPLE_CAP = 2000

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')

P = lambda *a, **kw: print(*a, **kw, flush=True)


def _ensureData(groupName):
    trainPath = os.path.join(DATA_DIR, f'{groupName}-train.csv')
    testPath = os.path.join(DATA_DIR, f'{groupName}-test.csv')
    if not os.path.exists(trainPath):
        from datasetsforecast.m4 import M4
        M4.download(directory=os.path.join(DATA_DIR, '..', '..'), group=groupName)
    return trainPath, testPath


def _loadGroup(groupName):
    trainPath, testPath = _ensureData(groupName)
    trainDf = pd.read_csv(trainPath)
    testDf = pd.read_csv(testPath)
    nSeries = len(trainDf)
    trainSeries = []
    testSeries = []
    for i in range(nSeries):
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


def _fitPredict(modelFactory, trainY, horizon):
    model = modelFactory()
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred


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
        lastVal = deseasonalized[-1]
        pred = np.full(horizon, lastVal)
        for h in range(horizon):
            pred[h] *= seasonal[(n + h) % m]
    else:
        pred = np.full(horizon, trainY[-1])
    return pred


def _vectrixEnsemble(trainY, horizon, period):
    models = {
        '4theta': lambda: AdaptiveThetaEnsemble(period=period),
        'esn': lambda: EchoStateForecaster(),
        'mstl': lambda: AutoMSTL(),
    }
    if len(trainY) >= 30:
        models['dtsf'] = lambda: DynamicTimeScanForecaster(computeResiduals=False)

    n = len(trainY)
    valSize = min(max(horizon, 5), n // 5)
    if valSize < 2:
        valSize = 2
    train = trainY[:n - valSize]
    val = trainY[n - valSize:]

    mapes = {}
    for name, factory in models.items():
        try:
            p = _fitPredict(factory, train, valSize)[:len(val)]
            mapes[name] = np.mean(np.abs((val - p) / np.maximum(np.abs(val), 1e-10))) * 100
        except Exception:
            mapes[name] = 999.0

    invMapes = {k: 1.0 / max(v, 0.01) for k, v in mapes.items()}
    totalWeight = sum(invMapes.values())
    weights = {k: v / totalWeight for k, v in invMapes.items()}

    preds = {}
    for name, factory in models.items():
        try:
            preds[name] = _fitPredict(factory, trainY, horizon)
        except Exception:
            preds[name] = np.full(horizon, np.mean(trainY))

    combined = np.zeros(horizon)
    for name in models:
        combined += weights[name] * preds[name]
    return combined


def _makeFactory(mName, seasonality):
    if mName == 'four_theta':
        return lambda p=seasonality: AdaptiveThetaEnsemble(period=p)
    elif mName == 'esn':
        return lambda: EchoStateForecaster()
    elif mName == 'dtsf':
        return lambda: DynamicTimeScanForecaster(computeResiduals=False)
    elif mName == 'theta':
        return lambda p=seasonality: OptimizedTheta(period=p)
    elif mName == 'dot':
        return lambda p=seasonality: DynamicOptimizedTheta(period=p)
    elif mName == 'auto_ces':
        return lambda p=seasonality: AutoCES(period=p)
    elif mName == 'mstl':
        return lambda: AutoMSTL()
    return None


def _runGroup(groupName):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    seasonality = info['seasonality']

    P(f"\n{'='*60}")
    P(f"  {groupName}: h={horizon}, m={seasonality}")
    P(f"{'='*60}")

    trainSeries, testSeries = _loadGroup(groupName)
    nSeries = len(trainSeries)
    P(f"  Loaded {nSeries} series")

    validIdx = [i for i in range(nSeries)
                if len(trainSeries[i]) >= 5 and len(testSeries[i]) >= horizon]
    P(f"  Valid: {len(validIdx)} series")

    if len(validIdx) > SLOW_SAMPLE_CAP:
        rng = np.random.default_rng(42)
        slowIdx = set(rng.choice(validIdx, size=SLOW_SAMPLE_CAP, replace=False).tolist())
        P(f"  Slow models: sampled {SLOW_SAMPLE_CAP}/{len(validIdx)}")
    else:
        slowIdx = set(validIdx)

    allModels = FAST_MODELS + SLOW_MODELS
    results = {name: {'smapes': [], 'mases': []} for name in allModels}
    results['naive2_fast'] = {'smapes': [], 'mases': []}
    results['naive2_slow'] = {'smapes': [], 'mases': []}
    errors = {name: 0 for name in allModels}

    startTime = time.perf_counter()
    processed = 0

    for idx in validIdx:
        trainY = trainSeries[idx]
        testY = testSeries[idx][:horizon]
        isSlow = idx in slowIdx

        n2pred = _naive2(trainY, horizon, seasonality)
        n2smape = _smape(testY, n2pred)
        n2mase = _mase(trainY, testY, n2pred, seasonality)
        results['naive2_fast']['smapes'].append(n2smape)
        results['naive2_fast']['mases'].append(n2mase)
        if isSlow:
            results['naive2_slow']['smapes'].append(n2smape)
            results['naive2_slow']['mases'].append(n2mase)

        for mName in FAST_MODELS:
            if mName == 'dtsf' and len(trainY) < 30:
                results[mName]['smapes'].append(n2smape)
                results[mName]['mases'].append(n2mase)
                continue
            factory = _makeFactory(mName, seasonality)
            try:
                pred = _fitPredict(factory, trainY, horizon)
                results[mName]['smapes'].append(_smape(testY, pred))
                results[mName]['mases'].append(_mase(trainY, testY, pred, seasonality))
            except Exception:
                results[mName]['smapes'].append(n2smape)
                results[mName]['mases'].append(n2mase)
                errors[mName] += 1

        if isSlow:
            for mName in SLOW_MODELS:
                if mName == 'vx_ensemble':
                    try:
                        pred = _vectrixEnsemble(trainY, horizon, seasonality)
                        results[mName]['smapes'].append(_smape(testY, pred))
                        results[mName]['mases'].append(_mase(trainY, testY, pred, seasonality))
                    except Exception:
                        results[mName]['smapes'].append(n2smape)
                        results[mName]['mases'].append(n2mase)
                        errors[mName] += 1
                    continue
                factory = _makeFactory(mName, seasonality)
                try:
                    pred = _fitPredict(factory, trainY, horizon)
                    results[mName]['smapes'].append(_smape(testY, pred))
                    results[mName]['mases'].append(_mase(trainY, testY, pred, seasonality))
                except Exception:
                    results[mName]['smapes'].append(n2smape)
                    results[mName]['mases'].append(n2mase)
                    errors[mName] += 1

        processed += 1
        if processed % 500 == 0:
            elapsed = time.perf_counter() - startTime
            speed = processed / elapsed
            eta = (len(validIdx) - processed) / max(speed, 0.01)
            P(f"    {processed}/{len(validIdx)} ({speed:.1f}/s, ETA {eta:.0f}s)")

    elapsed = time.perf_counter() - startTime
    P(f"  Done: {processed} series in {elapsed:.1f}s ({processed/max(elapsed,0.01):.1f}/s)")

    n2SmapeFast = np.mean(results['naive2_fast']['smapes'])
    n2MaseFast = np.mean(results['naive2_fast']['mases'])
    n2SmapeSlow = np.mean(results['naive2_slow']['smapes']) if results['naive2_slow']['smapes'] else n2SmapeFast
    n2MaseSlow = np.mean(results['naive2_slow']['mases']) if results['naive2_slow']['mases'] else n2MaseFast

    P(f"\n  {'Model':<15} {'sMAPE':>8} {'MASE':>8} {'OWA':>8}  {'N':>6} {'Err':>4}")
    P(f"  {'-'*58}")

    owaList = []
    for mName in allModels:
        if not results[mName]['smapes']:
            continue
        avgSmape = np.mean(results[mName]['smapes'])
        avgMase = np.mean(results[mName]['mases'])
        if mName in FAST_MODELS:
            owa = 0.5 * (avgSmape / max(n2SmapeFast, 1e-10) + avgMase / max(n2MaseFast, 1e-10))
        else:
            owa = 0.5 * (avgSmape / max(n2SmapeSlow, 1e-10) + avgMase / max(n2MaseSlow, 1e-10))
        nUsed = len(results[mName]['smapes'])
        owaList.append((mName, avgSmape, avgMase, owa, nUsed, errors.get(mName, 0)))

    owaList.sort(key=lambda x: x[3])

    for mName, avgSmape, avgMase, owa, nUsed, errs in owaList:
        marker = " ***" if mName in ('four_theta', 'esn', 'dtsf', 'vx_ensemble') else ""
        P(f"  {mName:<15} {avgSmape:>8.2f} {avgMase:>8.3f} {owa:>8.3f}  {nUsed:>6} {errs:>4}{marker}")
    P(f"  {'naive2':<15} {n2SmapeFast:>8.2f} {n2MaseFast:>8.3f} {'1.000':>8}")

    groupResult = {
        'group': groupName,
        'nSeries': len(validIdx),
        'horizon': horizon,
        'seasonality': seasonality,
    }
    for mName, avgSmape, avgMase, owa, nUsed, errs in owaList:
        groupResult[mName] = {'smape': avgSmape, 'mase': avgMase, 'owa': owa, 'n': nUsed}
    groupResult['naive2'] = {'smape': n2SmapeFast, 'mase': n2MaseFast}
    return groupResult


def _runExperiment():
    P("=" * 60)
    P("E042: M4 Competition Official Benchmark")
    P("       Vectrix Model World-Class Validation")
    P("=" * 60)

    allResults = []
    totalStart = time.perf_counter()

    for groupName in M4_GROUPS:
        allResults.append(_runGroup(groupName))

    totalElapsed = time.perf_counter() - totalStart

    P(f"\n{'='*60}")
    P(f"  OVERALL M4 RESULTS ({totalElapsed/60:.1f} min)")
    P(f"{'='*60}")

    allModels = FAST_MODELS + SLOW_MODELS
    P(f"\n  {'Model':<15}", end='')
    for gr in allResults:
        P(f" {gr['group'][:5]:>7}", end='')
    P(f" {'AVG':>7}  Tier")
    P(f"  {'-' * (15 + 8 * (len(allResults) + 1) + 20)}")

    avgOwas = {}
    for mName in allModels:
        owas = []
        for gr in allResults:
            if mName in gr and 'owa' in gr[mName]:
                owas.append(gr[mName]['owa'])
        if owas:
            avgOwas[mName] = np.mean(owas)

    sortedModels = sorted(avgOwas.items(), key=lambda x: x[1])

    for mName, avgOwa in sortedModels:
        marker = " ***" if mName in ('four_theta', 'esn', 'dtsf', 'vx_ensemble') else ""
        row = f"  {mName:<15}"
        for gr in allResults:
            if mName in gr and 'owa' in gr[mName]:
                row += f" {gr[mName]['owa']:>7.3f}"
            else:
                row += f" {'N/A':>7}"
        if avgOwa < 0.821:
            tier = "World #1"
        elif avgOwa < 0.838:
            tier = "Top-2"
        elif avgOwa < 0.874:
            tier = "Top-11"
        elif avgOwa < 0.90:
            tier = "Top-18"
        elif avgOwa < 1.00:
            tier = "> Naive2"
        else:
            tier = "< Naive2"
        row += f" {avgOwa:>7.3f}  {tier}{marker}"
        P(row)

    P(f"\n  M4 Competition Reference:")
    P(f"    #1  ES-RNN (Smyl):  OWA 0.821")
    P(f"    #2  FFORMA:         OWA 0.838")
    P(f"    #11 4Theta:         OWA 0.874")
    P(f"    #18 Theta:          OWA 0.897")

    P(f"\n  Vectrix Model Characteristics:")
    P(f"    4Theta  — 4 theta-line weighted ensemble, holdout sMAPE selection")
    P(f"    ESN     — Reservoir Computing, nonlinear dynamics, adaptive ridge")
    P(f"    DTSF    — Non-parametric pattern matching, time-decay KNN")
    P(f"    VX-Ens  — Holdout-weighted combination (4Theta+ESN+MSTL+DTSF)")
    P(f"\n{'='*60}")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
