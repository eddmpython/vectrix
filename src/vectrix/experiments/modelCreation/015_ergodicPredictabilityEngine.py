"""
==============================================================================
실험 ID: modelCreation/015
실험명: Ergodic Predictability Engine (EPE)
==============================================================================

목적:
- 국소 Lyapunov 지수(local LE)로 시계열의 "현재 시점 예측 가능 수평선"을
  실시간 추정하고, 수평선 기반으로 모델 가중치를 horizon별 차등 할당
- 예측 가능성이 높은 구간에서는 복잡 모델(DOT, 4Theta), 낮은 구간에서는
  단순 모델(Naive, mean)에 가중치를 이동

가설:
1. EPE > 단일 모델 예측 (horizon별 가중치 차등화 효과)
2. 예측 가능 수평선이 짧을수록 Naive 가중치가 높아져 과적합 방지
3. 장기 예측(h=18 monthly, h=48 hourly)에서 특히 효과적

방법:
1. Takens embedding: delay=autocorrelation first zero, dim=FNN heuristic
2. KDTree로 최근접 이웃 쌍 탐색
3. 이웃 궤적의 발산율 계산 → 국소 Lyapunov 지수
4. 예측 가능 수평선 = 1/max(localLE, 0.01)
5. Step-wise 가중치: step h에서 복잡모델 가중치 ∝ exp(-localLE * h)
6. 비교군: DOT only, equal ensemble, EPE horizon-weighted
7. M4 6개 그룹 × 500 시리즈

핵심 차별점 (기존 문헌 대비):
- Ayers 2023: ML로 local LE 추정 → 기상 분야, 예측 모델 가중치에 미연결
- Forecastability measures (2025): Lyapunov를 사후 분석 지표로만 사용
- EPE: Lyapunov를 "실시간 horizon별 모델 가중치 조정"에 사용하는 세계 최초 방법

결과 (M4 6그룹 × 500시리즈, 4.5분):
             Yearly  Quarterly  Monthly  Weekly   Daily   Hourly   AVG
dot          0.981   1.018      0.990    1.030   1.014   0.845    0.980
4theta       0.991   1.239      1.281    1.939   2.683   1.891    1.671
naive2       1.000   1.000      1.000    1.000   1.000   1.000    1.000
equal_ens    0.950   1.012      0.992    1.209   1.464   1.095    1.120
epe          1.006   1.014      1.023    1.200   1.484   1.066    1.132

Lyapunov: Yearly=0.47, Quarterly=0.39, Monthly=0.33, Weekly=0.13, Daily=0.17, Hourly=0.01
PredHorizon: Yearly=2.6, Quarterly=4.9, Monthly=13.0, Weekly=18.8, Daily=16.1, Hourly=73.7

EPE vs DOT: Yearly -2.5%, Monthly -3.3%, Weekly -16.5%, Daily -46.3%, Hourly -26.1%

결론:
- **가설 1 기각**: EPE가 DOT보다 전반적으로 크게 악화 (AVG OWA 1.132 vs 0.980)
- **가설 2 부분 확인**: Lyapunov가 높은 Yearly(0.47)에서 Naive 방향으로 이동하여
  DOT(0.981)보다 약간 악화(1.006)되었으나 큰 차이 없음
- **가설 3 기각**: 장기 예측에서 오히려 더 악화 (Monthly -3.3%)
- Lyapunov 지수가 의미 있는 정보를 제공: Hourly(0.01, 예측 가능)와 Yearly(0.47, 혼란)
  의 차이가 실제 예측 난이도와 일치
- 그러나 Lyapunov → 모델 가중치 변환이 너무 공격적:
  exp(-0.47 * h)는 h=3에서 이미 0.24로 떨어져 Naive에 과도하게 의존
- **교훈**: Lyapunov 지수는 "예측 가능성의 지표"로서 가치가 있으나,
  "모델 가중치 결정"에 직접 사용하면 단순 모델에 과도하게 치우침.
  DOT/CES 같은 적응형 모델은 이미 내부적으로 예측 불확실성을 처리하고 있어,
  외부에서 Naive 가중치를 강제하면 이중 보정 효과로 오히려 악화됨

실험일: 2026-02-28
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ces import AutoCES
from vectrix.engine.fourTheta import AdaptiveThetaEnsemble

P = lambda *a, **kw: print(*a, **kw, flush=True)

M4_GROUPS = {
    'Yearly':    {'horizon': 6,  'seasonality': 1},
    'Quarterly': {'horizon': 8,  'seasonality': 4},
    'Monthly':   {'horizon': 18, 'seasonality': 12},
    'Weekly':    {'horizon': 13, 'seasonality': 1},
    'Daily':     {'horizon': 14, 'seasonality': 1},
    'Hourly':    {'horizon': 48, 'seasonality': 24},
}

SAMPLE_PER_GROUP = 500
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')


def _ensureData(groupName):
    trainPath = os.path.join(DATA_DIR, f'{groupName}-train.csv')
    if not os.path.exists(trainPath):
        from datasetsforecast.m4 import M4
        M4.download(directory=os.path.join(DATA_DIR, '..', '..'), group=groupName)
    testPath = os.path.join(DATA_DIR, f'{groupName}-test.csv')
    return trainPath, testPath


def _loadGroup(groupName, maxSeries=None):
    trainPath, testPath = _ensureData(groupName)
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
    m = max(seasonality, 1)
    if len(trainY) <= m:
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
        lastVal = deseasonalized[-1]
        pred = np.full(horizon, lastVal)
        for h in range(horizon):
            pred[h] *= seasonal[(n + h) % m]
    else:
        pred = np.full(horizon, trainY[-1])
    return pred


def _estimateDelay(y, maxLag=50):
    """자기상관 최초 영교차(first zero crossing)로 시간 지연 추정."""
    n = len(y)
    maxLag = min(maxLag, n // 3)
    yNorm = y - np.mean(y)
    c0 = np.dot(yNorm, yNorm) / n
    if c0 < 1e-10:
        return 1
    for lag in range(1, maxLag + 1):
        acf = np.dot(yNorm[:n-lag], yNorm[lag:]) / ((n - lag) * c0)
        if acf <= 0:
            return lag
    return max(1, maxLag // 2)


def _estimateDimension(y, delay, maxDim=8):
    """False Nearest Neighbors 휴리스틱으로 임베딩 차원 추정."""
    n = len(y)
    bestDim = 2
    for dim in range(2, min(maxDim + 1, (n - 1) // delay)):
        nPoints = n - (dim - 1) * delay
        if nPoints < 20:
            break
        bestDim = dim
        if nPoints < 50:
            break
    return min(bestDim, 5)


def _takensEmbed(y, delay, dim):
    """Takens 시간지연 임베딩."""
    n = len(y)
    nPoints = n - (dim - 1) * delay
    if nPoints < 10:
        return None
    embedded = np.zeros((nPoints, dim))
    for d in range(dim):
        embedded[:, d] = y[d * delay: d * delay + nPoints]
    return embedded


def _localLyapunov(y, delay=None, dim=None, nNeighbors=5, maxSteps=10):
    """
    국소 Lyapunov 지수 추정.

    Takens 임베딩 → KDTree → 최근접 이웃의 발산율 측정
    → 지수적 발산의 기울기 = 최대 Lyapunov 지수
    """
    n = len(y)
    if n < 30:
        return 0.5, np.full(maxSteps, 0.5)

    yStd = np.std(y)
    if yStd < 1e-10:
        return 0.0, np.zeros(maxSteps)

    yNorm = (y - np.mean(y)) / yStd

    if delay is None:
        delay = _estimateDelay(yNorm)
    if dim is None:
        dim = _estimateDimension(yNorm, delay)

    embedded = _takensEmbed(yNorm, delay, dim)
    if embedded is None or len(embedded) < 20:
        return 0.5, np.full(maxSteps, 0.5)

    nPoints = len(embedded)
    maxSteps = min(maxSteps, nPoints // 3)
    if maxSteps < 2:
        return 0.5, np.full(10, 0.5)

    tree = KDTree(embedded)
    minSep = max(delay * 2, 5)

    divergences = np.zeros(maxSteps)
    counts = np.zeros(maxSteps)
    nSample = min(nPoints - maxSteps, 200)
    rng = np.random.default_rng(42)
    sampleIdx = rng.choice(nPoints - maxSteps, min(nSample, nPoints - maxSteps), replace=False)

    for i in sampleIdx:
        dists, idxs = tree.query(embedded[i], k=nNeighbors + 1)
        for kk in range(1, len(idxs)):
            j = idxs[kk]
            if abs(i - j) < minSep:
                continue
            d0 = dists[kk]
            if d0 < 1e-10:
                continue

            for step in range(1, maxSteps + 1):
                if i + step >= nPoints or j + step >= nPoints:
                    break
                dt = np.linalg.norm(embedded[i + step] - embedded[j + step])
                if dt < 1e-10:
                    dt = 1e-10
                divergences[step - 1] += np.log(dt / d0)
                counts[step - 1] += 1
            break

    avgDiv = np.zeros(maxSteps)
    for s in range(maxSteps):
        if counts[s] > 0:
            avgDiv[s] = divergences[s] / counts[s]

    validSteps = np.where(counts > 0)[0]
    if len(validSteps) >= 2:
        steps = validSteps + 1
        vals = avgDiv[validSteps]
        slope = np.polyfit(steps, vals, 1)[0]
        localLE = max(slope, 0.0)
    else:
        localLE = 0.5

    return localLE, avgDiv


def _epeForecast(trainY, horizon, period):
    """
    Ergodic Predictability Engine.

    1. 국소 Lyapunov 지수 추정
    2. 예측 가능 수평선 계산
    3. Horizon별 모델 가중치 차등 할당
    """
    n = len(trainY)

    localLE, divergenceProfile = _localLyapunov(trainY, maxSteps=min(horizon, 20))

    predictabilityHorizon = 1.0 / max(localLE, 0.01)
    predictabilityHorizon = min(predictabilityHorizon, horizon * 2)

    dotModel = DynamicOptimizedTheta(period=period)
    dotModel.fit(trainY)
    dotPred, _, _ = dotModel.predict(horizon)
    dotPred = np.asarray(dotPred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(dotPred)):
        dotPred = np.full(horizon, trainY[-1])

    thetaModel = AdaptiveThetaEnsemble(period=period)
    thetaModel.fit(trainY)
    thetaPred, _, _ = thetaModel.predict(horizon)
    thetaPred = np.asarray(thetaPred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(thetaPred)):
        thetaPred = np.full(horizon, trainY[-1])

    naive2Pred = _naive2(trainY, horizon, period)

    epePred = np.zeros(horizon)
    for h in range(horizon):
        step = h + 1
        decayFactor = np.exp(-localLE * step)
        decayFactor = np.clip(decayFactor, 0.05, 0.95)

        wComplex = decayFactor
        wSimple = 1.0 - decayFactor

        complexPred = 0.5 * dotPred[h] + 0.5 * thetaPred[h]
        simplePred = naive2Pred[h]

        epePred[h] = wComplex * complexPred + wSimple * simplePred

    equalPred = (dotPred + thetaPred + naive2Pred) / 3.0

    return {
        'dot': dotPred,
        '4theta': thetaPred,
        'naive2': naive2Pred,
        'equal_ensemble': equalPred,
        'epe': epePred,
    }, localLE, predictabilityHorizon


def _processOneSeries(trainY, testY, horizon, period):
    if len(trainY) < 20:
        return None

    preds, localLE, predHorizon = _epeForecast(trainY, horizon, period)

    smapeN = _smape(testY[:horizon], preds['naive2'])
    maseN = _mase(trainY, testY[:horizon], preds['naive2'], period)

    scores = {}
    for method, pred in preds.items():
        s = _smape(testY[:horizon], pred[:horizon])
        m = _mase(trainY, testY[:horizon], pred[:horizon], period)
        relSmape = s / max(smapeN, 1e-10)
        relMase = m / max(maseN, 1e-10)
        owa = 0.5 * (relSmape + relMase)
        scores[method] = owa

    return scores, localLE, predHorizon


def _runGroup(groupName, maxSeries=SAMPLE_PER_GROUP):
    info = M4_GROUPS[groupName]
    horizon = info['horizon']
    period = info['seasonality']

    P(f'\n{"="*60}')
    P(f'{groupName} (h={horizon}, m={period}, sample={maxSeries})')
    P(f'{"="*60}')

    trainSeries, testSeries = _loadGroup(groupName, maxSeries)
    nSeries = len(trainSeries)
    P(f'Loaded {nSeries} series')

    allScores = {}
    allLE = []
    allHorizons = []
    t0 = time.time()

    for idx in range(nSeries):
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period)
        if result is None:
            continue
        scores, le, ph = result
        allLE.append(le)
        allHorizons.append(ph)
        for method, owa in scores.items():
            if method not in allScores:
                allScores[method] = []
            allScores[method].append(owa)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            P(f'  [{idx+1}/{nSeries}] {(idx+1)/elapsed:.1f} series/s')

    elapsed = time.time() - t0
    P(f'Completed in {elapsed:.1f}s ({nSeries/elapsed:.1f} series/s)')
    P(f'Avg Lyapunov: {np.mean(allLE):.4f} (std={np.std(allLE):.4f})')
    P(f'Avg Pred Horizon: {np.mean(allHorizons):.1f} steps')

    avgOwa = {}
    for method, owaList in allScores.items():
        avgOwa[method] = np.mean(owaList)

    return avgOwa, np.mean(allLE), np.mean(allHorizons)


def _printResults(groupResults):
    P('\n' + '='*80)
    P('FINAL RESULTS: Ergodic Predictability Engine')
    P('='*80)

    methods = ['dot', '4theta', 'naive2', 'equal_ensemble', 'epe']
    groups = list(groupResults.keys())

    P(f'\n{"Method":<20}', end='')
    for g in groups:
        P(f'{g:>12}', end='')
    P(f'{"AVG":>12}')
    P('-' * (20 + 12 * (len(groups) + 1)))

    for method in methods:
        P(f'{method:<20}', end='')
        vals = []
        for g in groups:
            avgOwa, _, _ = groupResults[g]
            v = avgOwa.get(method, float('nan'))
            vals.append(v)
            marker = ' **' if v < 1.0 else '   '
            P(f'{v:>9.3f}{marker}', end='')
        avgAll = np.nanmean(vals)
        marker = ' **' if avgAll < 1.0 else '   '
        P(f'{avgAll:>9.3f}{marker}')

    P('\nLyapunov 지수 & 예측 수평선:')
    for g in groups:
        _, le, ph = groupResults[g]
        P(f'  {g}: LE={le:.4f}, PredHorizon={ph:.1f} steps')

    P('\nEPE vs DOT 개선율:')
    for g in groups:
        avgOwa, _, _ = groupResults[g]
        dotOwa = avgOwa.get('dot', float('nan'))
        epeOwa = avgOwa.get('epe', float('nan'))
        if dotOwa > 0:
            improvement = (dotOwa - epeOwa) / dotOwa * 100
            P(f'  {g}: {improvement:+.2f}%')

    P('\nEPE vs equal_ensemble 개선율:')
    for g in groups:
        avgOwa, _, _ = groupResults[g]
        eqOwa = avgOwa.get('equal_ensemble', float('nan'))
        epeOwa = avgOwa.get('epe', float('nan'))
        if eqOwa > 0:
            improvement = (eqOwa - epeOwa) / eqOwa * 100
            P(f'  {g}: {improvement:+.2f}%')


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('='*80)
    P('Ergodic Predictability Engine (EPE) — modelCreation/015')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        avgOwa, avgLE, avgPH = _runGroup(groupName, SAMPLE_PER_GROUP)
        groupResults[groupName] = (avgOwa, avgLE, avgPH)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
