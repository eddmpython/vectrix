"""
==============================================================================
실험 ID: modelCreation/021
실험명: Topological Forecaster (TF)
==============================================================================

목적:
- 시계열을 숫자가 아닌 "형태(shape)"로 보는 완전히 새로운 접근
- 시간지연 임베딩 → 고차원 궤적 → 위상적 특성(Betti numbers) 추출
- 유사한 위상 구조를 가진 과거 구간 탐색 → 그 직후 값으로 예측
- 값의 크기가 아닌 "궤적의 기하학적 구조"로 미래를 예측

가설:
1. TF가 Naive2보다 우수 (OWA < 1.0) — 위상 구조가 예측 정보 포함
2. TF와 DOT 잔차 상관 < 0.3 — 근본적으로 다른 판단 기준
3. 비선형/주기적 데이터(Monthly, Quarterly)에서 특히 강함

방법:
1. Takens 시간지연 임베딩: Y(t) → [Y(t), Y(t-τ), Y(t-2τ), ..., Y(t-(d-1)τ)]
   τ = first minimum of autocorrelation, d = max(2, min(5, n//50))
2. 간소화된 위상 특성: Vietoris-Rips 대신 궤적의 이웃 그래프 기반
   - 궤적 점들 간 거리 행렬
   - 다양한 반경 ε에서 연결 성분 수의 변화 = "persistence-like" 시그니처
3. 최근 윈도우의 시그니처와 과거 윈도우들의 시그니처 비교 (L2 거리)
4. 가장 유사한 K개 윈도우의 직후 값들의 가중 평균 = 예측
5. 비교군: DOT, AutoCES, Naive2
6. M4 6개 그룹 × 300 시리즈

핵심 차별점 (기존 문헌 대비):
- Seversky 2016: TDA + SVM for time series classification
- Perea & Harer 2015: Sliding window persistent homology — 패턴 감지에만 사용
- 본 실험: 위상 시그니처를 "예측 생성"에 직접 사용하는 최초 시도
  기존 TDA 시계열 연구는 분류/이상치에만 적용

결과 (M4 6그룹 × 300시리즈, 2.9분):
             Yearly  Quarterly  Monthly  Weekly   Daily   Hourly   AVG
tf           1.217   3.695      2.024    19.750  19.730   5.130    8.591
dot          1.014   0.990      0.983    1.011   0.998    0.858    0.976
ces          0.987   0.987      1.018    0.987   1.005    0.844    0.971
naive2       1.000   1.000      1.000    1.000   1.000    1.000    1.000

TF-DOT 잔차 상관: Y=0.66, Q=0.74, M=0.74, W=0.82, D=0.82, H=0.38, AVG=0.69

결론:
- **가설 1 완전 기각**: TF가 Naive2보다 모든 그룹에서 대폭 악화 (AVG OWA 8.591)
  Weekly(19.75x), Daily(19.73x)에서 치명적으로 나쁨
- **가설 2 기각**: 잔차 상관 AVG=0.69로 기존 모델과 유사하게 틀림.
  Hourly(0.38)만 약간 낮음
- **가설 3 완전 기각**: Monthly(2.02x), Quarterly(3.70x)에서 오히려 크게 악화
- **근본 문제**: 위상 시그니처(ε에 따른 연결 성분 수)는 너무 조잡한 요약.
  시계열의 미세한 값 변화를 포착하지 못하고, 대략적 형태만 비교.
  "비슷한 형태" ≠ "비슷한 값이 뒤따름". 위상적 유사성은
  시계열의 질적 구조를 반영하지만, 양적 예측에는 정보가 부족.
- **Weekly/Daily 참사**: 짧은 윈도우에서 위상 시그니처가 거의 동일해져
  가중 평균이 전체 평균으로 수렴 → 무의미한 예측
- **핵심 교훈**: TDA의 위상 특성은 분류/이상치 감지에 적합하지만,
  점 예측(regression)에는 정보 손실이 너무 큼.
  "같은 형태" → "같은 미래"라는 가정이 성립하지 않음

실험일: 2026-03-03
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ces import AutoCES

P = lambda *a, **kw: print(*a, **kw, flush=True)

M4_GROUPS = {
    'Yearly':    {'horizon': 6,  'seasonality': 1},
    'Quarterly': {'horizon': 8,  'seasonality': 4},
    'Monthly':   {'horizon': 18, 'seasonality': 12},
    'Weekly':    {'horizon': 13, 'seasonality': 1},
    'Daily':     {'horizon': 14, 'seasonality': 1},
    'Hourly':    {'horizon': 48, 'seasonality': 24},
}

SAMPLE_PER_GROUP = 300

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
        lastVal = deseasonalized[-1]
        pred = np.full(horizon, lastVal)
        for h in range(horizon):
            pred[h] *= seasonal[(n + h) % m]
    else:
        pred = np.full(horizon, trainY[-1])
    return pred


def _estimateDelay(y, maxLag=50):
    n = len(y)
    maxLag = min(maxLag, n // 3)
    if maxLag < 2:
        return 1
    yMean = np.mean(y)
    yStd = np.std(y)
    if yStd < 1e-10:
        return 1
    yNorm = (y - yMean) / yStd
    for lag in range(1, maxLag):
        acf = np.mean(yNorm[lag:] * yNorm[:-lag])
        if acf < 0:
            return lag
    return maxLag // 2


def _embed(y, dim, delay):
    n = len(y)
    nVecs = n - (dim - 1) * delay
    if nVecs < 5:
        return None
    embedded = np.zeros((nVecs, dim))
    for d in range(dim):
        embedded[:, d] = y[d * delay: d * delay + nVecs]
    return embedded


def _persistenceSignature(embedded, nBins=10):
    n = len(embedded)
    if n < 3:
        return np.zeros(nBins)

    dists = pdist(embedded)
    maxDist = np.max(dists)
    if maxDist < 1e-10:
        return np.zeros(nBins)

    distMatrix = squareform(dists)
    epsilons = np.linspace(0, maxDist * 0.8, nBins + 1)[1:]

    signature = np.zeros(nBins)
    for i, eps in enumerate(epsilons):
        adj = distMatrix <= eps
        visited = np.zeros(n, dtype=bool)
        nComponents = 0
        for start in range(n):
            if visited[start]:
                continue
            nComponents += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                neighbors = np.where(adj[node] & ~visited)[0]
                stack.extend(neighbors.tolist())
        signature[i] = nComponents / n

    return signature


def _topologicalPredict(trainY, horizon, windowSize=0, K=5, nBins=10):
    n = len(trainY)
    delay = _estimateDelay(trainY)
    dim = max(2, min(5, n // 50))
    if windowSize <= 0:
        windowSize = min(max(dim * delay * 2, 20), n // 4)

    yStd = np.std(trainY)
    if yStd < 1e-10:
        return np.full(horizon, trainY[-1])
    yNorm = (trainY - np.mean(trainY)) / yStd

    signatures = []
    sigStartIndices = []
    step = max(1, windowSize // 4)

    for start in range(0, n - windowSize - 1, step):
        windowData = yNorm[start:start + windowSize]
        emb = _embed(windowData, dim, delay)
        if emb is None:
            continue
        sig = _persistenceSignature(emb, nBins)
        signatures.append(sig)
        sigStartIndices.append(start)

    if len(signatures) < K + 1:
        return np.full(horizon, trainY[-1])

    signatures = np.array(signatures)

    queryWindow = yNorm[n - windowSize:]
    queryEmb = _embed(queryWindow, dim, delay)
    if queryEmb is None:
        return np.full(horizon, trainY[-1])
    querySig = _persistenceSignature(queryEmb, nBins)

    dists = np.linalg.norm(signatures[:-1] - querySig, axis=1)

    nValid = min(K, len(dists))
    if nValid < 1:
        return np.full(horizon, trainY[-1])

    if nValid >= len(dists):
        topK = np.arange(len(dists))
    else:
        topK = np.argpartition(dists, nValid)[:nValid]

    predictions = np.zeros(horizon)
    weights = np.zeros(nValid)

    for i, idx in enumerate(topK):
        origIdx = sigStartIndices[idx]
        afterIdx = origIdx + windowSize
        d = dists[idx]
        weights[i] = 1.0 / (d + 1e-10)

    wSum = weights.sum()
    if wSum < 1e-10:
        weights = np.ones(nValid) / nValid
    else:
        weights = weights / wSum

    for h in range(horizon):
        val = 0.0
        wUsed = 0.0
        for i, idx in enumerate(topK):
            origIdx = sigStartIndices[idx]
            futureIdx = origIdx + windowSize + h
            if futureIdx < n:
                val += weights[i] * trainY[futureIdx]
                wUsed += weights[i]
        if wUsed > 0:
            predictions[h] = val / wUsed
        else:
            predictions[h] = trainY[-1]

    return predictions


def _safePredict(modelClass, trainY, horizon, period):
    model = modelClass(period=period)
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred


def _processOneSeries(trainY, testY, horizon, period):
    n = len(trainY)
    if n < 30:
        return None

    results = {}

    tfPred = _topologicalPredict(trainY, horizon)
    results['tf'] = tfPred

    dotPred = _safePredict(DynamicOptimizedTheta, trainY, horizon, period)
    results['dot'] = dotPred

    cesPred = _safePredict(AutoCES, trainY, horizon, period)
    results['ces'] = cesPred

    naive2Pred = _naive2(trainY, horizon, period)
    results['naive2'] = naive2Pred

    smapeN = _smape(testY[:horizon], naive2Pred)
    maseN = _mase(trainY, testY[:horizon], naive2Pred, period)

    scores = {}
    for method, pred in results.items():
        s = _smape(testY[:horizon], pred[:horizon])
        m = _mase(trainY, testY[:horizon], pred[:horizon], period)
        relSmape = s / max(smapeN, 1e-10)
        relMase = m / max(maseN, 1e-10)
        owa = 0.5 * (relSmape + relMase)
        scores[method] = owa

    dotResid = testY[:horizon] - dotPred[:horizon]
    tfResid = testY[:horizon] - tfPred[:horizon]
    if len(dotResid) > 2:
        corr = np.corrcoef(dotResid, tfResid)[0, 1]
        if not np.isfinite(corr):
            corr = 0.0
    else:
        corr = 0.0

    return scores, corr


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
    allCorrs = []
    t0 = time.time()

    for idx in range(nSeries):
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period)
        if result is None:
            continue
        scores, corr = result
        for method, owa in scores.items():
            if method not in allScores:
                allScores[method] = []
            allScores[method].append(owa)
        allCorrs.append(corr)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            P(f'  [{idx+1}/{nSeries}] {speed:.1f} series/s')

    elapsed = time.time() - t0
    P(f'Completed in {elapsed:.1f}s ({nSeries/elapsed:.1f} series/s)')

    avgOwa = {}
    for method, owaList in allScores.items():
        avgOwa[method] = np.mean(owaList)

    avgCorr = np.mean(allCorrs) if allCorrs else 0.0

    return avgOwa, avgCorr


def _printResults(groupResults):
    P('\n' + '='*80)
    P('FINAL RESULTS: Topological Forecaster (TF)')
    P('='*80)

    methods = ['tf', 'dot', 'ces', 'naive2']
    groups = list(groupResults.keys())

    P(f'\n{"Method":<15}', end='')
    for g in groups:
        P(f'{g:>12}', end='')
    P(f'{"AVG":>12}')
    P('-' * (15 + 12 * (len(groups) + 1)))

    for method in methods:
        P(f'{method:<15}', end='')
        vals = []
        for g in groups:
            avgOwa, _ = groupResults[g]
            v = avgOwa.get(method, float('nan'))
            vals.append(v)
            marker = ' **' if v < 1.0 else '   '
            P(f'{v:>9.3f}{marker}', end='')
        avgAll = np.nanmean(vals)
        marker = ' **' if avgAll < 1.0 else '   '
        P(f'{avgAll:>9.3f}{marker}')

    P(f'\n{"TF-DOT Residual Correlation":}')
    for g in groups:
        _, corr = groupResults[g]
        P(f'  {g}: {corr:.4f}')
    avgCorr = np.mean([groupResults[g][1] for g in groups])
    P(f'  AVG: {avgCorr:.4f}')


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('='*80)
    P('Topological Forecaster (TF) — modelCreation/021')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        avgOwa, avgCorr = _runGroup(groupName, SAMPLE_PER_GROUP)
        groupResults[groupName] = (avgOwa, avgCorr)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
