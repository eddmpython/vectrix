"""
==============================================================================
실험 ID: modelCreation/022
실험명: Gravitational Forecaster (GF)
==============================================================================

목적:
- 물리학의 중력 개념을 시계열 예측에 적용하는 새로운 접근
- 과거 데이터 포인트가 "질량을 가진 입자"로 미래 값에 인력을 행사
- 인력의 평형점 = 예측값, 인력 분포 = 자연스러운 불확실성 구간
- DTSF(패턴 매칭)와 비슷하지만 이산 매칭이 아닌 연속적 힘장 개념

가설:
1. GF가 Naive2보다 우수 (OWA < 1.0) — 중력장이 패턴 포착
2. GF와 DOT 잔차 상관 < 0.3 — 근본적으로 다른 판단 기준
3. 추세+계절 혼합 데이터에서 자연스러운 감쇠 효과로 강점

방법:
1. 각 과거 값 Y(t)가 미래 값 Y(t+h)에 행사하는 인력:
   F(t, h) = mass(t) / distance(t, h)^2
   - mass(t) = 시간 감쇠 × 패턴 유사도
   - distance = |candidate - Y(t)| + softening
2. 시간 감쇠: exp(-λ * (n-t)) — 최근 값일수록 큰 인력
3. 패턴 유사도: 직전 W개 값의 유클리드 거리 기반 가중치
   가장 유사한 패턴 직후의 값이 큰 인력
4. 예측값 = 인력 가중 평균 (중력 평형점)
5. 비교군: DOT, AutoCES, Naive2
6. M4 6개 그룹 × 300 시리즈

핵심 차별점 (기존 문헌 대비):
- Gravitational search algorithm (Rashedi 2009): 최적화에만 사용
- Celestial mechanics in ML: 분류/클러스터링에 중력 사용
- 본 실험: 중력을 "예측 생성"에 직접 사용하는 최초 시도
  "미래값은 과거 데이터의 중력 평형점"이라는 프레임워크

결과 (M4 6그룹 × 300시리즈, 1.1분):
             Yearly  Quarterly  Monthly  Weekly   Daily   Hourly   AVG
gf           3.484   2.485      1.765    2.073   2.098   4.380    2.714
dot          0.987   0.985      0.983    1.011   0.998   0.858    0.970
ces          0.996   0.985      1.018    0.987   1.005   0.844    0.972
naive2       1.000   1.000      1.000    1.000   1.000   1.000    1.000

GF-DOT 잔차 상관: Y=0.51, Q=0.63, M=0.70, W=0.80, D=0.86, H=0.40, AVG=0.65

결론:
- **가설 1 완전 기각**: GF가 Naive2보다 모든 그룹에서 대폭 악화 (AVG OWA 2.714)
  Yearly(3.48x), Hourly(4.38x)에서 특히 나쁨
- **가설 2 부분 기각**: 잔차 상관 AVG=0.65로 기대(0.3)보다 높음.
  Yearly(0.51), Hourly(0.40)에서만 약간 다름
- **가설 3 기각**: 추세+계절 혼합에서 감쇠 효과 없이 오히려 악화
- **근본 문제**: "중력 평형점"이라는 개념은 시계열의 핵심 구조를
  무시함. 중력 가중 평균은 결국 시계열의 가중 평균에 수렴하는데,
  이는 패턴이나 추세를 반영하지 않고 "과거값의 중심"으로 회귀.
  시간 감쇠 + 패턴 유사도로 가중해도, 예측은 결국
  weighted mean(history) + small drift가 됨.
- **핵심 교훈**: 물리적 비유(중력)는 직관적이지만,
  시계열에서 "미래값은 과거값의 가중합"이라는 가정은
  분해 기반 모델보다 정보를 덜 활용함.
  ETS/Theta는 구조(트렌드, 계절, 오류)를 명시적으로 모델링하여
  각 성분을 최적 외삽하지만, 중력 모델은 구조 무시

실험일: 2026-03-03
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

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


def _gravitationalPredict(trainY, horizon, decayRate=0.05, windowSize=0, softening=0.0):
    n = len(trainY)
    if windowSize <= 0:
        windowSize = max(3, min(10, n // 20))

    yScale = np.std(trainY)
    if yScale < 1e-10:
        return np.full(horizon, trainY[-1])

    if softening <= 0:
        softening = yScale * 0.1

    timeDecay = np.exp(-decayRate * np.arange(n - 1, -1, -1))

    patternWeights = np.ones(n)
    if n > windowSize + 1:
        query = trainY[-(windowSize):]
        for t in range(windowSize, n):
            past = trainY[t - windowSize:t]
            dist = np.sqrt(np.mean((query - past) ** 2)) / yScale
            patternWeights[t] = np.exp(-dist)

    mass = timeDecay * patternWeights
    mass = mass / mass.sum()

    predictions = np.zeros(horizon)
    extended = trainY.copy()

    for h in range(horizon):
        nExt = len(extended)
        forces = np.zeros_like(extended)

        for t in range(nExt):
            forces[t] = mass[t] if t < len(mass) else mass[-1] * 0.5

        gravCenter = np.sum(forces * extended) / np.sum(forces)

        drift = extended[-1] - extended[-2] if nExt > 1 else 0.0
        driftDecay = np.exp(-0.1 * (h + 1))
        prediction = gravCenter + drift * driftDecay

        predictions[h] = prediction
        extended = np.append(extended, prediction)
        newMass = mass[-1] * np.exp(-decayRate)
        mass = np.append(mass, newMass)
        mass = mass / mass.sum()

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
    if n < 20:
        return None

    results = {}

    gfPred = _gravitationalPredict(trainY, horizon)
    results['gf'] = gfPred

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
    gfResid = testY[:horizon] - gfPred[:horizon]
    if len(dotResid) > 2:
        corr = np.corrcoef(dotResid, gfResid)[0, 1]
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
    P('FINAL RESULTS: Gravitational Forecaster (GF)')
    P('='*80)

    methods = ['gf', 'dot', 'ces', 'naive2']
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

    P(f'\n{"GF-DOT Residual Correlation":}')
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
    P('Gravitational Forecaster (GF) — modelCreation/022')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        avgOwa, avgCorr = _runGroup(groupName, SAMPLE_PER_GROUP)
        groupResults[groupName] = (avgOwa, avgCorr)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
