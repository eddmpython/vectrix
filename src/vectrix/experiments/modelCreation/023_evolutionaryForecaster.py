"""
==============================================================================
실험 ID: modelCreation/023
실험명: Evolutionary Forecaster (EF)
==============================================================================

목적:
- 유전 알고리즘(GA)을 시계열 예측에 직접 적용하는 새로운 접근
- 예측 후보(개체)를 생성 → 과거 데이터와의 일관성으로 적합도 평가 →
  선택/교배/돌연변이 → 생존자 평균 = 최종 예측
- 어떤 모델 구조도 가정하지 않음 — 적합성 함수만 정의

가설:
1. EF가 Naive2보다 우수 (OWA < 1.0) — 진화가 좋은 예측을 선택
2. EF와 DOT 잔차 상관 < 0.3 — 근본적으로 다른 생성 원리
3. 불규칙 패턴(Weekly, Daily)에서 모델-프리의 강점 발현

방법:
1. 초기 집단: 최근 값 기반 ±범위에서 무작위 예측 벡터 N=200개 생성
2. 적합도 함수:
   a. 연속성: 예측의 첫 값과 마지막 관측값의 차이 (작을수록 좋음)
   b. 역사적 일관성: 과거에서 유사한 패턴 직후 값과의 거리 (작을수록 좋음)
   c. 부드러움: 예측 벡터의 2차 차분 크기 (작을수록 좋음)
   d. 적합도 = -weighted_sum(a, b, c) (클수록 좋은 개체)
3. 진화:
   a. 선택: 토너먼트 선택 (k=3)
   b. 교배: 두 부모의 가중 평균 (crossover rate 0.8)
   c. 돌연변이: 가우시안 노이즈 추가 (mutation rate 0.1, 감쇠)
4. 세대: 50 세대 반복, 상위 10%의 평균 = 최종 예측
5. 비교군: DOT, AutoCES, Naive2
6. M4 6개 그룹 × 300 시리즈

핵심 차별점 (기존 문헌 대비):
- Evol. computing in forecasting (2020): GA로 ARIMA 파라미터 최적화 → 모델 구조 필요
- GP for symbolic regression (Koza 1992): 수식 진화 → 수식 가정 필요
- 본 실험: 예측값 자체를 직접 진화시키는 최초 시도
  모델도, 수식도, 파라미터도 없이 "좋은 예측"만 정의하면 됨

결과 (M4 6그룹 × 300시리즈, 12.0분):
             Yearly  Quarterly  Monthly  Weekly   Daily   Hourly   AVG
ef           1.114   1.432      2.076    1.548   1.650   27.555   5.896
dot          0.987   0.985      0.983    1.011   0.998   0.858    0.970
ces          0.996   0.985      1.018    0.987   1.005   0.844    0.972
naive2       1.000   1.000      1.000    1.000   1.000   1.000    1.000

EF-DOT 잔차 상관: Y=0.70, Q=0.59, M=0.53, W=0.71, D=0.75, H=0.16, AVG=0.57

결론:
- **가설 1 완전 기각**: EF가 Naive2보다 모든 그룹에서 악화 (AVG OWA 5.896)
  Hourly에서 27.56x로 치명적. Yearly(1.11x)이 가장 양호하지만 여전히 Naive2 이하
- **가설 2 부분 확인**: 잔차 상관 AVG=0.57로 기존 모델보다 낮은 편.
  특히 Hourly(0.16)에서 거의 무상관 → 앙상블 다양성 잠재력
- **가설 3 기각**: 불규칙 패턴에서 오히려 더 나쁨 (Daily 1.65x, Weekly 1.55x)
- **Hourly 참사 원인**: horizon=48로 긴 예측 벡터(48차원)를
  200개체 × 50세대로 탐색하기에 공간이 너무 넓음.
  차원의 저주로 좋은 예측을 찾지 못함
- **적합도 함수 한계**: "연속성 + 역사적 일관성 + 부드러움"은
  좋은 예측을 정의하기에 너무 느슨한 조건.
  실제로 좋은 예측은 "올바른 추세와 계절 패턴 외삽"이지만,
  이를 적합도 함수로 표현하면 결국 모델 구조가 필요해짐
- **핵심 교훈**: GA로 예측값을 직접 진화시키는 접근은
  "좋은 예측"의 정의가 모호하기 때문에 실패.
  모델 구조 없이 적합도 함수만으로는 시계열의 구조적 패턴을
  학습할 수 없음. GA는 파라미터 최적화에는 유효하지만,
  예측값 자체의 생성에는 부적합

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


def _findSimilarPatterns(trainY, windowSize, K=10):
    n = len(trainY)
    if n < windowSize + 2:
        return []
    query = trainY[-windowSize:]
    qStd = np.std(query)
    if qStd < 1e-10:
        qStd = 1.0

    patterns = []
    for t in range(windowSize, n - 1):
        past = trainY[t - windowSize:t]
        dist = np.sqrt(np.mean(((query - past) / qStd) ** 2))
        afterVal = trainY[t]
        patterns.append((dist, afterVal, t))

    patterns.sort(key=lambda x: x[0])
    return patterns[:K]


def _fitness(individual, trainY, horizon, patterns, wCont=1.0, wHist=1.0, wSmooth=0.5):
    lastVal = trainY[-1]
    contPenalty = abs(individual[0] - lastVal) / (abs(lastVal) + 1e-10)

    histPenalty = 0.0
    if patterns:
        for dist, afterVal, _ in patterns[:5]:
            weight = 1.0 / (dist + 1e-10)
            histPenalty += weight * abs(individual[0] - afterVal)
        histPenalty /= sum(1.0 / (p[0] + 1e-10) for p in patterns[:5])
        histPenalty /= (abs(lastVal) + 1e-10)

    if len(individual) > 2:
        d2 = np.diff(individual, n=2)
        smoothPenalty = np.mean(np.abs(d2)) / (np.std(trainY) + 1e-10)
    else:
        smoothPenalty = 0.0

    return -(wCont * contPenalty + wHist * histPenalty + wSmooth * smoothPenalty)


def _evolutionaryPredict(trainY, horizon, popSize=200, nGenerations=50, seed=42):
    rng = np.random.default_rng(seed)
    n = len(trainY)

    recentMean = np.mean(trainY[-min(20, n):])
    recentStd = np.std(trainY[-min(20, n):])
    if recentStd < 1e-10:
        recentStd = abs(recentMean) * 0.01 + 1e-10

    windowSize = max(3, min(10, n // 20))
    patterns = _findSimilarPatterns(trainY, windowSize, K=10)

    drift = 0.0
    if n > 3:
        recentDiffs = np.diff(trainY[-min(10, n):])
        drift = np.mean(recentDiffs)

    population = np.zeros((popSize, horizon))
    for i in range(popSize):
        base = trainY[-1] + drift * np.arange(1, horizon + 1)
        noise = rng.normal(0, recentStd * 0.5, horizon)
        population[i] = base + noise

    crossoverRate = 0.8
    mutationRate = 0.1
    eliteRatio = 0.1
    nElite = max(2, int(popSize * eliteRatio))

    for gen in range(nGenerations):
        fitnesses = np.array([
            _fitness(ind, trainY, horizon, patterns) for ind in population
        ])

        sortedIdx = np.argsort(fitnesses)[::-1]
        population = population[sortedIdx]
        fitnesses = fitnesses[sortedIdx]

        newPop = np.zeros_like(population)
        newPop[:nElite] = population[:nElite]

        for i in range(nElite, popSize):
            t1, t2, t3 = rng.choice(popSize, 3, replace=False)
            parent1Idx = t1 if fitnesses[t1] > fitnesses[t2] else t2
            t4, t5, t6 = rng.choice(popSize, 3, replace=False)
            parent2Idx = t4 if fitnesses[t4] > fitnesses[t5] else t5

            parent1 = population[parent1Idx]
            parent2 = population[parent2Idx]

            if rng.random() < crossoverRate:
                alpha = rng.uniform(0.2, 0.8)
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()

            mutScale = recentStd * 0.3 * (1.0 - gen / nGenerations)
            if rng.random() < mutationRate:
                child += rng.normal(0, mutScale, horizon)

            newPop[i] = child

        population = newPop

    finalFitnesses = np.array([
        _fitness(ind, trainY, horizon, patterns) for ind in population
    ])
    sortedIdx = np.argsort(finalFitnesses)[::-1]
    topN = max(5, int(popSize * eliteRatio))
    predictions = np.mean(population[sortedIdx[:topN]], axis=0)

    return predictions


def _safePredict(modelClass, trainY, horizon, period):
    model = modelClass(period=period)
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred


def _processOneSeries(trainY, testY, horizon, period, idx=0):
    n = len(trainY)
    if n < 20:
        return None

    results = {}

    efPred = _evolutionaryPredict(trainY, horizon, seed=42 + idx)
    results['ef'] = efPred

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
    efResid = testY[:horizon] - efPred[:horizon]
    if len(dotResid) > 2:
        corr = np.corrcoef(dotResid, efResid)[0, 1]
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
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period, idx)
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
    P('FINAL RESULTS: Evolutionary Forecaster (EF)')
    P('='*80)

    methods = ['ef', 'dot', 'ces', 'naive2']
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

    P(f'\n{"EF-DOT Residual Correlation":}')
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
    P('Evolutionary Forecaster (EF) — modelCreation/023')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        avgOwa, avgCorr = _runGroup(groupName, SAMPLE_PER_GROUP)
        groupResults[groupName] = (avgOwa, avgCorr)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
