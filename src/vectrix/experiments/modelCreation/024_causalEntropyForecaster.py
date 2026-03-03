"""
==============================================================================
실험 ID: modelCreation/024
실험명: Causal Entropy Forecaster (CEF)
==============================================================================

목적:
- 최대 엔트로피(MaxEnt) 원리를 시계열 예측에 적용하는 새로운 접근
- "가장 놀랍지 않은 미래" = 과거 패턴의 인과 제약을 만족하면서
  엔트로피를 최대화하는 분포의 기댓값
- Jaynes의 정보이론적 확률 해석: 가정을 최소화하는 예측
- 과적합 방지의 이론적 근거 → 보수적이지만 안정적인 예측

가설:
1. CEF가 Naive2보다 우수 (OWA < 1.0) — MaxEnt가 합리적 예측 생성
2. CEF와 DOT 잔차 상관 < 0.3 — 정보이론 vs 분해 기반의 차이
3. CEF가 노이즈 높은 데이터(Daily, Weekly)에서 과적합 방지로 강점

방법:
1. 과거 데이터에서 조건부 제약 추출:
   a. 1차 모멘트 제약: 조건부 평균 (최근 N개 값의 가중 평균)
   b. 2차 모멘트 제약: 조건부 분산 (최근 변동성)
   c. 추세 제약: 최근 기울기 (drift)
   d. 범위 제약: 역사적 최소/최대값 범위
2. MaxEnt 분포 = 제약을 만족하는 가장 넓은 분포:
   a. 평균+분산 제약 → 가우시안이 MaxEnt
   b. 기댓값 = 제약 기반 가우시안의 mean = 예측값
3. 다단계 예측: step-by-step, 각 step에서 제약 업데이트
   - 이전 step의 예측값이 다음 step의 제약에 반영
   - horizon이 길어질수록 분산 증가 (자연스러운 불확실성)
4. 비교군: DOT, AutoCES, Naive2
5. M4 6개 그룹 × 300 시리즈

핵심 차별점 (기존 문헌 대비):
- Jaynes 1957: MaxEnt 원리 → 물리학 통계역학에만 적용
- Rodriguez & Aler 2019: MaxEnt for load forecasting → 특정 도메인
- 본 실험: 범용 시계열 예측에 MaxEnt를 "제약 추출 + 분포 생성"으로
  체계화한 최초 시도. 기존 연구는 사전 정의된 제약 사용,
  본 방법은 데이터에서 제약을 자동 추출

결과 (M4 6그룹 × 300시리즈, 0.9분):
             Yearly  Quarterly  Monthly  Weekly   Daily   Hourly   AVG
cef          1.176   1.326      1.752    1.221   1.280   7.520    2.379
dot          0.987   0.985      0.983    1.011   0.998   0.858    0.970
ces          0.996   0.985      1.018    0.987   1.005   0.844    0.972
naive2       1.000   1.000      1.000    1.000   1.000   1.000    1.000

CEF-DOT 잔차 상관: Y=0.59, Q=0.69, M=0.54, W=0.90, D=0.90, H=0.16, AVG=0.63

결론:
- **가설 1 기각**: CEF가 Naive2보다 모든 그룹에서 악화 (AVG OWA 2.379)
  Hourly에서 7.52x로 치명적. Yearly(1.18x)이 가장 양호
- **가설 2 부분 확인**: 잔차 상관 AVG=0.63. Hourly(0.16)에서 거의 무상관,
  Monthly(0.54)에서도 낮은 편. 그러나 Weekly/Daily(0.90)에서는 높음
- **가설 3 기각**: 노이즈 데이터에서 과적합 방지 효과 없음.
  Weekly(1.22x), Daily(1.28x)에서도 Naive2보다 나쁨
- **5개 중 최선 (AVG 2.379)**: 5개 Novel Approach 중 가장 나쁘지 않음.
  AR1 + drift + seasonal 제약이 부분적으로 유효하지만,
  본질적으로 "가중 AR(1) + 계절 평균"과 동등 → MaxEnt의 이론적
  우아함에도 불구하고 실질적으로 단순 통계 모델과 유사
- **Hourly 참사 원인**: period=24 계절 분해에서 "계절 평균"만 사용.
  ETS/Theta의 적응형 계절 추정 대비 정보 손실이 큼
- **핵심 교훈**: MaxEnt 원리는 "가정 최소화"를 보장하지만,
  시계열 예측에서는 적절한 가정(트렌드 분해, 적응형 평활)이
  오히려 성능을 높임. "가정을 최소화"하면 정보도 최소화됨.
  과적합 방지 ≠ 좋은 예측. ETS/Theta의 구조적 가정이
  시계열의 실제 생성 과정과 더 잘 맞기 때문에 우수

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


def _extractConstraints(y, period=1):
    n = len(y)

    weights = np.exp(-0.05 * np.arange(n - 1, -1, -1))
    weights = weights / weights.sum()
    weightedMean = np.sum(weights * y)

    recentN = min(20, n)
    recentStd = np.std(y[-recentN:])
    if recentStd < 1e-10:
        recentStd = abs(weightedMean) * 0.01 + 1e-10

    driftN = min(10, n - 1)
    if driftN > 0:
        diffs = np.diff(y[-driftN - 1:])
        driftWeights = np.exp(-0.1 * np.arange(driftN - 1, -1, -1))
        driftWeights = driftWeights / driftWeights.sum()
        drift = np.sum(driftWeights * diffs)
    else:
        drift = 0.0

    histMin = np.min(y)
    histMax = np.max(y)
    rangeMargin = (histMax - histMin) * 0.2
    lowerBound = histMin - rangeMargin
    upperBound = histMax + rangeMargin

    seasonalComponent = None
    if period > 1 and n >= period * 2:
        nFull = (n // period) * period
        trimmed = y[-nFull:]
        reshaped = trimmed.reshape(-1, period)
        seasonalComponent = np.mean(reshaped, axis=0)
        globalMean = np.mean(trimmed)
        seasonalComponent = seasonalComponent - globalMean

    ar1Coef = 0.0
    if n > 2:
        yMean = np.mean(y)
        yDemeaned = y - yMean
        numerator = np.sum(yDemeaned[1:] * yDemeaned[:-1])
        denominator = np.sum(yDemeaned[:-1] ** 2)
        if abs(denominator) > 1e-10:
            ar1Coef = np.clip(numerator / denominator, -0.99, 0.99)

    return {
        'weightedMean': weightedMean,
        'recentStd': recentStd,
        'drift': drift,
        'lowerBound': lowerBound,
        'upperBound': upperBound,
        'seasonal': seasonalComponent,
        'period': period,
        'ar1': ar1Coef,
        'lastVal': y[-1],
    }


def _maxentPredict(trainY, horizon, period=1):
    n = len(trainY)
    constraints = _extractConstraints(trainY, period)

    predictions = np.zeros(horizon)
    extended = trainY.copy()

    for h in range(horizon):
        lastVal = extended[-1]
        drift = constraints['drift']
        ar1 = constraints['ar1']
        mean = constraints['weightedMean']

        arComponent = mean + ar1 * (lastVal - mean)
        driftComponent = lastVal + drift

        driftDecay = np.exp(-0.05 * (h + 1))
        arWeight = 0.5 + 0.3 * abs(ar1)
        driftWeight = (1.0 - arWeight) * driftDecay

        prediction = arWeight * arComponent + driftWeight * driftComponent
        residualWeight = 1.0 - arWeight - driftWeight
        if residualWeight > 0:
            prediction += residualWeight * mean

        if constraints['seasonal'] is not None:
            seasonIdx = (n + h) % constraints['period']
            prediction += constraints['seasonal'][seasonIdx]

        prediction = np.clip(prediction, constraints['lowerBound'], constraints['upperBound'])

        predictions[h] = prediction
        extended = np.append(extended, prediction)

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

    cefPred = _maxentPredict(trainY, horizon, period)
    results['cef'] = cefPred

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
    cefResid = testY[:horizon] - cefPred[:horizon]
    if len(dotResid) > 2:
        corr = np.corrcoef(dotResid, cefResid)[0, 1]
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
    P('FINAL RESULTS: Causal Entropy Forecaster (CEF)')
    P('='*80)

    methods = ['cef', 'dot', 'ces', 'naive2']
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

    P(f'\n{"CEF-DOT Residual Correlation":}')
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
    P('Causal Entropy Forecaster (CEF) — modelCreation/024')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        avgOwa, avgCorr = _runGroup(groupName, SAMPLE_PER_GROUP)
        groupResults[groupName] = (avgOwa, avgCorr)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
