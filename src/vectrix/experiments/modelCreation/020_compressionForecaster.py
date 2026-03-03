"""
==============================================================================
실험 ID: modelCreation/020
실험명: Compression-Based Forecaster (CBF)
==============================================================================

목적:
- 정보이론 기반의 완전히 새로운 예측 원리 검증
- "예측 = 압축": 시계열 뒤에 어떤 값이 올 때 전체 압축률이 가장 높은지 탐색
- Kolmogorov complexity의 실용적 근사 (zlib 사용)
- 기존 통계 모델과 근본적으로 다른 원리 → 잔차 상관 ~0 기대

가설:
1. CBF가 Naive2보다 우수 (OWA < 1.0) — 압축이 패턴을 포착
2. CBF와 DOT/CES 잔차 상관 < 0.3 — 근본적으로 다른 판단 기준
3. CBF가 규칙적 패턴(Yearly, Monthly)에서 강하고 노이즈(Daily)에서 약함

방법:
1. 시계열 Y를 바이트로 직렬화 (float64 → bytes)
2. 각 예측 시점 h에서:
   a. 후보값 grid 생성 (최근 값 기준 ±범위, N=50개)
   b. Y + 후보값을 연결한 바이트열의 zlib 압축 크기 측정
   c. 압축 크기가 가장 작은 후보값 = 예측값 (가장 자연스러운 연속)
   d. 선택된 값을 Y에 추가하고 다음 시점으로
3. 비교군: DOT, AutoCES, Naive2
4. M4 6개 그룹 × 300 시리즈

핵심 차별점 (기존 문헌 대비):
- Cilibrasi & Vitanyi 2005: NCD(Normalized Compression Distance)를 클러스터링에 사용
- Keogh 2004: Compression-based similarity for time series
- 본 실험: 압축을 "예측 생성"에 직접 사용하는 최초 시도
  기존 연구는 압축을 시계열 유사도 측정에만 사용했으며,
  "미래값 후보 중 압축률 최적 선택"이라는 예측 프레임워크는 전례 없음

결과 (M4 6그룹 × 300시리즈, 3.0분):
             Yearly  Quarterly  Monthly  Weekly   Daily   Hourly   AVG
cbf          8.766   4.690      2.696    3.036   2.969   7.777    4.989
dot          0.987   0.985      0.983    1.011   0.998   0.858    0.970
ces          0.996   0.985      1.018    0.987   1.005   0.844    0.972
naive2       1.000   1.000      1.000    1.000   1.000   1.000    1.000

CBF-DOT 잔차 상관: Y=0.66, Q=0.78, M=0.75, W=0.94, D=0.97, H=0.40, AVG=0.75

결론:
- **가설 1 완전 기각**: CBF가 Naive2보다 모든 그룹에서 대폭 악화 (AVG OWA 4.989)
  Yearly 8.77x, Hourly 7.78x로 특히 나쁨
- **가설 2 기각**: 잔차 상관 AVG=0.75로 기존 모델과 유사하게 틀림.
  Weekly(0.94), Daily(0.97)에서 거의 동일한 잔차 패턴
- **가설 3 기각**: 규칙적 패턴에서 오히려 가장 나쁨 (Yearly 8.77x)
- **근본 문제**: zlib 압축은 바이트 레벨 반복 패턴 감지하지만,
  float64의 바이트 표현에서 "값이 비슷한" 두 수의 바이트는 완전히 다름.
  100.0과 100.1의 바이트 표현은 전혀 달라서 압축 알고리즘이
  수치적 유사성을 포착 불가. 양자화가 필요하지만 정보 손실 발생.
- **핵심 교훈**: Kolmogorov complexity의 zlib 근사는 연속값 시계열에
  부적합. 텍스트/이산 시퀀스에는 유효하지만 부동소수점 바이트의
  비직관성 때문에 실패

실험일: 2026-03-03
"""

import os
import sys
import time
import zlib
import struct
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


def _seriesToBytes(y):
    return struct.pack(f'{len(y)}d', *y)


def _compressedSize(data):
    return len(zlib.compress(data, level=6))


def _compressionPredict(trainY, horizon, nCandidates=50, contextLen=0):
    n = len(trainY)
    if contextLen <= 0:
        contextLen = min(n, 200)
    context = trainY[-contextLen:]

    recentMean = np.mean(trainY[-min(20, n):])
    recentStd = np.std(trainY[-min(20, n):])
    if recentStd < 1e-10:
        recentStd = abs(recentMean) * 0.01 + 1e-10

    extended = context.copy()
    predictions = np.zeros(horizon)

    for h in range(horizon):
        lo = recentMean - 3.0 * recentStd
        hi = recentMean + 3.0 * recentStd
        candidates = np.linspace(lo, hi, nCandidates)

        baseBytes = _seriesToBytes(extended)
        baseSize = _compressedSize(baseBytes)

        bestVal = recentMean
        bestSize = float('inf')

        for cand in candidates:
            testSeq = np.append(extended, cand)
            testBytes = _seriesToBytes(testSeq)
            cSize = _compressedSize(testBytes)
            if cSize < bestSize:
                bestSize = cSize
                bestVal = cand

        refineLo = bestVal - (hi - lo) / nCandidates
        refineHi = bestVal + (hi - lo) / nCandidates
        refineCandidates = np.linspace(refineLo, refineHi, 20)
        for cand in refineCandidates:
            testSeq = np.append(extended, cand)
            testBytes = _seriesToBytes(testSeq)
            cSize = _compressedSize(testBytes)
            if cSize < bestSize:
                bestSize = cSize
                bestVal = cand

        predictions[h] = bestVal
        extended = np.append(extended, bestVal)

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

    cbfPred = _compressionPredict(trainY, horizon)
    results['cbf'] = cbfPred

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
    cbfResid = testY[:horizon] - cbfPred[:horizon]
    if len(dotResid) > 2:
        corr = np.corrcoef(dotResid, cbfResid)[0, 1]
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
    P('FINAL RESULTS: Compression-Based Forecaster (CBF)')
    P('='*80)

    methods = ['cbf', 'dot', 'ces', 'naive2']
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

    P(f'\n{"CBF-DOT Residual Correlation":}')
    for g in groups:
        _, corr = groupResults[g]
        P(f'  {g}: {corr:.4f}')
    avgCorr = np.mean([groupResults[g][1] for g in groups])
    P(f'  AVG: {avgCorr:.4f}')

    P('\nCBF vs DOT (OWA difference):')
    for g in groups:
        avgOwa, _ = groupResults[g]
        cbfOwa = avgOwa.get('cbf', float('nan'))
        dotOwa = avgOwa.get('dot', float('nan'))
        diff = cbfOwa - dotOwa
        P(f'  {g}: {diff:+.4f} ({"CBF better" if diff < 0 else "DOT better"})')


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('='*80)
    P('Compression-Based Forecaster (CBF) — modelCreation/020')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        avgOwa, avgCorr = _runGroup(groupName, SAMPLE_PER_GROUP)
        groupResults[groupName] = (avgOwa, avgCorr)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
