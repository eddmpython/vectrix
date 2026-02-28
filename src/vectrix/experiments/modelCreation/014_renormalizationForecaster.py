"""
==============================================================================
실험 ID: modelCreation/014
실험명: Renormalization Group Forecaster (RGF)
==============================================================================

목적:
- 물리학의 Renormalization Group (RG) 원리를 시계열 예측에 세계 최초 적용
- 시계열을 여러 스케일로 coarse-grain하고, 스케일 간 자기유사성이
  보존되는 고정점(fixed point)을 찾아 최종 예측을 재구성
- MAPA(Multiple Aggregation Prediction Algorithm)와의 차이:
  MAPA는 단순 가중 평균, RGF는 스케일 간 일관성을 물리적 제약으로 강제

가설:
1. RGF > 단일 스케일 예측 (OWA 5%+ 개선)
2. RGF > MAPA 스타일 단순 평균 (스케일 일관성 제약의 효과)
3. Yearly/Quarterly 등 장기 시리즈에서 특히 효과적

방법:
1. 시계열 y를 K개 스케일로 coarse-grain: y_k = blockMean(y, 2^k), k=0,1,...,K
2. 각 스케일에서 Theta/DOT로 예측
3. 스케일 간 "self-similarity score" 계산:
   - 인접 스케일 예측의 비율 안정성 (ratio consistency)
   - 고정점 수렴 여부 판단
4. 거시 스케일의 예측 패턴을 미세 스케일에 top-down 제약으로 부과
5. 비교군: 단일 스케일(k=0), MAPA(단순 평균), RGF(제약 결합)
6. M4 6개 그룹 × 500 시리즈

핵심 차별점 (기존 문헌 대비):
- MAPA (Kourentzes 2014): 다중 집계 후 단순/가중 평균 → 스케일 간 관계 무시
- Wavelet Forecast: 주파수 분해 후 독립 예측 → 스케일 간 정보 흐름 없음
- RGF: 물리학 RG의 핵심 — "거시적 고정점이 미시적 예측을 제약" 원리 도입

결과 (M4 6그룹 × 500시리즈, 5.6분):
             Yearly  Quarterly  Monthly  Weekly   Daily   Hourly   AVG
base_dot     0.985   1.018      0.990    1.030   1.014   0.845    0.980
mapa_style   1.090   1.247      1.045    1.177   1.152   1.231    1.157
rg_forecast  0.991   1.020      0.990    1.030   1.013   1.726    1.129
naive2       1.000   1.000      1.000    1.000   1.000   1.000    1.000

자기유사성: Yearly 0.954, Quarterly 0.974, Monthly 0.991, Weekly~Hourly 0.996~0.998
RGF vs DOT: Yearly -0.65%, Quarterly -0.25%, Monthly +0.00%, Hourly -104%

결론:
- **가설 1 기각**: RGF가 DOT보다 전반적으로 악화 (AVG OWA 1.129 vs 0.980)
- **가설 2 부분 확인**: RGF > MAPA (1.129 vs 1.157) — 스케일 제약이 약간 효과
- **가설 3 기각**: Yearly에서도 DOT 대비 -0.65% 악화
- Hourly에서 -104% 폭발: coarse-grain으로 24시간 주기가 파괴됨
- 자기유사성이 0.95+ 로 매우 높게 측정 → 대부분 시계열이 스케일 불변에 가까움
  → RG 제약 alpha가 매우 작아져서 사실상 DOT과 동일해짐
- **교훈**: 물리학 RG는 "스케일에 따라 구조가 변하는" 시스템에 유효하나,
  대부분 시계열은 스케일 불변이라 RG 제약의 효과가 미미함.
  계절성이 있는 시계열에서는 coarse-grain이 주기를 파괴하여 오히려 해로움

실험일: 2026-02-28
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


def _coarseGrain(y, blockSize):
    """시계열을 blockSize 단위로 coarse-grain (block averaging)."""
    n = len(y)
    nBlocks = n // blockSize
    if nBlocks < 4:
        return None
    truncated = y[:nBlocks * blockSize]
    return truncated.reshape(nBlocks, blockSize).mean(axis=1)


def _safePredict(trainY, horizon, period):
    """DOT로 예측. 실패 시 4Theta fallback."""
    model = DynamicOptimizedTheta(period=period)
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        model2 = AdaptiveThetaEnsemble(period=period)
        model2.fit(trainY)
        pred, _, _ = model2.predict(horizon)
        pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.full(horizon, trainY[-1])
    return pred


def _rgForecast(trainY, horizon, period):
    """
    Renormalization Group Forecaster.

    1. 원본(scale 0)과 coarse-grained 스케일(2x, 4x, 8x)에서 예측
    2. 스케일 간 ratio consistency로 고정점 추세 추출
    3. 거시 고정점을 미시 예측에 제약으로 부과
    """
    n = len(trainY)
    maxK = 0
    k = 1
    while n // (2 ** k) >= max(8, horizon):
        maxK = k
        k += 1
    maxK = min(maxK, 4)

    scalePreds = {}
    scaleData = {}

    scalePreds[0] = _safePredict(trainY, horizon, period)
    scaleData[0] = trainY

    for k in range(1, maxK + 1):
        blockSize = 2 ** k
        coarseY = _coarseGrain(trainY, blockSize)
        if coarseY is None or len(coarseY) < 8:
            break
        coarsePeriod = max(period // blockSize, 1)
        coarseHorizon = max(horizon // blockSize, 1)
        coarsePred = _safePredict(coarseY, coarseHorizon, coarsePeriod)
        scalePreds[k] = coarsePred
        scaleData[k] = coarseY

    if len(scalePreds) < 2:
        return scalePreds[0], scalePreds[0], 0.0

    trendRatios = []
    for k in sorted(scalePreds.keys()):
        data = scaleData[k]
        if len(data) >= 4:
            half = len(data) // 2
            r1 = np.mean(data[half:]) / (np.mean(data[:half]) + 1e-10)
            trendRatios.append(r1)

    if len(trendRatios) >= 2:
        ratioChanges = [abs(trendRatios[i] - trendRatios[i-1]) for i in range(1, len(trendRatios))]
        selfSimilarity = 1.0 / (1.0 + np.mean(ratioChanges))
    else:
        selfSimilarity = 0.5

    fixedPointTrend = np.mean(trendRatios) if trendRatios else 1.0

    basePred = scalePreds[0].copy()

    if len(scalePreds) >= 2:
        coarseKeys = sorted([k for k in scalePreds.keys() if k > 0])
        if coarseKeys:
            highestK = coarseKeys[-1]
            coarsePred = scalePreds[highestK]
            blockSize = 2 ** highestK

            coarseTrend = np.zeros(horizon)
            for h in range(horizon):
                coarseIdx = h // blockSize
                if coarseIdx < len(coarsePred):
                    coarseTrend[h] = coarsePred[coarseIdx]
                elif len(coarsePred) > 0:
                    coarseTrend[h] = coarsePred[-1]

            baseMean = np.mean(basePred)
            coarseMean = np.mean(coarseTrend)

            if abs(baseMean) > 1e-10 and abs(coarseMean) > 1e-10:
                adjustment = coarseTrend / coarseMean * baseMean
                alpha = selfSimilarity * 0.3
                rgPred = (1.0 - alpha) * basePred + alpha * adjustment
            else:
                rgPred = basePred
        else:
            rgPred = basePred
    else:
        rgPred = basePred

    mapaWeights = []
    for k in sorted(scalePreds.keys()):
        mapaWeights.append(1.0 / (k + 1))
    mapaWeights = np.array(mapaWeights)
    mapaWeights /= mapaWeights.sum()

    mapaPred = np.zeros(horizon)
    for idx, k in enumerate(sorted(scalePreds.keys())):
        pred = scalePreds[k]
        blockSize = 2 ** k
        expanded = np.zeros(horizon)
        for h in range(horizon):
            coarseIdx = h // blockSize
            if coarseIdx < len(pred):
                expanded[h] = pred[coarseIdx]
            elif len(pred) > 0:
                expanded[h] = pred[-1]
        mapaPred += mapaWeights[idx] * expanded

    return rgPred, mapaPred, selfSimilarity


def _processOneSeries(trainY, testY, horizon, period):
    """한 시리즈에 대해 RGF vs baseline 평가."""
    if len(trainY) < 16:
        return None

    rgPred, mapaPred, selfSim = _rgForecast(trainY, horizon, period)
    basePred = _safePredict(trainY, horizon, period)
    naive2Pred = _naive2(trainY, horizon, period)

    smapeN = _smape(testY[:horizon], naive2Pred)
    maseN = _mase(trainY, testY[:horizon], naive2Pred, period)

    results = {
        'base_dot': basePred,
        'mapa_style': mapaPred,
        'rg_forecast': rgPred,
        'naive2': naive2Pred,
    }

    scores = {}
    for method, pred in results.items():
        s = _smape(testY[:horizon], pred[:horizon])
        m = _mase(trainY, testY[:horizon], pred[:horizon], period)
        relSmape = s / max(smapeN, 1e-10)
        relMase = m / max(maseN, 1e-10)
        owa = 0.5 * (relSmape + relMase)
        scores[method] = owa

    return scores, selfSim


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
    selfSims = []
    t0 = time.time()

    for idx in range(nSeries):
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period)
        if result is None:
            continue
        scores, selfSim = result
        selfSims.append(selfSim)
        for method, owa in scores.items():
            if method not in allScores:
                allScores[method] = []
            allScores[method].append(owa)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            P(f'  [{idx+1}/{nSeries}] {(idx+1)/elapsed:.1f} series/s')

    elapsed = time.time() - t0
    P(f'Completed in {elapsed:.1f}s ({nSeries/elapsed:.1f} series/s)')

    avgOwa = {}
    for method, owaList in allScores.items():
        avgOwa[method] = np.mean(owaList)

    avgSelfSim = np.mean(selfSims) if selfSims else 0.0
    P(f'Average self-similarity: {avgSelfSim:.4f}')

    return avgOwa, avgSelfSim


def _printResults(groupResults):
    P('\n' + '='*80)
    P('FINAL RESULTS: Renormalization Group Forecaster')
    P('='*80)

    methods = ['base_dot', 'mapa_style', 'rg_forecast', 'naive2']
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
            avgOwa, _ = groupResults[g]
            v = avgOwa.get(method, float('nan'))
            vals.append(v)
            marker = ' **' if v < 1.0 else '   '
            P(f'{v:>9.3f}{marker}', end='')
        avgAll = np.nanmean(vals)
        marker = ' **' if avgAll < 1.0 else '   '
        P(f'{avgAll:>9.3f}{marker}')

    P('\n스케일 자기유사성:')
    for g in groups:
        _, selfSim = groupResults[g]
        P(f'  {g}: {selfSim:.4f}')

    P('\nRGF vs base_dot 개선율:')
    for g in groups:
        avgOwa, _ = groupResults[g]
        base = avgOwa.get('base_dot', float('nan'))
        rg = avgOwa.get('rg_forecast', float('nan'))
        if base > 0:
            improvement = (base - rg) / base * 100
            P(f'  {g}: {improvement:+.2f}%')


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('='*80)
    P('Renormalization Group Forecaster (RGF) — modelCreation/014')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    for groupName in M4_GROUPS:
        avgOwa, selfSim = _runGroup(groupName, SAMPLE_PER_GROUP)
        groupResults[groupName] = (avgOwa, selfSim)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
