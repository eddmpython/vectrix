"""
==============================================================================
실험 ID: modelCreation/013
실험명: Wasserstein Diversity Ensemble (WDE)
==============================================================================

목적:
- 세상에 없는 완전히 새로운 앙상블 원리 검증
- 기존 inverse-error weighting (Bates & Granger 1969)은 모델의 "정확도"만 고려
- WDE는 모델 잔차의 "분포적 다양성"을 Wasserstein 거리로 측정하여
  "다르게 실패하는 모델"에 가중치 보너스를 부여

가설:
1. Wasserstein 다양성 가중치 > inverse-MAPE 가중치 (OWA 3%+ 개선)
2. 잔차 분포가 다른 모델(DTSF, ESN)일수록 높은 다양성 점수
3. 정확도 + 다양성 결합이 어느 한쪽만보다 우수

방법:
1. M4 6개 그룹에서 시리즈별로:
   a. 각 모델 fit → in-sample 잔차 분포 계산
   b. 모든 모델 쌍의 1D Wasserstein 거리 행렬 구성
   c. 다양성 점수 = 해당 모델과 타 모델 잔차 간 평균 W-거리
   d. 정확도 점수 = holdout MAPE 역수
   e. 최종 가중치 = (1-α)·정확도 + α·다양성 (α ∈ {0, 0.1, ..., 1.0})
2. 비교군: equal, inverse_mape, diversity_only, combined(최적 α)
3. 최소 1000 시리즈 (M4 각 그룹에서 500 샘플 × 6 = 3000)

핵심 차별점 (기존 문헌 대비):
- Bates & Granger (1969): 점 추정 정확도(MSE)만 사용
- Pawlikowski M4#3 (2018): inverse MAPE → 여전히 점 추정
- FFORMA M4#2: 42개 시계열 피처 → 모델 자체 피처, 잔차 분포 무시
- WDE: 잔차의 전체 분포 구조(skew, fat tail, multimodality)를
  Wasserstein 거리로 비교 → "분포적 다양성"이라는 새 축 도입

결과 (M4 4그룹: Yearly 500 + Quarterly 500 + Monthly 500 + Hourly 200, 47.6분):
                 Yearly  Quarterly  Monthly  Hourly   AVG
inv_mape         0.986   1.067      1.086    0.768    0.977   ← 최적
wde_a0.1         0.996   1.095      1.110    0.785    0.997
wde_a0.2         1.009   1.142      1.141    0.810    1.026
wde_a0.5         1.071   1.334      1.259    0.902    1.142
diversity_only   1.226   1.724      1.502    1.079    1.383   ← 최악
equal            1.195   1.502      1.393    1.110    1.300
single_ces       1.002   0.975      0.978    0.843    0.949   ← 단일 최강
single_dot       1.008   1.018      0.990    0.872    0.972

최적 alpha: 전 그룹에서 alpha=0.0 (순수 inv_mape)이 최적

Wasserstein 거리 (Hourly):
  dot-dtsf: 0.607, ces-dtsf: 0.611, 4theta-dtsf: 0.623  ← DTSF가 가장 다름
  dot-esn: 0.225, ces-esn: 0.408  ← ESN은 중간 정도

가중치 비교 (Hourly):
  4theta: 정확도 0.136, 다양성 0.199 (+0.063)  ← 다양성으로 과대평가
  dot: 정확도 0.244, 다양성 0.169 (-0.075)  ← 다양성으로 과소평가

결론:
- **가설 1 완전 기각**: Wasserstein 다양성 가중치를 추가하면 성능이 일관되게 악화
- **가설 2 확인**: DTSF가 Wasserstein 거리 최대(0.6+) → 가장 "다르게 실패" 확인
- **가설 3 완전 기각**: alpha=0.0이 모든 그룹에서 최적 → 다양성 추가 가치 없음
- **핵심 교훈**: "다르게 실패하는 모델"이 존재하는 것은 사실이나,
  그 다양성에 보너스 가중치를 주면 정확도가 낮은 모델(DTSF, ESN)의 비중이 증가하여
  오히려 앙상블 성능이 악화됨.
  Wasserstein 거리는 "다양성 측정"에는 유효하나 "가중치 결정"에는 부적합.
  가장 정확한 모델에 가장 높은 가중치를 주는 inv_mape가 여전히 최적.
- **부가 발견**: 단일 모델 CES(0.949)가 모든 앙상블(0.977+)보다 우수
  → 이 모델 풀(5개)에서는 앙상블 자체의 가치도 의문

실험일: 2026-02-28
"""

import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
from vectrix.engine.esn import EchoStateForecaster
from vectrix.engine.dtsf import DynamicTimeScanForecaster
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


def _safePredict(modelFactory, trainY, horizon):
    model = modelFactory()
    model.fit(trainY)
    pred, _, _ = model.predict(horizon)
    pred = np.asarray(pred[:horizon], dtype=np.float64)
    if not np.all(np.isfinite(pred)):
        pred = np.where(np.isfinite(pred), pred, np.mean(trainY))
    return pred, model


def _getResiduals(model, trainY):
    if hasattr(model, 'residuals') and model.residuals is not None:
        res = model.residuals
        if callable(res):
            res = res()
        res = np.asarray(res, dtype=np.float64)
        if len(res) > 0:
            return res
    if hasattr(model, '_fitted') and model._fitted is not None:
        fitted = np.asarray(model._fitted, dtype=np.float64)
        if len(fitted) == len(trainY):
            return trainY - fitted
    return trainY[1:] - trainY[:-1]


def _wassersteinDiversityMatrix(residualsDict):
    """모든 모델 쌍의 1D Wasserstein 거리 행렬을 계산."""
    names = list(residualsDict.keys())
    nModels = len(names)
    wMatrix = np.zeros((nModels, nModels))
    for i in range(nModels):
        for j in range(i + 1, nModels):
            ri = residualsDict[names[i]]
            rj = residualsDict[names[j]]
            if len(ri) < 2 or len(rj) < 2:
                dist = 0.0
            else:
                ri_std = np.std(ri)
                rj_std = np.std(rj)
                scale = max(ri_std, rj_std, 1e-10)
                dist = wasserstein_distance(ri / scale, rj / scale)
            wMatrix[i, j] = dist
            wMatrix[j, i] = dist
    return wMatrix, names


def _diversityScores(wMatrix):
    """각 모델의 다양성 점수 = 타 모델과의 평균 Wasserstein 거리."""
    nModels = wMatrix.shape[0]
    scores = np.zeros(nModels)
    for i in range(nModels):
        others = [wMatrix[i, j] for j in range(nModels) if j != i]
        scores[i] = np.mean(others) if others else 0.0
    return scores


def _combinedWeights(accuracyWeights, diversityWeights, alpha):
    """정확도 가중치와 다양성 가중치를 alpha 비율로 결합."""
    accNorm = accuracyWeights / (accuracyWeights.sum() + 1e-10)
    divNorm = diversityWeights / (diversityWeights.sum() + 1e-10)
    combined = (1.0 - alpha) * accNorm + alpha * divNorm
    combined = combined / (combined.sum() + 1e-10)
    return combined


def _processOneSeries(trainY, testY, horizon, period):
    """한 시리즈에 대해 모든 앙상블 전략을 평가."""
    n = len(trainY)
    holdoutSize = min(max(horizon, 4), n // 4)
    if holdoutSize < 2:
        holdoutSize = 2
    fitY = trainY[:-holdoutSize]
    holdoutY = trainY[-holdoutSize:]

    if len(fitY) < 10:
        return None

    modelFactories = {
        'dot': lambda: DynamicOptimizedTheta(period=period),
        'ces': lambda: AutoCES(period=period),
        '4theta': lambda: AdaptiveThetaEnsemble(period=period),
    }

    if n >= 30:
        modelFactories['dtsf'] = lambda: DynamicTimeScanForecaster()
    modelFactories['esn'] = lambda: EchoStateForecaster()

    predictions = {}
    holdoutPreds = {}
    residualsDict = {}
    holdoutMapes = {}

    for mName, factory in modelFactories.items():
        pred, model = _safePredict(factory, fitY, holdoutSize)
        holdoutPreds[mName] = pred
        mape = np.mean(np.abs(holdoutY - pred) / (np.abs(holdoutY) + 1e-10))
        holdoutMapes[mName] = max(mape, 1e-10)

        res = _getResiduals(model, fitY)
        if len(res) < 2:
            res = fitY[1:] - fitY[:-1]
        residualsDict[mName] = res

    for mName, factory in modelFactories.items():
        pred, _ = _safePredict(factory, trainY, horizon)
        predictions[mName] = pred

    names = list(predictions.keys())
    nModels = len(names)

    if nModels < 2:
        return None

    predMatrix = np.column_stack([predictions[nm] for nm in names])

    accWeights = np.array([1.0 / holdoutMapes[nm] for nm in names])
    accWeights = accWeights / accWeights.sum()

    wMatrix, wNames = _wassersteinDiversityMatrix(
        {nm: residualsDict[nm] for nm in names}
    )
    divScores = _diversityScores(wMatrix)
    divWeights = divScores / (divScores.sum() + 1e-10)

    results = {}

    equalPred = predMatrix.mean(axis=1)
    results['equal'] = equalPred

    results['inv_mape'] = predMatrix @ accWeights

    results['diversity_only'] = predMatrix @ divWeights

    alphas = np.arange(0.0, 1.05, 0.1)
    for alpha in alphas:
        key = f'wde_a{alpha:.1f}'
        cw = _combinedWeights(accWeights, divWeights, alpha)
        results[key] = predMatrix @ cw

    naive2Pred = _naive2(trainY, horizon, period)
    results['naive2'] = naive2Pred

    for nm in names:
        results[f'single_{nm}'] = predictions[nm]

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

    wMatrixData = {
        'names': names,
        'accWeights': {nm: float(accWeights[i]) for i, nm in enumerate(names)},
        'divWeights': {nm: float(divWeights[i]) for i, nm in enumerate(names)},
        'wDistances': {f'{names[i]}-{names[j]}': float(wMatrix[i, j])
                       for i in range(nModels) for j in range(i + 1, nModels)},
    }

    return scores, wMatrixData


def _runGroup(groupName, maxSeries=SAMPLE_PER_GROUP):
    """M4 한 그룹에 대해 WDE 실험 실행."""
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
    allWData = []
    t0 = time.time()

    for idx in range(nSeries):
        result = _processOneSeries(trainSeries[idx], testSeries[idx], horizon, period)
        if result is None:
            continue
        scores, wData = result
        for method, owa in scores.items():
            if method not in allScores:
                allScores[method] = []
            allScores[method].append(owa)
        allWData.append(wData)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed
            P(f'  [{idx+1}/{nSeries}] {speed:.1f} series/s')

    elapsed = time.time() - t0
    P(f'Completed in {elapsed:.1f}s ({nSeries/elapsed:.1f} series/s)')

    avgOwa = {}
    for method, owaList in allScores.items():
        avgOwa[method] = np.mean(owaList)

    return avgOwa, allWData


def _printResults(groupResults):
    """전체 결과를 정리하여 출력."""
    P('\n' + '='*80)
    P('FINAL RESULTS: Wasserstein Diversity Ensemble')
    P('='*80)

    allMethods = set()
    for gName, (avgOwa, _) in groupResults.items():
        allMethods.update(avgOwa.keys())

    coreMethods = ['equal', 'inv_mape', 'diversity_only']
    wdeMethods = sorted([m for m in allMethods if m.startswith('wde_a')])
    singleMethods = sorted([m for m in allMethods if m.startswith('single_')])
    displayMethods = coreMethods + wdeMethods + singleMethods + ['naive2']
    displayMethods = [m for m in displayMethods if m in allMethods]

    groups = list(groupResults.keys())

    P(f'\n{"Method":<20}', end='')
    for g in groups:
        P(f'{g:>12}', end='')
    P(f'{"AVG":>12}')
    P('-' * (20 + 12 * (len(groups) + 1)))

    for method in displayMethods:
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

    P('\n' + '-'*80)
    P('최적 alpha 탐색:')
    for g in groups:
        avgOwa, _ = groupResults[g]
        bestAlpha = None
        bestOwa = float('inf')
        for m, v in avgOwa.items():
            if m.startswith('wde_a') and v < bestOwa:
                bestOwa = v
                bestAlpha = m
        if bestAlpha:
            invMape = avgOwa.get('inv_mape', float('nan'))
            improvement = (invMape - bestOwa) / invMape * 100 if invMape > 0 else 0
            P(f'  {g}: best={bestAlpha} (OWA={bestOwa:.4f}), inv_mape={invMape:.4f}, '
              f'improvement={improvement:+.2f}%')

    P('\nWasserstein 거리 분석 (마지막 그룹):')
    lastGroup = groups[-1]
    _, wDataList = groupResults[lastGroup]
    if wDataList:
        allDists = {}
        for wd in wDataList:
            for pair, dist in wd['wDistances'].items():
                if pair not in allDists:
                    allDists[pair] = []
                allDists[pair].append(dist)
        P(f'  모델 쌍별 평균 Wasserstein 거리 ({lastGroup}):')
        for pair in sorted(allDists.keys()):
            P(f'    {pair}: {np.mean(allDists[pair]):.4f} (std={np.std(allDists[pair]):.4f})')

        allAcc = {}
        allDiv = {}
        for wd in wDataList:
            for nm, w in wd['accWeights'].items():
                if nm not in allAcc:
                    allAcc[nm] = []
                allAcc[nm].append(w)
            for nm, w in wd['divWeights'].items():
                if nm not in allDiv:
                    allDiv[nm] = []
                allDiv[nm].append(w)
        P(f'\n  평균 가중치 비교 ({lastGroup}):')
        P(f'  {"Model":<12} {"Accuracy":>12} {"Diversity":>12} {"Diff":>12}')
        for nm in sorted(allAcc.keys()):
            aW = np.mean(allAcc[nm])
            dW = np.mean(allDiv.get(nm, [0]))
            P(f'  {nm:<12} {aW:>12.4f} {dW:>12.4f} {dW-aW:>+12.4f}')


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', line_buffering=True)

    P('='*80)
    P('Wasserstein Diversity Ensemble (WDE) — modelCreation/013')
    P('='*80)

    groupResults = {}
    totalStart = time.time()

    runGroups = ['Yearly', 'Quarterly', 'Monthly', 'Hourly']
    groupSamples = {
        'Yearly': 500, 'Quarterly': 500, 'Monthly': 500, 'Hourly': 200,
    }
    for groupName in runGroups:
        nSample = groupSamples.get(groupName, SAMPLE_PER_GROUP)
        avgOwa, wData = _runGroup(groupName, nSample)
        groupResults[groupName] = (avgOwa, wData)

    _printResults(groupResults)

    totalElapsed = time.time() - totalStart
    P(f'\nTotal time: {totalElapsed/60:.1f} min')
