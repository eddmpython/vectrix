"""
실험 ID: dataProfiling/015
실험명: Rolling Holdout Competition — 다중 검증으로 상관 개선

========================================================================
배경
========================================================================

E013에서 holdout 1-period 선택의 상관이 r=0.267으로 너무 약했다.
이 실험은 rolling window(여러 시점의 holdout)로 평균하여 상관을 개선한다.

핵심 아이디어:
- holdout 1회의 결과는 노이즈가 크다
- 여러 시점에서 반복 평가하면 노이즈가 평균화되어 진짜 성능이 드러난다
- 이건 cross-validation의 시계열 버전이다

========================================================================

목적:
- Rolling holdout(2~4회)가 1회 대비 선택 정확도를 개선하는지
- Holdout↔Test 상관이 r>0.5로 올라가는지
- 실질 OWA 개선이 발생하는지

가설:
1. Rolling 3-fold holdout↔test 상관이 E013(r=0.267)보다 높다
2. Rolling 승자가 DOT 대비 OWA를 개선한다
3. diff 전처리의 과적합이 rolling에서 걸러진다

방법:
1. M4 Monthly 500개 (seed=42)
2. Rolling: 마지막 2~4 period를 각각 holdout으로 사용하여 평균 OWA
3. 조합: 3모델 × 2전처리(raw, log) — diff 제외/포함 비교
4. Rolling 평균 OWA로 최적 조합 선택 → test 예측

결과 (실험 후 작성):
- 유효 시리즈: 413 (MIN_LEN=72로 500에서 감소)
- DOT-raw 기준선: OWA 0.8086

| Config           | Holdout OWA | DOT 대비  | H↔T 상관 | Gap 캡처율 | Oracle 일치 |
|------------------|-------------|-----------|----------|-----------|-------------|
| 1-fold safe      | 0.8136      | -0.62%    | r=0.285  | -3.9%     | 27.4%       |
| 2-fold safe      | 0.8008      | +0.96%    | r=0.299  | 6.1%      | 25.7%       |
| 3-fold safe      | 0.8015      | +0.88%    | r=0.299  | 5.6%      | 25.2%       |
| 3-fold all(+diff)| 0.8411      | -4.02%    | r=0.357  | -15.4%    | 17.4%       |

- Oracle OWA: 0.6820 (+15.66%), 3-fold+diff Oracle: 0.5975 (+26.11%)

결론:
- 가설 1 (Rolling↔test 상관 > E013) → 부분 확인. r=0.285→0.299 (미미한 개선)
- 가설 2 (Rolling 승자가 DOT 대비 개선) → 확인! 2-fold safe가 +0.96%
- 가설 3 (diff 과적합이 rolling에서 걸러짐) → 기각. 3-fold에서도 diff 포함 시 -4.02% 악화
- Rolling 2-fold가 최적. 3-fold는 추가 이득 없고 비용만 증가
- 상관 자체는 여전히 낮지만 (r=0.299), 평균 효과로는 양의 개선 달성
- 하지만 +0.96%는 실용적으로 미미. Oracle gap(15.66%)의 6.1%만 캡처
- diff 전처리는 holdout에서 체계적으로 과적합 — 어떤 fold 수에서도 해결 불가
- 최종 판단: 부분 성공. Rolling 2-fold safe는 DOT보다 약간 낫지만, 복잡도 대비 이득이 미미

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

M4_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')
SEED = 42
PERIOD = 12
HORIZON = 18
MIN_LEN = 72
N_SAMPLE = 500

MODELS = ['dot', 'auto_ces', 'four_theta']
PREPROCESS_SAFE = ['raw', 'log']
PREPROCESS_ALL = ['raw', 'log', 'diff']


def loadM4Monthly(n=N_SAMPLE):
    trainDf = pd.read_csv(os.path.join(M4_DIR, 'Monthly-train.csv'))
    testDf = pd.read_csv(os.path.join(M4_DIR, 'Monthly-test.csv'))
    np.random.seed(SEED)
    ids = np.random.choice(trainDf['V1'].values, size=n, replace=False)
    series = []
    for sid in ids:
        trainRow = trainDf[trainDf['V1'] == sid].iloc[0, 1:].dropna().values.astype(float)
        testRow = testDf[testDf['V1'] == sid].iloc[0, 1:].dropna().values.astype(float)
        if len(trainRow) >= MIN_LEN:
            series.append((sid, trainRow, testRow))
    return series


def applyPreprocess(y, method):
    if method == 'raw':
        return y, lambda pred: pred
    if method == 'log':
        minVal = np.min(y)
        shift = abs(minVal) + 1.0 if minVal <= 0 else 0.0
        return np.log(y + shift), lambda pred: np.exp(pred) - shift
    if method == 'diff':
        lastVal = y[-1]
        def inverseDiff(pred):
            result = np.zeros(len(pred))
            result[0] = lastVal + pred[0]
            for i in range(1, len(pred)):
                result[i] = result[i-1] + pred[i]
            return result
        return np.diff(y), inverseDiff
    return y, lambda pred: pred


def fitPredict(modelId, y, period, steps):
    from vectrix.engine.registry import createModel
    try:
        model = createModel(modelId, period)
        model.fit(y)
        pred, _, _ = model.predict(steps)
        if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
            return None
        if np.max(np.abs(pred)) > np.max(np.abs(y)) * 100:
            return None
        return pred
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return None


def computeSmape(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted))
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    return np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def computeMase(actual, predicted, trainY, period):
    naiveErrors = np.abs(trainY[period:] - trainY[:-period])
    naiveMae = np.mean(naiveErrors)
    if naiveMae == 0:
        return 0.0
    return np.mean(np.abs(actual - predicted)) / naiveMae


def computeOwa(smape, mase, naiveSmape, naiveMase):
    if naiveSmape == 0 or naiveMase == 0:
        return 1.0
    return 0.5 * (smape / naiveSmape + mase / naiveMase)


def naiveSeasonalPred(trainY, period, steps):
    return np.tile(trainY[-period:], (steps // period + 1))[:steps]


def rollingHoldoutEval(trainY, modelId, preproc, nFolds, holdoutSize=PERIOD):
    """Rolling holdout으로 평균 OWA를 계산한다."""
    owas = []

    for fold in range(nFolds):
        endIdx = len(trainY) - fold * holdoutSize
        startHoldout = endIdx - holdoutSize

        if startHoldout < PERIOD * 2 + 10:
            continue

        trainPart = trainY[:startHoldout]
        holdout = trainY[startHoldout:endIdx]

        procTrain, invFunc = applyPreprocess(trainPart, preproc)
        if len(procTrain) < PERIOD + 5:
            continue

        pred = fitPredict(modelId, procTrain, PERIOD, holdoutSize)
        if pred is None:
            continue
        predOrig = invFunc(pred[:holdoutSize])

        naivePred = naiveSeasonalPred(trainPart, PERIOD, holdoutSize)
        naiveSmape = computeSmape(holdout, naivePred)
        naiveMase = computeMase(holdout, naivePred, trainPart, PERIOD)

        predSmape = computeSmape(holdout, predOrig)
        predMase = computeMase(holdout, predOrig, trainPart, PERIOD)
        owa = computeOwa(predSmape, predMase, naiveSmape, naiveMase)

        owas.append(owa)

    return np.mean(owas) if owas else None


def runRollingCompetition(trainY, testY, nFolds, preprocList):
    """Rolling holdout 경쟁."""
    from vectrix.engine.registry import createModel

    candidates = {}

    for modelId in MODELS:
        for preproc in preprocList:
            key = f"{modelId}_{preproc}"
            holdoutOwa = rollingHoldoutEval(trainY, modelId, preproc, nFolds)
            if holdoutOwa is None:
                continue

            procTrainFull, invFuncFull = applyPreprocess(trainY, preproc)
            testPred = fitPredict(modelId, procTrainFull, PERIOD, HORIZON)
            if testPred is None:
                continue

            actual = testY[:HORIZON]
            testPredOrig = invFuncFull(testPred[:len(actual)])

            naivePred = naiveSeasonalPred(trainY, PERIOD, len(actual))
            naiveSmape = computeSmape(actual, naivePred)
            naiveMase = computeMase(actual, naivePred, trainY, PERIOD)

            tSmape = computeSmape(actual, testPredOrig)
            tMase = computeMase(actual, testPredOrig, trainY, PERIOD)
            tOwa = computeOwa(tSmape, tMase, naiveSmape, naiveMase)

            candidates[key] = {
                'holdoutOwa': holdoutOwa,
                'testOwa': tOwa,
            }

    if not candidates:
        return None

    dotRaw = candidates.get('dot_raw')
    holdoutBest = min(candidates.items(), key=lambda x: x[1]['holdoutOwa'])
    testBest = min(candidates.items(), key=lambda x: x[1]['testOwa'])

    return {
        'candidates': candidates,
        'holdoutBestKey': holdoutBest[0],
        'holdoutBestOwa': holdoutBest[1]['testOwa'],
        'oracleBestKey': testBest[0],
        'oracleBestOwa': testBest[1]['testOwa'],
        'dotRawOwa': dotRaw['testOwa'] if dotRaw else None,
    }


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    startTime = time.time()

    print("=" * 70)
    print("E015: Rolling Holdout Competition")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series (MIN_LEN={MIN_LEN})")

    from scipy.stats import spearmanr

    configs = [
        ('1-fold safe(no diff)', 1, PREPROCESS_SAFE),
        ('2-fold safe', 2, PREPROCESS_SAFE),
        ('3-fold safe', 3, PREPROCESS_SAFE),
        ('3-fold all(+diff)', 3, PREPROCESS_ALL),
    ]

    for configName, nFolds, preprocList in configs:
        print(f"\n{'='*70}")
        print(f"  Config: {configName}")
        print(f"  {len(MODELS)} models × {len(preprocList)} preprocess = {len(MODELS)*len(preprocList)} candidates")
        print(f"{'='*70}")

        allResults = []
        for idx, (sid, trainY, testY) in enumerate(series):
            result = runRollingCompetition(trainY, testY, nFolds, preprocList)
            if result is not None:
                allResults.append(result)
            if (idx + 1) % 100 == 0:
                elapsed = time.time() - startTime
                print(f"    {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

        n = len(allResults)
        if n == 0:
            print("  결과 없음")
            continue

        dotOwas = [r['dotRawOwa'] for r in allResults if r['dotRawOwa'] is not None]
        holdoutOwas = [r['holdoutBestOwa'] for r in allResults]
        oracleOwas = [r['oracleBestOwa'] for r in allResults]

        avgDot = np.mean(dotOwas)
        avgHoldout = np.mean(holdoutOwas)
        avgOracle = np.mean(oracleOwas)

        wins = sum(1 for r in allResults if r['dotRawOwa'] is not None and r['holdoutBestOwa'] < r['dotRawOwa'])
        matchOracle = sum(1 for r in allResults if r['holdoutBestKey'] == r['oracleBestKey'])

        perSeriesCorrs = []
        for r in allResults:
            if len(r['candidates']) >= 3:
                hList = [c['holdoutOwa'] for c in r['candidates'].values()]
                tList = [c['testOwa'] for c in r['candidates'].values()]
                if np.std(hList) > 0 and np.std(tList) > 0:
                    c, _ = spearmanr(hList, tList)
                    if np.isfinite(c):
                        perSeriesCorrs.append(c)

        avgCorr = np.mean(perSeriesCorrs) if perSeriesCorrs else 0

        oracleGap = avgDot - avgOracle
        holdoutImpr = (avgDot - avgHoldout) / avgDot * 100
        captureRate = (avgDot - avgHoldout) / oracleGap * 100 if oracleGap > 0 else 0

        print(f"\n  DOT-raw:        OWA {avgDot:.4f}")
        print(f"  Holdout 승자:   OWA {avgHoldout:.4f} ({holdoutImpr:+.2f}%)")
        print(f"  Oracle:         OWA {avgOracle:.4f} ({(avgDot-avgOracle)/avgDot*100:+.2f}%)")
        print(f"  Gap 캡처율:     {captureRate:.1f}%")
        print(f"  승률:           {wins}/{n} ({wins/n:.1%})")
        print(f"  Oracle 일치율:  {matchOracle}/{n} ({matchOracle/n:.1%})")
        print(f"  H↔T 상관:      r={avgCorr:.3f}")

    elapsed = time.time() - startTime
    print(f"\n\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
