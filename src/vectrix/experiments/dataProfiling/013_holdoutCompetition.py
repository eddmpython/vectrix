"""
실험 ID: dataProfiling/013
실험명: Holdout Dynamic Competition — 시도 → 평가 → 선택

========================================================================
배경
========================================================================

E001~E012의 결론: "관찰 → 판단 → 적용"은 시계열에서 작동하지 않는다.
DNA, Shape, 붕괴점 통계 등 어떤 사전 관찰 기반 표현도 전략 매핑에 실패했다.

유일하게 작동이 확인된 원리는 "블렌딩 > 선택" (E005~E007)과
"시도 → 평가" (holdout 기반 동적 검증)이다.

이 실험은 그 원리를 직접 구현한다:
1. 각 시리즈에서 마지막 1 period를 holdout으로 분리
2. 여러 (모델 × 전처리) 조합으로 holdout 예측
3. holdout OWA가 가장 좋은 조합을 선택
4. 선택된 조합으로 전체 데이터에서 최종 예측

이 접근의 핵심 질문:
- holdout 1 period의 성능이 미래 horizon의 성능과 상관이 있는가?
- oracle gap(-17.4%)의 몇 %를 실제로 캡처할 수 있는가?
- 속도 비용(N배 fit) 대비 정확도 개선이 가치 있는가?

========================================================================

목적:
- Holdout 동적 경쟁이 DOT 단독 대비 OWA를 개선하는지 검증
- 모델(DOT, CES, 4Theta) × 전처리(raw, log, diff) 조합 경쟁
- Holdout 성능과 실제 test 성능의 상관 분석
- Oracle gap 캡처율 측정

가설:
1. Holdout 승자가 DOT 단독보다 낫다 (OWA 개선 >1%)
2. Holdout OWA와 test OWA의 상관 r > 0.5
3. Oracle gap(-17.4%)의 20% 이상 캡처 (OWA 개선 >3.5%)
4. 전처리 변경이 모델 변경보다 효과적인 시리즈가 존재

방법:
1. M4 Monthly 500개 (seed=42)
2. Holdout = 마지막 12개월 (1 period)
3. 조합: 3모델(DOT, CES, 4Theta) × 3전처리(raw, log, diff) = 9가지
4. 각 조합으로 holdout 예측 → holdout OWA 측정
5. 최적 조합으로 전체 train 데이터에서 test 예측
6. 비교: DOT-raw(기준), holdout 승자, per-series oracle

결과 (실험 후 작성):
- DOT-raw 기준: OWA 0.8120
- Holdout 승자: OWA 1.0061 (-23.9% 악화!)
- Oracle (사후 최적): OWA 0.6022 (+25.8% 개선)
- Oracle gap 캡처율: -92.6% (악화 방향!)
- 승률: 39.0% (500개 중 195개에서만 DOT보다 나음)
- Holdout↔Test 상관: 시리즈 내 r=0.267 (약함), r>0.5 비율 42.6%
- Holdout=Oracle 일치율: 18.6% (9가지 중 맞는 확률)
- Safe(min of DOT, holdout): OWA 0.7342 (+9.6%) — 이론적 상한
- Holdout 승자 분포: auto_ces_diff 19.4% 최다. DOT 독점이 아님
- diff 전처리가 모든 모델에서 활발히 선택됨 (E009에서는 "폭발 위험" 판정)

결론:
- 가설 1 기각: Holdout 승자가 DOT보다 나쁘다 (OWA 1.006 vs 0.812)
- 가설 2 기각: Holdout↔Test 상관 r=0.267 < 0.5 목표
- 가설 3 기각: Oracle gap을 캡처하기는커녕 악화
- 가설 4 채택: diff 전처리가 다수 시리즈에서 oracle 최적 (auto_ces_diff 18.6%)
- 핵심 교훈:
  (1) Holdout 1 period로는 미래 성능을 예측할 수 없다 (r=0.267)
  (2) diff 전처리는 holdout에서 과적합한다 — holdout에서 좋지만 test에서 폭발
  (3) Safe 전략 0.734의 의미: 39%의 시리즈에서 holdout이 진짜 좋은 조합을 찾는다
      → 문제는 "이 시리즈가 39%에 속하는지" 사전에 모른다는 것
  (4) 18.6%의 조합 일치율은 9가지 중 랜덤(11.1%)보다 약간 높음
      → holdout이 완전 무작위는 아니지만 신뢰하기엔 부족

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
MIN_LEN = 48
N_SAMPLE = 500

MODELS = ['dot', 'auto_ces', 'four_theta']
PREPROCESS = ['raw', 'log', 'diff']


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
    """전처리 적용. 역변환 함수도 반환한다."""
    if method == 'raw':
        return y, lambda pred: pred

    if method == 'log':
        minVal = np.min(y)
        if minVal <= 0:
            shift = abs(minVal) + 1.0
        else:
            shift = 0.0
        transformed = np.log(y + shift)
        return transformed, lambda pred: np.exp(pred) - shift

    if method == 'diff':
        lastVal = y[-1]
        transformed = np.diff(y)
        def inverseDiff(pred):
            result = np.zeros(len(pred))
            result[0] = lastVal + pred[0]
            for i in range(1, len(pred)):
                result[i] = result[i-1] + pred[i]
            return result
        return transformed, inverseDiff

    return y, lambda pred: pred


def fitPredict(modelId, y, period, steps):
    """모델 생성 → 학습 → 예측. 실패 시 None."""
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


def runCompetition(trainY, testY, holdoutSize=PERIOD):
    """한 시리즈에서 holdout 경쟁을 실행한다.

    Returns: dict with all results, or None if failed
    """
    if len(trainY) < holdoutSize + PERIOD + 10:
        return None

    trainPart = trainY[:-holdoutSize]
    holdout = trainY[-holdoutSize:]

    actual = testY[:HORIZON]
    naivePred = naiveSeasonalPred(trainY, PERIOD, len(actual))
    naiveSmape = computeSmape(actual, naivePred)
    naiveMase = computeMase(actual, naivePred, trainY, PERIOD)

    holdoutNaive = naiveSeasonalPred(trainPart, PERIOD, holdoutSize)
    holdoutNaiveSmape = computeSmape(holdout, holdoutNaive)
    holdoutNaiveMase = computeMase(holdout, holdoutNaive, trainPart, PERIOD)

    candidates = {}

    for modelId in MODELS:
        for preproc in PREPROCESS:
            key = f"{modelId}_{preproc}"

            procTrain, invFunc = applyPreprocess(trainPart, preproc)
            if len(procTrain) < PERIOD + 5:
                continue

            period = PERIOD if preproc != 'diff' else PERIOD

            holdPred = fitPredict(modelId, procTrain, period, holdoutSize)
            if holdPred is None:
                continue
            holdPredOrig = invFunc(holdPred[:holdoutSize])

            hSmape = computeSmape(holdout, holdPredOrig)
            hMase = computeMase(holdout, holdPredOrig, trainPart, PERIOD)
            hOwa = computeOwa(hSmape, hMase, holdoutNaiveSmape, holdoutNaiveMase)

            procTrainFull, invFuncFull = applyPreprocess(trainY, preproc)
            testPred = fitPredict(modelId, procTrainFull, period, HORIZON)
            if testPred is None:
                continue
            testPredOrig = invFuncFull(testPred[:len(actual)])

            tSmape = computeSmape(actual, testPredOrig)
            tMase = computeMase(actual, testPredOrig, trainY, PERIOD)
            tOwa = computeOwa(tSmape, tMase, naiveSmape, naiveMase)

            candidates[key] = {
                'holdoutOwa': hOwa,
                'testOwa': tOwa,
                'modelId': modelId,
                'preproc': preproc,
            }

    if not candidates:
        return None

    dotRawKey = 'dot_raw'
    dotRawTestOwa = candidates[dotRawKey]['testOwa'] if dotRawKey in candidates else None

    holdoutBest = min(candidates.items(), key=lambda x: x[1]['holdoutOwa'])
    testBest = min(candidates.items(), key=lambda x: x[1]['testOwa'])

    return {
        'candidates': candidates,
        'holdoutBestKey': holdoutBest[0],
        'holdoutBestOwa': holdoutBest[1]['testOwa'],
        'holdoutBestHoldoutOwa': holdoutBest[1]['holdoutOwa'],
        'oracleBestKey': testBest[0],
        'oracleBestOwa': testBest[1]['testOwa'],
        'dotRawOwa': dotRawTestOwa,
        'naiveSmape': naiveSmape,
        'naiveMase': naiveMase,
    }


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    startTime = time.time()

    print("=" * 70)
    print("E013: Holdout Dynamic Competition")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series (MIN_LEN={MIN_LEN})")

    print(f"\n조합: {len(MODELS)} models × {len(PREPROCESS)} preprocess = {len(MODELS)*len(PREPROCESS)} candidates")
    print(f"  Models: {MODELS}")
    print(f"  Preprocess: {PREPROCESS}")

    print("\n--- Running holdout competition ---")
    allResults = []
    failCount = 0

    for idx, (sid, trainY, testY) in enumerate(series):
        result = runCompetition(trainY, testY)
        if result is None:
            failCount += 1
            continue
        allResults.append(result)

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - startTime
            print(f"  {idx + 1}/{len(series)}... ({elapsed:.0f}s)")

    n = len(allResults)
    print(f"\n  완료: {n} 시리즈 (실패: {failCount})")

    print("\n" + "=" * 70)
    print("=== Phase 1: 기준선 비교 ===")
    print("=" * 70)

    dotOwas = [r['dotRawOwa'] for r in allResults if r['dotRawOwa'] is not None]
    holdoutOwas = [r['holdoutBestOwa'] for r in allResults]
    oracleOwas = [r['oracleBestOwa'] for r in allResults]

    avgDot = np.mean(dotOwas)
    avgHoldout = np.mean(holdoutOwas)
    avgOracle = np.mean(oracleOwas)

    print(f"  DOT-raw (기준):      OWA {avgDot:.4f}")
    print(f"  Holdout 승자:        OWA {avgHoldout:.4f} ({(avgDot-avgHoldout)/avgDot*100:+.2f}%)")
    print(f"  Per-series Oracle:   OWA {avgOracle:.4f} ({(avgDot-avgOracle)/avgDot*100:+.2f}%)")

    oracleGap = avgDot - avgOracle
    holdoutCapture = (avgDot - avgHoldout) / oracleGap * 100 if oracleGap > 0 else 0
    print(f"  Oracle gap 캡처율:   {holdoutCapture:.1f}%")

    wins = sum(1 for r in allResults if r['dotRawOwa'] is not None and r['holdoutBestOwa'] < r['dotRawOwa'])
    ties = sum(1 for r in allResults if r['dotRawOwa'] is not None and abs(r['holdoutBestOwa'] - r['dotRawOwa']) < 0.001)
    print(f"  승률 (holdout > DOT): {wins}/{n} ({wins/n:.1%})")

    print("\n" + "=" * 70)
    print("=== Phase 2: Holdout↔Test 상관 ===")
    print("=" * 70)

    from scipy.stats import spearmanr

    holdoutOwaList = []
    testOwaList = []
    for r in allResults:
        for key, cand in r['candidates'].items():
            holdoutOwaList.append(cand['holdoutOwa'])
            testOwaList.append(cand['testOwa'])

    corr, pval = spearmanr(holdoutOwaList, testOwaList)
    print(f"  전체 (조합 수준): r={corr:.3f} (p={pval:.2e}, n={len(holdoutOwaList)})")

    perSeriesCorrs = []
    for r in allResults:
        if len(r['candidates']) >= 3:
            hList = [c['holdoutOwa'] for c in r['candidates'].values()]
            tList = [c['testOwa'] for c in r['candidates'].values()]
            if np.std(hList) > 0 and np.std(tList) > 0:
                c, _ = spearmanr(hList, tList)
                if np.isfinite(c):
                    perSeriesCorrs.append(c)

    perSeriesCorrs = np.array(perSeriesCorrs)
    print(f"  시리즈 내 상관 (조합 간 순위): 평균 r={np.mean(perSeriesCorrs):.3f}, 중앙값 {np.median(perSeriesCorrs):.3f}")
    print(f"  r > 0.5인 비율: {np.mean(perSeriesCorrs > 0.5):.1%}")
    print(f"  r > 0인 비율: {np.mean(perSeriesCorrs > 0):.1%}")

    print("\n" + "=" * 70)
    print("=== Phase 3: Holdout 승자의 조합 분포 ===")
    print("=" * 70)

    from collections import Counter
    holdoutWinners = [r['holdoutBestKey'] for r in allResults]
    oracleWinners = [r['oracleBestKey'] for r in allResults]

    print(f"\n  Holdout 승자 분포:")
    for key, count in Counter(holdoutWinners).most_common():
        print(f"    {key:25s}: {count:4d} ({count/n:.1%})")

    print(f"\n  Oracle 승자 분포:")
    for key, count in Counter(oracleWinners).most_common():
        print(f"    {key:25s}: {count:4d} ({count/n:.1%})")

    holdoutMatchOracle = sum(1 for r in allResults if r['holdoutBestKey'] == r['oracleBestKey'])
    print(f"\n  Holdout = Oracle 일치율: {holdoutMatchOracle}/{n} ({holdoutMatchOracle/n:.1%})")

    print("\n" + "=" * 70)
    print("=== Phase 4: 모델 vs 전처리 효과 분리 ===")
    print("=" * 70)

    modelWins = Counter()
    preprocWins = Counter()
    for r in allResults:
        best = r['holdoutBestKey']
        parts = best.split('_', 1)
        if len(parts) == 2:
            modelWins[parts[0]] += 1
            preprocWins[parts[1]] += 1

    print(f"  모델별 승률:")
    for model in MODELS:
        cnt = modelWins.get(model, 0)
        print(f"    {model:15s}: {cnt:4d} ({cnt/n:.1%})")

    print(f"  전처리별 승률:")
    for pp in PREPROCESS:
        cnt = preprocWins.get(pp, 0)
        print(f"    {pp:15s}: {cnt:4d} ({cnt/n:.1%})")

    preprocChangedWins = sum(1 for r in allResults
                            if r['holdoutBestKey'] != 'dot_raw'
                            and r['holdoutBestKey'].endswith('_raw') == False
                            and r['holdoutBestOwa'] < (r['dotRawOwa'] or 999))
    print(f"\n  전처리 변경으로 DOT보다 개선된 시리즈: {preprocChangedWins}/{n} ({preprocChangedWins/n:.1%})")

    print("\n" + "=" * 70)
    print("=== Phase 5: 안전한 전략 비교 ===")
    print("=" * 70)

    safeOwas = []
    for r in allResults:
        dotOwa = r['dotRawOwa']
        holdOwa = r['holdoutBestOwa']
        if dotOwa is not None:
            safeOwas.append(min(dotOwa, holdOwa))

    print(f"  DOT-raw:                    OWA {avgDot:.4f}")
    print(f"  Holdout 승자:               OWA {avgHoldout:.4f}")
    print(f"  Safe(min of DOT, holdout):  OWA {np.mean(safeOwas):.4f}")
    print(f"  Oracle:                     OWA {avgOracle:.4f}")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  DOT-raw:           OWA {avgDot:.4f}")
    print(f"  Holdout 승자:      OWA {avgHoldout:.4f} ({(avgDot-avgHoldout)/avgDot*100:+.2f}%)")
    print(f"  Oracle:            OWA {avgOracle:.4f} ({(avgDot-avgOracle)/avgDot*100:+.2f}%)")
    print(f"  Gap 캡처율:        {holdoutCapture:.1f}%")
    print(f"  승률:              {wins/n:.1%}")
    print(f"  Holdout↔Test 상관: r={np.mean(perSeriesCorrs):.3f}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
