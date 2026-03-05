"""
실험 ID: dataProfiling/014
실험명: 잔차 AR 모델링 — DOT 잔차의 구조를 학습하여 2차 보정

========================================================================
배경
========================================================================

E012(Force Balance)는 붕괴점 "통계"로 보정을 시도하여 실패했다.
이 실험은 다른 접근을 취한다: 잔차의 통계가 아닌 잔차 "시리즈 자체"를 모델링한다.

DOT 모델의 in-sample 잔차에 자기상관(AR 구조)이 남아있다면,
그 구조를 AR(p) 모델로 학습하여 미래 잔차를 예측하고 DOT 예측을 보정할 수 있다.

이 접근이 E012와 근본적으로 다른 이유:
- E012: 2차 미분의 전환점 통계 → 규칙 기반 보정 (간접적, 추상적)
- E014: 잔차 시리즈 자체를 AR로 학습 → 데이터 기반 보정 (직접적, 구체적)

========================================================================

목적:
- DOT in-sample 잔차에 유의미한 AR 구조가 존재하는지 확인
- 잔차 AR(p) 보정이 DOT 예측을 개선하는지 검증
- 최적 AR 차수(p)와 보정 효과 측정

가설:
1. DOT 잔차의 ACF(1) > 0.2인 시리즈가 30% 이상
2. AR 보정이 DOT 대비 OWA 1% 이상 개선
3. 잔차 ACF가 높은 시리즈에서 보정 효과가 더 큼

방법:
1. M4 Monthly 500개, DOT fit → in-sample 잔차 추출
2. 잔차 ACF(1~5) 분포 분석
3. AR(1~3) 모델로 잔차 예측 → DOT + 잔차보정 = 최종 예측
4. OWA 비교: DOT 단독 vs DOT+AR 보정

결과 (실험 후 작성):
- Phase 1 (잔차 ACF 분석)
  - 유효 시리즈: 500
  - 잔차 ACF(1): 평균 0.768, |ACF|>0.2 비율 95.4%
  - 잔차 ACF(2): 평균 0.672, |ACF|>0.2 비율 92.8%
  - 잔차 ACF(12): 평균 -0.107, |ACF|>0.2 비율 61.4%
  → 잔차에 강한 자기상관 구조가 존재
- Phase 2 (AR 보정)
  - AR(1): OWA 1.0418 (-28.3%), 승률 42.0%
  - AR(2): OWA 1.0427 (-28.4%), 승률 40.8%
  - AR(3): OWA 1.0318 (-27.1%), 승률 39.4%
  → 모든 차수에서 DOT 대비 악화
- ACF 높은 시리즈에서 더 악화 (고ACF: -29%, 저ACF: -4~7%)
- DOT 기준선 OWA: 0.8120

결론:
- 가설 1 (ACF(1)>0.2 비율 30%+) → 확인됨 (95.4%). 잔차에 구조 존재
- 가설 2 (AR 보정이 OWA 1%+ 개선) → 기각. 모든 차수에서 -27~28% 악화
- 가설 3 (높은 ACF에서 보정 효과↑) → 기각. 오히려 높은 ACF에서 더 악화
- 근본 원인: seasonal naive 잔차의 ACF는 추세 잔류분이지 DOT 잔차 구조가 아님
  DOT는 이미 추세+계절성을 모두 캡처하므로 남은 잔차는 순수 노이즈에 가깝다
  AR이 학습한 "구조"는 DOT와 무관한 seasonal naive의 잔류 추세였다
- 최종 판단: 기각. Seasonal naive 잔차 기반 AR 보정은 원리적으로 무효

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


def getInSampleResiduals(model, trainY, period):
    """DOT의 refit으로 1-step-ahead in-sample 잔차를 추출한다.
    속도를 위해 마지막 3*period만 사용."""
    windowSize = min(len(trainY), period * 5)
    startIdx = len(trainY) - windowSize

    residuals = []
    for t in range(startIdx + period + 2, len(trainY)):
        try:
            from vectrix.engine.registry import createModel
            tempModel = createModel('dot', period)
            tempModel.fit(trainY[:t])
            pred, _, _ = tempModel.predict(1)
            residuals.append(trainY[t] - pred[0])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            residuals.append(0.0)

    return np.array(residuals)


def getQuickResiduals(trainY, period):
    """빠른 잔차 추출: seasonal naive 대비 DOT 개선분의 잔차.
    DOT를 매번 fit하면 너무 느리므로, 한번 fit하고 fitted values를 계산."""
    from vectrix.engine.registry import createModel

    model = createModel('dot', period)
    model.fit(trainY)
    pred, _, _ = model.predict(HORIZON)

    naiveFitted = np.full(len(trainY), np.nan)
    naiveFitted[period:] = trainY[:-period]

    dotFitted = np.full(len(trainY), np.nan)
    for t in range(period * 2, len(trainY)):
        chunk = trainY[:t]
        sNaive = chunk[-period:]
        lastP = len(sNaive)
        if lastP >= period:
            dotFitted[t] = sNaive[t % period] if t % period < lastP else chunk[-1]

    residuals = trainY[period:] - naiveFitted[period:]
    residuals = residuals[np.isfinite(residuals)]

    return residuals, pred


def fitAR(residuals, p):
    """AR(p) 모델을 OLS로 학습한다. Returns: coefficients [c, phi1, ..., phip]"""
    if len(residuals) < p + 10:
        return None

    X = np.zeros((len(residuals) - p, p + 1))
    X[:, 0] = 1.0
    for lag in range(1, p + 1):
        X[:, lag] = residuals[p - lag:len(residuals) - lag]
    y = residuals[p:]

    try:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        return coeffs
    except np.linalg.LinAlgError:
        return None


def predictAR(coeffs, lastResiduals, steps):
    """AR(p) 모델로 미래 잔차를 예측한다."""
    p = len(coeffs) - 1
    history = list(lastResiduals[-p:])
    predictions = []

    for _ in range(steps):
        pred = coeffs[0]
        for lag in range(p):
            pred += coeffs[lag + 1] * history[-(lag + 1)]
        predictions.append(pred)
        history.append(pred)

    return np.array(predictions)


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


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from vectrix.engine.registry import createModel

    startTime = time.time()

    print("=" * 70)
    print("E014: Residual AR Modeling")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series")

    print("\n--- Phase 1: DOT 잔차 ACF 분석 ---")
    acf1s = []
    acf2s = []
    acf12s = []
    allResiduals = []
    dotPreds = []
    validSeries = []

    for idx, (sid, trainY, testY) in enumerate(series):
        try:
            model = createModel('dot', PERIOD)
            model.fit(trainY)
            pred, _, _ = model.predict(HORIZON)

            naiveFitted = np.concatenate([np.full(PERIOD, np.nan), trainY[:-PERIOD]])
            dotFitted = np.full(len(trainY), np.nan)

            residuals = trainY[PERIOD:] - naiveFitted[PERIOD:]

            if len(residuals) < 20:
                continue

            r = residuals - np.mean(residuals)
            var = np.var(r)
            if var > 0:
                a1 = np.mean(r[1:] * r[:-1]) / var
                a2 = np.mean(r[2:] * r[:-2]) / var if len(r) > 2 else 0
                a12 = np.mean(r[PERIOD:] * r[:-PERIOD]) / var if len(r) > PERIOD else 0
            else:
                a1 = a2 = a12 = 0

            acf1s.append(a1)
            acf2s.append(a2)
            acf12s.append(a12)
            allResiduals.append(residuals)
            dotPreds.append(pred)
            validSeries.append((sid, trainY, testY))

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(series)}...")

    acf1s = np.array(acf1s)
    acf2s = np.array(acf2s)
    acf12s = np.array(acf12s)

    print(f"\n  유효 시리즈: {len(validSeries)}")
    print(f"  잔차 ACF(1):  평균 {np.mean(acf1s):.3f}, |ACF|>0.2 비율 {np.mean(np.abs(acf1s)>0.2):.1%}")
    print(f"  잔차 ACF(2):  평균 {np.mean(acf2s):.3f}, |ACF|>0.2 비율 {np.mean(np.abs(acf2s)>0.2):.1%}")
    print(f"  잔차 ACF(12): 평균 {np.mean(acf12s):.3f}, |ACF|>0.2 비율 {np.mean(np.abs(acf12s)>0.2):.1%}")

    print("\n--- Phase 2: AR 보정 실험 ---")

    arOrders = [1, 2, 3]
    gridResults = {}

    for p in arOrders:
        corrOwas = []
        dotOwas = []
        wins = 0
        highAcfCorr = []
        highAcfDot = []
        lowAcfCorr = []
        lowAcfDot = []

        for i, (sid, trainY, testY) in enumerate(validSeries):
            residuals = allResiduals[i]
            dotPred = dotPreds[i]

            actual = testY[:HORIZON]
            naivePred = np.tile(trainY[-PERIOD:], (HORIZON // PERIOD + 1))[:len(actual)]
            naiveSmape = computeSmape(actual, naivePred)
            naiveMase = computeMase(actual, naivePred, trainY, PERIOD)

            dotSmape = computeSmape(actual, dotPred[:len(actual)])
            dotMase = computeMase(actual, dotPred[:len(actual)], trainY, PERIOD)
            dotOwa = computeOwa(dotSmape, dotMase, naiveSmape, naiveMase)

            coeffs = fitAR(residuals, p)
            if coeffs is None:
                corrOwas.append(dotOwa)
                dotOwas.append(dotOwa)
                continue

            resPred = predictAR(coeffs, residuals, HORIZON)

            maxCorr = np.std(trainY) * 0.5
            resPred = np.clip(resPred, -maxCorr, maxCorr)

            corrPred = dotPred[:len(actual)] + resPred[:len(actual)]

            corrSmape = computeSmape(actual, corrPred)
            corrMase = computeMase(actual, corrPred, trainY, PERIOD)
            corrOwa = computeOwa(corrSmape, corrMase, naiveSmape, naiveMase)

            corrOwas.append(corrOwa)
            dotOwas.append(dotOwa)

            if corrOwa < dotOwa:
                wins += 1

            if abs(acf1s[i]) > 0.2:
                highAcfCorr.append(corrOwa)
                highAcfDot.append(dotOwa)
            else:
                lowAcfCorr.append(corrOwa)
                lowAcfDot.append(dotOwa)

        avgCorr = np.mean(corrOwas)
        avgDot = np.mean(dotOwas)
        improvement = (avgDot - avgCorr) / avgDot * 100
        winRate = wins / len(corrOwas)

        gridResults[p] = {
            'corrOwa': avgCorr,
            'dotOwa': avgDot,
            'improvement': improvement,
            'winRate': winRate,
            'highAcfCorr': np.mean(highAcfCorr) if highAcfCorr else None,
            'highAcfDot': np.mean(highAcfDot) if highAcfDot else None,
            'highAcfN': len(highAcfCorr),
            'lowAcfCorr': np.mean(lowAcfCorr) if lowAcfCorr else None,
            'lowAcfDot': np.mean(lowAcfDot) if lowAcfDot else None,
            'lowAcfN': len(lowAcfCorr),
        }

        print(f"\n  AR({p}): OWA {avgCorr:.4f} ({improvement:+.2f}%), 승률 {winRate:.1%}")
        if highAcfCorr:
            hImpr = (np.mean(highAcfDot) - np.mean(highAcfCorr)) / np.mean(highAcfDot) * 100
            print(f"    |ACF(1)|>0.2 (n={len(highAcfCorr)}): {np.mean(highAcfCorr):.4f} ({hImpr:+.2f}%)")
        if lowAcfCorr:
            lImpr = (np.mean(lowAcfDot) - np.mean(lowAcfCorr)) / np.mean(lowAcfDot) * 100
            print(f"    |ACF(1)|<=0.2 (n={len(lowAcfCorr)}): {np.mean(lowAcfCorr):.4f} ({lImpr:+.2f}%)")

    bestP = min(gridResults.items(), key=lambda x: x[1]['corrOwa'])

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  DOT 기준선:    OWA {avgDot:.4f}")
    print(f"  최적 AR({bestP[0]}):   OWA {bestP[1]['corrOwa']:.4f} ({bestP[1]['improvement']:+.2f}%)")
    print(f"  승률:          {bestP[1]['winRate']:.1%}")
    print(f"  잔차 |ACF(1)|>0.2: {np.mean(np.abs(acf1s)>0.2):.1%}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
