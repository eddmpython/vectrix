"""
실험 ID: dataProfiling/012
실험명: Force Balance Model — 균형 붕괴 기반 예측 보정

========================================================================
이론적 배경
========================================================================

1. 관측 사실 (E011에서 확인)

   M4 Monthly 500개 시리즈의 2차 미분(가속도) 부호 전환점을 분석한 결과:

   (a) 유의미한 전환점은 시리즈 길이의 약 10% (평균 21.9개)
   (b) 전환점은 균일하게 분포하지 않고 몰려서 나타난다 (CV 1.314 > 랜덤 0.855)
       → "격동기"(연속 전환)와 "고요기"(긴 안정)가 교대
   (c) 전환 방향은 73.8% 확률로 교대 (상승→하강 다음 하강→상승)
       → 50%(랜덤)보다 유의미하게 높음
   (d) 전환 크기에 자기상관 존재 (ACF(1) = 0.187, 75.1%가 양)
       → 큰 전환 뒤에 큰 전환이 올 가능성이 높음

2. 핵심 아이디어

   기존 예측 모델(DOT, ETS, Theta 등)은 시계열을 Level + Trend + Seasonality로
   분해하고 각 성분을 매끄럽게 외삽한다. 이 접근의 근본적 한계는 "매끄럽지 않은
   변화"를 포착하지 못한다는 것이다.

   Force Balance Model은 시계열을 다르게 본다:

       y(t) = Inertia(t) + Force(t) + Noise(t)

   - Inertia(t): 관성. 현재 상태를 유지하려는 힘. Level과 Trend의 합.
                  기존 모델이 잘 포착하는 부분.
   - Force(t):   교란력. 현재 상태를 변화시키는 힘. 불연속적으로 작용.
                  기존 모델이 포착하지 못하는 부분. 잔차에 남는다.
   - Noise(t):   순수 노이즈. 예측 불가능.

   Force(t)는 대부분의 시간에 0이지만, "균형이 무너지는 순간"에 비영(非零)이 된다.
   이 순간이 E011에서 관측한 "유의미 가속도 전환점"이다.

3. Force(t)의 세 가지 통계적 특성

   E011 결과를 Force(t)의 언어로 재해석하면:

   특성 1 — Clustering (몰림)
     Force(t) ≠ 0인 시점이 몰려서 나타난다.
     → 교란이 한번 시작되면 연속적으로 발생. 마치 지진의 여진처럼.
     → 시사점: "최근 교란이 있었다면 추가 교란 가능성이 높다"

   특성 2 — Alternation (교대, 73.8%)
     Force(t)가 양(상승 교란)이었으면 다음 Force는 73.8% 확률로 음(하강 교란).
     → 시스템에 복원력이 존재. 한 방향으로 밀리면 되돌아오려 한다.
     → 시사점: "마지막 교란의 반대 방향으로 보정하면 유리하다"

   특성 3 — Persistence (지속, ACF(1) = 0.187)
     |Force(t)|와 |Force(t+1)|은 양의 상관.
     → 교란의 크기가 전파된다. 큰 교란 뒤에는 큰 교란이 온다.
     → 시사점: "보정의 크기는 최근 교란 크기에 비례해야 한다"

4. 보정 공식

   기존 모델의 예측값 ŷ(t+h)에 Force 보정항 F̂(t+h)를 더한다:

       ŷ_corrected(t+h) = ŷ_DOT(t+h) + F̂(t+h)

   F̂(t+h)의 추정:
       F̂(t+h) = sign × amplitude × decay^h

   각 항의 의미:
   - sign: 마지막 붕괴점의 반대 방향 (+1 또는 -1)
           교대율 73.8%에 기반한 확률적 판단
   - amplitude: 최근 K개 붕괴점 크기의 가중 평균
                자기상관에 기반 — 최근일수록 가중치 높음
   - decay: 감쇠율 (0 < decay < 1)
            horizon이 멀어질수록 보정 효과가 줄어든다
            격동기에는 decay가 높고 (보정 지속), 고요기에는 낮다 (빠른 소멸)

5. 검증 방법

   M4 Monthly 500개 시리즈에서:
   (a) DOT 단독 예측의 OWA (기준선)
   (b) DOT + Force 보정의 OWA (실험군)
   (c) 보정이 효과적인 시리즈의 특성 분석
   (d) decay, amplitude scaling 등 파라미터 민감도 분석

6. 이 이론이 틀릴 수 있는 지점

   (a) E011의 73.8% 교대율은 학습 데이터 "내부"에서 측정된 값.
       미래(horizon)에서도 유지된다는 보장이 없다.
   (b) 크기 자기상관 0.187은 약하다. 보정 크기가 노이즈에 묻힐 수 있다.
   (c) "격동기 감지"를 현재 시점에서 할 수 있는지 불확실.
       사후적으로만 보이는 것일 수 있다.
   (d) 보정을 적용하면 일부 시리즈에서 악화될 수 있다 (E005~E009의 교훈).

========================================================================

목적:
- Force Balance Model 이론을 M4 Monthly에서 실증 검증
- DOT 예측에 Force 보정을 적용하여 OWA가 개선되는지 확인
- 보정 파라미터(decay, amplitude, lookback)의 최적 범위 탐색
- 보정이 효과적인/비효과적인 시리즈의 특성 식별

가설:
1. Force 보정이 DOT 단독 대비 OWA를 개선한다 (>0.5% 개선)
2. 격동기에 있는 시리즈에서 보정 효과가 더 크다
3. 교대율이 높은 시리즈에서 보정 효과가 더 크다
4. decay = 0.5~0.8 범위가 최적이다 (너무 빠르면 효과 없음, 너무 느리면 누적 오차)

방법:
1. M4 Monthly 500개 (seed=42) — DOT 기준선 OWA 계산
2. 각 시리즈의 train 데이터에서 붕괴점 추출 + Force 특성 계산
3. DOT 예측 + Force 보정 적용 (파라미터 그리드 탐색)
4. OWA 비교: 전체, 격동기/고요기별, 교대율별
5. 최적 파라미터에서의 승률(보정이 개선하는 시리즈 비율) 확인

결과 (실험 후 작성):
- DOT 기준선 OWA: 0.8120
- 모든 파라미터 조합에서 악화. 최선(decay=0.3, lb=3) OWA 0.9055 (-11.5%)
- 최악(decay=0.95, lb=8) OWA 1.5613 (-92.3%)
- 승률 최대 23.8% (decay=0.85) — 76%의 시리즈에서 악화
- 격동기(n=93) 더 크게 악화 (OWA 1.074~3.050 vs DOT 기준)
- 고요기(n=407) 소폭 악화 (OWA 0.867~1.216)
- decay가 작을수록(빠른 감쇠) 덜 나쁨 → 보정 자체가 노이즈

결론:
- 가설 1 기각: Force 보정은 모든 파라미터에서 DOT를 악화시킨다
- 가설 2 기각: 격동기에서 보정이 "더 크게" 악화. 교란이 큰 시리즈에 큰 보정 = 재앙
- 가설 3~4 미검증: 전체가 악화하므로 세부 분석 무의미
- 핵심 교훈: E011의 통계적 특성(73.8% 교대, ACF 0.187)은 "사실"이지만,
  이를 "보정"으로 변환하면 노이즈만 추가된다. 이유:
  (1) 73.8%가 "다음 붕괴점"의 방향이지, "다음 시점"의 방향이 아니다
      — 붕괴점은 전체의 10%에서만 발생. 90%의 시점에서는 보정이 잘못됨
  (2) amplitude 추정의 불확실성이 너무 크다
      — 과거 크기로 미래 크기를 추정하는데 ACF 0.187은 너무 약하다
  (3) "언제" 다음 붕괴가 오는지 모른다
      — 방향과 크기를 알아도 타이밍을 모르면 보정할 수 없다
- Force Balance Model의 이론적 프레임워크는 유지하되,
  직접 보정이 아닌 다른 적용 방식을 모색해야 한다

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
MIN_LEN = 36
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


def extractBreakpoints(y, threshold='auto'):
    """2차 미분의 부호 전환점 중 유의미한 것만 추출한다."""
    if len(y) < 4:
        return []

    d1 = np.diff(y)
    d2 = np.diff(d1)

    allBps = []
    for i in range(1, len(d2)):
        if d2[i-1] * d2[i] < 0:
            mag = abs(d2[i] - d2[i-1])
            allBps.append({
                'pos': i + 1,
                'accelBefore': d2[i-1],
                'accelAfter': d2[i],
                'magnitude': mag,
                'direction': 'up_to_down' if d2[i-1] > 0 and d2[i] < 0 else 'down_to_up',
            })

    if not allBps:
        return []

    if threshold is None:
        return allBps

    mags = np.array([bp['magnitude'] for bp in allBps])
    if threshold == 'auto':
        cutoff = np.median(mags) * 2.0
    else:
        cutoff = threshold

    return [bp for bp in allBps if bp['magnitude'] >= cutoff]


def estimateForceCorrection(trainY, horizon, decay=0.7, lookback=5):
    """Force Balance Model 보정값을 계산한다.

    Returns: 길이 horizon의 보정 배열, 또는 보정 불가 시 영벡터
    """
    bps = extractBreakpoints(trainY, threshold='auto')

    if len(bps) < 3:
        return np.zeros(horizon), {'turbulent': False, 'nBps': 0}

    recentBps = bps[-lookback:]

    lastDirection = recentBps[-1]['direction']
    if lastDirection == 'up_to_down':
        sign = 1.0
    else:
        sign = -1.0

    weights = np.array([0.5 ** i for i in range(len(recentBps) - 1, -1, -1)])
    weights /= weights.sum()
    mags = np.array([bp['magnitude'] for bp in recentBps])
    amplitude = np.dot(weights, mags)

    scale = np.std(trainY[-PERIOD*3:]) if len(trainY) >= PERIOD*3 else np.std(trainY)
    if scale > 0:
        amplitude = amplitude / scale
    else:
        return np.zeros(horizon), {'turbulent': False, 'nBps': len(bps)}

    lastBpPos = bps[-1]['pos']
    distFromLastBp = len(trainY) - lastBpPos
    turbulent = distFromLastBp <= 3

    if not turbulent:
        amplitude *= 0.3

    correction = np.zeros(horizon)
    for h in range(horizon):
        correction[h] = sign * amplitude * (decay ** h) * scale

    info = {
        'turbulent': turbulent,
        'nBps': len(bps),
        'amplitude': amplitude,
        'sign': sign,
        'distFromLastBp': distFromLastBp,
        'recentAltRate': _recentAlternatingRate(bps, lookback=10),
    }

    return correction, info


def _recentAlternatingRate(bps, lookback=10):
    """최근 N개 붕괴점의 교대율."""
    recent = bps[-lookback:]
    if len(recent) < 2:
        return 0.5
    dirs = [bp['direction'] for bp in recent]
    alternations = sum(1 for i in range(1, len(dirs)) if dirs[i] != dirs[i-1])
    return alternations / (len(dirs) - 1)


def computeSmape(actual, predicted):
    """sMAPE 계산."""
    denominator = (np.abs(actual) + np.abs(predicted))
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    return np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]) * 100


def computeMase(actual, predicted, trainY, period):
    """MASE 계산."""
    naiveErrors = np.abs(trainY[period:] - trainY[:-period])
    naiveMae = np.mean(naiveErrors)
    if naiveMae == 0:
        return 0.0
    return np.mean(np.abs(actual - predicted)) / naiveMae


def computeOwa(smape, mase, naiveSmape, naiveMase):
    """OWA 계산."""
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
    print("E012: Force Balance Model — 균형 붕괴 기반 예측 보정")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series")

    print("\n--- Phase 1: DOT 기준선 + Force 보정 ---")

    decayValues = [0.3, 0.5, 0.7, 0.85, 0.95]
    lookbackValues = [3, 5, 8]

    results = {}
    dotBaseline = {'smapes': [], 'mases': [], 'owas': []}
    naiveBaseline = {'smapes': [], 'mases': []}

    seriesInfos = []

    print("  DOT 기준선 계산 중...")
    for idx, (sid, trainY, testY) in enumerate(series):
        try:
            model = createModel('dot', PERIOD)
            model.fit(trainY)
            pred, _, _ = model.predict(HORIZON)

            actual = testY[:HORIZON]
            dotPred = pred[:len(actual)]

            naivePred = np.tile(trainY[-PERIOD:], (HORIZON // PERIOD + 1))[:len(actual)]

            dotSmape = computeSmape(actual, dotPred)
            dotMase = computeMase(actual, dotPred, trainY, PERIOD)
            naiveSmape = computeSmape(actual, naivePred)
            naiveMase = computeMase(actual, naivePred, trainY, PERIOD)
            dotOwa = computeOwa(dotSmape, dotMase, naiveSmape, naiveMase)

            dotBaseline['smapes'].append(dotSmape)
            dotBaseline['mases'].append(dotMase)
            dotBaseline['owas'].append(dotOwa)
            naiveBaseline['smapes'].append(naiveSmape)
            naiveBaseline['mases'].append(naiveMase)

            seriesInfos.append({
                'idx': idx,
                'sid': sid,
                'trainY': trainY,
                'actual': actual,
                'dotPred': dotPred,
                'naiveSmape': naiveSmape,
                'naiveMase': naiveMase,
            })

        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

        if (idx + 1) % 100 == 0:
            print(f"    {idx + 1}/{len(series)}...")

    avgDotOwa = np.mean(dotBaseline['owas'])
    print(f"\n  DOT 기준선 OWA: {avgDotOwa:.4f} (n={len(seriesInfos)})")

    print("\n  파라미터 그리드 탐색 중...")
    bestOwa = avgDotOwa
    bestParams = None
    allGridResults = []

    for decay in decayValues:
        for lookback in lookbackValues:
            key = f"d{decay}_lb{lookback}"
            correctedOwas = []
            wins = 0
            turbulentOwas = []
            calmOwas = []
            turbulentDotOwas = []
            calmDotOwas = []

            for i, info in enumerate(seriesInfos):
                correction, forceInfo = estimateForceCorrection(
                    info['trainY'], HORIZON, decay=decay, lookback=lookback
                )
                corrPred = info['dotPred'] + correction[:len(info['actual'])]

                corrSmape = computeSmape(info['actual'], corrPred)
                corrMase = computeMase(info['actual'], corrPred, info['trainY'], PERIOD)
                corrOwa = computeOwa(corrSmape, corrMase, info['naiveSmape'], info['naiveMase'])

                correctedOwas.append(corrOwa)
                dotOwaI = dotBaseline['owas'][i]

                if corrOwa < dotOwaI:
                    wins += 1

                if forceInfo['turbulent']:
                    turbulentOwas.append(corrOwa)
                    turbulentDotOwas.append(dotOwaI)
                else:
                    calmOwas.append(corrOwa)
                    calmDotOwas.append(dotOwaI)

            avgCorrOwa = np.mean(correctedOwas)
            winRate = wins / len(correctedOwas)
            improvement = (avgDotOwa - avgCorrOwa) / avgDotOwa * 100

            gridResult = {
                'decay': decay,
                'lookback': lookback,
                'owa': avgCorrOwa,
                'improvement': improvement,
                'winRate': winRate,
                'turbulentOwa': np.mean(turbulentOwas) if turbulentOwas else None,
                'turbulentDotOwa': np.mean(turbulentDotOwas) if turbulentDotOwas else None,
                'turbulentN': len(turbulentOwas),
                'calmOwa': np.mean(calmOwas) if calmOwas else None,
                'calmDotOwa': np.mean(calmDotOwas) if calmDotOwas else None,
                'calmN': len(calmOwas),
            }
            allGridResults.append(gridResult)

            if avgCorrOwa < bestOwa:
                bestOwa = avgCorrOwa
                bestParams = gridResult

    print("\n--- Phase 2: 파라미터 그리드 결과 ---")
    print(f"  {'decay':>5s} {'lb':>3s} {'OWA':>7s} {'Δ%':>7s} {'승률':>6s} {'격동N':>5s} {'격동OWA':>8s} {'고요OWA':>8s}")
    print("  " + "-" * 60)
    for r in sorted(allGridResults, key=lambda x: x['owa']):
        turbStr = f"{r['turbulentOwa']:.4f}" if r['turbulentOwa'] is not None else "  N/A "
        calmStr = f"{r['calmOwa']:.4f}" if r['calmOwa'] is not None else "  N/A "
        print(f"  {r['decay']:>5.2f} {r['lookback']:>3d} {r['owa']:>7.4f} {r['improvement']:>+6.2f}% {r['winRate']:>5.1%} {r['turbulentN']:>5d} {turbStr:>8s} {calmStr:>8s}")

    print(f"\n  DOT 기준선: OWA {avgDotOwa:.4f}")
    if bestParams and bestParams['owa'] < avgDotOwa:
        print(f"  최적 보정:  OWA {bestParams['owa']:.4f} (decay={bestParams['decay']}, lb={bestParams['lookback']})")
        print(f"  개선:       {bestParams['improvement']:+.2f}%, 승률 {bestParams['winRate']:.1%}")
    else:
        print(f"  보정 효과 없음 — 모든 파라미터에서 DOT 기준선 이하")

    print("\n--- Phase 3: 격동기 vs 고요기 세부 분석 ---")
    if bestParams:
        bp = bestParams
        if bp['turbulentOwa'] is not None and bp['turbulentDotOwa'] is not None:
            turbImpr = (bp['turbulentDotOwa'] - bp['turbulentOwa']) / bp['turbulentDotOwa'] * 100
            print(f"  격동기 (n={bp['turbulentN']})")
            print(f"    DOT:  {bp['turbulentDotOwa']:.4f}")
            print(f"    보정: {bp['turbulentOwa']:.4f} ({turbImpr:+.2f}%)")
        if bp['calmOwa'] is not None and bp['calmDotOwa'] is not None:
            calmImpr = (bp['calmDotOwa'] - bp['calmOwa']) / bp['calmDotOwa'] * 100
            print(f"  고요기 (n={bp['calmN']})")
            print(f"    DOT:  {bp['calmDotOwa']:.4f}")
            print(f"    보정: {bp['calmOwa']:.4f} ({calmImpr:+.2f}%)")

    print("\n--- Phase 4: 교대율별 효과 ---")
    if bestParams:
        decay = bestParams['decay']
        lookback = bestParams['lookback']

        highAltOwas = []
        highAltDotOwas = []
        lowAltOwas = []
        lowAltDotOwas = []

        for i, info in enumerate(seriesInfos):
            correction, forceInfo = estimateForceCorrection(
                info['trainY'], HORIZON, decay=decay, lookback=lookback
            )
            corrPred = info['dotPred'] + correction[:len(info['actual'])]
            corrSmape = computeSmape(info['actual'], corrPred)
            corrMase = computeMase(info['actual'], corrPred, info['trainY'], PERIOD)
            corrOwa = computeOwa(corrSmape, corrMase, info['naiveSmape'], info['naiveMase'])

            altRate = forceInfo['recentAltRate']
            if altRate >= 0.7:
                highAltOwas.append(corrOwa)
                highAltDotOwas.append(dotBaseline['owas'][i])
            else:
                lowAltOwas.append(corrOwa)
                lowAltDotOwas.append(dotBaseline['owas'][i])

        if highAltOwas:
            hImpr = (np.mean(highAltDotOwas) - np.mean(highAltOwas)) / np.mean(highAltDotOwas) * 100
            print(f"  교대율 >= 0.7 (n={len(highAltOwas)})")
            print(f"    DOT:  {np.mean(highAltDotOwas):.4f}")
            print(f"    보정: {np.mean(highAltOwas):.4f} ({hImpr:+.2f}%)")
        if lowAltOwas:
            lImpr = (np.mean(lowAltDotOwas) - np.mean(lowAltOwas)) / np.mean(lowAltDotOwas) * 100
            print(f"  교대율 < 0.7 (n={len(lowAltOwas)})")
            print(f"    DOT:  {np.mean(lowAltDotOwas):.4f}")
            print(f"    보정: {np.mean(lowAltOwas):.4f} ({lImpr:+.2f}%)")

    print("\n--- Phase 5: 보정 크기 분포 ---")
    if bestParams:
        decay = bestParams['decay']
        lookback = bestParams['lookback']
        corrMagnitudes = []
        seriesStds = []
        for info in seriesInfos:
            correction, _ = estimateForceCorrection(
                info['trainY'], HORIZON, decay=decay, lookback=lookback
            )
            corrMag = np.abs(correction[0])
            serStd = np.std(info['actual'])
            if serStd > 0:
                corrMagnitudes.append(corrMag / serStd)
                seriesStds.append(serStd)

        corrMagnitudes = np.array(corrMagnitudes)
        print(f"  1단계 보정 크기 / 시리즈 std")
        print(f"    평균: {np.mean(corrMagnitudes):.4f}")
        print(f"    중앙값: {np.median(corrMagnitudes):.4f}")
        print(f"    최대: {np.max(corrMagnitudes):.4f}")
        print(f"    0.1 이상: {np.mean(corrMagnitudes > 0.1):.1%}")
        print(f"    0.5 이상: {np.mean(corrMagnitudes > 0.5):.1%}")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== FINAL SUMMARY ===")
    print("=" * 70)
    print(f"  DOT 기준선 OWA: {avgDotOwa:.4f}")
    if bestParams and bestParams['owa'] < avgDotOwa:
        print(f"  최적 보정 OWA:  {bestParams['owa']:.4f}")
        print(f"  개선:           {bestParams['improvement']:+.2f}%")
        print(f"  승률:           {bestParams['winRate']:.1%}")
        print(f"  파라미터:       decay={bestParams['decay']}, lookback={bestParams['lookback']}")
    else:
        print(f"  결론: Force 보정은 DOT를 개선하지 못함")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
