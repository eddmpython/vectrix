"""
실험 ID: dataProfiling/011
실험명: 균형 붕괴점 역학 (Breakpoint Dynamics)

목적:
- 시계열의 2차 미분에서 부호 전환점(균형 붕괴점)을 추출
- 붕괴점의 특성(간격, 크기, 방향, 비대칭)이 시리즈별로 안정적 패턴을 보이는지
- 붕괴점 패턴으로 "다음 붕괴"의 방향/크기를 예측할 수 있는지
- 이것이 기존 예측(DOT)의 잔차를 설명하는지

가설:
1. 붕괴점 간격은 시리즈 내에서 특성적 분포를 가진다 (순수 랜덤이 아니다)
2. 상승→하강 붕괴와 하강→상승 붕괴의 크기에 비대칭이 존재한다
3. 최근 N개 붕괴점의 패턴이 다음 붕괴 방향을 예측한다 (50% 이상)
4. DOT 잔차의 큰 오차가 붕괴점 근처에 집중된다

방법:
1. M4 Monthly 500개 (seed=42)
2. 각 시리즈: 1차 미분(속도) → 2차 미분(가속도) → 부호 전환점 추출
3. 붕괴점 특성 분석: 간격 분포, 크기 분포, 방향 비대칭
4. 예측 실험: 마지막 3~5개 붕괴점으로 다음 붕괴 방향 예측
5. DOT 잔차와 붕괴점 위치의 관계 분석

결과 (실험 후 작성):
- 유의미 붕괴점(median×2 threshold): 평균 21.9개 (시리즈 길이의 10%)
- 간격 불규칙: 실제 CV 1.314 > 랜덤 0.855. 붕괴점은 clustering(몰려서 등장)
- 교대율 0.736 (랜덤 0.5 vs 필연 1.0). 방향 예측 73.8% (alternating)
- 방향 비대칭 -0.020 (거의 대칭), 상승→하강 비율 0.486 (균등)
- 붕괴 크기 ACF(1) = 0.187, 75.1%가 양의 상관 → 큰 붕괴 다음 큰 붕괴
- 최대 붕괴 크기↔MASE r=-0.288 (p<0.001): 큰 변동 시리즈가 오히려 예측 쉬움
- 붕괴점 수↔MASE r=-0.075 (무관)

결론:
- 가설 1 기각: 간격은 랜덤보다 불규칙 (규칙적이지 않음). 그러나 clustering 구조 존재
- 가설 2 기각: 비대칭 거의 없음 (-0.020)
- 가설 3 채택: alternating 73.8% > 50%. 방향은 예측 가능
- 가설 4 부분 기각: 붕괴점 수↔오차 무관. 최대 크기↔오차는 역상관
- 핵심 발견 3가지:
  (1) 붕괴는 clustering한다 — 격동기/고요기가 있음 (GARCH 유사)
  (2) 크기에 자기상관 — 큰 붕괴 뒤 큰 붕괴 (변동성 클러스터링)
  (3) 방향은 73.8% 교대 — 완전 예측은 아니지만 랜덤 아님
- 다음 실험: 이 3가지 특성으로 DOT 예측을 사후 보정할 수 있는지

실험일: 2026-03-05
"""

import sys
import os
import time
import warnings

import numpy as np
import pandas as pd
from collections import Counter

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
    """2차 미분의 부호 전환점 중 유의미한 것만 추출한다.

    threshold: 'auto' = 가속도 중앙값의 2배 이상만, float = 절대 기준, None = 전부
    Returns: list of dict {pos, accel_before, accel_after, magnitude, direction}
    """
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
                'relPos': (i + 1) / len(y),
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


def analyzeBreakpointPattern(breakpoints):
    """붕괴점 패턴의 특성을 분석한다."""
    if len(breakpoints) < 3:
        return None

    positions = [bp['pos'] for bp in breakpoints]
    gaps = np.diff(positions)
    magnitudes = [bp['magnitude'] for bp in breakpoints]
    directions = [bp['direction'] for bp in breakpoints]

    upToDown = [bp['magnitude'] for bp in breakpoints if bp['direction'] == 'up_to_down']
    downToUp = [bp['magnitude'] for bp in breakpoints if bp['direction'] == 'down_to_up']

    gapCv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 999

    directionRuns = 1
    for i in range(1, len(directions)):
        if directions[i] != directions[i-1]:
            directionRuns += 1
    alternatingRatio = directionRuns / len(directions)

    return {
        'nBreakpoints': len(breakpoints),
        'meanGap': np.mean(gaps),
        'gapCv': gapCv,
        'gapRegularity': 1.0 / (1.0 + gapCv),
        'meanMagnitude': np.mean(magnitudes),
        'magnitudeCv': np.std(magnitudes) / np.mean(magnitudes) if np.mean(magnitudes) > 0 else 999,
        'upToDownRatio': len(upToDown) / len(breakpoints),
        'asymmetry': (np.mean(upToDown) - np.mean(downToUp)) / np.mean(magnitudes) if upToDown and downToUp and np.mean(magnitudes) > 0 else 0,
        'alternatingRatio': alternatingRatio,
        'meanUpMag': np.mean(upToDown) if upToDown else 0,
        'meanDownMag': np.mean(downToUp) if downToUp else 0,
    }


def predictNextDirection(breakpoints, lookback=5):
    """최근 N개 붕괴점의 방향 패턴으로 다음 방향을 예측한다.

    전략들:
    1. alternating: 마지막과 반대 방향
    2. momentum: 마지막과 같은 방향
    3. majority: 최근 N개 중 다수 방향
    4. pattern2: 마지막 2개의 패턴이 반복된다고 가정
    """
    if len(breakpoints) < 3:
        return {}

    directions = [bp['direction'] for bp in breakpoints]

    predictions = {}

    predictions['alternating'] = 'down_to_up' if directions[-1] == 'up_to_down' else 'up_to_down'
    predictions['momentum'] = directions[-1]

    recent = directions[-lookback:]
    counter = Counter(recent)
    predictions['majority'] = counter.most_common(1)[0][0]

    if len(directions) >= 3:
        lastTwo = (directions[-2], directions[-1])
        for i in range(len(directions) - 2):
            if (directions[i], directions[i+1]) == lastTwo:
                if i + 2 < len(directions):
                    predictions['pattern2'] = directions[i + 2]
        if 'pattern2' not in predictions:
            predictions['pattern2'] = predictions['alternating']

    return predictions


def evaluateDirectionPrediction(series):
    """각 시리즈에서 leave-last-out으로 방향 예측 정확도를 측정한다."""
    results = {s: {'correct': 0, 'total': 0} for s in ['alternating', 'momentum', 'majority', 'pattern2']}

    for sid, trainY, testY in series:
        bps = extractBreakpoints(trainY)
        if len(bps) < 6:
            continue

        for evalIdx in range(5, len(bps)):
            pastBps = bps[:evalIdx]
            actualDir = bps[evalIdx]['direction']

            preds = predictNextDirection(pastBps)
            for strategy, predDir in preds.items():
                results[strategy]['total'] += 1
                if predDir == actualDir:
                    results[strategy]['correct'] += 1

    return results


def countHorizonBreakpoints(series, thresholdMultiplier=2.0):
    """horizon 내 큰 붕괴점 수 vs DOT 오차의 상관."""
    from vectrix.engine.registry import createModel

    bpCountsH = []
    mases = []
    maxBpMags = []

    count = 0
    for sid, trainY, testY in series[:300]:
        try:
            model = createModel('dot', PERIOD)
            model.fit(trainY)
            pred, _, _ = model.predict(HORIZON)

            fullY = np.concatenate([trainY, testY])
            allBpsRaw = extractBreakpoints(fullY, threshold=None)
            if not allBpsRaw:
                continue
            mags = np.array([bp['magnitude'] for bp in allBpsRaw])
            cutoff = np.median(mags) * thresholdMultiplier

            horizonStart = len(trainY)
            horizonEnd = len(trainY) + len(testY)
            horizonBps = [bp for bp in allBpsRaw
                          if horizonStart <= bp['pos'] < horizonEnd and bp['magnitude'] >= cutoff]

            errors = np.abs(testY[:len(pred)] - pred[:len(testY)])
            maseDenom = np.mean(np.abs(np.diff(trainY[-(PERIOD+1):])))
            if maseDenom > 0:
                mase = np.mean(errors) / maseDenom
            else:
                continue

            bpCountsH.append(len(horizonBps))
            mases.append(mase)
            maxBpMags.append(max([bp['magnitude'] for bp in horizonBps], default=0) / np.std(fullY) if np.std(fullY) > 0 else 0)

            count += 1
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

    return np.array(bpCountsH), np.array(mases), np.array(maxBpMags)


def analyzeBreakpointsVsRandom(series):
    """붕괴점 간격이 랜덤보다 규칙적인지 확인한다."""
    realGapCvs = []
    randomGapCvs = []

    for sid, trainY, testY in series:
        bps = extractBreakpoints(trainY)
        if len(bps) < 5:
            continue

        positions = [bp['pos'] for bp in bps]
        gaps = np.diff(positions)
        realCv = np.std(gaps) / np.mean(gaps) if np.mean(gaps) > 0 else 999
        realGapCvs.append(realCv)

        simCvs = []
        for _ in range(100):
            simPositions = sorted(np.random.choice(range(2, len(trainY)), size=len(bps), replace=False))
            simGaps = np.diff(simPositions)
            simCv = np.std(simGaps) / np.mean(simGaps) if np.mean(simGaps) > 0 else 999
            simCvs.append(simCv)
        randomGapCvs.append(np.mean(simCvs))

    return np.array(realGapCvs), np.array(randomGapCvs)


def analyzeForecstHorizonBreakpoints(series):
    """테스트 기간(horizon)에 붕괴점이 있는 시리즈 vs 없는 시리즈의 DOT 성능 차이."""
    from vectrix.engine.registry import createModel

    withBp = []
    withoutBp = []

    count = 0
    for sid, trainY, testY in series[:300]:
        try:
            model = createModel('dot', PERIOD)
            model.fit(trainY)
            pred, _, _ = model.predict(HORIZON)

            fullY = np.concatenate([trainY, testY])
            allBps = extractBreakpoints(fullY)

            horizonStart = len(trainY)
            horizonEnd = len(trainY) + len(testY)
            horizonBps = [bp for bp in allBps if horizonStart <= bp['pos'] < horizonEnd]

            errors = np.abs(testY[:len(pred)] - pred[:len(testY)])
            mase_denom = np.mean(np.abs(np.diff(trainY[-(PERIOD+1):])))
            if mase_denom > 0:
                mase = np.mean(errors) / mase_denom
            else:
                mase = np.mean(errors)

            if horizonBps:
                withBp.append(mase)
            else:
                withoutBp.append(mase)

            count += 1
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

    return np.array(withBp), np.array(withoutBp)


if __name__ == '__main__':
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    startTime = time.time()

    print("=" * 70)
    print("E011: Breakpoint Dynamics — 균형 붕괴점 역학")
    print("=" * 70)

    series = loadM4Monthly()
    print(f"\nLoaded {len(series)} Monthly series")

    print("\n--- Phase 1: threshold별 붕괴점 수 ---")
    for label, th in [('전체(no filter)', None), ('auto(median×2)', 'auto')]:
        counts = []
        for sid, trainY, testY in series:
            bps = extractBreakpoints(trainY, threshold=th)
            counts.append(len(bps))
        counts = np.array(counts)
        avgLen = np.mean([len(s[1]) for s in series])
        print(f"  {label:20s}: 평균 {np.mean(counts):.1f}개, 시리즈 대비 {np.mean(counts)/avgLen*100:.1f}%")

    print("\n--- Phase 2: 유의미한 붕괴점(auto) 특성 ---")
    allPatterns = []
    bpCounts = []
    for sid, trainY, testY in series:
        bps = extractBreakpoints(trainY, threshold='auto')
        pattern = analyzeBreakpointPattern(bps)
        bpCounts.append(len(bps))
        if pattern:
            allPatterns.append(pattern)

    bpCounts = np.array(bpCounts)
    print(f"  분석 가능: {len(allPatterns)}/{len(series)}")

    if allPatterns:
        gapRegularities = [p['gapRegularity'] for p in allPatterns]
        asymmetries = [p['asymmetry'] for p in allPatterns]
        altRatios = [p['alternatingRatio'] for p in allPatterns]
        upRatios = [p['upToDownRatio'] for p in allPatterns]

        print(f"  간격 규칙성 평균: {np.mean(gapRegularities):.3f} (1=규칙, 0=불규칙)")
        print(f"  교대 비율: {np.mean(altRatios):.3f} (1=완전교대, 0.5=랜덤)")
        print(f"  방향 비대칭: {np.mean(asymmetries):.3f} (0=대칭)")
        print(f"  상승→하강 비율: {np.mean(upRatios):.3f} (0.5=균등)")

    print("\n--- Phase 3: 간격 규칙성 — 실제 vs 랜덤 ---")
    realCvs, randomCvs = analyzeBreakpointsVsRandom(series)
    print(f"  실제 간격 CV: {np.mean(realCvs):.3f}")
    print(f"  랜덤 간격 CV: {np.mean(randomCvs):.3f}")
    moreRegular = np.mean(realCvs < randomCvs)
    print(f"  실제가 랜덤보다 규칙적: {moreRegular:.1%}")

    print("\n--- Phase 4: 방향 예측 (유의미 붕괴점만) ---")
    dirResults = evaluateDirectionPrediction(series)
    for strategy, res in dirResults.items():
        if res['total'] > 0:
            acc = res['correct'] / res['total']
            print(f"  {strategy:15s}: {acc:.1%} ({res['correct']}/{res['total']})")
    bestStrategy = max(dirResults.items(), key=lambda x: x[1]['correct'] / max(x[1]['total'], 1))
    bestAcc = bestStrategy[1]['correct'] / bestStrategy[1]['total']
    print(f"  → 최고: {bestStrategy[0]} ({bestAcc:.1%})")

    print("\n--- Phase 5: Horizon 붕괴점 수 vs DOT 오차 상관 ---")
    bpCountsH, mases, maxMags = countHorizonBreakpoints(series)
    from scipy.stats import spearmanr
    if len(bpCountsH) > 10:
        corrCount, pCount = spearmanr(bpCountsH, mases)
        corrMag, pMag = spearmanr(maxMags, mases)
        print(f"  시리즈 수: {len(bpCountsH)}")
        print(f"  붕괴점 수↔MASE: r={corrCount:.3f} (p={pCount:.4f})")
        print(f"  최대 붕괴 크기↔MASE: r={corrMag:.3f} (p={pMag:.4f})")

        q3 = np.percentile(bpCountsH, 75)
        q1 = np.percentile(bpCountsH, 25)
        highBp = mases[bpCountsH >= q3] if np.any(bpCountsH >= q3) else np.array([])
        lowBp = mases[bpCountsH <= q1] if np.any(bpCountsH <= q1) else np.array([])
        if len(highBp) > 0 and len(lowBp) > 0:
            print(f"  상위25% 붕괴점: 평균 MASE={np.mean(highBp):.3f} (n={len(highBp)})")
            print(f"  하위25% 붕괴점: 평균 MASE={np.mean(lowBp):.3f} (n={len(lowBp)})")
            print(f"  비율: {np.mean(highBp)/np.mean(lowBp):.2f}x")

    print("\n--- Phase 6: 붕괴 크기 시퀀스의 자기상관 ---")
    acfValues = []
    for sid, trainY, testY in series:
        bps = extractBreakpoints(trainY, threshold='auto')
        if len(bps) < 10:
            continue
        mags = np.array([bp['magnitude'] for bp in bps])
        mags = (mags - np.mean(mags)) / (np.std(mags) + 1e-10)
        acf1 = np.corrcoef(mags[:-1], mags[1:])[0, 1]
        if np.isfinite(acf1):
            acfValues.append(acf1)

    if len(acfValues) > 0:
        acfValues = np.array(acfValues)
        print(f"  붕괴 크기 ACF(1) 평균: {np.mean(acfValues):.3f}")
        print(f"  ACF(1) > 0.3인 비율: {np.mean(acfValues > 0.3):.1%}")
        print(f"  ACF(1) > 0인 비율: {np.mean(acfValues > 0):.1%}")
        print(f"  → {'크기에 자기상관 있음 — 큰 붕괴 다음 큰 붕괴' if np.mean(acfValues) > 0.15 else '크기에 자기상관 약함'}")

    elapsed = time.time() - startTime

    print("\n" + "=" * 70)
    print("=== SUMMARY ===")
    print("=" * 70)
    print(f"  유의미 붕괴점(auto): 평균 {np.mean(bpCounts):.1f}개")
    print(f"  간격 규칙성: 실제 > 랜덤 {moreRegular:.1%}")
    if allPatterns:
        print(f"  교대율: {np.mean(altRatios):.3f}")
    print(f"  방향 예측 최고: {bestStrategy[0]} {bestAcc:.1%}")
    if len(bpCountsH) > 10:
        print(f"  붕괴점 수↔오차: r={corrCount:.3f}")
        print(f"  붕괴 크기↔오차: r={corrMag:.3f}")
    if len(acfValues) > 0:
        print(f"  붕괴 크기 ACF(1): {np.mean(acfValues):.3f}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print("=" * 70)
