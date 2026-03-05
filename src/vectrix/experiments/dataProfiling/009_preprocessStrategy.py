"""
실험 ID: dataProfiling/009
실험명: DNA 기반 전처리 전략 선택

목적:
- "어떤 모델을 쓸까"가 아닌 "어떤 전처리를 할까"로 패러다임 전환
- DNA 특성으로 최적 전처리를 선택하면 DOT 정확도를 올릴 수 있는지 검증
- 전처리 전략: raw(없음), log변환, 차분, 디노이징(MA), STL 분해 잔차

가설:
1. 변동성 높은 시리즈 → log변환이 DOT OWA를 5%+ 개선
2. 트렌드 강한 시리즈 → 차분 후 DOT가 개선
3. DNA 특성으로 최적 전처리를 선택하면 평균 OWA -2% 이상 개선

방법:
1. M4 Monthly 500개 샘플
2. DNA 특성 추출 (간이: cv, trend, seasonality, acf_lag1)
3. 5가지 전처리 × DOT 예측
4. per-series 최적 전처리 oracle gap 측정
5. DNA 기반 간단한 규칙으로 전처리 선택 → 실제 개선 측정

결과 (실험 후 작성):
- M4 Monthly 500개 샘플

전처리별 DOT OWA
  raw:  0.812 (기준선)
  log:  0.819 (+0.8%)
  diff: 2.357 (+190% — 폭발)
  ma3:  0.912 (+12.3%)
  stl:  1.692 (+108% — 폭발)

Oracle gap (per-series 최적 전처리)
  Raw: 0.812 → Oracle: 0.671 (-17.4%)
  raw 최적 20.4%, log 22.6%, diff 25.6%, ma3 18.6%, stl 12.8%
  → 어떤 전처리도 다수에게 지배적이지 않음

DNA → 전처리 상관
  가장 강한 상관: acf1↔ma3개선 (r=0.239), seasonality↔stl개선 (r=-0.210)
  전반적으로 r < 0.25 — DNA로 전처리를 선택할 수 없음

규칙 기반 선택
  OWA 1.091 (+34.4%) — raw보다 대폭 악화!
  diff와 stl이 "맞으면 최적이지만 틀리면 폭발"하기 때문

결론:
- **Oracle gap -17.4%는 매력적** — 전처리 최적화의 이론적 가치 존재
- **하지만 선택이 불가능** — DNA 특성으로 안전한 전처리 선택이 안 됨
- **diff/stl은 high risk high reward** — 25.6%에서 최적이지만 나머지에서 폭발
- **"잘 고르면 좋지만 잘못 고르면 폭발"** — 모델 선택과 동일한 구조적 문제
- **안전한 전처리는 raw뿐** — log/ma3는 미미한 차이, diff/stl은 위험
- **패러다임 전환 결론**: "무엇을 할까" 선택 자체가 DNA 수준으로는 불가능
  - 더 풍부한 피처(예: 정상성 검정, 분산 안정성 검정 등)이 필요하거나
  - 학습 기반 접근(holdout 검증으로 전처리 선택)이 필요

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


def loadM4Monthly(nSample):
    trainPath = os.path.join(M4_DIR, 'Monthly-train.csv')
    testPath = os.path.join(M4_DIR, 'Monthly-test.csv')
    trainDf = pd.read_csv(trainPath, index_col=0)
    testDf = pd.read_csv(testPath, index_col=0)

    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(trainDf), min(nSample, len(trainDf)), replace=False)
    trainSampled = trainDf.iloc[idx]
    testSampled = testDf.iloc[idx]

    data = {}
    for sid in trainSampled.index:
        trainVals = trainSampled.loc[sid].dropna().values.astype(np.float64)
        testVals = testSampled.loc[sid].dropna().values.astype(np.float64)
        if len(trainVals) >= MIN_LEN and len(testVals) >= HORIZON:
            data[str(sid)] = {'train': trainVals, 'test': testVals[:HORIZON]}
    return data


def computeSmape(actual, predicted):
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(2 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100


def computeMase(actual, predicted, trainY, period):
    n = len(trainY)
    if n <= period:
        naiveErr = np.mean(np.abs(np.diff(trainY)))
    else:
        naiveErr = np.mean(np.abs(trainY[period:] - trainY[:-period]))
    if naiveErr < 1e-10:
        naiveErr = 1e-10
    return np.mean(np.abs(actual - predicted)) / naiveErr


def computeOwa(smape, mase, naiveSmape, naiveMase):
    if naiveSmape < 1e-10:
        naiveSmape = 1e-10
    if naiveMase < 1e-10:
        naiveMase = 1e-10
    return 0.5 * (smape / naiveSmape) + 0.5 * (mase / naiveMase)


def naiveSeasonal(trainY, steps, period):
    lastSeason = trainY[-period:]
    reps = (steps // period) + 1
    return np.tile(lastSeason, reps)[:steps]


def forecastDot(trainY, steps, period):
    from vectrix.engine.registry import createModel
    try:
        model = createModel('dot', period)
        model.fit(trainY)
        pred, _, _ = model.predict(steps)
        pred = np.where(np.isfinite(pred), pred, np.nanmean(trainY))
        return pred
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return np.full(steps, np.nanmean(trainY))


def preprocessLog(trainY):
    minVal = np.min(trainY)
    if minVal <= 0:
        shift = abs(minVal) + 1.0
    else:
        shift = 0.0
    return np.log1p(trainY + shift), shift


def inverseLog(pred, shift):
    return np.expm1(pred) - shift


def preprocessDiff(trainY):
    return np.diff(trainY), trainY[-1]


def inverseDiff(pred, lastVal):
    return lastVal + np.cumsum(pred)


def preprocessMA(trainY, window=3):
    kernel = np.ones(window) / window
    smoothed = np.convolve(trainY, kernel, mode='same')
    smoothed[:window // 2] = trainY[:window // 2]
    smoothed[-(window // 2):] = trainY[-(window // 2):]
    return smoothed


def preprocessSTL(trainY, period):
    n = len(trainY)
    if n < period * 2:
        return trainY, np.zeros_like(trainY), np.zeros(1)

    nCycles = n // period
    truncated = trainY[:nCycles * period]
    reshaped = truncated.reshape(nCycles, period)
    seasonal = np.mean(reshaped, axis=0)
    seasonal = seasonal - np.mean(seasonal)
    seasonalFull = np.tile(seasonal, nCycles + 1)[:n]
    deseasonalized = trainY - seasonalFull

    from scipy.ndimage import uniform_filter1d
    trend = uniform_filter1d(deseasonalized, size=max(3, period // 2))
    remainder = deseasonalized - trend

    return remainder, seasonalFull, trend


def inverseSTL(predRemainder, seasonalPattern, lastTrend, steps, period):
    seasonal = np.tile(seasonalPattern, (steps // period) + 1)[:steps]
    trendSlope = lastTrend[-1] - lastTrend[-2] if len(lastTrend) >= 2 else 0
    trendExt = lastTrend[-1] + trendSlope * np.arange(1, steps + 1)
    return predRemainder + seasonal + trendExt


def simpleDna(trainY, period):
    cv = np.std(trainY) / (np.mean(np.abs(trainY)) + 1e-10)

    n = len(trainY)
    if n > period * 2:
        nCycles = n // period
        truncated = trainY[:nCycles * period]
        reshaped = truncated.reshape(nCycles, period)
        seasonalMean = np.mean(reshaped, axis=0)
        seasonalVar = np.var(seasonalMean)
        totalVar = np.var(trainY)
        seasStrength = seasonalVar / (totalVar + 1e-10)
    else:
        seasStrength = 0.0

    x = np.arange(n, dtype=np.float64)
    xNorm = x - x.mean()
    slope = np.sum(xNorm * (trainY - trainY.mean())) / (np.sum(xNorm ** 2) + 1e-10)
    trendStrength = abs(slope) * n / (np.std(trainY) + 1e-10)

    if n > 1:
        y = trainY - np.mean(trainY)
        acf1 = np.sum(y[:-1] * y[1:]) / (np.sum(y ** 2) + 1e-10)
    else:
        acf1 = 0.0

    return {'cv': cv, 'seasonality': seasStrength, 'trend': trendStrength, 'acf1': acf1}


if __name__ == '__main__':
    if sys.platform == 'win32':
        sys.stdout = __import__('io').TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("E009: DNA-Based Preprocessing Strategy Selection")
    print("=" * 70)

    t0 = time.time()

    data = loadM4Monthly(N_SAMPLE)
    sids = list(data.keys())
    print(f"\nLoaded {len(data)} Monthly series")

    STRATEGIES = ['raw', 'log', 'diff', 'ma3', 'stl']

    naiveSmapes = []
    naiveMases = []
    stratOwas = {s: [] for s in STRATEGIES}
    dnaList = []

    for i, sid in enumerate(sids):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sids)}...", flush=True)

        trainY = data[sid]['train']
        testY = data[sid]['test']

        naive2Pred = naiveSeasonal(trainY, HORIZON, PERIOD)
        nSmape = computeSmape(testY, naive2Pred)
        nMase = computeMase(testY, naive2Pred, trainY, PERIOD)
        naiveSmapes.append(nSmape)
        naiveMases.append(nMase)

        dna = simpleDna(trainY, PERIOD)
        dnaList.append(dna)

        rawPred = forecastDot(trainY, HORIZON, PERIOD)
        s = computeSmape(testY, rawPred)
        a = computeMase(testY, rawPred, trainY, PERIOD)
        stratOwas['raw'].append(computeOwa(s, a, nSmape, nMase))

        logY, shift = preprocessLog(trainY)
        logPred = forecastDot(logY, HORIZON, PERIOD)
        logPredInv = inverseLog(logPred, shift)
        logPredInv = np.where(np.isfinite(logPredInv), logPredInv, np.nanmean(trainY))
        s = computeSmape(testY, logPredInv)
        a = computeMase(testY, logPredInv, trainY, PERIOD)
        stratOwas['log'].append(computeOwa(s, a, nSmape, nMase))

        if len(trainY) > HORIZON + 2:
            diffY, lastVal = preprocessDiff(trainY)
            diffPred = forecastDot(diffY, HORIZON, max(1, PERIOD))
            diffPredInv = inverseDiff(diffPred, lastVal)
            diffPredInv = np.where(np.isfinite(diffPredInv), diffPredInv, np.nanmean(trainY))
            s = computeSmape(testY, diffPredInv)
            a = computeMase(testY, diffPredInv, trainY, PERIOD)
            stratOwas['diff'].append(computeOwa(s, a, nSmape, nMase))
        else:
            stratOwas['diff'].append(stratOwas['raw'][-1])

        maY = preprocessMA(trainY, window=3)
        maPred = forecastDot(maY, HORIZON, PERIOD)
        maPred = np.where(np.isfinite(maPred), maPred, np.nanmean(trainY))
        s = computeSmape(testY, maPred)
        a = computeMase(testY, maPred, trainY, PERIOD)
        stratOwas['ma3'].append(computeOwa(s, a, nSmape, nMase))

        remainder, seasonalFull, trend = preprocessSTL(trainY, PERIOD)
        if len(remainder) > 10:
            stlPred = forecastDot(remainder, HORIZON, 1)
            seasonal = seasonalFull[:PERIOD]
            stlPredFull = inverseSTL(stlPred, seasonal, trend, HORIZON, PERIOD)
            stlPredFull = np.where(np.isfinite(stlPredFull), stlPredFull, np.nanmean(trainY))
            s = computeSmape(testY, stlPredFull)
            a = computeMase(testY, stlPredFull, trainY, PERIOD)
            stratOwas['stl'].append(computeOwa(s, a, nSmape, nMase))
        else:
            stratOwas['stl'].append(stratOwas['raw'][-1])

    avgNaiveSmape = np.mean(naiveSmapes)
    avgNaiveMase = np.mean(naiveMases)

    print(f"\n{'=' * 70}")
    print(f"=== Phase 1: Strategy Performance ===")
    print(f"{'=' * 70}")

    for s in STRATEGIES:
        owas = np.array(stratOwas[s])
        print(f"  {s:>5}: mean OWA={owas.mean():.4f}  median={np.median(owas):.4f}  "
              f"Q25={np.percentile(owas, 25):.4f}  Q75={np.percentile(owas, 75):.4f}")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 2: Oracle Gap (per-series best strategy) ===")
    print(f"{'=' * 70}")

    owaMatrix = np.array([stratOwas[s] for s in STRATEGIES])
    oracleBest = np.min(owaMatrix, axis=0)
    rawOwas = np.array(stratOwas['raw'])
    oracleOwa = np.mean(oracleBest)
    rawOwa = np.mean(rawOwas)

    print(f"  Raw DOT:       {rawOwa:.4f}")
    print(f"  Oracle best:   {oracleOwa:.4f} ({oracleOwa - rawOwa:+.4f}, {(oracleOwa - rawOwa)/rawOwa*100:+.1f}%)")

    bestStratPerSeries = np.argmin(owaMatrix, axis=0)
    for j, s in enumerate(STRATEGIES):
        count = np.sum(bestStratPerSeries == j)
        print(f"    {s:>5} is best for {count:>4} series ({count/len(sids)*100:.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 3: DNA → Strategy Correlation ===")
    print(f"{'=' * 70}")

    dnaDf = pd.DataFrame(dnaList)
    for s in STRATEGIES:
        dnaDf[f'owa_{s}'] = stratOwas[s]
    dnaDf['bestStrat'] = bestStratPerSeries

    for feat in ['cv', 'seasonality', 'trend', 'acf1']:
        print(f"\n  {feat}:")
        for s in STRATEGIES:
            improvCol = np.array(stratOwas['raw']) - np.array(stratOwas[s])
            corr = np.corrcoef(dnaDf[feat].values, improvCol)[0, 1]
            print(f"    raw-{s:>4} improvement vs {feat}: r={corr:+.3f}")

    print(f"\n{'=' * 70}")
    print(f"=== Phase 4: Simple Rules ===")
    print(f"{'=' * 70}")

    cvArr = dnaDf['cv'].values
    seasArr = dnaDf['seasonality'].values
    trendArr = dnaDf['trend'].values
    acfArr = dnaDf['acf1'].values

    rules = [
        ("high_cv→log", lambda: cvArr > np.median(cvArr), 'log'),
        ("low_cv→raw", lambda: cvArr <= np.median(cvArr), 'raw'),
        ("high_seas→stl", lambda: seasArr > np.median(seasArr), 'stl'),
        ("high_trend→diff", lambda: trendArr > np.percentile(trendArr, 75), 'diff'),
        ("low_acf→ma3", lambda: acfArr < np.median(acfArr), 'ma3'),
    ]

    for name, maskFn, strat in rules:
        mask = maskFn()
        rawSubset = rawOwas[mask].mean()
        stratSubset = np.array(stratOwas[strat])[mask].mean()
        print(f"  {name:>20}: n={mask.sum():>3}, raw={rawSubset:.4f}, {strat}={stratSubset:.4f}, diff={stratSubset-rawSubset:+.4f}")

    print(f"\n  --- Combined Rule ---")
    selected = np.array(stratOwas['raw']).copy()
    stratNames = ['raw'] * len(sids)
    for j in range(len(sids)):
        cv = cvArr[j]
        seas = seasArr[j]
        if cv > np.percentile(cvArr, 75):
            selected[j] = stratOwas['log'][j]
            stratNames[j] = 'log'
        elif seas > np.percentile(seasArr, 75):
            selected[j] = stratOwas['stl'][j]
            stratNames[j] = 'stl'

    ruleOwa = np.mean(selected)
    print(f"  Rule-based: OWA={ruleOwa:.4f} vs raw={rawOwa:.4f} ({ruleOwa - rawOwa:+.4f})")

    from collections import Counter
    stratCount = Counter(stratNames)
    for s, c in stratCount.most_common():
        print(f"    {s}: {c} series")

    print(f"\n{'=' * 70}")
    print(f"=== FINAL SUMMARY ===")
    print(f"{'=' * 70}")
    print(f"  Raw DOT:        OWA={rawOwa:.4f}")
    print(f"  Oracle preproc:  OWA={oracleOwa:.4f} ({(oracleOwa-rawOwa)/rawOwa*100:+.1f}%)")
    print(f"  Rule-based:      OWA={ruleOwa:.4f} ({(ruleOwa-rawOwa)/rawOwa*100:+.1f}%)")
    print(f"  Gap captured:    {(rawOwa - ruleOwa) / (rawOwa - oracleOwa) * 100:.1f}%")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"{'=' * 70}")
