"""
실험 ID: foundationAttack/006
실험명: DNA 특성 선별 — 모델 성능과 상관 높은 특성 추출

목적:
- DNA 65+ 특성 중 DOT MASE와 상관이 높은 특성을 식별한다
- "이 특성이 높으면 DOT가 잘 맞힌다" / "이 특성이 높으면 DOT가 틀린다"를 찾는다
- 이를 통해 Learned Selection의 입력 특성을 정제한다

가설:
1. forecastability, seasonalStrength, trendStrength가 MASE와 음의 상관 (높을수록 쉬움)
2. volatilityClustering, approximateEntropy가 MASE와 양의 상관 (높을수록 어려움)
3. Top-10 특성으로 MASE 분산의 30%+ 설명 가능

방법:
1. 004의 DNA 프로파일 + 002의 DOT MASE를 시리즈 단위로 결합
2. DOT를 각 시리즈에서 다시 돌려 시리즈별 MASE 확보 (002와 동일한 설정)
3. Pearson/Spearman 상관계수 계산
4. Ridge 회귀로 DNA → MASE 예측 (R² 측정)
5. Top-10 특성 식별

데이터 리니지:
- 출처: GIFT-Eval + 004 DNA 프로파일
- 시리즈 수: 004와 동일 (데이터셋당 50개)
- 학습/검증 분할: 5-Fold CV
- 시드: 42

결과 (실험 후 작성):
- (아래에 기록)

결론:
- (실험 후 작성)

실험일: 2026-03-05
"""

import sys
import io
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

GIFT_EVAL_DIR = Path("data/gift_eval")

FREQ_TO_PERIOD = {
    "Y": 1, "A": 1, "A-DEC": 1,
    "Q": 4, "QS": 4, "Q-DEC": 4,
    "M": 12, "MS": 12,
    "W": 52, "W-MON": 52, "W-SUN": 52, "W-FRI": 52, "W-THU": 52, "W-TUE": 52, "W-WED": 52,
    "D": 7, "B": 5,
    "H": 24, "h": 24,
    "5T": 288, "5min": 288, "10T": 144, "10min": 144, "15T": 96, "15min": 96,
    "10S": 8640, "10s": 8640,
}

M4_PRED = {"Y": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
STD_PRED = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}

FEATURE_NAMES = [
    "trendStrength", "trendSlope", "trendLinearity", "trendCurvature",
    "seasonalStrength", "seasonalPeakPeriod", "seasonalAmplitude",
    "seasonalPhaseConsistency", "seasonalAutoCorr", "multiSeasonalScore",
    "acf1", "acf2", "acf3", "acfSum5", "acfDecayRate", "pacfLag1",
    "approximateEntropy", "turningPointRate", "nonlinearAutocorr", "asymmetry",
    "stabilityMean", "stabilityVariance", "levelShiftCount", "structuralBreakScore",
    "hurstExponent", "diffStationary", "spectralEntropy", "forecastability",
    "signalToNoise", "regularityIndex",
    "zeroRatio", "cv", "cv2", "demandDensity",
    "volatility", "volatilityClustering", "garchEffect", "extremeValueRatio", "tailIndex",
    "flatSpotRate", "crossingRate", "peakCount", "longestRun", "binEntropy",
    "ljungBoxStat", "kurtosis", "skewness",
]


def getPeriod(freq):
    freq = str(freq).strip()
    for key in sorted(FREQ_TO_PERIOD.keys(), key=len, reverse=True):
        if freq == key or freq.startswith(key):
            return FREQ_TO_PERIOD[key]
    return 1


def getPredLength(dsName, freq):
    freqKey = freq[0] if len(freq) > 0 else "D"
    for k in ["5T", "10T", "15T", "5min", "10min", "15min", "10S", "10s"]:
        if k in freq:
            freqKey = "T"
            break
    if dsName.startswith("m4_"):
        return M4_PRED.get(freqKey, 12)
    return STD_PRED.get(freqKey, 12)


def computeMASE(actual, predicted, insample, period):
    n = len(insample)
    if n <= period:
        naiveErr = np.mean(np.abs(np.diff(insample)))
    else:
        naiveErr = np.mean(np.abs(insample[period:] - insample[:-period]))
    if naiveErr < 1e-10:
        return np.nan
    return np.mean(np.abs(actual - predicted)) / naiveErr


def extractDnaAndMase(dsName, maxSeries=50, seed=42):
    import datasets as hfDatasets
    from vectrix.adaptive.dna import ForecastDNA
    from vectrix.engine.dot import DynamicOptimizedTheta

    dsPath = GIFT_EVAL_DIR / dsName
    if not dsPath.exists():
        return []

    ds = hfDatasets.load_from_disk(str(dsPath)).with_format("numpy")
    nTotal = len(ds)
    freq = str(ds[0].get("freq", "D"))
    period = getPeriod(freq)
    predLength = getPredLength(dsName, freq)

    rng = np.random.RandomState(seed)
    indices = rng.choice(nTotal, size=min(maxSeries, nTotal), replace=False)
    indices.sort()

    dna = ForecastDNA()
    results = []

    for idx in indices:
        entry = ds[int(idx)]
        target = entry["target"]
        if target.ndim > 1:
            target = target[0]
        y = target.astype(np.float64)

        if np.any(np.isnan(y)):
            for i in range(len(y)):
                if np.isnan(y[i]):
                    y[i] = y[i - 1] if i > 0 else 0.0

        if len(y) < predLength + 20:
            continue

        MAX_LEN = 5000
        if len(y) > MAX_LEN:
            y = y[-MAX_LEN:]

        safePeriod = min(period, len(y) // 3)
        if safePeriod < 1:
            safePeriod = 1

        trainY = y[:-predLength]
        testY = y[-predLength:]

        try:
            profile = dna.analyze(trainY, period=safePeriod)

            model = DynamicOptimizedTheta(period=safePeriod)
            model.fit(trainY)
            pred, _, _ = model.predict(predLength)

            validMask = ~np.isnan(testY)
            if np.sum(validMask) < 1:
                continue

            mase = computeMASE(testY[validMask], pred[validMask], trainY, safePeriod)
            if np.isnan(mase) or not np.isfinite(mase):
                continue

            results.append({
                "dataset": dsName,
                "seriesIdx": int(idx),
                "features": profile.features,
                "mase": float(mase),
            })
        except (ValueError, RuntimeError):
            pass

    return results


def computeCorrelations(allResults):
    X = []
    mases = []

    for r in allResults:
        row = []
        for fn in FEATURE_NAMES:
            val = r["features"].get(fn, 0.0)
            if not np.isfinite(val):
                val = 0.0
            row.append(val)
        X.append(row)
        mases.append(r["mase"])

    X = np.array(X)
    mases = np.array(mases)

    logMase = np.log1p(mases)

    print(f"\n  시리즈 수: {len(X)}")
    print(f"  MASE 평균: {np.mean(mases):.3f}, 중앙: {np.median(mases):.3f}")
    print(f"  log(1+MASE) 평균: {np.mean(logMase):.3f}")

    pearsonCorrs = []
    spearmanCorrs = []

    for i, fn in enumerate(FEATURE_NAMES):
        col = X[:, i]
        pCorr = np.corrcoef(col, logMase)[0, 1] if np.std(col) > 1e-10 else 0
        pearsonCorrs.append(pCorr)

        ranks1 = np.argsort(np.argsort(col)).astype(float)
        ranks2 = np.argsort(np.argsort(logMase)).astype(float)
        sCorr = np.corrcoef(ranks1, ranks2)[0, 1] if np.std(ranks1) > 1e-10 else 0
        spearmanCorrs.append(sCorr)

    pearsonCorrs = np.array(pearsonCorrs)
    spearmanCorrs = np.array(spearmanCorrs)

    print("\n" + "=" * 90)
    print("DNA 특성 × log(1+MASE) 상관계수 (|Spearman| 내림차순 Top-20)")
    print("=" * 90)
    print(f"\n  {'Rank':>4s} {'Feature':<25s} | {'Pearson':>8s} | {'Spearman':>8s} | {'해석':<20s}")
    print("  " + "-" * 80)

    sortIdx = np.argsort(np.abs(spearmanCorrs))[::-1]

    for rank, idx in enumerate(sortIdx[:20], 1):
        fn = FEATURE_NAMES[idx]
        pC = pearsonCorrs[idx]
        sC = spearmanCorrs[idx]

        if sC > 0.15:
            interp = "높을수록 DOT 어려움"
        elif sC < -0.15:
            interp = "높을수록 DOT 쉬움"
        else:
            interp = "약한 관계"

        print(f"  {rank:>4d} {fn:<25s} | {pC:+8.4f} | {sC:+8.4f} | {interp}")

    return X, logMase, pearsonCorrs, spearmanCorrs


def ridgeRegression(X, y, nFolds=5, seed=42):
    print("\n" + "=" * 90)
    print("Ridge 회귀: DNA → log(1+MASE) 예측")
    print("=" * 90)

    n = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    foldSize = n // nFolds

    r2Scores = []
    maeScores = []

    for fold in range(nFolds):
        testIdx = indices[fold * foldSize:(fold + 1) * foldSize]
        trainIdx = np.concatenate([indices[:fold * foldSize], indices[(fold + 1) * foldSize:]])

        XTrain, yTrain = X[trainIdx], y[trainIdx]
        XTest, yTest = X[testIdx], y[testIdx]

        mean = np.mean(XTrain, axis=0)
        std = np.std(XTrain, axis=0)
        std[std < 1e-10] = 1.0
        XTrainN = (XTrain - mean) / std
        XTestN = (XTest - mean) / std

        yMean = np.mean(yTrain)
        yTrainC = yTrain - yMean

        alpha = 10.0
        I = np.eye(XTrainN.shape[1])
        W = np.linalg.solve(XTrainN.T @ XTrainN + alpha * I, XTrainN.T @ yTrainC)

        yPred = XTestN @ W + yMean

        ssRes = np.sum((yTest - yPred) ** 2)
        ssTot = np.sum((yTest - np.mean(yTest)) ** 2)
        r2 = 1 - ssRes / ssTot if ssTot > 0 else 0
        mae = np.mean(np.abs(yTest - yPred))

        r2Scores.append(r2)
        maeScores.append(mae)

    print(f"\n  5-Fold CV R²: {np.mean(r2Scores):.4f} ± {np.std(r2Scores):.4f}")
    print(f"  5-Fold CV MAE: {np.mean(maeScores):.4f} ± {np.std(maeScores):.4f}")

    if np.mean(r2Scores) > 0.3:
        print("  → DNA가 MASE 분산의 30%+ 설명 — Learned Selection 가치 높음")
    elif np.mean(r2Scores) > 0.15:
        print("  → DNA가 MASE 분산의 15-30% 설명 — 개선 여지 있지만 기반은 됨")
    else:
        print("  → DNA가 MASE 분산을 잘 설명 못함 — 추가 특성 필요")

    return np.mean(r2Scores)


def analyzeTopFeatures(pearsonCorrs, spearmanCorrs):
    print("\n" + "=" * 90)
    print("Top-10 특성 (Learned Selection 입력 후보)")
    print("=" * 90)

    sortIdx = np.argsort(np.abs(spearmanCorrs))[::-1][:10]

    print(f"\n  {'Rank':>4s} {'Feature':<25s} | {'|Spearman|':>10s} | {'방향':<10s}")
    print("  " + "-" * 60)
    for rank, idx in enumerate(sortIdx, 1):
        fn = FEATURE_NAMES[idx]
        sC = spearmanCorrs[idx]
        direction = "MASE↑" if sC > 0 else "MASE↓"
        print(f"  {rank:>4d} {fn:<25s} | {abs(sC):10.4f} | {direction}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 90)
    print("DNA-MASE Correlation — Phase 1, Experiment 006")
    print("=" * 90)

    SHORT_DATASETS = (
        "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
        "electricity/15T electricity/H electricity/D electricity/W "
        "solar/10T solar/H solar/D solar/W "
        "hospital covid_deaths "
        "us_births/D us_births/M us_births/W "
        "saugeenday/D saugeenday/M saugeenday/W "
        "temperature_rain_with_missing "
        "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
        "car_parts_with_missing restaurant "
        "hierarchical_sales/D hierarchical_sales/W "
        "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
        "SZ_TAXI/15T SZ_TAXI/H "
        "M_DENSE/H M_DENSE/D "
        "ett1/15T ett1/H ett1/D ett1/W "
        "ett2/15T ett2/H ett2/D ett2/W "
        "jena_weather/10T jena_weather/H jena_weather/D "
        "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
        "bitbrains_rnd/5T bitbrains_rnd/H "
        "bizitobs_application bizitobs_service "
        "bizitobs_l2c/5T bizitobs_l2c/H"
    )

    MAX_SERIES = 20

    print(f"\n[설정] 데이터셋당 최대 {MAX_SERIES}개 시리즈 (DNA + DOT MASE 동시 추출)")

    datasets = SHORT_DATASETS.split()
    allResults = []

    print(f"\n{'Dataset':<45s} | {'N':>4s} | {'MASE mean':>9s} | {'Time':>6s}", flush=True)
    print("-" * 80, flush=True)

    for dsName in sorted(datasets):
        t0 = time.time()
        results = extractDnaAndMase(dsName, maxSeries=MAX_SERIES)
        elapsed = time.time() - t0

        if results:
            allResults.extend(results)
            meanMase = np.mean([r["mase"] for r in results])
            print(f"  {dsName:<43s} | {len(results):>4d} | {meanMase:9.3f} | {elapsed:5.1f}s", flush=True)

    print(f"\n총 {len(allResults)}개 시리즈")

    X, logMase, pCorrs, sCorrs = computeCorrelations(allResults)
    r2 = ridgeRegression(X, logMase)
    analyzeTopFeatures(pCorrs, sCorrs)
