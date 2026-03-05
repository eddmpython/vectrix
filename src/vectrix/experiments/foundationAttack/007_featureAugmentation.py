"""
실험 ID: foundationAttack/007
실험명: DNA 특성 증강 — 기존 47개 + 새 파생 특성으로 MASE 예측력 개선

목적:
- 006에서 Ridge R²=0.169 (DNA 47특성 → MASE) — 선형으로는 부족
- 기존 특성에서 교차 특성(interaction), 빈도 인코딩, 비선형 변환을 추가
- 증강된 특성으로 R²가 개선되는지 검증
- Phase 2 입력에 쓸 최종 특성 세트를 확정

가설:
1. 교차 특성(seasonalStrength × acf1 등)으로 R² > 0.25
2. 빈도 원핫 인코딩 추가로 R² > 0.20
3. Top-10 특성 선별 후 Ridge R²가 전체 특성과 비슷 (과적합 없음)

방법:
1. 004의 dna_profiles.json 로드 + 002의 DOT baseline에서 MASE 매칭
2. 기존 47특성 + 교차/빈도/변환 특성 = ~100개
3. Ridge 5-Fold CV로 R² 비교 (기존 vs 증강)
4. Lasso로 자동 특성 선별 → Top-K 확정

데이터 리니지:
- 출처: 006 실험과 동일한 방식으로 DNA+MASE 추출
- 시리즈 수: 006과 동일 (20/데이터셋)
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

BASE_FEATURES = [
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

FREQ_CATEGORIES = ["Y", "Q", "M", "W", "D", "H", "T", "S"]


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


def getFreqCategory(freq):
    freq = str(freq).strip()
    for k in ["5T", "10T", "15T", "5min", "10min", "15min"]:
        if k in freq:
            return "T"
    for k in ["10S", "10s"]:
        if k in freq:
            return "S"
    if freq.startswith("H") or freq == "h":
        return "H"
    if freq.startswith("D") or freq == "B":
        return "D"
    if freq.startswith("W"):
        return "W"
    if freq.startswith("M") or freq == "MS":
        return "M"
    if freq.startswith("Q") or freq == "QS":
        return "Q"
    if freq.startswith("Y") or freq.startswith("A"):
        return "Y"
    return "D"


def computeMASE(actual, predicted, insample, period):
    n = len(insample)
    if n <= period:
        naiveErr = np.mean(np.abs(np.diff(insample)))
    else:
        naiveErr = np.mean(np.abs(insample[period:] - insample[:-period]))
    if naiveErr < 1e-10:
        return np.nan
    return np.mean(np.abs(actual - predicted)) / naiveErr


def extractDnaAndMase(dsName, maxSeries=20, seed=42):
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
    freqCat = getFreqCategory(freq)

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
                "freq": freq,
                "freqCat": freqCat,
                "period": safePeriod,
                "length": len(trainY),
                "mase": float(mase),
            })
        except (ValueError, RuntimeError):
            pass

    return results


def buildFeatureMatrix(allResults):
    nBase = len(BASE_FEATURES)
    nFreq = len(FREQ_CATEGORIES)

    interactionPairs = [
        ("seasonalStrength", "acf1"),
        ("trendStrength", "volatility"),
        ("forecastability", "hurstExponent"),
        ("multiSeasonalScore", "seasonalAutoCorr"),
        ("cv", "approximateEntropy"),
        ("stabilityMean", "seasonalStrength"),
        ("trendStrength", "seasonalStrength"),
        ("volatilityClustering", "garchEffect"),
    ]
    nInteract = len(interactionPairs)

    nAugmented = nBase + nFreq + nInteract + 3

    featureNames = list(BASE_FEATURES)
    for fCat in FREQ_CATEGORIES:
        featureNames.append(f"freq_{fCat}")
    for f1, f2 in interactionPairs:
        featureNames.append(f"{f1}_x_{f2}")
    featureNames.extend(["log_period", "log_length", "period_to_length_ratio"])

    X = []
    mases = []

    for r in allResults:
        row = []

        for fn in BASE_FEATURES:
            val = r["features"].get(fn, 0.0)
            if not np.isfinite(val):
                val = 0.0
            row.append(val)

        for fCat in FREQ_CATEGORIES:
            row.append(1.0 if r["freqCat"] == fCat else 0.0)

        for f1, f2 in interactionPairs:
            v1 = r["features"].get(f1, 0.0)
            v2 = r["features"].get(f2, 0.0)
            if not np.isfinite(v1):
                v1 = 0.0
            if not np.isfinite(v2):
                v2 = 0.0
            row.append(v1 * v2)

        row.append(np.log1p(r["period"]))
        row.append(np.log1p(r["length"]))
        ratio = r["period"] / max(r["length"], 1)
        row.append(ratio)

        X.append(row)
        mases.append(r["mase"])

    X = np.array(X)
    mases = np.array(mases)
    logMase = np.log1p(mases)

    return X, logMase, featureNames


def ridgeCV(X, y, alpha=10.0, nFolds=5, seed=42):
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

        I = np.eye(XTrainN.shape[1])
        W = np.linalg.solve(XTrainN.T @ XTrainN + alpha * I, XTrainN.T @ yTrainC)

        yPred = XTestN @ W + yMean

        ssRes = np.sum((yTest - yPred) ** 2)
        ssTot = np.sum((yTest - np.mean(yTest)) ** 2)
        r2 = 1 - ssRes / ssTot if ssTot > 0 else 0
        mae = np.mean(np.abs(yTest - yPred))

        r2Scores.append(r2)
        maeScores.append(mae)

    return np.mean(r2Scores), np.std(r2Scores), np.mean(maeScores)


def lassoCV(X, y, alpha=0.01, nFolds=5, seed=42, maxIter=1000):
    n = len(X)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    foldSize = n // nFolds

    allWeights = np.zeros(X.shape[1])
    r2Scores = []

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

        nFeats = XTrainN.shape[1]
        W = np.zeros(nFeats)

        for _ in range(maxIter):
            for j in range(nFeats):
                residual = yTrainC - XTrainN @ W + XTrainN[:, j] * W[j]
                rho = XTrainN[:, j] @ residual / len(yTrainC)
                if rho > alpha:
                    W[j] = rho - alpha
                elif rho < -alpha:
                    W[j] = rho + alpha
                else:
                    W[j] = 0.0

        allWeights += np.abs(W)

        yPred = XTestN @ W + yMean
        ssRes = np.sum((yTest - yPred) ** 2)
        ssTot = np.sum((yTest - np.mean(yTest)) ** 2)
        r2 = 1 - ssRes / ssTot if ssTot > 0 else 0
        r2Scores.append(r2)

    return np.mean(r2Scores), allWeights / nFolds


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 90, flush=True)
    print("Feature Augmentation — Phase 1, Experiment 007", flush=True)
    print("=" * 90, flush=True)

    MAX_SERIES = 20
    print(f"\n[설정] 데이터셋당 최대 {MAX_SERIES}개 시리즈", flush=True)

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

    print(f"\n총 {len(allResults)}개 시리즈", flush=True)

    XBase = []
    logMaseBase = []
    for r in allResults:
        row = []
        for fn in BASE_FEATURES:
            val = r["features"].get(fn, 0.0)
            if not np.isfinite(val):
                val = 0.0
            row.append(val)
        XBase.append(row)
        logMaseBase.append(np.log1p(r["mase"]))
    XBase = np.array(XBase)
    logMaseBase = np.array(logMaseBase)

    print("\n" + "=" * 90, flush=True)
    print("비교 실험: 기존 특성 vs 증강 특성", flush=True)
    print("=" * 90, flush=True)

    r2Base, r2BaseStd, maeBase = ridgeCV(XBase, logMaseBase, alpha=10.0)
    print(f"\n  [기존 47특성] Ridge R²: {r2Base:.4f} ± {r2BaseStd:.4f}, MAE: {maeBase:.4f}", flush=True)

    XAug, logMaseAug, augNames = buildFeatureMatrix(allResults)
    r2Aug, r2AugStd, maeAug = ridgeCV(XAug, logMaseAug, alpha=10.0)
    print(f"  [증강 {XAug.shape[1]}특성] Ridge R²: {r2Aug:.4f} ± {r2AugStd:.4f}, MAE: {maeAug:.4f}", flush=True)

    nBase = len(BASE_FEATURES)
    nFreq = len(FREQ_CATEGORIES)
    XBaseFreq = XAug[:, :nBase + nFreq]
    r2BF, r2BFStd, maeBF = ridgeCV(XBaseFreq, logMaseAug, alpha=10.0)
    print(f"  [기존+빈도 {nBase+nFreq}특성] Ridge R²: {r2BF:.4f} ± {r2BFStd:.4f}, MAE: {maeBF:.4f}", flush=True)

    print(f"\n  개선율: 기존→증강 = {(r2Aug-r2Base)/max(abs(r2Base),0.001)*100:+.1f}%", flush=True)
    print(f"  개선율: 기존→기존+빈도 = {(r2BF-r2Base)/max(abs(r2Base),0.001)*100:+.1f}%", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("Lasso 특성 선별", flush=True)
    print("=" * 90, flush=True)

    for alpha in [0.005, 0.01, 0.02]:
        r2Lasso, lassoWeights = lassoCV(XAug, logMaseAug, alpha=alpha)
        nonZero = np.sum(np.abs(lassoWeights) > 1e-6)
        print(f"\n  Lasso α={alpha}: R²={r2Lasso:.4f}, 비영 특성: {nonZero}/{len(augNames)}", flush=True)

        topIdx = np.argsort(np.abs(lassoWeights))[::-1][:15]
        for rank, idx in enumerate(topIdx[:15], 1):
            if np.abs(lassoWeights[idx]) < 1e-6:
                break
            print(f"    {rank:>2d}. {augNames[idx]:<35s}: {lassoWeights[idx]:+.6f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("Alpha 튜닝 (Ridge)", flush=True)
    print("=" * 90, flush=True)

    for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
        r2, r2Std, mae = ridgeCV(XAug, logMaseAug, alpha=alpha)
        print(f"  α={alpha:>6.1f}: R²={r2:.4f} ± {r2Std:.4f}, MAE={mae:.4f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("결론", flush=True)
    print("=" * 90, flush=True)

    if r2Aug > 0.25:
        print("  → 증강 특성으로 R² > 0.25 달성! Phase 2 XGBoost에서 큰 개선 기대", flush=True)
    elif r2Aug > r2Base + 0.02:
        print("  → 증강 특성이 기존 대비 의미 있는 개선. Phase 2 진행 가치 있음", flush=True)
    else:
        print("  → 선형 모델의 한계. Phase 2의 비선형 모델이 핵심 개선점", flush=True)
