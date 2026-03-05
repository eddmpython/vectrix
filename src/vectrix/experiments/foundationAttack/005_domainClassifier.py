"""
실험 ID: foundationAttack/005
실험명: 도메인-DNA 매핑 학습 (DNA → 도메인 분류)

목적:
- DNA 65+ 특성만으로 시리즈의 도메인을 분류할 수 있는지 검증한다
- 분류 정확도가 높으면 → DNA가 도메인 특성을 잘 캡처한다는 증거
- 도메인 분류가 가능하면 → 도메인별 모델 선택 전략의 기반이 된다

가설:
1. DNA 특성으로 7개 도메인 분류 정확도 > 70% (random 14.3%)
2. 에너지/교통 도메인이 가장 잘 분류될 것 (계절성 패턴 뚜렷)
3. Econ/Fin은 다양한 패턴이 섞여 분류가 어려울 것

방법:
1. 004에서 생성한 dna_profiles.json 로드
2. 65+ DNA 특성을 입력, 도메인을 레이블로 분류
3. Ridge, RandomForest, XGBoost 3가지 모델 비교
4. Stratified 5-Fold CV로 평가
5. 도메인별 Precision/Recall + 혼동 행렬
6. 특성 중요도 Top-10 식별

데이터 리니지:
- 출처: 004 실험 결과 (data/gift_eval/dna_profiles.json)
- 시리즈 수: 004 실행 결과에 따름
- 학습/검증 분할: Stratified 5-Fold CV
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
from pathlib import Path
from collections import defaultdict

import numpy as np

GIFT_EVAL_DIR = Path("data/gift_eval")

FEATURE_NAMES = [
    "length", "mean", "std", "cv", "skewness", "kurtosis",
    "min", "max", "iqr", "rangeToMeanRatio",
    "trendStrength", "trendSlope", "trendLinearity", "trendCurvature", "trendDirection",
    "seasonalStrength", "seasonalPeakPeriod", "seasonalAmplitude",
    "seasonalPhaseConsistency", "seasonalHarmonicRatio", "seasonalAutoCorr",
    "seasonalAdjustedVariance", "multiSeasonalScore",
    "acf1", "acf2", "acf3", "acfSum5", "acfDecayRate",
    "pacfLag1", "acfFirstZero", "ljungBoxStat",
    "approximateEntropy", "turningPointRate", "thirdOrderAutoCorr",
    "asymmetry", "nonlinearAutocorr",
    "stabilityMean", "stabilityVariance", "levelShiftCount",
    "levelShiftMagnitude", "structuralBreakScore",
    "adfStatistic", "diffStationary", "hurstExponent", "unitRootIndicator",
    "spectralEntropy", "forecastability", "signalToNoise",
    "sampleEntropy", "regularityIndex",
    "zeroRatio", "adi", "cv2", "intermittencyType", "demandDensity",
    "volatility", "volatilityClustering", "garchEffect",
    "extremeValueRatio", "tailIndex",
    "flatSpotRate", "crossingRate", "peakCount", "longestRun", "binEntropy",
]


def loadProfiles():
    path = GIFT_EVAL_DIR / "dna_profiles.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[로드] {len(data)}개 프로파일 from {path}")
    return data


def buildDataset(profiles):
    X = []
    y = []
    domains = []

    for p in profiles:
        feats = p["features"]
        row = []
        valid = True
        for fn in FEATURE_NAMES:
            val = feats.get(fn, 0.0)
            if not np.isfinite(val):
                val = 0.0
            row.append(val)
        X.append(row)
        y.append(p["domain"])
        domains.append(p["domain"])

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    domainList = sorted(set(y))
    domainToIdx = {d: i for i, d in enumerate(domainList)}
    yIdx = np.array([domainToIdx[d] for d in y])

    print(f"\n  데이터셋: {X.shape[0]} 시리즈 × {X.shape[1]} 특성")
    print(f"  도메인 수: {len(domainList)}")
    for d in domainList:
        count = np.sum(y == d)
        print(f"    {d}: {count}개 ({100*count/len(y):.1f}%)")

    return X, yIdx, domainList


def stratifiedKFold(y, nFolds=5, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y)
    indices = np.arange(n)
    classes = np.unique(y)

    folds = [[] for _ in range(nFolds)]

    for c in classes:
        cIdx = indices[y == c]
        rng.shuffle(cIdx)
        for i, idx in enumerate(cIdx):
            folds[i % nFolds].append(idx)

    for f in folds:
        rng.shuffle(f)

    splits = []
    for i in range(nFolds):
        testIdx = np.array(folds[i])
        trainIdx = np.concatenate([np.array(folds[j]) for j in range(nFolds) if j != i])
        splits.append((trainIdx, testIdx))

    return splits


def ridgeClassifier(XTrain, yTrain, XTest, alpha=1.0):
    nClasses = len(np.unique(yTrain))
    nFeatures = XTrain.shape[1]

    mean = np.mean(XTrain, axis=0)
    std = np.std(XTrain, axis=0)
    std[std < 1e-10] = 1.0
    XTrainN = (XTrain - mean) / std
    XTestN = (XTest - mean) / std

    YOneHot = np.zeros((len(yTrain), nClasses))
    for i, c in enumerate(yTrain):
        YOneHot[i, c] = 1.0

    I = np.eye(nFeatures)
    W = np.linalg.solve(XTrainN.T @ XTrainN + alpha * I, XTrainN.T @ YOneHot)

    scores = XTestN @ W
    return np.argmax(scores, axis=1), W


def decisionStump(X, y, featureIdx, threshold):
    leftMask = X[:, featureIdx] <= threshold
    rightMask = ~leftMask
    return leftMask, rightMask


def simpleRandomForest(XTrain, yTrain, XTest, nTrees=100, maxDepth=5, seed=42):
    rng = np.random.RandomState(seed)
    nClasses = len(np.unique(yTrain))
    nSamples, nFeatures = XTrain.shape
    predictions = np.zeros((len(XTest), nClasses))

    for t in range(nTrees):
        bootIdx = rng.choice(nSamples, nSamples, replace=True)
        Xb = XTrain[bootIdx]
        yb = yTrain[bootIdx]

        featIdx = rng.choice(nFeatures, size=min(int(np.sqrt(nFeatures)) + 1, nFeatures), replace=False)

        bestFeat = None
        bestThresh = None
        bestGini = float("inf")

        for fi in featIdx:
            vals = np.unique(Xb[:, fi])
            if len(vals) <= 1:
                continue
            thresholds = rng.choice(vals, size=min(5, len(vals)), replace=False)
            for th in thresholds:
                left = yb[Xb[:, fi] <= th]
                right = yb[Xb[:, fi] > th]
                if len(left) == 0 or len(right) == 0:
                    continue
                giniLeft = 1.0 - sum((np.sum(left == c) / len(left)) ** 2 for c in range(nClasses))
                giniRight = 1.0 - sum((np.sum(right == c) / len(right)) ** 2 for c in range(nClasses))
                gini = (len(left) * giniLeft + len(right) * giniRight) / len(yb)
                if gini < bestGini:
                    bestGini = gini
                    bestFeat = fi
                    bestThresh = th

        if bestFeat is None:
            classCounts = np.bincount(yb, minlength=nClasses).astype(float)
            predictions += classCounts / classCounts.sum()
            continue

        testLeft = XTest[:, bestFeat] <= bestThresh
        testRight = ~testLeft

        leftY = yb[Xb[:, bestFeat] <= bestThresh]
        rightY = yb[Xb[:, bestFeat] > bestThresh]

        if len(leftY) > 0:
            leftCounts = np.bincount(leftY, minlength=nClasses).astype(float)
            predictions[testLeft] += leftCounts / leftCounts.sum()
        if len(rightY) > 0:
            rightCounts = np.bincount(rightY, minlength=nClasses).astype(float)
            predictions[testRight] += rightCounts / rightCounts.sum()

    return np.argmax(predictions, axis=1)


def evaluate(yTrue, yPred, domainList):
    acc = np.mean(yTrue == yPred)

    nClasses = len(domainList)
    confMatrix = np.zeros((nClasses, nClasses), dtype=int)
    for t, p in zip(yTrue, yPred):
        confMatrix[t, p] += 1

    print(f"\n  정확도: {acc:.4f} ({100*acc:.1f}%)")
    print(f"  (랜덤 기준: {100/nClasses:.1f}%)")

    print(f"\n  {'Domain':<14s} | {'Precision':>9s} | {'Recall':>9s} | {'F1':>9s} | {'Support':>7s}")
    print("  " + "-" * 60)

    for i, d in enumerate(domainList):
        tp = confMatrix[i, i]
        fp = np.sum(confMatrix[:, i]) - tp
        fn = np.sum(confMatrix[i, :]) - tp
        support = np.sum(confMatrix[i, :])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {d:<14s} | {precision:9.3f} | {recall:9.3f} | {f1:9.3f} | {support:7d}")

    return acc


def runExperiment():
    profiles = loadProfiles()
    X, y, domainList = buildDataset(profiles)

    folds = stratifiedKFold(y, nFolds=5, seed=42)

    ridgeAccs = []
    rfAccs = []
    allRidgePreds = np.zeros_like(y)
    allRfPreds = np.zeros_like(y)
    ridgeWeights = None

    for foldIdx, (trainIdx, testIdx) in enumerate(folds):
        XTrain, yTrain = X[trainIdx], y[trainIdx]
        XTest, yTest = X[testIdx], y[testIdx]

        ridgePred, W = ridgeClassifier(XTrain, yTrain, XTest, alpha=1.0)
        ridgeAcc = np.mean(yTest == ridgePred)
        ridgeAccs.append(ridgeAcc)
        allRidgePreds[testIdx] = ridgePred
        if ridgeWeights is None:
            ridgeWeights = W

        rfPred = simpleRandomForest(XTrain, yTrain, XTest, nTrees=200, seed=42 + foldIdx)
        rfAcc = np.mean(yTest == rfPred)
        rfAccs.append(rfAcc)
        allRfPreds[testIdx] = rfPred

        print(f"  Fold {foldIdx+1}: Ridge={ridgeAcc:.3f}, RF={rfAcc:.3f}")

    print("\n" + "=" * 80)
    print("Ridge Classifier (5-Fold CV)")
    print("=" * 80)
    print(f"  평균 정확도: {np.mean(ridgeAccs):.4f} ± {np.std(ridgeAccs):.4f}")
    evaluate(y, allRidgePreds, domainList)

    print("\n" + "=" * 80)
    print("Random Forest (5-Fold CV)")
    print("=" * 80)
    print(f"  평균 정확도: {np.mean(rfAccs):.4f} ± {np.std(rfAccs):.4f}")
    evaluate(y, allRfPreds, domainList)

    if ridgeWeights is not None:
        print("\n" + "=" * 80)
        print("특성 중요도 (Ridge 가중치 절대값 합)")
        print("=" * 80)
        importance = np.sum(np.abs(ridgeWeights), axis=1)
        topIdx = np.argsort(importance)[::-1][:15]
        for rank, idx in enumerate(topIdx, 1):
            print(f"  {rank:>2d}. {FEATURE_NAMES[idx]:<25s}: {importance[idx]:.4f}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 80)
    print("Domain Classifier from DNA — Phase 1, Experiment 005")
    print("=" * 80)

    runExperiment()
