"""
실험 ID: foundationAttack/009
실험명: Learned Model Selection — DNA → 최적 모델 분류

목적:
- 008에서 수집한 DNA + Oracle 레이블로 "DNA → 최적 모델" 분류기를 학습
- Ridge(선형) vs Gradient Boosted Trees(비선형) 비교
- 학습된 선택이 단일 모델/랜덤 선택 대비 MASE를 얼마나 낮추는지 측정
- Phase 2의 핵심 질문: "DNA만으로 최적 모델을 고를 수 있는가?"

가설:
1. GBT 선택 정확도 > Ridge 정확도 (비선형 상호작용 캡처)
2. GBT 선택 MASE < 최고 단일 모델(DOT) MASE
3. MASE 기준, GBT 선택은 Oracle의 50%+ 갭을 캡처

방법:
1. 008의 multi_model_oracle.json 로드
2. 증강 66특성 (007과 동일) → Oracle 모델 레이블
3. Ridge Classifier + 순수 GBT(from scratch) 5-Fold CV
4. 선택 정확도 + 실제 MASE로 평가

데이터 리니지:
- 출처: 008 실험 결과 (data/gift_eval/multi_model_oracle.json)
- 시리즈 수: 704개
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

MODEL_IDS = ["dot", "auto_ets", "auto_ces", "four_theta"]

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

INTERACTION_PAIRS = [
    ("seasonalStrength", "acf1"),
    ("trendStrength", "volatility"),
    ("forecastability", "hurstExponent"),
    ("multiSeasonalScore", "seasonalAutoCorr"),
    ("cv", "approximateEntropy"),
    ("stabilityMean", "seasonalStrength"),
    ("trendStrength", "seasonalStrength"),
    ("volatilityClustering", "garchEffect"),
]


def buildFeatures(r):
    row = []
    for fn in BASE_FEATURES:
        val = r["features"].get(fn, 0.0)
        if not np.isfinite(val):
            val = 0.0
        row.append(val)

    for fCat in FREQ_CATEGORIES:
        row.append(1.0 if r["freqCat"] == fCat else 0.0)

    for f1, f2 in INTERACTION_PAIRS:
        v1 = r["features"].get(f1, 0.0)
        v2 = r["features"].get(f2, 0.0)
        if not np.isfinite(v1): v1 = 0.0
        if not np.isfinite(v2): v2 = 0.0
        row.append(v1 * v2)

    row.append(np.log1p(r["period"]))
    row.append(np.log1p(r["length"]))
    row.append(r["period"] / max(r["length"], 1))

    return row


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
    mean = np.mean(XTrain, axis=0)
    std = np.std(XTrain, axis=0)
    std[std < 1e-10] = 1.0
    XTrainN = (XTrain - mean) / std
    XTestN = (XTest - mean) / std
    YOneHot = np.zeros((len(yTrain), nClasses))
    for i, c in enumerate(yTrain):
        YOneHot[i, c] = 1.0
    I = np.eye(XTrainN.shape[1])
    W = np.linalg.solve(XTrainN.T @ XTrainN + alpha * I, XTrainN.T @ YOneHot)
    scores = XTestN @ W
    return np.argmax(scores, axis=1)


class GradientBoostedClassifier:
    def __init__(self, nTrees=200, maxDepth=4, lr=0.1, subsample=0.8, seed=42):
        self.nTrees = nTrees
        self.maxDepth = maxDepth
        self.lr = lr
        self.subsample = subsample
        self.seed = seed
        self.trees = []
        self.nClasses = 0

    def _gini(self, y, nClasses):
        if len(y) == 0:
            return 0.0
        return 1.0 - sum((np.sum(y == c) / len(y)) ** 2 for c in range(nClasses))

    def _buildTree(self, X, residuals, rng, depth=0):
        n, nFeats = X.shape
        if depth >= self.maxDepth or n < 5:
            return float(np.mean(residuals))

        bestGain = -1
        bestFeat = None
        bestThresh = None

        featIdx = rng.choice(nFeats, size=min(int(np.sqrt(nFeats)) + 1, nFeats), replace=False)

        parentVar = np.var(residuals) * n

        for fi in featIdx:
            vals = np.unique(X[:, fi])
            if len(vals) <= 1:
                continue
            thresholds = vals[::max(1, len(vals) // 10)]
            for th in thresholds:
                leftMask = X[:, fi] <= th
                rightMask = ~leftMask
                nL = np.sum(leftMask)
                nR = np.sum(rightMask)
                if nL < 2 or nR < 2:
                    continue
                varL = np.var(residuals[leftMask]) * nL
                varR = np.var(residuals[rightMask]) * nR
                gain = parentVar - varL - varR
                if gain > bestGain:
                    bestGain = gain
                    bestFeat = fi
                    bestThresh = th

        if bestFeat is None:
            return float(np.mean(residuals))

        leftMask = X[:, bestFeat] <= bestThresh
        rightMask = ~leftMask

        return {
            "feat": bestFeat,
            "thresh": bestThresh,
            "left": self._buildTree(X[leftMask], residuals[leftMask], rng, depth + 1),
            "right": self._buildTree(X[rightMask], residuals[rightMask], rng, depth + 1),
        }

    def _predictTree(self, tree, x):
        if isinstance(tree, float):
            return tree
        if x[tree["feat"]] <= tree["thresh"]:
            return self._predictTree(tree["left"], x)
        return self._predictTree(tree["right"], x)

    def fit(self, X, y):
        self.nClasses = len(np.unique(y))
        n = len(X)
        rng = np.random.RandomState(self.seed)

        F = np.zeros((n, self.nClasses))
        classCounts = np.bincount(y, minlength=self.nClasses).astype(float)
        self.initProbs = np.log(classCounts / n + 1e-10)
        for c in range(self.nClasses):
            F[:, c] = self.initProbs[c]

        self.trees = []

        for t in range(self.nTrees):
            probs = np.exp(F)
            probs /= probs.sum(axis=1, keepdims=True)

            treesForRound = []
            subIdx = rng.choice(n, size=int(n * self.subsample), replace=False)

            for c in range(self.nClasses):
                targets = (y == c).astype(float)
                residuals = targets - probs[:, c]

                tree = self._buildTree(X[subIdx], residuals[subIdx], rng)
                treesForRound.append(tree)

                for i in range(n):
                    F[i, c] += self.lr * self._predictTree(tree, X[i])

            self.trees.append(treesForRound)

        return self

    def predict(self, X):
        n = len(X)
        F = np.zeros((n, self.nClasses))
        for c in range(self.nClasses):
            F[:, c] = self.initProbs[c]

        for treesForRound in self.trees:
            for c, tree in enumerate(treesForRound):
                for i in range(n):
                    F[i, c] += self.lr * self._predictTree(tree, X[i])

        return np.argmax(F, axis=1)


def evaluateSelection(allResults, yPred, modelIdxToId):
    selectedMases = []
    oracleMases = []
    dotMases = []

    for i, r in enumerate(allResults):
        selectedModel = modelIdxToId[yPred[i]]
        if selectedModel in r["mases"]:
            selectedMases.append(r["mases"][selectedModel])
        else:
            selectedMases.append(r["mases"].get("dot", 999))

        oracleMases.append(r["oracleMase"])
        dotMases.append(r["mases"].get("dot", 999))

    selectedMean = np.mean(selectedMases)
    oracleMean = np.mean(oracleMases)
    dotMean = np.mean(dotMases)

    gapTotal = dotMean - oracleMean
    gapCaptured = dotMean - selectedMean
    captureRate = gapCaptured / gapTotal * 100 if gapTotal > 0 else 0

    return selectedMean, oracleMean, dotMean, captureRate


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 90, flush=True)
    print("Learned Model Selection — Phase 2, Experiment 009", flush=True)
    print("=" * 90, flush=True)

    path = GIFT_EVAL_DIR / "multi_model_oracle.json"
    with open(path, "r", encoding="utf-8") as f:
        allResults = json.load(f)
    print(f"\n[로드] {len(allResults)}개 시리즈 from {path}", flush=True)

    validResults = [r for r in allResults if r["oracleModel"] in MODEL_IDS]
    print(f"[필터] auto_croston 제외 → {len(validResults)}개 시리즈", flush=True)

    modelIdToIdx = {mid: i for i, mid in enumerate(MODEL_IDS)}
    modelIdxToId = {i: mid for i, mid in enumerate(MODEL_IDS)}

    X = np.array([buildFeatures(r) for r in validResults])
    y = np.array([modelIdToIdx[r["oracleModel"]] for r in validResults])

    print(f"  특성: {X.shape[1]}개, 클래스: {len(MODEL_IDS)}개", flush=True)
    for mid in MODEL_IDS:
        count = np.sum(y == modelIdToIdx[mid])
        print(f"  {mid}: {count}개 ({100*count/len(y):.1f}%)", flush=True)

    folds = stratifiedKFold(y, nFolds=5, seed=42)

    ridgeAccs = []
    gbtAccs = []
    ridgePreds = np.zeros(len(y), dtype=int)
    gbtPreds = np.zeros(len(y), dtype=int)

    print("\n[학습] 5-Fold CV", flush=True)
    for foldIdx, (trainIdx, testIdx) in enumerate(folds):
        XTrain, yTrain = X[trainIdx], y[trainIdx]
        XTest, yTest = X[testIdx], y[testIdx]

        rPred = ridgeClassifier(XTrain, yTrain, XTest, alpha=10.0)
        ridgeAccs.append(np.mean(yTest == rPred))
        ridgePreds[testIdx] = rPred

        gbt = GradientBoostedClassifier(nTrees=150, maxDepth=4, lr=0.1, subsample=0.8, seed=42 + foldIdx)
        gbt.fit(XTrain, yTrain)
        gPred = gbt.predict(XTest)
        gbtAccs.append(np.mean(yTest == gPred))
        gbtPreds[testIdx] = gPred

        print(f"  Fold {foldIdx+1}: Ridge={ridgeAccs[-1]:.3f}, GBT={gbtAccs[-1]:.3f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("분류 정확도", flush=True)
    print("=" * 90, flush=True)
    print(f"  Ridge:  {np.mean(ridgeAccs):.4f} ± {np.std(ridgeAccs):.4f}", flush=True)
    print(f"  GBT:    {np.mean(gbtAccs):.4f} ± {np.std(gbtAccs):.4f}", flush=True)
    print(f"  랜덤:   {1/len(MODEL_IDS):.4f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("MASE 기반 실질 평가", flush=True)
    print("=" * 90, flush=True)

    rSel, oMean, dMean, rCapture = evaluateSelection(validResults, ridgePreds, modelIdxToId)
    gSel, _, _, gCapture = evaluateSelection(validResults, gbtPreds, modelIdxToId)

    print(f"\n  {'전략':<20s} | {'MASE':>8s} | {'vs DOT':>8s} | {'Oracle Gap 캡처':>15s}", flush=True)
    print("  " + "-" * 65, flush=True)
    print(f"  {'DOT (단일 최강)':<20s} | {dMean:8.3f} | {'기준':>8s} | {'0%':>15s}", flush=True)
    print(f"  {'Ridge 선택':<20s} | {rSel:8.3f} | {(dMean-rSel)/dMean*100:+7.1f}% | {rCapture:14.1f}%", flush=True)
    print(f"  {'GBT 선택':<20s} | {gSel:8.3f} | {(dMean-gSel)/dMean*100:+7.1f}% | {gCapture:14.1f}%", flush=True)
    print(f"  {'Oracle (상한)':<20s} | {oMean:8.3f} | {(dMean-oMean)/dMean*100:+7.1f}% | {'100%':>15s}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("도메인별 GBT 선택 vs DOT", flush=True)
    print("=" * 90, flush=True)

    domainResults = defaultdict(list)
    domainPreds = defaultdict(list)
    for i, r in enumerate(validResults):
        domainResults[r["domain"]].append(r)
        domainPreds[r["domain"]].append(gbtPreds[i])

    for domain in sorted(domainResults.keys()):
        drs = domainResults[domain]
        dpreds = domainPreds[domain]
        dDot = np.mean([r["mases"].get("dot", 999) for r in drs])
        dOracle = np.mean([r["oracleMase"] for r in drs])

        dSel = []
        for r, pred in zip(drs, dpreds):
            mid = modelIdxToId[pred]
            dSel.append(r["mases"].get(mid, r["mases"].get("dot", 999)))
        dSelMean = np.mean(dSel)

        imp = (dDot - dSelMean) / dDot * 100
        print(f"  {domain:<14s} | DOT={dDot:.3f} | GBT={dSelMean:.3f} | Oracle={dOracle:.3f} | vs DOT: {imp:+.1f}%", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("결론", flush=True)
    print("=" * 90, flush=True)

    if gCapture > 50:
        print(f"  → GBT 선택이 Oracle gap의 {gCapture:.1f}% 캡처! Learned Selection 가치 높음", flush=True)
    elif gCapture > 25:
        print(f"  → GBT 선택이 Oracle gap의 {gCapture:.1f}% 캡처. 개선 여지 있지만 가치 있음", flush=True)
    else:
        print(f"  → GBT 선택이 Oracle gap의 {gCapture:.1f}%만 캡처. 추가 특성/모델 필요", flush=True)

    if gSel < dMean:
        print(f"  → GBT MASE({gSel:.3f}) < DOT MASE({dMean:.3f}) — 학습된 선택이 단일 모델 격파!", flush=True)
    else:
        print(f"  → GBT MASE({gSel:.3f}) >= DOT MASE({dMean:.3f}) — 아직 단일 모델을 못 이김", flush=True)
