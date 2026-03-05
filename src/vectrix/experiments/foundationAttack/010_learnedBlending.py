"""
실험 ID: foundationAttack/010
실험명: Learned Blending — DNA → 모델 블렌딩 비율 매핑

목적:
- E009의 "하나를 고르는" 분류 대신, 4모델의 가중 평균 비율을 학습
- 분류 오류의 충격 완화: 선택이 틀려도 블렌딩이면 손실이 작다
- Oracle 블렌딩의 이론적 상한도 측정

가설:
1. Oracle 블렌딩(시리즈별 최적 비율) > Oracle 선택(시리즈별 최적 1개) — MASE -3% 이상
2. 학습된 블렌딩(GBT 회귀) > 학습된 선택(GBT 분류, E009) — MASE -2% 이상
3. 학습된 블렌딩이 Oracle gap의 40%+ 캡처 (E009의 31.3%에서 +9%p)

방법:
1. E008의 multi_model_oracle.json에서 4모델 MASE 로드
2. Oracle 블렌딩: 시리즈별 MASE 최소화하는 가중치 최적화 (scipy.minimize)
3. 학습된 블렌딩: DNA 66특성 → 4개 가중치(softmax 출력) 매핑
   - Ridge 회귀: 4개 독립 Ridge → softmax 정규화
   - GBT 회귀: 4개 독립 GBT → softmax 정규화
4. 평가: 블렌딩 MASE vs 선택 MASE vs DOT vs Oracle

데이터 리니지:
- 출처: GIFT-Eval (E008 결과)
- 파일: data/gift_eval/multi_model_oracle.json
- 시리즈 수: 704개
- 특성: 증강 66개 (47 DNA + 8 freq + 8 interaction + 3 meta)
- 시드: 42

실험일: 2026-03-05
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

DATA_PATH = ROOT / "data" / "gift_eval" / "multi_model_oracle.json"

MODELS = ["dot", "auto_ets", "auto_ces", "four_theta"]

INTERACTION_PAIRS = [
    ("seasonalStrength", "acf1"),
    ("trendStrength", "volatility"),
    ("multiSeasonalScore", "seasonalAutoCorr"),
    ("forecastability", "hurstExponent"),
    ("cv", "trendStrength"),
    ("entropy", "acf1"),
    ("peakCount", "turningPointRate"),
    ("stabilityMean", "stabilityVariance"),
]

FREQ_LIST = ["Y", "Q", "M", "W", "D", "H", "T", "S"]


def buildAugmentedFeatures(dnaDict, freq):
    baseKeys = sorted(k for k in dnaDict if isinstance(dnaDict[k], (int, float)) and not np.isnan(dnaDict[k]))
    baseVals = [float(dnaDict[k]) for k in baseKeys]

    freqOnehot = [1.0 if freq.upper().startswith(f) else 0.0 for f in FREQ_LIST]

    interactions = []
    for k1, k2 in INTERACTION_PAIRS:
        v1 = float(dnaDict.get(k1, 0.0))
        v2 = float(dnaDict.get(k2, 0.0))
        interactions.append(v1 * v2)

    period = float(dnaDict.get("period", 1))
    length = float(dnaDict.get("seriesLength", 100))
    meta = [
        np.log1p(period),
        np.log1p(length),
        period / max(length, 1),
    ]

    return np.array(baseVals + freqOnehot + interactions + meta)


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / ex.sum()


def blendedMase(weights, maseArr):
    w = softmax(weights)
    return np.dot(w, maseArr)


def oracleBlendWeights(maseArr):
    from scipy.optimize import minimize

    best = None
    bestVal = float("inf")

    for _ in range(5):
        x0 = np.random.randn(len(maseArr)) * 0.5
        res = minimize(blendedMase, x0, args=(maseArr,), method="Nelder-Mead",
                       options={"maxiter": 500, "xatol": 1e-6})
        if res.fun < bestVal:
            bestVal = res.fun
            best = softmax(res.x)

    return best, bestVal


class GradientBoostedRegressor:
    def __init__(self, nTrees=100, maxDepth=3, lr=0.1, minLeaf=5):
        self.nTrees = nTrees
        self.maxDepth = maxDepth
        self.lr = lr
        self.minLeaf = minLeaf
        self.trees = []
        self.initPred = 0.0

    def fit(self, X, y):
        self.initPred = np.mean(y)
        residuals = y - self.initPred
        self.trees = []

        for _ in range(self.nTrees):
            tree = self._buildTree(X, residuals, depth=0)
            self.trees.append(tree)
            preds = np.array([self._predictTree(tree, x) for x in X])
            residuals = residuals - self.lr * preds

        return self

    def predict(self, X):
        preds = np.full(len(X), self.initPred)
        for tree in self.trees:
            preds += self.lr * np.array([self._predictTree(tree, x) for x in X])
        return preds

    def _buildTree(self, X, y, depth):
        if depth >= self.maxDepth or len(y) < 2 * self.minLeaf:
            return {"leaf": True, "value": np.mean(y)}

        bestGain = 0
        bestFeat = None
        bestThresh = None
        totalVar = np.var(y) * len(y)

        nFeats = X.shape[1]
        featIdx = np.random.choice(nFeats, min(nFeats, max(10, nFeats // 3)), replace=False)

        for f in featIdx:
            vals = X[:, f]
            sortIdx = np.argsort(vals)
            sortY = y[sortIdx]
            sortV = vals[sortIdx]

            uniq = np.unique(sortV)
            if len(uniq) < 2:
                continue

            step = max(1, len(uniq) // 20)
            thresholds = uniq[::step]

            for t in thresholds:
                leftMask = vals <= t
                rightMask = ~leftMask
                nL = leftMask.sum()
                nR = rightMask.sum()
                if nL < self.minLeaf or nR < self.minLeaf:
                    continue

                varL = np.var(y[leftMask]) * nL
                varR = np.var(y[rightMask]) * nR
                gain = totalVar - varL - varR

                if gain > bestGain:
                    bestGain = gain
                    bestFeat = f
                    bestThresh = t

        if bestFeat is None:
            return {"leaf": True, "value": np.mean(y)}

        leftMask = X[:, bestFeat] <= bestThresh
        rightMask = ~leftMask

        return {
            "leaf": False,
            "feat": bestFeat,
            "thresh": bestThresh,
            "left": self._buildTree(X[leftMask], y[leftMask], depth + 1),
            "right": self._buildTree(X[rightMask], y[rightMask], depth + 1),
        }

    def _predictTree(self, node, x):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predictTree(node["left"], x)
        return self._predictTree(node["right"], x)


class RidgeRegressor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        A = X.T @ X + self.alpha * np.eye(d)
        self.w = np.linalg.solve(A, X.T @ y)
        self.b = np.mean(y - X @ self.w)
        return self

    def predict(self, X):
        return X @ self.w + self.b


def evaluateBlending(XTrain, yMaseTrain, XTest, yMaseTest, method="ridge"):
    nModels = yMaseTrain.shape[1]
    rawWeights = np.zeros((len(XTest), nModels))

    if method == "ridge":
        for m in range(nModels):
            reg = RidgeRegressor(alpha=10.0)
            targets = np.zeros(len(XTrain))
            for i in range(len(XTrain)):
                bestIdx = np.argmin(yMaseTrain[i])
                targets[i] = 1.0 if bestIdx == m else 0.0
            reg.fit(XTrain, targets)
            rawWeights[:, m] = reg.predict(XTest)
    elif method == "gbt":
        for m in range(nModels):
            reg = GradientBoostedRegressor(nTrees=100, maxDepth=3, lr=0.05, minLeaf=5)
            targets = np.zeros(len(XTrain))
            for i in range(len(XTrain)):
                bestIdx = np.argmin(yMaseTrain[i])
                targets[i] = 1.0 if bestIdx == m else 0.0
            reg.fit(XTrain, targets)
            rawWeights[:, m] = reg.predict(XTest)
    elif method == "gbt_mase":
        for m in range(nModels):
            reg = GradientBoostedRegressor(nTrees=100, maxDepth=3, lr=0.05, minLeaf=5)
            reg.fit(XTrain, yMaseTrain[:, m])
            rawWeights[:, m] = -reg.predict(XTest)

    blendedMases = np.zeros(len(XTest))
    for i in range(len(XTest)):
        w = softmax(rawWeights[i])
        blendedMases[i] = np.dot(w, yMaseTest[i])

    return blendedMases


def main():
    np.random.seed(42)
    print("=" * 90)
    print("Learned Blending — Phase 2, Experiment 010")
    print("=" * 90)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"\n[로드] {len(data)}개 시리즈 from {DATA_PATH.name}", flush=True)

    XList = []
    yMaseList = []
    domains = []
    freqs = []

    for item in data:
        masesDict = item.get("mases", {})
        mases = {}
        for m in MODELS:
            if m in masesDict and masesDict[m] is not None and not np.isnan(masesDict[m]):
                mases[m] = masesDict[m]
        if len(mases) < len(MODELS):
            continue

        dna = item.get("features", item.get("dna", {}))
        freq = item.get("freq", "M")
        feat = buildAugmentedFeatures(dna, freq)
        if np.any(np.isnan(feat)):
            continue

        XList.append(feat)
        yMaseList.append([mases[m] for m in MODELS])
        domains.append(item.get("domain", "Unknown"))
        freqs.append(freq)

    X = np.array(XList)
    yMase = np.array(yMaseList)
    domains = np.array(domains)
    freqs = np.array(freqs)
    N = len(X)

    print(f"[준비] {N}개 시리즈, {X.shape[1]}개 특성, {len(MODELS)}개 모델", flush=True)

    dotMase = yMase[:, MODELS.index("dot")]
    dotAvg = np.mean(dotMase)

    print(f"\n{'='*90}")
    print("Oracle 블렌딩 vs Oracle 선택")
    print(f"{'='*90}", flush=True)

    oracleSelectMases = np.min(yMase, axis=1)
    oracleSelectAvg = np.mean(oracleSelectMases)

    oracleBlendMases = np.zeros(N)
    for i in range(N):
        _, bestMase = oracleBlendWeights(yMase[i])
        oracleBlendMases[i] = bestMase

    oracleBlendAvg = np.mean(oracleBlendMases)

    print(f"  DOT 단독:          MASE = {dotAvg:.3f}")
    print(f"  Oracle 선택:       MASE = {oracleSelectAvg:.3f} (vs DOT: +{(1-oracleSelectAvg/dotAvg)*100:.1f}%)")
    print(f"  Oracle 블렌딩:     MASE = {oracleBlendAvg:.3f} (vs DOT: +{(1-oracleBlendAvg/dotAvg)*100:.1f}%)")
    print(f"  블렌딩 vs 선택:    {(1-oracleBlendAvg/oracleSelectAvg)*100:+.1f}%", flush=True)

    equalBlendMases = np.mean(yMase, axis=1)
    equalBlendAvg = np.mean(equalBlendMases)
    print(f"\n  균등 블렌딩(1/4):  MASE = {equalBlendAvg:.3f} (vs DOT: +{(1-equalBlendAvg/dotAvg)*100:.1f}%)", flush=True)

    print(f"\n{'='*90}")
    print("학습된 블렌딩 5-Fold CV")
    print(f"{'='*90}", flush=True)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    methods = {
        "Ridge 블렌딩": "ridge",
        "GBT 블렌딩 (분류기반)": "gbt",
        "GBT 블렌딩 (MASE역수)": "gbt_mase",
    }

    results = {name: [] for name in methods}
    dotResults = []
    oracleSelectResults = []
    oracleBlendResults = []
    equalBlendResults = []

    gbtBlendAllPreds = np.zeros(N)
    gbtMaseAllPreds = np.zeros(N)

    for fold, (trainIdx, testIdx) in enumerate(kf.split(X)):
        XTr, XTe = X[trainIdx], X[testIdx]
        yTr, yTe = yMase[trainIdx], yMase[testIdx]

        dotFold = np.mean(yTe[:, MODELS.index("dot")])
        dotResults.append(dotFold)
        oracleSelectResults.append(np.mean(np.min(yTe, axis=1)))
        equalBlendResults.append(np.mean(np.mean(yTe, axis=1)))

        oracleBlendFold = 0
        for i in range(len(yTe)):
            _, bm = oracleBlendWeights(yTe[i])
            oracleBlendFold += bm
        oracleBlendResults.append(oracleBlendFold / len(yTe))

        for name, method in methods.items():
            blendMases = evaluateBlending(XTr, yTr, XTe, yTe, method=method)
            results[name].append(np.mean(blendMases))

            if method == "gbt":
                gbtBlendAllPreds[testIdx] = blendMases
            elif method == "gbt_mase":
                gbtMaseAllPreds[testIdx] = blendMases

        print(f"  Fold {fold+1}: DOT={dotFold:.3f}, "
              f"Ridge={results['Ridge 블렌딩'][-1]:.3f}, "
              f"GBT={results['GBT 블렌딩 (분류기반)'][-1]:.3f}, "
              f"GBT-MASE={results['GBT 블렌딩 (MASE역수)'][-1]:.3f}", flush=True)

    print(f"\n{'='*90}")
    print("종합 비교")
    print(f"{'='*90}")

    dotFinal = np.mean(dotResults)
    oracleSelectFinal = np.mean(oracleSelectResults)
    oracleBlendFinal = np.mean(oracleBlendResults)
    equalBlendFinal = np.mean(equalBlendResults)
    oracleGap = dotFinal - oracleBlendFinal

    print(f"\n  {'전략':<30s} | {'MASE':>8s} | {'vs DOT':>8s} | {'Oracle Gap 캡처':>15s}")
    print(f"  {'-'*75}")
    print(f"  {'DOT (단일 최강)':<30s} | {dotFinal:8.3f} | {'기준':>8s} | {'0%':>15s}")
    print(f"  {'균등 블렌딩 (1/4)':<30s} | {equalBlendFinal:8.3f} | {(1-equalBlendFinal/dotFinal)*100:+7.1f}% | {(dotFinal-equalBlendFinal)/oracleGap*100:14.1f}%")

    for name in methods:
        avg = np.mean(results[name])
        capture = (dotFinal - avg) / oracleGap * 100
        print(f"  {name:<30s} | {avg:8.3f} | {(1-avg/dotFinal)*100:+7.1f}% | {capture:14.1f}%")

    print(f"  {'E009 GBT 선택 (참고)':<30s} | {'1.354':>8s} | {'+5.5%':>8s} | {'31.3%':>15s}")
    print(f"  {'Oracle 선택':<30s} | {oracleSelectFinal:8.3f} | {(1-oracleSelectFinal/dotFinal)*100:+7.1f}% | {(dotFinal-oracleSelectFinal)/oracleGap*100:14.1f}%")
    print(f"  {'Oracle 블렌딩':<30s} | {oracleBlendFinal:8.3f} | {(1-oracleBlendFinal/dotFinal)*100:+7.1f}% | {'100%':>15s}", flush=True)

    print(f"\n{'='*90}")
    print("도메인별 최적 블렌딩(GBT-MASE) vs DOT vs E009 선택")
    print(f"{'='*90}")

    bestMethod = "gbt_mase" if np.mean(results["GBT 블렌딩 (MASE역수)"]) < np.mean(results["GBT 블렌딩 (분류기반)"]) else "gbt"
    bestPreds = gbtMaseAllPreds if bestMethod == "gbt_mase" else gbtBlendAllPreds
    bestName = "GBT-MASE" if bestMethod == "gbt_mase" else "GBT-분류"

    for domain in sorted(set(domains)):
        mask = domains == domain
        dDot = np.mean(dotMase[mask])
        dBlend = np.mean(bestPreds[mask])
        dOracle = np.mean(np.min(yMase[mask], axis=1))
        imp = (1 - dBlend / dDot) * 100
        print(f"  {domain:<15s} | DOT={dDot:.3f} | {bestName}={dBlend:.3f} | Oracle={dOracle:.3f} | vs DOT: {imp:+.1f}%", flush=True)

    print(f"\n{'='*90}")
    print("결론")
    print(f"{'='*90}")

    bestAvg = min(np.mean(results[n]) for n in methods)
    bestCapture = (dotFinal - bestAvg) / oracleGap * 100
    e009Capture = 31.3

    if bestCapture > e009Capture:
        print(f"  → 블렌딩이 선택보다 우수! Oracle gap {bestCapture:.1f}% 캡처 (E009: {e009Capture}%)")
    else:
        print(f"  → 블렌딩({bestCapture:.1f}%)이 선택({e009Capture}%)보다 열등 또는 동등")

    h1 = oracleBlendAvg < oracleSelectAvg * 0.97
    h2 = bestAvg < 1.354 * 0.98
    h3 = bestCapture > 40

    print(f"\n  가설 1 (Oracle 블렌딩 > Oracle 선택 -3%): {'통과' if h1 else '기각'}")
    print(f"    Oracle 블렌딩={oracleBlendAvg:.3f} vs Oracle 선택={oracleSelectAvg:.3f}")
    print(f"  가설 2 (학습 블렌딩 > E009 GBT 선택 -2%): {'통과' if h2 else '기각'}")
    print(f"    최적 블렌딩={bestAvg:.3f} vs E009 GBT 선택=1.354")
    print(f"  가설 3 (Oracle gap 40%+ 캡처): {'통과' if h3 else '기각'}")
    print(f"    캡처율={bestCapture:.1f}%", flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
