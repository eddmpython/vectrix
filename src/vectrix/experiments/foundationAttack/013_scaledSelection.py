"""
실험 ID: foundationAttack/013
실험명: Scaled Selection — 확대 데이터로 GBT 모델 선택 재학습

목적:
- E012의 3,000+시리즈 / 6모델 데이터로 GBT 분류기 재학습
- E009(704개, 45.3% 정확도, 31.3% 캡처)와 직접 비교
- 데이터 5배 증가 + 모델 2개 추가의 효과 측정
- 도메인 전문화 재시도 (E011에서 데이터 부족으로 실패했던 것)

가설:
1. 확대 GBT 분류 정확도 > 50% (E009: 45.3%)
2. Oracle gap 캡처 > 40% (E009: 31.3%)
3. 도메인 전문 GBT가 범용 GBT보다 우수 (E011에서 기각된 것이 데이터 충분으로 역전)
4. 모든 도메인에서 DOT 이상 (E009에서 Econ/Fin, Sales가 악화된 것 해소)

방법:
1. E012의 scaled_oracle_6model.json 로드
2. 증강 특성 구축 (DNA + 빈도 원핫 + 교차 + 메타)
3. 범용 GBT (6모델 분류) — 5-Fold CV
4. 도메인별 전문 GBT — 5-Fold CV
5. 하이브리드: 범용 GBT + Sales DOT 폴백
6. 비교: 범용 vs 전문 vs 하이브리드 vs DOT vs Oracle

데이터 리니지:
- 출처: E012 결과 (scaled_oracle_6model.json)
- 시리즈 수: E012 결과에 따름 (기대 3,000+)
- 특성: 증강 (DNA + freq + interaction + meta)
- 시드: 42

실험일: 2026-03-05
"""

import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

DATA_PATH = ROOT / "data" / "gift_eval" / "scaled_oracle_6model.json"

MODELS = ["dot", "auto_ets", "auto_ces", "four_theta", "auto_mstl", "tbats"]

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


def buildFeatures(dnaDict, freq):
    baseKeys = sorted(k for k in dnaDict if isinstance(dnaDict[k], (int, float)) and not np.isnan(dnaDict[k]))
    baseVals = [float(dnaDict[k]) for k in baseKeys]

    freqOnehot = [1.0 if freq.upper().startswith(f) else 0.0 for f in FREQ_LIST]

    interactions = []
    for k1, k2 in INTERACTION_PAIRS:
        v1 = float(dnaDict.get(k1, 0.0))
        v2 = float(dnaDict.get(k2, 0.0))
        interactions.append(v1 * v2)

    period = float(dnaDict.get("period", 1))
    length = float(dnaDict.get("seriesLength", dnaDict.get("length", 100)))
    meta = [np.log1p(period), np.log1p(length), period / max(length, 1)]

    return np.array(baseVals + freqOnehot + interactions + meta)


class GBTClassifier:
    def __init__(self, nTrees=200, maxDepth=4, lr=0.1, minLeaf=5):
        self.nTrees = nTrees
        self.maxDepth = maxDepth
        self.lr = lr
        self.minLeaf = minLeaf
        self.classRegressors = {}
        self.classes = None
        self._singleClass = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        nC = len(self.classes)
        if nC <= 1:
            self._singleClass = self.classes[0] if nC == 1 else 0
            return self
        self._singleClass = None
        for c in self.classes:
            targets = (y == c).astype(float)
            reg = GBTRegressor(self.nTrees, self.maxDepth, self.lr, self.minLeaf)
            reg.fit(X, targets)
            self.classRegressors[c] = reg
        return self

    def predict(self, X):
        if self._singleClass is not None:
            return np.full(len(X), self._singleClass)
        scores = np.zeros((len(X), len(self.classes)))
        for i, c in enumerate(self.classes):
            scores[:, i] = self.classRegressors[c].predict(X)
        return self.classes[np.argmax(scores, axis=1)]


class GBTRegressor:
    def __init__(self, nTrees=200, maxDepth=4, lr=0.1, minLeaf=5):
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
        bestGain, bestFeat, bestThresh = 0, None, None
        totalVar = np.var(y) * len(y)
        nFeats = X.shape[1]
        featIdx = np.random.choice(nFeats, min(nFeats, max(10, nFeats // 3)), replace=False)
        for f in featIdx:
            vals = X[:, f]
            uniq = np.unique(vals)
            if len(uniq) < 2:
                continue
            step = max(1, len(uniq) // 20)
            for t in uniq[::step]:
                leftMask = vals <= t
                nL, nR = leftMask.sum(), len(y) - leftMask.sum()
                if nL < self.minLeaf or nR < self.minLeaf:
                    continue
                gain = totalVar - np.var(y[leftMask]) * nL - np.var(y[~leftMask]) * nR
                if gain > bestGain:
                    bestGain, bestFeat, bestThresh = gain, f, t
        if bestFeat is None:
            return {"leaf": True, "value": np.mean(y)}
        leftMask = X[:, bestFeat] <= bestThresh
        return {
            "leaf": False, "feat": bestFeat, "thresh": bestThresh,
            "left": self._buildTree(X[leftMask], y[leftMask], depth + 1),
            "right": self._buildTree(X[~leftMask], y[~leftMask], depth + 1),
        }

    def _predictTree(self, node, x):
        if node["leaf"]:
            return node["value"]
        return self._predictTree(node["left"] if x[node["feat"]] <= node["thresh"] else node["right"], x)


def main():
    np.random.seed(42)
    print("=" * 90)
    print("Scaled Selection — Phase 2.5, Experiment 013")
    print("=" * 90)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"\n[로드] {len(data)}개 시리즈 from {DATA_PATH.name}", flush=True)

    XList, yMaseList, yLabelList, domainList, freqList = [], [], [], [], []
    validModels = MODELS

    MASE_CAP = 50.0

    for item in data:
        masesDict = item.get("mases", {})
        mases = {}
        allValid = True
        for m in validModels:
            if m in masesDict and masesDict[m] is not None:
                v = masesDict[m]
                if np.isfinite(v) and v < MASE_CAP:
                    mases[m] = v
                else:
                    allValid = False
            else:
                allValid = False
        if not allValid or "dot" not in mases:
            continue

        dna = item.get("features", item.get("dna", {}))
        freq = item.get("freqCat", item.get("freq", "M"))
        feat = buildFeatures(dna, freq)
        if np.any(np.isnan(feat)):
            continue

        maseArr = [mases[m] for m in validModels]
        bestIdx = np.argmin(maseArr)
        modelIdx = bestIdx

        XList.append(feat)
        yMaseList.append({m: mases[m] for m in validModels})
        yLabelList.append(modelIdx)
        domainList.append(item.get("domain", "Unknown"))
        freqList.append(freq)

    X = np.array(XList)
    yLabel = np.array(yLabelList)
    domains = np.array(domainList)
    N = len(X)

    yMaseArr = np.zeros((N, len(validModels)))
    for i in range(N):
        for j, m in enumerate(validModels):
            yMaseArr[i, j] = yMaseList[i].get(m, 999.0)

    dotIdx = validModels.index("dot")
    dotMase = yMaseArr[:, dotIdx]
    oracleMase = np.min(yMaseArr, axis=1)

    print(f"[준비] {N}개 시리즈, {X.shape[1]}개 특성, {len(validModels)}개 모델", flush=True)
    print(f"  DOT MASE: {np.mean(dotMase):.3f}")
    print(f"  Oracle MASE: {np.mean(oracleMase):.3f}")
    print(f"  Oracle gap: {(1-np.mean(oracleMase)/np.mean(dotMase))*100:+.1f}%", flush=True)

    for i, m in enumerate(validModels):
        cnt = np.sum(yLabel == i)
        print(f"  {m}: {cnt}개 ({cnt/N*100:.1f}%)", flush=True)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'='*90}")
    print("범용 GBT 5-Fold CV")
    print(f"{'='*90}", flush=True)

    globalPreds = np.zeros(N, dtype=int)
    globalAccs = []

    for fold, (trIdx, teIdx) in enumerate(kf.split(X)):
        gbt = GBTClassifier(nTrees=200, maxDepth=4, lr=0.1, minLeaf=5)
        gbt.fit(X[trIdx], yLabel[trIdx])
        preds = gbt.predict(X[teIdx])
        globalPreds[teIdx] = preds
        acc = np.mean(preds == yLabel[teIdx])
        globalAccs.append(acc)

        selectedMase = np.mean([yMaseArr[teIdx[i], preds[i]] for i in range(len(teIdx))])
        dotFold = np.mean(dotMase[teIdx])
        print(f"  Fold {fold+1}: acc={acc:.3f}, MASE={selectedMase:.3f}, DOT={dotFold:.3f}", flush=True)

    globalMases = np.array([yMaseArr[i, globalPreds[i]] for i in range(N)])
    globalAvg = np.mean(globalMases)
    dotAvg = np.mean(dotMase)
    oracleAvg = np.mean(oracleMase)
    oracleGap = dotAvg - oracleAvg

    print(f"\n  범용 GBT 정확도: {np.mean(globalAccs):.3f} ± {np.std(globalAccs):.3f}")
    print(f"  범용 GBT MASE: {globalAvg:.3f}")
    print(f"  vs DOT: {(1-globalAvg/dotAvg)*100:+.1f}%")
    print(f"  Oracle gap 캡처: {(dotAvg-globalAvg)/oracleGap*100:.1f}%", flush=True)

    print(f"\n{'='*90}")
    print("도메인별 전문 GBT 5-Fold CV")
    print(f"{'='*90}", flush=True)

    domainPreds = np.zeros(N, dtype=int)

    for fold, (trIdx, teIdx) in enumerate(kf.split(X)):
        gbtFallback = GBTClassifier(nTrees=200, maxDepth=4, lr=0.1, minLeaf=5)
        gbtFallback.fit(X[trIdx], yLabel[trIdx])
        fallbackPreds = gbtFallback.predict(X[teIdx])

        for dom in np.unique(domains):
            trMask = domains[trIdx] == dom
            teMask = domains[teIdx] == dom
            trDomIdx = trIdx[trMask]
            teDomIdx = np.where(teMask)[0]

            if len(trDomIdx) < 30 or len(teDomIdx) == 0:
                for di in teDomIdx:
                    domainPreds[teIdx[di]] = fallbackPreds[di]
                continue

            domGbt = GBTClassifier(nTrees=150, maxDepth=3, lr=0.1, minLeaf=3)
            domGbt.fit(X[trDomIdx], yLabel[trDomIdx])
            domP = domGbt.predict(X[teIdx[teMask]])
            for k, di in enumerate(teDomIdx):
                domainPreds[teIdx[di]] = domP[k]

        foldMase = np.mean([yMaseArr[teIdx[i], domainPreds[teIdx[i]]] for i in range(len(teIdx))])
        print(f"  Fold {fold+1}: MASE={foldMase:.3f}", flush=True)

    domainMases = np.array([yMaseArr[i, domainPreds[i]] for i in range(N)])
    domainAvg = np.mean(domainMases)

    print(f"\n  도메인 전문 GBT MASE: {domainAvg:.3f}")
    print(f"  vs DOT: {(1-domainAvg/dotAvg)*100:+.1f}%")
    print(f"  Oracle gap 캡처: {(dotAvg-domainAvg)/oracleGap*100:.1f}%", flush=True)

    hybridMases = globalMases.copy()
    for i in range(N):
        if domains[i] == "Sales":
            hybridMases[i] = dotMase[i]
    hybridAvg = np.mean(hybridMases)

    print(f"\n{'='*90}")
    print("하이브리드 (범용 GBT + Sales DOT 폴백)")
    print(f"{'='*90}")
    print(f"  MASE: {hybridAvg:.3f}")
    print(f"  vs DOT: {(1-hybridAvg/dotAvg)*100:+.1f}%")
    print(f"  Oracle gap 캡처: {(dotAvg-hybridAvg)/oracleGap*100:.1f}%", flush=True)

    print(f"\n{'='*90}")
    print("종합 비교")
    print(f"{'='*90}")

    strategies = {
        "DOT (단일 최강)": dotAvg,
        "범용 GBT": globalAvg,
        "도메인 전문 GBT": domainAvg,
        "하이브리드 (GBT+폴백)": hybridAvg,
        "E009 GBT (참고, 4모델)": 1.354,
        "Oracle": oracleAvg,
    }

    print(f"\n  {'전략':<30s} | {'MASE':>8s} | {'vs DOT':>8s} | {'Oracle Gap':>12s}")
    print(f"  {'-'*70}")
    for name, mase in strategies.items():
        if name == "DOT (단일 최강)":
            print(f"  {name:<30s} | {mase:8.3f} | {'기준':>8s} | {'0%':>12s}")
        elif name == "Oracle":
            print(f"  {name:<30s} | {mase:8.3f} | {(1-mase/dotAvg)*100:+7.1f}% | {'100%':>12s}")
        else:
            capture = (dotAvg - mase) / oracleGap * 100
            print(f"  {name:<30s} | {mase:8.3f} | {(1-mase/dotAvg)*100:+7.1f}% | {capture:11.1f}%")

    print(f"\n{'='*90}")
    print("도메인별 비교")
    print(f"{'='*90}")

    bestAll = np.minimum(globalMases, domainMases)
    bestAll = np.minimum(bestAll, hybridMases)

    for dom in sorted(set(domains)):
        mask = domains == dom
        dDot = np.mean(dotMase[mask])
        dGlobal = np.mean(globalMases[mask])
        dDomain = np.mean(domainMases[mask])
        dHybrid = np.mean(hybridMases[mask])
        dOracle = np.mean(oracleMase[mask])

        best = min(dGlobal, dDomain, dHybrid)
        imp = (1 - best / dDot) * 100

        print(f"  {dom:<15s} | DOT={dDot:.3f} | 범용={dGlobal:.3f} | 전문={dDomain:.3f} | 하이브리드={dHybrid:.3f} | Oracle={dOracle:.3f} | best {imp:+.1f}%", flush=True)

    print(f"\n{'='*90}")
    print("결론")
    print(f"{'='*90}")

    bestOverall = min(globalAvg, domainAvg, hybridAvg)
    bestCapture = (dotAvg - bestOverall) / oracleGap * 100

    h1 = np.mean(globalAccs) > 0.50
    h2 = bestCapture > 40
    h3 = domainAvg < globalAvg
    h4 = True
    for dom in sorted(set(domains)):
        mask = domains == dom
        best = min(np.mean(globalMases[mask]), np.mean(domainMases[mask]), np.mean(hybridMases[mask]))
        if best > np.mean(dotMase[mask]) * 1.01:
            h4 = False
            break

    print(f"\n  가설 1 (분류 정확도 > 50%): {'통과' if h1 else '기각'} ({np.mean(globalAccs)*100:.1f}%)")
    print(f"  가설 2 (Oracle gap > 40%): {'통과' if h2 else '기각'} ({bestCapture:.1f}%)")
    print(f"  가설 3 (도메인 전문 > 범용): {'통과' if h3 else '기각'}")
    print(f"  가설 4 (모든 도메인 DOT 이상): {'통과' if h4 else '기각'}", flush=True)

    domBestMases = np.zeros(N)
    for dom in sorted(set(domains)):
        mask = domains == dom
        g = np.mean(globalMases[mask])
        d = np.mean(domainMases[mask])
        h = np.mean(hybridMases[mask])
        if g <= d and g <= h:
            domBestMases[mask] = globalMases[mask]
        elif d <= h:
            domBestMases[mask] = domainMases[mask]
        else:
            domBestMases[mask] = hybridMases[mask]
    compositeAvg = np.mean(domBestMases)
    compositeCapture = (dotAvg - compositeAvg) / oracleGap * 100
    print(f"\n  [보너스] 도메인별 최우수 조합: MASE={compositeAvg:.3f}, Oracle gap: {compositeCapture:.1f}%", flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
