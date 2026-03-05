"""
실험 ID: foundationAttack/011
실험명: Domain-Aware Routing — 도메인별 전문 모델 선택기

목적:
- E009의 범용 GBT(+5.5%)가 Econ/Fin(-1.6%), Sales(-4.5%)에서 악화
- 도메인별 전문 분류기를 학습하면 이 문제가 해결되는지 검증
- 2단계 라우팅: (1) DNA → 도메인 분류 (2) 도메인별 전문 모델 선택

가설:
1. 도메인별 전문 GBT가 E009 범용 GBT(+5.5%) 대비 +2%p 이상 개선
2. Econ/Fin, Sales에서 DOT 대비 악화가 해소 (→ 0% 이상)
3. Oracle gap 캡처율 40%+ (E009의 31.3%에서 +9%p)

방법:
1. 전략 A "도메인별 전문가": 도메인 7개 × 전문 GBT 분류기 7개
2. 전략 B "빈도별 전문가": 빈도 그룹(저빈도 Y/Q/M, 중빈도 W/D, 고빈도 H/T/S) × 전문 GBT 3개
3. 전략 C "2단계 라우팅": Ridge 도메인 분류(E005) → 도메인별 전문 GBT
4. 전략 D "도메인 DOT-폴백": 도메인별로 GBT가 DOT보다 나쁜 도메인은 DOT으로 폴백
5. 5-Fold CV로 전략 비교

데이터 리니지:
- 출처: GIFT-Eval (E008 결과)
- 파일: data/gift_eval/multi_model_oracle.json
- 시리즈 수: 704개
- 특성: 증강 (47 DNA + 8 freq + 8 interaction + 3 meta)
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

FREQ_GROUPS = {
    "low": ["Y", "Q", "M"],
    "mid": ["W", "D"],
    "high": ["H", "T", "S"],
}


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
    length = float(dnaDict.get("seriesLength", dnaDict.get("length", 100)))
    meta = [
        np.log1p(period),
        np.log1p(length),
        period / max(length, 1),
    ]

    return np.array(baseVals + freqOnehot + interactions + meta)


class GBTClassifier:
    def __init__(self, nTrees=150, maxDepth=4, lr=0.1, minLeaf=5):
        self.nTrees = nTrees
        self.maxDepth = maxDepth
        self.lr = lr
        self.minLeaf = minLeaf
        self.classRegressors = {}
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        nC = len(self.classes)

        if nC <= 1:
            self.classRegressors = {}
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
        featIdx = np.random.choice(nFeats, min(nFeats, max(8, nFeats // 3)), replace=False)

        for f in featIdx:
            vals = X[:, f]
            uniq = np.unique(vals)
            if len(uniq) < 2:
                continue
            step = max(1, len(uniq) // 15)
            thresholds = uniq[::step]

            for t in thresholds:
                leftMask = vals <= t
                nL = leftMask.sum()
                nR = len(y) - nL
                if nL < self.minLeaf or nR < self.minLeaf:
                    continue

                varL = np.var(y[leftMask]) * nL
                varR = np.var(y[~leftMask]) * nR
                gain = totalVar - varL - varR

                if gain > bestGain:
                    bestGain = gain
                    bestFeat = f
                    bestThresh = t

        if bestFeat is None:
            return {"leaf": True, "value": np.mean(y)}

        leftMask = X[:, bestFeat] <= bestThresh
        return {
            "leaf": False,
            "feat": bestFeat,
            "thresh": bestThresh,
            "left": self._buildTree(X[leftMask], y[leftMask], depth + 1),
            "right": self._buildTree(X[~leftMask], y[~leftMask], depth + 1),
        }

    def _predictTree(self, node, x):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thresh"]:
            return self._predictTree(node["left"], x)
        return self._predictTree(node["right"], x)


class RidgeClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.W = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        Y = np.zeros((len(y), len(self.classes)))
        for i, c in enumerate(self.classes):
            Y[:, i] = (y == c).astype(float)
        n, d = X.shape
        A = X.T @ X + self.alpha * np.eye(d)
        self.W = np.linalg.solve(A, X.T @ Y)
        return self

    def predict(self, X):
        scores = X @ self.W
        return self.classes[np.argmax(scores, axis=1)]


def getFreqGroup(freq):
    fu = freq.upper()
    for group, members in FREQ_GROUPS.items():
        for m in members:
            if fu.startswith(m):
                return group
    return "mid"


def main():
    np.random.seed(42)
    print("=" * 90)
    print("Domain-Aware Routing — Phase 2, Experiment 011")
    print("=" * 90)

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"\n[로드] {len(data)}개 시리즈 from {DATA_PATH.name}", flush=True)

    XList, yMaseList, yLabelList, domainList, freqList, freqGroupList = [], [], [], [], [], []

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

        maseArr = [mases[m] for m in MODELS]
        bestIdx = np.argmin(maseArr)

        XList.append(feat)
        yMaseList.append(maseArr)
        yLabelList.append(bestIdx)
        domainList.append(item.get("domain", "Unknown"))
        freqList.append(freq)
        freqGroupList.append(getFreqGroup(freq))

    X = np.array(XList)
    yMase = np.array(yMaseList)
    yLabel = np.array(yLabelList)
    domains = np.array(domainList)
    freqs = np.array(freqList)
    freqGroups = np.array(freqGroupList)
    N = len(X)

    print(f"[준비] {N}개 시리즈, {X.shape[1]}개 특성", flush=True)

    dotIdx = MODELS.index("dot")
    dotMase = yMase[:, dotIdx]
    oracleMase = np.min(yMase, axis=1)

    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    strategies = {
        "A_범용GBT": None,
        "B_도메인전문가": None,
        "C_빈도전문가": None,
        "D_2단계라우팅": None,
        "E_도메인폴백": None,
    }
    allPreds = {k: np.zeros(N, dtype=int) for k in strategies}
    allMases = {k: np.zeros(N) for k in strategies}

    print(f"\n{'='*90}")
    print("5-Fold CV 학습")
    print(f"{'='*90}", flush=True)

    for fold, (trainIdx, testIdx) in enumerate(kf.split(X)):
        XTr, XTe = X[trainIdx], X[testIdx]
        yTr, yTe = yLabel[trainIdx], yLabel[testIdx]
        maseTr, maseTe = yMase[trainIdx], yMase[testIdx]
        domTr, domTe = domains[trainIdx], domains[testIdx]
        fgTr, fgTe = freqGroups[trainIdx], freqGroups[testIdx]

        gbtGlobal = GBTClassifier(nTrees=150, maxDepth=4, lr=0.1, minLeaf=5)
        gbtGlobal.fit(XTr, yTr)
        predsA = gbtGlobal.predict(XTe)
        allPreds["A_범용GBT"][testIdx] = predsA
        allMases["A_범용GBT"][testIdx] = [maseTe[i, predsA[i]] for i in range(len(testIdx))]

        predsB = np.zeros(len(testIdx), dtype=int)
        for dom in np.unique(domTr):
            trMask = domTr == dom
            teMask = domTe == dom
            if trMask.sum() < 10 or teMask.sum() == 0:
                predsB[teMask] = predsA[teMask]
                continue
            localGbt = GBTClassifier(nTrees=100, maxDepth=3, lr=0.1, minLeaf=3)
            localGbt.fit(XTr[trMask], yTr[trMask])
            predsB[teMask] = localGbt.predict(XTe[teMask])
        for dom in np.unique(domTe):
            if dom not in domTr:
                teMask = domTe == dom
                predsB[teMask] = predsA[teMask]
        allPreds["B_도메인전문가"][testIdx] = predsB
        allMases["B_도메인전문가"][testIdx] = [maseTe[i, predsB[i]] for i in range(len(testIdx))]

        predsC = np.zeros(len(testIdx), dtype=int)
        for fg in np.unique(fgTr):
            trMask = fgTr == fg
            teMask = fgTe == fg
            if trMask.sum() < 10 or teMask.sum() == 0:
                predsC[teMask] = predsA[teMask]
                continue
            localGbt = GBTClassifier(nTrees=100, maxDepth=3, lr=0.1, minLeaf=3)
            localGbt.fit(XTr[trMask], yTr[trMask])
            predsC[teMask] = localGbt.predict(XTe[teMask])
        for fg in np.unique(fgTe):
            if fg not in fgTr:
                teMask = fgTe == fg
                predsC[teMask] = predsA[teMask]
        allPreds["C_빈도전문가"][testIdx] = predsC
        allMases["C_빈도전문가"][testIdx] = [maseTe[i, predsC[i]] for i in range(len(testIdx))]

        ridgeDom = RidgeClassifier(alpha=1.0)
        uniqueDoms = np.unique(domTr)
        domLabelTr = np.array([np.where(uniqueDoms == d)[0][0] for d in domTr])
        ridgeDom.fit(XTr, domLabelTr)
        domLabelTe = ridgeDom.predict(XTe)
        predDomTe = np.array([uniqueDoms[l] if l < len(uniqueDoms) else "Unknown" for l in domLabelTe])

        predsD = np.zeros(len(testIdx), dtype=int)
        domGbts = {}
        for dom in uniqueDoms:
            trMask = domTr == dom
            if trMask.sum() < 10:
                continue
            localGbt = GBTClassifier(nTrees=100, maxDepth=3, lr=0.1, minLeaf=3)
            localGbt.fit(XTr[trMask], yTr[trMask])
            domGbts[dom] = localGbt

        for i in range(len(testIdx)):
            pDom = predDomTe[i]
            if pDom in domGbts:
                predsD[i] = domGbts[pDom].predict(XTe[i:i+1])[0]
            else:
                predsD[i] = predsA[i]
        allPreds["D_2단계라우팅"][testIdx] = predsD
        allMases["D_2단계라우팅"][testIdx] = [maseTe[i, predsD[i]] for i in range(len(testIdx))]

        predsE = predsA.copy()
        for dom in np.unique(domTe):
            teMask = domTe == dom
            trDomMask = domTr == dom
            if trDomMask.sum() < 10:
                continue
            gbtDomMase = np.mean([maseTr[j, predsA[np.where(testIdx == trainIdx[j])[0][0]]]
                                  if trainIdx[j] in testIdx else 0
                                  for j in np.where(trDomMask)[0][:20]])
            dotDomMase = np.mean(maseTr[trDomMask, dotIdx])
            if gbtDomMase > dotDomMase:
                predsE[teMask] = dotIdx

        allPreds["E_도메인폴백"][testIdx] = predsE
        allMases["E_도메인폴백"][testIdx] = [maseTe[i, predsE[i]] for i in range(len(testIdx))]

        dotFold = np.mean(maseTe[:, dotIdx])
        print(f"  Fold {fold+1}: DOT={dotFold:.3f}, "
              f"범용={np.mean(allMases['A_범용GBT'][testIdx]):.3f}, "
              f"도메인={np.mean(allMases['B_도메인전문가'][testIdx]):.3f}, "
              f"빈도={np.mean(allMases['C_빈도전문가'][testIdx]):.3f}, "
              f"2단계={np.mean(allMases['D_2단계라우팅'][testIdx]):.3f}", flush=True)

    print(f"\n{'='*90}")
    print("종합 비교")
    print(f"{'='*90}")

    dotAvg = np.mean(dotMase)
    oracleAvg = np.mean(oracleMase)
    oracleGap = dotAvg - oracleAvg

    print(f"\n  {'전략':<25s} | {'MASE':>8s} | {'vs DOT':>8s} | {'Oracle Gap':>12s}")
    print(f"  {'-'*65}")
    print(f"  {'DOT (단일 최강)':<25s} | {dotAvg:8.3f} | {'기준':>8s} | {'0%':>12s}")

    for name in strategies:
        avg = np.mean(allMases[name])
        imp = (1 - avg / dotAvg) * 100
        capture = (dotAvg - avg) / oracleGap * 100
        print(f"  {name:<25s} | {avg:8.3f} | {imp:+7.1f}% | {capture:11.1f}%")

    print(f"  {'Oracle':<25s} | {oracleAvg:8.3f} | {(1-oracleAvg/dotAvg)*100:+7.1f}% | {'100%':>12s}", flush=True)

    print(f"\n{'='*90}")
    print("도메인별 최우수 전략 비교")
    print(f"{'='*90}")

    bestStrategy = max(strategies, key=lambda s: (dotAvg - np.mean(allMases[s])))
    print(f"\n  전체 최우수: {bestStrategy} (MASE={np.mean(allMases[bestStrategy]):.3f})")

    print(f"\n  {'도메인':<15s} | {'DOT':>7s} | {'범용GBT':>7s} | {'도메인전문':>8s} | {'빈도전문':>7s} | {'2단계':>7s} | {'최우수':>10s}")
    print(f"  {'-'*85}")

    for dom in sorted(set(domains)):
        mask = domains == dom
        dDot = np.mean(dotMase[mask])
        vals = {}
        for name in strategies:
            vals[name] = np.mean(allMases[name][mask])

        bestName = min(vals, key=vals.get)
        bestVal = vals[bestName]
        imp = (1 - bestVal / dDot) * 100

        print(f"  {dom:<15s} | {dDot:7.3f} | {vals['A_범용GBT']:7.3f} | {vals['B_도메인전문가']:8.3f} | {vals['C_빈도전문가']:7.3f} | {vals['D_2단계라우팅']:7.3f} | {bestName.split('_')[1][:4]} {imp:+.1f}%", flush=True)

    print(f"\n{'='*90}")
    print("빈도별 최우수 전략 비교")
    print(f"{'='*90}")

    for fg in ["low", "mid", "high"]:
        mask = freqGroups == fg
        if mask.sum() == 0:
            continue
        dDot = np.mean(dotMase[mask])
        print(f"\n  {fg} ({FREQ_GROUPS[fg]}): {mask.sum()}개")
        for name in strategies:
            avg = np.mean(allMases[name][mask])
            imp = (1 - avg / dDot) * 100
            print(f"    {name:<25s}: {avg:.3f} ({imp:+.1f}%)")

    print(f"\n{'='*90}")
    print("결론")
    print(f"{'='*90}")

    bestOverall = min(strategies, key=lambda s: np.mean(allMases[s]))
    bestMase = np.mean(allMases[bestOverall])
    bestCapture = (dotAvg - bestMase) / oracleGap * 100

    h1 = bestMase < 1.354 * 0.98
    h2 = True
    for dom in ["Econ/Fin", "Sales"]:
        mask = domains == dom
        if mask.sum() > 0:
            bestDomMase = min(np.mean(allMases[s][mask]) for s in strategies)
            if bestDomMase > np.mean(dotMase[mask]):
                h2 = False
    h3 = bestCapture > 40

    print(f"\n  최우수 전략: {bestOverall}")
    print(f"  MASE: {bestMase:.3f}, vs DOT: {(1-bestMase/dotAvg)*100:+.1f}%, Oracle gap: {bestCapture:.1f}%")
    print(f"\n  가설 1 (범용 GBT +5.5% 대비 +2%p): {'통과' if h1 else '기각'}")
    print(f"  가설 2 (Econ/Fin, Sales DOT 이상): {'통과' if h2 else '기각'}")
    print(f"  가설 3 (Oracle gap 40%+): {'통과' if h3 else '기각'}", flush=True)

    bestDomainMases = np.zeros(N)
    for dom in sorted(set(domains)):
        mask = domains == dom
        bestS = min(strategies, key=lambda s: np.mean(allMases[s][mask]))
        bestDomainMases[mask] = allMases[bestS][mask]

    compositeAvg = np.mean(bestDomainMases)
    compositeCapture = (dotAvg - compositeAvg) / oracleGap * 100
    print(f"\n  [보너스] 도메인별 최우수 조합: MASE={compositeAvg:.3f}, vs DOT: {(1-compositeAvg/dotAvg)*100:+.1f}%, Oracle gap: {compositeCapture:.1f}%", flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
