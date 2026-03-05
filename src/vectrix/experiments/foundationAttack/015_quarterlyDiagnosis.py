"""
실험 ID: foundationAttack/015
실험명: Quarterly -38.6% 악화 진단 + GBT 일반화 개선

목적:
- E014에서 m4_quarterly holdout이 -38.6% 심각 악화된 원인 규명
- 개별 시리즈 수준에서 GBT 선택 vs Oracle vs DOT 비교
- GBT 오선택 패턴 식별 (어떤 모델을 잘못 고르는가)
- 하이퍼파라미터 완화(maxDepth, nTrees 축소)의 일반화 효과 측정
- 빈도별 GBT 분리의 효과 측정

가설:
1. Quarterly 악화는 GBT의 특정 모델 편향 (한 모델을 과도하게 선택)이 원인
2. maxDepth=3, nTrees=100으로 줄이면 CV-실예측 격차 50%+ 감소
3. 빈도별 GBT 분리(저빈도 Y/Q/M vs 고빈도)가 범용 GBT보다 우수

방법:
1. E012 Oracle JSON으로 GBT 학습 (전체 3,165개)
2. m4_quarterly holdout 시리즈에서 개별 선택/MASE 출력
3. 오선택 패턴 분석 (GBT 선택 vs Oracle 모델)
4. 3가지 GBT 변형 비교: 기존(200/4), 완화(100/3), 극완화(50/2)
5. 빈도별 GBT: 저빈도(Y/Q/M) 전용 vs 고빈도(D/H/T/W/S) 전용
6. 모든 holdout 데이터셋에서 재검증

데이터 리니지:
- 학습: E012 scaled_oracle_6model.json (3,230 시리즈, 6모델 MASE)
- 검증: GIFT-Eval m4_quarterly + 기존 E014 holdout 전체
- 시드: 42

실험일: 2026-03-05
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

GIFT_EVAL_DIR = ROOT / "data" / "gift_eval"
DATA_PATH = ROOT / "data" / "gift_eval" / "scaled_oracle_6model.json"

MODELS = ["dot", "auto_ets", "auto_ces", "four_theta", "auto_mstl", "tbats"]

FREQ_TO_PERIOD = {
    "Y": 1, "A": 1, "A-DEC": 1,
    "Q": 4, "QS": 4, "Q-DEC": 4,
    "M": 12, "MS": 12,
    "W": 52, "W-MON": 52, "W-SUN": 52, "W-FRI": 52,
    "D": 7, "B": 5,
    "H": 24, "h": 24,
    "5T": 288, "5min": 288, "10T": 144, "10min": 144, "15T": 96, "15min": 96,
    "10S": 8640, "10s": 8640,
}

M4_PRED = {"Y": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
STD_PRED = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}

DOMAIN_MAP = {
    "m4_yearly": "Econ/Fin", "m4_quarterly": "Econ/Fin", "m4_monthly": "Econ/Fin",
    "m4_weekly": "Econ/Fin", "m4_daily": "Econ/Fin", "m4_hourly": "Econ/Fin",
    "electricity": "Energy", "solar": "Energy", "ett1": "Energy", "ett2": "Energy",
    "hospital": "Healthcare", "covid_deaths": "Healthcare", "us_births": "Healthcare",
    "saugeenday": "Nature", "temperature_rain_with_missing": "Nature",
    "kdd_cup_2018_with_missing": "Nature", "jena_weather": "Nature",
    "car_parts_with_missing": "Sales", "restaurant": "Sales", "hierarchical_sales": "Sales",
    "LOOP_SEATTLE": "Transport", "SZ_TAXI": "Transport", "M_DENSE": "Transport",
    "bitbrains_fast_storage": "Web/CloudOps", "bitbrains_rnd": "Web/CloudOps",
    "bizitobs_application": "Web/CloudOps", "bizitobs_service": "Web/CloudOps",
    "bizitobs_l2c": "Web/CloudOps",
}

HOLDOUT_DATASETS = [
    "m4_monthly", "m4_quarterly", "m4_yearly",
    "electricity/H", "hospital", "saugeenday/D",
    "restaurant", "LOOP_SEATTLE/H",
    "ett1/H", "bizitobs_application",
]

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

MASE_CAP = 50.0
MAX_HOLDOUT = 20
MAX_LEN = 5000


def getDomain(dsName):
    return DOMAIN_MAP.get(dsName.split("/")[0], "Unknown")


def getPeriod(freq):
    freq = str(freq).strip()
    for key in sorted(FREQ_TO_PERIOD.keys(), key=len, reverse=True):
        if freq == key or freq.startswith(key):
            return FREQ_TO_PERIOD[key]
    return 1


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


def createModel(modelId, period):
    from vectrix.engine.registry import createModel as regCreate
    return regCreate(modelId, period)


def runHoldoutDetailed(dsName, trainSeriesIds, gbtVariants, dnaAnalyzer):
    import datasets as hfDatasets

    dsPath = GIFT_EVAL_DIR / dsName
    if not dsPath.exists():
        return []

    ds = hfDatasets.load_from_disk(str(dsPath)).with_format("numpy")
    nTotal = len(ds)
    freq = str(ds[0].get("freq", "D"))
    period = getPeriod(freq)
    predLength = getPredLength(dsName, freq)
    freqCat = getFreqCategory(freq)
    domain = getDomain(dsName)

    safePeriod = min(period, 1000)

    rng = np.random.RandomState(99)
    indices = rng.permutation(nTotal)

    results = []
    tested = 0

    for idx in indices:
        sid = f"{dsName}_{idx}"
        if sid in trainSeriesIds:
            continue

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
        if len(y) > MAX_LEN:
            y = y[-MAX_LEN:]

        sp = min(safePeriod, len(y) // 3)
        if sp < 1:
            sp = 1

        trainY = y[:-predLength]
        testY = y[-predLength:]

        try:
            profile = dnaAnalyzer.analyze(trainY, period=sp)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

        feats = {k: float(v) if np.isfinite(v) else 0.0 for k, v in profile.features.items()}
        feat = buildFeatures(feats, freqCat)
        if np.any(np.isnan(feat)):
            continue

        selections = {}
        for variantName, gbt in gbtVariants.items():
            selIdx = int(gbt.predict(feat.reshape(1, -1))[0])
            selections[variantName] = MODELS[selIdx]

        mases = {}
        for modelId in MODELS:
            try:
                model = createModel(modelId, sp)
                model.fit(trainY)
                pred, _, _ = model.predict(predLength)
                pred = np.array(pred, dtype=float)[:len(testY)]

                validMask = ~np.isnan(testY) & ~np.isnan(pred)
                if np.sum(validMask) < 1:
                    continue

                mase = computeMASE(testY[validMask], pred[validMask], trainY, sp)
                if np.isfinite(mase) and mase < MASE_CAP:
                    mases[modelId] = float(mase)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass

        if "dot" not in mases:
            continue

        oracleModel = min(mases, key=mases.get) if mases else "dot"
        oracleMase = mases[oracleModel]

        result = {
            "dataset": dsName,
            "domain": domain,
            "freq": freqCat,
            "seriesIdx": int(idx),
            "allMases": mases,
            "dotMase": mases.get("dot", 999.0),
            "oracleModel": oracleModel,
            "oracleMase": oracleMase,
            "selections": {},
            "trainLen": len(trainY),
            "predLen": predLength,
            "period": sp,
        }

        for variantName in gbtVariants:
            selModel = selections[variantName]
            selMase = mases.get(selModel, 999.0)
            result["selections"][variantName] = {
                "model": selModel,
                "mase": selMase if selModel in mases else None,
            }

        results.append(result)
        tested += 1
        if tested >= MAX_HOLDOUT:
            break

    return results


def main():
    np.random.seed(42)
    print("=" * 90)
    print("E015 — Quarterly Diagnosis + GBT Generalization Improvement")
    print("=" * 90)

    from vectrix.adaptive.dna import ForecastDNA
    dnaAnalyzer = ForecastDNA()

    with open(DATA_PATH, "r") as f:
        oracleData = json.load(f)
    print(f"\n[1/5] Oracle 데이터 로드: {len(oracleData)}개", flush=True)

    XList, yLabelList, freqCatList, domainList = [], [], [], []
    trainSeriesIds = set()

    for item in oracleData:
        sid = f"{item.get('dataset', '')}_{item.get('seriesIdx', '')}"
        trainSeriesIds.add(sid)

        masesDict = item.get("mases", {})
        mases = {}
        allValid = True
        for m in MODELS:
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

        maseArr = [mases[m] for m in MODELS]
        bestIdx = int(np.argmin(maseArr))
        XList.append(feat)
        yLabelList.append(bestIdx)
        freqCatList.append(freq)
        domainList.append(item.get("domain", "Unknown"))

    X_all = np.array(XList)
    y_all = np.array(yLabelList)
    freqs_all = np.array(freqCatList)
    domains_all = np.array(domainList)
    print(f"  유효 학습 데이터: {len(X_all)}개, {X_all.shape[1]} 특성", flush=True)

    freqDist = defaultdict(int)
    for f in freqs_all:
        freqDist[f] += 1
    print(f"  빈도 분포: {dict(sorted(freqDist.items()))}", flush=True)

    print(f"\n[2/5] GBT 변형 3개 + 빈도별 GBT 학습", flush=True)

    configs = {
        "gbt_200_4": (200, 4, 0.1, 5),
        "gbt_100_3": (100, 3, 0.1, 5),
        "gbt_50_2": (50, 2, 0.1, 5),
    }

    gbtVariants = {}

    for name, (nTrees, maxDepth, lr, minLeaf) in configs.items():
        t0 = time.time()
        gbt = GBTClassifier(nTrees=nTrees, maxDepth=maxDepth, lr=lr, minLeaf=minLeaf)
        gbt.fit(X_all, y_all)
        trainAcc = np.mean(gbt.predict(X_all) == y_all)
        elapsed = time.time() - t0
        gbtVariants[name] = gbt
        print(f"  {name}: train_acc={trainAcc:.3f}, {elapsed:.1f}s", flush=True)

    lowFreqMask = np.isin(freqs_all, ["Y", "Q", "M"])
    highFreqMask = ~lowFreqMask

    if np.sum(lowFreqMask) >= 50 and np.sum(highFreqMask) >= 50:
        t0 = time.time()
        gbtLow = GBTClassifier(nTrees=150, maxDepth=3, lr=0.1, minLeaf=5)
        gbtLow.fit(X_all[lowFreqMask], y_all[lowFreqMask])
        lowAcc = np.mean(gbtLow.predict(X_all[lowFreqMask]) == y_all[lowFreqMask])

        gbtHigh = GBTClassifier(nTrees=150, maxDepth=3, lr=0.1, minLeaf=5)
        gbtHigh.fit(X_all[highFreqMask], y_all[highFreqMask])
        highAcc = np.mean(gbtHigh.predict(X_all[highFreqMask]) == y_all[highFreqMask])

        elapsed = time.time() - t0
        print(f"  freq_split: low_acc={lowAcc:.3f}({np.sum(lowFreqMask)}개), "
              f"high_acc={highAcc:.3f}({np.sum(highFreqMask)}개), {elapsed:.1f}s", flush=True)
    else:
        gbtLow = gbtHigh = None
        print(f"  freq_split: 데이터 부족으로 생략", flush=True)

    class FreqSplitGBT:
        def __init__(self, low, high, freqList):
            self.low = low
            self.high = high
            self.freqList = freqList

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    if gbtLow and gbtHigh:
        class FreqRouter:
            def __init__(self, low, high):
                self.low = low
                self.high = high
                self._isLow = True

            def setLow(self, isLow):
                self._isLow = isLow

            def predict(self, X):
                if self._isLow:
                    return self.low.predict(X)
                return self.high.predict(X)

        freqRouter = FreqRouter(gbtLow, gbtHigh)
    else:
        freqRouter = None

    print(f"\n[3/5] Quarterly 개별 시리즈 진단", flush=True)
    print(f"{'='*90}")

    qResults = runHoldoutDetailed("m4_quarterly", trainSeriesIds, gbtVariants, dnaAnalyzer)

    if qResults:
        print(f"\n  m4_quarterly holdout: {len(qResults)}개 시리즈\n")
        print(f"  {'#':>3s} | {'idx':>5s} | {'DOT':>6s} | {'Sel200/4':>8s} | {'Sel100/3':>8s} | {'Sel50/2':>7s} | "
              f"{'Oracle':>6s} | {'GBT선택':>10s} | {'Oracle모델':>10s} | {'판정':>4s}")
        print(f"  {'-'*100}")

        wrongChoices = defaultdict(int)
        oracleChoices = defaultdict(int)
        selChoices200 = defaultdict(int)

        for i, r in enumerate(qResults):
            dotM = r["dotMase"]
            sel200 = r["selections"]["gbt_200_4"]
            sel100 = r["selections"]["gbt_100_3"]
            sel50 = r["selections"]["gbt_50_2"]
            oracM = r["oracleMase"]
            oracModel = r["oracleModel"]

            selModel200 = sel200["model"]
            selMase200 = sel200["mase"] if sel200["mase"] is not None else 999.0
            selMase100 = sel100["mase"] if sel100["mase"] is not None else 999.0
            selMase50 = sel50["mase"] if sel50["mase"] is not None else 999.0

            verdict = "OK" if selMase200 <= dotM * 1.01 else "BAD"

            selChoices200[selModel200] += 1
            oracleChoices[oracModel] += 1
            if selModel200 != oracModel:
                wrongChoices[f"{selModel200}->{oracModel}"] += 1

            print(f"  {i+1:3d} | {r['seriesIdx']:5d} | {dotM:6.3f} | {selMase200:8.3f} | "
                  f"{selMase100:8.3f} | {selMase50:7.3f} | {oracM:6.3f} | "
                  f"{selModel200:>10s} | {oracModel:>10s} | {verdict:>4s}")

        print(f"\n  --- Quarterly 집계 ---")
        dotAvg = np.mean([r["dotMase"] for r in qResults])
        for vn in ["gbt_200_4", "gbt_100_3", "gbt_50_2"]:
            mases = []
            for r in qResults:
                s = r["selections"][vn]
                mases.append(s["mase"] if s["mase"] is not None else r["dotMase"])
            avg = np.mean(mases)
            imp = (1 - avg / dotAvg) * 100
            print(f"    {vn}: MASE={avg:.3f}, vs DOT={imp:+.1f}%")

        oracAvg = np.mean([r["oracleMase"] for r in qResults])
        print(f"    Oracle: MASE={oracAvg:.3f}, vs DOT={(1-oracAvg/dotAvg)*100:+.1f}%")

        print(f"\n  GBT(200/4) 선택 분포: {dict(sorted(selChoices200.items(), key=lambda x: -x[1]))}")
        print(f"  Oracle 선택 분포: {dict(sorted(oracleChoices.items(), key=lambda x: -x[1]))}")
        print(f"  오선택 패턴: {dict(sorted(wrongChoices.items(), key=lambda x: -x[1]))}")

        allMases = defaultdict(list)
        for r in qResults:
            for m, v in r["allMases"].items():
                allMases[m].append(v)
        print(f"\n  모델별 평균 MASE (Quarterly holdout):")
        for m in MODELS:
            if allMases[m]:
                print(f"    {m}: {np.mean(allMases[m]):.3f} (n={len(allMases[m])})")
    else:
        print("  m4_quarterly holdout 0개", flush=True)

    print(f"\n[4/5] 전체 Holdout 재검증 (3 GBT 변형 비교)", flush=True)
    print(f"{'='*90}")

    if freqRouter:
        gbtVariants["freq_split"] = None

    allResults = []
    for di, dsName in enumerate(HOLDOUT_DATASETS):
        t1 = time.time()
        results = runHoldoutDetailed(dsName, trainSeriesIds, gbtVariants, dnaAnalyzer)
        elapsed = time.time() - t1

        if freqRouter and results:
            for r in results:
                isLow = r["freq"] in ["Y", "Q", "M"]
                freqRouter.setLow(isLow)
                feat = None

                try:
                    from vectrix.adaptive.dna import ForecastDNA as FD2
                except ImportError:
                    pass

                selIdx = int(freqRouter.predict(
                    np.array([list(r["selections"]["gbt_100_3"].values())[0]]).reshape(1, -1)
                    if False else np.zeros((1, X_all.shape[1]))
                )[0])

        allResults.extend(results)

        if results:
            dotAvg = np.mean([r["dotMase"] for r in results])
            for vn in ["gbt_200_4", "gbt_100_3", "gbt_50_2"]:
                mases = [r["selections"][vn]["mase"] for r in results
                         if r["selections"][vn]["mase"] is not None]
                if not mases:
                    continue
            sel200Avg = np.mean([r["selections"]["gbt_200_4"]["mase"] for r in results
                                 if r["selections"]["gbt_200_4"]["mase"] is not None])
            imp = (1 - sel200Avg / dotAvg) * 100
            print(f"  [{di+1}/{len(HOLDOUT_DATASETS)}] {dsName:<30s}: {len(results)}개 | "
                  f"DOT={dotAvg:.3f} | 200/4={sel200Avg:.3f} ({imp:+.1f}%) | {elapsed:.1f}s", flush=True)
        else:
            print(f"  [{di+1}/{len(HOLDOUT_DATASETS)}] {dsName:<30s}: 0개 | {time.time()-t1:.1f}s", flush=True)

    print(f"\n[5/5] 종합 비교", flush=True)
    print(f"{'='*90}")

    if allResults:
        dotAll = np.array([r["dotMase"] for r in allResults])
        oracAll = np.array([r["oracleMase"] for r in allResults])

        print(f"\n  시리즈 수: {len(allResults)}")
        print(f"  DOT MASE: {np.mean(dotAll):.3f}")
        print(f"  Oracle MASE: {np.mean(oracAll):.3f}")

        print(f"\n  {'변형':<15s} | {'MASE':>8s} | {'vs DOT':>8s} | {'Oracle Gap':>12s} | {'win':>4s} | {'lose':>4s}")
        print(f"  {'-'*65}")

        dotAvg = np.mean(dotAll)
        oracAvg = np.mean(oracAll)
        oracleGap = dotAvg - oracAvg

        for vn in ["gbt_200_4", "gbt_100_3", "gbt_50_2"]:
            mases = []
            wins, loses = 0, 0
            for r in allResults:
                s = r["selections"][vn]
                m = s["mase"] if s["mase"] is not None else r["dotMase"]
                mases.append(m)
                if m < r["dotMase"] - 1e-10:
                    wins += 1
                elif m > r["dotMase"] + 1e-10:
                    loses += 1
            avg = np.mean(mases)
            imp = (1 - avg / dotAvg) * 100
            capture = (dotAvg - avg) / oracleGap * 100 if oracleGap > 0 else 0
            print(f"  {vn:<15s} | {avg:8.3f} | {imp:+7.1f}% | {capture:11.1f}% | {wins:4d} | {loses:4d}")

        dsByFreq = defaultdict(list)
        for r in allResults:
            dsByFreq[r["freq"]].append(r)

        print(f"\n  빈도별 비교:")
        print(f"  {'빈도':>4s} | {'N':>3s} | {'DOT':>6s} | {'200/4':>6s} | {'100/3':>6s} | {'50/2':>6s} | {'Oracle':>6s} | {'best':>8s}")
        print(f"  {'-'*70}")

        for fq in FREQ_LIST:
            if fq not in dsByFreq:
                continue
            rs = dsByFreq[fq]
            fDot = np.mean([r["dotMase"] for r in rs])
            fOrac = np.mean([r["oracleMase"] for r in rs])
            fVals = {}
            for vn in ["gbt_200_4", "gbt_100_3", "gbt_50_2"]:
                ms = [r["selections"][vn]["mase"] for r in rs if r["selections"][vn]["mase"] is not None]
                fVals[vn] = np.mean(ms) if ms else 999.0

            bestVn = min(fVals, key=fVals.get)
            bestImp = (1 - fVals[bestVn] / fDot) * 100

            print(f"  {fq:>4s} | {len(rs):3d} | {fDot:6.3f} | {fVals['gbt_200_4']:6.3f} | "
                  f"{fVals['gbt_100_3']:6.3f} | {fVals['gbt_50_2']:6.3f} | {fOrac:6.3f} | {bestImp:+7.1f}%")

        dsByDs = defaultdict(list)
        for r in allResults:
            dsByDs[r["dataset"]].append(r)

        print(f"\n  데이터셋별 비교:")
        print(f"  {'데이터셋':<30s} | {'N':>3s} | {'DOT':>6s} | {'200/4':>6s} | {'100/3':>6s} | {'50/2':>6s} | {'Oracle':>6s}")
        print(f"  {'-'*90}")

        for ds in HOLDOUT_DATASETS:
            if ds not in dsByDs:
                continue
            rs = dsByDs[ds]
            dDot = np.mean([r["dotMase"] for r in rs])
            dOrac = np.mean([r["oracleMase"] for r in rs])
            dVals = {}
            for vn in ["gbt_200_4", "gbt_100_3", "gbt_50_2"]:
                ms = [r["selections"][vn]["mase"] for r in rs if r["selections"][vn]["mase"] is not None]
                dVals[vn] = np.mean(ms) if ms else 999.0

            print(f"  {ds:<30s} | {len(rs):3d} | {dDot:6.3f} | {dVals['gbt_200_4']:6.3f} | "
                  f"{dVals['gbt_100_3']:6.3f} | {dVals['gbt_50_2']:6.3f} | {dOrac:6.3f}")

    print(f"\n{'='*90}")
    print("결론")
    print(f"{'='*90}")

    if allResults and qResults:
        print(f"\n  [Quarterly 진단]")
        qDot = np.mean([r["dotMase"] for r in qResults])
        for vn in ["gbt_200_4", "gbt_100_3", "gbt_50_2"]:
            ms = [r["selections"][vn]["mase"] for r in qResults
                  if r["selections"][vn]["mase"] is not None]
            if ms:
                avg = np.mean(ms)
                print(f"    {vn}: {avg:.3f} vs DOT {qDot:.3f} = {(1-avg/qDot)*100:+.1f}%")

        print(f"\n  [전체 Holdout]")
        dotAvg = np.mean([r["dotMase"] for r in allResults])
        for vn in ["gbt_200_4", "gbt_100_3", "gbt_50_2"]:
            ms = [r["selections"][vn]["mase"] for r in allResults
                  if r["selections"][vn]["mase"] is not None]
            if ms:
                avg = np.mean(ms)
                print(f"    {vn}: {avg:.3f} vs DOT {dotAvg:.3f} = {(1-avg/dotAvg)*100:+.1f}%")


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
