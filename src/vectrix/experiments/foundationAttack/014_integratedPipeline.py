"""
실험 ID: foundationAttack/014
실험명: Integrated Pipeline — Profile → Select → Predict 엔드투엔드

목적:
- E013의 GBT 선택기를 실제 예측 파이프라인으로 통합
- Oracle JSON 없이, 생 시계열만으로 Profile → Select → Predict 실행
- GIFT-Eval holdout 시리즈에서 실측 검증 (E012 학습에 미포함된 시리즈)
- M4 역검증 — 파이프라인이 M4에서도 기존 DOT 대비 개선되는지

가설:
1. 통합 파이프라인 MASE가 DOT 단독 대비 +10% 이상 개선 (E013에서 +15.3%)
2. M4 서브셋에서 DOT 대비 악화 < 5% (범용성 확인)
3. 파이프라인 실행 시간이 시리즈당 평균 5초 이내

방법:
1. E012 Oracle JSON으로 GBT 분류기 전체 학습 (train set = 3,165개)
2. GIFT-Eval 10개 데이터셋에서 E012에 미포함 시리즈로 holdout 구성
3. Holdout 시리즈에 Profile → Select → Predict 실행, 실제 MASE 측정
4. M4 서브셋(m4_monthly, m4_quarterly, m4_yearly)으로 역검증

데이터 리니지:
- 학습: E012 scaled_oracle_6model.json (3,230 시리즈, 6모델 MASE)
- 검증: GIFT-Eval holdout (학습에 미포함 시리즈, 데이터셋당 최대 20개)
- 역검증: GIFT-Eval m4_monthly, m4_quarterly, m4_yearly
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


def runHoldout(dsName, trainSeriesIds, gbt, dnaAnalyzer):
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

        t0 = time.time()

        try:
            profile = dnaAnalyzer.analyze(trainY, period=sp)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue

        feats = {k: float(v) if np.isfinite(v) else 0.0 for k, v in profile.features.items()}
        feat = buildFeatures(feats, freqCat)
        if np.any(np.isnan(feat)):
            continue

        selectedIdx = int(gbt.predict(feat.reshape(1, -1))[0])
        selectedModel = MODELS[selectedIdx]

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

        elapsed = time.time() - t0

        if selectedModel not in mases or "dot" not in mases:
            continue

        results.append({
            "dataset": dsName,
            "domain": domain,
            "freq": freqCat,
            "seriesIdx": int(idx),
            "selected": selectedModel,
            "selectedMase": mases[selectedModel],
            "dotMase": mases["dot"],
            "oracleMase": min(mases.values()),
            "allMases": mases,
            "elapsed": elapsed,
        })

        tested += 1
        if tested >= MAX_HOLDOUT:
            break

    return results


def main():
    np.random.seed(42)
    print("=" * 90)
    print("Integrated Pipeline — Phase 2.5, Experiment 014")
    print("=" * 90)

    from vectrix.adaptive.dna import ForecastDNA
    dnaAnalyzer = ForecastDNA()

    with open(DATA_PATH, "r") as f:
        oracleData = json.load(f)
    print(f"\n[1/4] Oracle 학습 데이터 로드: {len(oracleData)}개 시리즈", flush=True)

    XList, yLabelList = [], []

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

    X_train = np.array(XList)
    y_train = np.array(yLabelList)
    print(f"  학습 데이터: {len(X_train)}개, {X_train.shape[1]}개 특성", flush=True)
    print(f"  학습 시리즈 ID 수: {len(trainSeriesIds)}개", flush=True)

    print(f"\n[2/4] GBT 분류기 전체 학습 (200 trees, maxDepth=4)...", flush=True)
    t0 = time.time()
    gbt = GBTClassifier(nTrees=200, maxDepth=4, lr=0.1, minLeaf=5)
    gbt.fit(X_train, y_train)
    trainTime = time.time() - t0
    trainAcc = np.mean(gbt.predict(X_train) == y_train)
    print(f"  학습 완료: {trainTime:.1f}초, train acc={trainAcc:.3f}", flush=True)

    print(f"\n[3/4] GIFT-Eval Holdout 실예측 테스트", flush=True)
    print(f"  대상 데이터셋: {len(HOLDOUT_DATASETS)}개", flush=True)

    allResults = []

    for di, dsName in enumerate(HOLDOUT_DATASETS):
        t1 = time.time()
        results = runHoldout(dsName, trainSeriesIds, gbt, dnaAnalyzer)
        elapsed = time.time() - t1
        allResults.extend(results)

        if results:
            avgDot = np.mean([r["dotMase"] for r in results])
            avgSel = np.mean([r["selectedMase"] for r in results])
            imp = (1 - avgSel / avgDot) * 100 if avgDot > 0 else 0
            print(f"  [{di+1}/{len(HOLDOUT_DATASETS)}] {dsName:<30s}: "
                  f"{len(results)}개 | DOT={avgDot:.3f} | Sel={avgSel:.3f} | "
                  f"{imp:+.1f}% | {elapsed:.1f}s", flush=True)
        else:
            print(f"  [{di+1}/{len(HOLDOUT_DATASETS)}] {dsName:<30s}: 0개 (holdout 없음) | {elapsed:.1f}s", flush=True)

    print(f"\n{'='*90}")
    print("GIFT-Eval Holdout 종합 결과")
    print(f"{'='*90}", flush=True)

    if allResults:
        dotMases = np.array([r["dotMase"] for r in allResults])
        selMases = np.array([r["selectedMase"] for r in allResults])
        oracMases = np.array([r["oracleMase"] for r in allResults])
        times = np.array([r["elapsed"] for r in allResults])

        dotAvg = np.mean(dotMases)
        selAvg = np.mean(selMases)
        oracAvg = np.mean(oracMases)

        print(f"\n  시리즈 수: {len(allResults)}")
        print(f"  DOT MASE: {dotAvg:.3f}")
        print(f"  Selected MASE: {selAvg:.3f}")
        print(f"  Oracle MASE: {oracAvg:.3f}")
        if dotAvg > oracAvg:
            oracleGap = dotAvg - oracAvg
            print(f"  vs DOT: {(1-selAvg/dotAvg)*100:+.1f}%")
            print(f"  Oracle gap 캡처: {(dotAvg-selAvg)/oracleGap*100:.1f}%")
        print(f"  시리즈당 평균 시간: {np.mean(times):.2f}초")

        selCounts = defaultdict(int)
        for r in allResults:
            selCounts[r["selected"]] += 1
        print(f"\n  모델 선택 분포:")
        for m in MODELS:
            cnt = selCounts.get(m, 0)
            print(f"    {m}: {cnt}개 ({cnt/len(allResults)*100:.1f}%)")

        winCnt = np.sum(selMases < dotMases - 1e-10)
        tieCnt = np.sum(np.abs(selMases - dotMases) < 1e-10)
        loseCnt = np.sum(selMases > dotMases + 1e-10)
        print(f"\n  vs DOT: win={winCnt}, tie={tieCnt}, lose={loseCnt}")

        domainResults = defaultdict(list)
        for r in allResults:
            domainResults[r["domain"]].append(r)
        print(f"\n  도메인별 결과:")
        for dom in sorted(domainResults.keys()):
            rs = domainResults[dom]
            dDot = np.mean([r["dotMase"] for r in rs])
            dSel = np.mean([r["selectedMase"] for r in rs])
            dOrac = np.mean([r["oracleMase"] for r in rs])
            print(f"    {dom:<15s}: DOT={dDot:.3f}, Sel={dSel:.3f}, Oracle={dOrac:.3f}, "
                  f"vs DOT={((1-dSel/dDot)*100):+.1f}% ({len(rs)}개)")

        freqResults = defaultdict(list)
        for r in allResults:
            freqResults[r["freq"]].append(r)
        print(f"\n  빈도별 결과:")
        for fq in FREQ_LIST:
            if fq in freqResults:
                rs = freqResults[fq]
                fDot = np.mean([r["dotMase"] for r in rs])
                fSel = np.mean([r["selectedMase"] for r in rs])
                print(f"    {fq}: DOT={fDot:.3f}, Sel={fSel:.3f}, "
                      f"vs DOT={((1-fSel/fDot)*100):+.1f}% ({len(rs)}개)")
    else:
        print("\n  [경고] holdout 결과 없음", flush=True)

    print(f"\n[4/4] M4 역검증 (GIFT-Eval m4 서브셋)", flush=True)

    m4Datasets = ["m4_monthly", "m4_quarterly", "m4_yearly"]
    m4Results = {}

    for r in allResults:
        ds = r["dataset"]
        if ds in m4Datasets:
            if ds not in m4Results:
                m4Results[ds] = []
            m4Results[ds].append(r)

    if m4Results:
        print(f"\n  === M4 역검증 결과 ===")
        for ds in m4Datasets:
            if ds in m4Results:
                rs = m4Results[ds]
                mDot = np.mean([r["dotMase"] for r in rs])
                mSel = np.mean([r["selectedMase"] for r in rs])
                mOrac = np.mean([r["oracleMase"] for r in rs])
                print(f"  {ds} ({len(rs)}개): DOT={mDot:.3f}, Sel={mSel:.3f}, "
                      f"Oracle={mOrac:.3f}, vs DOT={((1-mSel/mDot)*100):+.1f}%", flush=True)

        allM4 = [r for ds in m4Datasets for r in m4Results.get(ds, [])]
        if allM4:
            mDotAll = np.mean([r["dotMase"] for r in allM4])
            mSelAll = np.mean([r["selectedMase"] for r in allM4])
            mOracAll = np.mean([r["oracleMase"] for r in allM4])
            print(f"\n  M4 종합: DOT={mDotAll:.3f}, Sel={mSelAll:.3f}, Oracle={mOracAll:.3f}")
            print(f"  M4 vs DOT: {(1-mSelAll/mDotAll)*100:+.1f}%", flush=True)
    else:
        print("  M4 서브셋 결과 없음 — holdout에 M4 시리즈 미포함", flush=True)

    print(f"\n{'='*90}")
    print("결론")
    print(f"{'='*90}")

    if allResults:
        selAvgH = np.mean([r["selectedMase"] for r in allResults])
        dotAvgH = np.mean([r["dotMase"] for r in allResults])
        impH = (1 - selAvgH / dotAvgH) * 100
        h1 = impH > 10
        print(f"\n  가설 1 (GIFT-Eval holdout +10%): {'통과' if h1 else '기각'} ({impH:+.1f}%)")

        allM4 = [r for ds in m4Datasets for r in m4Results.get(ds, [])]
        if allM4:
            mSelAll = np.mean([r["selectedMase"] for r in allM4])
            mDotAll = np.mean([r["dotMase"] for r in allM4])
            m4Imp = (1 - mSelAll / mDotAll) * 100
            h2 = m4Imp > -5
            print(f"  가설 2 (M4 악화 < 5%): {'통과' if h2 else '기각'} ({m4Imp:+.1f}%)")
        else:
            print(f"  가설 2: 검증 불가 (M4 holdout 없음)")

        avgTime = np.mean([r["elapsed"] for r in allResults])
        h3 = avgTime < 5.0
        print(f"  가설 3 (시리즈당 < 5초): {'통과' if h3 else '기각'} ({avgTime:.2f}초)")

        totalT = sum(r["elapsed"] for r in allResults)
        print(f"\n  총 실행 시간: {totalT:.1f}초")
    else:
        print("\n  결론 불가 — holdout 결과 없음")


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
