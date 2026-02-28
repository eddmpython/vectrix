"""
==============================================================================
실험 ID: modelCreation/008
실험명: Feature-Weighted Model Combiner — FFORMA 특성 기반 모델 결합
==============================================================================

목적:
- M4 Competition 2위 FFORMA 방법론 핵심 아이디어를 numpy로 재구현
- 시계열 특성(추세 강도, 계절성 강도, 엔트로피 등)을 추출
- 특성 기반으로 각 모델에 가중치를 부여하여 결합
- DNA 특성 추출 → 모델 가중치 학습 파이프라인

가설:
1. 단일 최적 모델 선택보다 가중 결합이 평균 순위 개선
2. 특성이 유사한 데이터에서 유사한 가중치 → 일반화 성능
3. 전체 평균 순위 상위 50% (4.0 이내)
4. leave-one-out cross-validation 기반 가중치가 균등 평균보다 우수

방법:
1. 시계열 특성 추출 (10개 핵심 특성)
   - 추세 강도, 계절성 강도, ACF1, 엔트로피, 변동성
   - 비선형성, Hurst 지수, 스펙트럼 엔트로피, 정상성, 전환점 비율
2. 학습 데이터셋(여러 시계열)으로 특성→가중치 매핑 학습
3. 새 시계열: 특성 추출 → 학습된 매핑으로 가중치 결정 → 모델 결합
4. 비교: 균등 앙상블 vs FFORMA vs 단일 최적 모델

성공 기준:
- 전체 평균 순위 상위 50%
- 균등 앙상블 대비 10%+ 개선

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (11개 데이터셋):
   - mstl: 3.36 (1위)
   - 4theta: 3.55 (2위)
   - fforma: 4.00 (3위) ***
   - dot: 4.27, auto_ces: 4.73
   - equal_avg: 4.73 (6위) ***
   - arima: 5.64, theta: 5.73

2. FFORMA vs Equal Average: 9승 2패
   - 대부분의 데이터에서 특성 기반 가중이 균등 평균보다 우수
   - 특히 volatile(70.5%↑), stationary(60.0%↑), trending(35.5%↑)에서 큰 차이

3. FFORMA vs Best Single Model: 1승 10패 (9.1%)
   - hourlyMultiSeasonal에서만 1위 (10.92% vs theta 11.69%)
   - 단일 최적 모델을 이기기는 어려움 → 결합의 안정성이 가치

4. 결합의 안정성:
   - fforma 순위: 2,2,2,3,4,2,6,6,5,6,6 → 변동 폭 2~6
   - 어떤 데이터에서도 최하위(8위)가 없음 — "안전한" 결합
   - energyUsage: fforma 3.53% (2위) — mstl 3.17%에 근접
   - retailSales: fforma 5.21% (2위) — mstl 3.34% 대비 양호

5. 가중치 분석:
   - mstl 가중치 항상 최대 (0.29~0.35) — mstl이 학습셋에서 지배적
   - 특성 기반 가중치가 데이터별 분화 미흡 → 더 다양한 학습 데이터 필요
   - 11개 학습셋으로는 k-NN 매핑이 과적합

6. 가설 검증:
   - 가설 1 (가중 > 단일): 기각 — 단일 최적 모델이 대부분 우세
   - 가설 2 (일반화): 부분 채택 — 비슷한 패턴 데이터에서 비슷한 가중치
   - 가설 3 (상위 50%): 채택 — 평균 순위 4.00 (3위)
   - 가설 4 (FFORMA > equal): 채택 — 9/11 승리

결론: 조건부 채택 — 메타러닝 프레임워크로 가치
- 평균 순위 4.00 (3위) — 안정적 결합 성능
- FFORMA > equal_avg: 9/11 → 특성 기반 가중이 유효
- 단일 최적 모델을 이기지는 못하나, "안전한 결합"으로 가치
- 약점: 학습 데이터 11개로는 메타러닝 불충분, 더 큰 코퍼스 필요
- 개선: M4 전체 시리즈로 메타모델 학습 시 진정한 FFORMA 효과 기대

==============================================================================
실험일: 2026-02-28
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.signal import periodogram

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def _extractFeatures(y):
    n = len(y)
    features = {}

    diff1 = np.diff(y)
    varY = np.var(y)
    varDiff = np.var(diff1)
    features['trendStrength'] = max(0, 1.0 - varDiff / max(varY, 1e-10))

    if n >= 14:
        detrended = y - np.linspace(y[0], y[-1], n)
        freqs, power = periodogram(detrended)
        validMask = freqs > 0
        freqs = freqs[validMask]
        power = power[validMask]
        if len(power) > 0:
            totalPower = np.sum(power)
            if totalPower > 0:
                peakPower = np.max(power)
                features['seasonalStrength'] = peakPower / totalPower
                normPower = power / totalPower
                normPower = normPower[normPower > 0]
                features['spectralEntropy'] = -np.sum(normPower * np.log(normPower)) / max(np.log(len(normPower)), 1e-10)
            else:
                features['seasonalStrength'] = 0.0
                features['spectralEntropy'] = 1.0
        else:
            features['seasonalStrength'] = 0.0
            features['spectralEntropy'] = 1.0
    else:
        features['seasonalStrength'] = 0.0
        features['spectralEntropy'] = 1.0

    if n > 1:
        yNorm = (y - np.mean(y)) / max(np.std(y), 1e-10)
        acf1 = np.corrcoef(yNorm[:-1], yNorm[1:])[0, 1] if len(yNorm) > 1 else 0.0
        features['acf1'] = acf1
    else:
        features['acf1'] = 0.0

    yStd = np.std(y)
    yMean = np.mean(np.abs(y))
    features['coeffOfVar'] = yStd / max(yMean, 1e-10)

    features['volatility'] = np.std(diff1) / max(np.mean(np.abs(diff1)), 1e-10) if len(diff1) > 0 else 0.0

    turningPoints = 0
    for i in range(1, n - 1):
        if (y[i] > y[i - 1] and y[i] > y[i + 1]) or (y[i] < y[i - 1] and y[i] < y[i + 1]):
            turningPoints += 1
    features['turningPointRate'] = turningPoints / max(n - 2, 1)

    if n > 10:
        halfN = n // 2
        var1 = np.var(y[:halfN])
        var2 = np.var(y[halfN:])
        features['heteroscedasticity'] = max(var1, var2) / max(min(var1, var2), 1e-10)
    else:
        features['heteroscedasticity'] = 1.0

    features['length'] = np.log(n)

    if n > 2:
        curvature = np.diff(diff1)
        features['nonlinearity'] = np.std(curvature) / max(np.std(diff1), 1e-10)
    else:
        features['nonlinearity'] = 0.0

    if n > 20:
        cumSum = np.cumsum(y - np.mean(y))
        R = np.max(cumSum) - np.min(cumSum)
        S = max(np.std(y), 1e-10)
        features['hurstApprox'] = np.log(R / S) / max(np.log(n), 1e-10)
    else:
        features['hurstApprox'] = 0.5

    return features


def _featuresToVector(features):
    keys = sorted(features.keys())
    return np.array([features[k] for k in keys], dtype=np.float64)


class FeatureWeightedCombiner:

    def __init__(self, baseModels=None, nNeighbors=3):
        self._baseModels = baseModels
        self._nNeighbors = nNeighbors
        self._trainingFeatures = []
        self._trainingWeights = []
        self._fittedModels = {}
        self._weights = None
        self._y = None

    def trainMetaModel(self, datasetDict, horizon=14):
        if self._baseModels is None:
            self._baseModels = _buildBaseModels()

        for dsName, values in datasetDict.items():
            n = len(values)
            if n < horizon + 30:
                continue

            trainEnd = n - horizon
            train = values[:trainEnd]
            actual = values[trainEnd:trainEnd + horizon]

            features = _extractFeatures(train)
            fVec = _featuresToVector(features)

            modelMapes = {}
            for mName, mFactory in self._baseModels.items():
                try:
                    model = mFactory()
                    model.fit(train)
                    pred, _, _ = model.predict(horizon)
                    pred = np.asarray(pred[:horizon], dtype=np.float64)
                    if not np.all(np.isfinite(pred)):
                        modelMapes[mName] = float('inf')
                        continue
                    mape = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
                    modelMapes[mName] = mape
                except Exception:
                    modelMapes[mName] = float('inf')

            mapeValues = np.array([modelMapes.get(m, float('inf')) for m in sorted(self._baseModels.keys())])
            invMapes = 1.0 / np.maximum(mapeValues, 1e-10)
            infMask = ~np.isfinite(invMapes)
            invMapes[infMask] = 0.0
            total = invMapes.sum()
            if total > 0:
                weights = invMapes / total
            else:
                weights = np.ones(len(self._baseModels)) / len(self._baseModels)

            self._trainingFeatures.append(fVec)
            self._trainingWeights.append(weights)

        return self

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()

        if self._baseModels is None:
            self._baseModels = _buildBaseModels()

        features = _extractFeatures(self._y)
        queryVec = _featuresToVector(features)

        if self._trainingFeatures:
            distances = []
            for fVec in self._trainingFeatures:
                d = np.sqrt(np.sum((queryVec - fVec) ** 2))
                distances.append(d)
            distances = np.array(distances)

            k = min(self._nNeighbors, len(distances))
            nearestIdx = np.argsort(distances)[:k]

            nearDist = distances[nearestIdx]
            kernelWeights = 1.0 / np.maximum(nearDist, 1e-10)
            kernelWeights /= kernelWeights.sum()

            self._weights = np.zeros(len(self._baseModels))
            for i, idx in enumerate(nearestIdx):
                self._weights += kernelWeights[i] * self._trainingWeights[idx]
        else:
            self._weights = np.ones(len(self._baseModels)) / len(self._baseModels)

        self._fittedModels = {}
        for mName, mFactory in sorted(self._baseModels.items()):
            model = mFactory()
            model.fit(self._y)
            self._fittedModels[mName] = model

        return self

    def predict(self, steps):
        allPreds = []
        modelNames = sorted(self._baseModels.keys())

        for mName in modelNames:
            model = self._fittedModels[mName]
            pred, _, _ = model.predict(steps)
            pred = np.asarray(pred[:steps], dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                pred = np.full(steps, np.mean(self._y))
            allPreds.append(pred)

        allPreds = np.array(allPreds)
        combined = np.average(allPreds, axis=0, weights=self._weights)

        residuals = []
        for mName in modelNames:
            model = self._fittedModels[mName]
            p, _, _ = model.predict(1)
            residuals.append(abs(self._y[-1] - p[0]))
        avgResid = max(np.mean(residuals), 1e-8)

        sigma = avgResid * np.sqrt(np.arange(1, steps + 1))
        lower = combined - 1.96 * sigma
        upper = combined + 1.96 * sigma

        return combined, lower, upper


class EqualWeightCombiner:

    def __init__(self, baseModels=None):
        self._baseModels = baseModels
        self._fittedModels = {}
        self._y = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        if self._baseModels is None:
            self._baseModels = _buildBaseModels()
        self._fittedModels = {}
        for mName, mFactory in self._baseModels.items():
            model = mFactory()
            model.fit(self._y)
            self._fittedModels[mName] = model
        return self

    def predict(self, steps):
        allPreds = []
        for mName in sorted(self._baseModels.keys()):
            model = self._fittedModels[mName]
            pred, _, _ = model.predict(steps)
            pred = np.asarray(pred[:steps], dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                pred = np.full(steps, np.mean(self._y))
            allPreds.append(pred)

        allPreds = np.array(allPreds)
        combined = np.mean(allPreds, axis=0)

        spread = np.std(allPreds, axis=0)
        sigma = np.maximum(spread, 1e-8) * np.sqrt(np.arange(1, steps + 1))
        lower = combined - 1.96 * sigma
        upper = combined + 1.96 * sigma

        return combined, lower, upper


def _buildBaseModels():
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.mstl import AutoMSTL
    from vectrix.engine.theta import OptimizedTheta

    return {
        "arima": lambda: AutoARIMA(),
        "auto_ces": lambda: AutoCES(),
        "dot": lambda: DynamicOptimizedTheta(),
        "mstl": lambda: AutoMSTL(),
        "theta": lambda: OptimizedTheta(),
    }


def _generateHourlyMultiSeasonal(n=720, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    base = 500.0
    hourly = 80.0 * np.sin(2.0 * np.pi * t / 24.0)
    daily = 40.0 * np.sin(2.0 * np.pi * t / (24.0 * 7.0))
    noise = rng.normal(0, 15, n)
    return base + hourly + daily + noise


def _runExperiment():
    from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS
    from vectrix.experiments.modelCreation.e034_adaptiveThetaEnsemble import AdaptiveThetaEnsemble

    print("=" * 70)
    print("E038: Feature-Weighted Model Combiner (FFORMA)")
    print("=" * 70)

    datasets = {}
    for name, genFunc in ALL_GENERATORS.items():
        if name == "intermittentDemand":
            continue
        if name == "multiSeasonalRetail":
            df = genFunc(n=730, seed=42)
        elif name == "stockPrice":
            df = genFunc(n=252, seed=42)
        else:
            df = genFunc(n=365, seed=42)
        datasets[name] = df["value"].values

    datasets["hourlyMultiSeasonal"] = _generateHourlyMultiSeasonal(720, 42)

    horizon = 14

    print("\n  Phase 1: Training meta-model with LOO cross-validation...")
    dsNames = sorted(datasets.keys())

    fformaModels = {}
    for testDs in dsNames:
        trainDatasets = {k: v for k, v in datasets.items() if k != testDs}
        combiner = FeatureWeightedCombiner()
        combiner.trainMetaModel(trainDatasets, horizon=horizon)
        fformaModels[testDs] = combiner

    existingModels = {
        "mstl": lambda: __import__('vectrix.engine.mstl', fromlist=['AutoMSTL']).AutoMSTL(),
        "arima": lambda: __import__('vectrix.engine.arima', fromlist=['AutoARIMA']).AutoARIMA(),
        "theta": lambda: __import__('vectrix.engine.theta', fromlist=['OptimizedTheta']).OptimizedTheta(),
        "auto_ces": lambda: __import__('vectrix.engine.ces', fromlist=['AutoCES']).AutoCES(),
        "dot": lambda: __import__('vectrix.engine.dot', fromlist=['DynamicOptimizedTheta']).DynamicOptimizedTheta(),
        "4theta": lambda: AdaptiveThetaEnsemble(),
    }

    allModelNames = list(existingModels.keys()) + ["fforma", "equal_avg"]
    results = {}

    print("  Phase 2: Evaluating on each dataset...\n")

    for dsName, values in datasets.items():
        n = len(values)
        if n < horizon + 30:
            continue

        trainEnd = n - horizon
        train = values[:trainEnd]
        actual = values[trainEnd:trainEnd + horizon]

        results[dsName] = {}

        for modelName, modelFactory in existingModels.items():
            try:
                model = modelFactory()
                model.fit(train)
                pred, _, _ = model.predict(horizon)
                pred = np.asarray(pred[:horizon], dtype=np.float64)
                if not np.all(np.isfinite(pred)):
                    results[dsName][modelName] = float('inf')
                    continue
                mape = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
                results[dsName][modelName] = mape
            except Exception:
                results[dsName][modelName] = float('inf')

        combiner = fformaModels[dsName]
        combiner.fit(train)
        pred, _, _ = combiner.predict(horizon)
        pred = np.asarray(pred[:horizon], dtype=np.float64)
        if np.all(np.isfinite(pred)):
            mape = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
            results[dsName]["fforma"] = mape
        else:
            results[dsName]["fforma"] = float('inf')

        eqCombiner = EqualWeightCombiner()
        eqCombiner.fit(train)
        pred, _, _ = eqCombiner.predict(horizon)
        pred = np.asarray(pred[:horizon], dtype=np.float64)
        if np.all(np.isfinite(pred)):
            mape = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
            results[dsName]["equal_avg"] = mape
        else:
            results[dsName]["equal_avg"] = float('inf')

    print("=" * 70)
    print("ANALYSIS 1: Rankings per Dataset")
    print("=" * 70)

    rankAccum = {m: [] for m in allModelNames}

    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        sortedModels = sorted(mapes.items(), key=lambda x: x[1])

        print(f"\n  [{dsName}]")
        for rank, (mName, mape) in enumerate(sortedModels, 1):
            marker = " ***" if mName in ("fforma", "equal_avg") else ""
            mapeStr = f"{mape:.2f}" if mape < 1e6 else f"{mape:.0f}"
            print(f"    {rank}. {mName:20s} MAPE={mapeStr:>14s}%{marker}")

        for rank, (mName, _) in enumerate(sortedModels, 1):
            if mName in rankAccum:
                rankAccum[mName].append(rank)

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Average Rank")
    print("=" * 70)

    avgRanks = []
    for mName, ranks in rankAccum.items():
        if ranks:
            avgRanks.append((mName, np.mean(ranks), len(ranks)))
    avgRanks.sort(key=lambda x: x[1])

    for mName, avgRank, count in avgRanks:
        marker = " ***" if mName in ("fforma", "equal_avg") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: FFORMA vs Equal Average vs Best Single")
    print("=" * 70)

    fformaVsEqual = {"win": 0, "lose": 0, "tie": 0}
    fformaVsSingle = {"win": 0, "lose": 0}

    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        fformaVal = mapes.get("fforma", float('inf'))
        equalVal = mapes.get("equal_avg", float('inf'))
        singleMapes = {k: v for k, v in mapes.items() if k not in ("fforma", "equal_avg")}
        bestSingle = min(singleMapes.values()) if singleMapes else float('inf')
        bestSingleName = min(singleMapes, key=singleMapes.get) if singleMapes else "none"

        if fformaVal < equalVal * 0.99:
            fformaVsEqual["win"] += 1
        elif fformaVal > equalVal * 1.01:
            fformaVsEqual["lose"] += 1
        else:
            fformaVsEqual["tie"] += 1

        if fformaVal <= bestSingle:
            fformaVsSingle["win"] += 1
        else:
            fformaVsSingle["lose"] += 1

        fVsE = (equalVal - fformaVal) / max(equalVal, 1e-8) * 100
        fVsS = (bestSingle - fformaVal) / max(bestSingle, 1e-8) * 100
        print(f"  {dsName:25s} | fforma {fformaVal:8.2f}% vs equal {equalVal:8.2f}% ({fVsE:+.1f}%) vs best({bestSingleName}) {bestSingle:8.2f}% ({fVsS:+.1f}%)")

    print(f"\n  FFORMA vs Equal: {fformaVsEqual}")
    print(f"  FFORMA vs Best Single: {fformaVsSingle}")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Feature-Weight Correlation")
    print("=" * 70)

    for dsName in ["retailSales", "trending", "volatile", "hourlyMultiSeasonal"]:
        if dsName not in datasets:
            continue
        combiner = fformaModels[dsName]
        values = datasets[dsName]
        train = values[:len(values) - horizon]
        combiner.fit(train)
        modelNames = sorted(_buildBaseModels().keys())
        wStr = ", ".join([f"{m}:{w:.3f}" for m, w in zip(modelNames, combiner._weights)])
        print(f"  {dsName:25s} | {wStr}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    total = fformaVsSingle["win"] + fformaVsSingle["lose"]
    if total > 0:
        print(f"\n    FFORMA vs Best Single: {fformaVsSingle['win']}/{total} ({fformaVsSingle['win']/total*100:.1f}%)")
    print(f"    FFORMA vs Equal Avg: {fformaVsEqual}")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
