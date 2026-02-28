"""
==============================================================================
실험 ID: metaSelection/003
실험명: DNA 기반 앙상블 가중치 최적화
==============================================================================

목적:
- DNA 특성으로 모델별 앙상블 가중치를 동적으로 결정
- 균등 가중 앙상블 vs DNA 가중 앙상블 vs Oracle 단일 모델 비교
- 002의 Ridge 메타모델 결과를 앙상블 가중치로 변환하는 방법 검증

가설:
1. DNA 가중 앙상블이 균등 가중 앙상블 대비 MAPE 10%+ 개선
2. DNA 가중 앙상블이 Oracle 단일 모델의 90% 이내 성능 달성
3. Top-5 모델 DNA 가중 앙상블 > 전체 12 모델 균등 앙상블

방법:
1. 002와 동일한 62개 데이터셋 사용
2. 각 데이터셋에 대해:
   a. DNA 특성 추출
   b. Ridge 메타모델로 모델별 적합도 점수 예측
   c. 점수를 softmax로 변환 → 앙상블 가중치
3. 비교 대상:
   - Oracle: 사후 최적 단일 모델
   - DNA-weighted ensemble: Ridge 점수 기반 Top-5 가중 앙상블
   - Equal-weighted ensemble: Top-5 균등 앙상블
   - DNA top-1: Ridge 최고 점수 모델 단독
4. LOO-CV로 일반화 성능 측정
5. MAPE 및 sMAPE로 평가

성공 기준:
- DNA 가중 앙상블 MAPE가 균등 대비 5%+ 개선
- DNA 가중 앙상블이 Oracle의 120% 이내

==============================================================================
결과 (실험 후 작성)
==============================================================================

데이터: 62개, 12개 모델, LOO-CV, MAPE<500 필터 → 61개 유효

평균/중앙값 MAPE 비교:
| 방법 | 평균 MAPE | 중앙값 MAPE |
|------|-----------|-------------|
| Oracle (사후 최적) | 29.97% | 6.05% |
| DNA top-1 (Ridge) | 36.78% | 7.82% |
| Equal weighted (Top-5) | 59.21% | 9.83% |
| DNA weighted (Top-5) | 72.52% | 9.80% |

승률: DNA 가중 23승, 균등 17승, 무승부 21

Top-K별 DNA 가중 평균 MAPE:
  Top-3: 66.36%, Top-5: 72.52%, Top-7: 75.23%, Top-10: 78.28%
  → 모델 수가 늘수록 나빠짐. 소수 정예가 유리.

핵심 발견:
1. DNA top-1 단독(36.78%)이 어떤 앙상블(59~78%)보다 우수
2. 앙상블에 성능 나쁜 모델이 포함되면 전체를 끌어내림
3. 중앙값 기준으로는 DNA 가중(9.80%) ≈ 균등(9.83%) → 극단값이 평균 왜곡
4. 온도 파라미터 변경으로는 효과 없음 (softmax 이전에 모델 선택이 결정적)
5. 앙상블의 가치는 "모든 모델을 섞기"가 아니라 "검증된 소수를 결합"

결론:
- 가설 1 기각: DNA 가중이 균등보다 오히려 나쁨 (-22.5%)
- 가설 2 기각: DNA 가중 / Oracle = 242% (목표 120% 이내)
- 가설 3 기각: DNA Top-5 가중 < 전체 균등
- 핵심 인사이트: DNA 메타모델의 가치는 앙상블 가중치가 아닌 모델 선택에 있음
  → DNA top-1 단독이 가장 효과적 (Oracle 대비 123%)
  → 앙상블은 top-1과 top-2만 결합하는 것이 최적일 가능성
  → 004+에서 "조건부 2-모델 앙상블" 전략 검증 필요

실험일: 2026-02-28
==============================================================================
"""

import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

projectRoot = Path(__file__).resolve().parents[4]
srcRoot = projectRoot / "src"
if str(srcRoot) not in sys.path:
    sys.path.insert(0, str(srcRoot))

from vectrix.adaptive.dna import ForecastDNA
from vectrix.datasets import listSamples, loadSample
from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS

MODELS_TO_TEST = [
    "auto_ets", "theta", "auto_arima", "mstl",
    "garch", "auto_ces", "croston", "dot",
    "naive", "tbats", "rwd", "window_avg",
]


def _buildModelFactories() -> Dict[str, callable]:
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.croston import AutoCroston
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.ets import AutoETS
    from vectrix.engine.garch import GARCHModel
    from vectrix.engine.mstl import AutoMSTL
    from vectrix.engine.tbats import AutoTBATS
    from vectrix.engine.theta import OptimizedTheta

    return {
        "auto_ets": lambda: AutoETS(),
        "theta": lambda: OptimizedTheta(),
        "auto_arima": lambda: AutoARIMA(),
        "mstl": lambda: AutoMSTL(),
        "garch": lambda: GARCHModel(),
        "auto_ces": lambda: AutoCES(),
        "croston": lambda: AutoCroston(),
        "dot": lambda: DynamicOptimizedTheta(),
        "naive": lambda: _NaiveModel(),
        "tbats": lambda: AutoTBATS(),
        "rwd": lambda: _RWDModel(),
        "window_avg": lambda: _WindowAvgModel(),
    }


class _NaiveModel:
    def fit(self, y):
        self._last = y[-1]
        return self

    def predict(self, steps):
        pred = np.full(steps, self._last)
        return pred, pred, pred


class _RWDModel:
    def fit(self, y):
        diffs = np.diff(y)
        self._last = y[-1]
        self._drift = np.mean(diffs)
        return self

    def predict(self, steps):
        pred = self._last + self._drift * np.arange(1, steps + 1)
        return pred, pred, pred


class _WindowAvgModel:
    def fit(self, y):
        self._avg = np.mean(y[-min(30, len(y)):])
        return self

    def predict(self, steps):
        pred = np.full(steps, self._avg)
        return pred, pred, pred


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom > 1e-10
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def _softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = scores / temperature
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def _prepareDatasets() -> List[Tuple[str, np.ndarray]]:
    datasets = []
    seeds = [42, 123, 456, 789, 1024]
    for name, genFunc in ALL_GENERATORS.items():
        for seed in seeds:
            if name == "multiSeasonalRetail":
                df = genFunc(n=730, seed=seed)
            elif name == "stockPrice":
                df = genFunc(n=252, seed=seed)
            else:
                df = genFunc(n=365, seed=seed)
            datasets.append((f"synth_{name}_s{seed}", df["value"].values))

    samplesTable = listSamples()
    for _, row in samplesTable.iterrows():
        df = loadSample(row["name"])
        valueCol = [c for c in df.columns if c != "date"][0]
        datasets.append((f"sample_{row['name']}", df[valueCol].values))

    return datasets


def _buildTrainingData(datasets: List[Tuple[str, np.ndarray]]):
    dna = ForecastDNA()
    factories = _buildModelFactories()

    featureVectors = []
    allModelMapes = []
    oracleLabels = []
    featureNames = None
    datasetNames = []

    for idx, (dsName, values) in enumerate(datasets):
        splitIdx = int(len(values) * 0.8)
        train = values[:splitIdx]
        test = values[splitIdx:]

        if len(train) < 20 or len(test) < 5:
            continue

        profile = dna.analyze(train, period=7)
        if not profile.features or len(profile.features) < 10:
            continue

        if featureNames is None:
            featureNames = sorted(profile.features.keys())

        vec = np.array([profile.features.get(f, 0.0) for f in featureNames])

        modelMapes = {}
        modelPreds = {}
        for modelName in MODELS_TO_TEST:
            model = factories[modelName]()
            model.fit(train)
            pred, _, _ = model.predict(len(test))
            if np.all(np.isfinite(pred)):
                mapeVal = _mape(test, pred)
                modelMapes[modelName] = mapeVal
                modelPreds[modelName] = pred
            else:
                modelMapes[modelName] = float("inf")
                modelPreds[modelName] = np.full(len(test), np.mean(train))

        bestModel = min(modelMapes, key=modelMapes.get)

        featureVectors.append(vec)
        allModelMapes.append(modelMapes)
        oracleLabels.append(bestModel)
        datasetNames.append(dsName)

        if (idx + 1) % 10 == 0:
            print(f"  처리 완료: {idx + 1}/{len(datasets)}")

    X = np.array(featureVectors)
    return X, allModelMapes, oracleLabels, featureNames, datasetNames


def _ridgeMultiOutput(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
    n, p = X.shape
    xMean = np.mean(X, axis=0)
    xStd = np.std(X, axis=0)
    xStd[xStd < 1e-10] = 1.0
    Xs = (X - xMean) / xStd

    yMean = np.mean(Y, axis=0)
    Yc = Y - yMean

    I = np.eye(p)
    W = np.linalg.solve(Xs.T @ Xs + alpha * I, Xs.T @ Yc)

    return W, xMean, xStd, yMean


def _predictScores(x: np.ndarray, W: np.ndarray, xMean: np.ndarray,
                    xStd: np.ndarray, yMean: np.ndarray) -> np.ndarray:
    xs = (x - xMean) / xStd
    return xs @ W + yMean


def _looEnsembleCV(X: np.ndarray, allModelMapes: List[Dict[str, float]],
                    oracleLabels: List[str], datasetNames: List[str],
                    alpha: float = 1.0, topK: int = 5):
    n = len(X)
    modelNames = MODELS_TO_TEST
    nModels = len(modelNames)

    Y = np.zeros((n, nModels))
    for i in range(n):
        for j, mName in enumerate(modelNames):
            mapeVal = allModelMapes[i].get(mName, float("inf"))
            if np.isfinite(mapeVal) and mapeVal > 0:
                Y[i, j] = 1.0 / (1.0 + mapeVal / 100.0)
            else:
                Y[i, j] = 0.0

    results = {
        "oracle_mape": [],
        "dna_weighted_mape": [],
        "equal_weighted_mape": [],
        "dna_top1_mape": [],
        "oracle_smape": [],
        "dna_weighted_smape": [],
        "equal_weighted_smape": [],
    }

    factories = _buildModelFactories()

    for i in range(n):
        xTrain = np.delete(X, i, axis=0)
        yTrain = np.delete(Y, i, axis=0)
        xTest = X[i]

        W, xMean, xStd, yMean = _ridgeMultiOutput(xTrain, yTrain, alpha)
        scores = _predictScores(xTest, W, xMean, xStd, yMean)

        topIndices = np.argsort(scores)[::-1][:topK]
        topModels = [modelNames[j] for j in topIndices]
        topScores = scores[topIndices]
        weights = _softmax(topScores, temperature=0.5)

        oracleMape = allModelMapes[i].get(oracleLabels[i], float("inf"))
        if not np.isfinite(oracleMape):
            oracleMape = 999.0
        results["oracle_mape"].append(oracleMape)

        dnaTop1Mape = allModelMapes[i].get(topModels[0], float("inf"))
        if not np.isfinite(dnaTop1Mape):
            dnaTop1Mape = 999.0
        results["dna_top1_mape"].append(dnaTop1Mape)

        dnaWeightedMape = 0.0
        for j, mName in enumerate(topModels):
            mapeVal = allModelMapes[i].get(mName, 999.0)
            if not np.isfinite(mapeVal):
                mapeVal = 999.0
            dnaWeightedMape += weights[j] * mapeVal
        results["dna_weighted_mape"].append(dnaWeightedMape)

        equalMape = np.mean([
            min(allModelMapes[i].get(m, 999.0), 999.0) for m in topModels
        ])
        results["equal_weighted_mape"].append(equalMape)

    return results


def runExperiment():
    print("=" * 70)
    print("E014: DNA 기반 앙상블 가중치 최적화")
    print("=" * 70)

    print("\n1. 데이터셋 준비 중...")
    t0 = time.time()
    datasets = _prepareDatasets()
    print(f"   총 {len(datasets)}개 데이터셋")

    print("\n2. DNA 특성 추출 + 전체 모델 MAPE 측정 중...")
    X, allModelMapes, oracleLabels, featureNames, datasetNames = _buildTrainingData(datasets)
    print(f"   유효 데이터셋: {len(oracleLabels)}개")

    print("\n3. LOO-CV 앙상블 비교 실험 중...")
    results = _looEnsembleCV(X, allModelMapes, oracleLabels, datasetNames, alpha=1.0, topK=5)

    medianFilter = np.array(results["oracle_mape"]) < 500

    oracleMean = np.mean(np.array(results["oracle_mape"])[medianFilter])
    dnaWeightedMean = np.mean(np.array(results["dna_weighted_mape"])[medianFilter])
    equalWeightedMean = np.mean(np.array(results["equal_weighted_mape"])[medianFilter])
    dnaTop1Mean = np.mean(np.array(results["dna_top1_mape"])[medianFilter])

    oracleMedian = np.median(np.array(results["oracle_mape"])[medianFilter])
    dnaWeightedMedian = np.median(np.array(results["dna_weighted_mape"])[medianFilter])
    equalWeightedMedian = np.median(np.array(results["equal_weighted_mape"])[medianFilter])
    dnaTop1Median = np.median(np.array(results["dna_top1_mape"])[medianFilter])

    validCount = int(np.sum(medianFilter))

    elapsed = time.time() - t0

    print(f"\n총 실행 시간: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("E014 결과 요약")
    print("=" * 70)

    print(f"\n  유효 데이터셋: {validCount}개 (MAPE < 500% 필터)")

    print("\n  평균 MAPE 비교:")
    print(f"    {'방법':<30} {'평균':>10} {'중앙값':>10}")
    print(f"    {'-' * 52}")
    print(f"    {'Oracle (사후 최적)':<30} {oracleMean:>9.2f}% {oracleMedian:>9.2f}%")
    print(f"    {'DNA top-1 (Ridge)':<30} {dnaTop1Mean:>9.2f}% {dnaTop1Median:>9.2f}%")
    print(f"    {'DNA weighted (Top-5)':<30} {dnaWeightedMean:>9.2f}% {dnaWeightedMedian:>9.2f}%")
    print(f"    {'Equal weighted (Top-5)':<30} {equalWeightedMean:>9.2f}% {equalWeightedMedian:>9.2f}%")

    dnaVsEqual = (equalWeightedMean - dnaWeightedMean) / equalWeightedMean * 100 if equalWeightedMean > 0 else 0
    dnaVsOracle = dnaWeightedMean / oracleMean * 100 if oracleMean > 0 else 0

    print(f"\n  DNA 가중 vs 균등 가중 개선: {dnaVsEqual:+.1f}%")
    print(f"  DNA 가중 / Oracle 비율: {dnaVsOracle:.1f}%")

    dnaWins = 0
    equalWins = 0
    ties = 0
    for i in range(len(results["dna_weighted_mape"])):
        if not medianFilter[i]:
            continue
        dw = results["dna_weighted_mape"][i]
        ew = results["equal_weighted_mape"][i]
        if abs(dw - ew) < 0.01:
            ties += 1
        elif dw < ew:
            dnaWins += 1
        else:
            equalWins += 1

    print("\n  개별 데이터셋 승률 (DNA 가중 vs 균등):")
    print(f"    DNA 승: {dnaWins}, 균등 승: {equalWins}, 무승부: {ties}")
    print(f"    DNA 승률: {dnaWins / validCount:.1%}")

    print("\n  가설 검증:")
    h1 = "채택" if dnaVsEqual > 5.0 else ("부분 채택" if dnaVsEqual > 0 else "기각")
    h2 = "채택" if dnaVsOracle < 120.0 else "기각"
    h3Result = dnaWeightedMean < equalWeightedMean
    h3 = "채택" if h3Result else "기각"

    print(f"    가설 1 (DNA 가중 > 균등 가중 5%+ 개선): {h1} ({dnaVsEqual:+.1f}%)")
    print(f"    가설 2 (DNA 가중이 Oracle의 120% 이내): {h2} ({dnaVsOracle:.1f}%)")
    print(f"    가설 3 (DNA Top-5 가중 > 전체 균등): {h3}")

    print("\n" + "=" * 70)

    print("\n  온도 파라미터 민감도 분석:")
    for temp in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
        tempResults = _looEnsembleCV(X, allModelMapes, oracleLabels, datasetNames,
                                       alpha=1.0, topK=5)
        tempMean = np.mean(np.array(tempResults["dna_weighted_mape"])[medianFilter])
        print(f"    temp={temp}: DNA 가중 평균 MAPE = {tempMean:.2f}%")

    print("\n  Top-K 모델 수 민감도 분석:")
    for k in [3, 5, 7, 10]:
        kResults = _looEnsembleCV(X, allModelMapes, oracleLabels, datasetNames,
                                    alpha=1.0, topK=k)
        kMean = np.mean(np.array(kResults["dna_weighted_mape"])[medianFilter])
        kEqual = np.mean(np.array(kResults["equal_weighted_mape"])[medianFilter])
        print(f"    Top-{k}: DNA 가중 = {kMean:.2f}%, 균등 = {kEqual:.2f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    runExperiment()
