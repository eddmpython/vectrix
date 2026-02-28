"""
==============================================================================
실험 ID: metaSelection/002
실험명: DNA 특성 중요도 분석 — 모델 선택을 위한 핵심 특성 식별
==============================================================================

목적:
- DNA 65+ 특성 중 모델 선택에 가장 중요한 특성 Top-10 식별
- Ridge 분류기로 "Oracle 모델 맞추기" 학습 후 계수 기반 중요도 추출
- 규칙 기반(001)에서 놓친 특성-모델 관계 발견
- 003 메타모델 설계를 위한 특성 선별 가이드 제공

가설:
1. 65+ 특성 중 상위 15개만으로도 전체 특성 대비 90% 이상 분류 성능 달성
2. seasonalStrength, trendStrength, acf1만으로는 부족 (규칙 기반의 한계)
3. 비선형 특성(approximateEntropy, nonlinearAutocorr)이 dot/auto_ces 선택에 핵심

방법:
1. 합성 데이터 11종 × 5개 seed + 내장 7종 = 62개 시리즈 생성
2. 각 시리즈: DNA 특성 벡터 추출 + 모든 모델 MAPE 측정 → Oracle 라벨
3. Ridge 다중 분류 (특성 → Oracle 모델) 학습
4. Leave-One-Out CV로 일반화 성능 측정
5. Ridge 계수 절대값 기반 특성 중요도 랭킹
6. 상위 K개 특성만으로 정확도 커브 (K=5,10,15,...,65)

성공 기준:
- LOO-CV top-3 accuracy > 40% (001 규칙 기반 38.9% 초과)
- 상위 15개 특성이 전체 대비 분류 성능 90% 이상 달성
- 비선형 특성이 top-15에 1개 이상 포함

==============================================================================
결과 (실험 후 작성)
==============================================================================

데이터: 62개 (합성 11종 × 5 seed + 내장 7종), 65개 DNA 특성, 12개 모델

Oracle 모델 분포:
  mstl 13 (21.0%), auto_arima 9 (14.5%), tbats 8 (12.9%), window_avg 6 (9.7%),
  naive 5 (8.1%), auto_ces 5 (8.1%), garch 4 (6.5%), rwd 3 (4.8%),
  dot 3 (4.8%), theta 2 (3.2%), auto_ets 2 (3.2%), croston 2 (3.2%)

Ridge LOO-CV 정확도:
| 특성 수 | top-1 | top-3 | top-5 |
|---------|-------|-------|-------|
| 전체 65 | 32.3% | 50.0% | 62.9% |
| Top-40  |       | 59.7% |       |
| Top-30  |       | 58.1% |       |
| Top-15  |       | 56.5% |       |
| Top-10  |       | 51.6% |       |
| Top-5   |       | 38.7% |       |
| 기본 3개 |      | 41.9% |       |

Ridge 50.0% vs 001 규칙 기반 38.9% → +11.1%p 개선 확인

Top-10 핵심 특성:
  1. volatilityClustering (0.1017)
  2. seasonalPeakPeriod (0.0968)
  3. nonlinearAutocorr (0.0886)
  4. demandDensity (0.0869)
  5. hurstExponent (0.0854)
  6. seasonalAutoCorr (0.0836)
  7. structuralBreakScore (0.0833)
  8. acfFirstZero (0.0827)
  9. crossingRate (0.0775)
  10. trendSlope (0.0770)

핵심 발견:
1. seasonalStrength, trendStrength가 Top-10에 없음 → 기존 규칙 기반의 근본 한계
2. volatilityClustering이 1위 → 변동성 패턴이 모델 선택에 가장 결정적
3. 비선형 특성(nonlinearAutocorr 3위, volatilityClustering 1위)이 핵심
4. Top-15만으로 56.5% 달성 (전체 65개 대비 113%)
5. K=30~40에서 최고 성능(58~60%) → 과적합 없이 안정적
6. MSTL이 Oracle 1위(21%) → 수정 후 실제로 강력한 모델

모델별 핵심 특성:
- garch: acfFirstZero(+), hurstExponent(-) → 장기기억 없는 노이즈에서 선택
- mstl: crossingRate(-), trendSlope(+) → 트렌드 있고 교차 적은 데이터
- tbats: volatilityClustering(+), ljungBoxStat(+) → 변동성+자기상관 패턴
- auto_ces: longestRun(-), tailIndex(+) → 불규칙한 꼬리 분포 데이터

결론:
- 가설 1 채택: Top-15 = 56.5% >= 전체 50.0% × 90% (45%)
- 가설 2 기각: 기본 3개(41.9%)가 의외로 준수 (전체의 84%)
- 가설 3 채택: nonlinearAutocorr(3위), volatilityClustering(1위)이 Top-15에 포함
- 003에서 Ridge 메타모델을 vectrix에 통합, Top-30 특성 사용 권장
- 기존 규칙 기반 대비 Ridge가 +11.1%p 우수 → 메타모델 전환 타당성 확인

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


def _evaluateModel(modelFactory, train: np.ndarray, test: np.ndarray) -> float:
    model = modelFactory()
    model.fit(train)
    pred, _, _ = model.predict(len(test))
    if not np.all(np.isfinite(pred)):
        return float("inf")
    return _mape(test, pred)


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


def _buildFeatureMatrix(datasets: List[Tuple[str, np.ndarray]]) -> Tuple[np.ndarray, List[str], List[str]]:
    dna = ForecastDNA()
    featureVectors = []
    oracleLabels = []
    datasetNames = []
    featureNames = None
    factories = _buildModelFactories()

    for idx, (dsName, values) in enumerate(datasets):
        splitIdx = int(len(values) * 0.8)
        train = values[:splitIdx]
        test = values[splitIdx:]

        if len(train) < 20 or len(test) < 5:
            continue

        profile = dna.analyze(train, period=7)
        if not profile.features or len(profile.features) < 10:
            continue

        bestModel = None
        bestMape = float("inf")
        for modelName in MODELS_TO_TEST:
            mapeVal = _evaluateModel(factories[modelName], train, test)
            if mapeVal < bestMape:
                bestMape = mapeVal
                bestModel = modelName

        if bestModel is None:
            continue

        if featureNames is None:
            featureNames = sorted(profile.features.keys())

        vec = np.array([profile.features.get(f, 0.0) for f in featureNames])
        featureVectors.append(vec)
        oracleLabels.append(bestModel)
        datasetNames.append(dsName)

        if (idx + 1) % 10 == 0:
            print(f"  처리 완료: {idx + 1}/{len(datasets)}")

    X = np.array(featureVectors)
    return X, oracleLabels, featureNames, datasetNames


def _ridgeMulticlass(X: np.ndarray, y: List[str], alpha: float = 1.0) -> Tuple[Dict[str, np.ndarray], List[str]]:
    classes = sorted(set(y))
    nClasses = len(classes)
    classToIdx = {c: i for i, c in enumerate(classes)}

    n, p = X.shape
    Y = np.zeros((n, nClasses))
    for i, label in enumerate(y):
        Y[i, classToIdx[label]] = 1.0

    xMean = np.mean(X, axis=0)
    xStd = np.std(X, axis=0)
    xStd[xStd < 1e-10] = 1.0
    Xs = (X - xMean) / xStd

    I = np.eye(p)
    W = np.linalg.solve(Xs.T @ Xs + alpha * I, Xs.T @ Y)

    coefs = {}
    for i, cls in enumerate(classes):
        coefs[cls] = W[:, i]

    return coefs, classes, xMean, xStd


def _predictRidge(X: np.ndarray, coefs: Dict[str, np.ndarray], classes: List[str],
                   xMean: np.ndarray, xStd: np.ndarray) -> List[List[str]]:
    Xs = (X - xMean) / xStd
    W = np.column_stack([coefs[c] for c in classes])
    scores = Xs @ W
    predictions = []
    for i in range(len(X)):
        ranked = np.argsort(scores[i])[::-1]
        predictions.append([classes[r] for r in ranked])
    return predictions


def _looCV(X: np.ndarray, y: List[str], alpha: float = 1.0) -> Tuple[float, float, float]:
    n = len(y)
    top1Hits = 0
    top3Hits = 0
    top5Hits = 0

    for i in range(n):
        xTrain = np.delete(X, i, axis=0)
        yTrain = [y[j] for j in range(n) if j != i]
        xTest = X[i:i + 1]

        coefs, classes, xMean, xStd = _ridgeMulticlass(xTrain, yTrain, alpha)
        preds = _predictRidge(xTest, coefs, classes, xMean, xStd)

        if preds[0][0] == y[i]:
            top1Hits += 1
        if y[i] in preds[0][:3]:
            top3Hits += 1
        if y[i] in preds[0][:5]:
            top5Hits += 1

    return top1Hits / n, top3Hits / n, top5Hits / n


def _featureImportance(coefs: Dict[str, np.ndarray], featureNames: List[str]) -> List[Tuple[str, float]]:
    allCoefs = np.column_stack([coefs[c] for c in coefs])
    importance = np.mean(np.abs(allCoefs), axis=1)

    ranked = sorted(zip(featureNames, importance), key=lambda x: -x[1])
    return ranked


def _accuracyCurve(X: np.ndarray, y: List[str], featureNames: List[str],
                    importanceRanking: List[Tuple[str, float]], alpha: float = 1.0) -> List[Tuple[int, float]]:
    curve = []
    kValues = [5, 10, 15, 20, 30, 40, 50, len(featureNames)]

    for k in kValues:
        if k > len(featureNames):
            break
        topFeatures = [name for name, _ in importanceRanking[:k]]
        featureIndices = [featureNames.index(f) for f in topFeatures]
        Xk = X[:, featureIndices]

        _, top3, _ = _looCV(Xk, y, alpha)
        curve.append((k, top3))
        print(f"    K={k}: top-3 accuracy = {top3:.1%}")

    return curve


def runExperiment():
    print("=" * 70)
    print("E013: DNA 특성 중요도 분석")
    print("=" * 70)

    print("\n1. 데이터셋 준비 중...")
    t0 = time.time()
    datasets = _prepareDatasets()
    print(f"   총 {len(datasets)}개 데이터셋 준비 완료")

    print("\n2. DNA 특성 추출 + Oracle 모델 식별 중...")
    X, oracleLabels, featureNames, datasetNames = _buildFeatureMatrix(datasets)
    print(f"   유효 데이터셋: {len(oracleLabels)}개")
    print(f"   특성 수: {len(featureNames)}개")
    print("   Oracle 모델 분포:")
    from collections import Counter
    oracleCounts = Counter(oracleLabels)
    for model, count in oracleCounts.most_common():
        print(f"     {model}: {count} ({count/len(oracleLabels):.1%})")

    print("\n3. Ridge 다중 분류 학습 (전체 데이터)...")
    coefs, classes, xMean, xStd = _ridgeMulticlass(X, oracleLabels, alpha=1.0)

    print("\n4. 특성 중요도 랭킹:")
    ranking = _featureImportance(coefs, featureNames)
    print(f"   {'순위':>4} {'특성명':<30} {'중요도':>10}")
    print(f"   {'-' * 48}")
    for i, (fname, imp) in enumerate(ranking[:20]):
        print(f"   {i + 1:>4} {fname:<30} {imp:>10.4f}")

    print("\n5. LOO-CV 평가 (전체 특성)...")
    top1Full, top3Full, top5Full = _looCV(X, oracleLabels, alpha=1.0)
    print(f"   전체 {len(featureNames)}개 특성:")
    print(f"     top-1 accuracy: {top1Full:.1%}")
    print(f"     top-3 accuracy: {top3Full:.1%}")
    print(f"     top-5 accuracy: {top5Full:.1%}")

    print("\n6. 특성 수별 정확도 커브 (LOO-CV)...")
    curve = _accuracyCurve(X, oracleLabels, featureNames, ranking, alpha=1.0)

    print("\n7. 모델별 핵심 특성 (Ridge 계수 Top-5):")
    for cls in sorted(coefs.keys()):
        coefsAbs = np.abs(coefs[cls])
        topIdx = np.argsort(coefsAbs)[::-1][:5]
        topFeats = [(featureNames[i], float(coefs[cls][i])) for i in topIdx]
        print(f"   {cls}:")
        for fname, val in topFeats:
            sign = "+" if val > 0 else "-"
            print(f"     {sign} {fname} ({abs(val):.3f})")

    elapsed = time.time() - t0
    print(f"\n총 실행 시간: {elapsed:.1f}s")

    print("\n" + "=" * 70)
    print("E013 결과 요약")
    print("=" * 70)

    top15Acc = None
    for k, acc in curve:
        if k == 15:
            top15Acc = acc
            break

    print(f"\n  Ridge LOO-CV (전체 {len(featureNames)}개 특성):")
    print(f"    top-1: {top1Full:.1%}")
    print(f"    top-3: {top3Full:.1%}")
    print(f"    top-5: {top5Full:.1%}")

    if top15Acc is not None:
        ratio = top15Acc / top3Full if top3Full > 0 else 0.0
        print(f"\n  Top-15 특성 top-3 accuracy: {top15Acc:.1%}")
        print(f"  전체 대비 비율: {ratio:.1%}")

    print("\n  Top-10 핵심 특성:")
    for i, (fname, imp) in enumerate(ranking[:10]):
        print(f"    {i + 1}. {fname} ({imp:.4f})")

    nonlinearInTop15 = [
        fname for fname, _ in ranking[:15]
        if fname in ("approximateEntropy", "nonlinearAutocorr", "thirdOrderAutoCorr",
                      "volatilityClustering", "garchEffect")
    ]
    print(f"\n  Top-15 중 비선형 특성: {nonlinearInTop15 if nonlinearInTop15 else '없음'}")

    print("\n  가설 검증:")

    h1 = "채택" if top15Acc is not None and top15Acc >= top3Full * 0.9 else "기각"
    print(f"    가설 1 (Top-15 >= 90% 성능): {h1}")

    basicOnly = ["seasonalStrength", "trendStrength", "acf1"]
    basicIdx = [featureNames.index(f) for f in basicOnly if f in featureNames]
    if basicIdx:
        Xbasic = X[:, basicIdx]
        _, basicTop3, _ = _looCV(Xbasic, oracleLabels, alpha=1.0)
        h2 = "채택" if basicTop3 < top3Full * 0.8 else "기각"
        print(f"    가설 2 (기본 3개만으로는 부족): {h2} (기본 3개: {basicTop3:.1%} vs 전체: {top3Full:.1%})")
    else:
        print("    가설 2: 평가 불가")

    h3 = "채택" if nonlinearInTop15 else "기각"
    print(f"    가설 3 (비선형 특성이 Top-15에 포함): {h3}")

    print("=" * 70)


if __name__ == "__main__":
    runExperiment()
