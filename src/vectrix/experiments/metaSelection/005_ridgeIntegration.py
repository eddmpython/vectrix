"""
==============================================================================
실험 ID: metaSelection/005
실험명: Ridge 메타모델 dna.py 통합 실현 가능성 검증
==============================================================================

목적:
- 002~004에서 Ridge 메타모델이 규칙 기반보다 우수함을 확인
- 실제 dna.py에 Ridge 계수를 내장(embed)하여 _recommendModels()를 교체할 수 있는지 검증
- 사전학습 계수 + 정규화 파라미터를 코드에 하드코딩하는 방식의 타당성 확인

가설:
1. 62개 합성/내장 데이터에서 학습한 Ridge 계수가 새로운 데이터에도 일반화됨
2. 사전학습 Ridge가 규칙 기반 대비 top-3 hit rate 10%p+ 개선
3. Ridge 추론 시간이 규칙 기반과 동등 (< 1ms)

방법:
1. 전체 62개 데이터에서 Ridge 학습 → 계수(W), 정규화(xMean, xStd) 추출
2. 계수를 numpy 배열로 직렬화 가능한 형태로 출력
3. LOO-CV로 일반화 성능 최종 확인
4. 규칙 기반 _recommendModels()와의 A/B 비교
5. 추론 시간 벤치마크

산출물:
- Ridge 계수 행렬 (65 features × N classes)
- 정규화 파라미터 (xMean, xStd)
- 클래스 목록 (모델 이름 순서)
- 특성 이름 목록 (정렬 순서)
- 통합 코드 스니펫

성공 기준:
- LOO-CV top-3 >= 50% (002 수준 유지)
- 규칙 기반 대비 top-3 hit 10%p+ 개선
- 추론 시간 < 1ms

==============================================================================
결과 (실험 후 작성)
==============================================================================

데이터: 62개 유효, 65개 DNA 특성, 12개 모델 클래스

Oracle 적중률 비교:
| 방법 | top-1 | top-3 | top-5 |
|------|-------|-------|-------|
| Ridge LOO-CV | 32.3% | 50.0% | 62.9% |
| 규칙 기반 DNA | 6.5% | 21.0% | 40.3% |
| Ridge 사전학습(train) | 93.5% | 100.0% | 100.0% |

→ Ridge는 규칙 기반 대비 모든 지표에서 압도적 우위 (+29%p top-3)
→ 그러나 train 100% vs LOO-CV 50% = 심각한 과적합 (gap 50%)

평균 MAPE 비교:
| 방법 | 평균 MAPE |
|------|-----------|
| Oracle | 29.97% |
| Ridge top-1 | 30.02% |
| 규칙 기반 top-1 | 73.32% |

→ Ridge MAPE(30.02%)가 Oracle(29.97%)과 거의 동일! (사전학습 기준)
→ 규칙 기반(73.32%)은 2.4배 나쁨

승률: Ridge 55승, 규칙 2승, 무승부 4 (61전)

추론 속도:
  Ridge: 0.0039 ms/prediction (규칙 기반 0.0050ms보다 빠름!)
  내장 메모리: 7,280 bytes (7.1 KB)
  파라미터 수: 910개

Alpha 하이퍼파라미터 탐색:
  alpha=0.01: top-3=53.2%, top-5=69.4% ← 최적
  alpha=0.50: top-3=53.2%, top-5=62.9%
  alpha=5.00: top-3=51.6%, top-5=66.1%
  → alpha=0.01이 최적이나, 과적합 우려

핵심 발견:
1. Ridge가 규칙 기반을 모든 지표에서 압도 (top-3: 50% vs 21%, MAPE: 30 vs 73%)
2. 사전학습 계수의 과적합이 심각 (train 100% vs test 50%)
   → 62개 학습 데이터가 12 클래스 분류에 부족 (클래스당 평균 5.2개)
3. 추론 속도는 규칙 기반보다 빠름 (0.004 vs 0.005ms)
4. 내장 크기 7.1KB로 매우 작음 — 코드 내장 가능
5. 과적합 해결 전략: 학습 데이터 확대 (M4 데이터 활용) 또는 정규화 강화

결론:
- 가설 1 기각: 과적합 갭 50% → 62개로는 일반화 불충분
- 가설 2 채택: top-3 +29%p 개선 (50% vs 21%)
- 가설 3 채택: 추론 0.004ms < 1ms
- 즉시 통합은 부적절 (과적합), 그러나 방향성은 확실
- 다음 단계: M4 데이터로 학습셋 확대 → 과적합 해소 후 통합
  또는 규칙 기반 + Ridge 하이브리드 (규칙으로 후보 축소 → Ridge로 최종 선택)

실험일: 2026-02-28
==============================================================================
"""

import io
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

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
    oracleLabels = []
    featureNames = None
    datasetNames = []
    allModelMapes = []
    dnaRecommendations = []

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
        for modelName in MODELS_TO_TEST:
            model = factories[modelName]()
            model.fit(train)
            pred, _, _ = model.predict(len(test))
            if np.all(np.isfinite(pred)):
                modelMapes[modelName] = _mape(test, pred)
            else:
                modelMapes[modelName] = float("inf")

        bestModel = min(modelMapes, key=modelMapes.get)

        featureVectors.append(vec)
        oracleLabels.append(bestModel)
        datasetNames.append(dsName)
        allModelMapes.append(modelMapes)
        dnaRecommendations.append(profile.recommendedModels[:5])

        if (idx + 1) % 10 == 0:
            print(f"  처리 완료: {idx + 1}/{len(datasets)}")

    X = np.array(featureVectors)
    return X, oracleLabels, featureNames, datasetNames, allModelMapes, dnaRecommendations


def _trainFullRidge(X: np.ndarray, y: List[str], alpha: float = 1.0):
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

    return W, classes, xMean, xStd


def _ridgePredict(x: np.ndarray, W: np.ndarray, classes: List[str],
                   xMean: np.ndarray, xStd: np.ndarray, topK: int = 5) -> List[str]:
    xs = (x - xMean) / xStd
    scores = xs @ W
    ranked = np.argsort(scores)[::-1][:topK]
    return [classes[r] for r in ranked]


def _looCV(X: np.ndarray, y: List[str], alpha: float = 1.0):
    n = len(y)
    top1Hits = 0
    top3Hits = 0
    top5Hits = 0

    for i in range(n):
        xTrain = np.delete(X, i, axis=0)
        yTrain = [y[j] for j in range(n) if j != i]
        xTest = X[i]

        W, classes, xMean, xStd = _trainFullRidge(xTrain, yTrain, alpha)
        preds = _ridgePredict(xTest, W, classes, xMean, xStd, topK=5)

        if preds[0] == y[i]:
            top1Hits += 1
        if y[i] in preds[:3]:
            top3Hits += 1
        if y[i] in preds[:5]:
            top5Hits += 1

    return top1Hits / n, top3Hits / n, top5Hits / n


def _benchmarkInference(X: np.ndarray, W: np.ndarray, classes: List[str],
                          xMean: np.ndarray, xStd: np.ndarray, nReps: int = 10000):
    x = X[0]

    t0 = time.perf_counter()
    for _ in range(nReps):
        _ridgePredict(x, W, classes, xMean, xStd, topK=5)
    elapsed = time.perf_counter() - t0

    return elapsed / nReps * 1000


def _exportCoefficients(W: np.ndarray, classes: List[str],
                         xMean: np.ndarray, xStd: np.ndarray,
                         featureNames: List[str]):
    result = {
        "featureNames": featureNames,
        "classes": classes,
        "xMean": xMean.tolist(),
        "xStd": xStd.tolist(),
        "W": W.tolist(),
    }

    print("\n  계수 행렬 크기: {} features x {} classes".format(W.shape[0], W.shape[1]))
    print(f"  클래스 목록: {classes}")
    print(f"  특성 수: {len(featureNames)}")

    wFlat = W.flatten()
    print(f"\n  계수 통계: mean={np.mean(wFlat):.6f}, std={np.std(wFlat):.6f}")
    print(f"  계수 범위: [{np.min(wFlat):.6f}, {np.max(wFlat):.6f}]")
    print(f"  0에 가까운 계수(|w|<0.01): {np.sum(np.abs(wFlat) < 0.01)}/{len(wFlat)}")

    return result


def _generateCodeSnippet(W: np.ndarray, classes: List[str],
                           xMean: np.ndarray, xStd: np.ndarray,
                           featureNames: List[str]):
    print("\n  === Ridge 메타모델 통합 코드 스니펫 ===")
    print(f"  _RIDGE_CLASSES = {classes}")
    print(f"  _RIDGE_FEATURES = {featureNames[:5]}... (총 {len(featureNames)}개)")
    print(f"  _RIDGE_W = np.array(...)  # shape ({W.shape[0]}, {W.shape[1]})")
    print(f"  _RIDGE_XMEAN = np.array(...)  # shape ({len(xMean)},)")
    print(f"  _RIDGE_XSTD = np.array(...)  # shape ({len(xStd)},)")
    print("\n  def _recommendModelsRidge(self, features):")
    print("      vec = np.array([features.get(f, 0.0) for f in _RIDGE_FEATURES])")
    print("      xs = (vec - _RIDGE_XMEAN) / _RIDGE_XSTD")
    print("      scores = xs @ _RIDGE_W")
    print("      ranked = np.argsort(scores)[::-1][:5]")
    print("      return [_RIDGE_CLASSES[r] for r in ranked]")

    totalBytes = W.nbytes + xMean.nbytes + xStd.nbytes
    print(f"\n  내장 시 메모리: {totalBytes:,} bytes ({totalBytes / 1024:.1f} KB)")

    nFeatures = len(featureNames)
    nClasses = len(classes)
    nParams = nFeatures * nClasses + nFeatures * 2
    print(f"  총 파라미터 수: {nParams:,}")


def runExperiment():
    print("=" * 70)
    print("E016: Ridge 메타모델 dna.py 통합 실현 가능성 검증")
    print("=" * 70)

    print("\n1. 데이터셋 준비 중...")
    t0 = time.time()
    datasets = _prepareDatasets()
    print(f"   총 {len(datasets)}개")

    print("\n2. DNA 특성 추출 + 전체 모델 MAPE 측정 중...")
    X, oracleLabels, featureNames, datasetNames, allModelMapes, dnaRecs = _buildTrainingData(datasets)
    print(f"   유효: {len(oracleLabels)}개, 특성: {len(featureNames)}개")

    oracleCounts = Counter(oracleLabels)
    print("   Oracle 분포:")
    for model, count in oracleCounts.most_common(5):
        print(f"     {model}: {count} ({count / len(oracleLabels):.1%})")

    print("\n3. LOO-CV 성능 확인...")
    top1, top3, top5 = _looCV(X, oracleLabels, alpha=1.0)
    print(f"   Ridge LOO-CV: top-1={top1:.1%}, top-3={top3:.1%}, top-5={top5:.1%}")

    print("\n4. 규칙 기반 DNA 성능 비교...")
    ruleTop1 = sum(1 for i, recs in enumerate(dnaRecs) if recs and recs[0] == oracleLabels[i])
    ruleTop3 = sum(1 for i, recs in enumerate(dnaRecs) if oracleLabels[i] in recs[:3])
    ruleTop5 = sum(1 for i, recs in enumerate(dnaRecs) if oracleLabels[i] in recs[:5])
    n = len(oracleLabels)
    print(f"   규칙 기반: top-1={ruleTop1 / n:.1%}, top-3={ruleTop3 / n:.1%}, top-5={ruleTop5 / n:.1%}")

    print("\n5. 전체 데이터 Ridge 학습 (사전학습 계수 추출)...")
    W, classes, xMean, xStd = _trainFullRidge(X, oracleLabels, alpha=1.0)

    print("\n6. 사전학습 계수 분석...")
    coeffData = _exportCoefficients(W, classes, xMean, xStd, featureNames)

    print("\n7. 추론 시간 벤치마크...")
    inferenceMs = _benchmarkInference(X, W, classes, xMean, xStd, nReps=10000)
    print(f"   Ridge 추론: {inferenceMs:.4f} ms/prediction")

    dna = ForecastDNA()
    t1 = time.perf_counter()
    for _ in range(1000):
        dna._recommendModels(dict(zip(featureNames, X[0])))
    ruleMs = (time.perf_counter() - t1) / 1000 * 1000
    print(f"   규칙 기반 추론: {ruleMs:.4f} ms/prediction")
    print(f"   속도 비율: Ridge/Rule = {inferenceMs / ruleMs:.2f}x")

    print("\n8. 사전학습 모델 vs LOO-CV 비교 (과적합 체크)...")
    fullTrainPreds = []
    for i in range(len(X)):
        preds = _ridgePredict(X[i], W, classes, xMean, xStd, topK=5)
        fullTrainPreds.append(preds)

    trainTop1 = sum(1 for i in range(n) if fullTrainPreds[i][0] == oracleLabels[i])
    trainTop3 = sum(1 for i in range(n) if oracleLabels[i] in fullTrainPreds[i][:3])
    trainTop5 = sum(1 for i in range(n) if oracleLabels[i] in fullTrainPreds[i][:5])
    print(f"   사전학습(train): top-1={trainTop1 / n:.1%}, top-3={trainTop3 / n:.1%}, top-5={trainTop5 / n:.1%}")
    print(f"   LOO-CV(test):    top-1={top1:.1%}, top-3={top3:.1%}, top-5={top5:.1%}")
    overfit = (trainTop3 / n - top3) / (trainTop3 / n) * 100 if trainTop3 > 0 else 0
    print(f"   과적합 갭 (top-3): {overfit:.1f}%")

    print("\n9. MAPE 비교 (Ridge vs 규칙 기반)...")
    mapeFilter = [m.get(oracleLabels[i], 999) < 500 for i, m in enumerate(allModelMapes)]

    ridgeMapes = []
    ruleMapes = []
    oracleMapes = []

    for i in range(n):
        if not mapeFilter[i]:
            continue
        ridgeTop1 = fullTrainPreds[i][0]
        ridgeMape = allModelMapes[i].get(ridgeTop1, 999.0)
        if not np.isfinite(ridgeMape):
            ridgeMape = 999.0
        ridgeMapes.append(ridgeMape)

        ruleTop1Model = dnaRecs[i][0] if dnaRecs[i] else "auto_ets"
        ruleMape = allModelMapes[i].get(ruleTop1Model, 999.0)
        if not np.isfinite(ruleMape):
            ruleMape = 999.0
        ruleMapes.append(ruleMape)

        oracleMape = allModelMapes[i].get(oracleLabels[i], 999.0)
        if not np.isfinite(oracleMape):
            oracleMape = 999.0
        oracleMapes.append(oracleMape)

    validN = len(ridgeMapes)
    print(f"   유효: {validN}개")
    print(f"   Oracle 평균 MAPE: {np.mean(oracleMapes):.2f}%")
    print(f"   Ridge top-1 평균 MAPE: {np.mean(ridgeMapes):.2f}%")
    print(f"   규칙 기반 top-1 평균 MAPE: {np.mean(ruleMapes):.2f}%")

    ridgeWins = sum(1 for r, ru in zip(ridgeMapes, ruleMapes) if r < ru)
    ruleWins = sum(1 for r, ru in zip(ridgeMapes, ruleMapes) if ru < r)
    ties = validN - ridgeWins - ruleWins
    print(f"   승률: Ridge {ridgeWins}승, 규칙 {ruleWins}승, 무승부 {ties}")

    _generateCodeSnippet(W, classes, xMean, xStd, featureNames)

    print("\n10. Alpha 하이퍼파라미터 탐색...")
    for alpha in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        aTop1, aTop3, aTop5 = _looCV(X, oracleLabels, alpha=alpha)
        print(f"    alpha={alpha:>5.2f}: top-1={aTop1:.1%}, top-3={aTop3:.1%}, top-5={aTop5:.1%}")

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("E016 결과 요약")
    print("=" * 70)

    print(f"\n  데이터셋: {validN}개 유효")

    print("\n  Oracle 적중률 비교:")
    print(f"  {'방법':<30} {'top-1':>8} {'top-3':>8} {'top-5':>8}")
    print(f"  {'-' * 58}")
    print(f"  {'Ridge LOO-CV':<30} {top1:.1%}{'':<4} {top3:.1%}{'':<4} {top5:.1%}")
    print(f"  {'규칙 기반 DNA':<30} {ruleTop1 / n:.1%}{'':<4} {ruleTop3 / n:.1%}{'':<4} {ruleTop5 / n:.1%}")
    print(f"  {'Ridge 사전학습(train)':<30} {trainTop1 / n:.1%}{'':<4} {trainTop3 / n:.1%}{'':<4} {trainTop5 / n:.1%}")

    print("\n  평균 MAPE:")
    print(f"    Oracle: {np.mean(oracleMapes):.2f}%")
    print(f"    Ridge: {np.mean(ridgeMapes):.2f}%")
    print(f"    규칙 기반: {np.mean(ruleMapes):.2f}%")

    print("\n  추론 속도:")
    print(f"    Ridge: {inferenceMs:.4f} ms")
    print(f"    규칙 기반: {ruleMs:.4f} ms")

    print(f"\n  내장 크기: {W.nbytes + xMean.nbytes + xStd.nbytes:,} bytes")
    print(f"  과적합 갭: {overfit:.1f}%")

    print("\n  가설 검증:")
    h1 = "채택" if overfit < 30 else "기각"
    print(f"    가설 1 (일반화 유지): {h1} (과적합 갭 {overfit:.1f}%)")

    ridgeVsRuleImprove = (top3 - ruleTop3 / n) * 100
    h2 = "채택" if ridgeVsRuleImprove >= 10 else ("부분 채택" if ridgeVsRuleImprove >= 5 else "기각")
    print(f"    가설 2 (top-3 10%p+ 개선): {h2} ({ridgeVsRuleImprove:+.1f}%p)")

    h3 = "채택" if inferenceMs < 1.0 else "기각"
    print(f"    가설 3 (추론 < 1ms): {h3} ({inferenceMs:.4f} ms)")

    print(f"\n  총 실행 시간: {elapsed:.1f}s")

    coeffPath = projectRoot / "src" / "vectrix" / "experiments" / "metaSelection" / "ridgeCoeffs.json"
    with open(coeffPath, "w", encoding="utf-8") as f:
        json.dump(coeffData, f, indent=2)
    print(f"\n  계수 저장: {coeffPath.name}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    runExperiment()
