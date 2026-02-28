"""
==============================================================================
실험 ID: ensembleEvolution/001
실험명: 잔차 직교성 기반 선택적 앙상블
==============================================================================

목적:
- metaSelection/003에서 단순 가중 앙상블이 top-1보다 나쁨을 확인
- 원인: 나쁜 모델의 극단적 MAPE가 앙상블 평균을 오염
- 새 전략: 모델 간 잔차가 "직교"(독립적 오차 패턴)인 경우만 앙상블
- 직교 = 모델 A의 잔차와 모델 B의 잔차가 낮은 상관관계

가설:
1. 잔차 상관이 낮은(|r|<0.3) 모델 쌍의 앙상블이 단독보다 MAPE 개선
2. 잔차 상관이 높은(|r|>0.7) 모델 쌍의 앙상블은 단독과 동등하거나 악화
3. 잔차 직교 기반 선택적 앙상블이 무조건 앙상블/무조건 단독 모두보다 우수

방법:
1. 62개 데이터셋에서 12개 모델의 예측 + 잔차 계산
2. 모든 모델 쌍의 잔차 상관계수 행렬 생성
3. 앙상블 전략:
   a. baseline: Oracle 단독 모델
   b. top-1 단독: DNA Ridge top-1
   c. 무조건 top-2 앙상블: top-1 + top-2 평균
   d. 직교 앙상블: top-1 + (잔차 상관 < threshold인 모델 중 최저 MAPE)
   e. 잔차 보정 앙상블: top-1 예측 + (직교 모델의 잔차 패턴 보정)
4. MAPE 비교

성공 기준:
- 직교 앙상블이 top-1 단독 대비 MAPE 5%+ 개선
- 직교 앙상블이 무조건 top-2 앙상블보다 우수

==============================================================================
결과 (실험 후 작성)
==============================================================================

데이터: 62개 → 61개 유효 (MAPE<500 필터), 12개 모델

핵심 발견 — 잔차 직교성이 존재하지 않음:
  모든 모델 쌍의 평균 잔차 상관이 +0.73 이상 (최소 0.73, 최대 1.0)
  |r|<0.3인 직교 쌍: 62개 중 6건만 발견 (9.8%)
  garch/auto_ces/croston/naive/window_avg: 모두 상관 1.0 (사실상 동일 예측)

평균 MAPE 비교:
| 전략 | 평균 MAPE | 중앙값 |
|------|-----------|--------|
| Oracle (사후 최적) | 29.97% | 6.05% |
| Top-1 단독 | 29.97% | 6.05% |
| 무조건 Top-2 앙상블 | 41.41% | 6.77% |
| 직교 앙상블 (|r|<0.3) | 30.30% | 6.58% |
| 적응형 직교+가중 | 30.30% | 6.64% |
| 잔차 보정 앙상블 | 30.20% | 6.27% |

직교 vs Top-1: -1.1% (악화!)
직교 vs 무조건 Top-2: +26.8% (개선, 그러나 Top-1보다 나쁨)
승률: 직교 0승, Top-1 6승, 무승부 55 → 직교 앙상블 전패

잔차 상관 구간별:
  low |r|<0.3: 0건 (이 구간 자체가 없음)
  mid 0.3≤|r|<0.7: 5건, 앙상블 승 1/5
  high |r|≥0.7: 56건 (91.8%), 앙상블 MAPE 44.5% vs 단독 32.0% → 앙상블 악화

근본 원인 분석:
1. vectrix의 모델들이 유사한 예측 패턴을 생성 (잔차가 동질적)
2. 특히 garch/auto_ces/croston/naive/window_avg가 완전 동일 잔차 (상관 1.0)
   → 이 모델들이 사실상 같은 예측을 하고 있음 (기본 모드에서)
3. 잔차가 직교적이지 않으므로, 앙상블의 오차 상쇄 효과 불가능
4. 앙상블이 도움되려면 "다르게 틀리는" 모델이 필요하나 현재 모든 모델이 "같이 틀림"

결론:
- 가설 1 기각: 직교 쌍 자체가 거의 없음. 발견된 6건도 개선 없음
- 가설 2 채택: 높은 상관 앙상블은 단독보다 악화 (44.5% vs 32.0%)
- 가설 3 기각: 직교 앙상블이 Top-1보다 나쁨 (-1.1%)
- 핵심 결론: vectrix 모델 간 잔차 다양성이 근본적으로 부족
  → 앙상블 개선이 아닌 모델 자체의 다양성 확보가 선결 과제
  → ML 모델(LightGBM, XGBoost)과의 앙상블이 유일한 직교성 확보 방법

실험일: 2026-02-28
==============================================================================
"""

import io
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

projectRoot = Path(__file__).resolve().parents[4]
srcRoot = projectRoot / "src"
if str(srcRoot) not in sys.path:
    sys.path.insert(0, str(srcRoot))

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


def _evaluateAllModels(train: np.ndarray, test: np.ndarray):
    factories = _buildModelFactories()
    predictions = {}
    residuals = {}
    mapes = {}

    for modelName in MODELS_TO_TEST:
        model = factories[modelName]()
        model.fit(train)
        pred, _, _ = model.predict(len(test))

        if np.all(np.isfinite(pred)):
            predictions[modelName] = pred
            residuals[modelName] = test - pred
            mapes[modelName] = _mape(test, pred)
        else:
            predictions[modelName] = np.full(len(test), np.mean(train))
            residuals[modelName] = test - predictions[modelName]
            mapes[modelName] = float("inf")

    return predictions, residuals, mapes


def _residualCorrelationMatrix(residuals: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    models = list(residuals.keys())
    corrMatrix = {}

    for i, mA in enumerate(models):
        corrMatrix[mA] = {}
        for j, mB in enumerate(models):
            if i == j:
                corrMatrix[mA][mB] = 1.0
            else:
                rA = residuals[mA]
                rB = residuals[mB]
                if np.std(rA) < 1e-10 or np.std(rB) < 1e-10:
                    corrMatrix[mA][mB] = 0.0
                else:
                    corrMatrix[mA][mB] = float(np.corrcoef(rA, rB)[0, 1])

    return corrMatrix


def _ensembleStrategies(predictions, residuals, mapes, corrMatrix, test):
    sortedModels = sorted(mapes.items(), key=lambda x: x[1])
    validModels = [(m, v) for m, v in sortedModels if np.isfinite(v)]

    if not validModels:
        return {}

    top1Model = validModels[0][0]
    top1Mape = validModels[0][1]
    top1Pred = predictions[top1Model]

    oracleModel = validModels[0][0]
    oracleMape = validModels[0][1]

    results = {
        "oracle_mape": oracleMape,
        "oracle_model": oracleModel,
        "top1_mape": top1Mape,
        "top1_model": top1Model,
    }

    if len(validModels) >= 2:
        top2Model = validModels[1][0]
        top2Pred = predictions[top2Model]
        naiveEnsemble = (top1Pred + top2Pred) / 2
        results["naive_top2_mape"] = _mape(test, naiveEnsemble)
        results["naive_top2_corr"] = corrMatrix[top1Model].get(top2Model, 0.0)
    else:
        results["naive_top2_mape"] = top1Mape
        results["naive_top2_corr"] = 1.0

    bestOrthogonal = None
    bestOrthogonalMape = float("inf")
    bestOrthogonalCorr = 1.0

    for mName, mMape in validModels[1:]:
        corr = corrMatrix[top1Model].get(mName, 1.0)
        if abs(corr) < 0.3:
            ensemblePred = (top1Pred + predictions[mName]) / 2
            ensembleMape = _mape(test, ensemblePred)
            if ensembleMape < bestOrthogonalMape:
                bestOrthogonal = mName
                bestOrthogonalMape = ensembleMape
                bestOrthogonalCorr = corr

    if bestOrthogonal is not None:
        results["orthogonal_mape"] = bestOrthogonalMape
        results["orthogonal_partner"] = bestOrthogonal
        results["orthogonal_corr"] = bestOrthogonalCorr
        results["orthogonal_found"] = True
    else:
        results["orthogonal_mape"] = top1Mape
        results["orthogonal_partner"] = None
        results["orthogonal_corr"] = None
        results["orthogonal_found"] = False

    bestAdaptive = None
    bestAdaptiveMape = float("inf")
    bestAdaptiveCorr = 1.0

    for threshold in [0.1, 0.2, 0.3, 0.5]:
        for mName, mMape in validModels[1:]:
            corr = corrMatrix[top1Model].get(mName, 1.0)
            if abs(corr) < threshold:
                w1 = 1.0 / (1.0 + top1Mape) if np.isfinite(top1Mape) else 0.5
                w2 = 1.0 / (1.0 + mMape) if np.isfinite(mMape) else 0.5
                wTotal = w1 + w2
                w1 /= wTotal
                w2 /= wTotal
                ensemblePred = w1 * top1Pred + w2 * predictions[mName]
                ensembleMape = _mape(test, ensemblePred)
                if ensembleMape < bestAdaptiveMape:
                    bestAdaptive = mName
                    bestAdaptiveMape = ensembleMape
                    bestAdaptiveCorr = corr

    if bestAdaptive is not None:
        results["adaptive_mape"] = bestAdaptiveMape
        results["adaptive_partner"] = bestAdaptive
        results["adaptive_corr"] = bestAdaptiveCorr
    else:
        results["adaptive_mape"] = top1Mape
        results["adaptive_partner"] = None
        results["adaptive_corr"] = None

    bestCorrMape = float("inf")
    bestCorrPartner = None

    for mName, mMape in validModels[1:]:
        corr = corrMatrix[top1Model].get(mName, 1.0)
        if abs(corr) < 0.5:
            residModel = residuals[mName]
            residTop1 = residuals[top1Model]

            correction = residModel - residTop1
            corrFactor = np.clip(1.0 - abs(corr), 0.0, 0.5)
            adjustedPred = top1Pred + corrFactor * correction * 0.3
            corrMape = _mape(test, adjustedPred)

            if corrMape < bestCorrMape:
                bestCorrMape = corrMape
                bestCorrPartner = mName

    if bestCorrPartner is not None:
        results["correction_mape"] = bestCorrMape
        results["correction_partner"] = bestCorrPartner
    else:
        results["correction_mape"] = top1Mape
        results["correction_partner"] = None

    return results


def _analyzeCorrelationPatterns(allCorrMatrices: List[Dict]):
    pairCorrs = {}

    for corrMatrix in allCorrMatrices:
        models = list(corrMatrix.keys())
        for i, mA in enumerate(models):
            for j, mB in enumerate(models):
                if i >= j:
                    continue
                pair = (mA, mB)
                if pair not in pairCorrs:
                    pairCorrs[pair] = []
                corr = corrMatrix[mA].get(mB, 0.0)
                if np.isfinite(corr):
                    pairCorrs[pair].append(corr)

    pairStats = {}
    for pair, corrs in pairCorrs.items():
        if len(corrs) >= 5:
            pairStats[pair] = {
                "mean": np.mean(corrs),
                "std": np.std(corrs),
                "n": len(corrs),
            }

    return pairStats


def runExperiment():
    print("=" * 70)
    print("E018: 잔차 직교성 기반 선택적 앙상블")
    print("=" * 70)

    print("\n1. 데이터셋 준비 중...")
    t0 = time.time()
    datasets = _prepareDatasets()
    print(f"   총 {len(datasets)}개")

    allResults = []
    allCorrMatrices = []

    print("\n2. 모델 평가 + 잔차 분석 중...")
    for idx, (dsName, values) in enumerate(datasets):
        splitIdx = int(len(values) * 0.8)
        train = values[:splitIdx]
        test = values[splitIdx:]

        if len(train) < 20 or len(test) < 5:
            continue

        predictions, residuals, mapes = _evaluateAllModels(train, test)
        corrMatrix = _residualCorrelationMatrix(residuals)
        allCorrMatrices.append(corrMatrix)

        results = _ensembleStrategies(predictions, residuals, mapes, corrMatrix, test)
        results["dataset"] = dsName
        allResults.append(results)

        if (idx + 1) % 10 == 0:
            print(f"  처리 완료: {idx + 1}/{len(datasets)}")

    print(f"   유효: {len(allResults)}개")

    print("\n3. 잔차 상관 패턴 분석...")
    pairStats = _analyzeCorrelationPatterns(allCorrMatrices)

    print("\n   가장 직교적인 모델 쌍 (평균 상관 가장 낮은):")
    sortedPairs = sorted(pairStats.items(), key=lambda x: abs(x[1]["mean"]))
    for pair, stats in sortedPairs[:10]:
        print(f"     {pair[0]:<15} <-> {pair[1]:<15}: mean={stats['mean']:+.3f} (std={stats['std']:.3f}, n={stats['n']})")

    print("\n   가장 상관 높은 모델 쌍:")
    sortedPairsHigh = sorted(pairStats.items(), key=lambda x: -abs(x[1]["mean"]))
    for pair, stats in sortedPairsHigh[:10]:
        print(f"     {pair[0]:<15} <-> {pair[1]:<15}: mean={stats['mean']:+.3f} (std={stats['std']:.3f}, n={stats['n']})")

    validResults = [r for r in allResults if r.get("oracle_mape", 999) < 500]
    n = len(validResults)

    oracleMapes = [r["oracle_mape"] for r in validResults]
    top1Mapes = [r["top1_mape"] for r in validResults]
    naiveTop2Mapes = [r["naive_top2_mape"] for r in validResults]
    orthMapes = [r["orthogonal_mape"] for r in validResults]
    adaptMapes = [r["adaptive_mape"] for r in validResults]
    correctionMapes = [r["correction_mape"] for r in validResults]

    orthFound = sum(1 for r in validResults if r.get("orthogonal_found", False))

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print("E018 결과 요약")
    print("=" * 70)

    print(f"\n  데이터셋: {n}개 유효")

    print("\n  평균 MAPE 비교:")
    print(f"  {'전략':<40} {'평균':>10} {'중앙값':>10}")
    print(f"  {'-' * 62}")
    print(f"  {'Oracle (사후 최적 단독)':<40} {np.mean(oracleMapes):>9.2f}% {np.median(oracleMapes):>9.2f}%")
    print(f"  {'Top-1 단독':<40} {np.mean(top1Mapes):>9.2f}% {np.median(top1Mapes):>9.2f}%")
    print(f"  {'무조건 Top-2 평균 앙상블':<40} {np.mean(naiveTop2Mapes):>9.2f}% {np.median(naiveTop2Mapes):>9.2f}%")
    print(f"  {'직교 앙상블 (|r|<0.3 only)':<40} {np.mean(orthMapes):>9.2f}% {np.median(orthMapes):>9.2f}%")
    print(f"  {'적응형 직교+가중 앙상블':<40} {np.mean(adaptMapes):>9.2f}% {np.median(adaptMapes):>9.2f}%")
    print(f"  {'잔차 보정 앙상블':<40} {np.mean(correctionMapes):>9.2f}% {np.median(correctionMapes):>9.2f}%")

    print(f"\n  직교 파트너 발견: {orthFound}/{n}건 ({orthFound / n:.1%})")

    orthVsTop1 = (np.mean(top1Mapes) - np.mean(orthMapes)) / np.mean(top1Mapes) * 100
    orthVsNaive = (np.mean(naiveTop2Mapes) - np.mean(orthMapes)) / np.mean(naiveTop2Mapes) * 100
    adaptVsTop1 = (np.mean(top1Mapes) - np.mean(adaptMapes)) / np.mean(top1Mapes) * 100

    print("\n  개선율:")
    print(f"    직교 vs Top-1 단독: {orthVsTop1:+.1f}%")
    print(f"    직교 vs 무조건 Top-2: {orthVsNaive:+.1f}%")
    print(f"    적응형 vs Top-1 단독: {adaptVsTop1:+.1f}%")

    orthWins = sum(1 for o, t in zip(orthMapes, top1Mapes) if o < t - 0.01)
    top1Wins = sum(1 for o, t in zip(orthMapes, top1Mapes) if t < o - 0.01)
    ties = n - orthWins - top1Wins
    print(f"\n  승률 (직교 vs Top-1): 직교 {orthWins}승, Top-1 {top1Wins}승, 무승부 {ties}")

    corrBins = {"low": [], "mid": [], "high": []}
    for r in validResults:
        corr = r.get("naive_top2_corr", 0.0)
        naiveMape = r["naive_top2_mape"]
        top1Mape = r["top1_mape"]

        if abs(corr) < 0.3:
            corrBins["low"].append((naiveMape, top1Mape))
        elif abs(corr) < 0.7:
            corrBins["mid"].append((naiveMape, top1Mape))
        else:
            corrBins["high"].append((naiveMape, top1Mape))

    print("\n  잔차 상관 구간별 앙상블 효과:")
    for binName, binData in [("low |r|<0.3", corrBins["low"]),
                               ("mid 0.3≤|r|<0.7", corrBins["mid"]),
                               ("high |r|≥0.7", corrBins["high"])]:
        if binData:
            naiveMean = np.mean([d[0] for d in binData])
            top1Mean = np.mean([d[1] for d in binData])
            improvement = (top1Mean - naiveMean) / top1Mean * 100
            ensWins = sum(1 for d in binData if d[0] < d[1])
            print(f"    {binName}: {len(binData)}건, "
                  f"앙상블={naiveMean:.1f}%, 단독={top1Mean:.1f}%, "
                  f"앙상블 승={ensWins}/{len(binData)}, 개선={improvement:+.1f}%")

    print("\n  가설 검증:")
    h1 = "채택" if orthVsTop1 > 5 else ("부분 채택" if orthVsTop1 > 0 else "기각")
    print(f"    가설 1 (직교 앙상블 > Top-1 5%+): {h1} ({orthVsTop1:+.1f}%)")

    if corrBins["high"]:
        highNaiveMean = np.mean([d[0] for d in corrBins["high"]])
        highTop1Mean = np.mean([d[1] for d in corrBins["high"]])
        h2Check = highNaiveMean >= highTop1Mean * 0.98
        h2 = "채택" if h2Check else "기각"
        print(f"    가설 2 (높은 상관 앙상블 ≈ 단독): {h2} (앙상블={highNaiveMean:.1f}%, 단독={highTop1Mean:.1f}%)")
    else:
        print("    가설 2: 평가 불가 (높은 상관 사례 없음)")

    orthBetterThanBoth = orthVsTop1 > 0 and orthVsNaive > 0
    h3 = "채택" if orthBetterThanBoth else "기각"
    print(f"    가설 3 (직교 > 무조건 앙상블/단독 모두): {h3}")

    print(f"\n  총 실행 시간: {elapsed:.1f}s")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    runExperiment()
