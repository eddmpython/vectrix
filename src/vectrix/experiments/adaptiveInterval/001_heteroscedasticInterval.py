"""
==============================================================================
실험 ID: adaptiveInterval/001
실험명: 잔차 이질 분산 검출 + 적응형 예측 구간
==============================================================================

목적:
- 현재 vectrix는 모든 모델에서 고정 1.96*sigma 구간 사용
- 실제 시계열은 변동성이 시간에 따라 변함 (이질적 분산)
- 잔차의 이질적 분산을 검출하고, 검출된 경우 적응형 구간으로 전환
- 적응형 구간의 coverage(실제 포함률)가 고정 구간보다 정확한지 검증

가설:
1. 합성 데이터의 30%+ 에서 잔차 이질적 분산이 검출됨
2. 적응형 구간의 실제 coverage가 목표 95%에 더 가까움 (|coverage - 0.95| 감소)
3. 적응형 구간의 평균 폭이 고정 구간보다 좁음 (불필요한 넓힘 방지)

방법:
1. 62개 데이터셋에서 각 모델별 잔차 계산
2. 잔차 이질적 분산 검출:
   a. White 검정 (잔차^2 vs 시간 회귀)
   b. ARCH 효과 검정 (잔차^2의 자기상관)
   c. 롤링 표준편차 변동 계수
3. 구간 전략 비교:
   a. 고정 구간: 1.96 * global_sigma * sqrt(h)
   b. 적응형 구간: 1.96 * local_sigma(t) * sqrt(h)
      - local_sigma: 최근 잔차의 EWMA 표준편차
   c. GARCH 구간: 조건부 분산 기반
4. 평가 지표:
   - Coverage: 실제 값이 구간 안에 들어가는 비율
   - Width: 평균 구간 폭
   - Winkler Score: coverage + width 종합

성공 기준:
- 적응형 구간의 coverage가 0.95에 5%p 이내
- 적응형 구간의 Winkler Score가 고정 구간 대비 5%+ 개선

==============================================================================
결과 (실험 후 작성)
==============================================================================

데이터: 62개 데이터셋, 6개 모델, 총 372건 테스트

이질적 분산 검출:
  전체: 325/372건 (87.4%) — 예상보다 훨씬 높음
  White 검정: 226건 (60.8%)
  ARCH 효과: 319건 (85.8%)
  롤링 CV > 0.3: 130건 (34.9%)
  데이터 유형별: volatile(27%), intermittentDemand(17%) 외 모두 100%

구간 전략 비교:
| 전략 | Coverage | Coverage Gap | Width | Winkler |
|------|----------|-------------|-------|---------|
| 고정 1.96σ | 0.964 | 0.079 | 2592 | 2797 |
| 적응형 EWMA | 0.993 | 0.050 | 3415 | 3499 |
| GARCH 기반 | 0.885 | 0.113 | 522 | 1487 |

Winkler 최저 승률:
  GARCH: 326승 (87.6%) — 압도적 1위
  적응형: 38승 (10.2%)
  고정: 8승 (2.2%)

이질 분산 검출 시:
  GARCH: 279승 (85.8%)
  적응형: 38승 (11.7%)
  고정: 8승 (2.5%)

핵심 발견:
1. 이질적 분산은 거의 모든 시계열에서 존재 (87.4%) — 고정 구간 가정 위반
2. GARCH 기반 구간이 Winkler Score에서 압도적 우위 (고정 대비 +46.9% 개선)
3. GARCH의 비결: 좁은 구간(522 vs 2592)으로 coverage 소폭 하락(0.885) 감수
   → Winkler Score 기준으로는 좁은 구간 + 적절한 coverage가 최적
4. 적응형 EWMA: coverage는 좋으나(0.993) 구간이 너무 넓어(3415) Winkler 악화
5. 고정 구간의 문제: 이질 분산이 있는 데이터에서 "평균적으로 맞추기" 때문에
   변동 구간에서는 너무 좁고, 안정 구간에서는 너무 넓음

결론:
- 가설 1 채택: 87.4% 이질 분산 검출 > 30% 목표 (대폭 초과)
- 가설 2 채택: 적응형 coverage gap 0.050 < 고정 0.079 (정확도 향상)
- 가설 3 기각: 적응형 구간이 고정보다 넓음 (3415 > 2592)
- GARCH 기반 구간을 비-GARCH 모델에도 적용하는 것이 최적 전략
- 다음 단계: GARCH 조건부 분산을 일반 모델(ETS, ARIMA 등)의 잔차에 적용하는
  "후처리 GARCH 구간" 모듈 검증

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

MODELS_TO_TEST = ["auto_ets", "theta", "auto_arima", "mstl", "auto_ces", "dot"]


def _buildModelFactories() -> Dict[str, callable]:
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.ets import AutoETS
    from vectrix.engine.mstl import AutoMSTL
    from vectrix.engine.theta import OptimizedTheta

    return {
        "auto_ets": lambda: AutoETS(),
        "theta": lambda: OptimizedTheta(),
        "auto_arima": lambda: AutoARIMA(),
        "mstl": lambda: AutoMSTL(),
        "auto_ces": lambda: AutoCES(),
        "dot": lambda: DynamicOptimizedTheta(),
    }


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


def _testHeteroscedasticity(residuals: np.ndarray) -> Dict[str, float]:
    n = len(residuals)
    if n < 20:
        return {"whiteP": 1.0, "archP": 1.0, "rollingCV": 0.0, "detected": False}

    r2 = residuals ** 2

    t = np.arange(n, dtype=np.float64)
    tMean = np.mean(t)
    r2Mean = np.mean(r2)
    cov = np.sum((t - tMean) * (r2 - r2Mean))
    varT = np.sum((t - tMean) ** 2)
    if varT < 1e-10:
        whiteP = 1.0
    else:
        slope = cov / varT
        intercept = r2Mean - slope * tMean
        r2Hat = intercept + slope * t
        ssRes = np.sum((r2 - r2Hat) ** 2)
        ssTot = np.sum((r2 - r2Mean) ** 2)
        if ssTot < 1e-10:
            whiteP = 1.0
        else:
            r2Stat = 1.0 - ssRes / ssTot
            fStat = r2Stat * (n - 2) / (1 - r2Stat + 1e-10)
            from scipy.stats import f as fdist
            whiteP = 1.0 - fdist.cdf(fStat, 1, n - 2)

    lags = min(5, n // 5)
    r2Centered = r2 - np.mean(r2)
    denom = np.sum(r2Centered ** 2)
    if denom < 1e-10:
        archP = 1.0
    else:
        acfVals = []
        for lag in range(1, lags + 1):
            acfVal = np.sum(r2Centered[lag:] * r2Centered[:-lag]) / denom
            acfVals.append(acfVal)
        ljungBox = n * (n + 2) * sum(a ** 2 / (n - k - 1) for k, a in enumerate(acfVals))
        from scipy.stats import chi2
        archP = 1.0 - chi2.cdf(ljungBox, lags)

    windowSize = max(20, n // 5)
    rollingStds = []
    for i in range(0, n - windowSize + 1, windowSize // 2):
        windowResid = residuals[i:i + windowSize]
        rollingStds.append(np.std(windowResid))
    rollingStds = np.array(rollingStds)
    if np.mean(rollingStds) > 1e-10:
        rollingCV = np.std(rollingStds) / np.mean(rollingStds)
    else:
        rollingCV = 0.0

    detected = whiteP < 0.05 or archP < 0.05 or rollingCV > 0.3

    return {
        "whiteP": whiteP,
        "archP": archP,
        "rollingCV": rollingCV,
        "detected": detected,
    }


def _fixedInterval(residuals: np.ndarray, steps: int, alpha: float = 0.05):
    z = 1.96
    sigma = np.std(residuals)
    horizonFactor = np.sqrt(np.arange(1, steps + 1))
    margin = z * sigma * horizonFactor
    return margin


def _adaptiveInterval(residuals: np.ndarray, steps: int, alpha: float = 0.05,
                       ewmaSpan: int = 30):
    z = 1.96
    n = len(residuals)
    r2 = residuals ** 2

    decayFactor = 2.0 / (ewmaSpan + 1)
    ewmaVar = np.zeros(n)
    ewmaVar[0] = r2[0]
    for i in range(1, n):
        ewmaVar[i] = decayFactor * r2[i] + (1 - decayFactor) * ewmaVar[i - 1]

    localSigma = np.sqrt(ewmaVar[-1])
    if localSigma < 1e-10:
        localSigma = np.std(residuals)

    horizonFactor = np.sqrt(np.arange(1, steps + 1))

    trendR2 = np.polyfit(np.arange(n), r2, 1)
    varSlope = trendR2[0]

    margin = np.zeros(steps)
    for h in range(steps):
        projectedVar = ewmaVar[-1] + varSlope * (h + 1)
        projectedVar = max(projectedVar, ewmaVar[-1] * 0.5)
        projectedSigma = np.sqrt(projectedVar)
        margin[h] = z * projectedSigma * horizonFactor[h]

    return margin


def _garchInterval(residuals: np.ndarray, steps: int, alpha: float = 0.05):
    z = 1.96
    n = len(residuals)
    r2 = residuals ** 2

    omega = np.var(residuals) * 0.05
    alphaG = 0.1
    betaG = 0.85
    unconditionalVar = omega / (1.0 - alphaG - betaG) if (alphaG + betaG) < 1.0 else np.var(residuals)

    sigma2 = np.zeros(n)
    sigma2[0] = unconditionalVar
    for t in range(1, n):
        sigma2[t] = omega + alphaG * r2[t - 1] + betaG * sigma2[t - 1]

    lastSigma2 = sigma2[-1]
    lastR2 = r2[-1]

    margin = np.zeros(steps)
    forecSigma2 = lastSigma2
    for h in range(steps):
        forecSigma2 = omega + (alphaG + betaG) * forecSigma2
        margin[h] = z * np.sqrt(forecSigma2)

    return margin


def _evaluateCoverage(actual: np.ndarray, predictions: np.ndarray,
                       margin: np.ndarray) -> Dict[str, float]:
    lower = predictions - margin
    upper = predictions + margin

    inInterval = (actual >= lower) & (actual <= upper)
    coverage = np.mean(inInterval)
    width = np.mean(upper - lower)

    targetCoverage = 0.95
    winklerPenalty = 0.0
    for i in range(len(actual)):
        intervalWidth = upper[i] - lower[i]
        if actual[i] < lower[i]:
            winklerPenalty += 2.0 * (lower[i] - actual[i]) / (1 - targetCoverage)
        elif actual[i] > upper[i]:
            winklerPenalty += 2.0 * (actual[i] - upper[i]) / (1 - targetCoverage)
        winklerPenalty += intervalWidth

    winklerScore = winklerPenalty / len(actual)

    return {
        "coverage": coverage,
        "width": width,
        "winklerScore": winklerScore,
        "coverageGap": abs(coverage - targetCoverage),
    }


def runExperiment():
    print("=" * 70)
    print("E025: 잔차 이질 분산 검출 + 적응형 예측 구간")
    print("=" * 70)

    print("\n1. 데이터셋 준비 중...")
    t0 = time.time()
    datasets = _prepareDatasets()
    print(f"   총 {len(datasets)}개")

    factories = _buildModelFactories()

    heteroResults = []
    intervalResults = {"fixed": [], "adaptive": [], "garch": []}
    detectedCount = 0

    print("\n2. 모델별 잔차 분석 + 구간 비교 중...")
    for idx, (dsName, values) in enumerate(datasets):
        splitIdx = int(len(values) * 0.8)
        train = values[:splitIdx]
        test = values[splitIdx:]

        if len(train) < 30 or len(test) < 5:
            continue

        for modelName in MODELS_TO_TEST:
            model = factories[modelName]()
            model.fit(train)
            pred, _, _ = model.predict(len(test))

            if not np.all(np.isfinite(pred)):
                continue

            trainResiduals = np.zeros(len(train))
            trainPred, _, _ = model.predict(len(train))
            if len(trainPred) >= len(train) and np.all(np.isfinite(trainPred)):
                trainResiduals = train - trainPred[:len(train)]
            else:
                model2 = factories[modelName]()
                nFit = int(len(train) * 0.7)
                model2.fit(train[:nFit])
                holdPred, _, _ = model2.predict(len(train) - nFit)
                if np.all(np.isfinite(holdPred)):
                    trainResiduals = np.zeros(len(train))
                    trainResiduals[nFit:] = train[nFit:] - holdPred[:len(train) - nFit]
                else:
                    continue

            if np.std(trainResiduals) < 1e-10:
                continue

            heteroTest = _testHeteroscedasticity(trainResiduals)
            heteroResults.append({
                "dataset": dsName,
                "model": modelName,
                **heteroTest,
            })
            if heteroTest["detected"]:
                detectedCount += 1

            steps = len(test)

            fixedMargin = _fixedInterval(trainResiduals, steps)
            fixedEval = _evaluateCoverage(test, pred, fixedMargin)
            intervalResults["fixed"].append({
                "dataset": dsName, "model": modelName,
                "hetero": heteroTest["detected"],
                **fixedEval,
            })

            adaptiveMargin = _adaptiveInterval(trainResiduals, steps)
            adaptiveEval = _evaluateCoverage(test, pred, adaptiveMargin)
            intervalResults["adaptive"].append({
                "dataset": dsName, "model": modelName,
                "hetero": heteroTest["detected"],
                **adaptiveEval,
            })

            garchMargin = _garchInterval(trainResiduals, steps)
            garchEval = _evaluateCoverage(test, pred, garchMargin)
            intervalResults["garch"].append({
                "dataset": dsName, "model": modelName,
                "hetero": heteroTest["detected"],
                **garchEval,
            })

        if (idx + 1) % 10 == 0:
            print(f"  처리 완료: {idx + 1}/{len(datasets)}")

    totalTests = len(heteroResults)
    elapsed = time.time() - t0

    print(f"   총 테스트: {totalTests}건")
    print(f"   이질적 분산 검출: {detectedCount}건 ({detectedCount / totalTests:.1%})")

    print("\n3. 이질적 분산 검출 상세...")
    whiteDetected = sum(1 for r in heteroResults if r["whiteP"] < 0.05)
    archDetected = sum(1 for r in heteroResults if r["archP"] < 0.05)
    cvDetected = sum(1 for r in heteroResults if r["rollingCV"] > 0.3)
    print(f"   White 검정 (p<0.05): {whiteDetected}건 ({whiteDetected / totalTests:.1%})")
    print(f"   ARCH 효과 (p<0.05): {archDetected}건 ({archDetected / totalTests:.1%})")
    print(f"   롤링 CV > 0.3: {cvDetected}건 ({cvDetected / totalTests:.1%})")

    print("\n4. 데이터 유형별 이질적 분산 비율...")
    synthHetero = [r for r in heteroResults if r["dataset"].startswith("synth_")]
    sampleHetero = [r for r in heteroResults if r["dataset"].startswith("sample_")]

    dsTypes = {}
    for r in heteroResults:
        parts = r["dataset"].split("_")
        if parts[0] == "synth":
            dsType = parts[1]
        else:
            dsType = "_".join(parts[1:])
        if dsType not in dsTypes:
            dsTypes[dsType] = {"total": 0, "detected": 0}
        dsTypes[dsType]["total"] += 1
        if r["detected"]:
            dsTypes[dsType]["detected"] += 1

    for dsType, counts in sorted(dsTypes.items(), key=lambda x: -x[1]["detected"] / max(x[1]["total"], 1)):
        rate = counts["detected"] / counts["total"] if counts["total"] > 0 else 0
        print(f"   {dsType:<25}: {counts['detected']}/{counts['total']} ({rate:.0%})")

    print("\n" + "=" * 70)
    print("E025 결과 요약")
    print("=" * 70)

    for stratName in ["fixed", "adaptive", "garch"]:
        results = intervalResults[stratName]
        allCov = [r["coverage"] for r in results]
        allWidth = [r["width"] for r in results]
        allWinkler = [r["winklerScore"] for r in results]
        allGap = [r["coverageGap"] for r in results]

        heteroCov = [r["coverage"] for r in results if r["hetero"]]
        nonHeteroCov = [r["coverage"] for r in results if not r["hetero"]]
        heteroWinkler = [r["winklerScore"] for r in results if r["hetero"]]
        nonHeteroWinkler = [r["winklerScore"] for r in results if not r["hetero"]]

        print(f"\n  [{stratName.upper()}]")
        print(f"    전체: coverage={np.mean(allCov):.3f}, width={np.mean(allWidth):.1f}, "
              f"winkler={np.mean(allWinkler):.1f}, coverageGap={np.mean(allGap):.3f}")
        if heteroCov:
            print(f"    이질분산O: coverage={np.mean(heteroCov):.3f}, winkler={np.mean(heteroWinkler):.1f} ({len(heteroCov)}건)")
        if nonHeteroCov:
            print(f"    이질분산X: coverage={np.mean(nonHeteroCov):.3f}, winkler={np.mean(nonHeteroWinkler):.1f} ({len(nonHeteroCov)}건)")

    print("\n  전략 간 비교:")
    fixedCovGap = np.mean([r["coverageGap"] for r in intervalResults["fixed"]])
    adaptCovGap = np.mean([r["coverageGap"] for r in intervalResults["adaptive"]])
    garchCovGap = np.mean([r["coverageGap"] for r in intervalResults["garch"]])

    fixedWinkler = np.mean([r["winklerScore"] for r in intervalResults["fixed"]])
    adaptWinkler = np.mean([r["winklerScore"] for r in intervalResults["adaptive"]])
    garchWinkler = np.mean([r["winklerScore"] for r in intervalResults["garch"]])

    fixedWidth = np.mean([r["width"] for r in intervalResults["fixed"]])
    adaptWidth = np.mean([r["width"] for r in intervalResults["adaptive"]])
    garchWidth = np.mean([r["width"] for r in intervalResults["garch"]])

    print(f"  {'전략':<15} {'Coverage Gap':>14} {'Winkler':>10} {'Width':>10}")
    print(f"  {'-' * 52}")
    print(f"  {'고정 구간':<15} {fixedCovGap:>13.3f} {fixedWinkler:>9.1f} {fixedWidth:>9.1f}")
    print(f"  {'적응형 EWMA':<15} {adaptCovGap:>13.3f} {adaptWinkler:>9.1f} {adaptWidth:>9.1f}")
    print(f"  {'GARCH 기반':<15} {garchCovGap:>13.3f} {garchWinkler:>9.1f} {garchWidth:>9.1f}")

    adaptWinklerImprove = (fixedWinkler - adaptWinkler) / fixedWinkler * 100
    garchWinklerImprove = (fixedWinkler - garchWinkler) / fixedWinkler * 100

    print("\n  Winkler 개선:")
    print(f"    적응형 vs 고정: {adaptWinklerImprove:+.1f}%")
    print(f"    GARCH vs 고정: {garchWinklerImprove:+.1f}%")

    fixedWins = 0
    adaptWins = 0
    garchWins = 0
    for i in range(len(intervalResults["fixed"])):
        scores = [
            ("fixed", intervalResults["fixed"][i]["winklerScore"]),
            ("adaptive", intervalResults["adaptive"][i]["winklerScore"]),
            ("garch", intervalResults["garch"][i]["winklerScore"]),
        ]
        bestStrat = min(scores, key=lambda x: x[1])[0]
        if bestStrat == "fixed":
            fixedWins += 1
        elif bestStrat == "adaptive":
            adaptWins += 1
        else:
            garchWins += 1

    totalComparisons = len(intervalResults["fixed"])
    print("\n  Winkler 최저 승률:")
    print(f"    고정: {fixedWins}승 ({fixedWins / totalComparisons:.1%})")
    print(f"    적응형: {adaptWins}승 ({adaptWins / totalComparisons:.1%})")
    print(f"    GARCH: {garchWins}승 ({garchWins / totalComparisons:.1%})")

    heteroFixedWins = 0
    heteroAdaptWins = 0
    heteroGarchWins = 0
    heteroCount = 0
    for i in range(len(intervalResults["fixed"])):
        if not intervalResults["fixed"][i]["hetero"]:
            continue
        heteroCount += 1
        scores = [
            ("fixed", intervalResults["fixed"][i]["winklerScore"]),
            ("adaptive", intervalResults["adaptive"][i]["winklerScore"]),
            ("garch", intervalResults["garch"][i]["winklerScore"]),
        ]
        bestStrat = min(scores, key=lambda x: x[1])[0]
        if bestStrat == "fixed":
            heteroFixedWins += 1
        elif bestStrat == "adaptive":
            heteroAdaptWins += 1
        else:
            heteroGarchWins += 1

    if heteroCount > 0:
        print(f"\n  이질적 분산 검출된 경우만 ({heteroCount}건):")
        print(f"    고정: {heteroFixedWins}승 ({heteroFixedWins / heteroCount:.1%})")
        print(f"    적응형: {heteroAdaptWins}승 ({heteroAdaptWins / heteroCount:.1%})")
        print(f"    GARCH: {heteroGarchWins}승 ({heteroGarchWins / heteroCount:.1%})")

    print("\n  가설 검증:")
    h1 = "채택" if detectedCount / totalTests > 0.3 else "기각"
    print(f"    가설 1 (30%+ 이질 분산 검출): {h1} ({detectedCount / totalTests:.1%})")

    h2 = "채택" if adaptCovGap < fixedCovGap else "기각"
    print(f"    가설 2 (적응형 coverage 더 정확): {h2} (적응형 gap={adaptCovGap:.3f} vs 고정 gap={fixedCovGap:.3f})")

    h3 = "채택" if adaptWidth < fixedWidth else "기각"
    print(f"    가설 3 (적응형 구간 더 좁음): {h3} (적응형={adaptWidth:.1f} vs 고정={fixedWidth:.1f})")

    print(f"\n  총 실행 시간: {elapsed:.1f}s")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    runExperiment()
