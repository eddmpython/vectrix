"""
==============================================================================
실험 ID: adaptiveInterval/002
실험명: Conformal vs Bootstrap vs GARCH 후처리 구간 비교
==============================================================================

목적:
- vectrix에 이미 구현된 ConformalInterval, BootstrapInterval과
  001에서 유효성이 확인된 GARCH 후처리 구간을 동일 조건에서 비교
- 어떤 구간 방법이 Winkler Score, Coverage, Width 측면에서 최적인지 결정
- 메인 파이프라인 통합 우선순위 결정

가설:
1. Conformal 구간이 Coverage 기준 가장 정확 (이론적 보장)
2. GARCH 후처리가 Winkler Score 기준 최적 (001 결과 확장)
3. Bootstrap이 다양한 데이터 유형에서 가장 안정적

방법:
1. 62개 데이터셋, 6개 핵심 모델
2. 각 모델의 학습 잔차로 구간 계산:
   a. 고정: 1.96 * global_sigma * sqrt(h)
   b. Conformal (Split): 기존 ConformalInterval 클래스 사용
   c. Conformal (Jackknife): Jackknife+ 방식
   d. Bootstrap: 기존 BootstrapInterval 클래스 사용
   e. GARCH 후처리: 001의 GARCH 기반 구간
3. 평가: Coverage, Width, Winkler Score, Coverage Gap

성공 기준:
- 최소 1개 고급 방법이 고정 대비 Winkler 5%+ 개선
- Coverage gap 최소 방법 식별

==============================================================================
결과 (실험 후 작성)
==============================================================================

데이터: 62개 데이터셋, 6개 모델, 총 372건

구간 방법별 성능:
| 방법 | Coverage | CovGap | Width | Winkler |
|------|----------|--------|-------|---------|
| 고정 1.96σ | 0.954 | 0.086 | 1858 | 2072 |
| Conformal Split | 1.000 | 0.050 | 3140 | 3155 |
| Conformal Jack | 1.000 | 0.050 | 14609 | 14609 |
| Bootstrap | 0.309 | 0.643 | 331 | 11436 |
| GARCH 후처리 | 0.809 | 0.166 | 469 | 1431 |

Winkler 최저 승률:
  GARCH: 272승 (73.1%) — 모든 모델에서 압도적 1위
  고정: 43승 (11.6%)
  Bootstrap: 43승 (11.6%)
  Conformal Split: 14승 (3.8%)
  Conformal Jack: 0승 (0.0%)

모델별 GARCH 승률: auto_ets 81%, mstl 79%, auto_ces 84%, auto_arima 65%, theta 71%, dot 60%

Coverage Gap 최저 승률:
  고정: 193승 (51.9%) — Coverage 정확도는 고정이 최고
  GARCH: 125승 (33.6%)

핵심 발견:
1. GARCH 후처리가 Winkler Score 압도적 1위 (고정 대비 +31% 개선)
2. Conformal Split: Coverage 완벽(1.0)이지만 구간이 너무 넓어(3140) 비실용적
3. Conformal Jackknife: 구간 폭 14609 — 극단적으로 넓어 사용 불가
4. Bootstrap: Coverage 0.309 — 심각하게 낮음. 구현에 문제 있을 가능성
5. 고정 구간이 의외로 Coverage 정확도(0.954) 최고 — 95% 목표에 가장 가까움
6. GARCH의 약점: Coverage 0.809 (under-coverage 15%) — 보정 필요

실질적 최적 전략:
- Winkler 기준: GARCH 후처리 (좁은 구간 + 적절한 적응)
- Coverage 기준: 고정 1.96σ (0.954)
- 하이브리드: GARCH 후처리 + coverage 보정 (z를 1.96에서 2.2~2.3으로 조정)

기존 Conformal/Bootstrap 구현의 문제점:
- Conformal: calibration 과정에서 구간 폭 과대 추정 (max residual 사용)
- Jackknife: fold 수 부족으로 극단값 편향
- Bootstrap: cumResiduals 감쇠(sqrt(1/h)) 방식이 과도 → under-coverage

결론:
- 가설 1 채택: Conformal Coverage 0.050 gap (최소)
- 가설 2 채택: GARCH가 Winkler 최적 (1431 vs 고정 2072)
- 가설 3 기각: Bootstrap이 가장 불안정 (std=30063 vs 고정 6267)
- GARCH 후처리를 메인 파이프라인 구간 방법으로 통합 권장
- Bootstrap 구현 개선 필요 (잔차 재표본 로직)
- Conformal은 구간 폭 보정 후 재검증 필요

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
from vectrix.intervals.bootstrap import BootstrapInterval
from vectrix.intervals.conformal import ConformalInterval

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


def _evaluateInterval(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Dict[str, float]:
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


def _garchInterval(residuals: np.ndarray, predictions: np.ndarray, steps: int):
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

    margin = np.zeros(steps)
    forecSigma2 = sigma2[-1]
    for h in range(steps):
        forecSigma2 = omega + (alphaG + betaG) * forecSigma2
        margin[h] = z * np.sqrt(forecSigma2)

    return predictions - margin, predictions + margin


def runExperiment():
    print("=" * 70)
    print("E026: Conformal vs Bootstrap vs GARCH 후처리 구간 비교")
    print("=" * 70)

    print("\n1. 데이터셋 준비 중...")
    t0 = time.time()
    datasets = _prepareDatasets()
    print(f"   총 {len(datasets)}개")

    factories = _buildModelFactories()

    methods = ["fixed", "conformal_split", "conformal_jack", "bootstrap", "garch"]
    results = {m: [] for m in methods}

    print("\n2. 모델별 구간 비교 중...")
    for idx, (dsName, values) in enumerate(datasets):
        splitIdx = int(len(values) * 0.8)
        train = values[:splitIdx]
        test = values[splitIdx:]

        if len(train) < 30 or len(test) < 5:
            continue

        steps = len(test)

        for modelName in MODELS_TO_TEST:
            modelFactory = factories[modelName]

            model = modelFactory()
            model.fit(train)
            pred, lo, hi = model.predict(steps)

            if not np.all(np.isfinite(pred)):
                continue

            sigma = np.std(train - np.mean(train))
            if hasattr(model, 'residuals') and model.residuals is not None and len(model.residuals) > 0:
                residuals = model.residuals
                sigma = np.std(residuals)
            else:
                nFit = int(len(train) * 0.7)
                model2 = modelFactory()
                model2.fit(train[:nFit])
                holdPred, _, _ = model2.predict(len(train) - nFit)
                if np.all(np.isfinite(holdPred)):
                    residuals = train[nFit:] - holdPred[:len(train) - nFit]
                    sigma = np.std(residuals)
                else:
                    residuals = np.diff(train)
                    sigma = np.std(residuals)

            if sigma < 1e-10:
                continue

            fixedMargin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
            fixedEval = _evaluateInterval(test, pred - fixedMargin, pred + fixedMargin)
            results["fixed"].append({"dataset": dsName, "model": modelName, **fixedEval})

            confSplit = ConformalInterval(method='split', coverageLevel=0.95)
            confSplit.calibrate(train, modelFactory, steps)
            cLo, cHi = confSplit.predict(pred)
            confSplitEval = _evaluateInterval(test, cLo, cHi)
            results["conformal_split"].append({"dataset": dsName, "model": modelName, **confSplitEval})

            confJack = ConformalInterval(method='jackknife', coverageLevel=0.95)
            confJack.calibrate(train, modelFactory, steps)
            jLo, jHi = confJack.predict(pred)
            confJackEval = _evaluateInterval(test, jLo, jHi)
            results["conformal_jack"].append({"dataset": dsName, "model": modelName, **confJackEval})

            boot = BootstrapInterval(nBoot=100, coverageLevel=0.95)
            boot.calibrate(train, modelFactory, steps)
            bLo, bHi = boot.predict(pred)
            bootEval = _evaluateInterval(test, bLo, bHi)
            results["bootstrap"].append({"dataset": dsName, "model": modelName, **bootEval})

            gLo, gHi = _garchInterval(residuals, pred, steps)
            garchEval = _evaluateInterval(test, gLo, gHi)
            results["garch"].append({"dataset": dsName, "model": modelName, **garchEval})

        if (idx + 1) % 10 == 0:
            print(f"  처리 완료: {idx + 1}/{len(datasets)}")

    n = len(results["fixed"])
    elapsed = time.time() - t0

    print(f"   유효: {n}건")

    print("\n" + "=" * 70)
    print("E026 결과 요약")
    print("=" * 70)

    print(f"\n  데이터셋: {n}건")
    print(f"\n  {'방법':<20} {'Coverage':>10} {'CovGap':>10} {'Width':>10} {'Winkler':>10}")
    print(f"  {'-' * 64}")

    for method in methods:
        mResults = results[method]
        cov = np.mean([r["coverage"] for r in mResults])
        gap = np.mean([r["coverageGap"] for r in mResults])
        width = np.mean([r["width"] for r in mResults])
        winkler = np.mean([r["winklerScore"] for r in mResults])
        methodLabel = {
            "fixed": "고정 1.96σ",
            "conformal_split": "Conformal Split",
            "conformal_jack": "Conformal Jack",
            "bootstrap": "Bootstrap",
            "garch": "GARCH 후처리",
        }[method]
        print(f"  {methodLabel:<20} {cov:>9.3f} {gap:>9.3f} {width:>9.1f} {winkler:>9.1f}")

    fixedWinkler = np.mean([r["winklerScore"] for r in results["fixed"]])
    print("\n  Winkler 개선 (vs 고정):")
    for method in methods[1:]:
        mWinkler = np.mean([r["winklerScore"] for r in results[method]])
        improve = (fixedWinkler - mWinkler) / fixedWinkler * 100
        print(f"    {method:<20}: {improve:+.1f}%")

    print("\n  Winkler 최저 승률:")
    wins = {m: 0 for m in methods}
    for i in range(n):
        bestMethod = min(methods, key=lambda m: results[m][i]["winklerScore"])
        wins[bestMethod] += 1

    for method in methods:
        print(f"    {method:<20}: {wins[method]}승 ({wins[method] / n:.1%})")

    print("\n  Coverage Gap 최저 승률:")
    gapWins = {m: 0 for m in methods}
    for i in range(n):
        bestMethod = min(methods, key=lambda m: results[m][i]["coverageGap"])
        gapWins[bestMethod] += 1

    for method in methods:
        print(f"    {method:<20}: {gapWins[method]}승 ({gapWins[method] / n:.1%})")

    print("\n  모델별 최적 방법 (Winkler 기준):")
    for modelName in MODELS_TO_TEST:
        modelWins = {m: 0 for m in methods}
        modelCount = 0
        for i in range(n):
            if results["fixed"][i]["model"] == modelName:
                bestMethod = min(methods, key=lambda m: results[m][i]["winklerScore"])
                modelWins[bestMethod] += 1
                modelCount += 1
        if modelCount > 0:
            bestForModel = max(modelWins, key=modelWins.get)
            print(f"    {modelName:<15}: {bestForModel} ({modelWins[bestForModel]}/{modelCount})")

    print("\n  가설 검증:")
    confCovGap = np.mean([r["coverageGap"] for r in results["conformal_split"]])
    fixedCovGap = np.mean([r["coverageGap"] for r in results["fixed"]])
    h1 = "채택" if confCovGap < fixedCovGap else "기각"
    print(f"    가설 1 (Conformal Coverage 가장 정확): {h1} "
          f"(Conformal gap={confCovGap:.3f} vs 고정 gap={fixedCovGap:.3f})")

    garchWinkler = np.mean([r["winklerScore"] for r in results["garch"]])
    bestWinklerMethod = min(methods, key=lambda m: np.mean([r["winklerScore"] for r in results[m]]))
    h2 = "채택" if bestWinklerMethod == "garch" else "기각"
    print(f"    가설 2 (GARCH Winkler 최적): {h2} (최적: {bestWinklerMethod})")

    bootStds = []
    for modelName in MODELS_TO_TEST:
        modelResults = [results["bootstrap"][i]["winklerScore"]
                         for i in range(n) if results["bootstrap"][i]["model"] == modelName]
        if modelResults:
            bootStds.append(np.std(modelResults))
    fixedStds = []
    for modelName in MODELS_TO_TEST:
        modelResults = [results["fixed"][i]["winklerScore"]
                         for i in range(n) if results["fixed"][i]["model"] == modelName]
        if modelResults:
            fixedStds.append(np.std(modelResults))
    h3 = "채택" if np.mean(bootStds) < np.mean(fixedStds) else "기각"
    print(f"    가설 3 (Bootstrap 가장 안정적): {h3} "
          f"(Bootstrap std={np.mean(bootStds):.1f} vs 고정 std={np.mean(fixedStds):.1f})")

    print(f"\n  총 실행 시간: {elapsed:.1f}s")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    runExperiment()
