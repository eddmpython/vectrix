"""
실험 ID: foundationAttack/002
실험명: DOT-Hybrid GIFT-Eval 기준선 측정

목적:
- 현재 Vectrix DOT 모델을 GIFT-Eval 55개 short 구성에서 돌려 MASE 기준선 확보
- 도메인별 DOT 성능 분포를 파악하여 "어디서 이길 수 있는지" 판단
- GIFT-Eval 평가 프레임워크(GluonTS evaluate_model)와의 호환성 확인

가설:
1. DOT는 Econ/Fin(M4) 도메인에서 가장 강할 것 (이미 검증됨)
2. Energy/Nature 도메인에서도 경쟁력 있을 것 (계절성+트렌드 데이터)
3. Web/CloudOps는 고빈도 노이즈가 많아 DOT가 약할 것

방법:
1. GIFT-Eval Dataset 클래스로 각 데이터셋 로드
2. test_data에서 각 시리즈의 과거 데이터 추출
3. DOT.fit() → DOT.predict() 실행
4. MASE, sMAPE 계산 (M4 OWA와 비교 가능하도록)
5. 도메인별/빈도별 결과 집계

데이터 리니지:
- 출처: GIFT-Eval (HuggingFace Salesforce/GiftEval)
- 도메인: 7개 전체
- 빈도: 10개 전체 (short term만)
- 시리즈 수: 55개 구성의 전체 시리즈 (144K+)
- 전처리: 원본 그대로
- 학습/검증 분할: GIFT-Eval 공식 split 사용
- 시드: N/A (결정론적 모델)

결과 (실험 후 작성):
- (아래에 기록)

결론:
- (실험 후 작성)

실험일: 2026-03-05
"""

import sys
import io
import os
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

GIFT_EVAL_DIR = Path("data/gift_eval")

FREQ_TO_PERIOD = {
    "Y": 1, "A": 1, "Q": 4, "QS": 4, "M": 12, "MS": 12,
    "W": 52, "W-MON": 52, "W-SUN": 52,
    "D": 7, "B": 5,
    "H": 24, "h": 24,
    "T": 60, "min": 60,
    "5T": 288, "5min": 288,
    "10T": 144, "10min": 144,
    "15T": 96, "15min": 96,
    "S": 60, "s": 60,
    "10S": 8640, "10s": 8640,
}

SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W "
    "hospital covid_deaths "
    "us_births/D us_births/M us_births/W "
    "saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing "
    "kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D "
    "SZ_TAXI/15T SZ_TAXI/H "
    "M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W "
    "ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H "
    "bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

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


def getDomain(dsName: str) -> str:
    return DOMAIN_MAP.get(dsName.split("/")[0], "Unknown")


def getPeriod(freq: str) -> int:
    freq = str(freq).strip()
    for key in sorted(FREQ_TO_PERIOD.keys(), key=len, reverse=True):
        if freq == key or freq.startswith(key):
            return FREQ_TO_PERIOD[key]
    return 1


def computeMASE(actual, predicted, insample, period):
    n = len(insample)
    if n <= period:
        naiveErr = np.mean(np.abs(np.diff(insample)))
    else:
        naiveErr = np.mean(np.abs(insample[period:] - insample[:-period]))

    if naiveErr < 1e-10:
        return np.nan

    forecastErr = np.mean(np.abs(actual - predicted))
    return forecastErr / naiveErr


def computeSMAPE(actual, predicted):
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom > 1e-10
    if not np.any(mask):
        return np.nan
    return 200.0 * np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask])


def runDotOnDataset(dsName: str, maxSeries: int = 0):
    import datasets as hfDatasets
    from vectrix.engine.dot import DynamicOptimizedTheta

    dsPath = GIFT_EVAL_DIR / dsName
    if not dsPath.exists():
        print(f"  [SKIP] {dsName} not found")
        return None

    ds = hfDatasets.load_from_disk(str(dsPath)).with_format("numpy")
    nTotal = len(ds)
    freq = str(ds[0].get("freq", "D"))
    period = getPeriod(freq)

    M4_PRED = {"Y": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
    STD_PRED = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}

    freqKey = freq
    for k in ["Y", "A", "Q", "M", "W", "D", "H", "T", "S"]:
        if freq.startswith(k) or freq.endswith(k):
            freqKey = k
            break
    for k in ["5T", "10T", "15T", "5min", "10min", "15min", "10S", "10s"]:
        if k in freq:
            freqKey = "T"
            break

    if dsName.startswith("m4_"):
        predLength = M4_PRED.get(freqKey, 12)
    else:
        predLength = STD_PRED.get(freqKey, 12)

    limit = nTotal if maxSeries <= 0 else min(maxSeries, nTotal)

    maseScores = []
    smapeScores = []
    failures = 0
    t0 = time.time()

    for i in range(limit):
        entry = ds[i]
        target = entry["target"]
        if target.ndim > 1:
            target = target[0]
        y = target.astype(np.float64)

        if len(y) < predLength + 10:
            failures += 1
            continue

        trainY = y[:-predLength]
        testY = y[-predLength:]

        if np.all(np.isnan(trainY)) or len(trainY) < 10:
            failures += 1
            continue

        validMask = ~np.isnan(trainY)
        if np.sum(validMask) < 10:
            failures += 1
            continue

        cleanTrain = trainY.copy()
        if np.any(np.isnan(cleanTrain)):
            nanIdx = np.where(np.isnan(cleanTrain))[0]
            for idx in nanIdx:
                if idx == 0:
                    nextValid = np.where(~np.isnan(cleanTrain[1:]))[0]
                    if len(nextValid) > 0:
                        cleanTrain[0] = cleanTrain[nextValid[0] + 1]
                    else:
                        cleanTrain[0] = 0.0
                else:
                    cleanTrain[idx] = cleanTrain[idx - 1]

        try:
            model = DynamicOptimizedTheta(period=min(period, len(cleanTrain) // 2))
            model.fit(cleanTrain)
            pred, _, _ = model.predict(predLength)

            validTest = ~np.isnan(testY)
            if np.sum(validTest) < 1:
                failures += 1
                continue

            mase = computeMASE(testY[validTest], pred[validTest], cleanTrain, min(period, len(cleanTrain) // 2))
            smape = computeSMAPE(testY[validTest], pred[validTest])

            if not np.isnan(mase):
                maseScores.append(mase)
            if not np.isnan(smape):
                smapeScores.append(smape)

        except Exception:
            failures += 1

    elapsed = time.time() - t0

    if not maseScores:
        return None

    result = {
        "dataset": dsName,
        "domain": getDomain(dsName),
        "freq": freq,
        "period": period,
        "predLength": predLength,
        "nTotal": nTotal,
        "nEvaluated": limit,
        "nSuccess": len(maseScores),
        "nFailed": failures,
        "mase_mean": float(np.mean(maseScores)),
        "mase_median": float(np.median(maseScores)),
        "smape_mean": float(np.mean(smapeScores)) if smapeScores else None,
        "elapsed_sec": round(elapsed, 2),
        "series_per_sec": round(len(maseScores) / elapsed, 1) if elapsed > 0 else 0,
    }
    return result


def runAllDatasets(maxSeriesPerDs: int = 100):
    datasets = SHORT_DATASETS.split()
    results = []

    print(f"\n{'Dataset':<45s} | {'Domain':<12s} | {'MASE':>8s} | {'sMAPE':>8s} | "
          f"{'N':>5s} | {'Time':>6s}")
    print("-" * 100)

    for dsName in sorted(datasets):
        result = runDotOnDataset(dsName, maxSeries=maxSeriesPerDs)
        if result:
            results.append(result)
            print(f"  {result['dataset']:<43s} | {result['domain']:<12s} | "
                  f"{result['mase_mean']:8.3f} | {result['smape_mean']:8.1f} | "
                  f"{result['nSuccess']:5d} | {result['elapsed_sec']:5.1f}s")
        else:
            print(f"  {dsName:<43s} | {'FAILED':<12s} |")

    return results


def summarizeResults(results):
    print("\n" + "=" * 80)
    print("도메인별 DOT MASE 요약")
    print("=" * 80)

    domainMase = defaultdict(list)
    for r in results:
        domainMase[r["domain"]].append(r["mase_mean"])

    for domain in sorted(domainMase.keys()):
        scores = domainMase[domain]
        print(f"  {domain:<14s} | datasets={len(scores):>2d} | "
              f"MASE mean={np.mean(scores):.3f} | "
              f"median={np.median(scores):.3f} | "
              f"min={np.min(scores):.3f} | max={np.max(scores):.3f}")

    print(f"\n  {'OVERALL':<14s} | datasets={len(results):>2d} | "
          f"MASE mean={np.mean([r['mase_mean'] for r in results]):.3f}")

    print("\n빈도별 DOT MASE 요약")
    print("-" * 60)
    freqMase = defaultdict(list)
    for r in results:
        freqMase[r["freq"]].append(r["mase_mean"])
    for freq in sorted(freqMase.keys()):
        scores = freqMase[freq]
        print(f"  {freq:<10s} | n={len(scores):>2d} | MASE={np.mean(scores):.3f}")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 80)
    print("DOT Baseline on GIFT-Eval — Phase 0, Experiment 002")
    print("=" * 80)

    MAX_SERIES = 100

    print(f"\n[설정] 데이터셋당 최대 {MAX_SERIES}개 시리즈 평가")
    print(f"[설정] 데이터 경로: {GIFT_EVAL_DIR}")

    if not GIFT_EVAL_DIR.exists():
        print("\n[ERROR] GIFT-Eval 데이터 없음. 001_giftEvalSetup.py 먼저 실행")
        sys.exit(1)

    results = runAllDatasets(maxSeriesPerDs=MAX_SERIES)

    if results:
        summarizeResults(results)

        outPath = GIFT_EVAL_DIR / "dot_baseline.json"
        with open(outPath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[저장] {outPath}")
