"""
실험 ID: foundationAttack/008
실험명: 다중 모델 Oracle 분석 — 도메인/빈도별 최적 모델 분포

목적:
- DOT뿐 아니라 5개 핵심 통계 모델의 시리즈별 MASE를 수집한다
- Oracle(시리즈별 최적 모델 선택)이 단일 모델 대비 얼마나 개선되는지 측정
- 도메인/빈도별로 어떤 모델이 가장 자주 승리하는지 분석
- Phase 2-009의 학습 데이터(DNA + 최적 모델 레이블) 생성

가설:
1. Oracle 선택은 최고 단일 모델 대비 MASE -15% 이상 개선
2. 도메인마다 승리 모델 분포가 다름 (→ 도메인 인식 선택의 가치)
3. 빈도별로도 승리 모델 분포가 다름 (Y/Q는 Theta, H/D는 DOT?)

방법:
1. GIFT-Eval 55개 short 구성에서 시리즈당 5개 모델 실행
2. 모델: DOT, AutoETS, AutoCES, FourTheta, AutoCroston
3. 시리즈별 Oracle = min(MASE) 모델
4. 도메인/빈도별 Oracle 분포 분석
5. DNA + Oracle 레이블 → JSON 저장 (009 입력)

데이터 리니지:
- 출처: GIFT-Eval short (55구성)
- 시리즈 수: 데이터셋당 최대 20개
- 시드: 42

결과 (실험 후 작성):
- (아래에 기록)

결론:
- (실험 후 작성)

실험일: 2026-03-05
"""

import sys
import io
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

GIFT_EVAL_DIR = Path("data/gift_eval")

FREQ_TO_PERIOD = {
    "Y": 1, "A": 1, "A-DEC": 1,
    "Q": 4, "QS": 4, "Q-DEC": 4,
    "M": 12, "MS": 12,
    "W": 52, "W-MON": 52, "W-SUN": 52, "W-FRI": 52, "W-THU": 52, "W-TUE": 52, "W-WED": 52,
    "D": 7, "B": 5,
    "H": 24, "h": 24,
    "5T": 288, "5min": 288, "10T": 144, "10min": 144, "15T": 96, "15min": 96,
    "10S": 8640, "10s": 8640,
}

M4_PRED = {"Y": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
STD_PRED = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}

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

MODEL_IDS = ["dot", "auto_ets", "auto_ces", "four_theta", "auto_croston"]


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


def createModel(modelId, period):
    from vectrix.engine.registry import createModel as regCreate
    return regCreate(modelId, period)


def runMultiModelOnDataset(dsName, maxSeries=20, seed=42):
    import datasets as hfDatasets
    from vectrix.adaptive.dna import ForecastDNA

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

    rng = np.random.RandomState(seed)
    indices = rng.choice(nTotal, size=min(maxSeries, nTotal), replace=False)
    indices.sort()

    dna = ForecastDNA()
    results = []

    for idx in indices:
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

        MAX_LEN = 5000
        if len(y) > MAX_LEN:
            y = y[-MAX_LEN:]

        safePeriod = min(period, len(y) // 3)
        if safePeriod < 1:
            safePeriod = 1

        trainY = y[:-predLength]
        testY = y[-predLength:]

        try:
            profile = dna.analyze(trainY, period=safePeriod)
        except (ValueError, RuntimeError):
            continue

        mases = {}
        for modelId in MODEL_IDS:
            try:
                model = createModel(modelId, safePeriod)
                model.fit(trainY)
                pred, _, _ = model.predict(predLength)

                validMask = ~np.isnan(testY) & ~np.isnan(pred)
                if np.sum(validMask) < 1:
                    continue

                mase = computeMASE(testY[validMask], pred[validMask], trainY, safePeriod)
                if np.isfinite(mase):
                    mases[modelId] = float(mase)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass

        if len(mases) < 2:
            continue

        oracleModel = min(mases, key=mases.get)
        oracleMase = mases[oracleModel]

        feats = {k: float(v) if np.isfinite(v) else 0.0 for k, v in profile.features.items()}

        results.append({
            "dataset": dsName,
            "domain": domain,
            "freq": freq,
            "freqCat": freqCat,
            "seriesIdx": int(idx),
            "period": safePeriod,
            "length": len(trainY),
            "mases": mases,
            "oracleModel": oracleModel,
            "oracleMase": oracleMase,
            "features": feats,
            "category": profile.category,
            "difficulty": profile.difficulty,
        })

    return results


def analyzeResults(allResults):
    print(f"\n총 {len(allResults)}개 시리즈 × {len(MODEL_IDS)}개 모델", flush=True)

    modelMases = defaultdict(list)
    oracleMases = []
    oracleWins = defaultdict(int)

    for r in allResults:
        for mid, mase in r["mases"].items():
            modelMases[mid].append(mase)
        oracleMases.append(r["oracleMase"])
        oracleWins[r["oracleModel"]] += 1

    print("\n" + "=" * 90, flush=True)
    print("모델별 평균 MASE", flush=True)
    print("=" * 90, flush=True)
    print(f"\n  {'Model':<15s} | {'Mean MASE':>9s} | {'Median':>8s} | {'Count':>5s}", flush=True)
    print("  " + "-" * 50, flush=True)

    for mid in MODEL_IDS:
        vals = modelMases[mid]
        if vals:
            print(f"  {mid:<15s} | {np.mean(vals):9.3f} | {np.median(vals):8.3f} | {len(vals):5d}", flush=True)

    print(f"  {'ORACLE':<15s} | {np.mean(oracleMases):9.3f} | {np.median(oracleMases):8.3f} | {len(oracleMases):5d}", flush=True)

    bestSingle = min(MODEL_IDS, key=lambda m: np.mean(modelMases[m]) if modelMases[m] else 999)
    bestMase = np.mean(modelMases[bestSingle])
    oracleMean = np.mean(oracleMases)
    improvement = (bestMase - oracleMean) / bestMase * 100

    print(f"\n  최고 단일 모델: {bestSingle} (MASE {bestMase:.3f})", flush=True)
    print(f"  Oracle MASE: {oracleMean:.3f}", flush=True)
    print(f"  Oracle 개선율: -{improvement:.1f}%", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("Oracle 승리 모델 분포", flush=True)
    print("=" * 90, flush=True)
    total = len(allResults)
    for mid in sorted(oracleWins.keys(), key=lambda m: oracleWins[m], reverse=True):
        pct = 100 * oracleWins[mid] / total
        bar = "#" * int(pct / 2)
        print(f"  {mid:<15s}: {oracleWins[mid]:>4d} ({pct:5.1f}%) {bar}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("도메인별 Oracle 승리 모델", flush=True)
    print("=" * 90, flush=True)

    domainResults = defaultdict(list)
    for r in allResults:
        domainResults[r["domain"]].append(r)

    for domain in sorted(domainResults.keys()):
        drs = domainResults[domain]
        dWins = defaultdict(int)
        dMases = defaultdict(list)
        dOracle = []
        for r in drs:
            dWins[r["oracleModel"]] += 1
            dOracle.append(r["oracleMase"])
            for mid, mase in r["mases"].items():
                dMases[mid].append(mase)

        print(f"\n  {domain} (n={len(drs)}, Oracle MASE={np.mean(dOracle):.3f})", flush=True)
        for mid in sorted(dWins.keys(), key=lambda m: dWins[m], reverse=True):
            avgM = np.mean(dMases[mid]) if dMases[mid] else 0
            print(f"    {mid:<15s}: {dWins[mid]:>3d} wins ({100*dWins[mid]/len(drs):4.1f}%), avg MASE={avgM:.3f}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("빈도별 Oracle 승리 모델", flush=True)
    print("=" * 90, flush=True)

    freqResults = defaultdict(list)
    for r in allResults:
        freqResults[r["freqCat"]].append(r)

    for fCat in sorted(freqResults.keys()):
        frs = freqResults[fCat]
        fWins = defaultdict(int)
        fOracle = []
        for r in frs:
            fWins[r["oracleModel"]] += 1
            fOracle.append(r["oracleMase"])

        print(f"\n  {fCat} (n={len(frs)}, Oracle MASE={np.mean(fOracle):.3f})", flush=True)
        for mid in sorted(fWins.keys(), key=lambda m: fWins[m], reverse=True):
            print(f"    {mid:<15s}: {fWins[mid]:>3d} wins ({100*fWins[mid]/len(frs):4.1f}%)", flush=True)

    return oracleMean, improvement


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 90, flush=True)
    print("Multi-Model Oracle — Phase 2, Experiment 008", flush=True)
    print("=" * 90, flush=True)

    MAX_SERIES = 20
    print(f"\n[설정] 데이터셋당 최대 {MAX_SERIES}개, 모델 {len(MODEL_IDS)}개", flush=True)
    print(f"[모델] {', '.join(MODEL_IDS)}", flush=True)

    datasets = SHORT_DATASETS.split()
    allResults = []

    print(f"\n{'Dataset':<45s} | {'N':>4s} | {'Oracle':>6s} | {'Best':>12s} | {'Time':>6s}", flush=True)
    print("-" * 90, flush=True)

    for dsName in sorted(datasets):
        t0 = time.time()
        results = runMultiModelOnDataset(dsName, maxSeries=MAX_SERIES)
        elapsed = time.time() - t0

        if results:
            allResults.extend(results)
            oM = np.mean([r["oracleMase"] for r in results])
            wins = defaultdict(int)
            for r in results:
                wins[r["oracleModel"]] += 1
            bestWin = max(wins.items(), key=lambda x: x[1])
            print(f"  {dsName:<43s} | {len(results):>4d} | {oM:6.3f} | {bestWin[0]:>12s} | {elapsed:5.1f}s", flush=True)

    oracleMean, improvement = analyzeResults(allResults)

    outPath = GIFT_EVAL_DIR / "multi_model_oracle.json"
    serializable = []
    for r in allResults:
        sr = dict(r)
        sr["features"] = {k: float(v) for k, v in r["features"].items()}
        serializable.append(sr)
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)
    print(f"\n[저장] {outPath} ({len(serializable)}개 시리즈)", flush=True)
