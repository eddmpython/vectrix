"""
실험 ID: foundationAttack/012
실험명: Scaled Oracle — 확대 데이터 + 6모델 Oracle 재구축

목적:
- E008(704 시리즈, 4모델)을 3,000+로 확대하여 학습 데이터 병목 해소
- 모델 풀 확대: auto_mstl, tbats 추가 → 6모델 Oracle
- 확대된 Oracle gap이 E008(-17.7%)보다 커지는지 확인
- E013의 GBT 재학습을 위한 대규모 학습 데이터 생성

가설:
1. 6모델 Oracle gap > 4모델 Oracle gap (-17.7%) — auto_mstl/tbats가 새 니치를 차지
2. 3,000+ 시리즈에서 Oracle 분포가 안정화 (E008의 704개 대비 분산 감소)
3. auto_mstl이 고빈도(H/T/S)에서 DOT보다 낮은 MASE

방법:
1. GIFT-Eval 55개 short 구성에서 시리즈당 최대 100개
2. 6모델: dot, auto_ets, auto_ces, four_theta, auto_mstl, tbats
3. DNA 66특성 + 6모델 MASE → JSON 저장
4. Oracle 분석: 도메인별/빈도별 분포

데이터 리니지:
- 출처: GIFT-Eval short (55구성)
- 시리즈당: 최대 100개 (E008의 5배)
- 기대 총량: 3,000~5,000개
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
OUTPUT_PATH = ROOT / "data" / "gift_eval" / "scaled_oracle_6model.json"

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

MODEL_IDS = ["dot", "auto_ets", "auto_ces", "four_theta", "auto_mstl", "tbats"]
MAX_SERIES = 100
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


def createModel(modelId, period):
    from vectrix.engine.registry import createModel as regCreate
    return regCreate(modelId, period)


def runDataset(dsName):
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

    rng = np.random.RandomState(42)
    nSample = min(MAX_SERIES, nTotal)
    indices = rng.choice(nTotal, size=nSample, replace=False)
    indices.sort()

    dna = ForecastDNA()
    results = []
    skipped = 0

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
            skipped += 1
            continue

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
            skipped += 1
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
            skipped += 1
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

    print(f"\n{'='*90}")
    print("모델별 평균 MASE")
    print(f"{'='*90}")
    print(f"\n  {'Model':<15s} | {'Mean MASE':>9s} | {'Median':>8s} | {'Count':>5s}", flush=True)
    print("  " + "-" * 50)

    for mid in MODEL_IDS:
        vals = modelMases[mid]
        if vals:
            print(f"  {mid:<15s} | {np.mean(vals):9.3f} | {np.median(vals):8.3f} | {len(vals):5d}", flush=True)

    print(f"  {'ORACLE':<15s} | {np.mean(oracleMases):9.3f} | {np.median(oracleMases):8.3f} | {len(oracleMases):5d}", flush=True)

    bestSingle = min(MODEL_IDS, key=lambda m: np.mean(modelMases[m]) if modelMases[m] else 999)
    bestMase = np.mean(modelMases[bestSingle])
    oracleMean = np.mean(oracleMases)
    improvement = (bestMase - oracleMean) / bestMase * 100

    print(f"\n  최고 단일 모델: {bestSingle} (MASE {bestMase:.3f})")
    print(f"  Oracle MASE: {oracleMean:.3f}")
    print(f"  Oracle 개선율: -{improvement:.1f}%", flush=True)

    N = len(allResults)
    print(f"\n{'='*90}")
    print("Oracle 승리 모델 분포")
    print(f"{'='*90}")
    for mid, cnt in sorted(oracleWins.items(), key=lambda x: -x[1]):
        print(f"  {mid:<15s}: {cnt:4d} ({cnt/N*100:.1f}%)", flush=True)

    print(f"\n{'='*90}")
    print("도메인별 Oracle 분포")
    print(f"{'='*90}")

    domainResults = defaultdict(list)
    for r in allResults:
        domainResults[r["domain"]].append(r)

    for dom in sorted(domainResults):
        drs = domainResults[dom]
        dWins = defaultdict(int)
        dMases = defaultdict(list)
        for r in drs:
            dWins[r["oracleModel"]] += 1
            for mid, m in r["mases"].items():
                dMases[mid].append(m)

        topModel = max(dWins, key=dWins.get)
        topPct = dWins[topModel] / len(drs) * 100
        dotMase = np.mean(dMases["dot"]) if dMases["dot"] else float("nan")
        oMase = np.mean([r["oracleMase"] for r in drs])

        print(f"\n  {dom} ({len(drs)}개)")
        print(f"    DOT MASE: {dotMase:.3f}, Oracle MASE: {oMase:.3f}, Gap: {(1-oMase/dotMase)*100:+.1f}%")
        for mid, cnt in sorted(dWins.items(), key=lambda x: -x[1])[:4]:
            print(f"    {mid}: {cnt} ({cnt/len(drs)*100:.1f}%)", flush=True)

    print(f"\n{'='*90}")
    print("빈도별 Oracle 분포")
    print(f"{'='*90}")

    freqResults = defaultdict(list)
    for r in allResults:
        freqResults[r["freqCat"]].append(r)

    for fc in ["Y", "Q", "M", "W", "D", "H", "T", "S"]:
        if fc not in freqResults:
            continue
        frs = freqResults[fc]
        fWins = defaultdict(int)
        fMases = defaultdict(list)
        for r in frs:
            fWins[r["oracleModel"]] += 1
            for mid, m in r["mases"].items():
                fMases[mid].append(m)

        topModel = max(fWins, key=fWins.get)
        dotMase = np.mean(fMases["dot"]) if fMases["dot"] else float("nan")
        oMase = np.mean([r["oracleMase"] for r in frs])

        print(f"\n  {fc} ({len(frs)}개)")
        print(f"    DOT MASE: {dotMase:.3f}, Oracle MASE: {oMase:.3f}")
        for mid, cnt in sorted(fWins.items(), key=lambda x: -x[1])[:3]:
            print(f"    {mid}: {cnt} ({cnt/len(frs)*100:.1f}%)", flush=True)

    e008Models = ["dot", "auto_ets", "auto_ces", "four_theta"]
    e008Oracle = []
    for r in allResults:
        e008Mases = {m: r["mases"][m] for m in e008Models if m in r["mases"]}
        if e008Mases:
            e008Oracle.append(min(e008Mases.values()))

    e008OracleAvg = np.mean(e008Oracle)
    fullOracleAvg = np.mean(oracleMases)
    mstlContrib = sum(1 for r in allResults if r["oracleModel"] == "auto_mstl")
    tbatsContrib = sum(1 for r in allResults if r["oracleModel"] == "tbats")

    print(f"\n{'='*90}")
    print("신규 모델 기여도 (auto_mstl + tbats)")
    print(f"{'='*90}")
    print(f"  4모델 Oracle MASE: {e008OracleAvg:.3f}")
    print(f"  6모델 Oracle MASE: {fullOracleAvg:.3f}")
    print(f"  추가 개선: {(1-fullOracleAvg/e008OracleAvg)*100:+.1f}%")
    print(f"  auto_mstl 승리: {mstlContrib}개 ({mstlContrib/N*100:.1f}%)")
    print(f"  tbats 승리: {tbatsContrib}개 ({tbatsContrib/N*100:.1f}%)", flush=True)


def main():
    np.random.seed(42)
    print("=" * 90)
    print(f"Scaled Oracle — 6모델, {MAX_SERIES}시리즈/데이터셋")
    print("=" * 90)

    dsList = SHORT_DATASETS.split()
    allResults = []
    totalTime = time.time()

    for i, dsName in enumerate(dsList):
        t0 = time.time()
        results = runDataset(dsName)
        dt = time.time() - t0
        allResults.extend(results)
        elapsed = time.time() - totalTime

        print(f"  [{i+1:2d}/{len(dsList)}] {dsName:<40s}: {len(results):3d}개 ({dt:.1f}s) | 누적 {len(allResults)}개 ({elapsed:.0f}s)", flush=True)

    print(f"\n총 소요시간: {time.time()-totalTime:.0f}초", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(allResults, f, indent=2)
    print(f"저장: {OUTPUT_PATH} ({len(allResults)}개)", flush=True)

    analyzeResults(allResults)

    dotMases = [r["mases"]["dot"] for r in allResults if "dot" in r["mases"]]
    oracleMases = [r["oracleMase"] for r in allResults]
    dotAvg = np.mean(dotMases)
    oracleAvg = np.mean(oracleMases)
    oracleGap = (1 - oracleAvg / dotAvg) * 100

    print(f"\n{'='*90}")
    print("결론")
    print(f"{'='*90}")
    print(f"  시리즈 수: {len(allResults)} (E008: 704)")
    print(f"  DOT MASE: {dotAvg:.3f}")
    print(f"  6모델 Oracle MASE: {oracleAvg:.3f}")
    print(f"  Oracle gap: {oracleGap:+.1f}%")

    h1 = oracleGap > 17.7
    print(f"\n  가설 1 (6모델 gap > 4모델 17.7%): {'통과' if h1 else '기각'} ({oracleGap:.1f}%)", flush=True)


if __name__ == "__main__":
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
