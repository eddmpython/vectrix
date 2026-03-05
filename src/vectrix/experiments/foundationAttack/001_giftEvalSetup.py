"""
실험 ID: foundationAttack/001
실험명: GIFT-Eval 데이터셋 확보 + 포맷 파악

목적:
- GIFT-Eval 벤치마크 데이터셋을 다운로드하고 구조를 파악한다
- 23개 데이터셋 × 3개 term(short/medium/long) = 최대 103개 구성의 메타데이터 수집
- 도메인별/빈도별 시리즈 수, 길이 분포, 예측 길이를 정리한다
- Vectrix 모델을 돌리기 위한 데이터 변환 파이프라인을 구축한다

가설:
1. GIFT-Eval 데이터는 M4보다 도메인이 명확하게 구분되어 있어,
   도메인별 특화 전략이 유효할 것이다
2. 고빈도(5T, 10T, 15T) 데이터가 상당 비중을 차지하며,
   이 영역에서 통계 모델이 파운데이션 모델과 경쟁 가능할 것이다

방법:
1. HuggingFace에서 GIFT-Eval 다운로드
2. 각 데이터셋의 메타데이터 수집 (시리즈 수, 길이, 빈도, 변수 수)
3. 도메인별 분류 및 통계 요약
4. Vectrix 입력 포맷으로의 변환 유틸리티 작성
5. 샘플 시리즈를 Vectrix DOT로 예측해보는 스모크 테스트

데이터 리니지:
- 출처: HuggingFace Salesforce/GiftEval
- 도메인: 7개 (에너지, 금융, 날씨, 교통, 헬스케어, 웹/클라우드, 소매)
- 빈도: 10개 (10S, 5T, 10T, 15T, H, D, W, M, Q, Y)
- 시리즈 수: 144K+
- 전처리: 원본 그대로
- 시드: N/A (탐색 실험)

결과 (실험 후 작성):
- (아래에 기록)

결론:
- (실험 후 작성)

실험일: 2026-03-05
"""

import sys
import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

GIFT_EVAL_DIR = Path("data/gift_eval")

DOMAIN_MAP = {
    "m4_yearly": "Econ/Fin",
    "m4_quarterly": "Econ/Fin",
    "m4_monthly": "Econ/Fin",
    "m4_weekly": "Econ/Fin",
    "m4_daily": "Econ/Fin",
    "m4_hourly": "Econ/Fin",
    "electricity": "Energy",
    "solar": "Energy",
    "ett1": "Energy",
    "ett2": "Energy",
    "hospital": "Healthcare",
    "covid_deaths": "Healthcare",
    "us_births": "Healthcare",
    "saugeenday": "Nature",
    "temperature_rain_with_missing": "Nature",
    "kdd_cup_2018_with_missing": "Nature",
    "jena_weather": "Nature",
    "car_parts_with_missing": "Sales",
    "restaurant": "Sales",
    "hierarchical_sales": "Sales",
    "LOOP_SEATTLE": "Transport",
    "SZ_TAXI": "Transport",
    "M_DENSE": "Transport",
    "bitbrains_fast_storage": "Web/CloudOps",
    "bitbrains_rnd": "Web/CloudOps",
    "bizitobs_application": "Web/CloudOps",
    "bizitobs_service": "Web/CloudOps",
    "bizitobs_l2c": "Web/CloudOps",
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

MED_LONG_DATASETS = (
    "electricity/15T electricity/H "
    "solar/10T solar/H "
    "kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "SZ_TAXI/15T "
    "M_DENSE/H "
    "ett1/15T ett1/H "
    "ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)

M4_PRED_LENGTH = {"Y": 6, "Q": 8, "M": 18, "W": 13, "D": 14, "H": 48}
PRED_LENGTH = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}

FREQ_TO_PERIOD = {
    "Y": 1, "Q": 4, "M": 12, "W": 52,
    "D": 7, "H": 24, "T": 60, "S": 60,
    "5T": 288, "10T": 144, "15T": 96,
    "10S": 8640,
}


def getDomain(dsName: str) -> str:
    baseName = dsName.split("/")[0]
    return DOMAIN_MAP.get(baseName, "Unknown")


def getFreqFromName(dsName: str) -> str:
    parts = dsName.split("/")
    if len(parts) > 1:
        return parts[1]
    if dsName.startswith("m4_"):
        freqMap = {
            "m4_yearly": "Y", "m4_quarterly": "Q", "m4_monthly": "M",
            "m4_weekly": "W", "m4_daily": "D", "m4_hourly": "H",
        }
        return freqMap.get(dsName, "?")
    return "?"


def getAllConfigs():
    shortDs = SHORT_DATASETS.split()
    medLongDs = MED_LONG_DATASETS.split()
    allDs = list(set(shortDs + medLongDs))

    configs = []
    for ds in sorted(allDs):
        configs.append((ds, "short"))
        if ds in medLongDs:
            configs.append((ds, "medium"))
            configs.append((ds, "long"))
    return configs


def downloadGiftEval():
    targetDir = GIFT_EVAL_DIR
    if targetDir.exists() and any(targetDir.iterdir()):
        print(f"[INFO] GIFT-Eval already exists at {targetDir}")
        return True

    print("[INFO] Downloading GIFT-Eval from HuggingFace...")
    print(f"[INFO] Target: {targetDir}")
    print()
    print("Run this command manually:")
    print(f"  huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir {targetDir}")
    print()
    print("Or in Python:")
    print("  from huggingface_hub import snapshot_download")
    print(f'  snapshot_download("Salesforce/GiftEval", repo_type="dataset", local_dir="{targetDir}")')
    return False


def exploreDataset(dsName: str):
    try:
        import datasets
    except ImportError:
        print("[ERROR] 'datasets' package not installed. Run: uv pip install datasets")
        return None

    dsPath = GIFT_EVAL_DIR / dsName
    if not dsPath.exists():
        print(f"[SKIP] {dsName} not found at {dsPath}")
        return None

    ds = datasets.load_from_disk(str(dsPath)).with_format("numpy")
    nSeries = len(ds)
    lengths = []
    for entry in ds:
        target = entry["target"]
        if target.ndim == 1:
            lengths.append(len(target))
        else:
            lengths.append(target.shape[1])

    freq = ds[0].get("freq", "?")
    nVariates = 1
    if ds[0]["target"].ndim > 1:
        nVariates = ds[0]["target"].shape[0]

    info = {
        "name": dsName,
        "domain": getDomain(dsName),
        "freq": freq if isinstance(freq, str) else str(freq),
        "nSeries": nSeries,
        "nVariates": nVariates,
        "minLength": int(np.min(lengths)),
        "maxLength": int(np.max(lengths)),
        "medianLength": int(np.median(lengths)),
        "meanLength": float(np.mean(lengths)),
        "totalPoints": int(np.sum(lengths)) * nVariates,
    }
    return info


def exploreAllDatasets():
    results = []
    shortDs = SHORT_DATASETS.split()

    for dsName in sorted(set(shortDs)):
        info = exploreDataset(dsName)
        if info is not None:
            results.append(info)
            print(f"  {dsName:<40s} | {info['domain']:<14s} | {info['freq']:<4s} | "
                  f"series={info['nSeries']:>6d} | variates={info['nVariates']:>2d} | "
                  f"len={info['minLength']:>6d}~{info['maxLength']:>6d} | "
                  f"points={info['totalPoints']:>12,d}")

    return results


def summarizeByDomain(results):
    print("\n" + "=" * 80)
    print("도메인별 요약")
    print("=" * 80)

    domainStats = defaultdict(lambda: {"datasets": 0, "series": 0, "points": 0, "freqs": set()})
    for r in results:
        d = domainStats[r["domain"]]
        d["datasets"] += 1
        d["series"] += r["nSeries"]
        d["points"] += r["totalPoints"]
        d["freqs"].add(r["freq"])

    for domain in sorted(domainStats.keys()):
        s = domainStats[domain]
        freqStr = ", ".join(sorted(s["freqs"]))
        print(f"  {domain:<14s} | datasets={s['datasets']:>2d} | series={s['series']:>7,d} | "
              f"points={s['points']:>14,d} | freqs={freqStr}")


def smokeTestVectrix(dsName: str):
    try:
        import datasets as hfDatasets
    except ImportError:
        print("[ERROR] 'datasets' package not installed")
        return

    dsPath = GIFT_EVAL_DIR / dsName
    if not dsPath.exists():
        print(f"[SKIP] {dsName} not found")
        return

    ds = hfDatasets.load_from_disk(str(dsPath)).with_format("numpy")

    entry = ds[0]
    target = entry["target"]
    if target.ndim > 1:
        target = target[0]
    y = target.astype(np.float64)

    if len(y) < 30:
        print(f"[SKIP] Series too short: {len(y)} points")
        return

    freq = getFreqFromName(dsName)
    period = FREQ_TO_PERIOD.get(freq, 1)

    print(f"\n[SMOKE TEST] {dsName}")
    print(f"  Series length: {len(y)}, freq: {freq}, period: {period}")
    print(f"  y[:10] = {y[:10]}")

    try:
        from vectrix.engine.dot import DynamicOptimizedTheta

        model = DynamicOptimizedTheta(period=period)
        model.fit(y)

        predLength = M4_PRED_LENGTH.get(freq, PRED_LENGTH.get(freq, 12))
        pred, lower, upper = model.predict(predLength)

        print(f"  Prediction length: {predLength}")
        print(f"  pred[:5] = {pred[:5]}")
        print(f"  lower[:5] = {lower[:5]}")
        print(f"  upper[:5] = {upper[:5]}")
        print(f"  [OK] DOT prediction successful")
    except Exception as e:
        print(f"  [FAIL] DOT prediction failed: {e}")


def printConfigSummary():
    configs = getAllConfigs()
    print(f"\n총 구성 수: {len(configs)}")
    print(f"  short: {sum(1 for _, t in configs if t == 'short')}")
    print(f"  medium: {sum(1 for _, t in configs if t == 'medium')}")
    print(f"  long: {sum(1 for _, t in configs if t == 'long')}")

    print("\n도메인별 데이터셋:")
    domainDs = defaultdict(list)
    for ds, _ in configs:
        if (ds, "short") in configs or _ == "short":
            domainDs[getDomain(ds)].append(ds)

    for domain in sorted(domainDs.keys()):
        unique = sorted(set(domainDs[domain]))
        print(f"  {domain}: {len(unique)}개")
        for ds in unique:
            freq = getFreqFromName(ds)
            print(f"    - {ds} ({freq})")


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 80)
    print("GIFT-Eval Setup — Phase 0, Experiment 001")
    print("=" * 80)

    print("\n[Step 1] 구성(Config) 목록 파악")
    printConfigSummary()

    print("\n[Step 2] 데이터셋 다운로드 확인")
    hasData = downloadGiftEval()

    if hasData:
        print("\n[Step 3] 데이터셋 탐색")
        results = exploreAllDatasets()

        if results:
            summarizeByDomain(results)

            print("\n[Step 4] Vectrix DOT 스모크 테스트")
            testDs = results[0]["name"]
            smokeTestVectrix(testDs)

            outPath = GIFT_EVAL_DIR / "metadata.json"
            with open(outPath, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n[저장] {outPath}")
    else:
        print("\n[Step 3] 데이터 없음 — 다운로드 후 재실행")
        print("\n먼저 다운로드 구성을 확인합니다:")
        printConfigSummary()
