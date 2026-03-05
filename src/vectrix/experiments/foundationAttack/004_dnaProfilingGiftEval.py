"""
실험 ID: foundationAttack/004
실험명: GIFT-Eval 7개 도메인 DNA 프로파일 추출 + 클러스터 분석

목적:
- GIFT-Eval 55개 short 구성에서 시리즈별 DNA 65+ 특성을 추출한다
- 도메인별 DNA 분포가 통계적으로 유의하게 다른지 검증한다
- DNA 특성 공간에서 도메인 클러스터가 형성되는지 시각화한다
- 이를 통해 "DNA만으로 도메인을 구분할 수 있는가?" 기초 검증

가설:
1. 같은 도메인 내 시리즈는 DNA 특성 공간에서 가까울 것이다
2. 에너지/교통 도메인은 계절성 특성이 강하게 나타날 것이다
3. 금융/경제 도메인은 변동성/비선형 특성이 두드러질 것이다
4. 도메인 간 DNA 분포 차이가 있다면 Learned Profiling의 가치가 있다

방법:
1. 각 데이터셋에서 최대 50개 시리즈 샘플링
2. ForecastDNA.analyze()로 65+ 특성 추출
3. 도메인별 특성 분포 비교 (평균/표준편차)
4. PCA 2D 투영으로 클러스터 시각화
5. 도메인 분리도 측정 (실루엣 스코어)

데이터 리니지:
- 출처: GIFT-Eval (HuggingFace Salesforce/GiftEval)
- 도메인: 7개 전체
- 빈도: 10개 전체 (short term)
- 시리즈 수: 데이터셋당 최대 50개 × 55개 구성
- 전처리: 멀티변량은 첫 변수만 사용, NaN은 forward fill
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

FREQ_TO_PERIOD = {
    "Y": 1, "A": 1, "A-DEC": 1,
    "Q": 4, "QS": 4, "Q-DEC": 4,
    "M": 12, "MS": 12,
    "W": 52, "W-MON": 52, "W-SUN": 52, "W-FRI": 52, "W-THU": 52, "W-TUE": 52, "W-WED": 52,
    "D": 7, "B": 5,
    "H": 24, "h": 24,
    "T": 60, "min": 60,
    "5T": 288, "5min": 288,
    "10T": 144, "10min": 144,
    "15T": 96, "15min": 96,
    "S": 60, "s": 60,
    "10S": 8640, "10s": 8640,
}


def getDomain(dsName):
    return DOMAIN_MAP.get(dsName.split("/")[0], "Unknown")


def getPeriod(freq):
    freq = str(freq).strip()
    for key in sorted(FREQ_TO_PERIOD.keys(), key=len, reverse=True):
        if freq == key or freq.startswith(key):
            return FREQ_TO_PERIOD[key]
    return 1


def extractDnaFromDataset(dsName, maxSeries=50, seed=42):
    import datasets as hfDatasets
    from vectrix.adaptive.dna import ForecastDNA

    dsPath = GIFT_EVAL_DIR / dsName
    if not dsPath.exists():
        return []

    ds = hfDatasets.load_from_disk(str(dsPath)).with_format("numpy")
    nTotal = len(ds)
    freq = str(ds[0].get("freq", "D"))
    period = getPeriod(freq)

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
            nanIdx = np.where(np.isnan(y))[0]
            for i in nanIdx:
                if i == 0:
                    nextValid = np.where(~np.isnan(y[1:]))[0]
                    y[0] = y[nextValid[0] + 1] if len(nextValid) > 0 else 0.0
                else:
                    y[i] = y[i - 1]

        if len(y) < 20:
            continue

        MAX_LEN = 5000
        if len(y) > MAX_LEN:
            y = y[-MAX_LEN:]

        safePeriod = min(period, len(y) // 3)
        if safePeriod < 1:
            safePeriod = 1

        try:
            profile = dna.analyze(y, period=safePeriod)
            results.append({
                "dataset": dsName,
                "domain": getDomain(dsName),
                "freq": freq,
                "seriesIdx": int(idx),
                "features": profile.features,
                "category": profile.category,
                "difficulty": profile.difficulty,
                "difficultyScore": profile.difficultyScore,
            })
        except (ValueError, RuntimeError):
            pass

    return results


def extractAll(maxSeriesPerDs=50):
    datasets = SHORT_DATASETS.split()
    allResults = []

    print(f"{'Dataset':<45s} | {'Domain':<12s} | {'N':>4s} | {'Time':>6s}", flush=True)
    print("-" * 80, flush=True)

    for dsName in sorted(datasets):
        t0 = time.time()
        results = extractDnaFromDataset(dsName, maxSeries=maxSeriesPerDs)
        elapsed = time.time() - t0

        if results:
            allResults.extend(results)
            print(f"  {dsName:<43s} | {results[0]['domain']:<12s} | {len(results):>4d} | {elapsed:5.1f}s", flush=True)
        else:
            print(f"  {dsName:<43s} | {'SKIP':<12s} |", flush=True)

    return allResults


def analyzeDomainProfiles(allResults):
    print("\n" + "=" * 100)
    print("도메인별 DNA 특성 분포")
    print("=" * 100)

    domainFeatures = defaultdict(list)
    for r in allResults:
        domainFeatures[r["domain"]].append(r["features"])

    keyFeatures = [
        "trendStrength", "seasonalStrength", "volatilityClustering",
        "acf1", "hurstExponent", "forecastability",
        "zeroRatio", "cv", "spectralEntropy",
        "approximateEntropy", "stabilityMean", "nonlinearAutocorr",
    ]

    print(f"\n{'Feature':<25s}", end="")
    domains = sorted(domainFeatures.keys())
    for d in domains:
        print(f" | {d:>12s}", end="")
    print()
    print("-" * (25 + 15 * len(domains)))

    domainMeans = {}
    for feat in keyFeatures:
        print(f"  {feat:<23s}", end="")
        for d in domains:
            vals = [f.get(feat, 0) for f in domainFeatures[d] if feat in f]
            mean = np.mean(vals) if vals else 0
            domainMeans.setdefault(d, {})[feat] = mean
            print(f" | {mean:12.4f}", end="")
        print()

    print(f"\n  {'N (시리즈 수)':<23s}", end="")
    for d in domains:
        print(f" | {len(domainFeatures[d]):12d}", end="")
    print()

    return domainFeatures, domainMeans


def analyzeCategoryDistribution(allResults):
    print("\n" + "=" * 100)
    print("도메인별 시계열 카테고리 분포")
    print("=" * 100)

    domainCats = defaultdict(lambda: defaultdict(int))
    for r in allResults:
        domainCats[r["domain"]][r["category"]] += 1

    allCats = sorted(set(r["category"] for r in allResults))
    domains = sorted(domainCats.keys())

    print(f"\n{'Category':<15s}", end="")
    for d in domains:
        print(f" | {d:>12s}", end="")
    print()
    print("-" * (15 + 15 * len(domains)))

    for cat in allCats:
        print(f"  {cat:<13s}", end="")
        for d in domains:
            total = sum(domainCats[d].values())
            count = domainCats[d].get(cat, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f" | {pct:10.1f}%", end="")
        print()


def analyzeDifficultyDistribution(allResults):
    print("\n" + "=" * 100)
    print("도메인별 예측 난이도 분포")
    print("=" * 100)

    domainDiff = defaultdict(list)
    for r in allResults:
        domainDiff[r["domain"]].append(r["difficultyScore"])

    domains = sorted(domainDiff.keys())
    print(f"\n{'Domain':<14s} | {'Mean':>6s} | {'Median':>6s} | {'Std':>6s} | {'Easy%':>6s} | {'Med%':>6s} | {'Hard%':>6s} | {'VHard%':>6s}")
    print("-" * 90)

    for d in domains:
        scores = domainDiff[d]
        mean = np.mean(scores)
        median = np.median(scores)
        std = np.std(scores)

        difficulties = [r["difficulty"] for r in allResults if r["domain"] == d]
        total = len(difficulties)
        easy = 100 * difficulties.count("easy") / total
        med = 100 * difficulties.count("medium") / total
        hard = 100 * difficulties.count("hard") / total
        vhard = 100 * difficulties.count("very_hard") / total

        print(f"  {d:<12s} | {mean:6.1f} | {median:6.1f} | {std:6.1f} | "
              f"{easy:5.1f}% | {med:5.1f}% | {hard:5.1f}% | {vhard:5.1f}%")


def runPcaAnalysis(allResults):
    print("\n" + "=" * 100)
    print("PCA 2D 클러스터 분석")
    print("=" * 100)

    featureNames = [
        "trendStrength", "seasonalStrength", "volatilityClustering",
        "acf1", "acf2", "acf3", "hurstExponent", "forecastability",
        "zeroRatio", "cv", "spectralEntropy", "approximateEntropy",
        "stabilityMean", "stabilityVariance", "nonlinearAutocorr",
        "trendSlope", "trendLinearity", "seasonalAmplitude",
        "volatility", "garchEffect", "extremeValueRatio",
        "flatSpotRate", "crossingRate", "turningPointRate",
        "acfDecayRate", "ljungBoxStat", "diffStationary",
    ]

    X = []
    labels = []
    for r in allResults:
        row = []
        skip = False
        for fn in featureNames:
            val = r["features"].get(fn, 0)
            if not np.isfinite(val):
                skip = True
                break
            row.append(val)
        if not skip:
            X.append(row)
            labels.append(r["domain"])

    X = np.array(X)
    labels = np.array(labels)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds < 1e-10] = 1.0
    Xnorm = (X - means) / stds

    cov = np.cov(Xnorm.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    sortIdx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sortIdx]
    eigenvectors = eigenvectors[:, sortIdx]

    pc1 = Xnorm @ eigenvectors[:, 0]
    pc2 = Xnorm @ eigenvectors[:, 1]

    varExplained1 = eigenvalues[0] / np.sum(eigenvalues) * 100
    varExplained2 = eigenvalues[1] / np.sum(eigenvalues) * 100

    print(f"\n  총 시리즈: {len(X)}")
    print(f"  특성 수: {len(featureNames)}")
    print(f"  PC1 설명 분산: {varExplained1:.1f}%")
    print(f"  PC2 설명 분산: {varExplained2:.1f}%")
    print(f"  PC1+PC2: {varExplained1 + varExplained2:.1f}%")

    domains = sorted(set(labels))
    print(f"\n{'Domain':<14s} | {'PC1 mean':>8s} | {'PC1 std':>8s} | {'PC2 mean':>8s} | {'PC2 std':>8s}")
    print("-" * 60)
    for d in domains:
        mask = labels == d
        print(f"  {d:<12s} | {np.mean(pc1[mask]):8.3f} | {np.std(pc1[mask]):8.3f} | "
              f"{np.mean(pc2[mask]):8.3f} | {np.std(pc2[mask]):8.3f}")

    print("\nPC1 주요 적재값 (top-5):")
    loadings1 = eigenvectors[:, 0]
    topIdx1 = np.argsort(np.abs(loadings1))[::-1][:5]
    for i in topIdx1:
        print(f"  {featureNames[i]:<25s}: {loadings1[i]:+.4f}")

    print("\nPC2 주요 적재값 (top-5):")
    loadings2 = eigenvectors[:, 1]
    topIdx2 = np.argsort(np.abs(loadings2))[::-1][:5]
    for i in topIdx2:
        print(f"  {featureNames[i]:<25s}: {loadings2[i]:+.4f}")

    return computeSilhouette(Xnorm, labels)


def computeSilhouette(X, labels):
    print("\n" + "=" * 100)
    print("실루엣 스코어 (도메인 분리도)")
    print("=" * 100)

    domains = sorted(set(labels))
    domainIdx = {d: i for i, d in enumerate(domains)}
    numericLabels = np.array([domainIdx[l] for l in labels])

    n = len(X)
    if n > 2000:
        rng = np.random.RandomState(42)
        sampleIdx = rng.choice(n, 2000, replace=False)
        X = X[sampleIdx]
        numericLabels = numericLabels[sampleIdx]
        labels = labels[sampleIdx]
        n = 2000

    silhouettes = []
    for i in range(n):
        myCluster = numericLabels[i]
        sameCluster = numericLabels == myCluster
        diffClusters = ~sameCluster

        if np.sum(sameCluster) <= 1:
            silhouettes.append(0)
            continue

        dists = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))

        a = np.mean(dists[sameCluster & (np.arange(n) != i)])

        bMin = float("inf")
        for c in range(len(domains)):
            if c == myCluster:
                continue
            cMask = numericLabels == c
            if np.sum(cMask) == 0:
                continue
            bMin = min(bMin, np.mean(dists[cMask]))

        if bMin == float("inf"):
            silhouettes.append(0)
        else:
            silhouettes.append((bMin - a) / max(a, bMin))

    overallSil = np.mean(silhouettes)
    print(f"\n  전체 실루엣 스코어: {overallSil:.4f}")
    print(f"  해석: ", end="")
    if overallSil > 0.5:
        print("강한 클러스터 구조 — 도메인이 DNA로 잘 분리됨")
    elif overallSil > 0.25:
        print("약한 클러스터 구조 — 일부 도메인만 분리됨")
    elif overallSil > 0:
        print("거의 분리 안 됨 — 도메인 간 DNA 중첩 큼")
    else:
        print("분리 없음 — DNA만으로 도메인 구분 불가")

    print(f"\n  도메인별 실루엣:")
    for d in domains:
        mask = labels == d
        domSil = np.mean(np.array(silhouettes)[mask])
        print(f"    {d:<14s}: {domSil:.4f}")

    return overallSil


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 100)
    print("DNA Profiling on GIFT-Eval — Phase 1, Experiment 004")
    print("=" * 100)

    MAX_SERIES = 50

    print(f"\n[설정] 데이터셋당 최대 {MAX_SERIES}개 시리즈")
    print(f"[설정] 시드: 42")

    t0 = time.time()
    allResults = extractAll(maxSeriesPerDs=MAX_SERIES)
    totalTime = time.time() - t0

    print(f"\n총 {len(allResults)}개 시리즈 DNA 추출 완료 ({totalTime:.1f}초)")

    domainFeatures, domainMeans = analyzeDomainProfiles(allResults)
    analyzeCategoryDistribution(allResults)
    analyzeDifficultyDistribution(allResults)
    silScore = runPcaAnalysis(allResults)

    outPath = GIFT_EVAL_DIR / "dna_profiles.json"
    serializable = []
    for r in allResults:
        sr = dict(r)
        sr["features"] = {k: float(v) if np.isfinite(v) else 0.0 for k, v in r["features"].items()}
        serializable.append(sr)
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)
    print(f"\n[저장] {outPath} ({len(serializable)}개 프로파일)")
