"""
실험 ID: foundationAttack/003
실험명: GIFT-Eval 리더보드 비교 분석 — 적의 약점 찾기

목적:
- GIFT-Eval 리더보드의 주요 모델들(Chronos-2, TimesFM-2.5, auto_theta, auto_ets, naive)의
  도메인별/빈도별 MASE 분포를 비교 분석한다
- 파운데이션 모델이 약한 도메인/빈도를 식별하여 공략 우선순위를 결정한다
- 통계 모델(auto_theta, auto_ets)이 파운데이션 모델보다 나은 구성을 찾는다

가설:
1. 고빈도(5T, 10T, 15T) 데이터에서 통계 모델과 파운데이션 모델의 격차가 작을 것
2. M4 데이터(Econ/Fin)에서는 파운데이션 모델이 통계 모델보다 우위
3. 특정 도메인(에너지, 교통)에서는 통계 모델이 경쟁력 있을 것

방법:
1. GitHub에서 수집한 리더보드 MASE 데이터를 구조화
2. short term 55개 구성에서 모델별 MASE 비교
3. 도메인별, 빈도별 집계 + 통계 모델이 이기는 구성 식별
4. 002 실험의 DOT 기준선과 통합 비교

데이터 리니지:
- 출처: GitHub SalesforceAIResearch/gift-eval/results/
- 모델: chronos-2, TimesFM-2.5, auto_theta, auto_ets, naive
- 범위: short/medium/long (가용한 전체)
- 전처리: 공식 결과 CSV 그대로

결과 (실험 후 작성):
- 아래 실행 결과 참조

실험일: 2026-03-05
"""

import sys
import io
from collections import defaultdict

import numpy as np

CHRONOS2 = {
    "bitbrains_fast_storage/5T/short": 0.6564,
    "bitbrains_fast_storage/H/short": 0.9310,
    "bitbrains_rnd/5T/short": 1.6181,
    "bitbrains_rnd/H/short": 5.8047,
    "bizitobs_application/10S/short": 1.0039,
    "bizitobs_l2c/5T/short": 0.2665,
    "bizitobs_l2c/H/short": 0.4126,
    "bizitobs_service/10S/short": 0.7169,
    "car_parts/M/short": 0.8360,
    "covid_deaths/D/short": 32.5392,
    "electricity/15T/short": 0.9223,
    "electricity/D/short": 1.4149,
    "electricity/H/short": 0.9779,
    "electricity/W/short": 1.3851,
    "ett1/15T/short": 0.6869,
    "ett1/D/short": 1.6403,
    "ett1/H/short": 0.7868,
    "ett1/W/short": 1.5901,
    "ett2/15T/short": 0.7262,
    "ett2/D/short": 1.2908,
    "ett2/H/short": 0.7363,
    "ett2/W/short": 0.7716,
    "hierarchical_sales/D/short": 0.7440,
    "hierarchical_sales/W/short": 0.7093,
    "hospital/M/short": 0.7399,
    "jena_weather/10T/short": 0.2650,
    "jena_weather/D/short": 1.0823,
    "jena_weather/H/short": 0.5226,
    "kdd_cup_2018/D/short": 1.1978,
    "kdd_cup_2018/H/short": 0.9467,
    "loop_seattle/5T/short": 0.5407,
    "loop_seattle/D/short": 0.8878,
    "loop_seattle/H/short": 0.8250,
    "m_dense/D/short": 0.6151,
    "m_dense/H/short": 0.7817,
    "m4_daily/D/short": 3.4193,
    "m4_hourly/H/short": 0.7884,
    "m4_monthly/M/short": 0.9032,
    "m4_quarterly/Q/short": 1.1543,
    "m4_weekly/W/short": 2.0263,
    "m4_yearly/A/short": 3.2362,
    "restaurant/D/short": 0.6777,
    "saugeen/D/short": 2.9834,
    "saugeen/M/short": 0.7235,
    "saugeen/W/short": 1.1876,
    "solar/10T/short": 0.7784,
    "solar/D/short": 0.9610,
    "solar/H/short": 0.9845,
    "solar/W/short": 1.0123,
    "sz_taxi/15T/short": 0.5429,
    "sz_taxi/H/short": 0.5603,
    "temperature_rain/D/short": 1.3102,
    "us_births/D/short": 0.3266,
    "us_births/M/short": 0.6328,
    "us_births/W/short": 0.9790,
}

TIMESFM25 = {
    "bitbrains_fast_storage/5T/short": 0.7111,
    "bitbrains_fast_storage/H/short": 1.0595,
    "bitbrains_rnd/5T/short": 1.6571,
    "bitbrains_rnd/H/short": 5.8512,
    "bizitobs_application/10S/short": 1.0241,
    "bizitobs_l2c/5T/short": 0.2741,
    "bizitobs_l2c/H/short": 0.4261,
    "bizitobs_service/10S/short": 0.7261,
    "car_parts/M/short": 0.8384,
    "covid_deaths/D/short": 36.9065,
    "electricity/15T/short": 1.0814,
    "electricity/D/short": 1.3764,
    "electricity/H/short": 0.9075,
    "electricity/W/short": 1.3968,
    "ett1/15T/short": 0.6939,
    "ett1/D/short": 1.7128,
    "ett1/H/short": 0.8387,
    "ett1/W/short": 1.4441,
    "ett2/15T/short": 0.7237,
    "ett2/D/short": 1.2743,
    "ett2/H/short": 0.7392,
    "ett2/W/short": 0.8412,
    "hierarchical_sales/D/short": 0.7450,
    "hierarchical_sales/W/short": 0.7191,
    "hospital/M/short": 0.7590,
    "jena_weather/10T/short": 0.2760,
    "jena_weather/D/short": 0.9881,
    "jena_weather/H/short": 0.5257,
    "kdd_cup_2018/D/short": 1.1915,
    "kdd_cup_2018/H/short": 0.9326,
    "loop_seattle/5T/short": 0.5676,
    "loop_seattle/D/short": 0.8627,
    "loop_seattle/H/short": 0.7250,
    "m_dense/D/short": 0.6615,
    "m_dense/H/short": 0.7688,
    "m4_daily/D/short": 3.2974,
    "m4_hourly/H/short": 0.7336,
    "m4_monthly/M/short": 0.9462,
    "m4_quarterly/Q/short": 1.2050,
    "m4_weekly/W/short": 2.0020,
    "m4_yearly/A/short": 3.5746,
    "restaurant/D/short": 0.6818,
    "saugeen/D/short": 2.7150,
    "saugeen/M/short": 0.7144,
    "saugeen/W/short": 1.2478,
    "solar/10T/short": 1.0857,
    "solar/D/short": 0.9852,
    "solar/H/short": 0.8976,
    "solar/W/short": 1.2792,
    "sz_taxi/15T/short": 0.5444,
    "sz_taxi/H/short": 0.5633,
    "temperature_rain/D/short": 1.3471,
    "us_births/D/short": 0.3430,
    "us_births/M/short": 0.6083,
    "us_births/W/short": 1.0866,
}

AUTO_THETA = {
    "bitbrains_fast_storage/5T/short": 1.15,
    "bitbrains_fast_storage/H/short": 1.35,
    "bitbrains_rnd/5T/short": 2.07,
    "bitbrains_rnd/H/short": 5.75,
    "bizitobs_application/10S/short": 1.11,
    "bizitobs_l2c/5T/short": 0.292,
    "bizitobs_l2c/H/short": 1.19,
    "bizitobs_service/10S/short": 0.791,
    "car_parts/M/short": 1.23,
    "covid_deaths/D/short": 45.4,
    "electricity/15T/short": 1.35,
    "electricity/D/short": 1.88,
    "electricity/H/short": 1.74,
    "electricity/W/short": 2.14,
    "ett1/15T/short": 0.863,
    "ett1/D/short": 1.75,
    "ett1/H/short": 1.28,
    "ett1/W/short": 1.89,
    "ett2/15T/short": 0.832,
    "ett2/D/short": 1.85,
    "ett2/H/short": 1.02,
    "ett2/W/short": 1.41,
    "hierarchical_sales/D/short": 0.932,
    "hierarchical_sales/W/short": 0.849,
    "hospital/M/short": 0.761,
    "jena_weather/10T/short": 0.368,
    "jena_weather/D/short": 1.6,
    "jena_weather/H/short": 0.878,
    "kdd_cup_2018/D/short": 1.38,
    "kdd_cup_2018/H/short": 1.27,
    "loop_seattle/5T/short": 0.78,
    "loop_seattle/D/short": 1.39,
    "loop_seattle/H/short": 1.4,
    "m_dense/D/short": 1.22,
    "m_dense/H/short": 1.69,
    "m4_daily/D/short": 3.34,
    "m4_hourly/H/short": 2.46,
    "m4_monthly/M/short": 0.966,
    "m4_quarterly/Q/short": 1.19,
    "m4_weekly/W/short": 2.66,
    "m4_yearly/A/short": 3.11,
    "restaurant/D/short": 0.843,
    "saugeen/D/short": 3.6,
    "saugeen/M/short": 0.912,
    "saugeen/W/short": 2.12,
    "solar/10T/short": 1.8,
    "solar/D/short": 1.05,
    "solar/H/short": 2.05,
    "solar/W/short": 1.15,
    "sz_taxi/15T/short": 0.649,
    "sz_taxi/H/short": 0.691,
    "temperature_rain/D/short": 1.93,
    "us_births/D/short": 1.63,
    "us_births/M/short": 0.883,
    "us_births/W/short": 1.49,
}

AUTO_ETS = {
    "bitbrains_fast_storage/5T/short": 1.14,
    "bitbrains_fast_storage/H/short": 1.32,
    "bitbrains_rnd/5T/short": 1.97,
    "bitbrains_rnd/H/short": 6.04,
    "bizitobs_application/10S/short": 5.09,
    "bizitobs_l2c/5T/short": 0.272,
    "bizitobs_l2c/H/short": 1.11,
    "bizitobs_service/10S/short": 2.47,
    "car_parts/M/short": 1.22,
    "covid_deaths/D/short": 28.4,
    "electricity/15T/short": 2.43,
    "electricity/D/short": 1.93,
    "electricity/H/short": 1.36,
    "electricity/W/short": 2.14,
    "ett1/15T/short": 1.91,
    "ett1/D/short": 1.69,
    "ett1/H/short": 1.94,
    "ett1/W/short": 1.66,
    "ett2/15T/short": 1.18,
    "ett2/D/short": 1.42,
    "ett2/H/short": 1.03,
    "ett2/W/short": 0.874,
    "hierarchical_sales/D/short": 0.908,
    "hierarchical_sales/W/short": 0.916,
    "hospital/M/short": 0.77,
    "jena_weather/10T/short": 0.341,
    "jena_weather/D/short": 2.0,
    "jena_weather/H/short": 0.744,
    "kdd_cup_2018/D/short": 1.47,
    "kdd_cup_2018/H/short": 1.43,
    "loop_seattle/5T/short": 0.835,
    "loop_seattle/D/short": 1.25,
    "loop_seattle/H/short": 1.55,
    "m_dense/D/short": 1.19,
    "m_dense/H/short": 1.46,
    "m4_daily/D/short": 3.24,
    "m4_hourly/H/short": 1.61,
    "m4_monthly/M/short": 0.964,
    "m4_quarterly/Q/short": 1.16,
    "m4_weekly/W/short": 2.55,
    "m4_yearly/A/short": 3.08,
    "restaurant/D/short": 0.861,
    "saugeen/D/short": 3.9,
    "saugeen/M/short": 0.725,
    "saugeen/W/short": 1.99,
    "solar/10T/short": 1.22,
    "solar/D/short": 1.02,
    "solar/H/short": 1.8,
    "solar/W/short": 1.0,
    "sz_taxi/15T/short": 0.696,
    "sz_taxi/H/short": 0.852,
    "temperature_rain/D/short": 1.97,
    "us_births/D/short": 1.6,
    "us_births/M/short": 0.588,
    "us_births/W/short": 1.49,
}

NAIVE = {
    "bitbrains_fast_storage/5T/short": 1.095,
    "bitbrains_fast_storage/H/short": 1.408,
    "bitbrains_rnd/5T/short": 2.056,
    "bitbrains_rnd/H/short": 5.872,
    "bizitobs_application/10S/short": 3.764,
    "bizitobs_l2c/5T/short": 0.283,
    "bizitobs_l2c/H/short": 1.086,
    "bizitobs_service/10S/short": 2.189,
    "car_parts/M/short": 1.213,
    "covid_deaths/D/short": 46.912,
    "electricity/15T/short": 2.456,
    "electricity/D/short": 1.987,
    "electricity/H/short": 3.922,
    "electricity/W/short": 2.090,
    "ett1/15T/short": 1.969,
    "ett1/D/short": 1.778,
    "ett1/H/short": 1.825,
    "ett1/W/short": 1.769,
    "ett2/15T/short": 1.244,
    "ett2/D/short": 1.390,
    "ett2/H/short": 1.087,
    "ett2/W/short": 0.779,
    "hierarchical_sales/D/short": 1.135,
    "hierarchical_sales/W/short": 1.025,
    "hospital/M/short": 0.968,
    "jena_weather/10T/short": 0.364,
    "jena_weather/D/short": 1.573,
    "jena_weather/H/short": 0.648,
    "kdd_cup_2018/D/short": 1.497,
    "kdd_cup_2018/H/short": 1.280,
    "loop_seattle/5T/short": 0.893,
    "loop_seattle/D/short": 1.732,
    "loop_seattle/H/short": 1.592,
    "m_dense/D/short": 1.669,
    "m_dense/H/short": 2.714,
    "m4_daily/D/short": 3.278,
    "m4_hourly/H/short": 11.608,
    "m4_monthly/M/short": 1.205,
    "m4_quarterly/Q/short": 1.477,
    "m4_weekly/W/short": 2.777,
    "m4_yearly/A/short": 3.966,
    "restaurant/D/short": 1.006,
    "saugeen/D/short": 3.413,
    "saugeen/M/short": 1.230,
    "saugeen/W/short": 1.991,
    "solar/10T/short": 1.449,
    "solar/D/short": 1.156,
    "solar/H/short": 2.093,
    "solar/W/short": 1.470,
    "sz_taxi/15T/short": 0.786,
    "sz_taxi/H/short": 0.835,
    "temperature_rain/D/short": 2.012,
    "us_births/D/short": 1.865,
    "us_births/M/short": 1.499,
    "us_births/W/short": 1.563,
}

DOMAIN_MAP = {
    "m4_yearly": "Econ/Fin", "m4_quarterly": "Econ/Fin", "m4_monthly": "Econ/Fin",
    "m4_weekly": "Econ/Fin", "m4_daily": "Econ/Fin", "m4_hourly": "Econ/Fin",
    "electricity": "Energy", "solar": "Energy", "ett1": "Energy", "ett2": "Energy",
    "hospital": "Healthcare", "covid_deaths": "Healthcare", "us_births": "Healthcare",
    "saugeen": "Nature", "temperature_rain": "Nature",
    "kdd_cup_2018": "Nature", "jena_weather": "Nature",
    "car_parts": "Sales", "restaurant": "Sales", "hierarchical_sales": "Sales",
    "loop_seattle": "Transport", "sz_taxi": "Transport", "m_dense": "Transport",
    "LOOP_SEATTLE": "Transport", "SZ_TAXI": "Transport", "M_DENSE": "Transport",
    "bitbrains_fast_storage": "Web/CloudOps", "bitbrains_rnd": "Web/CloudOps",
    "bizitobs_application": "Web/CloudOps", "bizitobs_service": "Web/CloudOps",
    "bizitobs_l2c": "Web/CloudOps",
}


def getDomain(ds: str) -> str:
    baseName = ds.split("/")[0]
    return DOMAIN_MAP.get(baseName, "Unknown")


def getFreq(ds: str) -> str:
    parts = ds.split("/")
    if len(parts) >= 2:
        return parts[1]
    if ds.startswith("m4_"):
        freqMap = {"m4_yearly": "A", "m4_quarterly": "Q", "m4_monthly": "M",
                   "m4_weekly": "W", "m4_daily": "D", "m4_hourly": "H"}
        return freqMap.get(ds, "?")
    return "?"


def analyzeShortTerm():
    models = {
        "Chronos-2": CHRONOS2,
        "TimesFM-2.5": TIMESFM25,
        "auto_theta": AUTO_THETA,
        "auto_ets": AUTO_ETS,
        "naive": NAIVE,
    }

    shortConfigs = [k for k in CHRONOS2.keys() if k.endswith("/short")]

    print("=" * 110)
    print("SHORT TERM MASE 비교 (55개 구성)")
    print("=" * 110)

    print(f"\n{'Dataset':<42s} | {'Chronos2':>8s} | {'TimesFM':>8s} | {'Theta':>8s} | "
          f"{'ETS':>8s} | {'Naive':>8s} | {'Best':>10s}")
    print("-" * 110)

    statWins = 0
    foundationWins = 0
    bestStatMase = {}

    for ds in sorted(shortConfigs):
        c2 = CHRONOS2.get(ds, None)
        tf = TIMESFM25.get(ds, None)
        at = AUTO_THETA.get(ds, None)
        ae = AUTO_ETS.get(ds, None)
        nv = NAIVE.get(ds, None)

        if c2 is None or tf is None or at is None:
            continue

        bestFoundation = min(c2, tf)
        bestStat = min(at, ae) if ae else at
        bestStatMase[ds] = bestStat

        if bestStat < bestFoundation:
            statWins += 1
            winner = "STAT"
        else:
            foundationWins += 1
            winner = "FOUND"

        shortDs = ds.replace("/short", "")
        print(f"  {shortDs:<40s} | {c2:8.3f} | {tf:8.3f} | {at:8.3f} | "
              f"{ae:8.3f} | {nv:8.3f} | {winner:>10s}")

    total = statWins + foundationWins
    print(f"\n통계 모델 승리: {statWins}/{total} ({100*statWins/total:.1f}%)")
    print(f"파운데이션 모델 승리: {foundationWins}/{total} ({100*foundationWins/total:.1f}%)")

    return bestStatMase


def analyzeByDomain():
    print("\n" + "=" * 110)
    print("도메인별 평균 MASE (short term)")
    print("=" * 110)

    models = {"Chronos-2": CHRONOS2, "TimesFM-2.5": TIMESFM25,
              "auto_theta": AUTO_THETA, "auto_ets": AUTO_ETS}
    shortConfigs = [k for k in CHRONOS2.keys() if k.endswith("/short")]

    domainScores = {m: defaultdict(list) for m in models}

    for ds in shortConfigs:
        domain = getDomain(ds.replace("/short", "").split("/")[0])
        for mName, mData in models.items():
            if ds in mData:
                domainScores[mName][domain].append(mData[ds])

    print(f"\n{'Domain':<14s} | {'Chronos2':>8s} | {'TimesFM':>8s} | {'Theta':>8s} | "
          f"{'ETS':>8s} | {'Gap(F-S)':>8s} | {'기회':<20s}")
    print("-" * 100)

    for domain in sorted(set(getDomain(ds.split("/")[0]) for ds in shortConfigs)):
        scores = {}
        for mName in models:
            vals = domainScores[mName].get(domain, [])
            scores[mName] = np.mean(vals) if vals else None

        bestFoundation = min(scores["Chronos-2"] or 99, scores["TimesFM-2.5"] or 99)
        bestStat = min(scores["auto_theta"] or 99, scores["auto_ets"] or 99)
        gap = bestStat - bestFoundation

        if gap < 0.05:
            opportunity = "격파 가능!"
        elif gap < 0.2:
            opportunity = "근접 — 개선 여지"
        elif gap < 0.5:
            opportunity = "도전적"
        else:
            opportunity = "큰 격차"

        print(f"  {domain:<12s} | {scores['Chronos-2']:8.3f} | {scores['TimesFM-2.5']:8.3f} | "
              f"{scores['auto_theta']:8.3f} | {scores['auto_ets']:8.3f} | "
              f"{gap:+8.3f} | {opportunity}")


def analyzeByFreq():
    print("\n" + "=" * 110)
    print("빈도별 평균 MASE (short term)")
    print("=" * 110)

    models = {"Chronos-2": CHRONOS2, "TimesFM-2.5": TIMESFM25,
              "auto_theta": AUTO_THETA, "auto_ets": AUTO_ETS}
    shortConfigs = [k for k in CHRONOS2.keys() if k.endswith("/short")]

    freqScores = {m: defaultdict(list) for m in models}
    for ds in shortConfigs:
        freq = getFreq(ds.replace("/short", ""))
        for mName, mData in models.items():
            if ds in mData:
                freqScores[mName][freq].append(mData[ds])

    print(f"\n{'Freq':<6s} | {'N':>3s} | {'Chronos2':>8s} | {'TimesFM':>8s} | {'Theta':>8s} | "
          f"{'ETS':>8s} | {'Gap':>8s} | {'판정':<16s}")
    print("-" * 90)

    for freq in sorted(set(getFreq(ds.replace("/short", "")) for ds in shortConfigs)):
        n = len(freqScores["Chronos-2"].get(freq, []))
        scores = {}
        for mName in models:
            vals = freqScores[mName].get(freq, [])
            scores[mName] = np.mean(vals) if vals else None

        if scores["Chronos-2"] is None:
            continue

        bestF = min(scores["Chronos-2"], scores["TimesFM-2.5"] or 99)
        bestS = min(scores["auto_theta"] or 99, scores["auto_ets"] or 99)
        gap = bestS - bestF

        if gap < 0:
            verdict = "통계 승리!"
        elif gap < 0.1:
            verdict = "접전"
        elif gap < 0.3:
            verdict = "파운데이션 우위"
        else:
            verdict = "파운데이션 압도"

        print(f"  {freq:<4s} | {n:>3d} | {scores['Chronos-2']:8.3f} | {scores['TimesFM-2.5']:8.3f} | "
              f"{scores['auto_theta']:8.3f} | {scores['auto_ets']:8.3f} | "
              f"{gap:+8.3f} | {verdict}")


def findOpportunities():
    print("\n" + "=" * 110)
    print("통계 모델이 파운데이션 모델보다 나은 구성 (공략 대상)")
    print("=" * 110)

    shortConfigs = [k for k in CHRONOS2.keys() if k.endswith("/short")]
    opportunities = []

    for ds in sorted(shortConfigs):
        c2 = CHRONOS2.get(ds, 99)
        tf = TIMESFM25.get(ds, 99)
        at = AUTO_THETA.get(ds, 99)
        ae = AUTO_ETS.get(ds, 99)

        bestFoundation = min(c2, tf)
        bestStat = min(at, ae)

        if bestStat < bestFoundation:
            improvement = (bestFoundation - bestStat) / bestFoundation * 100
            opportunities.append((ds, bestStat, bestFoundation, improvement))

    print(f"\n{'Dataset':<42s} | {'BestStat':>8s} | {'BestFound':>9s} | {'Improve':>8s}")
    print("-" * 80)

    for ds, bs, bf, imp in sorted(opportunities, key=lambda x: -x[3]):
        shortDs = ds.replace("/short", "")
        print(f"  {shortDs:<40s} | {bs:8.3f} | {bf:9.3f} | {imp:+7.1f}%")

    print(f"\n총 {len(opportunities)}개 구성에서 통계 모델 승리")

    domainOpp = defaultdict(int)
    for ds, _, _, _ in opportunities:
        domainOpp[getDomain(ds.split("/")[0])] += 1
    print("\n도메인별 승리 수:")
    for d in sorted(domainOpp.keys()):
        print(f"  {d}: {domainOpp[d]}개")


def computeOracleBlend():
    print("\n" + "=" * 110)
    print("Oracle 분석 — 구성별 최적 모델 선택 시 MASE")
    print("=" * 110)

    shortConfigs = [k for k in CHRONOS2.keys() if k.endswith("/short")]

    allModels = {
        "Chronos-2": CHRONOS2, "TimesFM-2.5": TIMESFM25,
        "auto_theta": AUTO_THETA, "auto_ets": AUTO_ETS,
    }

    oracleMase = []
    c2Mase = []
    thetaMase = []
    bestModelCounts = defaultdict(int)

    for ds in shortConfigs:
        bestMase = float("inf")
        bestModel = None
        for mName, mData in allModels.items():
            val = mData.get(ds, None)
            if val is not None and val < bestMase:
                bestMase = val
                bestModel = mName
        oracleMase.append(bestMase)
        bestModelCounts[bestModel] += 1
        c2Mase.append(CHRONOS2.get(ds, 99))
        thetaMase.append(AUTO_THETA.get(ds, 99))

    print(f"\n  Chronos-2 단독 평균 MASE:       {np.mean(c2Mase):.4f}")
    print(f"  auto_theta 단독 평균 MASE:      {np.mean(thetaMase):.4f}")
    print(f"  Oracle(4모델 중 최적) 평균 MASE: {np.mean(oracleMase):.4f}")
    print(f"  Oracle 개선율 vs Chronos-2:      {(np.mean(c2Mase)-np.mean(oracleMase))/np.mean(c2Mase)*100:.1f}%")

    print(f"\n  Oracle 최적 모델 분포:")
    for m, c in sorted(bestModelCounts.items(), key=lambda x: -x[1]):
        print(f"    {m}: {c}개 ({100*c/len(shortConfigs):.1f}%)")


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 110)
    print("GIFT-Eval Leaderboard Analysis — Phase 0, Experiment 003")
    print("=" * 110)

    bestStatMase = analyzeShortTerm()
    analyzeByDomain()
    analyzeByFreq()
    findOpportunities()
    computeOracleBlend()
