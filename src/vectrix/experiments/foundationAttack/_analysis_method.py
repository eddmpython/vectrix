"""임시 분석: 구체적 방법론 도출을 위한 수치 분석"""
import sys
import io
import numpy as np
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

CHRONOS2 = {'bitbrains_fast_storage/5T': 0.6564, 'bitbrains_fast_storage/H': 0.931, 'bitbrains_rnd/5T': 1.6181, 'bitbrains_rnd/H': 5.8047, 'bizitobs_application/10S': 1.0039, 'bizitobs_l2c/5T': 0.2665, 'bizitobs_l2c/H': 0.4126, 'bizitobs_service/10S': 0.7169, 'car_parts/M': 0.836, 'covid_deaths/D': 32.5392, 'electricity/15T': 0.9223, 'electricity/D': 1.4149, 'electricity/H': 0.9779, 'electricity/W': 1.3851, 'ett1/15T': 0.6869, 'ett1/D': 1.6403, 'ett1/H': 0.7868, 'ett1/W': 1.5901, 'ett2/15T': 0.7262, 'ett2/D': 1.2908, 'ett2/H': 0.7363, 'ett2/W': 0.7716, 'hierarchical_sales/D': 0.744, 'hierarchical_sales/W': 0.7093, 'hospital/M': 0.7399, 'jena_weather/10T': 0.265, 'jena_weather/D': 1.0823, 'jena_weather/H': 0.5226, 'kdd_cup_2018/D': 1.1978, 'kdd_cup_2018/H': 0.9467, 'loop_seattle/5T': 0.5407, 'loop_seattle/D': 0.8878, 'loop_seattle/H': 0.825, 'm_dense/D': 0.6151, 'm_dense/H': 0.7817, 'm4_daily/D': 3.4193, 'm4_hourly/H': 0.7884, 'm4_monthly/M': 0.9032, 'm4_quarterly/Q': 1.1543, 'm4_weekly/W': 2.0263, 'm4_yearly/A': 3.2362, 'restaurant/D': 0.6777, 'saugeen/D': 2.9834, 'saugeen/M': 0.7235, 'saugeen/W': 1.1876, 'solar/10T': 0.7784, 'solar/D': 0.961, 'solar/H': 0.9845, 'solar/W': 1.0123, 'sz_taxi/15T': 0.5429, 'sz_taxi/H': 0.5603, 'temperature_rain/D': 1.3102, 'us_births/D': 0.3266, 'us_births/M': 0.6328, 'us_births/W': 0.979}
TIMESFM25 = {'bitbrains_fast_storage/5T': 0.7111, 'bitbrains_fast_storage/H': 1.0595, 'bitbrains_rnd/5T': 1.6571, 'bitbrains_rnd/H': 5.8512, 'bizitobs_application/10S': 1.0241, 'bizitobs_l2c/5T': 0.2741, 'bizitobs_l2c/H': 0.4261, 'bizitobs_service/10S': 0.7261, 'car_parts/M': 0.8384, 'covid_deaths/D': 36.9065, 'electricity/15T': 1.0814, 'electricity/D': 1.3764, 'electricity/H': 0.9075, 'electricity/W': 1.3968, 'ett1/15T': 0.6939, 'ett1/D': 1.7128, 'ett1/H': 0.8387, 'ett1/W': 1.4441, 'ett2/15T': 0.7237, 'ett2/D': 1.2743, 'ett2/H': 0.7392, 'ett2/W': 0.8412, 'hierarchical_sales/D': 0.745, 'hierarchical_sales/W': 0.7191, 'hospital/M': 0.759, 'jena_weather/10T': 0.276, 'jena_weather/D': 0.9881, 'jena_weather/H': 0.5257, 'kdd_cup_2018/D': 1.1915, 'kdd_cup_2018/H': 0.9326, 'loop_seattle/5T': 0.5676, 'loop_seattle/D': 0.8627, 'loop_seattle/H': 0.725, 'm_dense/D': 0.6615, 'm_dense/H': 0.7688, 'm4_daily/D': 3.2974, 'm4_hourly/H': 0.7336, 'm4_monthly/M': 0.9462, 'm4_quarterly/Q': 1.205, 'm4_weekly/W': 2.002, 'm4_yearly/A': 3.5746, 'restaurant/D': 0.6818, 'saugeen/D': 2.715, 'saugeen/M': 0.7144, 'saugeen/W': 1.2478, 'solar/10T': 1.0857, 'solar/D': 0.9852, 'solar/H': 0.8976, 'solar/W': 1.2792, 'sz_taxi/15T': 0.5444, 'sz_taxi/H': 0.5633, 'temperature_rain/D': 1.3471, 'us_births/D': 0.343, 'us_births/M': 0.6083, 'us_births/W': 1.0866}
AUTO_THETA = {'bitbrains_fast_storage/5T': 1.15, 'bitbrains_fast_storage/H': 1.35, 'bitbrains_rnd/5T': 2.07, 'bitbrains_rnd/H': 5.75, 'bizitobs_application/10S': 1.11, 'bizitobs_l2c/5T': 0.292, 'bizitobs_l2c/H': 1.19, 'bizitobs_service/10S': 0.791, 'car_parts/M': 1.23, 'covid_deaths/D': 45.4, 'electricity/15T': 1.35, 'electricity/D': 1.88, 'electricity/H': 1.74, 'electricity/W': 2.14, 'ett1/15T': 0.863, 'ett1/D': 1.75, 'ett1/H': 1.28, 'ett1/W': 1.89, 'ett2/15T': 0.832, 'ett2/D': 1.85, 'ett2/H': 1.02, 'ett2/W': 1.41, 'hierarchical_sales/D': 0.932, 'hierarchical_sales/W': 0.849, 'hospital/M': 0.761, 'jena_weather/10T': 0.368, 'jena_weather/D': 1.6, 'jena_weather/H': 0.878, 'kdd_cup_2018/D': 1.38, 'kdd_cup_2018/H': 1.27, 'loop_seattle/5T': 0.78, 'loop_seattle/D': 1.39, 'loop_seattle/H': 1.4, 'm_dense/D': 1.22, 'm_dense/H': 1.69, 'm4_daily/D': 3.34, 'm4_hourly/H': 2.46, 'm4_monthly/M': 0.966, 'm4_quarterly/Q': 1.19, 'm4_weekly/W': 2.66, 'm4_yearly/A': 3.11, 'restaurant/D': 0.843, 'saugeen/D': 3.6, 'saugeen/M': 0.912, 'saugeen/W': 2.12, 'solar/10T': 1.8, 'solar/D': 1.05, 'solar/H': 2.05, 'solar/W': 1.15, 'sz_taxi/15T': 0.649, 'sz_taxi/H': 0.691, 'temperature_rain/D': 1.93, 'us_births/D': 1.63, 'us_births/M': 0.883, 'us_births/W': 1.49}
AUTO_ETS = {'bitbrains_fast_storage/5T': 1.14, 'bitbrains_fast_storage/H': 1.32, 'bitbrains_rnd/5T': 1.97, 'bitbrains_rnd/H': 6.04, 'bizitobs_application/10S': 5.09, 'bizitobs_l2c/5T': 0.272, 'bizitobs_l2c/H': 1.11, 'bizitobs_service/10S': 2.47, 'car_parts/M': 1.22, 'covid_deaths/D': 28.4, 'electricity/15T': 2.43, 'electricity/D': 1.93, 'electricity/H': 1.36, 'electricity/W': 2.14, 'ett1/15T': 1.91, 'ett1/D': 1.69, 'ett1/H': 1.94, 'ett1/W': 1.66, 'ett2/15T': 1.18, 'ett2/D': 1.42, 'ett2/H': 1.03, 'ett2/W': 0.874, 'hierarchical_sales/D': 0.908, 'hierarchical_sales/W': 0.916, 'hospital/M': 0.77, 'jena_weather/10T': 0.341, 'jena_weather/D': 2.0, 'jena_weather/H': 0.744, 'kdd_cup_2018/D': 1.47, 'kdd_cup_2018/H': 1.43, 'loop_seattle/5T': 0.835, 'loop_seattle/D': 1.25, 'loop_seattle/H': 1.55, 'm_dense/D': 1.19, 'm_dense/H': 1.46, 'm4_daily/D': 3.24, 'm4_hourly/H': 1.61, 'm4_monthly/M': 0.964, 'm4_quarterly/Q': 1.16, 'm4_weekly/W': 2.55, 'm4_yearly/A': 3.08, 'restaurant/D': 0.861, 'saugeen/D': 3.9, 'saugeen/M': 0.725, 'saugeen/W': 1.99, 'solar/10T': 1.22, 'solar/D': 1.02, 'solar/H': 1.8, 'solar/W': 1.0, 'sz_taxi/15T': 0.696, 'sz_taxi/H': 0.852, 'temperature_rain/D': 1.97, 'us_births/D': 1.6, 'us_births/M': 0.588, 'us_births/W': 1.49}

configs = sorted(CHRONOS2.keys())

print("=" * 80)
print("방법론 도출을 위한 정밀 분석")
print("=" * 80)

print("\n--- 1. 전체 구성 평균 ---")
c2_avg = np.mean([CHRONOS2[ds] for ds in configs])
tf_avg = np.mean([TIMESFM25[ds] for ds in configs])
at_avg = np.mean([AUTO_THETA[ds] for ds in configs])
ae_avg = np.mean([AUTO_ETS[ds] for ds in configs])

print(f"  Chronos-2:   {c2_avg:.3f}")
print(f"  TimesFM-2.5: {tf_avg:.3f}")
print(f"  auto_theta:  {at_avg:.3f}")
print(f"  auto_ets:    {ae_avg:.3f}")

best_found = [min(CHRONOS2[ds], TIMESFM25[ds]) for ds in configs]
best_stat = [min(AUTO_THETA[ds], AUTO_ETS[ds]) for ds in configs]
print(f"\n  best_foundation 평균: {np.mean(best_found):.3f}")
print(f"  best_stat 평균:       {np.mean(best_stat):.3f}")
print(f"  격차:                 {np.mean(best_stat) - np.mean(best_found):.3f}")

print("\n--- 2. 구성별 Oracle(전체 4모델) ---")
oracle_all = [min(CHRONOS2[ds], TIMESFM25[ds], AUTO_THETA[ds], AUTO_ETS[ds]) for ds in configs]
print(f"  Oracle(4모델) 평균:   {np.mean(oracle_all):.3f}")
print(f"  vs Chronos-2:         {(c2_avg - np.mean(oracle_all))/c2_avg*100:+.1f}%")

oracle_stat = [min(AUTO_THETA[ds], AUTO_ETS[ds]) for ds in configs]
oracle_stat6 = []
for ds in configs:
    oracle_stat6.append(min(AUTO_THETA[ds], AUTO_ETS[ds]))

print("\n--- 3. 빈도별 격차 (핵심) ---")
freq_data = defaultdict(lambda: {'c2': [], 'tf': [], 'bf': [], 'bs': [], 'oracle': []})
for ds in configs:
    parts = ds.split('/')
    freq = parts[1] if len(parts) >= 2 else '?'
    c2 = CHRONOS2[ds]
    tf = TIMESFM25[ds]
    at = AUTO_THETA[ds]
    ae = AUTO_ETS[ds]
    bf = min(c2, tf)
    bs = min(at, ae)
    oa = min(c2, tf, at, ae)
    freq_data[freq]['c2'].append(c2)
    freq_data[freq]['tf'].append(tf)
    freq_data[freq]['bf'].append(bf)
    freq_data[freq]['bs'].append(bs)
    freq_data[freq]['oracle'].append(oa)

print(f"\n  {'Freq':<5s} {'n':>3s} {'BestFound':>10s} {'BestStat':>10s} {'Gap':>8s} {'Oracle4':>10s} {'O4vsC2':>8s} {'판정':<12s}")
print("  " + "-" * 72)

freq_order = ['A', 'Q', 'M', 'W', 'D', 'H', '15T', '10T', '5T', '10S']
for freq in freq_order:
    if freq not in freq_data:
        continue
    d = freq_data[freq]
    bf_avg = np.mean(d['bf'])
    bs_avg = np.mean(d['bs'])
    oa_avg = np.mean(d['oracle'])
    c2_f = np.mean(d['c2'])
    gap = (bf_avg - bs_avg) / bf_avg * 100
    o4_gap = (c2_f - oa_avg) / c2_f * 100
    n = len(d['c2'])

    if gap < 0:
        verdict = "통계 승리!"
    elif gap < 5:
        verdict = "접전"
    elif gap < 15:
        verdict = "F 우위"
    else:
        verdict = "F 압도"

    print(f"  {freq:<5s} {n:>3d} {bf_avg:>10.3f} {bs_avg:>10.3f} {gap:>+8.1f}% {oa_avg:>10.3f} {o4_gap:>+8.1f}% {verdict}")

print("\n--- 4. 통계 모델이 이기는 구성 상세 ---")
stat_wins = []
for ds in configs:
    bf = min(CHRONOS2[ds], TIMESFM25[ds])
    bs = min(AUTO_THETA[ds], AUTO_ETS[ds])
    if bs < bf:
        gap = (bf - bs) / bf * 100
        stat_wins.append((ds, bs, bf, gap))

for ds, bs, bf, gap in sorted(stat_wins, key=lambda x: -x[3]):
    freq = ds.split('/')[1] if '/' in ds else '?'
    best_model = 'ets' if AUTO_ETS[ds] < AUTO_THETA[ds] else 'theta'
    print(f"  {ds:<35s} stat={bs:.3f} ({best_model})  found={bf:.3f}  {gap:+.1f}%")

print(f"\n  통계 승리: {len(stat_wins)}/55 = {len(stat_wins)/55*100:.1f}%")

print("\n--- 5. 핵심 질문: Oracle 캡처율별 Chronos-2 대비 결과 ---")
print("  (이것이 방법론의 실현 가능성을 결정한다)")
print()
gap_total = c2_avg - np.mean(oracle_all)
print(f"  Chronos-2 평균:    {c2_avg:.3f}")
print(f"  Oracle(4모델):     {np.mean(oracle_all):.3f}")
print(f"  Oracle gap:        {gap_total:.3f} ({gap_total/c2_avg*100:.1f}%)")
print()
for capture in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    achieved = c2_avg - gap_total * capture
    vs_c2 = (c2_avg - achieved) / c2_avg * 100
    better = "Chronos-2 격파!" if achieved < c2_avg else ""
    print(f"  캡처 {capture:.0%}: MASE {achieved:.3f} (vs C2: {vs_c2:+.1f}%) {better}")

print("\n--- 6. 현실적 시나리오 ---")
print()

print("  시나리오 A: 현재 GBT (E014 실예측 +4.1%)")
gbt_real = c2_avg * (1 - 0.041)
print(f"    우리 MASE: ~{gbt_real:.3f}")
print(f"    vs Chronos-2: {(c2_avg-gbt_real)/c2_avg*100:+.1f}%")
print(f"    → Chronos-2보다 좋지만, 이건 DOT 대비 +4.1%이지 C2 대비가 아님!")
print()

print("  시나리오 B: DOT 대비 +4.1%를 MASE로 환산")
dot_gift_mase = 1.804
dot_improved = dot_gift_mase * (1 - 0.041)
print(f"    DOT GIFT-Eval MASE: {dot_gift_mase:.3f}")
print(f"    +4.1% 개선:        {dot_improved:.3f}")
print(f"    vs Chronos-2({c2_avg:.3f}): {'이김' if dot_improved < c2_avg else '짐'}")
print()

print("  시나리오 C: E013 CV 기준 +15.3%")
dot_cv = dot_gift_mase * (1 - 0.153)
print(f"    +15.3% 개선:       {dot_cv:.3f}")
print(f"    vs Chronos-2({c2_avg:.3f}): {'이김' if dot_cv < c2_avg else '짐'}")
print()

print("  시나리오 D: E014 실예측 결과 직접 사용")
e014_selected_mase = 1.795
print(f"    E014 실측 MASE:    {e014_selected_mase:.3f}")
print(f"    vs Chronos-2({c2_avg:.3f}): {'이김' if e014_selected_mase < c2_avg else '짐'}")

print("\n--- 7. 결론: 어디서 이기고 어디서 지는가? ---")
print()

stat_win_freqs = defaultdict(int)
stat_loss_freqs = defaultdict(int)
for ds in configs:
    freq = ds.split('/')[1] if '/' in ds else '?'
    bf = min(CHRONOS2[ds], TIMESFM25[ds])
    bs = min(AUTO_THETA[ds], AUTO_ETS[ds])
    if bs < bf:
        stat_win_freqs[freq] += 1
    else:
        stat_loss_freqs[freq] += 1

print(f"  {'Freq':<5s} {'승리':>4s} {'패배':>4s} {'승률':>6s}")
print("  " + "-" * 25)
for freq in freq_order:
    if freq not in freq_data:
        continue
    w = stat_win_freqs.get(freq, 0)
    l = stat_loss_freqs.get(freq, 0)
    total = w + l
    print(f"  {freq:<5s} {w:>4d} {l:>4d} {w/total*100:>5.0f}%")
