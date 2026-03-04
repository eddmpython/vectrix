"""
실험 ID: modelCreation/050
실험명: Powell Optimizer Full Pipeline — ETS Powell이 전체 forecast() OWA에 미치는 영향

목적:
- E049에서 Powell이 AutoETS 단독 OWA를 36% 개선 (2.031 → 1.291)
- 특히 Daily OWA 4.584 → 0.600 (87% 개선)
- 이 개선이 전체 forecast() 파이프라인(DOT-Hybrid + 모델 선택 + 앙상블)에서도 유효한지 검증
- 현재 DOT-Hybrid 기준선 AVG OWA 0.877

가설:
1. ETS optimizer를 Powell로 교체하면 전체 forecast() AVG OWA가 0.877에서 개선
2. 특히 Daily OWA 0.996 개선 기대 (ETS가 Daily에서 보조 모델로 사용되므로)
3. 속도 열화 없음 (E049에서 0.94x 확인)

방법:
1. M4 6개 그룹 × 100개 시리즈 = 600개 시리즈
2. 기존 forecast() vs Powell-patched forecast() 비교
3. sMAPE, MASE, OWA + 속도 측정

결과 (실험 후 작성):
- M4 600개 시리즈(그룹당 100개) 전체 forecast() 파이프라인 비교

  OWA 비교 (전체 파이프라인: DOT-Hybrid + 모델 선택 + 앙상블)
    baseline  AVG OWA=1.4563 (Y=1.266, Q=1.077, M=1.332, W=1.177, D=1.115, H=2.770)
    powell    AVG OWA=1.4491 (Y=1.269, Q=1.052, M=1.344, W=1.177, D=1.115, H=2.738)

  차이 (powell - baseline)
    Yearly    +0.003 (미미한 악화)
    Quarterly -0.026 (소폭 개선)
    Monthly   +0.012 (소폭 악화)
    Weekly     0.000 (동일)
    Daily      0.000 (동일)
    Hourly    -0.032 (소폭 개선)
    AVG       -0.007 (0.5% 개선)

  속도 비교
    baseline  총 766ms (per-series 평균 128ms)
    powell    총 857ms (per-series 평균 143ms)
    비율: 0.89x (Powell이 11% 느림)

  핵심 발견
  1. E049에서 AutoETS 단독 36% 개선은 전체 파이프라인에서 0.5% 개선으로 희석
  2. DOT-Hybrid가 최종 모델 선택에서 지배적 → ETS 개선이 최종 결과에 거의 반영 안 됨
  3. Weekly/Daily는 완전 동일 → 이 그룹에서 ETS가 아예 선택되지 않음
  4. Quarterly(-0.026)와 Hourly(-0.032)에서만 미미한 개선 → ETS가 보조 모델로 기여하는 그룹
  5. 속도는 오히려 11% 느려짐 — forecast() 레벨에서는 overhead가 다른 곳에서 발생

결론:
- **기각** — 전체 파이프라인에서 Powell 교체 효과가 미미 (AVG OWA -0.007, 0.5%)
- DOT-Hybrid가 최종 모델 선택을 지배하므로 ETS optimizer 개선이 희석됨
- 속도도 11% 악화 → 속도+정확도 모두에서 교체 정당성 없음
- E049의 AutoETS 단독 36% 개선은 "ETS만 쓸 때"의 이야기. 전체 파이프라인에서는 무의미
- **교훈**: 개별 모델 개선이 전체 파이프라인에 전파되려면, 해당 모델이 최종 선택에 기여하는 비율이 높아야 함
- DOT-Hybrid 자체의 속도 최적화가 더 효과적 방향

실험일: 2026-03-05
"""
import sys
import io
import os
import time

import numpy as np
import pandas as pd

if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')

    from vectrix.engine.ets import ETSModel
    from scipy.optimize import minimize

    origOptimize = ETSModel._optimizeParameters

    def powellOptimize(self, y):
        bounds = [(0.001, 0.999)]
        if self.trendType != 'N':
            bounds.append((0.001, 0.999))
        if self.seasonalType != 'N':
            bounds.append((0.001, 0.999))
        if self.damped:
            bounds.append((0.8, 0.999))

        def objective(params):
            return self._computeSSE(y, params)

        x0 = [0.3]
        if self.trendType != 'N':
            x0.append(0.1)
        if self.seasonalType != 'N':
            x0.append(0.1)
        if self.damped:
            x0.append(0.98)

        try:
            result = minimize(
                objective, x0, method='Powell',
                bounds=bounds, options={'maxiter': 50, 'ftol': 1e-4}
            )
            self._setParams(result.x)
        except Exception:
            pass

    from vectrix import Vectrix

    GROUPS = {
        'Yearly':    {'period': 1,  'horizon': 6},
        'Quarterly': {'period': 4,  'horizon': 8},
        'Monthly':   {'period': 12, 'horizon': 18},
        'Weekly':    {'period': 52, 'horizon': 13},
        'Daily':     {'period': 7,  'horizon': 14},
        'Hourly':    {'period': 24, 'horizon': 48},
    }

    N_SERIES = 100

    M4_SMAPE = {
        'Yearly': 13.528, 'Quarterly': 9.733, 'Monthly': 12.126,
        'Weekly': 7.817, 'Daily': 3.045, 'Hourly': 9.328,
    }
    M4_MASE = {
        'Yearly': 2.980, 'Quarterly': 1.111, 'Monthly': 0.836,
        'Weekly': 2.108, 'Daily': 3.278, 'Hourly': 0.821,
    }

    def smape(actual, predicted):
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        a, p = actual[mask], predicted[mask]
        denom = np.abs(a) + np.abs(p)
        denom = np.where(denom == 0, 1, denom)
        return np.mean(2 * np.abs(a - p) / denom) * 100

    def mase(actual, predicted, trainY, period):
        if period >= len(trainY):
            period = 1
        naiveDiffs = np.abs(trainY[period:] - trainY[:-period])
        scale = np.mean(naiveDiffs) if len(naiveDiffs) > 0 else 1.0
        if scale < 1e-10:
            scale = 1.0
        return np.mean(np.abs(actual - predicted)) / scale

    VARIANTS = ['baseline', 'powell']
    allResults = {v: {} for v in VARIANTS}
    allTimes = {v: {} for v in VARIANTS}

    for groupName, gCfg in GROUPS.items():
        period = gCfg['period']
        horizon = gCfg['horizon']

        trainFile = os.path.join(DATA_DIR, f'{groupName}-train.csv')
        testFile = os.path.join(DATA_DIR, f'{groupName}-test.csv')
        if not os.path.exists(trainFile):
            print(f"[SKIP] {groupName}")
            continue

        trainDf = pd.read_csv(trainFile)
        testDf = pd.read_csv(testFile)
        nRows = min(N_SERIES, len(trainDf))

        print(f"\n=== {groupName} (period={period}, h={horizon}, n={nRows}) ===")

        for variant in VARIANTS:
            if variant == 'powell':
                ETSModel._optimizeParameters = powellOptimize
            else:
                ETSModel._optimizeParameters = origOptimize

            varSmapes = []
            varMases = []
            varTimes = []

            for rowIdx in range(nRows):
                trainY = trainDf.iloc[rowIdx, 1:].dropna().values.astype(float)
                testY = testDf.iloc[rowIdx, 1:].dropna().values[:horizon].astype(float)

                if len(trainY) < 20 or len(testY) < horizon:
                    continue

                try:
                    dates = pd.date_range('2000-01-01', periods=len(trainY), freq='D')
                    df = pd.DataFrame({'date': dates, 'value': trainY})

                    vx = Vectrix()

                    t0 = time.perf_counter()
                    result = vx.forecast(df, dateCol='date', valueCol='value', steps=horizon)
                    fitTime = (time.perf_counter() - t0) * 1000

                    pred = np.array(result.predictions[:horizon])
                    pred = np.clip(pred, -1e15, 1e15)
                    pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)

                    s = smape(testY, pred)
                    m = mase(testY, pred, trainY, period)

                    varSmapes.append(s)
                    varMases.append(m)
                    varTimes.append(fitTime)
                except Exception as e:
                    varSmapes.append(np.nan)
                    varMases.append(np.nan)
                    varTimes.append(0)

            avgSmape = np.nanmean(varSmapes)
            avgMase = np.nanmean(varMases)
            avgTime = np.mean(varTimes)

            allResults[variant][groupName] = {
                'smape': avgSmape, 'mase': avgMase, 'time': avgTime,
            }
            allTimes[variant][groupName] = avgTime

            print(f"  {variant:12s}: sMAPE={avgSmape:.3f}  MASE={avgMase:.3f}  time={avgTime:.1f}ms")

    ETSModel._optimizeParameters = origOptimize

    print("\n\n" + "=" * 70)
    print("OWA COMPARISON — FULL PIPELINE")
    print("=" * 70)

    for variant in VARIANTS:
        groupOwas = []
        for groupName in GROUPS:
            if groupName not in allResults[variant]:
                continue
            r = allResults[variant][groupName]
            relSmape = r['smape'] / M4_SMAPE[groupName]
            relMase = r['mase'] / M4_MASE[groupName]
            owa = (relSmape + relMase) / 2
            groupOwas.append(owa)
            print(f"  {variant:12s} {groupName:10s}: OWA={owa:.4f}")
        if groupOwas:
            print(f"  {variant:12s} {'AVG':10s}: OWA={np.mean(groupOwas):.4f}")
        print()

    print("\n" + "=" * 70)
    print("SPEED COMPARISON (ms)")
    print("=" * 70)

    for groupName in GROUPS:
        times = []
        for variant in VARIANTS:
            if groupName in allTimes[variant]:
                times.append(f"{variant}={allTimes[variant][groupName]:.1f}")
        print(f"  {groupName:10s}: {', '.join(times)}")

    baseTotal = sum(allTimes['baseline'].get(g, 0) for g in GROUPS)
    powellTotal = sum(allTimes['powell'].get(g, 0) for g in GROUPS)
    if powellTotal > 0:
        print(f"\n  baseline vs powell: {baseTotal:.0f}ms -> {powellTotal:.0f}ms ({baseTotal/powellTotal:.2f}x)")

    print("\n\nDone.")
