"""
실험 ID: modelCreation/049
실험명: ETS Optimizer Speed — L-BFGS-B vs Nelder-Mead vs Powell

목적:
- cProfile에서 ETS _optimizeParameters → scipy L-BFGS-B → approx_derivative가
  forecast() 전체 시간의 ~70%를 차지 (59ms/90ms)
- L-BFGS-B는 gradient 기반이므로 수치 미분(approx_derivative)을 매번 호출
- Gradient-free 방법(Nelder-Mead, Powell)으로 전환하면 이 오버헤드 제거
- R forecast 패키지도 Nelder-Mead 사용 (ETS 표준)
- 정확도 손실 없이 속도 개선 가능한지 M4 600개 시리즈로 검증

가설:
1. Nelder-Mead/Powell은 L-BFGS-B 대비 ETS fit 속도 2x+ 개선
2. OWA 열화 < 0.3% (M4 기준선 대비)
3. AICc 기준 모델 선택 결과가 동일 (>90% 일치)

방법:
1. M4 6개 그룹 × 100개 시리즈 = 600개 시리즈
2. 3개 optimizer로 AutoETS fit → sMAPE, MASE, OWA 비교
3. 모델 선택 일치율 비교
4. fit 시간 비교

결과 (실험 후 작성):
- M4 600개 시리즈(그룹당 100개) 3개 optimizer로 AutoETS fit

  OWA 비교 (AutoETS 단독, DOT-Hybrid 아님)
    L-BFGS-B     AVG OWA=2.031 (Yearly=1.559, Quarterly=0.932, Monthly=1.277, Weekly=1.085, Daily=4.584, Hourly=2.750)
    Nelder-Mead  AVG OWA=폭발 (Yearly/Quarterly/Daily에서 MASE 10^11 수준)
    Powell       AVG OWA=1.291 (Yearly=1.178, Quarterly=0.749, Monthly=1.296, Weekly=1.061, Daily=0.600, Hourly=2.858)

  속도 비교 (평균 fit time per series)
    L-BFGS-B     총 223ms (Yearly=33, Quarterly=32, Monthly=33, Weekly=44, Daily=30, Hourly=51)
    Nelder-Mead  총 656ms (3x 느림!)
    Powell       총 236ms (0.94x, 거의 동일)

  모델 선택 일치율
    Nelder-Mead vs L-BFGS-B: 57.1%
    Powell vs L-BFGS-B: 59.8%

  핵심 발견
  1. Nelder-Mead는 bounds 미지원으로 파라미터가 [0,1] 범위 초과 → 폭발 (완전 부적합)
  2. Powell은 bounds 지원 + gradient-free, 정확도가 L-BFGS-B보다 오히려 좋음 (AVG OWA 36% 개선)
  3. 속도 차이는 미미 (0.94x) — maxiter=30에서 approx_derivative 호출 횟수가 적기 때문
  4. 모델 선택 일치율 60% — 다른 optimizer는 다른 최적점에 수렴, 다른 모델 선택
  5. Powell의 Daily OWA 0.600 vs L-BFGS-B 4.584 — L-BFGS-B가 Daily에서 local minima에 빠짐

결론:
- **가설 1 기각** — 속도 차이 미미 (0.94x). maxiter=30에서는 gradient 계산 비용이 작음
- **가설 3 기각** — 모델 선택 일치율 60% (90% 미달)
- **의외의 발견: Powell이 정확도에서 L-BFGS-B를 압도** (특히 Daily -87%)
- Nelder-Mead는 bounds가 없어 ETS에 완전 부적합 (R forecast 패키지가 Nelder-Mead 사용하는 것은 별도 penalty 처리 때문)
- **Powell 채택 검토 가능** — 속도 동등 + 정확도 개선. 단, 모델 선택이 달라지므로 전체 파이프라인 영향 확인 필요
- 속도 최적화 방향 전환 필요 — optimizer 교체로는 ETS 속도 개선 불가. maxiter 축소 또는 ETS 평가 모델 수 축소가 대안

실험일: 2026-03-05
"""
import sys
import io
import os
import time
import copy

import numpy as np
import pandas as pd

if __name__ == '__main__':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data', 'm4', 'm4', 'datasets')

    from vectrix.engine.ets import AutoETS, ETSModel
    from scipy.optimize import minimize

    GROUPS = {
        'Yearly':    {'period': 1,  'horizon': 6},
        'Quarterly': {'period': 4,  'horizon': 8},
        'Monthly':   {'period': 12, 'horizon': 18},
        'Weekly':    {'period': 52, 'horizon': 13},
        'Daily':     {'period': 7,  'horizon': 14},
        'Hourly':    {'period': 24, 'horizon': 48},
    }

    N_SERIES = 100

    METHODS = ['L-BFGS-B', 'Nelder-Mead', 'Powell']

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

    origOptimize = ETSModel._optimizeParameters

    def patchedOptimize(self, y, method='L-BFGS-B'):
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
            if method == 'L-BFGS-B':
                result = minimize(
                    objective, x0, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 30, 'ftol': 1e-4}
                )
            elif method == 'Nelder-Mead':
                result = minimize(
                    objective, x0, method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
                )
            elif method == 'Powell':
                result = minimize(
                    objective, x0, method='Powell',
                    bounds=bounds, options={'maxiter': 50, 'ftol': 1e-4}
                )
            self._setParams(result.x)
        except Exception:
            pass

    currentMethod = 'L-BFGS-B'

    def monkeyOptimize(self, y):
        patchedOptimize(self, y, method=currentMethod)

    allResults = {m: {} for m in METHODS}
    allTimes = {m: {} for m in METHODS}

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

        for method in METHODS:
            currentMethod = method
            ETSModel._optimizeParameters = monkeyOptimize

            methodSmapes = []
            methodMases = []
            methodTimes = []
            methodModels = []

            for rowIdx in range(nRows):
                trainY = trainDf.iloc[rowIdx, 1:].dropna().values.astype(float)
                testY = testDf.iloc[rowIdx, 1:].dropna().values[:horizon].astype(float)

                if len(trainY) < 20 or len(testY) < horizon:
                    continue

                try:
                    auto = AutoETS(period=period)

                    t0 = time.perf_counter()
                    bestModel = auto.fit(trainY)
                    fitTime = (time.perf_counter() - t0) * 1000

                    pred, _, _ = bestModel.predict(horizon)
                    pred = np.clip(pred, -1e15, 1e15)
                    pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)

                    s = smape(testY, pred)
                    m = mase(testY, pred, trainY, period)

                    methodSmapes.append(s)
                    methodMases.append(m)
                    methodTimes.append(fitTime)
                    methodModels.append(f"{bestModel.errorType}{bestModel.trendType}{bestModel.seasonalType}")
                except Exception:
                    methodSmapes.append(np.nan)
                    methodMases.append(np.nan)
                    methodTimes.append(0)
                    methodModels.append('FAIL')

            avgSmape = np.nanmean(methodSmapes)
            avgMase = np.nanmean(methodMases)
            avgTime = np.mean(methodTimes)

            allResults[method][groupName] = {
                'smape': avgSmape, 'mase': avgMase,
                'time': avgTime, 'models': methodModels,
            }
            allTimes[method][groupName] = avgTime

            print(f"  {method:12s}: sMAPE={avgSmape:.3f}  MASE={avgMase:.3f}  time={avgTime:.1f}ms")

    ETSModel._optimizeParameters = origOptimize

    print("\n\n" + "=" * 70)
    print("OWA COMPARISON")
    print("=" * 70)

    for method in METHODS:
        groupOwas = []
        for groupName in GROUPS:
            if groupName not in allResults[method]:
                continue
            r = allResults[method][groupName]
            relSmape = r['smape'] / M4_SMAPE[groupName]
            relMase = r['mase'] / M4_MASE[groupName]
            owa = (relSmape + relMase) / 2
            groupOwas.append(owa)
            print(f"  {method:12s} {groupName:10s}: OWA={owa:.4f}")
        if groupOwas:
            print(f"  {method:12s} {'AVG':10s}: OWA={np.mean(groupOwas):.4f}")
        print()

    print("\n" + "=" * 70)
    print("SPEED COMPARISON (ms)")
    print("=" * 70)

    for groupName in GROUPS:
        times = []
        for method in METHODS:
            if groupName in allTimes[method]:
                times.append(f"{method}={allTimes[method][groupName]:.1f}")
        print(f"  {groupName:10s}: {', '.join(times)}")

    baseTime = sum(allTimes['L-BFGS-B'].get(g, 0) for g in GROUPS)
    for method in METHODS[1:]:
        altTime = sum(allTimes[method].get(g, 0) for g in GROUPS)
        if altTime > 0:
            print(f"\n  {method} vs L-BFGS-B: {baseTime:.0f}ms -> {altTime:.0f}ms ({baseTime/altTime:.2f}x)")

    print("\n\n" + "=" * 70)
    print("MODEL SELECTION AGREEMENT")
    print("=" * 70)

    for method in METHODS[1:]:
        total = 0
        agree = 0
        for groupName in GROUPS:
            if groupName not in allResults['L-BFGS-B'] or groupName not in allResults[method]:
                continue
            baseModels = allResults['L-BFGS-B'][groupName]['models']
            altModels = allResults[method][groupName]['models']
            for bm, am in zip(baseModels, altModels):
                total += 1
                if bm == am:
                    agree += 1
        if total > 0:
            print(f"  {method} vs L-BFGS-B: {agree}/{total} = {agree/total*100:.1f}% agreement")

    print("\n\nDone.")
