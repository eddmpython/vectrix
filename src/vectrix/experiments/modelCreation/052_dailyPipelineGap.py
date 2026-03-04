"""
실험 ID: modelCreation/052
실험명: Daily Pipeline Gap — 파이프라인 각 단계별 OWA 분해

목적:
- E051에서 DOT 단독 OWA=0.605인데 forecast() OWA=0.908 (50% 악화)
- 파이프라인의 어느 단계에서 성능이 악화되는지 정확히 분해
- 단계: (A) 전체 데이터 fit → (B) 80/20 split + 선택 → (C) refit → (D) 앙상블 → (E) flat defense

가설:
1. 80/20 split으로 인한 학습 데이터 감소가 주요 원인
2. refit 과정에서 config 불일치 가능
3. 앙상블이 최선 모델을 희석

방법:
1. M4 Daily 200개 시리즈
2. 5가지 시나리오 비교
   A) 전체 데이터 fit (E051 방식) — oracle ceiling
   B) 80% fit → test에서 평가 (split 효과)
   C) 80% fit → 최선 모델 → 전체 refit (refit 효과)
   D) forecast() 그대로 (full pipeline)
   E) forecast() with ensemble='best' (앙상블 제거)

결과 (실험 후 작성):
- M4 Daily 200개 시리즈 파이프라인 분해 분석

  시나리오별 OWA
    A_fullFit_oracle       = 0.709  (4모델 best, 전체 데이터)
    B_fullFit_dot          = 0.820  (DOT, 전체 데이터, period=7)
    C_splitFit_dot         = 4.865  (DOT, 80% 데이터) — 폭발!
    D_splitFit_bestOf4     = 4.532  (4모델 best, 80% 데이터)
    E_splitFit_refit_dot   = 0.820  (80% → refit full) — 완전 복구
    G_pipeline_default     = 1.307  (forecast() 전체)

  핵심 발견: pipeline detected period 확인 결과
    대부분의 Daily 시리즈에서 FFT가 period=53,72,144,168 등을 감지
    이 잘못된 period가 DOT-Hybrid에 전달되어 성능 붕괴

결론:
- **근본 원인 발견**: FFT period 감지가 Daily에서 스퓨리어스 period 반환
- autoAnalyzer._detectSeasonalPeriods()의 FFT 로직이 n/i 계산으로 큰 period를 감지
- DOT full fit (period=7) = 0.820이지만, 파이프라인이 period=53+ 전달 → 1.307으로 악화
- E053에서 수정 실험 진행 → basePeriod_only 전략이 모든 그룹에서 개선 확인

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

    from vectrix.engine.registry import createModel
    from vectrix import Vectrix

    PERIOD = 7
    HORIZON = 14
    N_SERIES = 200

    M4_SMAPE_DAILY = 3.045
    M4_MASE_DAILY = 3.278

    trainFile = os.path.join(DATA_DIR, 'Daily-train.csv')
    testFile = os.path.join(DATA_DIR, 'Daily-test.csv')

    trainDf = pd.read_csv(trainFile)
    testDf = pd.read_csv(testFile)
    nRows = min(N_SERIES, len(trainDf))

    MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']

    def smape(actual, predicted):
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        a, p = actual[mask], predicted[mask]
        if len(a) == 0:
            return np.nan
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

    scenarios = {
        'A_fullFit_oracle': {'smapes': [], 'mases': []},
        'B_fullFit_dot': {'smapes': [], 'mases': []},
        'C_splitFit_dot': {'smapes': [], 'mases': []},
        'D_splitFit_bestOf4': {'smapes': [], 'mases': []},
        'E_splitFit_refit_dot': {'smapes': [], 'mases': []},
        'F_pipeline_ensembleBest': {'smapes': [], 'mases': []},
        'G_pipeline_default': {'smapes': [], 'mases': []},
    }

    print(f"=== Daily Pipeline Gap Analysis (n={nRows}) ===\n")

    for rowIdx in range(nRows):
        trainY = trainDf.iloc[rowIdx, 1:].dropna().values.astype(float)
        testY = testDf.iloc[rowIdx, 1:].dropna().values[:HORIZON].astype(float)

        if len(trainY) < 30 or len(testY) < HORIZON:
            continue

        n = len(trainY)
        splitIdx = int(n * 0.8)
        train80 = trainY[:splitIdx]
        val20 = trainY[splitIdx:]

        allModelPreds = {}
        allModelSmapes = {}

        for modelId in MODELS:
            try:
                model = createModel(modelId, PERIOD)
                model.fit(trainY)
                pred, _, _ = model.predict(HORIZON)
                pred = np.clip(pred, -1e15, 1e15)
                pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)
                allModelPreds[modelId] = pred
                allModelSmapes[modelId] = smape(testY, pred)
            except Exception:
                pass

        if allModelPreds:
            bestFull = min(allModelPreds, key=lambda m: allModelSmapes[m])
            bestPred = allModelPreds[bestFull]
            s = smape(testY, bestPred)
            m = mase(testY, bestPred, trainY, PERIOD)
            scenarios['A_fullFit_oracle']['smapes'].append(s)
            scenarios['A_fullFit_oracle']['mases'].append(m)

        if 'dot' in allModelPreds:
            s = smape(testY, allModelPreds['dot'])
            m = mase(testY, allModelPreds['dot'], trainY, PERIOD)
            scenarios['B_fullFit_dot']['smapes'].append(s)
            scenarios['B_fullFit_dot']['mases'].append(m)

        splitPreds = {}
        splitSmapes = {}
        for modelId in MODELS:
            try:
                model = createModel(modelId, PERIOD)
                model.fit(train80)
                pred80, _, _ = model.predict(HORIZON)
                pred80 = np.clip(pred80, -1e15, 1e15)
                pred80 = np.where(np.isnan(pred80), np.nanmean(train80), pred80)
                splitPreds[modelId] = pred80
                splitSmapes[modelId] = smape(testY, pred80)
            except Exception:
                pass

        if 'dot' in splitPreds:
            s = smape(testY, splitPreds['dot'])
            m = mase(testY, splitPreds['dot'], trainY, PERIOD)
            scenarios['C_splitFit_dot']['smapes'].append(s)
            scenarios['C_splitFit_dot']['mases'].append(m)

        if splitSmapes:
            bestSplit = min(splitSmapes, key=lambda m: splitSmapes[m])

            bestSplitPred = splitPreds[bestSplit]
            s = smape(testY, bestSplitPred)
            m = mase(testY, bestSplitPred, trainY, PERIOD)
            scenarios['D_splitFit_bestOf4']['smapes'].append(s)
            scenarios['D_splitFit_bestOf4']['mases'].append(m)

        try:
            dotModel = createModel('dot', PERIOD)
            dotModel.fit(train80)
            if hasattr(dotModel, 'refit'):
                dotModel.refit(trainY)
            else:
                dotModel.fit(trainY)
            refitPred, _, _ = dotModel.predict(HORIZON)
            refitPred = np.clip(refitPred, -1e15, 1e15)
            refitPred = np.where(np.isnan(refitPred), np.nanmean(trainY), refitPred)
            s = smape(testY, refitPred)
            m = mase(testY, refitPred, trainY, PERIOD)
            scenarios['E_splitFit_refit_dot']['smapes'].append(s)
            scenarios['E_splitFit_refit_dot']['mases'].append(m)
        except Exception:
            pass

        try:
            dates = pd.date_range('2000-01-01', periods=n, freq='D')
            df = pd.DataFrame({'date': dates, 'value': trainY})
            vx = Vectrix()
            result = vx.forecast(df, dateCol='date', valueCol='value', steps=HORIZON, ensemble='best')
            pred = np.array(result.predictions[:HORIZON])
            pred = np.clip(pred, -1e15, 1e15)
            pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)
            s = smape(testY, pred)
            m = mase(testY, pred, trainY, PERIOD)
            scenarios['F_pipeline_ensembleBest']['smapes'].append(s)
            scenarios['F_pipeline_ensembleBest']['mases'].append(m)
        except Exception:
            scenarios['F_pipeline_ensembleBest']['smapes'].append(np.nan)
            scenarios['F_pipeline_ensembleBest']['mases'].append(np.nan)

        try:
            dates = pd.date_range('2000-01-01', periods=n, freq='D')
            df = pd.DataFrame({'date': dates, 'value': trainY})
            vx = Vectrix()
            result = vx.forecast(df, dateCol='date', valueCol='value', steps=HORIZON)
            pred = np.array(result.predictions[:HORIZON])
            pred = np.clip(pred, -1e15, 1e15)
            pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)
            s = smape(testY, pred)
            m = mase(testY, pred, trainY, PERIOD)
            scenarios['G_pipeline_default']['smapes'].append(s)
            scenarios['G_pipeline_default']['mases'].append(m)
        except Exception:
            scenarios['G_pipeline_default']['smapes'].append(np.nan)
            scenarios['G_pipeline_default']['mases'].append(np.nan)

        if (rowIdx + 1) % 50 == 0:
            print(f"  Processed {rowIdx + 1}/{nRows}...")

    print(f"\n{'='*70}")
    print("SCENARIO COMPARISON — Daily OWA")
    print(f"{'='*70}\n")

    print(f"  {'Scenario':<35s} {'sMAPE':>8s} {'MASE':>8s} {'OWA':>8s}")
    print("  " + "-" * 65)

    for name, data in scenarios.items():
        if not data['smapes']:
            continue
        avgS = np.nanmean(data['smapes'])
        avgM = np.nanmean(data['mases'])
        owa = ((avgS / M4_SMAPE_DAILY) + (avgM / M4_MASE_DAILY)) / 2
        print(f"  {name:<35s} {avgS:8.3f} {avgM:8.3f} {owa:8.4f}")

    print(f"\n{'='*70}")
    print("GAP ATTRIBUTION")
    print(f"{'='*70}\n")

    def getOwa(name):
        data = scenarios[name]
        if not data['smapes']:
            return np.nan
        avgS = np.nanmean(data['smapes'])
        avgM = np.nanmean(data['mases'])
        return ((avgS / M4_SMAPE_DAILY) + (avgM / M4_MASE_DAILY)) / 2

    owaA = getOwa('A_fullFit_oracle')
    owaB = getOwa('B_fullFit_dot')
    owaC = getOwa('C_splitFit_dot')
    owaD = getOwa('D_splitFit_bestOf4')
    owaE = getOwa('E_splitFit_refit_dot')
    owaF = getOwa('F_pipeline_ensembleBest')
    owaG = getOwa('G_pipeline_default')

    print(f"  Oracle ceiling (A)                  = {owaA:.4f}")
    print(f"  DOT full-data fit (B)               = {owaB:.4f}")
    print(f"  DOT 80% fit (C)                     = {owaC:.4f}")
    print(f"    → Split cost (C-B)                = {owaC - owaB:+.4f}")
    print(f"  Best-of-4 80% fit (D)               = {owaD:.4f}")
    print(f"    → Model selection gain (D-C)      = {owaD - owaC:+.4f}")
    print(f"  DOT 80% fit → refit full (E)        = {owaE:.4f}")
    print(f"    → Refit recovery (E-C)            = {owaE - owaC:+.4f}")
    print(f"  Pipeline ensemble=best (F)          = {owaF:.4f}")
    print(f"    → Pipeline overhead (F-E)         = {owaF - owaE:+.4f}")
    print(f"  Pipeline default ensemble (G)       = {owaG:.4f}")
    print(f"    → Ensemble effect (G-F)           = {owaG - owaF:+.4f}")
    print(f"  Total gap (G-B)                     = {owaG - owaB:+.4f}")

    print("\n\nDone.")
