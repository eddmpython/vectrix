"""
실험 ID: modelCreation/051
실험명: Daily OWA 0.996 원인 진단 — 모델별 성능 분해 + 실패 패턴 분석

목적:
- Daily OWA 0.996은 Naive2 수준으로, 30+ 모델이 사실상 무의미
- 어떤 모델이 선택되고 있는지, 왜 Naive2를 이기지 못하는지 진단
- 모델별/시리즈별 세밀한 분석으로 개선 방향 도출

가설:
1. DOT-Hybrid가 Daily에서도 지배적으로 선택되지만, period=7에서 계절성 포착이 불충분
2. 특정 시리즈 유형(고노이즈, 트렌드 없음 등)에서 집중적으로 실패
3. Naive2 대비 승률이 50% 미만인 모델이 존재

방법:
1. M4 Daily 전체 (4227개 시리즈) 중 500개 시리즈 분석
2. 각 모델별 개별 sMAPE/MASE + forecast() 최종 결과 비교
3. Naive2 vs 각 모델 per-series 승률
4. 실패 시리즈의 데이터 특성 분석 (길이, 변동성, 트렌드, 계절성)
5. 모델 선택 분포 확인

결과 (실험 후 작성):
- M4 Daily 500개 시리즈 분석

  모델별 성능 (전체 데이터 fit, period=7 직접 전달)
    Naive2       OWA=0.900
    DOT          OWA=0.605  (70.4% 승률 vs Naive2)
    AutoCES      OWA=0.604  (71.4% 승률)
    4Theta       OWA=0.865  (54.4% 승률)
    AutoETS      OWA=1.432  (70.8% 승률)
    forecast()   OWA=0.908  (52.2% 승률)

  Oracle (4모델 best) OWA = 0.516
  forecast()가 개별 모델(0.605)보다 나쁨 (0.908) → 파이프라인 문제

  모델 승률 분포: auto_ets 32.4%, dot 24.6%, auto_ces 22.4%, four_theta 20.6%
  Naive2 승률: 80.4% (402/500)
  Winner vs Loser: 길이/변동성 차이 미미

결론:
- **핵심 발견**: DOT/CES 단독 OWA 0.60인데 forecast()는 0.91 → 파이프라인이 50% 악화시킴
- E052에서 원인 분석 → E053에서 FFT period 감지가 근본 원인으로 확인
- 개별 모델 성능은 충분함 (DOT 0.605, CES 0.604) — 파이프라인 수정이 핵심

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

    from vectrix.engine.registry import createModel, getRegistry
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix import Vectrix

    PERIOD = 7
    HORIZON = 14

    M4_SMAPE_DAILY = 3.045
    M4_MASE_DAILY = 3.278

    trainFile = os.path.join(DATA_DIR, 'Daily-train.csv')
    testFile = os.path.join(DATA_DIR, 'Daily-test.csv')

    trainDf = pd.read_csv(trainFile)
    testDf = pd.read_csv(testFile)

    N_SERIES = 500
    nRows = min(N_SERIES, len(trainDf))

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

    def naive2Forecast(trainY, period, horizon):
        n = len(trainY)
        if period > 1 and n >= period * 2:
            seasonal = np.zeros(period)
            for i in range(period):
                indices = list(range(n - 1 - i, -1, -period))[:3]
                if indices:
                    seasonal[i] = np.mean(trainY[indices])
            pred = np.array([seasonal[(n + h) % period] for h in range(horizon)])
        else:
            pred = np.full(horizon, trainY[-1])
        return pred

    MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets']

    print(f"=== Daily Diagnosis (period={PERIOD}, h={HORIZON}, n={nRows}) ===\n")

    print("--- Phase 1: Individual Model Performance ---")

    modelSmapes = {m: [] for m in MODELS}
    modelMases = {m: [] for m in MODELS}
    naive2Smapes = []
    naive2Mases = []

    pipelineSmapes = []
    pipelineMases = []

    seriesInfo = []

    for rowIdx in range(nRows):
        trainY = trainDf.iloc[rowIdx, 1:].dropna().values.astype(float)
        testY = testDf.iloc[rowIdx, 1:].dropna().values[:HORIZON].astype(float)
        sid = trainDf.iloc[rowIdx, 0]

        if len(trainY) < 20 or len(testY) < HORIZON:
            continue

        n2Pred = naive2Forecast(trainY, PERIOD, HORIZON)
        n2S = smape(testY, n2Pred)
        n2M = mase(testY, n2Pred, trainY, PERIOD)
        naive2Smapes.append(n2S)
        naive2Mases.append(n2M)

        bestModelSmape = np.inf
        bestModelName = None

        for modelId in MODELS:
            try:
                model = createModel(modelId, PERIOD)
                model.fit(trainY)
                pred, _, _ = model.predict(HORIZON)
                pred = np.clip(pred, -1e15, 1e15)
                pred = np.where(np.isnan(pred), np.nanmean(trainY), pred)

                s = smape(testY, pred)
                m = mase(testY, pred, trainY, PERIOD)

                modelSmapes[modelId].append(s)
                modelMases[modelId].append(m)

                if s < bestModelSmape:
                    bestModelSmape = s
                    bestModelName = modelId
            except Exception:
                modelSmapes[modelId].append(np.nan)
                modelMases[modelId].append(np.nan)

        try:
            dates = pd.date_range('2000-01-01', periods=len(trainY), freq='D')
            df = pd.DataFrame({'date': dates, 'value': trainY})
            vx = Vectrix()
            result = vx.forecast(df, dateCol='date', valueCol='value', steps=HORIZON)
            pPred = np.array(result.predictions[:HORIZON])
            pPred = np.clip(pPred, -1e15, 1e15)
            pPred = np.where(np.isnan(pPred), np.nanmean(trainY), pPred)

            pS = smape(testY, pPred)
            pM = mase(testY, pPred, trainY, PERIOD)
            pipelineSmapes.append(pS)
            pipelineMases.append(pM)
            selectedModel = result.modelUsed if hasattr(result, 'modelUsed') else 'unknown'
        except Exception:
            pipelineSmapes.append(np.nan)
            pipelineMases.append(np.nan)
            selectedModel = 'error'

        cv = np.std(trainY) / (np.mean(np.abs(trainY)) + 1e-10)
        trendSlope = np.polyfit(np.arange(len(trainY)), trainY, 1)[0] / (np.std(trainY) + 1e-10)

        seriesInfo.append({
            'sid': sid,
            'n': len(trainY),
            'cv': cv,
            'trendSlope': abs(trendSlope),
            'mean': np.mean(trainY),
            'bestModel': bestModelName,
            'bestSmape': bestModelSmape,
            'naive2Smape': n2S,
            'pipelineSmape': pipelineSmapes[-1] if pipelineSmapes else np.nan,
            'selectedModel': selectedModel,
            'beatsNaive2': bestModelSmape < n2S,
        })

        if (rowIdx + 1) % 100 == 0:
            print(f"  Processed {rowIdx + 1}/{nRows} series...")

    print(f"\n  Total valid series: {len(naive2Smapes)}")

    print("\n--- Phase 2: Model-Level Results ---\n")

    print(f"  {'Model':15s} {'sMAPE':>8s} {'MASE':>8s} {'OWA':>8s} {'vs Naive2':>10s}")
    print("  " + "-" * 55)

    n2AvgSmape = np.nanmean(naive2Smapes)
    n2AvgMase = np.nanmean(naive2Mases)
    n2Owa = ((n2AvgSmape / M4_SMAPE_DAILY) + (n2AvgMase / M4_MASE_DAILY)) / 2
    print(f"  {'Naive2':15s} {n2AvgSmape:8.3f} {n2AvgMase:8.3f} {n2Owa:8.4f} {'baseline':>10s}")

    for modelId in MODELS:
        avgS = np.nanmean(modelSmapes[modelId])
        avgM = np.nanmean(modelMases[modelId])
        owa = ((avgS / M4_SMAPE_DAILY) + (avgM / M4_MASE_DAILY)) / 2
        winRate = sum(1 for ms, ns in zip(modelSmapes[modelId], naive2Smapes)
                     if not np.isnan(ms) and ms < ns) / len(naive2Smapes) * 100
        print(f"  {modelId:15s} {avgS:8.3f} {avgM:8.3f} {owa:8.4f} {winRate:8.1f}% win")

    pAvgSmape = np.nanmean(pipelineSmapes)
    pAvgMase = np.nanmean(pipelineMases)
    pOwa = ((pAvgSmape / M4_SMAPE_DAILY) + (pAvgMase / M4_MASE_DAILY)) / 2
    pWinRate = sum(1 for ps, ns in zip(pipelineSmapes, naive2Smapes)
                   if not np.isnan(ps) and ps < ns) / len(naive2Smapes) * 100
    print(f"  {'forecast()':15s} {pAvgSmape:8.3f} {pAvgMase:8.3f} {pOwa:8.4f} {pWinRate:8.1f}% win")

    print("\n--- Phase 3: Oracle Analysis ---\n")

    oracleSmapes = []
    oracleMases = []
    for i in range(len(naive2Smapes)):
        allS = [modelSmapes[m][i] for m in MODELS if not np.isnan(modelSmapes[m][i])]
        allM = [modelMases[m][i] for m in MODELS if not np.isnan(modelMases[m][i])]
        if allS:
            bestIdx = np.argmin(allS)
            oracleSmapes.append(allS[bestIdx])
            oracleMases.append(allM[bestIdx])

    oAvgS = np.nanmean(oracleSmapes)
    oAvgM = np.nanmean(oracleMases)
    oOwa = ((oAvgS / M4_SMAPE_DAILY) + (oAvgM / M4_MASE_DAILY)) / 2
    print(f"  Oracle (per-series best of 4 models) OWA = {oOwa:.4f}")
    print(f"  Oracle sMAPE = {oAvgS:.3f}, MASE = {oAvgM:.3f}")

    print("\n--- Phase 4: Best Model Distribution ---\n")

    from collections import Counter
    bestDist = Counter(s['bestModel'] for s in seriesInfo if s['bestModel'])
    total = sum(bestDist.values())
    for m, c in bestDist.most_common():
        print(f"  {m:15s}: {c:4d} ({c/total*100:.1f}%)")

    print("\n--- Phase 5: Failure Pattern Analysis ---\n")

    infodf = pd.DataFrame(seriesInfo)

    losers = infodf[~infodf['beatsNaive2']]
    winners = infodf[infodf['beatsNaive2']]

    print(f"  Beat Naive2: {len(winners)}/{len(infodf)} ({len(winners)/len(infodf)*100:.1f}%)")
    print(f"  Lost to Naive2: {len(losers)}/{len(infodf)} ({len(losers)/len(infodf)*100:.1f}%)")

    print(f"\n  Winner characteristics (mean)")
    print(f"    series length: {winners['n'].mean():.0f}")
    print(f"    CV: {winners['cv'].mean():.4f}")
    print(f"    trend slope: {winners['trendSlope'].mean():.4f}")
    print(f"    data mean: {winners['mean'].mean():.1f}")

    print(f"\n  Loser characteristics (mean)")
    print(f"    series length: {losers['n'].mean():.0f}")
    print(f"    CV: {losers['cv'].mean():.4f}")
    print(f"    trend slope: {losers['trendSlope'].mean():.4f}")
    print(f"    data mean: {losers['mean'].mean():.1f}")

    print("\n--- Phase 6: Series Length Impact ---\n")

    for minLen in [50, 100, 200, 500, 1000]:
        longSeries = infodf[infodf['n'] >= minLen]
        if len(longSeries) < 10:
            continue
        winR = longSeries['beatsNaive2'].mean() * 100
        avgBestSmape = longSeries['bestSmape'].mean()
        avgN2Smape = longSeries['naive2Smape'].mean()
        print(f"  n>={minLen:4d}: {len(longSeries):3d} series, win rate={winR:.1f}%, "
              f"best sMAPE={avgBestSmape:.3f} vs Naive2 sMAPE={avgN2Smape:.3f}")

    print("\n--- Phase 7: Model Selection in Pipeline ---\n")

    selDist = Counter(s['selectedModel'] for s in seriesInfo)
    for m, c in selDist.most_common():
        print(f"  {m:20s}: {c:4d} ({c/len(seriesInfo)*100:.1f}%)")

    print("\n\nDone.")
