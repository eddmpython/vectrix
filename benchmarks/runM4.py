"""
M4 Competition 벤치마크 실행

vectrix의 각 모델을 M4 데이터셋에서 평가합니다.
sMAPE, MASE, OWA를 계산하여 공식 결과와 비교합니다.

사용법:
    python benchmarks/runM4.py --freq Monthly --n 100
    python benchmarks/runM4.py --freq Yearly --n 0     # 전체
    python benchmarks/runM4.py --all --n 50             # 전 주기 50개씩
"""

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")

from m4Loader import M4_FREQUENCIES, loadM4, mase, owa, smape


def naive2Forecast(train: np.ndarray, horizon: int, period: int) -> np.ndarray:
    if period <= 1 or len(train) < 2 * period:
        return np.full(horizon, train[-1])
    deseasonalized = np.copy(train).astype(np.float64)
    nFull = len(train) // period
    if nFull >= 2:
        seasonalIndices = np.ones(period)
        for j in range(period):
            vals = [train[i * period + j] for i in range(nFull) if i * period + j < len(train)]
            if vals:
                seasonalIndices[j] = np.mean(vals)
        grandMean = np.mean(seasonalIndices)
        if grandMean > 0:
            seasonalIndices = seasonalIndices / grandMean
        for i in range(len(train)):
            idx = i % period
            if seasonalIndices[idx] > 0:
                deseasonalized[i] = train[i] / seasonalIndices[idx]
        lastVal = deseasonalized[-1]
        pred = np.full(horizon, lastVal)
        for h in range(horizon):
            idx = (len(train) + h) % period
            pred[h] = pred[h] * seasonalIndices[idx]
        return pred
    return np.full(horizon, train[-1])


def _isSane(pred: np.ndarray, train: np.ndarray) -> bool:
    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        return False
    trainRange = np.max(train) - np.min(train)
    if trainRange < 1e-10:
        return True
    maxDev = max(np.abs(pred - np.mean(train)).max(), 1e-10)
    if maxDev > 10 * trainRange:
        return False
    return True


def vectrixForecast(train: np.ndarray, horizon: int, period: int) -> np.ndarray:
    from vectrix.engine.ets import AutoETS
    from vectrix.engine.theta import OptimizedTheta
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.arima import AutoARIMA

    n = len(train)
    effectivePeriod = period if n >= period * 3 else 1

    candidates = []
    candidates.append(("theta", lambda: OptimizedTheta(period=effectivePeriod)))
    candidates.append(("dot", lambda: DynamicOptimizedTheta(period=effectivePeriod)))
    candidates.append(("ets", lambda: AutoETS(period=effectivePeriod)))
    candidates.append(("ces", lambda: AutoCES(period=effectivePeriod)))
    if n >= 30:
        candidates.append(("arima", lambda: AutoARIMA(maxP=3, maxD=2, maxQ=3, seasonalPeriod=effectivePeriod)))

    if n < 2 * horizon + 5:
        bestPred = np.full(horizon, train[-1])
        for name, factory in candidates:
            try:
                model = factory()
                model.fit(train)
                pred, _, _ = model.predict(horizon)
                pred = np.array(pred[:horizon], dtype=np.float64)
                if _isSane(pred, train):
                    return pred
            except Exception:
                continue
        return bestPred

    valSize = min(horizon, max(n // 5, 3))
    valTrain = train[:-valSize]
    valTest = train[-valSize:]

    bestPred = np.full(horizon, train[-1])
    bestErr = float("inf")

    for name, factory in candidates:
        try:
            model = factory()
            model.fit(valTrain)
            valPred, _, _ = model.predict(valSize)
            valPred = np.array(valPred[:valSize], dtype=np.float64)

            if not _isSane(valPred, train):
                continue

            err = smape(valTest, valPred)

            if err < bestErr:
                bestErr = err

                fullModel = factory()
                fullModel.fit(train)
                fullPred, _, _ = fullModel.predict(horizon)
                fullPred = np.array(fullPred[:horizon], dtype=np.float64)

                if _isSane(fullPred, train):
                    bestPred = fullPred
        except Exception:
            continue

    return bestPred


def runBenchmark(
    frequency: str,
    nSeries: int = 0,
    methods: list = None,
) -> pd.DataFrame:
    nArg = nSeries if nSeries > 0 else None
    trainSeries, testSeries, horizon = loadM4(frequency, nSeries=nArg)

    info = M4_FREQUENCIES[frequency]
    period = info["period"]
    if frequency == "Monthly":
        period = 12
    elif frequency == "Quarterly":
        period = 4
    elif frequency == "Hourly":
        period = 24
    elif frequency == "Weekly":
        period = 52
    elif frequency == "Daily":
        period = 7

    if methods is None:
        methods = ["naive2", "vectrix"]

    methodFns = {
        "naive2": lambda tr, h: naive2Forecast(tr, h, period),
        "vectrix": lambda tr, h: vectrixForecast(tr, h, period),
    }

    results = {m: {"smape": [], "mase": [], "time": []} for m in methods}
    total = len(trainSeries)
    completed = 0

    for seriesId in trainSeries:
        train = trainSeries[seriesId]
        if seriesId not in testSeries:
            continue
        test = testSeries[seriesId]
        actualHorizon = min(horizon, len(test))

        for methodName in methods:
            fn = methodFns[methodName]
            t0 = time.time()
            pred = fn(train, actualHorizon)
            elapsed = time.time() - t0

            pred = pred[:actualHorizon]
            actual = test[:actualHorizon]

            sVal = smape(actual, pred)
            mVal = mase(actual, pred, train, period)

            results[methodName]["smape"].append(sVal)
            results[methodName]["mase"].append(mVal)
            results[methodName]["time"].append(elapsed)

        completed += 1
        if completed % 50 == 0 or completed == total:
            print(f"  [{completed}/{total}]")

    rows = []
    for methodName in methods:
        avgSmape = np.mean(results[methodName]["smape"]) if results[methodName]["smape"] else 0
        avgMase = np.mean(results[methodName]["mase"]) if results[methodName]["mase"] else 0
        avgTime = np.mean(results[methodName]["time"]) if results[methodName]["time"] else 0
        totalTime = np.sum(results[methodName]["time"]) if results[methodName]["time"] else 0

        rows.append({
            "method": methodName,
            "frequency": frequency,
            "nSeries": completed,
            "sMAPE": round(avgSmape, 3),
            "MASE": round(avgMase, 3),
            "avgTime_s": round(avgTime, 4),
            "totalTime_s": round(totalTime, 2),
        })

    naive2Row = [r for r in rows if r["method"] == "naive2"]
    if naive2Row:
        n2Smape = naive2Row[0]["sMAPE"]
        n2Mase = naive2Row[0]["MASE"]
        for row in rows:
            row["OWA"] = round(owa(row["sMAPE"], row["MASE"], n2Smape, n2Mase), 3)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="M4 Benchmark for Vectrix")
    parser.add_argument("--freq", type=str, default="Monthly", help="Frequency: Yearly, Quarterly, Monthly, Weekly, Daily, Hourly")
    parser.add_argument("--n", type=int, default=100, help="Number of series (0 = all)")
    parser.add_argument("--all", action="store_true", help="Run all frequencies")
    args = parser.parse_args()

    print("=" * 60)
    print("  Vectrix M4 Competition Benchmark")
    print("=" * 60)

    if args.all:
        frequencies = list(M4_FREQUENCIES.keys())
    else:
        frequencies = [args.freq]

    allResults = []
    for freq in frequencies:
        print(f"\n--- {freq} ---")
        df = runBenchmark(freq, nSeries=args.n)
        allResults.append(df)
        print(df.to_string(index=False))

    if len(allResults) > 1:
        combined = pd.concat(allResults, ignore_index=True)
        print("\n" + "=" * 60)
        print("  COMBINED RESULTS")
        print("=" * 60)
        print(combined.to_string(index=False))

        outPath = Path(__file__).parent / "m4Results.csv"
        combined.to_csv(outPath, index=False)
        print(f"\nResults saved to {outPath}")


if __name__ == "__main__":
    main()
