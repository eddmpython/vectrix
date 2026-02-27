"""
M3 Competition 벤치마크 실행

사용법:
    python benchmarks/runM3.py --cat M3Month --n 100
    python benchmarks/runM3.py --all --n 50
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

from m3Loader import M3_INFO, loadM3, mase, smape


def naive2Forecast(train: np.ndarray, horizon: int, period: int) -> np.ndarray:
    if period <= 1 or len(train) < 2 * period:
        return np.full(horizon, train[-1])
    nFull = len(train) // period
    if nFull >= 2:
        seasonalIndices = np.ones(period)
        for j in range(period):
            vals = [train[i * period + j] for i in range(nFull) if i * period + j < len(train)]
            if vals:
                seasonalIndices[j] = np.mean(vals)
        grandMean = np.mean(seasonalIndices)
        if grandMean > 0:
            seasonalIndices /= grandMean
        deseasonalized = np.copy(train).astype(np.float64)
        for i in range(len(train)):
            idx = i % period
            if seasonalIndices[idx] > 0:
                deseasonalized[i] = train[i] / seasonalIndices[idx]
        lastVal = deseasonalized[-1]
        pred = np.full(horizon, lastVal)
        for h in range(horizon):
            idx = (len(train) + h) % period
            pred[h] *= seasonalIndices[idx]
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


def runBenchmark(category: str, nSeries: int = 0) -> pd.DataFrame:
    nArg = nSeries if nSeries > 0 else None
    trainSeries, testSeries, horizon = loadM3(category, nSeries=nArg)

    info = M3_INFO[category]
    period = info["period"]

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
        totalTime = np.sum(results[methodName]["time"]) if results[methodName]["time"] else 0

        rows.append({
            "method": methodName,
            "category": category,
            "nSeries": completed,
            "sMAPE": round(avgSmape, 3),
            "MASE": round(avgMase, 3),
            "totalTime_s": round(totalTime, 2),
        })

    naive2Row = [r for r in rows if r["method"] == "naive2"]
    if naive2Row:
        n2Smape = naive2Row[0]["sMAPE"]
        n2Mase = naive2Row[0]["MASE"]
        for row in rows:
            if n2Smape > 0 and n2Mase > 0:
                row["OWA"] = round(0.5 * (row["sMAPE"] / n2Smape + row["MASE"] / n2Mase), 3)
            else:
                row["OWA"] = 1.0

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="M3 Benchmark for Vectrix")
    parser.add_argument("--cat", type=str, default="M3Month", help="Category: M3Year, M3Quart, M3Month, M3Other")
    parser.add_argument("--n", type=int, default=100, help="Number of series (0 = all)")
    parser.add_argument("--all", action="store_true", help="Run all categories")
    args = parser.parse_args()

    print("=" * 60)
    print("  Vectrix M3 Competition Benchmark")
    print("=" * 60)

    if args.all:
        categories = list(M3_INFO.keys())
    else:
        categories = [args.cat]

    allResults = []
    for cat in categories:
        print(f"\n--- {cat} ---")
        df = runBenchmark(cat, nSeries=args.n)
        allResults.append(df)
        print(df.to_string(index=False))

    if len(allResults) > 1:
        combined = pd.concat(allResults, ignore_index=True)
        print("\n" + "=" * 60)
        print("  COMBINED RESULTS")
        print("=" * 60)
        print(combined.to_string(index=False))

        outPath = Path(__file__).parent / "m3Results.csv"
        combined.to_csv(outPath, index=False)
        print(f"\nResults saved to {outPath}")


if __name__ == "__main__":
    main()
