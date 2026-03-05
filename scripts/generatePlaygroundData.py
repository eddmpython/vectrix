"""
Playground 페이지용 사전 계산 JSON 데이터 생성.

6개 다양한 샘플 데이터셋에 대해 forecast + analyze 결과를 생성하여
landing/static/playground/data.json에 저장한다.

사용법:
    cd /c/Users/MSI/OneDrive/Desktop/sideProject/vectrix
    uv run python scripts/generatePlaygroundData.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import vectrix


DATASETS = [
    {
        "id": "airline",
        "label": "Airline Passengers",
        "domain": "Transportation",
        "frequency": "monthly",
        "steps": 12,
        "description": "Classic monthly airline passenger data (1949-1960) with trend and multiplicative seasonality.",
    },
    {
        "id": "retail",
        "label": "Retail Sales",
        "domain": "Retail",
        "frequency": "daily",
        "steps": 30,
        "description": "Daily retail sales with weekly/annual seasonality and holiday spikes.",
    },
    {
        "id": "stock",
        "label": "Stock Price",
        "domain": "Finance",
        "frequency": "business_daily",
        "steps": 20,
        "description": "Daily stock closing price — random walk with volatility clustering.",
    },
    {
        "id": "energy",
        "label": "Energy Consumption",
        "domain": "Energy",
        "frequency": "hourly",
        "steps": 48,
        "description": "Hourly energy consumption (kWh) with daily and weekly patterns.",
    },
    {
        "id": "web",
        "label": "Web Traffic",
        "domain": "Technology",
        "frequency": "daily",
        "steps": 30,
        "description": "Daily web pageviews with exponential growth and weekly pattern.",
    },
    {
        "id": "intermittent",
        "label": "Intermittent Demand",
        "domain": "Supply Chain",
        "frequency": "daily",
        "steps": 30,
        "description": "Sparse/lumpy demand pattern — common in spare parts and low-volume products.",
    },
]

VALUE_COLS = {
    "airline": "passengers",
    "retail": "sales",
    "stock": "close",
    "energy": "consumption_kwh",
    "web": "pageviews",
    "intermittent": "demand",
}

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "landing" / "static" / "playground" / "data.json"


def sanitizeValue(v):
    if isinstance(v, (np.floating, float)):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(float(v), 4)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [sanitizeValue(x) for x in v]
    if isinstance(v, (list, tuple)):
        return [sanitizeValue(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def processDataset(config):
    datasetId = config["id"]
    valueCol = VALUE_COLS[datasetId]
    steps = config["steps"]

    print(f"  Loading {datasetId}...")
    df = vectrix.loadSample(datasetId)

    dates = df["date"].astype(str).tolist()
    values = df[valueCol].tolist()

    print(f"  Running forecast (steps={steps})...")
    forecastResult = vectrix.forecast(df, steps=steps)

    print(f"  Running analyze...")
    analysisResult = vectrix.analyze(df)

    print(f"  Extracting model comparison...")
    comparison = forecastResult.compare()

    allForecasts = forecastResult.allForecasts()
    allForecastDates = allForecasts["date"].astype(str).tolist()
    modelForecasts = {}
    for col in allForecasts.columns:
        if col == "date":
            continue
        modelForecasts[col] = [sanitizeValue(v) for v in allForecasts[col].tolist()]

    dna = analysisResult.dna
    selectedFeatures = [
        "trendStrength", "seasonalStrength", "volatility", "forecastability",
        "hurstExponent", "spectralEntropy", "cv", "skewness", "kurtosis",
        "garchEffect", "trendSlope", "trendDirection", "trendLinearity",
        "trendCurvature", "seasonalPeakPeriod", "nonlinearAutocorr",
        "approximateEntropy", "acf1", "demandDensity", "volatilityClustering",
    ]
    dnaFeatures = {}
    for key in selectedFeatures:
        if key in dna.features:
            dnaFeatures[key] = sanitizeValue(dna.features[key])

    chars = analysisResult.characteristics
    characteristics = {
        "length": chars.length,
        "frequency": chars.frequency,
        "period": chars.period,
        "hasTrend": chars.hasTrend,
        "hasSeasonality": chars.hasSeasonality,
        "isStationary": chars.isStationary,
        "trendDirection": chars.trendDirection,
        "trendStrength": sanitizeValue(chars.trendStrength),
        "seasonalStrength": sanitizeValue(chars.seasonalStrength),
        "volatility": sanitizeValue(chars.volatility),
        "volatilityLevel": chars.volatilityLevel,
        "missingRatio": sanitizeValue(chars.missingRatio),
        "outlierCount": chars.outlierCount,
        "predictabilityScore": sanitizeValue(chars.predictabilityScore),
    }

    modelComparisonRows = []
    for _, row in comparison.iterrows():
        modelComparisonRows.append({
            "model": row["model"],
            "mape": sanitizeValue(row["mape"]),
            "rmse": sanitizeValue(row["rmse"]),
            "mae": sanitizeValue(row["mae"]),
            "smape": sanitizeValue(row["smape"]),
            "timeMs": sanitizeValue(row["time_ms"]),
            "selected": bool(row["selected"]),
        })

    metrics = {
        "mape": sanitizeValue(forecastResult.mape),
        "rmse": sanitizeValue(forecastResult.rmse),
        "mae": sanitizeValue(forecastResult.mae),
        "smape": sanitizeValue(forecastResult.smape),
    }
    if metrics["mape"] is None and len(modelComparisonRows) > 0:
        best = modelComparisonRows[0]
        metrics = {
            "mape": best["mape"],
            "rmse": best["rmse"],
            "mae": best["mae"],
            "smape": best["smape"],
        }

    return {
        "id": datasetId,
        "label": config["label"],
        "domain": config["domain"],
        "frequency": config["frequency"],
        "description": config["description"],
        "timeSeries": {
            "dates": dates,
            "values": [sanitizeValue(v) for v in values],
            "valueCol": valueCol,
            "length": len(values),
        },
        "forecast": {
            "model": forecastResult.model,
            "steps": steps,
            "dates": [str(d) for d in forecastResult.dates],
            "predictions": [sanitizeValue(v) for v in forecastResult.predictions],
            "lower": [sanitizeValue(v) for v in forecastResult.lower],
            "upper": [sanitizeValue(v) for v in forecastResult.upper],
            "metrics": metrics,
            "rankedModels": forecastResult.models,
            "modelComparison": modelComparisonRows,
            "allForecasts": {
                "dates": allForecastDates,
                "models": modelForecasts,
            },
        },
        "analysis": {
            "dna": {
                "fingerprint": dna.fingerprint,
                "difficulty": dna.difficulty,
                "difficultyScore": sanitizeValue(dna.difficultyScore),
                "category": dna.category,
                "recommendedModels": dna.recommendedModels,
                "summary": dna.summary,
                "features": dnaFeatures,
            },
            "changepoints": [int(x) for x in analysisResult.changepoints],
            "anomalies": [int(x) for x in analysisResult.anomalies],
            "characteristics": characteristics,
        },
    }


def main():
    print("=== Vectrix Playground Data Generator ===\n")

    results = []
    for config in DATASETS:
        print(f"Processing: {config['label']} ({config['id']})")
        try:
            data = processDataset(config)
            results.append(data)
            print(f"  Done.\n")
        except (ValueError, RuntimeError, KeyError) as e:
            print(f"  FAILED: {e}\n")

    output = {
        "generatedAt": pd.Timestamp.now().isoformat(),
        "version": "1.0.0",
        "datasetCount": len(results),
        "datasets": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    sizeKb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Output: {OUTPUT_PATH}")
    print(f"Size: {sizeKb:.1f} KB")
    print(f"Datasets: {len(results)}/{len(DATASETS)}")


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()
