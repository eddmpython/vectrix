"""
M3 Competition 데이터 로더

M3 데이터셋(3,003개 시계열)을 로드합니다.
https://forecasters.org/resources/time-series-data/m3-competition/
"""

import io
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


M3_URLS = {
    "M3Year": "https://forvis.github.io/data/M3_yearly_TSTS.csv",
    "M3Quart": "https://forvis.github.io/data/M3_quarterly_TSTS.csv",
    "M3Month": "https://forvis.github.io/data/M3_monthly_TSTS.csv",
    "M3Other": "https://forvis.github.io/data/M3_other_TSTS.csv",
}

M3_INFO = {
    "M3Year": {"period": 1, "horizon": 6, "prefix": "Y"},
    "M3Quart": {"period": 4, "horizon": 8, "prefix": "Q"},
    "M3Month": {"period": 12, "horizon": 18, "prefix": "M"},
    "M3Other": {"period": 1, "horizon": 8, "prefix": "O"},
}


def loadM3(
    category: str = "M3Month",
    nSeries: Optional[int] = None,
    cacheDir: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    if category not in M3_URLS:
        raise ValueError(f"Unknown category: {category}. Choose from {list(M3_URLS.keys())}")

    if cacheDir is None:
        cacheDir = Path(__file__).parent / ".cache"
    cacheDir.mkdir(parents=True, exist_ok=True)

    cacheFile = f"M3_{category}.csv"
    cachePath = cacheDir / cacheFile

    if cachePath.exists():
        df = pd.read_csv(cachePath)
    else:
        url = M3_URLS[category]
        print(f"Downloading M3 {category} dataset...")
        response = urllib.request.urlopen(url)
        data = response.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(data))
        df.to_csv(cachePath, index=False)

    info = M3_INFO[category]
    horizon = info["horizon"]

    uniqueIds = df["series_id"].unique()
    limit = nSeries if nSeries else len(uniqueIds)

    trainSeries = {}
    testSeries = {}

    for sid in uniqueIds[:limit]:
        subset = df[df["series_id"] == sid]["value"].values.astype(np.float64)
        if len(subset) <= horizon:
            continue
        trainSeries[str(sid)] = subset[:-horizon]
        testSeries[str(sid)] = subset[-horizon:]

    print(f"  Loaded {len(trainSeries)} M3 {category} series (horizon={horizon})")
    return trainSeries, testSeries, horizon


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    denominator = np.abs(actual) + np.abs(predicted)
    mask = denominator > 0
    if not np.any(mask):
        return 0.0
    return float(200.0 * np.mean(np.abs(actual[mask] - predicted[mask]) / denominator[mask]))


def mase(actual: np.ndarray, predicted: np.ndarray, trainData: np.ndarray, period: int) -> float:
    n = len(trainData)
    if n <= period:
        naiveDiff = np.abs(np.diff(trainData))
    else:
        naiveDiff = np.abs(trainData[period:] - trainData[:-period])
    scaleFactor = np.mean(naiveDiff) if len(naiveDiff) > 0 else 1.0
    if scaleFactor == 0:
        scaleFactor = 1.0
    return float(np.mean(np.abs(actual - predicted)) / scaleFactor)
