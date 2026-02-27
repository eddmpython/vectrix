"""
M4 Competition 데이터 로더

M4 데이터셋(100,000개 시계열)을 GitHub에서 다운로드하고 파싱합니다.
https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
"""

import io
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


M4_BASE_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"

M4_FREQUENCIES = {
    "Yearly": {"period": 1, "horizon": 6},
    "Quarterly": {"period": 4, "horizon": 8},
    "Monthly": {"period": 12, "horizon": 18},
    "Weekly": {"period": 1, "horizon": 13},
    "Daily": {"period": 1, "horizon": 14},
    "Hourly": {"period": 24, "horizon": 48},
}


def downloadCsv(url: str, cacheDir: Optional[Path] = None) -> pd.DataFrame:
    if cacheDir is not None:
        cacheDir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        cachePath = cacheDir / filename
        if cachePath.exists():
            return pd.read_csv(cachePath)

    response = urllib.request.urlopen(url)
    data = response.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(data))

    if cacheDir is not None:
        df.to_csv(cachePath, index=False)

    return df


def loadM4(
    frequency: str = "Monthly",
    nSeries: Optional[int] = None,
    cacheDir: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int]:
    if frequency not in M4_FREQUENCIES:
        raise ValueError(f"frequency must be one of {list(M4_FREQUENCIES.keys())}")

    info = M4_FREQUENCIES[frequency]
    horizon = info["horizon"]

    if cacheDir is None:
        cacheDir = Path(__file__).parent / ".cache"

    trainUrl = f"{M4_BASE_URL}/Train/{frequency}-train.csv"
    testUrl = f"{M4_BASE_URL}/Test/{frequency}-test.csv"

    print(f"Loading M4 {frequency} data...")
    trainDf = downloadCsv(trainUrl, cacheDir)
    testDf = downloadCsv(testUrl, cacheDir)

    trainSeries = {}
    testSeries = {}

    idCol = trainDf.columns[0]
    trainIds = trainDf[idCol].values
    testIds = testDf[idCol].values

    limit = nSeries if nSeries else len(trainIds)

    for i in range(min(limit, len(trainIds))):
        seriesId = str(trainIds[i])
        trainRow = trainDf.iloc[i, 1:].dropna().values.astype(np.float64)
        trainSeries[seriesId] = trainRow

        if i < len(testIds):
            testRow = testDf.iloc[i, 1:].dropna().values.astype(np.float64)
            testSeries[seriesId] = testRow

    print(f"  Loaded {len(trainSeries)} series (horizon={horizon})")
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


def owa(smapeVal: float, maseVal: float, smapeNaive2: float, maseNaive2: float) -> float:
    if smapeNaive2 == 0 or maseNaive2 == 0:
        return 1.0
    return 0.5 * (smapeVal / smapeNaive2 + maseVal / maseNaive2)
