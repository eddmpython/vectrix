"""
실험용 합성 데이터 생성기 통합 모듈.

기존 e001, e005, e006, e007, e011의 중복 생성기를 통합.
모든 생성기는 pd.DataFrame(columns=['date', 'value'])을 반환.
"""

import numpy as np
import pandas as pd


def generateRetailSales(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """주간+연간 계절성 + 추세 + 휴일 스파이크 소매 판매 데이터."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    base = 1000.0
    trend = 200.0 * t / n
    weekly = 150.0 * np.sin(2.0 * np.pi * t / 7.0)
    yearly = 200.0 * np.sin(2.0 * np.pi * t / 365.0)
    noise = rng.normal(0, 50, n)

    values = base + trend + weekly + yearly + noise
    values = np.maximum(values, 100.0)

    return pd.DataFrame({"date": dates, "value": values})


def generateStockPrice(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """GBM + 변동성 클러스터링 주가 데이터."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    mu = 0.0005
    sigma = 0.02
    returns = rng.normal(mu, sigma, n)

    for i in range(1, n):
        if abs(returns[i - 1]) > 0.03:
            returns[i] *= 1.5

    prices = 100.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame({"date": dates, "value": prices})


def generateTemperature(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """연간 계절성 + 이상 기후 기온 데이터."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    base = 12.0
    yearly = 15.0 * np.cos(2.0 * np.pi * (t - 180) / 365.0)
    noise = rng.normal(0, 3, n)

    values = base + yearly + noise

    outlierIdx = rng.choice(n, size=10, replace=False)
    values[outlierIdx] += rng.choice([-10.0, 10.0], size=10)

    return pd.DataFrame({"date": dates, "value": values})


def generateEnergyUsage(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """주간+연간 다중 계절성 에너지 소비 데이터."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    base = 500.0
    trend = 20.0 * t / n
    weekly = 50.0 * np.sin(2.0 * np.pi * t / 7.0)
    yearly = 100.0 * np.sin(2.0 * np.pi * (t - 30) / 365.0)
    noise = rng.normal(0, 20, n)

    values = base + trend + weekly + yearly + noise
    values = np.maximum(values, 200.0)

    return pd.DataFrame({"date": dates, "value": values})


def generateManufacturing(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """추세 + 주기적 설비 점검 드롭 제조 데이터."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    base = 1000.0
    trend = 100.0 * t / n
    noise = rng.normal(0, 30, n)

    values = base + trend + noise

    dropPeriod = 90
    dropDuration = 7
    for start in range(dropPeriod, n, dropPeriod):
        end = min(start + dropDuration, n)
        values[start:end] *= 0.7

    values = np.maximum(values, 100.0)

    return pd.DataFrame({"date": dates, "value": values})


def generateIntermittentDemand(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """포아송 기반 간헐적 수요 데이터. 대부분 0, 간헐적 양수."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    demandProb = 0.15
    hasDemand = rng.random(n) < demandProb
    values = np.zeros(n, dtype=np.float64)
    values[hasDemand] = rng.poisson(5, size=int(hasDemand.sum())).astype(np.float64)

    return pd.DataFrame({"date": dates, "value": values})


def generateMultiSeasonalRetail(n: int = 730, seed: int = 42) -> pd.DataFrame:
    """2년 소매 데이터. 주간+연간 강한 다중 계절성."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    base = 1000.0
    trend = 300.0 * t / n
    weekly = 150.0 * np.sin(2.0 * np.pi * t / 7.0)
    yearly = 250.0 * np.sin(2.0 * np.pi * t / 365.0)
    noise = rng.normal(0, 40, n)

    values = base + trend + weekly + yearly + noise
    values = np.maximum(values, 100.0)

    return pd.DataFrame({"date": dates, "value": values})


def generateVolatile(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """GARCH 스타일 변동성 클러스터링 데이터."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    omega = 0.1
    alpha = 0.15
    beta = 0.8
    sigma2 = np.zeros(n)
    values = np.zeros(n)
    sigma2[0] = 1.0
    values[0] = 100.0

    for i in range(1, n):
        sigma2[i] = omega + alpha * (values[i - 1] - 100.0) ** 2 / 100.0 + beta * sigma2[i - 1]
        values[i] = 100.0 + rng.normal(0, np.sqrt(max(sigma2[i], 0.01)))

    return pd.DataFrame({"date": dates, "value": values})


def generateTrending(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """강한 비선형 추세 + 약한 노이즈 데이터."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    base = 100.0
    trend = 200.0 * (1.0 - np.exp(-3.0 * t / n))
    noise = rng.normal(0, 5, n)

    values = base + trend + noise

    return pd.DataFrame({"date": dates, "value": values})


def generateStationary(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """정상 시계열. AR(1) 과정."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    phi = 0.7
    values = np.zeros(n)
    values[0] = 100.0

    for i in range(1, n):
        values[i] = 100.0 + phi * (values[i - 1] - 100.0) + rng.normal(0, 3)

    return pd.DataFrame({"date": dates, "value": values})


def generateRegimeShift(n: int = 365, seed: int = 42) -> pd.DataFrame:
    """2~3개 레짐이 존재하는 시계열. 레짐별 평균/분산 변화."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    changePoints = sorted(rng.choice(range(n // 4, 3 * n // 4), size=2, replace=False))
    cp1, cp2 = changePoints[0], changePoints[1]

    values = np.zeros(n)
    values[:cp1] = 100.0 + rng.normal(0, 5, cp1)
    values[cp1:cp2] = 150.0 + rng.normal(0, 15, cp2 - cp1)
    values[cp2:] = 80.0 + rng.normal(0, 8, n - cp2)

    return pd.DataFrame({"date": dates, "value": values})


ALL_GENERATORS = {
    "retailSales": generateRetailSales,
    "stockPrice": generateStockPrice,
    "temperature": generateTemperature,
    "energyUsage": generateEnergyUsage,
    "manufacturing": generateManufacturing,
    "intermittentDemand": generateIntermittentDemand,
    "multiSeasonalRetail": generateMultiSeasonalRetail,
    "volatile": generateVolatile,
    "trending": generateTrending,
    "stationary": generateStationary,
    "regimeShift": generateRegimeShift,
}


def generateAll(n: int = 365, seed: int = 42) -> dict:
    """모든 합성 데이터를 한번에 생성. {name: DataFrame} 반환."""
    result = {}
    for name, genFunc in ALL_GENERATORS.items():
        if name == "multiSeasonalRetail":
            result[name] = genFunc(n=max(n, 730), seed=seed)
        elif name == "stockPrice":
            result[name] = genFunc(n=min(n, 252), seed=seed)
        else:
            result[name] = genFunc(n=n, seed=seed)
    return result
