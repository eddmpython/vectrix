"""
Built-in sample datasets for quick testing and demos.

    >>> from vectrix import loadSample
    >>> df = loadSample("airline")
    >>> from vectrix import forecast
    >>> result = forecast(df, date="date", value="passengers", steps=12)
"""

from typing import Dict

import numpy as np
import pandas as pd


def _generateAirline() -> pd.DataFrame:
    """
    Classic airline passengers dataset (1949-1960, 144 monthly observations).
    Multiplicative trend + seasonality pattern.
    """
    nMonths = 144
    dates = pd.date_range("1949-01-01", periods=nMonths, freq="MS")
    t = np.arange(nMonths, dtype=np.float64)

    trend = 110 + 2.2 * t + 0.015 * t**2
    seasonalPattern = np.array([
        -20, -30, -5, 5, 15, 40, 60, 55, 25, -10, -30, -25
    ], dtype=np.float64)
    seasonal = np.tile(seasonalPattern, nMonths // 12)
    scaleFactor = 1.0 + t * 0.004

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 5, nMonths)

    values = trend + seasonal * scaleFactor + noise
    values = np.maximum(values, 50)

    return pd.DataFrame({"date": dates, "passengers": np.round(values, 1)})


def _generateRetailSales() -> pd.DataFrame:
    """
    Daily retail sales (2 years, 730 observations).
    Weekly seasonality + holiday spikes + trend.
    """
    nDays = 730
    dates = pd.date_range("2023-01-01", periods=nDays, freq="D")
    t = np.arange(nDays, dtype=np.float64)

    trend = 1000 + 1.5 * t
    weekday = np.array([dates[i].weekday() for i in range(nDays)])
    weeklyPattern = np.array([1.0, 0.95, 0.92, 0.97, 1.1, 1.35, 1.25])
    seasonal = np.array([weeklyPattern[w] for w in weekday]) * 200

    yearDay = np.array([d.timetuple().tm_yday for d in dates])
    annualSeasonal = 80 * np.sin(2 * np.pi * (yearDay - 80) / 365)

    rng = np.random.default_rng(123)
    noise = rng.normal(0, 40, nDays)

    values = trend + seasonal + annualSeasonal + noise

    for i, d in enumerate(dates):
        if d.month == 12 and d.day >= 20:
            values[i] *= 1.4
        elif d.month == 11 and d.day >= 24 and d.day <= 27:
            values[i] *= 1.3

    values = np.maximum(values, 100)
    return pd.DataFrame({"date": dates, "sales": np.round(values, 2)})


def _generateStockPrice() -> pd.DataFrame:
    """
    Simulated stock price (252 trading days, ~1 year).
    Geometric Brownian motion with volatility clustering.
    """
    nDays = 252
    dates = pd.bdate_range("2024-01-02", periods=nDays)

    rng = np.random.default_rng(7)

    price = np.zeros(nDays, dtype=np.float64)
    price[0] = 150.0
    mu = 0.0003
    sigma = 0.015

    vol = sigma
    for i in range(1, nDays):
        shock = rng.normal()
        vol = 0.94 * vol + 0.06 * sigma * (1 + 0.5 * abs(shock))
        logReturn = mu - 0.5 * vol**2 + vol * shock
        price[i] = price[i - 1] * np.exp(logReturn)

    return pd.DataFrame({"date": dates, "close": np.round(price, 2)})


def _generateTemperature() -> pd.DataFrame:
    """
    Daily temperature (3 years, 1095 observations).
    Strong annual seasonality + daily noise.
    """
    nDays = 1095
    dates = pd.date_range("2022-01-01", periods=nDays, freq="D")
    t = np.arange(nDays, dtype=np.float64)

    annual = 15 + 12 * np.sin(2 * np.pi * (t - 80) / 365.25)
    rng = np.random.default_rng(99)
    noise = rng.normal(0, 3.5, nDays)

    values = annual + noise + 0.002 * t

    return pd.DataFrame({"date": dates, "temperature": np.round(values, 1)})


def _generateEnergy() -> pd.DataFrame:
    """
    Hourly energy consumption (30 days, 720 observations).
    Daily pattern + weekly variation.
    """
    nHours = 720
    dates = pd.date_range("2024-06-01", periods=nHours, freq="h")
    t = np.arange(nHours, dtype=np.float64)

    hourOfDay = t % 24
    hourlyPattern = (
        200
        + 100 * np.sin(2 * np.pi * (hourOfDay - 6) / 24)
        + 60 * np.sin(2 * np.pi * (hourOfDay - 14) / 12)
    )

    dayOfWeek = (t // 24) % 7
    weekendFactor = np.where(dayOfWeek >= 5, 0.75, 1.0)

    rng = np.random.default_rng(55)
    noise = rng.normal(0, 15, nHours)

    values = hourlyPattern * weekendFactor + noise
    values = np.maximum(values, 50)

    return pd.DataFrame({"date": dates, "consumption_kwh": np.round(values, 1)})


def _generateWebTraffic() -> pd.DataFrame:
    """
    Daily website page views (180 days).
    Exponential growth + weekly seasonality + marketing spikes.
    """
    nDays = 180
    dates = pd.date_range("2024-07-01", periods=nDays, freq="D")
    t = np.arange(nDays, dtype=np.float64)

    growth = 5000 * np.exp(0.008 * t)

    weekday = np.array([d.weekday() for d in dates])
    weeklyPattern = np.array([1.0, 1.05, 1.08, 1.1, 0.95, 0.7, 0.6])
    weekly = np.array([weeklyPattern[w] for w in weekday])

    rng = np.random.default_rng(33)
    spikes = np.zeros(nDays)
    spikeIdx = rng.choice(nDays, size=5, replace=False)
    for idx in spikeIdx:
        spikes[idx] = rng.uniform(0.3, 0.8) * growth[idx]

    noise = rng.normal(0, 0.05, nDays)

    values = growth * weekly * (1 + noise) + spikes
    values = np.maximum(values, 100)

    return pd.DataFrame({"date": dates, "pageviews": np.round(values).astype(int)})


def _generateIntermittent() -> pd.DataFrame:
    """
    Intermittent/sparse demand (365 days).
    Many zeros with occasional bursts — classic Croston pattern.
    """
    nDays = 365
    dates = pd.date_range("2024-01-01", periods=nDays, freq="D")

    rng = np.random.default_rng(77)

    demandProb = 0.15
    occurs = rng.random(nDays) < demandProb
    sizes = rng.poisson(lam=8, size=nDays).astype(np.float64)
    values = np.where(occurs, sizes, 0.0)

    burstDays = rng.choice(nDays, size=3, replace=False)
    for d in burstDays:
        values[d] = rng.integers(30, 80)

    return pd.DataFrame({"date": dates, "demand": values.astype(int)})


_REGISTRY: Dict[str, dict] = {
    "airline": {
        "fn": _generateAirline,
        "desc": "Airline passengers (monthly, 144 obs) — trend + multiplicative seasonality",
        "dateCol": "date",
        "valueCol": "passengers",
        "freq": "monthly",
    },
    "retail": {
        "fn": _generateRetailSales,
        "desc": "Retail sales (daily, 730 obs) — weekly + annual seasonality + holiday spikes",
        "dateCol": "date",
        "valueCol": "sales",
        "freq": "daily",
    },
    "stock": {
        "fn": _generateStockPrice,
        "desc": "Stock price (daily, 252 obs) — random walk with volatility clustering",
        "dateCol": "date",
        "valueCol": "close",
        "freq": "business_daily",
    },
    "temperature": {
        "fn": _generateTemperature,
        "desc": "Temperature (daily, 1095 obs) — strong annual seasonality",
        "dateCol": "date",
        "valueCol": "temperature",
        "freq": "daily",
    },
    "energy": {
        "fn": _generateEnergy,
        "desc": "Energy consumption (hourly, 720 obs) — daily + weekly pattern",
        "dateCol": "date",
        "valueCol": "consumption_kwh",
        "freq": "hourly",
    },
    "web": {
        "fn": _generateWebTraffic,
        "desc": "Web traffic (daily, 180 obs) — exponential growth + weekly pattern",
        "dateCol": "date",
        "valueCol": "pageviews",
        "freq": "daily",
    },
    "intermittent": {
        "fn": _generateIntermittent,
        "desc": "Intermittent demand (daily, 365 obs) — sparse/lumpy demand pattern",
        "dateCol": "date",
        "valueCol": "demand",
        "freq": "daily",
    },
}


def loadSample(name: str) -> pd.DataFrame:
    """
    Load a built-in sample dataset for quick testing.

    Parameters
    ----------
    name : str
        Dataset name. One of:
        'airline', 'retail', 'stock', 'temperature',
        'energy', 'web', 'intermittent'

    Returns
    -------
    pd.DataFrame
        DataFrame with 'date' column and a value column.

    Examples
    --------
    >>> from vectrix import loadSample, forecast
    >>> df = loadSample("airline")
    >>> result = forecast(df, date="date", value="passengers", steps=12)

    >>> df = loadSample("retail")
    >>> result = forecast(df, date="date", value="sales", steps=30)
    """
    nameLower = name.lower().strip()
    if nameLower not in _REGISTRY:
        available = ", ".join(f"'{k}'" for k in _REGISTRY)
        raise ValueError(
            f"Unknown dataset: '{name}'. Available: {available}"
        )
    return _REGISTRY[nameLower]["fn"]()


def listSamples() -> pd.DataFrame:
    """
    List all available sample datasets.

    Returns
    -------
    pd.DataFrame
        Table with name, description, value column, frequency, and size.

    Examples
    --------
    >>> from vectrix import listSamples
    >>> print(listSamples())
    """
    rows = []
    for name, info in _REGISTRY.items():
        df = info["fn"]()
        rows.append({
            "name": name,
            "description": info["desc"],
            "valueCol": info["valueCol"],
            "frequency": info["freq"],
            "rows": len(df),
        })
    return pd.DataFrame(rows)


__all__ = ["loadSample", "listSamples"]
