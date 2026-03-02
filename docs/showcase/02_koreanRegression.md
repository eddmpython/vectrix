# Showcase 02 — Korean Regression Analysis

**Regression analysis on real Korean datasets with full diagnostics.**

## Overview

Two regression analyses using publicly available Korean data:

1. **Seoul Bike Sharing Demand** — Predict hourly bike rentals from weather features (UCI ML Repository)
2. **Korean Macro Regression** — Exchange rate determinants using economic indicators (FRED)

## Run Interactively

```bash
pip install vectrix pandas marimo
marimo run docs/showcase/en/02_koreanRegression.py
```

## 1. Seoul Bike Sharing

8,760 hourly observations from Seoul's public bike-sharing system.

```python
import pandas as pd
from vectrix import regress

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
bikeDf = pd.read_csv(url, encoding="unicode_escape")

bikeDf = bikeDf.rename(columns={
    "Rented Bike Count": "rentedBikeCount",
    "Hour": "hour",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "windSpeed",
    "Rainfall(mm)": "rainfall",
    "Solar Radiation (MJ/m2)": "solarRadiation",
})
tempCols = [c for c in bikeDf.columns if "Temperature" in c and "Dew" not in c]
if tempCols:
    bikeDf = bikeDf.rename(columns={tempCols[0]: "temperature"})

result = regress(
    data=bikeDf,
    formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + hour + solarRadiation",
)
```

### Key Findings

| Variable | Effect |
|----------|--------|
| Temperature +1°C | More rentals |
| Humidity +1% | Fewer rentals |
| Rainfall +1mm | Significantly fewer rentals |
| Hour | Peak during commute hours |

```python
print(f"R-squared: {result.r_squared:.4f}")
print(f"F-statistic: {result.f_stat:.2f}")
print(result.diagnose())
```

## 2. Korean Macro Regression

Exchange rate determinants using FRED economic indicators.

```python
import pandas as pd
from vectrix import regress

series = {
    "EXKOUS": "exchangeRate",
    "IRLTLT01KRM156N": "bondYield",
    "LRUNTTTTMKM156S": "unemployment",
    "SPASTT01KRM661N": "kospi",
    "KORCPIALLMINMEI": "cpi",
}

dfs = []
for code, name in series.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    df = pd.read_csv(url)
    df.columns = ["date", name]
    df["date"] = pd.to_datetime(df["date"])
    dfs.append(df)

merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(merged, df, on="date", how="inner")
merged = merged.dropna()

result = regress(
    data=merged,
    formula="exchangeRate ~ bondYield + unemployment + kospi + cpi",
)
```

## Data Sources

| Source | Dataset | Auth |
|--------|---------|:----:|
| UCI ML Repository | Seoul Bike Sharing Demand | No |
| FRED | Korean Economic Indicators | No |

!!! note "Disclaimer"
    For educational purposes only.

---

**Next:** [Showcase 03 — Model Comparison](03_modelComparison.md)
