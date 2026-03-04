---
title: Korean Regression Analysis
---

# Korean Regression Analysis

Two regression analyses using real Korean datasets: Seoul Bike Sharing Demand (UCI ML Repository) and Korean Macroeconomic Regression (FRED).

## Data Sources

| Dataset | Source | Observations | Description |
|---------|--------|:------------:|-------------|
| Seoul Bike Sharing | [UCI ML Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) | 8,760 (hourly) | Hourly bike rental count with weather conditions |
| Korean Macro Regression | [FRED](https://fred.stlouisfed.org/) | ~300 (monthly) | Exchange rate, interest rate, trade balance, CPI |

## Seoul Bike Sharing Demand

Analyzing how weather conditions affect hourly bike rental demand in Seoul.

### Loading and Preparing Data

```python
import pandas as pd
from vectrix import regress

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv"
bikeDf = pd.read_csv(url, encoding="unicode_escape")

colMap = {
    "Rented Bike Count": "rentedBikeCount",
    "Hour": "hour",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "windSpeed",
    "Rainfall(mm)": "rainfall",
    "Solar Radiation (MJ/m2)": "solarRadiation",
}

tempCols = [c for c in bikeDf.columns if "Temperature" in c and "Dew" not in c]
if tempCols:
    colMap[tempCols[0]] = "temperature"

bikeDf = bikeDf.rename(columns=colMap)
```

### Running the Regression

```python
result = regress(
    data=bikeDf,
    formula="rentedBikeCount ~ temperature + humidity + windSpeed + rainfall + hour + solarRadiation",
    summary=False,
)

print(f"R-squared:     {result.r_squared:.4f}")
print(f"Adj R-squared: {result.adj_r_squared:.4f}")
print(f"F-statistic:   {result.f_stat:.2f}")
```

### Coefficient Table

```python
labels = ["Intercept", "Temperature", "Humidity", "Wind Speed",
          "Rainfall", "Hour", "Solar Radiation"]

for name, coef, pval in zip(labels, result.coefficients, result.pvalues):
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"{name:20s}  {coef:+10.3f}  p={pval:.4f} {sig}")
```

### Interpretation

- **Temperature** +1C increases bike rentals (warmer = more outdoor activity)
- **Humidity** +1% decreases bike rentals (uncomfortable conditions)
- **Rainfall** +1mm sharply decreases rentals (rain deters cycling)

### Diagnostics

```python
print(result.diagnose())
```

This runs Durbin-Watson (autocorrelation), Breusch-Pagan (heteroscedasticity), VIF (multicollinearity), and Jarque-Bera (normality) tests.

## Korean Macro Regression

Modeling the USD/KRW exchange rate as a function of macroeconomic factors.

### Data from FRED

```python
import pandas as pd
from vectrix import regress

series = {
    "EXKOUS": "exchangeRate",
    "INTDSRKRM193N": "interestRate",
    "XTEXVA01KRM667S": "exports",
    "KORCPIALLMINMEI": "cpi",
}

frames = []
for code, name in series.items():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    tmp = pd.read_csv(url)
    tmp.columns = ["date", name]
    tmp["date"] = pd.to_datetime(tmp["date"])
    frames.append(tmp.set_index("date"))

macroDf = pd.concat(frames, axis=1).dropna().reset_index()
```

### Running the Regression

```python
result = regress(
    data=macroDf,
    formula="exchangeRate ~ interestRate + exports + cpi",
    summary=False,
)

print(f"R-squared:     {result.r_squared:.4f}")
print(f"Adj R-squared: {result.adj_r_squared:.4f}")
print(f"F-statistic:   {result.f_stat:.2f}")
print(result.summary())
```

### Diagnostics

```python
print(result.diagnose())
```

> **Note:** Macroeconomic regression on levels often exhibits autocorrelation and non-stationarity. For production analysis, consider differencing or cointegration tests. This showcase demonstrates the API, not econometric best practices.

> **Disclaimer:** For educational purposes only. Data source: [UCI ML Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand) and [FRED](https://fred.stlouisfed.org/).
