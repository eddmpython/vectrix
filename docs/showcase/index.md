# Showcase

Real-world forecasting and regression with **publicly available Korean data**.
No API key required — all data is fetched live from open sources.

## How to Run

```bash
pip install "vectrix[tutorials]"
marimo run docs/showcase/en/01_koreanEconomy.py
```

## Available Showcases

| # | Topic | English | Korean | Data Source |
|---|-------|---------|--------|-------------|
| 01 | Korean Economy Forecasting | `en/01_koreanEconomy.py` | `ko/01_koreanEconomy.py` | FRED |
| 02 | Korean Regression Analysis | `en/02_koreanRegression.py` | `ko/02_koreanRegression.py` | UCI / FRED |

## Showcase Descriptions

### 01 — Korean Economy Forecasting

Forecast real Korean economic indicators using data from FRED (Federal Reserve Economic Data):

- **USD/KRW Exchange Rate** — Monthly data from 1981, 12-month forecast
- **KOSPI Stock Index** — Monthly stock market index, 12-month forecast
- **Consumer Price Index (CPI)** — Inflation tracking, 12-month forecast
- **Multi-indicator DNA Analysis** — Compare difficulty and characteristics across 4 indicators

All data downloaded automatically from `fred.stlouisfed.org` — no registration needed.

### 02 — Korean Regression Analysis

Regression analysis on Korean datasets:

- **Seoul Bike Sharing Demand** — 8,760 hourly observations, predict rental count from weather features (UCI Machine Learning Repository)
- **Korean Macro Regression** — Exchange rate determinants: bond yield, unemployment, stock index, CPI (FRED)
- **Bike Data → Time Series** — Same dataset used for both regression AND forecasting

## Data Sources

| Source | URL | Auth Required |
|--------|-----|:---:|
| FRED | `fred.stlouisfed.org` | No |
| UCI ML Repository | `archive.ics.uci.edu` | No |
| Open-Meteo | `open-meteo.com` | No |

!!! note "Disclaimer"
    These analyses are for educational purposes only. Do not use for actual investment or business decisions.
