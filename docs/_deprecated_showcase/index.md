# Showcase

Real-world forecasting examples with **publicly available data**.
Each showcase has a detailed guide below and an interactive marimo notebook you can run locally.

## Available Showcases

| # | Topic | Guide | Interactive |
|---|-------|-------|-------------|
| 01 | [Korean Economy Forecasting](01_koreanEconomy.md) | FRED economic indicators | `marimo run docs/showcase/en/01_koreanEconomy.py` |
| 02 | [Korean Regression Analysis](02_koreanRegression.md) | Bike sharing + macro regression | `marimo run docs/showcase/en/02_koreanRegression.py` |
| 03 | [Model Comparison](03_modelComparison.md) | 30+ models side-by-side | `marimo run docs/showcase/en/03_modelComparison.py` |
| 04 | [Business Intelligence](04_businessIntelligence.md) | Anomaly, scenarios, backtesting | `marimo run docs/showcase/en/04_businessIntelligence.py` |

## How to Run Interactively

Showcases are built with [marimo](https://marimo.io) — reactive Python notebooks.

```bash
pip install vectrix pandas numpy marimo
marimo run docs/showcase/en/01_koreanEconomy.py
```

Or browse the guides above to see code and explanations directly on this site.

## Showcase Descriptions

### 01 — Korean Economy Forecasting

Forecast real Korean economic indicators using FRED data:

- **USD/KRW Exchange Rate** — Monthly data from 1981, 12-month forecast
- **KOSPI Stock Index** — Monthly stock market index, 12-month forecast
- **Consumer Price Index (CPI)** — Inflation tracking, 12-month forecast
- **Multi-indicator DNA Analysis** — Compare difficulty and characteristics

### 02 — Korean Regression Analysis

Regression analysis on Korean datasets:

- **Seoul Bike Sharing Demand** — 8,760 hourly observations, predict rental count from weather (UCI ML Repository)
- **Korean Macro Regression** — Exchange rate determinants from FRED indicators

### 03 — Model Comparison & Adaptive Intelligence

Compare all 30+ forecasting models:

- **DNA Analysis** — Data difficulty, category, fingerprint
- **Model Ranking** — Every model's MAPE, RMSE, MAE, sMAPE
- **All Forecasts DataFrame** — All model predictions side-by-side
- **Quick Compare** — One-liner `compare()` function

### 04 — Business Intelligence

End-to-end business workflow:

- **Anomaly Detection** — Find unusual data points
- **What-If Scenarios** — Growth, recession, supply shock analysis
- **Backtesting** — Walk-forward model validation
- **Business Metrics** — MAPE, RMSE, MAE, bias, tracking signal

## Data Sources

| Source | URL | Auth Required |
|--------|-----|:---:|
| FRED | `fred.stlouisfed.org` | No |
| UCI ML Repository | `archive.ics.uci.edu` | No |

!!! note "Disclaimer"
    These analyses are for educational purposes only. Do not use for actual investment or business decisions.
