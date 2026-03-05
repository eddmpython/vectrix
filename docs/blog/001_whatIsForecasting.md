---
title: "What Is Forecasting?"
---

# What Is Forecasting?

![Forecasting — using historical data to predict the future](/vectrix/docs/blog/assets/forecasting-hero.svg)

Every decision you make is a forecast. When you grab an umbrella before leaving home, you're forecasting rain. When a company orders extra inventory before the holidays, it's forecasting demand. When a government adjusts interest rates, it's forecasting inflation.

**Forecasting is the art and science of making informed predictions about the future using available information.**

That's it. No complex formulas needed to understand the concept. But beneath this simple definition lies a discipline that powers trillion-dollar decisions every day.

---

## Why Forecasting Matters

Consider these numbers

- A **1% improvement** in demand forecasting can save a retailer **$10 million** annually in reduced waste and stockouts
- Airlines use forecasting to set prices for over **3 billion** passenger trips each year
- Energy grids forecast power demand every **15 minutes** to keep the lights on
- Central banks forecast GDP and inflation to set policies that affect **billions** of people

Forecasting isn't an academic exercise. It's the invisible engine behind modern economies. Every supply chain, every financial market, every hospital staffing plan relies on some form of prediction.

> "Prediction is very difficult, especially about the future." — Niels Bohr

Bohr was right — but we don't need perfect predictions. We need predictions that are **good enough** to make better decisions than guessing.

---

## The Two Camps: Qualitative vs. Quantitative

Broadly, forecasting approaches fall into two camps.

![Qualitative vs. Quantitative forecasting approaches](/vectrix/docs/blog/assets/qualitative-vs-quantitative.svg)

### Qualitative Forecasting

This relies on human judgment, expertise, and intuition. Examples include

- **Expert opinion**: A seasoned retail buyer predicting next season's trends
- **Delphi method**: A panel of experts iteratively refining their predictions
- **Market research**: Surveys and focus groups gauging consumer intent
- **Scenario planning**: "What if oil prices double? What if a new competitor enters?"

Qualitative methods shine when **data is scarce or nonexistent** — launching a brand-new product, entering an unknown market, or predicting the impact of unprecedented events.

### Quantitative Forecasting

This uses mathematical models applied to historical data. This is where tools like Vectrix live. Examples include

- **Time series models**: Analyzing patterns in sequential data (ETS, ARIMA, Theta)
- **Regression models**: Finding relationships between variables ("sales increase by X when temperature rises by Y")
- **Machine learning**: Neural networks, gradient boosting, and ensemble methods
- **Foundation models**: Pre-trained deep learning models that generalize across datasets

Quantitative methods shine when **historical data exists and the future somewhat resembles the past**.

In practice, the best forecasters combine both. Numbers inform judgment; judgment corrects numbers.

---

## The Three Patterns Every Forecaster Must Know

When you look at any time series data, you're looking for three fundamental patterns

### 1. Trend

Is the data going up, going down, or staying flat over time?

A company's revenue growing 10% year-over-year has an **upward trend**. A declining population in a rural town shows a **downward trend**. A stable utility bill shows **no trend**.

Trend tells you the direction. It's the "big picture" of your data.

### 2. Seasonality

Are there repeating patterns at regular intervals?

Ice cream sales peak every summer. E-commerce traffic spikes every Black Friday. Hospital admissions rise every flu season. These are **seasonal patterns** — they repeat at predictable intervals.

Seasonality can be
- **Daily**: Restaurant traffic peaks at lunch and dinner
- **Weekly**: Gym attendance drops on weekends
- **Monthly**: Rent payments on the 1st of each month
- **Yearly**: Holiday shopping in December

### 3. Noise

Everything that isn't trend or seasonality is noise — random fluctuations that can't be predicted.

A sudden spike in website traffic because a celebrity mentioned your product. An unexpected dip in sales because a water main broke outside your store. These are **random events** that no model can foresee.

The goal of forecasting is to capture the **signal** (trend + seasonality) and accept the **noise** as irreducible uncertainty.

![Time Series Decomposition — Data = Trend + Seasonality + Noise](/vectrix/docs/blog/assets/time-series-decomposition.svg)

That's the fundamental equation. Every forecasting model, from the simplest to the most complex, is trying to separate signal from noise.

---

## How Forecasting Actually Works (The 10,000-Foot View)

Here's the process, stripped to its essence

![The 6-step forecasting process](/vectrix/docs/blog/assets/forecasting-process.svg)

**Step 1: Collect historical data.**
You need past observations. Monthly sales for the last 3 years. Daily temperatures for the last decade. Hourly website traffic for the last quarter. More data usually means better forecasts — but not always.

**Step 2: Identify patterns.**
Look for the three patterns: trend, seasonality, and noise. Is there a clear direction? Are there repeating cycles? How "noisy" is the data?

**Step 3: Choose a model.**
Select a mathematical model that can capture the patterns you've identified. Simple data might need a simple model. Complex data might need a sophisticated one. The best model is the simplest one that captures the patterns adequately.

**Step 4: Fit the model.**
Feed your historical data into the model. The model "learns" the patterns — estimating trend slopes, seasonal factors, and noise levels.

**Step 5: Generate forecasts.**
Use the fitted model to project into the future. The model extrapolates the learned patterns forward.

**Step 6: Quantify uncertainty.**
Every forecast is uncertain. Good forecasters don't just give a number — they give a **range**. "We expect 1,000 units, but it could be anywhere from 800 to 1,200."

**Step 7: Evaluate and iterate.**
Compare your forecasts against what actually happened. Learn from errors. Adjust. Repeat.

---

## The Honest Truth About Forecasting

Let's be upfront about what forecasting can and cannot do.

**Forecasting CAN**
- Identify likely ranges for future values
- Detect and extrapolate existing patterns
- Quantify the uncertainty in predictions
- Provide a disciplined framework for decision-making
- Improve consistently with better data and methods

**Forecasting CANNOT**
- Predict black swan events (pandemics, wars, market crashes)
- Guarantee accuracy — all forecasts are wrong, some are useful
- Replace domain expertise and human judgment
- Work well without sufficient historical data
- Predict the future when the future is fundamentally different from the past

The legendary statistician George Box said it best

> "All models are wrong, but some are useful."

The goal isn't perfection. The goal is to be **less wrong** than the alternative — which is often gut feeling, wishful thinking, or no plan at all.

---

## What's Next?

This post covered the "what" and "why" of forecasting. In upcoming posts, we'll dive deeper

- **[How Do We Know If a Forecast Is Any Good?](/vectrix/blog/how-we-know-forecasts-work)** — Benchmarks, metrics, and the M Competitions
- **[Python Forecasting Libraries Compared](/vectrix/blog/python-forecasting-libraries)** — Which tool should you use?
- **Your First Forecast in Python** — From data to prediction in 5 minutes
- **Time Series Models Explained** — ETS, ARIMA, Theta, and when to use each

Forecasting is a journey from intuition to evidence. Welcome aboard.

---

*Want to try forecasting right now? [Install Vectrix](https://eddmpython.github.io/vectrix/docs/getting-started/installation) and run your first forecast in one line of code.*
