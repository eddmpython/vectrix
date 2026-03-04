---
title: "How Do We Know If a Forecast Is Any Good?"
---

# How Do We Know If a Forecast Is Any Good?

![Benchmarking — measuring what matters in forecasting](/vectrix/blog/benchmark-hero.svg)

In the [last post](/vectrix/blog/what-is-forecasting), we established that forecasting is about making informed predictions using historical data. We talked about trend, seasonality, and noise. We outlined the process from data collection to uncertainty quantification.

But we left out a critical question — **how do you know if your forecast is actually good?**

A forecast that says "sales will be between 0 and infinity" is technically never wrong. But it's also completely useless. The real challenge isn't just making predictions — it's making predictions that are **measurably better** than the alternatives.

This is where **benchmarks** come in.

---

## The Problem with "It Looks Right"

Imagine you build a forecasting model for monthly sales. It produces a nice-looking chart. The line goes up when the data goes up. It captures the seasonal bumps. Your manager glances at it and says, "Looks good."

But "looks good" is not a metric. Consider these scenarios

- Your model predicts 1,000 units. Actual sales are 1,100. Is that good?
- Your model predicts 1,000 units for a product that sells between 900 and 1,100 every month. Is the model adding any value, or is it just echoing the average?
- Your model is 5% more accurate than last year's model — but a simple "repeat last month" approach is 3% more accurate. Was the effort worth it?

Without a **systematic way to measure accuracy**, you can't answer any of these questions. You're flying blind, making decisions based on aesthetics rather than evidence.

---

## What Is a Benchmark?

A **benchmark** is a standardized test that lets you objectively compare forecasting methods against each other and against simple baselines.

![What makes a good benchmark](/vectrix/blog/benchmark-anatomy.svg)

A good benchmark has four components

**1. A dataset everyone agrees on.**
Not your company's proprietary data. Not a cherry-picked example. A large, diverse, publicly available collection of time series that represents real-world challenges.

**2. A baseline to beat.**
Usually the simplest reasonable method — like "repeat last year's value" or "use the historical average." If your sophisticated model can't beat this, it's not adding value.

**3. Clear rules.**
How much historical data can you use? How far ahead must you predict? Are you allowed to tune hyperparameters on the test set? Without strict rules, comparisons become meaningless.

**4. Agreed-upon metrics.**
Everyone measures accuracy the same way, using the same formulas, on the same data. No cherry-picking the metric that makes your method look best.

Think of it like the Olympics. Everyone runs the same distance, on the same track, measured by the same clock. That's what makes comparison meaningful.

---

## The M Competitions — Forecasting's Olympics

The most influential benchmarks in forecasting are the **M Competitions**, organized by Professor Spyros Makridakis starting in 1982.

![The M Competition timeline](/vectrix/blog/m-competition-timeline.svg)

### M1 (1982) — The Wake-Up Call

**111 time series.** The first large-scale comparison of forecasting methods. The shocking finding? **Simple methods often beat complex ones.** Sophisticated statistical models that took hours to fit were outperformed by basic exponential smoothing methods that took seconds.

This humbled the academic community and established a principle that holds to this day — **complexity does not guarantee accuracy.**

### M2 (1986) — Adding Judgment

**29 time series** with the twist that participants could use judgment and external information. The results reinforced M1's findings — simple methods remained competitive even when experts had access to additional context.

### M3 (2000) — Scaling Up

**3,003 time series** across Yearly, Quarterly, Monthly, and Other categories. The Theta method (Assimakopoulos & Nikolopoulos) stunned the community by winning with an elegantly simple approach — decomposing data into two "theta lines" and combining their forecasts.

Key takeaway: **combination and simplicity still won.** No single complex model dominated across all categories.

### M4 (2018) — The Game Changer

**100,000 time series** across 6 frequencies (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly). This was the big one — 50x larger than M3, representing the most comprehensive forecasting benchmark ever created.

The results reshaped the field

| Rank | Method | Type | OWA |
|:----:|--------|------|:---:|
| 1 | ES-RNN (Smyl) | Hybrid (LSTM + ETS) | 0.821 |
| 2 | FFORMA | Meta-learning ensemble | 0.838 |
| 3 | Theta (Fiorucci) | Statistical | 0.854 |
| 18 | Original Theta | Statistical | 0.897 |

**Three revolutionary findings**

1. **A hybrid method won for the first time.** ES-RNN combined deep learning (LSTM) with classical exponential smoothing — proving that the future lies in combining statistical rigor with machine learning flexibility.

2. **Pure machine learning methods performed poorly.** Standalone neural networks and ML models without statistical components couldn't compete with well-tuned statistical methods. The data wasn't enough — you needed structural assumptions too.

3. **Combination is king.** The top methods all combined multiple approaches. No single model, no matter how sophisticated, could dominate across all frequencies.

### M5 (2020) — Real Retail Data

**42,840 products from Walmart.** Unlike previous M Competitions that used anonymized data, M5 used real retail sales data from Walmart. This added real-world messiness — zero sales, promotions, holidays, and hierarchical product structures.

The winners were almost exclusively **gradient boosting** methods (LightGBM, XGBoost) — a stark contrast to M4 where statistical methods dominated. This revealed an important truth — **the best method depends on the data structure.**

### M6 (2022) — Uncertainty and Investment

Focused on **financial forecasting and uncertainty estimation.** It evaluated not just point forecasts but the quality of prediction intervals and investment decisions based on those forecasts.

---

## Why Benchmarks Matter Beyond Academia

You might think benchmarks are an academic exercise. They're not. Here's why they matter for practitioners

### 1. They prevent self-deception

It's easy to build a model, test it on your own data, and convince yourself it works. Benchmarks force honesty. If your method ranks 50th out of 60 on a standardized dataset, the problem is your method, not the data.

### 2. They guide tool selection

When choosing a forecasting library, you want evidence. "Our library uses advanced algorithms" is marketing. "Our library achieves OWA 0.885 on the M4 Competition dataset" is a measurable claim you can verify.

### 3. They reveal method strengths and weaknesses

No method wins everywhere. M4 showed that statistical methods excel on yearly data while machine learning methods dominate with large-volume daily/hourly data. Benchmarks help you match the right tool to the right problem.

### 4. They push the field forward

Every M Competition has produced breakthroughs. M1 proved simplicity matters. M3 gave us the Theta method. M4 launched the hybrid era. M5 validated gradient boosting for retail. Without standardized comparison, these discoveries might never have happened.

---

## The Metrics — How Accuracy Is Measured

Benchmarks need metrics. Here are the ones that matter most

### MAE — Mean Absolute Error

The simplest accuracy measure. How far off are you, on average?

```
MAE = average(|actual - forecast|)
```

If your MAE is 50, your predictions are off by 50 units on average. Simple, intuitive, but **scale-dependent** — an MAE of 50 means very different things for a product selling 100 units versus 10,000 units.

### MAPE — Mean Absolute Percentage Error

Makes errors scale-independent by expressing them as percentages.

```
MAPE = average(|actual - forecast| / actual) × 100%
```

A MAPE of 10% means you're off by 10% on average. Easy to communicate to stakeholders. But it has a fatal flaw — **it explodes when actual values are near zero** and penalizes over-forecasting more than under-forecasting.

### sMAPE — Symmetric MAPE

Fixes MAPE's asymmetry by using the average of actual and forecast in the denominator.

```
sMAPE = average(|actual - forecast| / ((|actual| + |forecast|) / 2)) × 100%
```

Used in the M3 and M4 Competitions. Bounded between 0% and 200%. More balanced than MAPE, but still has quirks with near-zero values.

### MASE — Mean Absolute Scaled Error

The gold standard for comparing across different series. It scales errors against a **naive seasonal forecast** — the simplest possible reasonable method.

```
MASE = MAE of your model / MAE of naive seasonal forecast
```

- MASE below 1.0 → Your model beats the naive method
- MASE equal to 1.0 → Your model equals the naive method
- MASE above 1.0 → Your model is worse than just repeating last year

This is the metric that keeps forecasters honest. If you can't beat "repeat what happened last year," your model isn't adding value.

### OWA — Overall Weighted Average

The M4 Competition's headline metric. Combines sMAPE and MASE relative to a Naive2 baseline

```
OWA = 0.5 × (your sMAPE / Naive2 sMAPE) + 0.5 × (your MASE / Naive2 MASE)
```

OWA below 1.0 means you beat Naive2. The lower, the better. This single number captures both percentage accuracy and scaled accuracy in one comparable metric.

---

## The Naive Baseline — Your Minimum Bar

Every benchmark revolves around a **baseline** — the simplest reasonable prediction. In forecasting, this is usually the **Naive method**

- **Naive1** — Repeat the last observed value. "Tomorrow will be like today."
- **Naive2** — Repeat last year's value, adjusted for trend. "Next January will be like last January, plus growth."
- **Seasonal Naive** — Repeat the value from the same season last year. "This December will be like last December."

These sound absurdly simple, and they are. But here's the uncomfortable truth — **a surprising number of sophisticated models can't consistently beat them.**

The M4 Competition showed that across 100,000 series, the average submission barely outperformed Naive2. Many individual models, including some using deep learning, scored OWA above 1.0 — meaning they were **worse than doing nothing sophisticated at all.**

This is why benchmarking against naive methods isn't optional — it's the **minimum bar for credibility.**

---

## What Does This Mean for You?

If you're choosing a forecasting tool or building your own models, here's what benchmarks teach us

**1. Always compare against naive baselines.**
Before celebrating your model's accuracy, check: does it beat Seasonal Naive? If not, use Seasonal Naive and save yourself the complexity.

**2. Test across diverse data.**
A model that works brilliantly on monthly retail data might fail miserably on hourly energy data. Use benchmarks that cover multiple frequencies and domains.

**3. Look at multiple metrics.**
MAPE alone doesn't tell the full story. Look at MASE (does it beat naive?) and prediction interval coverage (are the uncertainty bounds calibrated?).

**4. Be skeptical of claims without evidence.**
"Our AI-powered forecasting achieves 99% accuracy" is almost certainly misleading. Ask: on what dataset? Against what baseline? Using what metric? Over what horizon?

**5. Combination usually wins.**
The most consistent finding across 40 years of M Competitions: combining multiple models outperforms any single model. Don't bet everything on one approach.

---

## Where Vectrix Stands

Transparency matters. Here's how Vectrix performs on the M4 benchmark, using 2,000 randomly sampled series per frequency

| Frequency | Vectrix OWA | Context |
|-----------|:-----------:|---------|
| Yearly | **0.797** | Near M4 winner level |
| Quarterly | **0.905** | Competitive with top methods |
| Monthly | **0.933** | Solid mid-table |
| Weekly | **0.959** | Beats Naive2 |
| Daily | **0.996** | Near parity with Naive2 |
| Hourly | **0.722** | World-class |
| **Average** | **0.885** | **Outperforms M4 #18 Theta (0.897)** |

These numbers aren't cherry-picked or inflated. They represent honest performance — strong in some frequencies, room for improvement in others. We publish our benchmark code so you can [reproduce every number](https://eddmpython.github.io/vectrix/docs/benchmarks/).

That's what benchmarks are for. Not marketing. **Evidence.**

---

## What's Next?

Now you know what forecasting is (Post 1) and how we measure whether it works (Post 2). In upcoming posts

- **Your First Forecast in Python** — From zero to prediction in 5 minutes
- **Time Series Models Explained** — ETS, ARIMA, Theta, and when to use each
- **The Art of Model Selection** — How auto-selection works and why it matters

Forecasting without measurement is guessing. Measurement without benchmarks is wishful thinking. Now you have the framework to tell the difference.

---

*Want to see these benchmarks in action? [Install Vectrix](https://eddmpython.github.io/vectrix/docs/getting-started/installation) and run your own M4 benchmark in minutes.*
