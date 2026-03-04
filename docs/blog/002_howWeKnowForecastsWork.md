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

The most influential benchmarks in forecasting are the **[M Competitions](https://en.wikipedia.org/wiki/Makridakis_Competitions)**, organized by Professor Spyros Makridakis starting in 1982. The "M" stands for Makridakis — the researcher who had the audacity to ask, "Do complex models actually work better?"

![The M Competition timeline](/vectrix/blog/m-competition-timeline.svg)

### M1 (1982) — The Wake-Up Call

**111 time series.** The first large-scale comparison of forecasting methods. The shocking finding? **Simple methods often beat complex ones.** Sophisticated statistical models that took hours to fit were outperformed by basic exponential smoothing methods that took seconds.

This humbled the academic community and established a principle that holds to this day — **complexity does not guarantee accuracy.**

> Original paper: [Makridakis, S. et al. (1982). "The accuracy of extrapolation (time series) methods"](https://doi.org/10.1002/for.3980010202), *Journal of Forecasting*

### M2 (1986) — Adding Judgment

**29 time series** with the twist that participants could use judgment and external information alongside their models. The results reinforced M1's findings — simple methods remained competitive even when experts had access to additional context.

### M3 (2000) — Scaling Up

**3,003 time series** across Yearly, Quarterly, Monthly, and Other categories. The [Theta method](https://doi.org/10.1016/S0169-2070(00)00066-2) (Assimakopoulos & Nikolopoulos) stunned the community by winning with an elegantly simple approach — decomposing data into two "theta lines" and combining their forecasts.

Key takeaway: **combination and simplicity still won.** No single complex model dominated across all categories.

> Original paper: [Makridakis & Hibon (2000). "The M3-Competition"](https://doi.org/10.1016/S0169-2070(00)00057-1), *International Journal of Forecasting*

### M4 (2018) — The Game Changer

**[100,000 time series](https://github.com/Mcompetitions/M4-methods)** across 6 frequencies (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly). This was the big one — 50x larger than M3, representing the most comprehensive forecasting benchmark ever created.

The M4 was run as an open competition on the [M4 Competition website](https://mofc.unic.ac.cy/m4/), where 49 teams from around the world submitted their methods. The results reshaped the field

| Rank | Method | Type | OWA |
|:----:|--------|------|:---:|
| 1 | ES-RNN (Smyl) | Hybrid (LSTM + ETS) | 0.821 |
| 2 | FFORMA (Montero-Manso) | Meta-learning ensemble | 0.838 |
| 3 | Theta (Fiorucci) | Statistical | 0.854 |
| 18 | Original Theta | Statistical | 0.897 |

**Three revolutionary findings**

1. **A hybrid method won for the first time.** ES-RNN combined deep learning (LSTM networks — a type of neural network that remembers long-term patterns) with classical exponential smoothing (ETS — a statistical method that weights recent observations more heavily). This proved that the future lies in combining statistical rigor with machine learning flexibility.

2. **Pure machine learning methods performed poorly.** Standalone neural networks and ML models without statistical components couldn't compete with well-tuned statistical methods. Having lots of data wasn't enough — you needed structural assumptions about how time series behave (like trend and seasonality) too.

3. **Combination is king.** The top methods all combined multiple approaches. No single model, no matter how sophisticated, could dominate across all 6 frequencies. This echoed every previous M Competition.

> Original paper: [Makridakis, Spiliotis & Assimakopoulos (2020). "The M4 Competition"](https://doi.org/10.1016/j.ijforecast.2019.04.014), *International Journal of Forecasting*

### M5 (2020) — Real Retail Data

**[42,840 products from Walmart](https://mofc.unic.ac.cy/m5/).** Unlike previous M Competitions that used anonymized data, M5 used real retail sales data from Walmart stores. This added real-world messiness — zero sales days, promotional effects, holiday spikes, and hierarchical product structures (item → department → store → state).

The winners were almost exclusively **gradient boosting** methods (LightGBM, XGBoost) — a stark contrast to M4 where statistical methods dominated. Why? Because M5 included **external features** (prices, promotions, calendar events) that tree-based ML models can exploit directly, while traditional time series models only look at the history of the series itself. This revealed an important truth — **the best method depends on the data structure.**

> Original paper: [Makridakis, Spiliotis & Assimakopoulos (2022). "M5 accuracy competition"](https://doi.org/10.1016/j.ijforecast.2021.11.013), *International Journal of Forecasting*

### M6 (2022) — Uncertainty and Investment

The [M6 Competition](https://mofc.unic.ac.cy/m6/) focused on **financial forecasting and uncertainty estimation.** Unlike previous competitions that measured point forecast accuracy, M6 evaluated how well participants could estimate the *probability distribution* of future outcomes — and whether their predictions were good enough to make profitable investment decisions.

This was a fundamentally different challenge. In financial markets, being "approximately right" isn't enough — you need to know *how confident* you are in your prediction, because that determines how much to bet.

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

Benchmarks need metrics — standardized formulas that turn "how wrong was this forecast?" into a single comparable number. Each metric answers a slightly different question, which is why competitions use multiple metrics together.

### MAE — Mean Absolute Error

The simplest accuracy measure. How far off are you, on average?

```
MAE = average(|actual - forecast|)
```

**Example.** You predicted 100, 200, 300 units for three months. Actual sales were 110, 180, 320. Your errors are 10, 20, 20. MAE = (10 + 20 + 20) / 3 = **16.7 units**.

Simple and intuitive, but **scale-dependent** — an MAE of 50 means very different things for a product selling 100 units versus 10,000 units. You can't use MAE to compare forecasts across different products or datasets.

### MAPE — Mean Absolute Percentage Error

Solves MAE's scale problem by expressing errors as percentages of the actual values.

```
MAPE = average(|actual - forecast| / actual) × 100%
```

**Example.** Same numbers. Errors as percentages: 10/110 = 9.1%, 20/180 = 11.1%, 20/320 = 6.3%. MAPE = **8.8%**. Now you can compare across products — "Product A has 8.8% MAPE, Product B has 15% MAPE."

Easy to communicate to stakeholders ("we're off by about 9%"). But MAPE has two fatal flaws

- **It explodes when actual values are near zero.** If actual sales = 1 unit and you predicted 2, MAPE = 100%. If actual = 0, MAPE is undefined (division by zero). This makes MAPE useless for intermittent demand (products that sell 0 on many days).
- **It's asymmetric.** Predicting 150 when actual is 100 gives MAPE = 50%. But predicting 50 when actual is 100 also gives MAPE = 50%. Seems fair? Not quite — the formula punishes under-forecasting more than over-forecasting due to the smaller denominator.

### sMAPE — Symmetric MAPE

Fixes MAPE's asymmetry by using the average of actual and forecast in the denominator instead of just the actual.

```
sMAPE = average(2 × |actual - forecast| / (|actual| + |forecast|)) × 100%
```

**Why "symmetric"?** Because predicting 150 when actual is 100 now gives the same error as predicting 100 when actual is 150 — the denominator is the same in both cases ((100+150)/2 = 125). This removes the bias that MAPE has.

Used as the primary metric in the M3 and M4 Competitions. Bounded between 0% and 200%. More balanced than MAPE, but still has quirks with near-zero values.

### MASE — Mean Absolute Scaled Error

This is where it gets interesting. MASE asks a fundamentally different question: **"How much better is your model than the simplest possible approach?"**

Instead of measuring absolute error or percentage error, MASE compares your model's errors to the errors of a **naive seasonal forecast** — a method that simply repeats what happened one season ago.

```
MASE = MAE of your model / MAE of naive seasonal forecast
```

**Example.** Your model has MAE = 16.7 units. The naive method (just repeating last year's values) has MAE = 25 units. MASE = 16.7 / 25 = **0.67**. Your model is 33% better than doing nothing.

The interpretation is intuitive

- MASE below 1.0 → Your model beats the naive method (good)
- MASE equal to 1.0 → Your model is no better than naive (bad — why bother?)
- MASE above 1.0 → Your model is worse than naive (very bad — you'd be better off using the simplest possible approach)

MASE was [proposed by Rob Hyndman](https://doi.org/10.1016/j.ijforecast.2006.03.001) in 2006 and has become the gold standard for comparing forecasts across different series. Because it's scaled relative to naive, a MASE of 0.7 means the same thing whether you're forecasting daily retail sales or yearly GDP.

### OWA — Overall Weighted Average

The M4 Competition needed a single number to rank all 49 submissions. OWA was designed for exactly this — it combines sMAPE and MASE into one headline metric, both measured relative to the **Naive2** baseline (explained in the next section)

```
OWA = 0.5 × (your sMAPE / Naive2 sMAPE) + 0.5 × (your MASE / Naive2 MASE)
```

**Why two metrics?** Because sMAPE measures *percentage accuracy* (are you close in relative terms?) while MASE measures *scaled accuracy* (are you better than naive?). A model could game one metric but not both. By combining them with equal weight, OWA gives a balanced picture.

**Example.** Your model's sMAPE = 11.5%, Naive2's sMAPE = 13.0%. Your MASE = 0.85, Naive2's MASE = 1.0. OWA = 0.5 × (11.5/13.0) + 0.5 × (0.85/1.0) = 0.5 × 0.885 + 0.5 × 0.85 = **0.867**.

OWA below 1.0 means you beat Naive2. The lower, the better. In the M4 Competition, the winning method (ES-RNN) achieved OWA 0.821, meaning it was about 18% better than Naive2 across all 100,000 series.

---

## The Naive Baselines — Your Minimum Bar

Every benchmark revolves around a **baseline** — the simplest reasonable prediction. If your fancy model can't beat the baseline, it's not adding value. In forecasting, several naive methods serve as baselines, each capturing a different aspect of "doing the obvious thing."

### Naive1 (Random Walk)

The simplest possible forecast. Whatever the last value was, predict that it continues forever.

```
Forecast for any future period = last observed value
```

**Example.** Sales last month were 1,000 units. Naive1 predicts 1,000 for next month, the month after, and every month after that. No trend, no seasonality — just "things stay as they are."

This sounds useless, but for highly volatile data like stock prices or exchange rates, Naive1 is surprisingly hard to beat. That's why the "random walk hypothesis" in finance says stock prices already contain all available information — the best prediction of tomorrow's price is today's price.

### Seasonal Naive

Repeat the value from the same season last year. This captures seasonality but ignores trend.

```
Forecast for month M = actual value from month M last year
```

**Example.** Sales in December 2024 were 3,000 units. Seasonal Naive predicts December 2025 will also be 3,000. January 2025 forecast = January 2024 actual. And so on.

This is the baseline that MASE uses for its denominator. If your model can't beat "just repeat last year's pattern," it has no value.

### Naive2 — The M4 Benchmark Baseline

This is the critical one. **Naive2 is the official baseline of the M4 Competition**, and it's more sophisticated than it sounds.

Naive2 works in two steps

**Step 1 — Deseasonalize.** If the data has seasonality, Naive2 first removes it using classical seasonal decomposition. This separates the data into a trend+remainder component and a seasonal component.

**Step 2 — Apply Naive1 to the deseasonalized data.** Predict the deseasonalized series using the last observed value (random walk), then add the seasonal pattern back.

```
Naive2 = Naive1 applied to deseasonalized data, then reseasonalized
```

**Why not just Seasonal Naive?** Because Naive2 adapts to the most recent level of the data. If your sales have been trending upward, Seasonal Naive would predict last year's (lower) values. Naive2 captures last year's seasonal *shape* but at this year's *level* — a surprisingly good prediction for many series.

**For non-seasonal data** (no repeating pattern), Naive2 reduces to plain Naive1 — just repeat the last value.

This is why OWA uses Naive2 as the denominator. It's a strong baseline that already handles both level changes and seasonality. Beating it requires your model to capture patterns that go beyond these two basic features — like complex trend dynamics, multiple seasonal cycles, or non-linear relationships.

### Why Baselines Matter So Much

These sound absurdly simple, and they are. But here's the uncomfortable truth — **a surprising number of sophisticated models can't consistently beat them.**

The M4 Competition showed that across 100,000 series, the average submission barely outperformed Naive2. Many individual models, including some using deep learning, scored OWA above 1.0 — meaning they were **worse than doing nothing sophisticated at all.**

Think about what that means. Teams spent weeks or months developing complex algorithms, training deep neural networks, tuning hundreds of hyperparameters — and the result was worse than a method you could implement in 10 lines of code.

This is why benchmarking against naive methods isn't optional — it's the **minimum bar for credibility.** If someone tells you their model "achieves 95% accuracy" but doesn't mention what the naive baseline achieves, the number is meaningless. Maybe naive achieves 96%.

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
