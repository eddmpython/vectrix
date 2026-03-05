---
title: "Forecasting Models Explained — What ETS, ARIMA, Theta, and Friends Actually Do"
description: "A deep dive into all 22 time series forecasting models. Learn what ETS, Holt-Winters, CES, Theta, DOT, FourTheta, ARIMA, TBATS, MSTL, GARCH, Croston, DTSF, and ESN do, when each shines, when each struggles, and why having many diverse models matters."
---

# Forecasting Models Explained — What ETS, ARIMA, Theta, and Friends Actually Do

![Different models see the same data through different lenses](/vectrix/blog/assets/models-hero.svg)

In [Post 3](/vectrix/blog/python-forecasting-libraries), we compared forecasting *libraries*. In [Post 4](/vectrix/blog/your-first-forecast-in-python), we ran our first forecast. Now comes the deeper question — **what was the model actually doing?**

When you call `forecast(data, steps=12)`, Vectrix evaluates 22 models behind the scenes. But what *are* those models? Why are there so many? Do they all do the same thing in slightly different ways?

No. Each model looks at your data through a fundamentally different lens. ETS decomposes your data into building blocks. ARIMA studies how today's value depends on yesterday's. Theta controls how aggressively to project a trend. GARCH tracks volatility itself.

This post explains every major model family — not the math (you can read papers for that), but the *intuition*. What does each model see when it looks at your data? When does it shine? When does it fail? And why does having many different perspectives make the final forecast better?

---

## Why So Many Models?

Here's the fundamental insight: **no model is universally best**.

The M3 Competition (2000) tested 24 methods on 3,003 time series. The winner? Theta — a simple method that barely anyone had heard of. It beat complex neural networks and sophisticated state-space models.

The M4 Competition (2018) tested even more methods on 100,000 series. The winner? A hybrid of statistical and neural approaches. Theta still did well. ARIMA did okay. Simple exponential smoothing did surprisingly okay too.

The lesson is always the same: different data needs different models. Monthly airline passengers follow smooth seasonal curves — ETS handles that perfectly. Daily stock returns jump around chaotically — GARCH is built for that. Spare parts demand is 90% zeros — Croston specializes in exactly that.

![Forecasting model families — smoothing, theta, ARIMA, specialists, baselines](/vectrix/blog/assets/model-family-tree.svg)

The real power isn't in any single model. It's in having a *diverse* set of models that make *different kinds of errors*, then combining them intelligently. That's why Vectrix ships 22 models instead of 1.

Let's meet each family.

---

## The Smoothing Family — Level, Trend, Season

### ETS (Error, Trend, Seasonal)

ETS is the Swiss Army knife of forecasting. It decomposes your time series into three building blocks:

- **Level** — where the series is right now (the "center of gravity")
- **Trend** — whether it's going up, down, or staying flat
- **Seasonal** — the repeating pattern (weekly, monthly, yearly cycles)

![ETS decomposition — original series broken into level, trend, seasonal, and error](/vectrix/blog/assets/ets-decomposition.svg)

Each component can behave differently. The trend might be additive (growing by a fixed amount each period) or multiplicative (growing by a fixed percentage). Same for seasonality. This gives ETS 18 possible configurations — from the simplest "just track the level" to the most complex "multiplicative trend with multiplicative seasonality and damping".

**The smoothing part**: ETS uses exponentially weighted averages. Recent observations matter more than old ones. How much more? That's controlled by smoothing parameters (α for level, β for trend, γ for seasonality). Small α means the model has a long memory and changes slowly. Large α means it reacts quickly to new data.

**AutoETS** searches through all valid configurations and picks the one with the best information criterion (AICc) — balancing fit quality against model complexity.

```python
from vectrix.engine.ets import AutoETS
import numpy as np

data = np.array([120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215,
                 125, 140, 155, 138, 160, 175, 168, 185, 200, 190, 210, 225])

model = AutoETS(period=12)
model.fit(data)
predictions, lower, upper = model.predict(steps=6)
print(f"Next 6: {predictions.round(1)}")
```

**When ETS shines**: Stable patterns with clear trend and/or seasonality. Monthly sales, quarterly revenue, annual temperature cycles. If your data follows a recognizable rhythm, ETS will capture it.

**When ETS struggles**: Sudden regime changes (ETS adapts slowly). Data with multiple seasonal periods (ETS handles only one). Highly nonlinear dynamics. Intermittent demand (lots of zeros).

**Key insight**: ETS is a *generative* model — it tells a story about *how* the data was created (level shifted by trend, modulated by season, corrupted by noise). This makes it highly interpretable.

---

### ETS Variants — When You Know Your Data

AutoETS searches all 18 configurations and picks the winner. But sometimes *you* know something AutoETS doesn't. Vectrix registers two fixed ETS configurations as standalone models:

**ETS(A,A,N) — Holt's Linear Method**

This is ETS with additive error, additive trend, and *no* seasonality. It's the right tool when your data trends up or down but has no repeating seasonal pattern.

Think quarterly GDP, population growth, or cumulative user signups. There's a clear direction, but no "January is always high" pattern. Holt's Linear captures the level and the slope, nothing more.

```python
from vectrix.engine.ets import ETSModel
import numpy as np

data = np.array([100, 108, 114, 123, 130, 140, 148, 157, 165, 175, 183, 192])

model = ETSModel('A', 'A', 'N', period=1)
model.fit(data)
predictions, _, _ = model.predict(steps=6)
print(f"Holt's Linear: {predictions.round(1)}")
```

**When it shines**: Clean trending data without seasonality. Faster than AutoETS because it doesn't search — it just fits.

**When it struggles**: Seasonal data (it ignores seasons entirely). Nonlinear trends (it assumes constant slope).

**ETS(A,A,A) — Holt-Winters Additive**

The classic Holt-Winters method: additive error, additive trend, additive seasonality. This is what most people picture when they think "exponential smoothing with seasonality."

Monthly ice cream sales go up every summer by roughly the *same amount* (additive seasonality). Holt-Winters tracks the level, the trend slope, and a seasonal index for each period.

```python
from vectrix.engine.ets import ETSModel
import numpy as np

data = np.array([120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215,
                 125, 140, 155, 138, 160, 175, 168, 185, 200, 190, 210, 225])

model = ETSModel('A', 'A', 'A', period=12)
model.fit(data)
predictions, _, _ = model.predict(steps=6)
print(f"Holt-Winters: {predictions.round(1)}")
```

**When it shines**: Classic seasonal data where the seasonal amplitude stays roughly constant over time (the December spike is always about +30 units, not +30%).

**When it struggles**: Multiplicative seasonality (seasonal amplitude grows with the level). If your December spike doubles when your overall sales double, you need ETS(A,A,M) or ETS(M,A,M) — let AutoETS find it.

**Why both exist alongside AutoETS**: In ensembles, fixed-configuration models provide *stability*. AutoETS might switch from ETS(A,A,A) to ETS(M,N,M) on slightly different data, creating forecast instability. The fixed variants are anchors — their behavior is predictable, which helps the ensemble stay grounded.

---

### CES (Complex Exponential Smoothing)

CES is ETS's mathematically sophisticated cousin. The "complex" refers to complex numbers (real + imaginary), not difficulty.

The idea: instead of separate level and growth parameters, CES uses a single *complex-valued* smoothing parameter. The real part controls the level. The imaginary part controls latent oscillatory dynamics — hidden cycles that aren't captured by standard seasonality.

**Why does this matter?** Some patterns aren't cleanly seasonal. They oscillate, but not at fixed periods. Business cycles, for instance, have roughly 5-7 year wavelengths but don't repeat exactly. CES can capture these quasi-periodic dynamics because complex exponentials naturally model oscillations.

```python
from vectrix.engine.ces import AutoCES
import numpy as np

data = np.array([100, 108, 115, 110, 120, 128, 125, 135, 142, 138,
                 145, 150, 105, 113, 120, 115, 125, 133, 130, 140])

model = AutoCES(period=10)
model.fit(data)
predictions, _, _ = model.predict(steps=6)
print(f"CES forecast: {predictions.round(1)}")
```

**When CES shines**: Data with hidden cyclic components. Macroeconomic indicators, commodity prices with irregular cycles, anything where seasonality isn't the whole story.

**When CES struggles**: Pure trend data with no oscillation. Intermittent demand. Very short series where complex parameters can't be reliably estimated.

---

## The Theta Family — Controlling Curvature

### The Original Theta Method

Theta is beautifully simple. Published in 2000, it won the M3 Competition and confused everyone — how could something so basic beat sophisticated ARIMA and neural network models?

The core idea: decompose the series into **theta lines**. A theta line is the original series with its curvature amplified or dampened.

- **θ = 0** — removes all curvature, leaving only the linear trend
- **θ = 1** — the original series, unchanged
- **θ = 2** — doubles the curvature, emphasizing short-term fluctuations
- **θ > 2** — even more curvature, more conservative forecasts

![Theta lines at different θ values — from pure trend to conservative smoothing](/vectrix/blog/assets/theta-curvature.svg)

The classic Theta method forecasts the θ=0 line with linear regression (trend) and the θ=2 line with Simple Exponential Smoothing (SES), then averages the two predictions 50/50.

That's it. Linear trend + smoothed recent behavior, combined equally. And it beat almost everything in M3.

**Why it works**: Most time series have both a trend component and a mean-reverting component. By giving each its own forecasting method and combining, Theta captures both without overfitting either.

```python
from vectrix.engine.theta import OptimizedTheta
import numpy as np

data = np.array([100, 105, 108, 115, 120, 118, 125, 130, 128, 135,
                 140, 138, 145, 150, 148, 155, 160, 158, 165, 170])

model = OptimizedTheta(period=1)
model.fit(data)
predictions, _, _ = model.predict(steps=6)
print(f"Theta forecast: {predictions.round(1)}")
```

---

### DOT (Dynamic Optimized Theta) — Vectrix's Workhorse

DOT takes the Theta concept and supercharges it. Instead of fixing θ=0 and θ=2 with a 50/50 split, DOT simultaneously optimizes three parameters:

- **θ** — the decomposition parameter (not fixed at 0 or 2)
- **α** — the SES smoothing parameter
- **drift** — an explicit trend term

For shorter series (below 24 data points), DOT switches to a completely different strategy: it evaluates 8 model configurations (linear/exponential trend × additive/multiplicative × with/without seasonality) and picks the best one via holdout validation.

This hybrid approach is why DOT is Vectrix's strongest single model. On the M4 benchmark, DOT achieves an OWA of 0.848 — close to the M4 #2 method (FFORMA at 0.838), using pure statistics with no machine learning.

```python
from vectrix.engine.dot import DynamicOptimizedTheta
import numpy as np

data = np.array([120, 135, 148, 130, 155, 170, 162, 180, 195, 185, 200, 215,
                 125, 140, 155, 138, 160, 175, 168, 185, 200, 190, 210, 225])

model = DynamicOptimizedTheta(period=12)
model.fit(data)
predictions, _, _ = model.predict(steps=6)
print(f"DOT forecast: {predictions.round(1)}")
```

**When DOT shines**: General-purpose forecasting across frequencies. Monthly, quarterly, weekly data — DOT adapts to each.

**When DOT struggles**: Extreme volatility changes. Pure intermittent demand (mostly zeros).

---

### FourTheta — The Ensemble Within an Ensemble

FourTheta fits four theta lines (θ=0, 1, 2, 3) independently, then combines them using inverse-error weights from a holdout set. If the θ=0 line (pure trend) does well on recent data, it gets more weight. If θ=2 (smoothing) does better, that gets more weight.

This makes FourTheta adaptive — it doesn't commit to a fixed balance between trend and smoothing. It lets the data decide.

```python
from vectrix.engine.fourTheta import AdaptiveThetaEnsemble
import numpy as np

data = np.array([100, 105, 115, 110, 125, 130, 120, 135, 145, 140,
                 155, 160, 150, 165, 175, 170, 185, 190, 180, 195])

model = AdaptiveThetaEnsemble(period=1)
model.fit(data)
predictions, _, _ = model.predict(steps=6)
print(f"FourTheta forecast: {predictions.round(1)}")
```

**When FourTheta shines**: Unstable trends where you're unsure whether the current trajectory will continue or revert. FourTheta hedges that uncertainty by combining all four perspectives. It also excels when you have limited data and can't reliably choose between Theta, DOT, and ARIMA — FourTheta's internal ensemble is more robust to small-sample noise.

**When FourTheta struggles**: Very short series (fewer than 15 points) where even four theta lines can't be reliably estimated. Highly seasonal data where the Theta decomposition itself is less meaningful.

**FourTheta vs. Theta vs. DOT — when to use which?**

- **Theta**: Fast, simple, good default. Best when you want speed and simplicity.
- **DOT**: Strongest single model — optimizes all parameters jointly. Best for general-purpose production use.
- **FourTheta**: Most *stable* forecast. By averaging four perspectives, extreme predictions get dampened. Best when you value predictability over raw accuracy.

In Vectrix's ensemble, all three participate — their errors are correlated but not identical, so each adds incremental diversity.

---

## ARIMA — Learning from Autocorrelation

ARIMA stands for AutoRegressive Integrated Moving Average. The name sounds intimidating, but the idea is elegant: **the best predictor of the future is a weighted combination of past values and past errors**.

Three components:

- **AR (AutoRegressive)** — today's value depends on the last *p* values. If p=2, the model learns something like "tomorrow = 0.7 × today + 0.2 × yesterday + noise"
- **I (Integrated)** — differencing to make the series stationary. Instead of modeling raw sales, model the *change* in sales. d=1 means one round of differencing.
- **MA (Moving Average)** — today's value depends on the last *q* forecast errors. This captures short-lived shocks that persist briefly then fade.

Together, ARIMA(p,d,q) is a powerful framework for modeling linear dependencies in time series.

**Seasonal ARIMA** (SARIMA) adds seasonal AR and MA terms, handling patterns like "January is always like last January."

```python
from vectrix.engine.arima import AutoARIMA
import numpy as np

data = np.random.randn(100).cumsum() + 200

model = AutoARIMA()
model.fit(data)
predictions, lower, upper = model.predict(steps=12)
print(f"ARIMA forecast: {predictions[:6].round(1)}")
```

**AutoARIMA** automates the painful part — choosing p, d, and q. It uses stepwise AICc search (similar to R's `auto.arima()`) to find the best order without you needing to read ACF/PACF plots.

**When ARIMA shines**: Stationary or near-stationary data with linear autocorrelation structure. Economic indicators, temperature anomalies, anything where "how things changed recently" is a good predictor of how they'll change next.

**When ARIMA struggles**: Strong seasonality without enough seasonal data (SARIMA needs multiple complete seasons). Nonlinear dynamics. Regime changes. Data where the *level* matters more than the *changes*.

**Key insight**: ARIMA thinks in *differences*. It's fundamentally about modeling the *changes* between observations, not the observations themselves. This makes it naturally good at detecting momentum and mean-reversion, but it can miss level shifts.

---

## The Multi-Season Specialists

### TBATS — Fourier-Powered Seasonality

Some data has multiple seasonal patterns layered on top of each other. Daily electricity demand, for example, has:

- A daily pattern (peak during business hours)
- A weekly pattern (weekdays vs weekends)
- An annual pattern (summer cooling, winter heating)

Standard ETS handles one seasonal period. TBATS handles all of them simultaneously using a clever trick: instead of tracking individual seasonal indices (which would require storing 365 values for yearly seasonality), it represents each seasonal pattern as a sum of sine and cosine waves (Fourier series).

TBATS stands for: **T**rigonometric seasonality, **B**ox-Cox transformation, **A**RMA errors, **T**rend, **S**easonal components.

```python
from vectrix.engine.tbats import AutoTBATS
import numpy as np

t = np.arange(200)
daily = 10 * np.sin(2 * np.pi * t / 7)
annual = 20 * np.sin(2 * np.pi * t / 365)
data = 100 + 0.3 * t + daily + annual + np.random.randn(200) * 5

model = AutoTBATS(period=7)
model.fit(data)
predictions, _, _ = model.predict(steps=14)
print(f"TBATS 14-day forecast: {predictions.round(1)}")
```

**When TBATS shines**: High-frequency data with multiple overlapping seasonal patterns. Hourly energy, daily retail with day-of-week and month effects, call center volumes.

**When TBATS struggles**: Short series (Fourier harmonics need enough data to estimate). Negative values (Box-Cox transform requires positivity). Simple, single-season data (overkill — use ETS instead).

---

### MSTL — Decompose First, Forecast the Pieces

MSTL (Multiple Seasonal-Trend decomposition using LOESS) takes a different approach to multi-seasonality: instead of building it into a single model, it **peels off seasonal layers one at a time**.

The process:
1. Extract the first seasonal pattern (e.g., weekly) using STL decomposition
2. Remove it from the data
3. Extract the next seasonal pattern (e.g., yearly)
4. Remove it
5. What remains is the trend + residuals
6. Forecast the trend with ARIMA
7. Add the seasonal patterns back

```python
from vectrix.engine.mstl import AutoMSTL
import numpy as np

data = np.random.randn(200).cumsum() + 500
data += 20 * np.sin(2 * np.pi * np.arange(200) / 7)

model = AutoMSTL()
model.fit(data)
predictions, _, _ = model.predict(steps=14)
print(f"MSTL forecast: {predictions[:7].round(1)}")
```

**When MSTL shines**: Same situations as TBATS, but MSTL tends to be more robust when the seasonal patterns are well-separated and stable.

**When MSTL struggles**: When seasonal patterns interact (multiplicative seasonality) rather than stack (additive). Very short series.

**MSTL vs AutoMSTL — what's the difference?**

Vectrix registers two MSTL variants:

- **MSTL** (`MSTLDecomposition`) — you specify the seasonal periods manually (e.g., `periods=[7, 365]`). When Vectrix calls it, it uses the detected primary period from the data's frequency.
- **AutoMSTL** (`AutoMSTL`) — automatically detects *all* relevant seasonal periods using autocorrelation analysis, then decomposes. No manual period input needed.

Use MSTL when you *know* the seasonal periods (most business data has well-known cycles). Use AutoMSTL when you're not sure, or when the data might have unexpected periodicities.

In Vectrix's pipeline, both are evaluated. AutoMSTL sometimes discovers seasonal patterns that the primary period misses, making it a valuable complement.

---

## The Domain Specialists

### GARCH — Forecasting Volatility, Not Values

Every model we've discussed so far forecasts the *value* of the time series. GARCH does something different — it forecasts the *variance*.

In financial markets, volatility clusters. Calm periods are followed by more calm. Turbulent periods are followed by more turbulence. GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models this clustering.

The core equation:

```
σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
```

Translation: tomorrow's variance equals a baseline (ω) plus how large yesterday's shock was (α × ε²) plus how high yesterday's variance was (β × σ²).

Vectrix includes three GARCH variants:

| Model | What's Special |
|-------|---------------|
| GARCH | Standard symmetric variance modeling |
| EGARCH | Log-variance form, allows negative shocks to have bigger impact |
| GJR-GARCH | Threshold model, explicit "leverage effect" for bad news vs good news |

```python
from vectrix.engine.garch import GARCHModel
import numpy as np

returns = np.random.randn(200) * 0.01
returns[50:60] *= 3
returns[120:130] *= 4

model = GARCHModel()
model.fit(returns)
predictions, lower, upper = model.predict(steps=10)
print(f"GARCH mean forecast: {predictions[:5].round(4)}")
print(f"Confidence band width: {(upper - lower)[:5].round(4)}")
```

**When GARCH shines**: Financial returns, any data where the *variability* changes over time. The mean forecast from GARCH is often boring (close to zero for returns), but the **confidence intervals** are where the value is — they widen during volatile periods and narrow during calm ones.

**When GARCH struggles**: Predicting the level of non-financial series. Data with strong trends (GARCH assumes stationarity of returns). Anything where you care about "what will the value be" rather than "how uncertain is the prediction."

---

### Croston — The Intermittent Demand Expert

Most forecasting models assume data arrives continuously. But what about spare parts? Or slow-moving luxury goods? Or rare event counts?

```
0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 7, 0, 0, 0, 2, 0
```

This is intermittent demand — most periods have zero demand, with occasional bursts. Feed this to ETS or ARIMA, and they'll predict something like 0.85 every period. Useless.

Croston's method splits the problem into two:

1. **Demand size** — when demand occurs, how large is it? (Forecasted with SES)
2. **Demand interval** — how many periods between demand events? (Also forecasted with SES)

The forecast is then: size ÷ interval = expected demand per period.

Vectrix includes three Croston variants:

| Variant | Improvement |
|---------|------------|
| Classic | Original Croston method |
| SBA | Syntetos-Boylan Approximation — corrects a systematic upward bias |
| TSB | Teunter-Syntetos-Babai — directly estimates demand probability, handles obsolescence |

```python
from vectrix.engine.croston import AutoCroston
import numpy as np

demand = np.array([0, 0, 0, 5, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 7, 0, 0, 0, 2, 0,
                   0, 0, 0, 4, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 3, 0, 0, 0, 5, 0])

model = AutoCroston()
model.fit(demand)
predictions, _, _ = model.predict(steps=10)
print(f"Expected demand per period: {predictions[:5].round(2)}")
```

**When Croston shines**: Inventory planning for slow-moving items. Spare parts forecasting. Any series where more than 30% of observations are zero.

**When Croston struggles**: Continuous demand (complete overkill). Data where zeros mean "missing" rather than "no demand."

---

## The Unconventional Models

### DTSF (Dynamic Time Scan Forecaster) — Pattern Matching

While every other model builds a mathematical formula, DTSF does something radically different: it finds **historical windows that look like the recent past** and uses what happened after those windows as the forecast.

Think of it like this: "The last 30 days look a lot like days 100-130 and days 250-280. After day 130, demand went up 15%. After day 280, demand went up 12%. So my forecast is that demand will go up roughly 13%."

```python
from vectrix.engine.dtsf import DynamicTimeScanForecaster
import numpy as np

t = np.arange(300)
data = 100 + 20 * np.sin(2 * np.pi * t / 30) + np.random.randn(300) * 5

model = DynamicTimeScanForecaster()
model.fit(data)
predictions, _, _ = model.predict(steps=14)
print(f"DTSF forecast: {predictions[:7].round(1)}")
```

**How it works in practice**: DTSF slides a window across the historical data, computing a distance metric between the recent window and each historical window. The closest matches are selected, and the values that followed those matches are combined (typically averaged) to create the forecast. The window size and number of matches are tuned automatically.

**When DTSF shines**: Repeating complex patterns that parametric models can't capture. Hourly data with irregular but recurring events (holidays, promotions). Data where "this happened before and it played out like this" is a valid forecasting strategy. Long series with rich history give DTSF more patterns to match.

**When DTSF struggles**: Short series (not enough history for pattern matching). Truly novel situations with no historical precedent. Data with strong trends (pattern matching works best on detrended data). Very noisy data where similar-looking windows lead to very different outcomes.

**Why it matters for ensembles**: DTSF makes fundamentally different errors than parametric models. Where ETS might miss a sudden shift, DTSF might catch it because it's seen a similar shift before. Where ARIMA extrapolates a linear trend, DTSF might recognize that "the last time the series looked like this, it reversed." This orthogonality makes DTSF valuable in ensembles even when its standalone accuracy is modest.

---

### ESN (Echo State Network) — Reservoir Computing

ESN is Vectrix's lightweight neural approach. Instead of training a full neural network (which requires lots of data and compute), ESN uses a fixed random "reservoir" — a network of interconnected nodes with random weights that are *never trained*.

**How reservoir computing works**: Imagine dropping a pebble into a pond. The ripples create complex interference patterns across the surface. If you could read those patterns, you'd learn something about the pebble (size, speed, angle). ESN works the same way:

1. Each data point enters the reservoir (the "pebble")
2. It activates hundreds of randomly connected neurons (the "ripples")
3. Each neuron combines the input with its own state and its neighbors' states through nonlinear activation functions
4. The result is hundreds of different nonlinear transformations of the input history
5. A simple linear regression maps these reservoir states to predictions

The key insight: the reservoir creates a *high-dimensional nonlinear expansion* of the input. Linear regression in this expanded space can capture highly nonlinear relationships — without the gradient descent, backpropagation, and training instability of conventional neural networks.

```python
from vectrix.engine.esn import EchoStateForecaster
import numpy as np

data = np.random.randn(200).cumsum() + 100

model = EchoStateForecaster()
model.fit(data)
predictions, _, _ = model.predict(steps=10)
print(f"ESN forecast: {predictions[:5].round(1)}")
```

**When ESN shines**: Capturing nonlinear dynamics that statistical models miss — regime switches, asymmetric cycles, threshold effects. Data with complex dependencies where the relationship between past and future values isn't a simple linear function. ESN also trains extremely fast (no iterative optimization), making it practical for production pipelines.

**When ESN struggles**: Linear, well-behaved data (overkill — ETS or ARIMA will do better with less risk of overfitting). Very short series (the reservoir needs enough data to "warm up" its internal states). Pure random walks with no structure. ESN can also be sensitive to hyperparameters (reservoir size, spectral radius), though Vectrix tunes these automatically.

**Why ESN matters for the ensemble**: ESN's errors are systematically different from statistical model errors. Statistical models struggle with nonlinearities but handle trends well. ESN handles nonlinearities but can be noisy on simple patterns. Combining them gives the best of both worlds.

---

## The Baselines — Your Sanity Check

Baselines aren't trying to be good. They're trying to be **the minimum bar that any real model must clear**.

| Model | What It Does | When It's Actually Useful |
|-------|-------------|--------------------------|
| **Naive** | Repeats the last observed value forever | Benchmark for all others. If your model can't beat Naive, something is wrong. |
| **Seasonal Naive** | Repeats the last complete season | The M4 Competition baseline. Surprisingly hard to beat on stable seasonal data. |
| **Mean** | Forecasts the historical average | Good for stationary data with no trend. |
| **Random Walk with Drift** | Last value + average change per period | Simple trending benchmark. |
| **Window Average** | Average of the last N observations | Recent-history benchmark for stable data. |

The M4 Competition uses **Naive2** (Seasonal Naive with deseasonalization) as its reference. A model's OWA (Overall Weighted Average) score is normalized against Naive2 — below 1.0 means you beat the baseline, above 1.0 means the baseline was better.

**Never skip baselines.** They serve two purposes:

1. **Sanity check** — if your sophisticated model loses to Naive, debug it before deploying it
2. **Communication** — "Our model is 15% better than repeating last year's numbers" is more compelling than "Our MAPE is 4.2%"

```python
from vectrix import compare, loadSample

df = loadSample("airline")
ranking = compare(df, steps=12)
print(ranking[["model", "mape"]].head(10))
```

---

## How Vectrix Puts It All Together

When you call `forecast()`, here's what actually happens:

1. **DNA Profiling** — 65+ statistical features are extracted. Is the data trending? Seasonal? Volatile? Intermittent? How forecastable is it?
2. **Model Ranking** — based on the DNA profile, models are ranked by expected fitness. Seasonal data? ETS and Theta go first. Intermittent? Croston leads.
3. **Validation** — all candidate models are trained on a subset and evaluated on a holdout set. Real performance, not theoretical.
4. **Selection or Ensemble** — if one model clearly wins, it's selected. If several perform similarly, they're combined into a weighted ensemble using inverse-error weights.

The key insight: different models make *different kinds of errors*. ETS might over-smooth a sudden jump. ARIMA might over-react to a noisy spike. Theta might miss a seasonal effect. But when you average their forecasts, the errors partially cancel out.

This is why **ensemble diversity matters more than individual model accuracy**. Adding a mediocre model that makes *different* errors can improve the ensemble, while adding a great model that makes the *same* errors won't help.

```python
from vectrix import forecast, loadSample

df = loadSample("airline")
result = forecast(df, steps=12)

print(f"Best model: {result.model}")
print(f"MAPE: {result.mape:.1f}%")
print(f"All models ranked: {result.models[:5]}")
```

---

## Quick Reference — When to Use What

| Your Data Looks Like... | Best Model Families | Why |
|------------------------|--------------------|----|
| Smooth trend + clear seasonality | AutoETS, Holt-Winters, DOT | Built for exactly this pattern |
| Trending without seasonality | Holt's Linear, ARIMA, Theta | Trend specialists, no seasonal overhead |
| Growing/shrinking with momentum | ARIMA, Theta, RWD | Capture autocorrelation and trend |
| Multiple seasonal cycles (daily+weekly+yearly) | TBATS, MSTL, AutoMSTL | Multi-season specialists |
| Financial returns / volatile data | GARCH, EGARCH, GJR-GARCH | Model variance, not just mean |
| Mostly zeros (spare parts, rare events) | Croston (SBA/TSB) | Separate size vs. interval |
| Complex nonlinear patterns | CES, ESN, DTSF | Different nonlinear approaches |
| Repeating historical patterns | DTSF | Non-parametric pattern matching |
| Unstable trend, need robust forecast | FourTheta | Four perspectives, dampened extremes |
| "I don't know, just give me a forecast" | `forecast(data)` | Vectrix tests everything automatically |

The beauty of automated model selection is that you don't *have to* choose. But understanding what's happening under the hood helps you interpret results, diagnose failures, and communicate findings to stakeholders.

---

## What's Next?

Now that you know what each model does, the natural question is: **how does Vectrix decide which one to use?** That's the topic of our next post — the art and science of model selection, DNA profiling, and why the best model for monthly data is almost never the best model for daily data.

**Previous posts in this series:**

- [What Is Forecasting?](/vectrix/blog/what-is-forecasting) — The fundamentals
- [How Do We Know If a Forecast Is Any Good?](/vectrix/blog/how-we-know-forecasts-work) — Metrics and benchmarks
- [Python Forecasting Libraries Compared](/vectrix/blog/python-forecasting-libraries) — The tool landscape
- [Your First Forecast in Python](/vectrix/blog/your-first-forecast-in-python) — Hands-on practice

**Want to try these models yourself?** Check out the [Models Tutorial](/vectrix/docs/tutorials/models/) for runnable code examples with every model family.

---
