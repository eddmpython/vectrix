"""
Individual chart functions for Vectrix visualization.

Each function takes a Vectrix result object and returns a Plotly figure.
"""

import pandas as pd

from .theme import COLORS, PALETTE, HEIGHT, applyTheme

try:
    import plotly.graph_objects as go
except ImportError:
    raise ImportError(
        "plotly is required for vectrix.viz. "
        "Install it with: pip install vectrix[viz]"
    )


def _detectColumns(df):
    """Detect date and value columns from a DataFrame."""
    dateCols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    dateCol = dateCols[0] if dateCols else df.columns[0]
    valueCols = [c for c in df.columns if c != dateCol and pd.api.types.is_numeric_dtype(df[c])]
    valueCol = valueCols[0] if valueCols else df.columns[1]
    return dateCol, valueCol


def forecastChart(forecastResult, historical=None, title=None, theme="dark"):
    """
    Interactive forecast chart with confidence intervals.

    Parameters
    ----------
    forecastResult : EasyForecastResult
        Result from forecast().
    historical : pd.DataFrame, optional
        Historical data with date and value columns.
    title : str, optional
        Chart title. Auto-generated if None.
    theme : str
        'dark' (default) or 'light'.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fcDf = forecastResult.toDataframe()
    fcDates = pd.to_datetime(fcDf["date"])

    if historical is not None:
        dateCol, valueCol = _detectColumns(historical)
        histDates = pd.to_datetime(historical[dateCol])

        fig.add_trace(go.Scatter(
            x=histDates, y=historical[valueCol],
            name="Historical",
            line=dict(color=COLORS["muted"], width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.upper,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.lower,
        fill="tonexty", name="95% CI",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(width=0), hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.predictions,
        name=f"Forecast ({forecastResult.model})",
        line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>",
    ))

    autoTitle = title or f"Forecast — {forecastResult.model} (MAPE {forecastResult.mape:.1f}%)"
    return applyTheme(fig, title=autoTitle, height=HEIGHT["chart"], theme=theme)


def dnaRadar(analysisResult, title=None, theme="dark"):
    """
    DNA profile radar chart showing key time series features.

    Displays 6 normalized features on a polar chart: trend strength,
    seasonality, memory (Hurst), volatility clustering, nonlinearity,
    and forecastability.

    Parameters
    ----------
    analysisResult : EasyAnalysisResult
        Result from analyze().
    title : str, optional
    theme : str
        'dark' or 'light'.

    Returns
    -------
    go.Figure
    """
    dna = analysisResult.dna
    feat = dna.features

    keys = [
        "trendStrength", "seasonalStrength", "hurstExponent",
        "volatilityClustering", "nonlinearAutocorr", "forecastability",
    ]
    labels = [
        "Trend", "Seasonality", "Memory",
        "Vol. Clustering", "Nonlinear", "Forecastability",
    ]

    values = []
    for k in keys:
        v = feat.get(k, 0)
        values.append(max(0.0, min(float(v) if v is not None else 0, 1.0)))
    values.append(values[0])
    labelsClosed = labels + [labels[0]]

    bgColor = COLORS["card"] if theme == "dark" else "#f1f5f9"
    gridColor = "rgba(255,255,255,0.1)" if theme == "dark" else "rgba(0,0,0,0.1)"

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labelsClosed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color=COLORS["primary"], width=2),
        name="DNA Profile",
    ))

    autoTitle = title or f"DNA — {dna.category} ({dna.difficulty}, {dna.difficultyScore:.0f}/100)"
    fig.update_layout(
        polar=dict(
            bgcolor=bgColor,
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=gridColor),
            angularaxis=dict(gridcolor=gridColor),
        ),
    )
    return applyTheme(fig, title=autoTitle, height=HEIGHT["chart"], theme=theme)


def modelHeatmap(comparisonDf, top=10, title=None, theme="dark"):
    """
    Model comparison heatmap with normalized error metrics.

    Lower (greener) is better. Values are min-max normalized per column.

    Parameters
    ----------
    comparisonDf : pd.DataFrame
        Result from compare(). Assumed sorted by MAPE (best first).
    top : int
        Number of top models to show.
    title : str, optional
    theme : str
        'dark' or 'light'.

    Returns
    -------
    go.Figure
    """
    topDf = comparisonDf.head(top).copy()
    metricCols = ["mape", "rmse", "mae", "smape"]
    available = [c for c in metricCols if c in topDf.columns]

    normalized = topDf[available].copy()
    for col in available:
        mn, mx = normalized[col].min(), normalized[col].max()
        if mx > mn:
            normalized[col] = (normalized[col] - mn) / (mx - mn)
        else:
            normalized[col] = 0.5

    fig = go.Figure(data=go.Heatmap(
        z=normalized[available].values,
        x=[c.upper() for c in available],
        y=topDf["model"].values,
        colorscale=[
            [0, COLORS["positive"]],
            [0.5, COLORS["warning"]],
            [1, COLORS["negative"]],
        ],
        text=topDf[available].round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=12),
        showscale=False,
        hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
    ))

    fig.update_layout(yaxis=dict(autorange="reversed"))
    autoTitle = title or f"Top {top} Models — Error Heatmap"
    return applyTheme(fig, title=autoTitle, height=max(250, top * 35), theme=theme)


def scenarioChart(scenarios, dates=None, title=None, theme="dark"):
    """
    What-if scenario comparison chart.

    Parameters
    ----------
    scenarios : list[ScenarioResult]
        Result from WhatIfAnalyzer.analyze().
    dates : list or pd.DatetimeIndex, optional
        Forecast dates for the X-axis. If None, uses numeric steps.
    title : str, optional
    theme : str
        'dark' or 'light'.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, s in enumerate(scenarios):
        if dates is not None:
            xVals = pd.to_datetime(dates[:len(s.predictions)])
            hoverFmt = f"{s.name}<br>%{{x|%Y-%m-%d}}<br>%{{y:,.1f}}<extra></extra>"
        else:
            xVals = list(range(1, len(s.predictions) + 1))
            hoverFmt = f"{s.name}<br>Step %{{x}}<br>%{{y:,.1f}}<extra></extra>"

        fig.add_trace(go.Scatter(
            x=xVals,
            y=s.predictions,
            name=f"{s.name} ({s.impact:+.1f}%)",
            line=dict(
                color=PALETTE[i % len(PALETTE)],
                width=2,
                dash="solid" if i == 0 else "dash",
            ),
            hovertemplate=hoverFmt,
        ))

    autoTitle = title or "What-If Scenario Analysis"
    return applyTheme(fig, title=autoTitle, height=HEIGHT["chart"], theme=theme)


def backtestChart(backtestResult, metric="mape", title=None, theme="dark"):
    """
    Backtest performance bar chart by fold.

    Parameters
    ----------
    backtestResult : BacktestResult
        Result from Backtester.run().
    metric : str
        Which metric to show: 'mape' (default) or 'rmse'.
    title : str, optional
    theme : str
        'dark' or 'light'.

    Returns
    -------
    go.Figure
    """
    if metric == "rmse":
        foldValues = [getattr(f, "rmse", 0) for f in backtestResult.folds]
        avgValue = getattr(backtestResult, "avgRMSE", 0)
        metricLabel = "RMSE"
        fmt = ",.1f"
    else:
        foldValues = [f.mape for f in backtestResult.folds]
        avgValue = backtestResult.avgMAPE
        metricLabel = "MAPE"
        fmt = ".1f"

    foldNums = [f"Fold {f.fold}" for f in backtestResult.folds]

    barColors = [
        COLORS["positive"] if m == min(foldValues) else
        COLORS["negative"] if m == max(foldValues) else
        COLORS["primary"]
        for m in foldValues
    ]

    suffix = "%" if metric == "mape" else ""

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=foldNums, y=foldValues,
        marker_color=barColors,
        text=[f"{m:{fmt}}{suffix}" for m in foldValues],
        textposition="auto",
        hovertemplate=f"Fold %{{x}}<br>{metricLabel}: %{{y:{fmt}}}{suffix}<extra></extra>",
    ))
    fig.add_hline(
        y=avgValue, line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Avg {avgValue:{fmt}}{suffix}",
    )

    autoTitle = title or f"Backtest — {avgValue:{fmt}}{suffix} Average {metricLabel}"
    return applyTheme(fig, title=autoTitle, height=HEIGHT["chart"], theme=theme)


def metricsCard(metricsDict, title=None, thresholds=None, theme="dark"):
    """
    Business metrics scorecard with color-coded indicators.

    Parameters
    ----------
    metricsDict : dict
        Result from BusinessMetrics.calculate().
    title : str, optional
    thresholds : dict, optional
        Custom thresholds. Keys: 'accuracy' (default 95), 'bias' (3),
        'wape' (5), 'mase' (1.0). Values above threshold turn red.
    theme : str
        'dark' or 'light'.

    Returns
    -------
    go.Figure
    """
    colors = COLORS if theme == "dark" else {**COLORS, "text": "#0f172a"}

    t = thresholds or {}
    accThresh = t.get("accuracy", 95)
    biasThresh = t.get("bias", 3)
    wapeThresh = t.get("wape", 5)
    maseThresh = t.get("mase", 1.0)

    items = [
        ("Accuracy", metricsDict.get("forecastAccuracy", 0), ".1f", "%", accThresh, True),
        ("Bias", metricsDict.get("biasPercent", 0), "+.2f", "%", biasThresh, False),
        ("WAPE", metricsDict.get("wape", 0), ".2f", "%", wapeThresh, False),
        ("MASE", metricsDict.get("mase", 0), ".3f", "", maseThresh, False),
    ]

    fig = go.Figure()
    for i, (name, val, fmt, suffix, thresh, higherBetter) in enumerate(items):
        if name == "Bias":
            color = colors["positive"] if abs(val) < thresh else colors["warning"] if abs(val) < thresh * 1.67 else colors["negative"]
        elif higherBetter:
            color = colors["positive"] if val >= thresh else colors["negative"]
        else:
            color = colors["positive"] if val < thresh else colors["negative"]

        fig.add_trace(go.Indicator(
            mode="number",
            value=val,
            number=dict(
                font=dict(size=40, color=color),
                valueformat=fmt,
                suffix=suffix,
            ),
            title=dict(text=name, font=dict(size=16, color=colors["text"])),
            domain=dict(row=0, column=i),
        ))

    fig.update_layout(grid=dict(rows=1, columns=len(items), pattern="independent"))
    autoTitle = title or "Business Metrics"
    return applyTheme(fig, title=autoTitle, height=HEIGHT["card"], theme=theme)
