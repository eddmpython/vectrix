"""
Individual chart functions for Vectrix visualization.

Each function takes a Vectrix result object and returns a Plotly figure.
"""

import numpy as np
import pandas as pd

from .theme import COLORS, PALETTE, applyTheme

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError(
        "plotly is required for vectrix.viz. "
        "Install it with: pip install vectrix[viz]"
    )


def forecastChart(forecastResult, historical=None, title=None):
    """
    Interactive forecast chart with confidence intervals.

    Parameters
    ----------
    forecastResult : EasyForecastResult
        Result from forecast().
    historical : pd.DataFrame, optional
        Historical data with 'date' and value columns.
    title : str, optional
        Chart title. Auto-generated if None.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    fcDf = forecastResult.toDataframe()
    fcDates = pd.to_datetime(fcDf["date"])

    if historical is not None:
        dateCols = [c for c in historical.columns if "date" in c.lower()]
        dateCol = dateCols[0] if dateCols else historical.columns[0]
        valueCols = [c for c in historical.columns if c != dateCol]
        valueCol = valueCols[0] if valueCols else historical.columns[1]
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
    return applyTheme(fig, title=autoTitle, height=450)


def dnaRadar(analysisResult, title=None):
    """
    DNA profile radar chart showing key time series features.

    Parameters
    ----------
    analysisResult : EasyAnalysisResult
        Result from analyze().
    title : str, optional

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
        values.append(min(float(v) if v is not None else 0, 1.0))
    values.append(values[0])
    labelsClosed = labels + [labels[0]]

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
            bgcolor=COLORS["card"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
    )
    return applyTheme(fig, title=autoTitle, height=450)


def modelHeatmap(comparisonDf, top=10, title=None):
    """
    Model comparison heatmap with normalized error metrics.

    Parameters
    ----------
    comparisonDf : pd.DataFrame
        Result from compare().
    top : int
        Number of top models to show.
    title : str, optional

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
            normalized[col] = 0

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
    return applyTheme(fig, title=autoTitle, height=max(250, top * 35))


def scenarioChart(scenarios, title=None):
    """
    What-if scenario comparison chart.

    Parameters
    ----------
    scenarios : list[ScenarioResult]
        Result from WhatIfAnalyzer.analyze().
    title : str, optional

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()
    for i, s in enumerate(scenarios):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(s.predictions) + 1)),
            y=s.predictions,
            name=f"{s.name} ({s.impact:+.1f}%)",
            line=dict(
                color=PALETTE[i % len(PALETTE)],
                width=2,
                dash="solid" if i == 0 else "dash",
            ),
            hovertemplate=f"{s.name}<br>Step %{{x}}<br>%{{y:,.1f}}<extra></extra>",
        ))

    autoTitle = title or "What-If Scenario Analysis"
    return applyTheme(fig, title=autoTitle, height=400)


def backtestChart(backtestResult, title=None):
    """
    Backtest performance bar chart (MAPE + RMSE by fold).

    Parameters
    ----------
    backtestResult : BacktestResult
        Result from Backtester.run().
    title : str, optional

    Returns
    -------
    go.Figure
    """
    foldMapes = [f.mape for f in backtestResult.folds]
    foldNums = [f"Fold {f.fold}" for f in backtestResult.folds]

    barColors = [
        COLORS["positive"] if m == min(foldMapes) else
        COLORS["negative"] if m == max(foldMapes) else
        COLORS["primary"]
        for m in foldMapes
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=foldNums, y=foldMapes,
        marker_color=barColors,
        text=[f"{m:.1f}%" for m in foldMapes],
        textposition="auto",
        hovertemplate="Fold %{x}<br>MAPE: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(
        y=backtestResult.avgMAPE, line_dash="dash",
        line_color=COLORS["warning"],
        annotation_text=f"Avg {backtestResult.avgMAPE:.1f}%",
    )

    autoTitle = title or f"Backtest — {backtestResult.avgMAPE:.1f}% Average MAPE"
    return applyTheme(fig, title=autoTitle, height=400)


def metricsCard(metricsDict, title=None):
    """
    Business metrics scorecard with color-coded indicators.

    Parameters
    ----------
    metricsDict : dict
        Result from BusinessMetrics.calculate().
    title : str, optional

    Returns
    -------
    go.Figure
    """
    items = [
        ("Accuracy", metricsDict.get("forecastAccuracy", 0), ".1f", "%", 95, True),
        ("Bias", metricsDict.get("biasPercent", 0), "+.2f", "%", 3, False),
        ("WAPE", metricsDict.get("wape", 0), ".2f", "%", 5, False),
        ("MASE", metricsDict.get("mase", 0), ".3f", "", 1.0, False),
    ]

    fig = go.Figure()
    for i, (name, val, fmt, suffix, thresh, higherBetter) in enumerate(items):
        if name == "Bias":
            color = COLORS["positive"] if abs(val) < thresh else COLORS["warning"] if abs(val) < 5 else COLORS["negative"]
        elif higherBetter:
            color = COLORS["positive"] if val >= thresh else COLORS["negative"]
        else:
            color = COLORS["positive"] if val < thresh else COLORS["negative"]

        fig.add_trace(go.Indicator(
            mode="number",
            value=val,
            number=dict(
                font=dict(size=40, color=color),
                valueformat=fmt,
                suffix=suffix,
            ),
            title=dict(text=name, font=dict(size=16, color=COLORS["text"])),
            domain=dict(row=0, column=i),
        ))

    fig.update_layout(grid=dict(rows=1, columns=len(items), pattern="independent"))
    autoTitle = title or "Business Metrics"
    return applyTheme(fig, title=autoTitle, height=220)
