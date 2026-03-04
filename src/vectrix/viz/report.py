"""
Composite report generators that combine multiple charts.

Each function returns a single Plotly figure with subplots.
"""

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


def forecastReport(forecastResult, historical=None, title=None):
    """
    Comprehensive forecast report with predictions, confidence bands, and metrics.

    Creates a 2-row layout:
    - Top: forecast line chart with CI and optional historical data
    - Bottom: key metrics summary bar

    Parameters
    ----------
    forecastResult : EasyForecastResult
        Result from forecast().
    historical : pd.DataFrame, optional
        Historical data with 'date' and value columns.
    title : str, optional
        Report title. Auto-generated if None.

    Returns
    -------
    go.Figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.12,
        subplot_titles=["Forecast", "Error Metrics"],
    )

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
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.upper,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.lower,
        fill="tonexty", name="95% CI",
        fillcolor="rgba(99,102,241,0.12)",
        line=dict(width=0), hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.predictions,
        name=f"Forecast ({forecastResult.model})",
        line=dict(color=COLORS["primary"], width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>",
    ), row=1, col=1)

    metricNames = ["MAPE", "RMSE", "MAE"]
    metricValues = [
        getattr(forecastResult, "mape", 0),
        getattr(forecastResult, "rmse", 0),
        getattr(forecastResult, "mae", 0),
    ]
    metricColors = []
    for i, v in enumerate(metricValues):
        if i == 0:
            metricColors.append(COLORS["positive"] if v < 10 else COLORS["warning"] if v < 20 else COLORS["negative"])
        else:
            metricColors.append(COLORS["primary"])

    fig.add_trace(go.Bar(
        x=metricNames, y=metricValues,
        marker_color=metricColors,
        text=[f"{v:.2f}" for v in metricValues],
        textposition="auto",
        showlegend=False,
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    autoTitle = title or f"Forecast Report — {forecastResult.model}"
    return applyTheme(fig, title=autoTitle, height=600)


def analysisReport(analysisResult, title=None):
    """
    Comprehensive analysis report with DNA radar, feature bars, and summary.

    Creates a 2x2 layout:
    - Top-left: DNA radar chart
    - Top-right: feature importance bars
    - Bottom: summary text indicators

    Parameters
    ----------
    analysisResult : EasyAnalysisResult
        Result from analyze().
    title : str, optional
        Report title. Auto-generated if None.

    Returns
    -------
    go.Figure
    """
    dna = analysisResult.dna
    feat = dna.features

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "polar"}, {"type": "xy"}],
            [{"type": "domain", "colspan": 2}, None],
        ],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        subplot_titles=["DNA Profile", "Key Features", ""],
    )

    radarKeys = [
        "trendStrength", "seasonalStrength", "hurstExponent",
        "volatilityClustering", "nonlinearAutocorr", "forecastability",
    ]
    radarLabels = [
        "Trend", "Seasonality", "Memory",
        "Vol. Clustering", "Nonlinear", "Forecastability",
    ]
    radarValues = []
    for k in radarKeys:
        v = feat.get(k, 0)
        radarValues.append(min(float(v) if v is not None else 0, 1.0))
    radarValues.append(radarValues[0])
    radarLabelsClosed = radarLabels + [radarLabels[0]]

    fig.add_trace(go.Scatterpolar(
        r=radarValues, theta=radarLabelsClosed,
        fill="toself",
        fillcolor="rgba(99,102,241,0.2)",
        line=dict(color=COLORS["primary"], width=2),
        name="DNA",
    ), row=1, col=1)

    barKeys = [
        "trendStrength", "seasonalStrength", "hurstExponent",
        "volatilityClustering", "nonlinearAutocorr", "forecastability",
        "entropy", "acfSum",
    ]
    barLabels = [
        "Trend", "Season", "Hurst",
        "Vol.Clust", "Nonlinear", "Fcast",
        "Entropy", "ACF Sum",
    ]
    barValues = []
    validLabels = []
    for k, lbl in zip(barKeys, barLabels):
        v = feat.get(k)
        if v is not None:
            barValues.append(float(v))
            validLabels.append(lbl)

    barColors = [PALETTE[i % len(PALETTE)] for i in range(len(barValues))]

    fig.add_trace(go.Bar(
        x=validLabels, y=barValues,
        marker_color=barColors,
        text=[f"{v:.3f}" for v in barValues],
        textposition="auto",
        showlegend=False,
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ), row=1, col=2)

    summaryItems = [
        ("Category", dna.category),
        ("Difficulty", f"{dna.difficulty} ({dna.difficultyScore:.0f}/100)"),
        ("Changepoints", str(len(analysisResult.changepoints))),
        ("Anomalies", str(len(analysisResult.anomalies))),
    ]
    summaryText = "  |  ".join(f"<b>{k}</b>: {v}" for k, v in summaryItems)

    fig.add_trace(go.Indicator(
        mode="number",
        value=dna.difficultyScore,
        number=dict(
            font=dict(size=48, color=COLORS["primary"]),
            valueformat=".0f",
            suffix="/100",
        ),
        title=dict(
            text=f"{dna.category} — {dna.difficulty}<br>"
                 f"<span style='font-size:13px'>{len(analysisResult.changepoints)} changepoints, "
                 f"{len(analysisResult.anomalies)} anomalies</span>",
            font=dict(size=16, color=COLORS["text"]),
        ),
        domain=dict(row=1, column=0),
    ), row=2, col=1)

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["card"],
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        ),
        grid=dict(rows=2, columns=2, pattern="independent"),
    )

    autoTitle = title or f"Analysis Report — {dna.category}"
    return applyTheme(fig, title=autoTitle, height=650)
