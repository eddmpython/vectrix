"""
Composite report generators that combine multiple charts.

Each function returns a single Plotly figure with subplots.
Design language: Cyan→Purple gradient, dark navy, Inter typography.
"""

import pandas as pd

from .theme import HEIGHT, PALETTE, _colors, applyTheme

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
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


def forecastReport(forecastResult, historical=None, title=None, theme="dark"):
    """
    Comprehensive forecast report with predictions, confidence bands, and metrics.

    Creates a 2-row layout:
    - Top (75%): forecast line chart with CI and optional historical data
    - Bottom (25%): key metrics summary bar (MAPE, RMSE, MAE, sMAPE)

    Parameters
    ----------
    forecastResult : EasyForecastResult
        Result from forecast().
    historical : pd.DataFrame, optional
        Historical data with date and value columns.
    title : str, optional
    theme : str
        'dark' or 'light'.

    Returns
    -------
    go.Figure
    """
    c = _colors(theme)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.10,
        subplot_titles=[
            f"<b>Forecast</b>  <span style='font-size:11px;color:{c['textMuted']}'>"
            f"{forecastResult.model}  ·  {len(forecastResult.predictions)} steps</span>",
            "<b>Error Metrics</b>",
        ],
    )

    fcDf = forecastResult.toDataframe()
    fcDates = pd.to_datetime(fcDf["date"])

    if historical is not None:
        dateCol, valueCol = _detectColumns(historical)
        histDates = pd.to_datetime(historical[dateCol])

        fig.add_trace(go.Scatter(
            x=histDates, y=historical[valueCol],
            name="Historical",
            line=dict(color=c["muted"], width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>",
        ), row=1, col=1)

        lastHistDate = histDates.iloc[-1]
        lastHistVal = historical[valueCol].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[lastHistDate, fcDates.iloc[0]],
            y=[lastHistVal, forecastResult.predictions[0]],
            line=dict(color=c["muted"], width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.upper,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.lower,
        fill="tonexty", name="95% CI",
        fillcolor="rgba(6,182,212,0.10)",
        line=dict(width=0), hoverinfo="skip",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=fcDates, y=forecastResult.predictions,
        name=f"Forecast ({forecastResult.model})",
        line=dict(color=c["primary"], width=2.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>",
    ), row=1, col=1)

    metricNames = ["MAPE", "RMSE", "MAE", "sMAPE"]
    metricValues = [
        getattr(forecastResult, "mape", 0),
        getattr(forecastResult, "rmse", 0),
        getattr(forecastResult, "mae", 0),
        getattr(forecastResult, "smape", 0),
    ]

    metricColors = []
    for i, v in enumerate(metricValues):
        if i in (0, 3):
            metricColors.append(c["positive"] if v < 10 else c["warning"] if v < 20 else c["negative"])
        else:
            metricColors.append(c["primary"])

    fig.add_trace(go.Bar(
        x=metricNames, y=metricValues,
        marker=dict(color=metricColors, line=dict(width=0)),
        text=[f"{v:.2f}" for v in metricValues],
        textposition="auto",
        textfont=dict(size=12, color=c["text"]),
        showlegend=False,
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ), row=2, col=1)

    autoTitle = title or f"Forecast Report — {forecastResult.model}"
    return applyTheme(fig, title=autoTitle, height=HEIGHT["report"], theme=theme)


def analysisReport(analysisResult, title=None, theme="dark"):
    """
    Comprehensive analysis report with DNA radar, feature bars, and summary.

    Creates a 2x2 layout:
    - Top-left: DNA radar chart (6 features on polar)
    - Top-right: feature importance bar chart (8 features)
    - Bottom: summary indicator (difficulty score + metadata)

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
    c = _colors(theme)
    dna = analysisResult.dna
    feat = dna.features

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "polar"}, {"type": "xy"}],
            [{"type": "domain", "colspan": 2}, None],
        ],
        row_heights=[0.65, 0.35],
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
        subplot_titles=[
            "<b>DNA Profile</b>",
            "<b>Key Features</b>",
            "",
        ],
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
        radarValues.append(max(0.0, min(float(v) if v is not None else 0, 1.0)))
    radarValues.append(radarValues[0])
    radarLabelsClosed = radarLabels + [radarLabels[0]]

    fig.add_trace(go.Scatterpolar(
        r=radarValues, theta=radarLabelsClosed,
        fill="toself",
        fillcolor="rgba(6,182,212,0.12)",
        line=dict(color=c["primary"], width=2.5),
        marker=dict(size=5, color=c["primary"]),
        name="DNA",
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
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
        marker=dict(color=barColors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in barValues],
        textposition="auto",
        textfont=dict(size=11, color=c["text"]),
        showlegend=False,
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ), row=1, col=2)

    scoreColor = c["positive"] if dna.difficultyScore < 40 else c["warning"] if dna.difficultyScore < 70 else c["negative"]

    fig.add_trace(go.Indicator(
        mode="number",
        value=dna.difficultyScore,
        number=dict(
            font=dict(size=44, color=scoreColor),
            valueformat=".0f",
            suffix="/100",
        ),
        title=dict(
            text=(
                f"<b>{dna.category}</b> — {dna.difficulty}<br>"
                f"<span style='font-size:12px;color:{c['textMuted']}'>"
                f"{len(analysisResult.changepoints)} changepoints  ·  "
                f"{len(analysisResult.anomalies)} anomalies</span>"
            ),
            font=dict(size=15, color=c["text"]),
        ),
        domain=dict(row=1, column=0),
    ), row=2, col=1)

    fig.update_layout(
        polar=dict(
            bgcolor=c["bg"],
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=c["grid"],
                tickfont=dict(size=10, color=c["dim"]),
                tickvals=[0.25, 0.5, 0.75, 1.0],
            ),
            angularaxis=dict(
                gridcolor=c["grid"],
                tickfont=dict(size=11, color=c["textMuted"]),
            ),
        ),
        grid=dict(rows=2, columns=2, pattern="independent"),
    )

    autoTitle = title or f"Analysis Report — {dna.category}"
    return applyTheme(fig, title=autoTitle, height=HEIGHT["analysis"], theme=theme)
