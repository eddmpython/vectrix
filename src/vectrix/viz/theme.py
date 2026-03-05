"""
Vectrix brand theme for Plotly charts.

Design tokens unified with the landing page (SvelteKit).
Cyan→Purple gradient identity, dark navy background, Inter typography.
"""


COLORS = {
    "primary": "#06b6d4",
    "primaryDark": "#0891b2",
    "primaryLight": "#22d3ee",
    "accent": "#8b5cf6",
    "accentLight": "#a78bfa",
    "positive": "#10b981",
    "negative": "#ef4444",
    "warning": "#f59e0b",
    "muted": "#94a3b8",
    "dim": "#64748b",
    "bg": "#0f172a",
    "bgDarker": "#020617",
    "card": "#1e293b",
    "cardHover": "#334155",
    "text": "#f8fafc",
    "textMuted": "#94a3b8",
    "border": "#334155",
    "grid": "rgba(255,255,255,0.05)",
}

LIGHT_COLORS = {
    "primary": "#06b6d4",
    "primaryDark": "#0891b2",
    "primaryLight": "#0e7490",
    "accent": "#8b5cf6",
    "accentLight": "#7c3aed",
    "positive": "#059669",
    "negative": "#dc2626",
    "warning": "#d97706",
    "muted": "#64748b",
    "dim": "#94a3b8",
    "bg": "#ffffff",
    "bgDarker": "#f8fafc",
    "card": "#f1f5f9",
    "cardHover": "#e2e8f0",
    "text": "#0f172a",
    "textMuted": "#64748b",
    "border": "#e2e8f0",
    "grid": "rgba(0,0,0,0.05)",
}

PALETTE = [
    "#06b6d4",
    "#8b5cf6",
    "#10b981",
    "#f59e0b",
    "#ef4444",
    "#38bdf8",
    "#a78bfa",
    "#34d399",
    "#fb923c",
    "#e879f9",
]

GRADIENT_COLORSCALE = [
    [0.0, "#06b6d4"],
    [0.5, "#8b5cf6"],
    [1.0, "#e879f9"],
]

HEATMAP_COLORSCALE = [
    [0.0, "#10b981"],
    [0.35, "#06b6d4"],
    [0.65, "#8b5cf6"],
    [1.0, "#ef4444"],
]

HEIGHT = {
    "chart": 480,
    "card": 240,
    "report": 640,
    "analysis": 680,
    "small": 360,
}

FONT = "Inter, system-ui, -apple-system, sans-serif"
FONT_MONO = "JetBrains Mono, ui-monospace, monospace"


def _colors(theme):
    return COLORS if theme == "dark" else LIGHT_COLORS


def applyTheme(fig, title=None, subtitle=None, height=480, theme="dark"):
    """Apply Vectrix brand theme to a Plotly figure."""
    c = _colors(theme)

    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        paper_bgcolor=c["bgDarker"],
        plot_bgcolor=c["bg"],
        font=dict(family=FONT, color=c["text"], size=13),
        margin=dict(l=56, r=24, t=72 if title else 40, b=48),
        height=height,
        legend=dict(
            orientation="h",
            y=-0.14,
            x=0.5,
            xanchor="center",
            font=dict(size=12, color=c["textMuted"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor=c["card"],
            bordercolor=c["border"],
            font=dict(family=FONT, size=12, color=c["text"]),
        ),
    )

    if title:
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>{title}</b><br>"
                    f"<span style='font-size:12px;color:{c['textMuted']}'>{subtitle}</span>"
                    if subtitle else f"<b>{title}</b>"
                ),
                font=dict(size=17, color=c["text"]),
                x=0.02,
                xanchor="left",
            )
        )

    fig.update_xaxes(
        gridcolor=c["grid"],
        zeroline=False,
        linecolor=c["border"],
        linewidth=1,
        tickfont=dict(size=11, color=c["textMuted"]),
    )
    fig.update_yaxes(
        gridcolor=c["grid"],
        zeroline=False,
        linecolor=c["border"],
        linewidth=1,
        tickfont=dict(size=11, color=c["textMuted"]),
    )

    return fig
