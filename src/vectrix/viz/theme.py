"""
Vectrix brand theme for Plotly charts.

Provides consistent colors, layout defaults, and helper functions
for creating publication-quality dark-themed visualizations.
"""


COLORS = {
    "primary": "#6366f1",
    "accent": "#a855f7",
    "positive": "#22c55e",
    "negative": "#ef4444",
    "warning": "#f59e0b",
    "muted": "#94a3b8",
    "bg": "#0f172a",
    "card": "#1e293b",
    "text": "#f1f5f9",
    "grid": "rgba(255,255,255,0.06)",
}

LIGHT_COLORS = {
    "primary": "#6366f1",
    "accent": "#a855f7",
    "positive": "#16a34a",
    "negative": "#dc2626",
    "warning": "#d97706",
    "muted": "#64748b",
    "bg": "#ffffff",
    "card": "#f8fafc",
    "text": "#0f172a",
    "grid": "rgba(0,0,0,0.06)",
}

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"]),
    margin=dict(l=60, r=30, t=60, b=40),
)

PALETTE = [
    COLORS["primary"],
    COLORS["accent"],
    COLORS["positive"],
    COLORS["warning"],
    COLORS["negative"],
    "#38bdf8",
    "#fb923c",
    "#e879f9",
    "#34d399",
    "#f87171",
]

HEIGHT = {
    "chart": 450,
    "card": 220,
    "report": 600,
    "analysis": 650,
    "small": 350,
}


def applyTheme(fig, title=None, height=450, theme="dark"):
    """Apply Vectrix brand theme to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
    title : str, optional
    height : int
    theme : str
        'dark' (default) or 'light'.
    """
    colors = COLORS if theme == "dark" else LIGHT_COLORS
    layout = dict(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        paper_bgcolor=colors["bg"],
        plot_bgcolor=colors["bg"],
        font=dict(family="Inter, system-ui, sans-serif", color=colors["text"]),
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.update_layout(
        **layout,
        height=height,
        legend=dict(orientation="h", y=-0.15),
    )
    if title:
        fig.update_layout(title=dict(text=title, font_size=18))
    fig.update_xaxes(gridcolor=colors["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False)
    return fig
