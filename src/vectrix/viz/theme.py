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

LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"]),
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


def applyTheme(fig, title=None, height=450):
    """Apply Vectrix brand theme to a Plotly figure."""
    fig.update_layout(
        **LAYOUT,
        height=height,
        legend=dict(orientation="h", y=-0.15),
    )
    if title:
        fig.update_layout(title=dict(text=title, font_size=18))
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig
