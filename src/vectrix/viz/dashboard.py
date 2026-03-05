"""
HTML dashboard generator for Vectrix.

Produces a self-contained HTML report with embedded Plotly charts,
KPI cards, tables, and analysis summaries. Design follows shadcn/ui
patterns with Vectrix brand tokens (Cyan->Purple, dark navy).

Report narrative: Overview -> Data Profile -> Forecast Results -> Charts

Usage:
    from vectrix.viz import dashboard
    report = dashboard(forecast=result, analysis=analysis, historical=df)
    report.show()            # Jupyter inline or open browser
    report.save("report.html")
"""

import datetime
import html as htmllib
import math
import tempfile
import webbrowser
from pathlib import Path

from .theme import PALETTE

try:
    import plotly.io as pio
except ImportError:
    raise ImportError(
        "plotly is required for vectrix.viz. "
        "Install it with: pip install vectrix[viz]"
    )


_CSS = """
:root {
  --bg: #020617;
  --surface: #0f172a;
  --card: #1e293b;
  --card-hover: #263548;
  --border: rgba(255,255,255,0.08);
  --border-strong: rgba(255,255,255,0.12);
  --text: #f8fafc;
  --text-secondary: #cbd5e1;
  --text-muted: #94a3b8;
  --dim: #64748b;
  --subtle: #475569;
  --primary: #06b6d4;
  --primary-dark: #0891b2;
  --primary-muted: rgba(6,182,212,0.15);
  --accent: #8b5cf6;
  --accent-muted: rgba(139,92,246,0.15);
  --positive: #10b981;
  --positive-muted: rgba(16,185,129,0.12);
  --negative: #ef4444;
  --negative-muted: rgba(239,68,68,0.12);
  --warning: #f59e0b;
  --warning-muted: rgba(245,158,11,0.12);
  --radius: 8px;
  --radius-lg: 12px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: Inter, -apple-system, system-ui, sans-serif;
  line-height: 1.6;
  padding: 40px 32px;
  max-width: 1200px;
  margin: 0 auto;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Header */
.vx-header {
  padding-bottom: 32px;
  margin-bottom: 40px;
  border-bottom: 1px solid var(--border);
}
.vx-header-brand {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
}
.vx-header-wordmark {
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.vx-header h1 {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  margin-bottom: 12px;
}
.vx-tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.vx-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 12px;
  border-radius: 9999px;
  font-size: 12px;
  font-weight: 500;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--text-muted);
}
.vx-tag-primary { border-color: rgba(6,182,212,0.25); color: var(--primary); }
.vx-tag-accent { border-color: rgba(139,92,246,0.25); color: var(--accent); }
.vx-tag-positive { border-color: rgba(16,185,129,0.25); color: var(--positive); }
.vx-dot {
  width: 6px; height: 6px; border-radius: 50%%;
  display: inline-block; flex-shrink: 0;
}

/* Section */
.vx-section {
  margin-bottom: 40px;
}
.vx-section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 16px;
}
.vx-section-title {
  font-size: 16px;
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--text);
}
.vx-section-badge {
  font-size: 12px;
  color: var(--text-muted);
}

/* Card base */
.vx-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.vx-card-pad {
  padding: 24px;
}

/* KPI grid */
.vx-kpi-grid {
  display: grid;
  gap: 16px;
}
.vx-kpi-grid.cols-5 { grid-template-columns: repeat(5, 1fr); }
.vx-kpi-grid.cols-4 { grid-template-columns: repeat(4, 1fr); }
.vx-kpi-grid.cols-3 { grid-template-columns: repeat(3, 1fr); }
.vx-kpi-grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
.vx-kpi {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  transition: border-color 0.15s;
}
.vx-kpi:hover {
  border-color: var(--border-strong);
}
.vx-kpi-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 8px;
}
.vx-kpi-value {
  font-size: 28px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
  line-height: 1.1;
}
.vx-kpi-sub {
  font-size: 12px;
  color: var(--dim);
  margin-top: 6px;
}
.vx-kpi-indicator {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%%;
  margin-right: 6px;
  vertical-align: middle;
}

/* Color utilities */
.vx-good { color: var(--positive); }
.vx-bad { color: var(--negative); }
.vx-warn { color: var(--warning); }
.vx-cyan { color: var(--primary); }
.vx-purple { color: var(--accent); }
.vx-muted { color: var(--dim); }
.vx-text { color: var(--text-secondary); }

/* Executive summary */
.vx-summary-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px 28px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}
.vx-summary-col h3 {
  font-size: 13px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--dim);
  margin-bottom: 12px;
}
.vx-summary-col p, .vx-summary-col li {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.8;
}
.vx-insight-list {
  list-style: none;
  padding: 0;
}
.vx-insight-list li {
  padding: 3px 0 3px 18px;
  position: relative;
}
.vx-insight-list li::before {
  content: "";
  position: absolute;
  left: 0;
  top: 12px;
  width: 5px;
  height: 5px;
  border-radius: 50%%;
  background: var(--primary);
}

/* Feature bars */
.vx-feature-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.vx-feat {
  display: grid;
  grid-template-columns: 130px 1fr 50px;
  align-items: center;
  padding: 7px 0;
  gap: 12px;
}
.vx-feat-name {
  font-size: 13px;
  color: var(--text-muted);
}
.vx-feat-bar-bg {
  height: 6px;
  border-radius: 3px;
  background: rgba(255,255,255,0.05);
  overflow: hidden;
}
.vx-feat-bar {
  height: 100%%;
  border-radius: 3px;
}
.vx-feat-val {
  font-size: 12px;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: var(--text-secondary);
  text-align: right;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
}

/* Chart grid */
.vx-chart-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.vx-chart-grid.cols-1 { grid-template-columns: 1fr; }
.vx-chart-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 4px;
  overflow: hidden;
}
.vx-chart-card.full { grid-column: 1 / -1; }

/* Table */
.vx-table {
  width: 100%%;
  border-collapse: collapse;
  font-size: 13px;
}
.vx-table thead { background: rgba(255,255,255,0.02); }
.vx-table th {
  padding: 12px 20px;
  text-align: left;
  font-weight: 500;
  font-size: 12px;
  color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}
.vx-table th.r { text-align: right; }
.vx-table td {
  padding: 10px 20px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  font-variant-numeric: tabular-nums;
}
.vx-table td.r {
  text-align: right;
  font-family: 'JetBrains Mono', ui-monospace, monospace;
  font-size: 12px;
}
.vx-table tr:last-child td { border-bottom: none; }
.vx-table tr:hover td { background: rgba(255,255,255,0.02); }
.vx-table .vx-rank {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px; height: 22px;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 600;
  margin-right: 10px;
  background: rgba(255,255,255,0.05);
  color: var(--dim);
}
.vx-table .vx-rank-1 {
  background: var(--primary-muted);
  color: var(--primary);
}

/* Stats row */
.vx-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1px;
  background: var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.vx-stat {
  background: var(--surface);
  padding: 16px 20px;
}
.vx-stat-label {
  font-size: 12px;
  color: var(--dim);
  margin-bottom: 4px;
}
.vx-stat-value {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-secondary);
  font-variant-numeric: tabular-nums;
}

/* Footer */
.vx-footer {
  margin-top: 40px;
  padding: 32px 0 8px;
  border-top: 1px solid var(--border);
}
.vx-footer-brand {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 12px;
}
.vx-footer-logo {
  font-size: 15px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.vx-footer-tagline {
  font-size: 12px;
  color: var(--dim);
}
.vx-footer-features {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 6px 16px;
  margin-bottom: 16px;
}
.vx-footer-feat {
  font-size: 11px;
  color: var(--subtle);
}
.vx-footer-links {
  text-align: center;
  font-size: 11px;
  color: var(--dim);
}
.vx-footer-links a {
  color: var(--primary);
  text-decoration: none;
}
.vx-footer-links a:hover { text-decoration: underline; }

@media (max-width: 768px) {
  .vx-chart-grid { grid-template-columns: 1fr; }
  .vx-summary-card { grid-template-columns: 1fr; }
  .vx-kpi-grid { grid-template-columns: repeat(2, 1fr); }
  .vx-tag-row { gap: 6px; }
  body { padding: 20px 16px; }
}
"""


class DashboardResult:
    """Container for generated HTML dashboard."""

    def __init__(self, htmlContent, title):
        self._html = htmlContent
        self._title = title

    @property
    def html(self):
        return self._html

    def save(self, path):
        """Save dashboard to an HTML file."""
        Path(path).write_text(self._html, encoding="utf-8")
        return path

    def show(self):
        """Display dashboard. Jupyter -> inline, terminal -> browser."""
        try:
            from IPython.display import HTML, display
            get_ipython()  # noqa: F821
            display(HTML(self._html))
        except (ImportError, NameError):
            tmp = Path(tempfile.mkdtemp()) / f"vectrix_{self._title.replace(' ', '_')}.html"
            self.save(tmp)
            webbrowser.open(f"file://{tmp}")

    def _repr_html_(self):
        return self._html


def _safe(val, fmt=".2f", suffix="", fallback="--"):
    if val is None:
        return fallback
    try:
        f = float(val)
        if math.isinf(f) or math.isnan(f):
            return fallback
        return f"{f:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return fallback


def _safeFloat(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isinf(f) or math.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _gradeCls(val, low, high):
    if val is None:
        return "vx-muted"
    v = _safeFloat(val)
    if v is None:
        return "vx-muted"
    if v < low:
        return "vx-good"
    if v < high:
        return "vx-warn"
    return "vx-bad"


def _figToHtml(fig, height=400):
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
        default_height=f"{height}px",
    )


_LOGO_B64 = (
    "data:image/png;base64,"
    "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"
    "UVORK5CYII="
)


def _buildHeader(title, forecastResult, analysisResult):
    escapedTitle = htmllib.escape(title)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    tags = []

    if forecastResult:
        model = getattr(forecastResult, "model", None)
        if model:
            tags.append(f'<span class="vx-tag vx-tag-primary"><span class="vx-dot" style="background:var(--primary)"></span>{htmllib.escape(str(model))}</span>')
        steps = getattr(forecastResult, "predictions", None)
        if steps is not None:
            tags.append(f'<span class="vx-tag vx-tag-accent"><span class="vx-dot" style="background:var(--accent)"></span>{len(steps)} steps ahead</span>')

    if analysisResult:
        dna = getattr(analysisResult, "dna", None)
        if dna:
            tags.append(f'<span class="vx-tag vx-tag-positive"><span class="vx-dot" style="background:var(--positive)"></span>{htmllib.escape(dna.difficulty)}</span>')

    tags.append(f'<span class="vx-tag">{now}</span>')
    tagHtml = "".join(tags)

    return f"""
    <div class="vx-header">
      <div class="vx-header-brand">
        <img src="{_LOGO_B64}" alt="Vectrix" width="28" height="28" style="border-radius:4px" />
        <span class="vx-header-wordmark">Vectrix</span>
      </div>
      <h1>{escapedTitle}</h1>
      <div class="vx-tag-row">{tagHtml}</div>
    </div>
    """


def _buildOverviewKpis(forecastResult, analysisResult, comparison, historical):
    cards = []
    chars = getattr(analysisResult, "characteristics", None) if analysisResult else None

    if chars:
        length = getattr(chars, "length", None)
        if length:
            cards.append(("Observations", f"{length:,}", "vx-text", "Data points"))
        freq = getattr(chars, "frequency", None)
        if freq:
            freqStr = freq.value if hasattr(freq, "value") else str(freq)
            cards.append(("Frequency", htmllib.escape(freqStr), "vx-text", "Sampling rate"))
    elif historical is not None:
        cards.append(("Observations", f"{len(historical):,}", "vx-text", "Data points"))

    if analysisResult:
        dna = getattr(analysisResult, "dna", None)
        if dna:
            score = dna.difficultyScore
            cls = "vx-good" if score < 40 else "vx-warn" if score < 70 else "vx-bad"
            cards.append(("Difficulty", f"{score:.0f}/100", cls, dna.difficulty))

    if forecastResult:
        steps = getattr(forecastResult, "predictions", None)
        if steps is not None:
            cards.append(("Horizon", f"{len(steps)}", "vx-cyan", "Forecast steps"))

    if not cards:
        return ""

    n = min(len(cards), 5)
    kpiHtml = ""
    for label, value, cls, sub in cards[:5]:
        kpiHtml += f"""
        <div class="vx-kpi">
          <div class="vx-kpi-label">{htmllib.escape(label)}</div>
          <div class="vx-kpi-value {cls}">{value}</div>
          <div class="vx-kpi-sub">{htmllib.escape(sub)}</div>
        </div>"""

    return f"""
    <div class="vx-section">
      <div class="vx-section-head">
        <div class="vx-section-title">Overview</div>
      </div>
      <div class="vx-kpi-grid cols-{n}">{kpiHtml}</div>
    </div>
    """


def _buildProfileSection(analysisResult, historical):
    if analysisResult is None:
        return ""

    dna = getattr(analysisResult, "dna", None)
    if dna is None:
        return ""

    feat = dna.features
    chars = getattr(analysisResult, "characteristics", None)

    featureSpec = [
        ("Trend Strength", "trendStrength"),
        ("Seasonality", "seasonalStrength"),
        ("Forecastability", "forecastability"),
        ("Hurst Exponent", "hurstExponent"),
        ("Vol. Clustering", "volatilityClustering"),
        ("Nonlinearity", "nonlinearAutocorr"),
        ("Entropy", "entropy"),
        ("ACF Sum", "acfSum"),
    ]

    featureRows = ""
    for i, (label, key) in enumerate(featureSpec):
        v = _safeFloat(feat.get(key), 0)
        pct = max(0, min(v * 100, 100))
        color = PALETTE[i % len(PALETTE)]
        featureRows += f"""
        <div class="vx-feat">
          <span class="vx-feat-name">{htmllib.escape(label)}</span>
          <div class="vx-feat-bar-bg">
            <div class="vx-feat-bar" style="width:{pct:.0f}%;background:{color};opacity:0.8"></div>
          </div>
          <span class="vx-feat-val">{v:.3f}</span>
        </div>"""

    insights = _buildInsightItems(analysisResult)
    insightItems = "".join(f"<li>{htmllib.escape(s)}</li>" for s in insights) if insights else ""
    insightHtml = f'<ul class="vx-insight-list">{insightItems}</ul>' if insightItems else '<p style="color:var(--dim);font-size:13px">No insights available.</p>'

    cps = getattr(analysisResult, "changepoints", [])
    anoms = getattr(analysisResult, "anomalies", [])

    statItems = [
        ("Category", htmllib.escape(dna.category)),
        ("Difficulty", f"{dna.difficultyScore:.0f}/100 ({htmllib.escape(dna.difficulty)})"),
    ]

    if chars:
        freq = getattr(chars, "frequency", None)
        if freq:
            freqStr = freq.value if hasattr(freq, "value") else str(freq)
            statItems.append(("Frequency", htmllib.escape(freqStr)))
        period = getattr(chars, "period", None)
        if period:
            statItems.append(("Period", str(period)))
        dateRange = getattr(chars, "dateRange", None)
        if dateRange and len(dateRange) == 2:
            statItems.append(("Date Range", f"{dateRange[0]} to {dateRange[1]}"))
        length = getattr(chars, "length", None)
        if length:
            statItems.append(("Observations", f"{length:,}"))
        volLevel = getattr(chars, "volatilityLevel", None)
        if volLevel:
            statItems.append(("Volatility", htmllib.escape(str(volLevel))))
        predScore = getattr(chars, "predictabilityScore", None)
        if predScore is not None:
            statItems.append(("Predictability", _safe(predScore, ".0f", "/100")))
    elif historical is not None:
        statItems.append(("Observations", f"{len(historical):,}"))

    statItems.append(("Changepoints", str(len(cps))))
    statItems.append(("Anomalies", str(len(anoms))))

    descriptiveHtml = ""
    mean = _safeFloat(feat.get("mean"))
    std = _safeFloat(feat.get("std"))
    cv = _safeFloat(feat.get("cv"))
    skew = _safeFloat(feat.get("skewness"))
    kurt = _safeFloat(feat.get("kurtosis"))
    fMin = _safeFloat(feat.get("min"))
    fMax = _safeFloat(feat.get("max"))
    iqr = _safeFloat(feat.get("iqr"))
    snr = _safeFloat(feat.get("signalToNoise"))
    acf1 = _safeFloat(feat.get("acf1"))

    descItems = []
    if mean is not None:
        descItems.append(("Mean", _safe(mean, ",.2f")))
    if std is not None:
        descItems.append(("Std Dev", _safe(std, ",.2f")))
    if fMin is not None and fMax is not None:
        descItems.append(("Range", f"{_safe(fMin, ',.1f')} — {_safe(fMax, ',.1f')}"))
    if iqr is not None:
        descItems.append(("IQR", _safe(iqr, ",.2f")))
    if cv is not None:
        descItems.append(("CV", _safe(cv, ".3f")))
    if skew is not None:
        descItems.append(("Skewness", _safe(skew, ".3f")))
    if kurt is not None:
        descItems.append(("Kurtosis", _safe(kurt, ".3f")))
    if snr is not None:
        descItems.append(("SNR", _safe(snr, ".1f", "dB")))
    if acf1 is not None:
        descItems.append(("ACF(1)", _safe(acf1, ".3f")))

    if descItems:
        descCells = "".join(
            f'<div class="vx-stat"><div class="vx-stat-label">{k}</div><div class="vx-stat-value">{v}</div></div>'
            for k, v in descItems
        )
        descriptiveHtml = f"""
        <div class="vx-section-head" style="margin-top:20px">
          <div class="vx-section-title" style="font-size:14px">Descriptive Statistics</div>
        </div>
        <div class="vx-stats">{descCells}</div>
        """

    recModels = getattr(dna, "recommendedModels", [])
    recHtml = ""
    if recModels:
        badges = "".join(
            f'<span class="vx-tag vx-tag-primary" style="font-size:11px">{htmllib.escape(m)}</span>'
            for m in recModels[:5]
        )
        recHtml = f"""
        <div style="margin-top:16px">
          <div style="font-size:12px;color:var(--dim);margin-bottom:8px">Recommended Models</div>
          <div class="vx-tag-row">{badges}</div>
        </div>
        """

    fingerprint = getattr(dna, "fingerprint", "")
    fpHtml = f' &middot; <span style="font-family:\'JetBrains Mono\',monospace;font-size:11px">{htmllib.escape(fingerprint)}</span>' if fingerprint else ""

    statCells = "".join(
        f'<div class="vx-stat"><div class="vx-stat-label">{k}</div><div class="vx-stat-value">{v}</div></div>'
        for k, v in statItems
    )

    return f"""
    <div class="vx-section">
      <div class="vx-section-head">
        <div class="vx-section-title">Data Profile</div>
        <div class="vx-section-badge">DNA Analysis{fpHtml}</div>
      </div>
      <div class="vx-stats" style="margin-bottom:20px">{statCells}</div>
      <div class="vx-summary-card">
        <div class="vx-summary-col">
          <h3>Feature Profile</h3>
          <div class="vx-feature-list">{featureRows}</div>
          {recHtml}
        </div>
        <div class="vx-summary-col">
          <h3>Key Insights</h3>
          {insightHtml}
        </div>
      </div>
      {descriptiveHtml}
    </div>
    """


def _buildInsightItems(analysisResult):
    dna = getattr(analysisResult, "dna", None)
    if dna is None:
        return []

    feat = dna.features
    items = []

    trend = _safeFloat(feat.get("trendStrength"), 0)
    if trend > 0.6:
        items.append(f"Strong upward/downward trend (strength {trend:.3f}) — trend-following models recommended")
    elif trend > 0.3:
        items.append(f"Moderate trend present (strength {trend:.3f})")
    else:
        items.append(f"No significant trend (strength {trend:.3f}) — stationary behavior")

    season = _safeFloat(feat.get("seasonalStrength"), 0)
    if season > 0.5:
        items.append(f"Strong seasonal pattern (strength {season:.3f}) — seasonal decomposition recommended")
    elif season > 0.2:
        items.append(f"Mild seasonality detected (strength {season:.3f})")
    else:
        items.append("No significant seasonality — non-seasonal models preferred")

    fcast = _safeFloat(feat.get("forecastability"), 0)
    if fcast > 0.7:
        items.append(f"Highly predictable signal (forecastability {fcast:.3f})")
    elif fcast > 0.4:
        items.append(f"Moderate predictability (forecastability {fcast:.3f})")
    else:
        items.append(f"Low predictability (forecastability {fcast:.3f}) — wider confidence intervals expected")

    hurst = _safeFloat(feat.get("hurstExponent"), 0.5)
    if hurst > 0.6:
        items.append(f"Long memory process (Hurst {hurst:.3f}) — past patterns persist")
    elif hurst < 0.4:
        items.append(f"Mean-reverting dynamics (Hurst {hurst:.3f}) — deviations self-correct")

    vol = _safeFloat(feat.get("volatilityClustering"), 0)
    if vol > 0.3:
        items.append(f"Volatility clustering detected ({vol:.3f}) — risk varies over time")

    cps = getattr(analysisResult, "changepoints", [])
    anoms = getattr(analysisResult, "anomalies", [])
    if len(cps) > 0:
        items.append(f"{len(cps)} structural break(s) found — regime changes in the data")
    if len(anoms) > 0:
        items.append(f"{len(anoms)} anomalous point(s) — may affect model accuracy")

    return items


def _buildPerformanceSection(forecastResult, comparison=None):
    if forecastResult is None:
        return ""

    model = getattr(forecastResult, "model", "Unknown")
    steps = getattr(forecastResult, "predictions", [])
    horizon = len(steps) if steps is not None else 0

    mape = _safeFloat(getattr(forecastResult, "mape", None))
    rmse = _safeFloat(getattr(forecastResult, "rmse", None))
    mae = _safeFloat(getattr(forecastResult, "mae", None))
    smape = _safeFloat(getattr(forecastResult, "smape", None))

    metricsSource = ""
    if mape is None and comparison is not None and len(comparison) > 0:
        best = comparison.iloc[0]
        mape = _safeFloat(best.get("mape")) if "mape" in comparison.columns else None
        rmse = _safeFloat(best.get("rmse")) if "rmse" in comparison.columns else None
        mae = _safeFloat(best.get("mae")) if "mae" in comparison.columns else None
        smape = _safeFloat(best.get("smape")) if "smape" in comparison.columns else None
        bestModel = best.get("model", "")
        if bestModel:
            model = str(bestModel)
        metricsSource = "Cross-validation"
    else:
        metricsSource = "Holdout"

    accuracy = (100 - mape) if mape is not None else None
    accCls = "vx-good" if accuracy and accuracy >= 90 else "vx-warn" if accuracy and accuracy >= 80 else "vx-bad" if accuracy else "vx-muted"

    kpis = [
        ("Forecast Accuracy", _safe(accuracy, ".1f", "%"), accCls, "100% - MAPE"),
        ("MAPE", _safe(mape, ".2f", "%"), _gradeCls(mape, 10, 20), "Mean absolute % error"),
        ("RMSE", _safe(rmse, ",.2f"), "vx-text", "Root mean square error"),
        ("MAE", _safe(mae, ",.2f"), "vx-text", "Mean absolute error"),
        ("sMAPE", _safe(smape, ".2f", "%"), _gradeCls(smape, 10, 20), "Symmetric MAPE"),
    ]

    kpiHtml = ""
    for label, value, cls, sub in kpis:
        kpiHtml += f"""
        <div class="vx-kpi">
          <div class="vx-kpi-label">{htmllib.escape(label)}</div>
          <div class="vx-kpi-value {cls}">{value}</div>
          <div class="vx-kpi-sub">{htmllib.escape(sub)}</div>
        </div>"""

    import numpy as np

    detailItems = []
    preds = getattr(forecastResult, "predictions", None)
    upper = getattr(forecastResult, "upper", None)
    lower = getattr(forecastResult, "lower", None)
    dates = getattr(forecastResult, "dates", None)

    if preds is not None and len(preds) > 0:
        predArr = np.array(preds, dtype=float)
        detailItems.append(("Prediction Range", f"{_safe(predArr.min(), ',.1f')} — {_safe(predArr.max(), ',.1f')}"))

    if upper is not None and lower is not None:
        upperArr = np.array(upper, dtype=float)
        lowerArr = np.array(lower, dtype=float)
        avgWidth = float(np.mean(upperArr - lowerArr))
        detailItems.append(("Avg CI Width", _safe(avgWidth, ",.1f")))
        detailItems.append(("Confidence Level", "95%"))

    if dates is not None and len(dates) > 0:
        detailItems.append(("Forecast Start", str(dates[0])[:10]))
        detailItems.append(("Forecast End", str(dates[-1])[:10]))

    models = getattr(forecastResult, "models", None)
    if models and len(models) > 0:
        detailItems.append(("Ensemble Members", str(len(models))))

    detailHtml = ""
    if detailItems:
        detailCells = "".join(
            f'<div class="vx-stat"><div class="vx-stat-label">{k}</div><div class="vx-stat-value">{v}</div></div>'
            for k, v in detailItems
        )
        detailHtml = f'<div class="vx-stats" style="margin-top:16px">{detailCells}</div>'

    compTableHtml = _buildComparisonTableInline(comparison)

    return f"""
    <div class="vx-section">
      <div class="vx-section-head">
        <div class="vx-section-title">Forecast Results</div>
        <div class="vx-section-badge">{htmllib.escape(str(model))} &middot; {horizon} steps &middot; {metricsSource}</div>
      </div>
      <div class="vx-kpi-grid cols-5">{kpiHtml}</div>
      {detailHtml}
      {compTableHtml}
    </div>
    """


def _buildComparisonTableInline(comparison, top=10):
    if comparison is None or len(comparison) == 0:
        return ""

    df = comparison.head(top)
    metricCols = [c for c in ["model", "mape", "rmse", "mae", "smape"] if c in df.columns]
    if not metricCols:
        return ""

    timeCols = [c for c in ["time_ms"] if c in df.columns]
    allCols = metricCols + timeCols

    headers = ""
    for col in allCols:
        if col == "model":
            headers += "<th>Model</th>"
        elif col == "time_ms":
            headers += '<th class="r">Time</th>'
        else:
            headers += f'<th class="r">{col.upper()}</th>'

    rows = ""
    for idx, (_, row) in enumerate(df.iterrows()):
        cells = ""
        for col in allCols:
            val = row[col]
            if col == "model":
                rankCls = "vx-rank-1" if idx == 0 else ""
                cells += f'<td><span class="vx-rank {rankCls}">{idx + 1}</span>{htmllib.escape(str(val))}</td>'
            elif col == "time_ms":
                cells += f'<td class="r">{_safe(val, ".0f", "ms")}</td>'
            elif col == "mape":
                fv = _safeFloat(val)
                cls = _gradeCls(fv, 10, 20)
                cells += f'<td class="r {cls}">{_safe(val, ".2f", "%")}</td>'
            elif col == "smape":
                cells += f'<td class="r">{_safe(val, ".2f", "%")}</td>'
            else:
                cells += f'<td class="r">{_safe(val, ".2f")}</td>'
        rows += f"<tr>{cells}</tr>"

    return f"""
      <div style="margin-top:20px">
        <div class="vx-section-head">
          <div class="vx-section-title" style="font-size:14px">Model Comparison</div>
          <div class="vx-section-badge">Top {min(top, len(df))} &middot; ranked by MAPE</div>
        </div>
        <div class="vx-card">
          <table class="vx-table">
            <thead><tr>{headers}</tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
      </div>
    """


def _buildChartsSection(forecastResult=None, analysisResult=None, historical=None, theme="dark"):
    from .charts import dnaRadar, forecastChart

    charts = []

    if forecastResult:
        fig = forecastChart(forecastResult, historical=historical, theme=theme)
        fig.update_layout(margin=dict(l=48, r=20, t=56, b=40), height=400)
        charts.append(("Forecast", _figToHtml(fig, 400), False))

    if analysisResult:
        fig = dnaRadar(analysisResult, theme=theme)
        fig.update_layout(margin=dict(l=48, r=20, t=56, b=40), height=400)
        charts.append(("DNA Radar", _figToHtml(fig, 400), False))

    if not charts:
        return ""

    chartCards = ""
    for name, chtml, full in charts:
        fullCls = " full" if full else ""
        chartCards += f'<div class="vx-chart-card{fullCls}">{chtml}</div>'
    gridCls = "cols-1" if len(charts) == 1 else ""

    return f"""
    <div class="vx-section">
      <div class="vx-section-head">
        <div class="vx-section-title">Visualizations</div>
        <div class="vx-section-badge">{len(charts)} charts</div>
      </div>
      <div class="vx-chart-grid {gridCls}">{chartCards}</div>
    </div>
    """


def _buildFooter():
    features = [
        "30+ forecasting models",
        "Built-in Rust engine",
        "Forecast DNA profiling",
        "Automatic model selection",
        "Confidence intervals",
        "Anomaly detection",
        "What-if scenarios",
        "Zero-config API",
    ]
    featHtml = "".join(f'<span class="vx-footer-feat">{f}</span>' for f in features)

    return f"""
    <div class="vx-footer">
      <div class="vx-footer-brand">
        <span class="vx-footer-logo">Vectrix</span>
        <span class="vx-footer-tagline">Python syntax, Rust speed</span>
      </div>
      <div class="vx-footer-features">{featHtml}</div>
      <div class="vx-footer-links">
        <a href="https://github.com/eddmpython/vectrix">GitHub</a>
        &nbsp;&middot;&nbsp;
        <a href="https://pypi.org/project/vectrix/">PyPI</a>
        &nbsp;&middot;&nbsp;
        <a href="https://eddmpython.github.io/vectrix/">Docs</a>
        &nbsp;&middot;&nbsp;
        pip install vectrix
      </div>
    </div>
    """


def dashboard(
    forecast=None,
    analysis=None,
    comparison=None,
    historical=None,
    title="Vectrix Report",
    theme="dark",
):
    """
    Generate a self-contained HTML dashboard.

    Report follows a narrative flow:
    1. Overview KPIs (observations, frequency, difficulty, horizon)
    2. Data Profile (DNA features, statistical properties, insights)
    3. Forecast Results (accuracy metrics + model comparison table)
    4. Visualizations (forecast chart, DNA radar)

    Parameters
    ----------
    forecast : EasyForecastResult, optional
        Result from forecast().
    analysis : EasyAnalysisResult, optional
        Result from analyze().
    comparison : pd.DataFrame, optional
        Result from compare().
    historical : pd.DataFrame, optional
        Historical data for the forecast chart.
    title : str
        Dashboard title.
    theme : str
        'dark' (default) or 'light'.

    Returns
    -------
    DashboardResult
        Call .show() to display, .save(path) to export.
    """
    import sys
    import time

    isTerminal = not _isNotebook()
    t0 = time.time()

    def _log(msg):
        if isTerminal:
            elapsed = time.time() - t0
            sys.stderr.write(f"\r\033[K[vectrix] {msg} ({elapsed:.1f}s)")
            sys.stderr.flush()

    css = _CSS

    _log("Building header...")
    header = _buildHeader(title, forecast, analysis)

    _log("Building overview...")
    overview = _buildOverviewKpis(forecast, analysis, comparison, historical)

    _log("Analyzing data profile...")
    profile = _buildProfileSection(analysis, historical)

    _log("Computing performance metrics...")
    performance = _buildPerformanceSection(forecast, comparison)

    _log("Rendering charts...")
    charts = _buildChartsSection(forecast, analysis, historical, theme)

    footer = _buildFooter()

    _log("Assembling report...")
    escapedTitle = htmllib.escape(title)

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escapedTitle} — Vectrix</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{css}</style>
</head>
<body>

{header}
{overview}
{profile}
{performance}
{charts}
{footer}

</body>
</html>"""

    elapsed = time.time() - t0
    if isTerminal:
        sys.stderr.write(f"\r\033[K[vectrix] Dashboard ready ({elapsed:.1f}s, {len(page):,} bytes)\n")
        sys.stderr.flush()

    return DashboardResult(page, title)


def _isNotebook():
    try:
        get_ipython()  # noqa: F821
        return True
    except NameError:
        return False
