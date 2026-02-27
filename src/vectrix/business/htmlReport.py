"""자체 포함 HTML 예측 보고서"""
from datetime import datetime

import numpy as np


class HTMLReportGenerator:
    """CDN 없이 완전 자체 포함 HTML 보고서 생성"""

    def generate(
        self,
        historicalData: np.ndarray,
        predictions: np.ndarray,
        lower95: np.ndarray,
        upper95: np.ndarray,
        modelName: str = "Auto",
        dates: list = None,
        title: str = "Vectrix 예측 보고서",
        outputPath: str = "report.html",
    ) -> str:
        """HTML 보고서 생성 후 파일 경로 반환"""

        sections = []
        sections.append(self._overviewCard(historicalData, predictions, modelName))
        sections.append(self._forecastChart(historicalData, predictions, lower95, upper95, modelName))
        sections.append(self._metricsTable(historicalData, predictions))
        sections.append(self._dataTable(predictions, lower95, upper95, dates))

        html = self._assemble(title, sections)
        with open(outputPath, 'w', encoding='utf-8') as f:
            f.write(html)
        return outputPath

    def _overviewCard(self, historical, predictions, modelName):
        mean = float(np.mean(predictions))
        std = float(np.std(predictions))
        histMean = float(np.mean(historical))
        change = (mean - histMean) / abs(histMean) * 100 if abs(histMean) > 1e-10 else 0.0
        changeColor = '#27ae60' if change >= 0 else '#e74c3c'
        changeArrow = '\u2191' if change >= 0 else '\u2193'

        return f'''
        <div class="card">
            <h2>개요</h2>
            <div class="metrics-row">
                <div class="metric">
                    <div class="value">{modelName}</div>
                    <div class="label">최적 모델</div>
                </div>
                <div class="metric">
                    <div class="value">{len(predictions)}</div>
                    <div class="label">예측 기간</div>
                </div>
                <div class="metric">
                    <div class="value">{mean:,.1f}</div>
                    <div class="label">예측 평균</div>
                </div>
                <div class="metric">
                    <div class="value" style="color:{changeColor}">{changeArrow} {abs(change):.1f}%</div>
                    <div class="label">변화율</div>
                </div>
                <div class="metric">
                    <div class="value">{std:,.1f}</div>
                    <div class="label">예측 표준편차</div>
                </div>
            </div>
        </div>'''

    def _forecastChart(self, historical, predictions, lower, upper, modelName):
        """matplotlib -> SVG 인라인"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import io

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(14, 5))

            nHist = len(historical)
            nPred = len(predictions)

            ax.plot(range(nHist), historical, color='#2c3e50', linewidth=1.2, label='실측')
            predX = range(nHist, nHist + nPred)
            ax.plot(predX, predictions, color='#e74c3c', linewidth=2, label=f'예측 ({modelName})')
            ax.fill_between(predX, lower, upper, color='#e74c3c', alpha=0.15, label='95% CI')
            ax.axvline(x=nHist - 0.5, color='#95a5a6', linestyle='--', alpha=0.5)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_title('시계열 예측 결과', fontsize=14, fontweight='bold')
            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='svg', bbox_inches='tight')
            plt.close(fig)
            svgStr = buf.getvalue().decode('utf-8')

            return f'<div class="card"><h2>예측 그래프</h2>{svgStr}</div>'
        except ImportError:
            return '<div class="card"><h2>예측 그래프</h2><p>matplotlib 미설치 — 그래프 생략</p></div>'

    def _metricsTable(self, historical, predictions):
        histMean = float(np.mean(historical))
        histStd = float(np.std(historical))
        predMean = float(np.mean(predictions))
        predMin = float(np.min(predictions))
        predMax = float(np.max(predictions))

        return f'''
        <div class="card">
            <h2>주요 지표</h2>
            <table>
                <tr><th>지표</th><th>과거</th><th>예측</th></tr>
                <tr><td>평균</td><td>{histMean:,.2f}</td><td>{predMean:,.2f}</td></tr>
                <tr><td>표준편차</td><td>{histStd:,.2f}</td><td>{float(np.std(predictions)):,.2f}</td></tr>
                <tr><td>최솟값</td><td>{float(np.min(historical)):,.2f}</td><td>{predMin:,.2f}</td></tr>
                <tr><td>최댓값</td><td>{float(np.max(historical)):,.2f}</td><td>{predMax:,.2f}</td></tr>
            </table>
        </div>'''

    def _dataTable(self, predictions, lower, upper, dates):
        rows = []
        for i in range(len(predictions)):
            dateStr = dates[i] if dates and i < len(dates) else f"Step {i+1}"
            rows.append(
                f'<tr><td>{dateStr}</td><td>{predictions[i]:,.2f}</td>'
                f'<td>{lower[i]:,.2f}</td><td>{upper[i]:,.2f}</td></tr>'
            )
        tableRows = '\n'.join(rows)

        return f'''
        <div class="card">
            <h2>예측 데이터</h2>
            <table>
                <tr><th>날짜</th><th>예측</th><th>하한 (95%)</th><th>상한 (95%)</th></tr>
                {tableRows}
            </table>
        </div>'''

    def _assemble(self, title, sections):
        body = '\n'.join(sections)
        return f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: 'Segoe UI', -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; padding: 24px; background: #f0f2f5; color: #333; }}
h1 {{ color: #1a1a2e; margin-bottom: 4px; }}
.subtitle {{ color: #666; margin-bottom: 24px; }}
.card {{ background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card h2 {{ color: #1a1a2e; font-size: 18px; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 8px; }}
.metrics-row {{ display: flex; flex-wrap: wrap; gap: 16px; }}
.metric {{ flex: 1; min-width: 120px; text-align: center; padding: 16px; background: #f8f9fa; border-radius: 8px; }}
.metric .value {{ font-size: 24px; font-weight: 700; color: #2c3e50; }}
.metric .label {{ font-size: 12px; color: #7f8c8d; margin-top: 4px; text-transform: uppercase; }}
table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
th, td {{ padding: 10px 14px; text-align: right; border-bottom: 1px solid #eee; }}
th {{ background: #f8f9fa; font-weight: 600; text-align: left; }}
td:first-child, th:first-child {{ text-align: left; }}
tr:hover {{ background: #f8f9fa; }}
svg {{ width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Vectrix v3.0.0</p>
{body}
</body>
</html>'''
