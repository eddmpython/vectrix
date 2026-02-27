"""
Report Generator

종합 예측 보고서 생성:
- Dict/JSON 형식 (API/프로그래밍용)
- HTML 형식 (시각화용, matplotlib optional)
- 텍스트 형식 (터미널/로그용)
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from .anomaly import AnomalyDetector
from .explain import ForecastExplainer
from .metrics import BusinessMetrics


class ReportGenerator:
    """
    종합 예측 보고서 생성기

    Usage:
        >>> gen = ReportGenerator()
        >>> report = gen.generate(y, predictions, period=7)
    """

    def __init__(self, locale: str = 'ko'):
        self.locale = locale
        self.anomalyDetector = AnomalyDetector()
        self.explainer = ForecastExplainer()
        self.metrics = BusinessMetrics()

    def generate(
        self,
        historicalData: np.ndarray,
        predictions: np.ndarray,
        lower95: Optional[np.ndarray] = None,
        upper95: Optional[np.ndarray] = None,
        period: int = 7,
        modelName: str = 'Vectrix',
        dates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        종합 보고서 생성

        Returns
        -------
        Dict with sections: overview, dataAnalysis, forecast, anomalies,
                            explanation, qualityMetrics
        """
        n = len(historicalData)
        steps = len(predictions)

        report = {
            'generatedAt': datetime.now().isoformat(),
            'version': 'Vectrix Report v1.0',
            'locale': self.locale,
        }

        report['overview'] = self._generateOverview(
            historicalData, predictions, modelName, period, dates
        )

        report['dataAnalysis'] = self._analyzeData(historicalData, period)

        report['forecast'] = self._generateForecastSection(
            historicalData, predictions, lower95, upper95, dates
        )

        anomalyResult = self.anomalyDetector.detect(historicalData, method='auto', period=period)
        report['anomalies'] = {
            'count': anomalyResult.nAnomalies,
            'ratio': round(anomalyResult.anomalyRatio * 100, 2),
            'details': anomalyResult.details[:10]
        }

        explanation = self.explainer.explain(historicalData, predictions, period, self.locale)
        report['explanation'] = explanation

        if n > steps:
            trainPred = predictions[:min(steps, n)]
            trainActual = historicalData[-len(trainPred):]
            report['qualityMetrics'] = self._computeQuality(trainActual, trainPred)
        else:
            report['qualityMetrics'] = {}

        report['recommendations'] = self._generateRecommendations(
            report, historicalData, predictions
        )

        return report

    def _generateOverview(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        modelName: str,
        period: int,
        dates: Optional[List[str]]
    ) -> Dict[str, Any]:
        return {
            'dataPoints': len(y),
            'forecastHorizon': len(predictions),
            'model': modelName,
            'seasonalPeriod': period,
            'dateRange': {
                'start': dates[0] if dates else 'N/A',
                'end': dates[-1] if dates else 'N/A',
            } if dates else {},
            'statistics': {
                'mean': round(float(np.mean(y)), 2),
                'std': round(float(np.std(y)), 2),
                'min': round(float(np.min(y)), 2),
                'max': round(float(np.max(y)), 2),
                'cv': round(float(np.std(y) / abs(np.mean(y))) if abs(np.mean(y)) > 0 else 0, 4),
            }
        }

    def _analyzeData(self, y: np.ndarray, period: int) -> Dict[str, Any]:
        n = len(y)

        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, y, 1)[0]
        trendDirection = 'upward' if slope > 0.01 * np.std(y) / n else 'downward' if slope < -0.01 * np.std(y) / n else 'flat'

        hasSeasonal = False
        if period > 1 and n >= period * 2:
            detrended = y - (slope * x + np.mean(y))
            seasonal = np.zeros(n)
            for i in range(period):
                vals = detrended[i::period]
                seasonal[i::period] = np.mean(vals)
            seasonalVar = np.var(seasonal)
            totalVar = np.var(y)
            hasSeasonal = seasonalVar > 0.05 * totalVar

        recentTrend = np.mean(np.diff(y[-min(10, n):])) if n > 1 else 0

        return {
            'trend': {
                'direction': trendDirection,
                'slope': round(float(slope), 6),
                'strength': round(min(abs(slope) * n / np.std(y), 1.0), 2) if np.std(y) > 0 else 0,
            },
            'seasonality': {
                'detected': hasSeasonal,
                'period': period,
            },
            'recentMomentum': round(float(recentTrend), 4),
            'stationarity': 'likely' if abs(slope) * n < 0.5 * np.std(y) else 'unlikely',
        }

    def _generateForecastSection(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        lower95: Optional[np.ndarray],
        upper95: Optional[np.ndarray],
        dates: Optional[List[str]]
    ) -> Dict[str, Any]:
        section = {
            'values': predictions.tolist(),
            'mean': round(float(np.mean(predictions)), 2),
            'min': round(float(np.min(predictions)), 2),
            'max': round(float(np.max(predictions)), 2),
        }

        if lower95 is not None and upper95 is not None:
            section['lower95'] = lower95.tolist()
            section['upper95'] = upper95.tolist()
            section['avgWidth'] = round(float(np.mean(upper95 - lower95)), 2)

        predChange = (predictions[-1] - y[-1]) / abs(y[-1]) * 100 if abs(y[-1]) > 0 else 0
        section['overallChange'] = round(float(predChange), 2)
        section['direction'] = 'up' if predChange > 1 else 'down' if predChange < -1 else 'stable'

        if dates:
            section['dates'] = dates

        return section

    def _computeQuality(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
        return self.metrics.calculate(actual, predicted)

    def _generateRecommendations(
        self,
        report: Dict,
        y: np.ndarray,
        predictions: np.ndarray
    ) -> List[str]:
        recs = []

        anomalyCount = report.get('anomalies', {}).get('count', 0)
        if anomalyCount > 0:
            if self.locale == 'ko':
                recs.append(f"이상치 {anomalyCount}개 감지됨 -데이터 품질 확인 권장")
            else:
                recs.append(f"{anomalyCount} anomalies detected -review data quality")

        cv = report.get('overview', {}).get('statistics', {}).get('cv', 0)
        if cv > 0.5:
            if self.locale == 'ko':
                recs.append("높은 변동성 -예측 신뢰구간에 주의")
            else:
                recs.append("High variability -pay attention to prediction intervals")

        conf = report.get('explanation', {}).get('confidence', {})
        if conf.get('level') == 'low':
            if self.locale == 'ko':
                recs.append("예측 신뢰도 낮음 -더 많은 데이터 수집 또는 외부 요인 검토 권장")
            else:
                recs.append("Low forecast confidence -consider collecting more data")

        if not recs:
            if self.locale == 'ko':
                recs.append("예측 품질 양호 -정기적 모니터링 권장")
            else:
                recs.append("Forecast quality is good -regular monitoring recommended")

        return recs

    def toText(self, report: Dict[str, Any]) -> str:
        """보고서를 텍스트로 변환"""
        lines = []
        lines.append("=" * 60)
        lines.append("Vectrix 예측 보고서" if self.locale == 'ko' else "Vectrix Forecast Report")
        lines.append(f"생성: {report.get('generatedAt', '')}")
        lines.append("=" * 60)

        overview = report.get('overview', {})
        lines.append(f"\n데이터: {overview.get('dataPoints', 0)}개 관측값")
        lines.append(f"예측 기간: {overview.get('forecastHorizon', 0)}스텝")
        lines.append(f"모델: {overview.get('model', 'N/A')}")

        stats = overview.get('statistics', {})
        lines.append(f"평균: {stats.get('mean', 0)}, 표준편차: {stats.get('std', 0)}")

        forecast = report.get('forecast', {})
        lines.append(f"\n예측 방향: {forecast.get('direction', 'N/A')}")
        lines.append(f"전체 변화: {forecast.get('overallChange', 0)}%")

        explanation = report.get('explanation', {})
        summary = explanation.get('summary', '')
        if summary:
            lines.append(f"\n{summary}")

        anomalies = report.get('anomalies', {})
        lines.append(f"\n이상치: {anomalies.get('count', 0)}개 ({anomalies.get('ratio', 0)}%)")

        recs = report.get('recommendations', [])
        if recs:
            lines.append("\n권장사항:")
            for r in recs:
                lines.append(f"  - {r}")

        lines.append("\n" + "=" * 60)
        return '\n'.join(lines)
