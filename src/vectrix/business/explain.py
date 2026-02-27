"""
Forecast Explainer

예측 결과를 비전문가에게 설명하는 모듈:
- 성분 분해 (추세, 계절성, 잔차 기여도)
- 자연어 설명 생성
- 핵심 드라이버 식별
"""

import numpy as np
from typing import Dict, Any, List, Optional


class ForecastExplainer:
    """
    예측 설명기

    Usage:
        >>> explainer = ForecastExplainer()
        >>> explanation = explainer.explain(y, predictions, characteristics)
    """

    def explain(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        period: int = 7,
        locale: str = 'ko'
    ) -> Dict[str, Any]:
        """
        예측 결과 설명

        Returns
        -------
        Dict with keys: drivers, narrative, decomposition, confidence
        """
        n = len(y)
        drivers = self._identifyDrivers(y, period)
        decomposition = self._decompose(y, period)
        narrative = self._generateNarrative(y, predictions, drivers, period, locale)
        confidence = self._assessConfidence(y, predictions)

        return {
            'drivers': drivers,
            'narrative': narrative,
            'decomposition': decomposition,
            'confidence': confidence,
            'summary': self._generateSummary(drivers, locale)
        }

    def _identifyDrivers(self, y: np.ndarray, period: int) -> List[Dict[str, Any]]:
        n = len(y)
        totalVar = np.var(y) + 1e-10
        drivers = []

        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, y, 1)[0]
        trendLine = slope * x + np.polyfit(x, y, 1)[1]
        trendVar = np.var(trendLine)
        trendPct = min(trendVar / totalVar * 100, 100)

        direction = 'upward' if slope > 0 else 'downward' if slope < 0 else 'flat'
        drivers.append({
            'name': 'trend',
            'contribution': round(trendPct, 1),
            'direction': direction,
            'slope': float(slope)
        })

        if period > 1 and n >= period * 2:
            detrended = y - trendLine
            seasonal = np.zeros(n)
            for i in range(period):
                vals = detrended[i::period]
                seasonal[i::period] = np.mean(vals)

            seasonalVar = np.var(seasonal)
            seasonalPct = min(seasonalVar / totalVar * 100, 100 - trendPct)
            drivers.append({
                'name': 'seasonality',
                'contribution': round(seasonalPct, 1),
                'period': period,
                'amplitude': float(np.max(seasonal) - np.min(seasonal))
            })

        recentMomentum = np.mean(np.diff(y[-min(10, n):])) if n > 1 else 0
        momentumPct = max(0, 100 - sum(d['contribution'] for d in drivers))
        drivers.append({
            'name': 'momentum',
            'contribution': round(momentumPct, 1),
            'direction': 'positive' if recentMomentum > 0 else 'negative',
            'value': float(recentMomentum)
        })

        drivers.sort(key=lambda d: -d['contribution'])
        return drivers

    def _decompose(self, y: np.ndarray, period: int) -> Dict[str, np.ndarray]:
        n = len(y)
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, y, 1)
        trend = coeffs[0] * x + coeffs[1]

        detrended = y - trend
        seasonal = np.zeros(n)

        if period > 1 and n >= period * 2:
            for i in range(period):
                vals = detrended[i::period]
                seasonal[i::period] = np.mean(vals)

        residual = y - trend - seasonal

        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }

    def _generateNarrative(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        drivers: List[Dict],
        period: int,
        locale: str
    ) -> str:
        topDriver = drivers[0] if drivers else {'name': 'unknown', 'contribution': 0}
        predChange = (predictions[-1] - y[-1]) / abs(y[-1]) * 100 if abs(y[-1]) > 0 else 0
        direction = '상승' if predChange > 1 else '하락' if predChange < -1 else '유지'

        if locale == 'ko':
            lines = [f"예측 결과: 향후 {len(predictions)}기간 동안 {direction} 전망"]

            for d in drivers:
                if d['name'] == 'trend':
                    dirKo = '상승' if d.get('direction') == 'upward' else '하락' if d.get('direction') == 'downward' else '수평'
                    lines.append(f"  - 추세 ({d['contribution']}%): {dirKo} 추세")
                elif d['name'] == 'seasonality':
                    lines.append(f"  - 계절성 ({d['contribution']}%): {d.get('period', period)}일 주기")
                elif d['name'] == 'momentum':
                    dirKo = '양' if d.get('direction') == 'positive' else '음'
                    lines.append(f"  - 최근 모멘텀 ({d['contribution']}%): {dirKo}의 방향")

            return '\n'.join(lines)
        else:
            lines = [f"Forecast: {direction} trend over next {len(predictions)} periods"]
            for d in drivers:
                lines.append(f"  - {d['name']} ({d['contribution']}%)")
            return '\n'.join(lines)

    def _assessConfidence(self, y: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
        dataStd = np.std(y)
        predStd = np.std(predictions)
        cv = dataStd / abs(np.mean(y)) if abs(np.mean(y)) > 0 else 1.0

        if cv < 0.1:
            level = 'high'
            score = 85
        elif cv < 0.3:
            level = 'medium'
            score = 65
        else:
            level = 'low'
            score = 40

        return {
            'level': level,
            'score': score,
            'dataVariability': float(cv),
            'predictionStability': float(predStd / (dataStd + 1e-10))
        }

    def _generateSummary(self, drivers: List[Dict], locale: str) -> str:
        parts = []
        for d in drivers[:3]:
            parts.append(f"{d['contribution']}% {d['name']}")

        if locale == 'ko':
            return f"예측 주요 요인: {', '.join(parts)}"
        return f"Key drivers: {', '.join(parts)}"
