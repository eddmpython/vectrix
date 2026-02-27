"""
Level 1: 일직선 예측 사전 진단

예측 수행 전에 일직선 예측이 발생할 위험도를 진단합니다.
"""

from typing import Dict, List, Optional

import numpy as np

from ..types import MODEL_INFO, DataCharacteristics, FlatRiskAssessment, RiskLevel


class FlatRiskDiagnostic:
    """
    일직선 예측 위험도 사전 진단

    예측 전에 데이터를 분석하여 일직선 예측이 발생할 가능성을 평가합니다.
    위험도에 따라 적절한 모델 선택 전략을 권장합니다.
    """

    def __init__(self, period: int = 7):
        self.period = period

    def diagnose(
        self,
        values: np.ndarray,
        characteristics: Optional[DataCharacteristics] = None
    ) -> FlatRiskAssessment:
        """
        일직선 예측 위험도 진단

        Parameters
        ----------
        values : np.ndarray
            시계열 데이터
        characteristics : DataCharacteristics, optional
            이미 분석된 데이터 특성 (없으면 간단히 분석)

        Returns
        -------
        FlatRiskAssessment
            위험도 평가 결과
        """
        n = len(values)

        if n < 4:
            return FlatRiskAssessment(
                riskScore=1.0,
                riskLevel=RiskLevel.CRITICAL,
                riskFactors={'shortData': True},
                warnings=['데이터가 너무 적습니다 (최소 4개 필요)'],
                recommendedStrategy='naive_only',
                recommendedModels=['naive']
            )

        riskFactors = {
            'lowVariance': self._checkLowVariance(values),
            'weakSeasonality': self._checkWeakSeasonality(values),
            'noTrend': self._checkNoTrend(values),
            'shortData': self._checkShortData(values),
            'highNoise': self._checkHighNoise(values),
            'flatRecent': self._checkFlatRecent(values)
        }

        weights = {
            'lowVariance': 0.25,
            'weakSeasonality': 0.20,
            'noTrend': 0.15,
            'shortData': 0.15,
            'highNoise': 0.15,
            'flatRecent': 0.10
        }

        riskScore = sum(
            weights[k] * (1.0 if v else 0.0)
            for k, v in riskFactors.items()
        )

        if riskScore >= 0.7:
            riskLevel = RiskLevel.CRITICAL
        elif riskScore >= 0.5:
            riskLevel = RiskLevel.HIGH
        elif riskScore >= 0.3:
            riskLevel = RiskLevel.MEDIUM
        else:
            riskLevel = RiskLevel.LOW

        strategy, models = self._getRecommendation(riskLevel, riskFactors, n)
        warnings = self._generateWarnings(riskFactors, riskLevel)

        return FlatRiskAssessment(
            riskScore=riskScore,
            riskLevel=riskLevel,
            riskFactors=riskFactors,
            recommendedStrategy=strategy,
            recommendedModels=models,
            warnings=warnings
        )

    def _checkLowVariance(self, values: np.ndarray) -> bool:
        """변동성이 너무 낮은지 확인"""
        std = np.std(values)
        mean = np.mean(np.abs(values))

        if mean < 1e-10:
            return True

        cv = std / mean  # 변동계수
        return cv < 0.05  # 5% 미만이면 변동성 부족

    def _checkWeakSeasonality(self, values: np.ndarray) -> bool:
        """계절성이 약한지 확인"""
        n = len(values)
        period = self.period

        if n < period * 2:
            return True  # 데이터 부족으로 계절성 판단 불가

        try:
            seasonalMeans = []
            for i in range(period):
                indices = list(range(i, n, period))
                if indices:
                    seasonalMeans.append(np.mean(values[indices]))

            if not seasonalMeans:
                return True

            seasonalVar = np.var(seasonalMeans)
            totalVar = np.var(values)

            if totalVar < 1e-10:
                return True

            seasonalStrength = seasonalVar / totalVar
            return seasonalStrength < 0.15  # 15% 미만이면 계절성 약함

        except Exception:
            return True

    def _checkNoTrend(self, values: np.ndarray) -> bool:
        """추세가 없는지 확인"""
        n = len(values)

        if n < 10:
            return True

        try:
            x = np.arange(n)
            slope, _ = np.polyfit(x, values, 1)

            valueRange = np.max(values) - np.min(values)
            if valueRange < 1e-10:
                return True

            trendStrength = abs(slope * n) / valueRange
            return trendStrength < 0.1  # 추세 기여도 10% 미만

        except Exception:
            return True

    def _checkShortData(self, values: np.ndarray) -> bool:
        """데이터가 부족한지 확인"""
        n = len(values)
        minRequired = self.period * 2
        return n < max(minRequired, 20)

    def _checkHighNoise(self, values: np.ndarray) -> bool:
        """노이즈가 과다한지 확인"""
        n = len(values)

        if n < 5:
            return False

        try:
            # 이동평균으로 추세 추정
            windowSize = min(5, n // 3)
            if windowSize < 2:
                return False

            smoothed = np.convolve(
                values,
                np.ones(windowSize) / windowSize,
                mode='valid'
            )

            # 원본과 평활화 데이터의 차이 (노이즈)
            startIdx = windowSize // 2
            endIdx = startIdx + len(smoothed)
            noise = values[startIdx:endIdx] - smoothed

            noiseRatio = np.std(noise) / (np.std(values) + 1e-10)
            return noiseRatio > 0.7  # 노이즈가 70% 이상

        except Exception:
            return False

    def _checkFlatRecent(self, values: np.ndarray) -> bool:
        """최근 데이터가 평평한지 확인"""
        n = len(values)
        recentN = min(10, n // 2)

        if recentN < 3:
            return False

        recent = values[-recentN:]
        recentStd = np.std(recent)
        totalStd = np.std(values)

        if totalStd < 1e-10:
            return True

        return (recentStd / totalStd) < 0.3  # 최근 변동이 전체의 30% 미만

    def _getRecommendation(
        self,
        riskLevel: RiskLevel,
        riskFactors: Dict[str, bool],
        dataLength: int
    ) -> tuple:
        """위험도에 따른 전략 및 모델 추천"""

        if riskLevel == RiskLevel.CRITICAL:
            strategy = "force_seasonal"
            models = ['seasonal_naive', 'snaive_drift']

        elif riskLevel == RiskLevel.HIGH:
            strategy = "seasonal_priority"
            models = ['seasonal_naive', 'snaive_drift', 'mstl', 'theta']

        elif riskLevel == RiskLevel.MEDIUM:
            strategy = "balanced"
            models = ['mstl', 'holt_winters', 'theta', 'auto_arima']

        else:
            strategy = "standard"
            models = ['auto_arima', 'auto_ets', 'theta', 'mstl']

        models = [
            m for m in models
            if dataLength >= MODEL_INFO.get(m, {}).get('minData', 10)
        ]

        if not models:
            models = ['seasonal_naive'] if dataLength >= 7 else ['naive']

        return strategy, models

    def _generateWarnings(
        self,
        riskFactors: Dict[str, bool],
        riskLevel: RiskLevel
    ) -> List[str]:
        """경고 메시지 생성"""
        warnings = []

        warningMessages = {
            'lowVariance': '데이터 변동성이 매우 낮습니다. 예측이 일직선이 될 수 있습니다.',
            'weakSeasonality': '명확한 계절 패턴이 감지되지 않습니다.',
            'noTrend': '상승/하락 추세가 감지되지 않습니다.',
            'shortData': '데이터가 부족합니다. 더 긴 기간의 데이터가 필요합니다.',
            'highNoise': '노이즈가 많아 패턴 감지가 어렵습니다.',
            'flatRecent': '최근 데이터가 평평합니다. 패턴 변화 가능성.'
        }

        for factor, isRisk in riskFactors.items():
            if isRisk and factor in warningMessages:
                warnings.append(warningMessages[factor])

        if riskLevel in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            warnings.insert(0, f'⚠️ 일직선 예측 위험도: {riskLevel.value.upper()}')

        return warnings
