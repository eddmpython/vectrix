"""
Level 2: 적응형 모델 선택기

일직선 예측 위험도와 데이터 특성에 따라 최적의 모델을 선택합니다.
"""

from typing import Any, Dict, List, Optional

from ..types import MODEL_INFO, DataCharacteristics, FlatRiskAssessment, RiskLevel


class AdaptiveModelSelector:
    """
    적응형 모델 선택기

    일직선 예측 위험도가 높으면 Seasonal Naive 등 계절성 강제 모델을 우선 선택하고,
    위험도가 낮으면 Auto 모델들을 사용합니다.
    """

    # 모델별 일직선 저항력
    FLAT_RESISTANCE = {
        'seasonal_naive': 0.95,   # 무조건 계절 패턴 반복
        'snaive_drift': 0.90,     # 계절 패턴 + 드리프트
        'mstl': 0.85,             # 다중 계절 분해
        'holt_winters': 0.80,     # 삼중 지수평활
        'theta': 0.75,            # Theta 분해
        'prophet': 0.70,          # 계절성 강제 모델링
        'auto_arima': 0.60,       # ARIMA (계절성에 따라 다름)
        'auto_ets': 0.55,         # ETS (A,N,N 위험)
        'auto_theta': 0.70,       # AutoTheta
        'auto_ces': 0.65,         # AutoCES
        'naive': 0.10,            # 가장 위험
        'mean': 0.05,             # 가장 위험
        'rwd': 0.60,              # 추세 있으면 양호
        'window_avg': 0.15,       # 대부분 수평
    }

    # 모델별 최소 데이터 요구량
    MIN_DATA = {
        'seasonal_naive': 14,
        'snaive_drift': 14,
        'mstl': 50,
        'holt_winters': 24,
        'theta': 10,
        'prophet': 60,
        'auto_arima': 30,
        'auto_ets': 20,
        'auto_theta': 10,
        'auto_ces': 20,
        'naive': 2,
        'mean': 2,
        'rwd': 5,
        'window_avg': 5,
    }

    def __init__(self):
        pass

    def selectModels(
        self,
        flatRisk: FlatRiskAssessment,
        characteristics: DataCharacteristics,
        maxModels: int = 5
    ) -> List[str]:
        """
        위험도 기반 모델 선택

        Parameters
        ----------
        flatRisk : FlatRiskAssessment
            일직선 예측 위험도 평가 결과
        characteristics : DataCharacteristics
            데이터 특성
        maxModels : int
            최대 모델 수

        Returns
        -------
        List[str]
            선택된 모델 ID 목록
        """
        dataLength = characteristics.length
        riskLevel = flatRisk.riskLevel

        # 위험도 레벨에 따른 기본 모델 집합
        if riskLevel == RiskLevel.CRITICAL:
            candidates = self._getCriticalModels()
        elif riskLevel == RiskLevel.HIGH:
            candidates = self._getHighRiskModels()
        elif riskLevel == RiskLevel.MEDIUM:
            candidates = self._getMediumRiskModels()
        else:
            candidates = self._getLowRiskModels()

        # 데이터 길이로 필터링
        validModels = [
            m for m in candidates
            if dataLength >= self.MIN_DATA.get(m, 10)
        ]

        # 계절성에 따른 조정
        if characteristics.hasSeasonality:
            validModels = self._prioritizeSeasonalModels(validModels)
        else:
            # 계절성 없으면 계절성 강제 모델 제외
            validModels = [
                m for m in validModels
                if m not in ['seasonal_naive', 'snaive_drift']
            ]

        # 폴백
        if not validModels:
            validModels = ['theta'] if dataLength >= 10 else ['naive']

        return validModels[:maxModels]

    def _getCriticalModels(self) -> List[str]:
        """위험도 Critical: 계절성 강제 모델만"""
        return ['seasonal_naive', 'snaive_drift', 'mstl']

    def _getHighRiskModels(self) -> List[str]:
        """위험도 High: 계절성 우선"""
        return ['seasonal_naive', 'snaive_drift', 'mstl', 'theta', 'holt_winters']

    def _getMediumRiskModels(self) -> List[str]:
        """위험도 Medium: 균형"""
        return ['mstl', 'holt_winters', 'theta', 'auto_arima', 'auto_theta']

    def _getLowRiskModels(self) -> List[str]:
        """위험도 Low: 표준 Auto 모델"""
        return ['auto_arima', 'auto_ets', 'auto_theta', 'theta', 'mstl']

    def _prioritizeSeasonalModels(self, models: List[str]) -> List[str]:
        """계절성 있을 때 계절성 모델 우선"""
        seasonalModels = ['mstl', 'seasonal_naive', 'snaive_drift', 'holt_winters']
        otherModels = [m for m in models if m not in seasonalModels]
        prioritized = [m for m in seasonalModels if m in models] + otherModels
        return prioritized

    def getModelInfo(self, modelId: str) -> Dict[str, Any]:
        """모델 정보 반환"""
        return MODEL_INFO.get(modelId, {
            'name': modelId,
            'description': '',
            'flatResistance': 0.5,
            'bestFor': [],
            'minData': 10
        })

    def rankByFlatResistance(self, models: List[str]) -> List[str]:
        """일직선 저항력 순으로 정렬"""
        return sorted(
            models,
            key=lambda m: self.FLAT_RESISTANCE.get(m, 0.5),
            reverse=True
        )

    def getRecommendation(
        self,
        flatRisk: FlatRiskAssessment,
        characteristics: DataCharacteristics
    ) -> Dict[str, Any]:
        """
        모델 추천 상세 정보

        Returns
        -------
        Dict
            추천 모델 목록과 이유
        """
        models = self.selectModels(flatRisk, characteristics)

        recommendations = []
        for i, modelId in enumerate(models):
            info = self.getModelInfo(modelId)
            recommendations.append({
                'rank': i + 1,
                'modelId': modelId,
                'name': info.get('name', modelId),
                'flatResistance': self.FLAT_RESISTANCE.get(modelId, 0.5),
                'reason': self._getRecommendationReason(
                    modelId, flatRisk.riskLevel, characteristics
                )
            })

        return {
            'riskLevel': flatRisk.riskLevel.value,
            'strategy': flatRisk.recommendedStrategy,
            'models': recommendations,
            'warning': self._getStrategyWarning(flatRisk.riskLevel)
        }

    def _getRecommendationReason(
        self,
        modelId: str,
        riskLevel: RiskLevel,
        characteristics: DataCharacteristics
    ) -> str:
        """추천 이유 생성"""
        reasons = {
            'seasonal_naive': '계절 패턴을 강제로 반복하여 일직선 방지',
            'snaive_drift': '계절 패턴 + 추세 반영으로 안정적 예측',
            'mstl': '다중 계절성 분해로 복잡한 패턴 포착',
            'holt_winters': '수준/추세/계절성 분리 모델링',
            'theta': 'M3 Competition 우승 모델, 범용적 성능',
            'auto_arima': '자동 파라미터 튜닝으로 최적 ARIMA',
            'auto_ets': '30가지 조합 중 자동 최적 선택',
            'auto_theta': '자동 Theta 분해',
            'prophet': '이상치에 강건하고 휴일 효과 반영'
        }

        baseReason = reasons.get(modelId, '범용 모델')

        if riskLevel in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if self.FLAT_RESISTANCE.get(modelId, 0) >= 0.8:
                baseReason += ' (일직선 위험 대응)'

        return baseReason

    def _getStrategyWarning(self, riskLevel: RiskLevel) -> Optional[str]:
        """전략 경고 메시지"""
        if riskLevel == RiskLevel.CRITICAL:
            return '⚠️ 일직선 예측 위험이 매우 높습니다. 계절성 강제 모델만 사용됩니다.'
        elif riskLevel == RiskLevel.HIGH:
            return '⚠️ 일직선 예측 위험이 높습니다. 계절성 모델이 우선됩니다.'
        elif riskLevel == RiskLevel.MEDIUM:
            return '일직선 예측 가능성이 있습니다. 결과를 주의 깊게 확인하세요.'
        return None
