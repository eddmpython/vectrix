"""
ChaniCast 핵심 데이터 타입 정의
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class FlatPredictionType(Enum):
    """일직선 예측 유형"""
    NONE = "none"
    HORIZONTAL = "horizontal"      # 수평 일직선: ────────
    DIAGONAL = "diagonal"          # 대각선 일직선: ╱╱╱╱╱╱
    MEAN_REVERSION = "mean_reversion"  # 평균 수렴: ∿→───


class RiskLevel(Enum):
    """위험도 수준"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Frequency(Enum):
    """데이터 주기"""
    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "M"
    QUARTERLY = "Q"
    YEARLY = "Y"
    HOURLY = "H"
    UNKNOWN = "unknown"


@dataclass
class DataCharacteristics:
    """데이터 특성 분석 결과"""

    # 기본 정보
    length: int = 0
    frequency: Frequency = Frequency.UNKNOWN
    period: int = 1
    dateRange: Tuple[str, str] = ("", "")

    # 추세 분석
    hasTrend: bool = False
    trendDirection: str = "none"  # "up", "down", "none"
    trendStrength: float = 0.0    # 0.0 ~ 1.0

    # 계절성 분석
    hasSeasonality: bool = False
    seasonalStrength: float = 0.0
    seasonalPeriods: List[int] = field(default_factory=list)
    hasMultipleSeasonality: bool = False

    # 정상성
    isStationary: bool = False

    # 변동성
    volatility: float = 0.0
    volatilityLevel: str = "normal"  # "low", "normal", "high"

    # 품질
    missingRatio: float = 0.0
    outlierCount: int = 0
    outlierRatio: float = 0.0

    # 예측 가능성
    predictabilityScore: float = 0.0  # 0 ~ 100


@dataclass
class FlatRiskAssessment:
    """일직선 예측 위험도 평가 결과"""

    # 종합 위험도
    riskScore: float = 0.0  # 0.0 ~ 1.0
    riskLevel: RiskLevel = RiskLevel.LOW

    # 개별 위험 요소
    riskFactors: Dict[str, bool] = field(default_factory=dict)
    # - lowVariance: 변동성 부족
    # - weakSeasonality: 계절성 약함
    # - noTrend: 추세 없음
    # - shortData: 데이터 부족
    # - highNoise: 노이즈 과다

    # 권장 전략
    recommendedStrategy: str = "standard"
    recommendedModels: List[str] = field(default_factory=list)

    # 경고 메시지
    warnings: List[str] = field(default_factory=list)


@dataclass
class FlatPredictionInfo:
    """일직선 예측 감지 정보"""

    isFlat: bool = False
    flatType: FlatPredictionType = FlatPredictionType.NONE

    # 감지 지표
    predictionStd: float = 0.0
    originalStd: float = 0.0
    stdRatio: float = 0.0
    varianceRatio: float = 0.0

    # 보정 정보
    correctionApplied: bool = False
    correctionMethod: str = ""
    correctionStrength: float = 0.0

    # 메시지
    message: str = ""
    suggestion: str = ""


@dataclass
class ModelResult:
    """개별 모델 예측 결과"""

    modelId: str = ""
    modelName: str = ""

    # 예측값
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    upper95: np.ndarray = field(default_factory=lambda: np.array([]))

    # 평가 지표
    mape: float = float('inf')
    rmse: float = float('inf')
    mae: float = float('inf')
    smape: float = float('inf')

    # 일직선 정보
    flatInfo: Optional[FlatPredictionInfo] = None

    # 메타데이터
    trainingTime: float = 0.0
    isValid: bool = True


@dataclass
class ForecastResult:
    """최종 예측 결과"""

    success: bool = False

    # 예측값
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    dates: List[str] = field(default_factory=list)
    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    upper95: np.ndarray = field(default_factory=lambda: np.array([]))

    # 선택된 모델
    bestModelId: str = ""
    bestModelName: str = ""

    # 모든 모델 결과
    allModelResults: Dict[str, ModelResult] = field(default_factory=dict)

    # 데이터 특성
    characteristics: Optional[DataCharacteristics] = None

    # 일직선 위험도
    flatRisk: Optional[FlatRiskAssessment] = None

    # 일직선 감지/보정 정보
    flatInfo: Optional[FlatPredictionInfo] = None

    # 해석
    interpretation: Dict[str, Any] = field(default_factory=dict)

    # 경고
    warnings: List[str] = field(default_factory=list)

    # 에러
    error: Optional[str] = None

    def hasWarning(self) -> bool:
        """경고가 있는지 확인"""
        return len(self.warnings) > 0 or (self.flatInfo and self.flatInfo.isFlat)

    def getSummary(self) -> Dict[str, Any]:
        """결과 요약 반환"""
        return {
            'success': self.success,
            'bestModel': self.bestModelName,
            'mape': self.allModelResults.get(self.bestModelId, ModelResult()).mape,
            'predictionLength': len(self.predictions),
            'hasWarning': self.hasWarning(),
            'flatDetected': self.flatInfo.isFlat if self.flatInfo else False,
            'warningCount': len(self.warnings)
        }


# 모델 정보 상수
MODEL_INFO = {
    'seasonal_naive': {
        'name': 'Seasonal Naive',
        'description': '지난 시즌 같은 시점 값 사용. 계절 패턴 무조건 반복.',
        'flatResistance': 0.95,
        'bestFor': ['강한 계절성', '일직선 위험 높을 때'],
        'minData': 14
    },
    'snaive_drift': {
        'name': 'Seasonal Naive + Drift',
        'description': '계절 패턴 반복 + 추세 반영',
        'flatResistance': 0.90,
        'bestFor': ['계절성 + 추세', '일직선 위험 높을 때'],
        'minData': 14
    },
    'mstl': {
        'name': 'MSTL',
        'description': '다중 계절성 분해 (LOESS) + ARIMA',
        'flatResistance': 0.85,
        'bestFor': ['다중 계절성', '복잡한 패턴'],
        'minData': 50
    },
    'holt_winters': {
        'name': 'Holt-Winters',
        'description': '삼중 지수평활 (수준 + 추세 + 계절)',
        'flatResistance': 0.80,
        'bestFor': ['계절성 데이터', '중기 예측'],
        'minData': 24
    },
    'theta': {
        'name': 'Theta',
        'description': 'M3 Competition 우승 모델. Theta 분해.',
        'flatResistance': 0.75,
        'bestFor': ['범용', '빠른 예측'],
        'minData': 10
    },
    'auto_arima': {
        'name': 'AutoARIMA',
        'description': '자동 ARIMA. AICc 기준 최적 파라미터.',
        'flatResistance': 0.60,
        'bestFor': ['정상성 데이터', '추세 예측'],
        'minData': 30
    },
    'auto_ets': {
        'name': 'AutoETS',
        'description': '자동 지수평활. 30가지 조합 자동 선택.',
        'flatResistance': 0.55,
        'bestFor': ['안정적 패턴', '단기 예측'],
        'minData': 20
    },
    'ensemble': {
        'name': 'Variability-Preserving Ensemble',
        'description': '변동성 보존 앙상블. 상위 모델 결합.',
        'flatResistance': 0.85,
        'bestFor': ['불확실한 패턴', '안정적 예측'],
        'minData': 30
    },
    'naive': {
        'name': 'Naive',
        'description': '마지막 관측값을 반복. 가장 단순한 벤치마크.',
        'flatResistance': 0.10,
        'bestFor': ['벤치마크', 'Random Walk 데이터'],
        'minData': 2
    },
    'mean': {
        'name': 'Mean',
        'description': '과거 평균값으로 예측. 정상 시계열 벤치마크.',
        'flatResistance': 0.05,
        'bestFor': ['벤치마크', '정상 시계열'],
        'minData': 2
    },
    'rwd': {
        'name': 'Random Walk with Drift',
        'description': '마지막 값 + 평균 추세. 추세 있는 벤치마크.',
        'flatResistance': 0.60,
        'bestFor': ['추세 데이터', '벤치마크'],
        'minData': 5
    },
    'window_avg': {
        'name': 'Window Average',
        'description': '최근 윈도우 평균으로 예측.',
        'flatResistance': 0.15,
        'bestFor': ['벤치마크', '안정적 데이터'],
        'minData': 5
    },
    'auto_ces': {
        'name': 'AutoCES',
        'description': '복소수 지수평활법. N/S/P/F 자동 선택.',
        'flatResistance': 0.65,
        'bestFor': ['비선형 패턴', '복잡한 계절성'],
        'minData': 20
    },
    'croston': {
        'name': 'Croston (Auto)',
        'description': '간헐적 수요 예측. Classic/SBA/TSB 자동 선택.',
        'flatResistance': 0.30,
        'bestFor': ['간헐적 수요', '0이 많은 시계열'],
        'minData': 10
    },
    'dot': {
        'name': 'Dynamic Optimized Theta',
        'description': 'Theta+alpha+drift 동시 L-BFGS-B 최적화.',
        'flatResistance': 0.80,
        'bestFor': ['추세 데이터', '범용'],
        'minData': 10
    },
    'tbats': {
        'name': 'TBATS',
        'description': 'Trigonometric Seasonal, Box-Cox, ARMA, Trend. 복잡한 다중 계절성.',
        'flatResistance': 0.85,
        'bestFor': ['다중 계절성', '시간별 데이터', '복잡한 패턴'],
        'minData': 30
    },
    'garch': {
        'name': 'GARCH(1,1)',
        'description': '조건부 분산 모델. 금융 변동성 예측.',
        'flatResistance': 0.50,
        'bestFor': ['금융 데이터', '변동성 예측', '수익률'],
        'minData': 50
    },
    'egarch': {
        'name': 'EGARCH',
        'description': '비대칭 변동성 모델. 레버리지 효과.',
        'flatResistance': 0.50,
        'bestFor': ['금융 데이터', '비대칭 변동성'],
        'minData': 50
    },
    'gjr_garch': {
        'name': 'GJR-GARCH',
        'description': '임계 비대칭 GARCH. 음의 충격 반응.',
        'flatResistance': 0.50,
        'bestFor': ['금융 데이터', '비대칭 변동성'],
        'minData': 50
    }
}
