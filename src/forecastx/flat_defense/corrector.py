"""
Level 3-4: 일직선 예측 보정

일직선 예측 감지시 지능적으로 보정합니다.
"""

import numpy as np
from typing import Tuple, Optional
from ..types import FlatPredictionInfo, FlatPredictionType


class FlatPredictionCorrector:
    """
    일직선 예측 보정기

    일직선 예측이 감지되면 원본 데이터의 패턴을 활용하여 보정합니다.
    단순히 노이즈를 추가하는 것이 아니라, 실제 패턴을 기반으로 보정합니다.
    """

    def __init__(
        self,
        seasonalStrength: float = 0.5,
        variationStrength: float = 0.3,
        maxCorrection: float = 0.5  # 원본 std 대비 최대 보정 비율
    ):
        """
        Parameters
        ----------
        seasonalStrength : float
            계절 패턴 주입 강도 (0.0 ~ 1.0)
        variationStrength : float
            변동 추가 강도 (0.0 ~ 1.0)
        maxCorrection : float
            최대 보정 비율 (원본 std 기준)
        """
        self.seasonalStrength = seasonalStrength
        self.variationStrength = variationStrength
        self.maxCorrection = maxCorrection

    def correct(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        flatInfo: FlatPredictionInfo,
        period: int = 7
    ) -> Tuple[np.ndarray, FlatPredictionInfo]:
        """
        일직선 예측 보정

        Parameters
        ----------
        predictions : np.ndarray
            원본 예측값
        originalData : np.ndarray
            원본 시계열 데이터
        flatInfo : FlatPredictionInfo
            일직선 감지 정보
        period : int
            계절 주기

        Returns
        -------
        Tuple[np.ndarray, FlatPredictionInfo]
            (보정된 예측값, 업데이트된 감지 정보)
        """
        if not flatInfo.isFlat:
            return predictions, flatInfo

        corrected = predictions.copy()
        correctionMethod = ""

        if flatInfo.flatType == FlatPredictionType.HORIZONTAL:
            corrected, correctionMethod = self._correctHorizontal(
                predictions, originalData, period
            )

        elif flatInfo.flatType == FlatPredictionType.DIAGONAL:
            corrected, correctionMethod = self._correctDiagonal(
                predictions, originalData, period
            )

        elif flatInfo.flatType == FlatPredictionType.MEAN_REVERSION:
            corrected, correctionMethod = self._correctMeanReversion(
                predictions, originalData, period
            )

        updatedInfo = FlatPredictionInfo(
            isFlat=flatInfo.isFlat,
            flatType=flatInfo.flatType,
            predictionStd=flatInfo.predictionStd,
            originalStd=flatInfo.originalStd,
            stdRatio=flatInfo.stdRatio,
            varianceRatio=flatInfo.varianceRatio,
            correctionApplied=True,
            correctionMethod=correctionMethod,
            correctionStrength=self.seasonalStrength,
            message=flatInfo.message,
            suggestion=f'보정 적용됨: {correctionMethod}'
        )

        return corrected, updatedInfo

    def _correctHorizontal(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        period: int
    ) -> Tuple[np.ndarray, str]:
        """
        수평 일직선 보정: 계절 패턴 주입

        원본 데이터에서 계절 패턴을 추출하여 예측에 주입합니다.
        """
        seasonal = self._extractSeasonalPattern(originalData, period)

        if seasonal is None:
            return self._addSimpleVariation(predictions, originalData), "simple_variation"

        # 예측 길이에 맞게 계절 패턴 반복
        nPred = len(predictions)
        seasonalExtended = np.tile(seasonal, nPred // len(seasonal) + 1)[:nPred]

        # 강도 조절하여 주입
        originalStd = np.std(originalData)
        maxAdjustment = originalStd * self.maxCorrection

        # 계절 패턴 스케일링
        seasonalAdjustment = seasonalExtended * self.seasonalStrength
        seasonalAdjustment = np.clip(seasonalAdjustment, -maxAdjustment, maxAdjustment)

        corrected = predictions + seasonalAdjustment

        return corrected, "seasonal_injection"

    def _correctDiagonal(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        period: int
    ) -> Tuple[np.ndarray, str]:
        """
        대각선 일직선 보정: 계절 변동 추가

        추세는 유지하면서 계절적 변동을 추가합니다.
        """
        seasonal = self._extractSeasonalPattern(originalData, period)

        if seasonal is None:
            return self._addSimpleVariation(predictions, originalData), "simple_variation"

        # 예측의 추세 보존
        nPred = len(predictions)
        trend = np.linspace(predictions[0], predictions[-1], nPred)

        # 계절 패턴 추가
        seasonalExtended = np.tile(seasonal, nPred // len(seasonal) + 1)[:nPred]

        originalStd = np.std(originalData)
        seasonalAdjustment = seasonalExtended * self.variationStrength * originalStd / (np.std(seasonal) + 1e-10)

        corrected = trend + seasonalAdjustment

        return corrected, "trend_plus_seasonal"

    def _correctMeanReversion(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        period: int
    ) -> Tuple[np.ndarray, str]:
        """
        평균 수렴 보정: 변동성 유지

        장기 예측에서도 변동성이 유지되도록 보정합니다.
        """
        nPred = len(predictions)
        originalStd = np.std(originalData[-min(30, len(originalData)):])

        # 현재 예측의 변동성 계산
        predStd = np.std(predictions)

        if predStd < 1e-10:
            # 완전히 평평하면 계절 패턴 주입
            return self._correctHorizontal(predictions, originalData, period)

        # 타겟 변동성 (원본의 일정 비율 유지)
        targetStd = originalStd * 0.7

        # 변동성 스케일링
        predMean = np.mean(predictions)
        scaleFactor = targetStd / predStd

        corrected = predMean + (predictions - predMean) * scaleFactor

        return corrected, "variance_scaling"

    def _extractSeasonalPattern(
        self,
        data: np.ndarray,
        period: int
    ) -> Optional[np.ndarray]:
        """
        데이터에서 계절 패턴 추출 (간단한 평균 기반)
        """
        n = len(data)

        if n < period:
            return None

        # 최근 데이터에서 계절 패턴 추출
        recentData = data[-min(period * 3, n):]
        nRecent = len(recentData)

        # 주기별 평균 계산
        seasonal = np.zeros(period)
        counts = np.zeros(period)

        for i in range(nRecent):
            idx = i % period
            seasonal[idx] += recentData[i]
            counts[idx] += 1

        counts[counts == 0] = 1  # 0으로 나누기 방지
        seasonal = seasonal / counts

        # 평균 제거 (계절 성분만)
        seasonal = seasonal - np.mean(seasonal)

        return seasonal

    def _addSimpleVariation(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray
    ) -> np.ndarray:
        """
        단순 변동 추가 (계절 패턴 추출 실패시)

        완전히 랜덤이 아니라, 최근 데이터의 변동 패턴을 모방합니다.
        """
        n = len(predictions)
        originalStd = np.std(originalData[-min(30, len(originalData)):])

        # 최근 변동 패턴
        recentDiffs = np.diff(originalData[-min(n + 10, len(originalData)):])

        if len(recentDiffs) < n:
            # 패턴 반복
            recentDiffs = np.tile(recentDiffs, n // len(recentDiffs) + 1)[:n]
        else:
            recentDiffs = recentDiffs[:n]

        # 변동 스케일 조정
        variation = recentDiffs * self.variationStrength
        maxVar = originalStd * self.maxCorrection
        variation = np.clip(variation, -maxVar, maxVar)

        corrected = predictions.copy()
        for i in range(1, n):
            corrected[i] = corrected[i - 1] + variation[i - 1]

        # 레벨 조정 (원래 예측의 평균 유지)
        corrected = corrected - np.mean(corrected) + np.mean(predictions)

        return corrected


def correctWithConfidenceInterval(
    predictions: np.ndarray,
    lower95: np.ndarray,
    upper95: np.ndarray,
    flatInfo: FlatPredictionInfo,
    originalData: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    신뢰구간도 함께 보정

    일직선 예측이면 신뢰구간을 확대합니다.
    """
    if not flatInfo.isFlat:
        return predictions, lower95, upper95

    corrector = FlatPredictionCorrector()
    correctedPred, _ = corrector.correct(
        predictions, originalData, flatInfo
    )

    # 신뢰구간 확대
    originalStd = np.std(originalData)
    steps = np.arange(1, len(predictions) + 1)

    # 일직선일수록 불확실성 증가
    uncertaintyMultiplier = 1.5 if flatInfo.flatType != FlatPredictionType.NONE else 1.0

    margin = 1.96 * originalStd * np.sqrt(steps) * uncertaintyMultiplier

    correctedLower = correctedPred - margin
    correctedUpper = correctedPred + margin

    return correctedPred, correctedLower, correctedUpper
