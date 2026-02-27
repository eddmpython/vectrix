"""
Level 3: 일직선 예측 감지

예측 수행 후 결과가 일직선인지 감지합니다.
"""

from typing import Optional

import numpy as np

from ..types import FlatPredictionInfo, FlatPredictionType


class FlatPredictionDetector:
    """
    일직선 예측 감지기

    예측 결과를 분석하여 일직선(수평, 대각선, 평균 수렴)인지 판단합니다.
    """

    def __init__(
        self,
        horizontalThreshold: float = 0.01,
        diagonalThreshold: float = 1e-8,
        varianceThreshold: float = 0.0001
    ):
        """
        Parameters
        ----------
        horizontalThreshold : float
            수평 일직선 판단 임계값 (예측 std / 원본 std)
        diagonalThreshold : float
            대각선 일직선 판단 임계값 (차분의 분산)
        varianceThreshold : float
            상대 분산 임계값
        """
        self.horizontalThreshold = horizontalThreshold
        self.diagonalThreshold = diagonalThreshold
        self.varianceThreshold = varianceThreshold

    def detect(
        self,
        predictions: np.ndarray,
        originalData: np.ndarray,
        originalStd: Optional[float] = None
    ) -> FlatPredictionInfo:
        """
        일직선 예측 감지

        Parameters
        ----------
        predictions : np.ndarray
            예측값
        originalData : np.ndarray
            원본 데이터
        originalStd : float, optional
            원본 데이터 표준편차 (없으면 계산)

        Returns
        -------
        FlatPredictionInfo
            감지 결과
        """
        if len(predictions) < 3:
            return FlatPredictionInfo(
                isFlat=False,
                flatType=FlatPredictionType.NONE,
                message='예측 길이가 너무 짧아 감지 불가'
            )

        if originalStd is None:
            originalStd = np.std(originalData)

        predStd = np.std(predictions)
        predVar = np.var(predictions)
        predMean = np.mean(np.abs(predictions))

        # 수평 일직선 감지
        if originalStd > 0:
            stdRatio = predStd / originalStd
            if stdRatio < self.horizontalThreshold:
                return FlatPredictionInfo(
                    isFlat=True,
                    flatType=FlatPredictionType.HORIZONTAL,
                    predictionStd=predStd,
                    originalStd=originalStd,
                    stdRatio=stdRatio,
                    message='수평 일직선 예측 감지: 모델이 계절성/변동을 감지하지 못함',
                    suggestion='Seasonal Naive 또는 MSTL 모델 사용 권장'
                )

        # 상대 분산으로 수평 일직선 감지
        if predMean > 0:
            relativeVar = predVar / (predMean ** 2)
            if relativeVar < self.varianceThreshold:
                return FlatPredictionInfo(
                    isFlat=True,
                    flatType=FlatPredictionType.HORIZONTAL,
                    predictionStd=predStd,
                    originalStd=originalStd,
                    varianceRatio=relativeVar,
                    message='수평 일직선 예측 감지: 예측값이 거의 변하지 않음',
                    suggestion='계절 패턴 강제 주입 권장'
                )

        # 대각선 일직선 감지 (일정한 기울기)
        diffs = np.diff(predictions)
        diffVar = np.var(diffs)

        if diffVar < self.diagonalThreshold and predVar > 1e-6:
            return FlatPredictionInfo(
                isFlat=True,
                flatType=FlatPredictionType.DIAGONAL,
                predictionStd=predStd,
                originalStd=originalStd,
                varianceRatio=diffVar,
                message='대각선 일직선 예측 감지: 추세만 반영되고 계절성 미감지',
                suggestion='계절 변동 추가 권장'
            )

        # 평균 수렴 감지 (장기 예측에서 변동 감소)
        if len(predictions) >= 10:
            firstHalfStd = np.std(predictions[:len(predictions)//2])
            secondHalfStd = np.std(predictions[len(predictions)//2:])

            if firstHalfStd > 0 and secondHalfStd / firstHalfStd < 0.3:
                return FlatPredictionInfo(
                    isFlat=True,
                    flatType=FlatPredictionType.MEAN_REVERSION,
                    predictionStd=predStd,
                    originalStd=originalStd,
                    stdRatio=secondHalfStd / firstHalfStd,
                    message='평균 수렴 예측 감지: 장기 예측에서 변동이 급감',
                    suggestion='예측 기간 단축 또는 불확실성 확대 권장'
                )

        return FlatPredictionInfo(
            isFlat=False,
            flatType=FlatPredictionType.NONE,
            predictionStd=predStd,
            originalStd=originalStd,
            stdRatio=predStd / originalStd if originalStd > 0 else 0
        )

    def detectMultiple(
        self,
        modelPredictions: dict,
        originalData: np.ndarray
    ) -> dict:
        """
        여러 모델의 예측 결과 일괄 감지

        Parameters
        ----------
        modelPredictions : dict
            {모델ID: 예측값} 딕셔너리
        originalData : np.ndarray
            원본 데이터

        Returns
        -------
        dict
            {모델ID: FlatPredictionInfo} 딕셔너리
        """
        originalStd = np.std(originalData)
        results = {}

        for modelId, predictions in modelPredictions.items():
            results[modelId] = self.detect(predictions, originalData, originalStd)

        return results

    def getFlatModels(self, detectionResults: dict) -> list:
        """일직선 예측을 생성한 모델 목록 반환"""
        return [
            modelId
            for modelId, info in detectionResults.items()
            if info.isFlat
        ]

    def getValidModels(self, detectionResults: dict) -> list:
        """유효한 예측을 생성한 모델 목록 반환"""
        return [
            modelId
            for modelId, info in detectionResults.items()
            if not info.isFlat
        ]
