"""
Level 4: 변동성 보존 앙상블

일반 앙상블은 평균을 내면서 변동성이 줄어드는 문제가 있습니다.
이 앙상블은 변동성을 보존하면서 정확도도 유지합니다.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from ...types import ModelResult, FlatPredictionInfo, FlatPredictionType


class VariabilityPreservingEnsemble:
    """
    변동성 보존 앙상블

    핵심 아이디어:
    1. 정확도 기반 가중치 + 변동성 보존 가중치 결합
    2. 일직선 예측 모델은 가중치 감소
    3. 앙상블 후 변동성이 과도하게 줄면 스케일링
    """

    def __init__(
        self,
        variabilityWeight: float = 0.3,
        minVariabilityRatio: float = 0.5,
        excludeFlatModels: bool = True
    ):
        """
        Parameters
        ----------
        variabilityWeight : float
            변동성 보존 가중치 비중 (0.0 ~ 1.0)
        minVariabilityRatio : float
            최소 변동성 비율 (원본 대비)
        excludeFlatModels : bool
            일직선 모델 제외 여부
        """
        self.variabilityWeight = variabilityWeight
        self.minVariabilityRatio = minVariabilityRatio
        self.excludeFlatModels = excludeFlatModels

    def ensemble(
        self,
        modelResults: Dict[str, ModelResult],
        originalData: np.ndarray,
        topK: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        변동성 보존 앙상블 수행

        Parameters
        ----------
        modelResults : Dict[str, ModelResult]
            모델별 결과
        originalData : np.ndarray
            원본 데이터
        topK : int
            상위 K개 모델만 사용

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]
            (예측값, lower95, upper95, 메타데이터)
        """
        originalStd = np.std(originalData[-min(30, len(originalData)):])

        # 유효한 모델 필터링
        validResults = self._filterValidModels(modelResults)

        if not validResults:
            return self._fallbackPrediction(originalData, modelResults)

        # MAPE 기준 정렬
        sortedModels = sorted(
            validResults.items(),
            key=lambda x: x[1].mape
        )[:topK]

        if not sortedModels:
            return self._fallbackPrediction(originalData, modelResults)

        # 가중치 계산
        weights = self._calculateWeights(sortedModels, originalStd)

        # 가중 앙상블
        predictions = self._weightedAverage(sortedModels, weights)

        # 변동성 보정
        predictions = self._correctVariability(predictions, originalStd)

        # 신뢰구간 계산
        lower95, upper95 = self._calculateConfidenceInterval(
            sortedModels, predictions, originalStd
        )

        metadata = {
            'modelsUsed': [m[0] for m in sortedModels],
            'weights': {m[0]: w for m, w in zip(sortedModels, weights)},
            'originalStd': originalStd,
            'ensembleStd': np.std(predictions),
            'variabilityRatio': np.std(predictions) / originalStd if originalStd > 0 else 0
        }

        return predictions, lower95, upper95, metadata

    def _filterValidModels(
        self,
        modelResults: Dict[str, ModelResult]
    ) -> Dict[str, ModelResult]:
        """유효한 모델 필터링"""
        valid = {}

        for modelId, result in modelResults.items():
            if modelId == 'ensemble':
                continue

            if not result.isValid:
                continue

            if self.excludeFlatModels and result.flatInfo:
                if result.flatInfo.isFlat:
                    continue

            if result.mape == float('inf') or np.isnan(result.mape):
                continue

            if len(result.predictions) == 0:
                continue

            valid[modelId] = result

        return valid

    def _calculateWeights(
        self,
        sortedModels: List[Tuple[str, ModelResult]],
        originalStd: float
    ) -> np.ndarray:
        """
        가중치 계산: 정확도 × 변동성 보존

        Parameters
        ----------
        sortedModels : List[Tuple[str, ModelResult]]
            (모델ID, 결과) 튜플 리스트
        originalStd : float
            원본 데이터 표준편차

        Returns
        -------
        np.ndarray
            정규화된 가중치
        """
        n = len(sortedModels)
        accuracyWeights = np.zeros(n)
        variabilityWeights = np.zeros(n)

        for i, (modelId, result) in enumerate(sortedModels):
            # 정확도 가중치 (MAPE 역수)
            accuracyWeights[i] = 1.0 / (result.mape + 1e-6)

            # 변동성 보존 가중치
            predStd = np.std(result.predictions)
            if originalStd > 0:
                varRatio = predStd / originalStd
                # 원본과 비슷한 변동성일수록 높은 점수
                variabilityWeights[i] = 1.0 / (1.0 + abs(varRatio - 1.0))
            else:
                variabilityWeights[i] = 1.0

        # 정규화
        if accuracyWeights.sum() > 0:
            accuracyWeights /= accuracyWeights.sum()
        if variabilityWeights.sum() > 0:
            variabilityWeights /= variabilityWeights.sum()

        # 결합
        alpha = self.variabilityWeight
        finalWeights = (1 - alpha) * accuracyWeights + alpha * variabilityWeights

        # 다시 정규화
        if finalWeights.sum() > 0:
            finalWeights /= finalWeights.sum()
        else:
            finalWeights = np.ones(n) / n

        return finalWeights

    def _weightedAverage(
        self,
        sortedModels: List[Tuple[str, ModelResult]],
        weights: np.ndarray
    ) -> np.ndarray:
        """가중 평균 계산"""
        # 예측 길이 통일
        predLength = min(len(m[1].predictions) for m in sortedModels)

        predictions = np.zeros(predLength)
        for i, (modelId, result) in enumerate(sortedModels):
            predictions += weights[i] * result.predictions[:predLength]

        return predictions

    def _correctVariability(
        self,
        predictions: np.ndarray,
        originalStd: float
    ) -> np.ndarray:
        """
        변동성 보정

        앙상블 후 변동성이 너무 줄었으면 스케일링
        """
        ensembleStd = np.std(predictions)

        if originalStd <= 0 or ensembleStd <= 0:
            return predictions

        varRatio = ensembleStd / originalStd

        # 변동성이 최소 비율 이하면 스케일업
        if varRatio < self.minVariabilityRatio:
            targetStd = originalStd * self.minVariabilityRatio * 0.8
            scaleFactor = targetStd / ensembleStd

            predMean = np.mean(predictions)
            predictions = predMean + (predictions - predMean) * scaleFactor

        return predictions

    def _calculateConfidenceInterval(
        self,
        sortedModels: List[Tuple[str, ModelResult]],
        predictions: np.ndarray,
        originalStd: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """신뢰구간 계산"""
        nPred = len(predictions)

        # 모델 간 불확실성
        modelPreds = [m[1].predictions[:nPred] for m in sortedModels]
        modelStd = np.std(modelPreds, axis=0)

        # 시간에 따른 불확실성 증가
        steps = np.arange(1, nPred + 1)
        timeUncertainty = originalStd * np.sqrt(steps) * 0.5

        # 결합
        totalUncertainty = np.sqrt(modelStd ** 2 + timeUncertainty ** 2)

        margin = 1.96 * totalUncertainty

        lower95 = predictions - margin
        upper95 = predictions + margin

        return lower95, upper95

    def _fallbackPrediction(
        self,
        originalData: np.ndarray,
        modelResults: Dict[str, ModelResult]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """폴백 예측 (유효한 모델 없을 때)"""
        # 아무 모델이나 하나 선택
        if modelResults:
            firstResult = next(iter(modelResults.values()))
            predictions = firstResult.predictions
        else:
            # 최후의 수단: Naive
            lastVal = originalData[-1]
            predictions = np.full(30, lastVal)

        originalStd = np.std(originalData[-30:])
        steps = np.arange(1, len(predictions) + 1)
        margin = 1.96 * originalStd * np.sqrt(steps)

        return (
            predictions,
            predictions - margin,
            predictions + margin,
            {'warning': '유효한 모델이 없어 폴백 예측 사용'}
        )


def quickEnsemble(
    modelPredictions: Dict[str, np.ndarray],
    modelMapes: Dict[str, float],
    originalStd: float
) -> np.ndarray:
    """
    간단한 앙상블 (결과 객체 없이)

    Parameters
    ----------
    modelPredictions : Dict[str, np.ndarray]
        모델별 예측값
    modelMapes : Dict[str, float]
        모델별 MAPE
    originalStd : float
        원본 데이터 표준편차

    Returns
    -------
    np.ndarray
        앙상블 예측값
    """
    if not modelPredictions:
        return np.array([])

    # 예측 길이 통일
    predLength = min(len(p) for p in modelPredictions.values())

    # MAPE 기반 가중치
    weights = {}
    totalWeight = 0

    for modelId, pred in modelPredictions.items():
        mape = modelMapes.get(modelId, 100)
        w = 1.0 / (mape + 1e-6)

        # 변동성 보너스
        predStd = np.std(pred)
        if originalStd > 0:
            varRatio = predStd / originalStd
            varBonus = 1.0 / (1.0 + abs(varRatio - 1.0))
            w *= (0.7 + 0.3 * varBonus)

        weights[modelId] = w
        totalWeight += w

    # 정규화
    for modelId in weights:
        weights[modelId] /= totalWeight

    # 앙상블
    ensemble = np.zeros(predLength)
    for modelId, pred in modelPredictions.items():
        ensemble += weights[modelId] * pred[:predLength]

    return ensemble
