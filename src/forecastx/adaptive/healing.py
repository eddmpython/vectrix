"""
Self-Healing Forecast

예측이 생성된 후 실제 데이터가 도착하면, 예측 오차를 실시간으로
모니터링하고 자동으로 교정하는 '살아있는 예측' 시스템.

핵심 알고리즘:
- CUSUM (Cumulative Sum): 체계적 편향(bias) 감지
- EWMA (Exponentially Weighted Moving Average): 최근 오차 추세 추적
- Adaptive Conformal Correction: 분포 가정 없는 예측 교정
- 온라인 잔차 학습 + 예측 업데이트

Usage:
    >>> from forecastx.adaptive.healing import SelfHealingForecast
    >>> healer = SelfHealingForecast(predictions, lower95, upper95, historicalData)
    >>> healer.observe(actual_values)
    >>> updated = healer.getUpdatedForecast()
    >>> status = healer.getStatus()
    >>> report = healer.getReport()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class HealingStatus:
    """
    Self-Healing 예측의 현재 상태 스냅샷

    observe() 호출 시마다 반환되며, 예측의 건강 상태,
    드리프트 감지 결과, 교정 적용 여부 등을 포함한다.
    """
    health: str                         # 'healthy', 'degrading', 'critical', 'healed'
    healthScore: float                  # 0-100
    observedCount: int
    remainingSteps: int
    driftDetected: bool
    driftDirection: Optional[str]       # 'upward_bias', 'downward_bias', None
    driftMagnitude: float
    correctionApplied: bool
    biasEstimate: float                 # 추정 편향
    mape: float                         # 현재까지의 MAPE
    mae: float                          # 현재까지의 MAE
    refitRecommended: bool
    refitReason: Optional[str]
    message: str                        # 사람이 읽을 수 있는 상태 메시지


@dataclass
class HealingReport:
    """
    Self-Healing 예측의 전체 치유 과정 보고서

    getReport() 호출 시 반환되며, 전체 치유 과정의 요약,
    교정 전후 MAPE 비교, 드리프트 이벤트 목록 등을 포함한다.
    """
    overallHealth: str
    healthScore: float
    totalObserved: int
    totalCorrected: int
    originalMape: float                 # 교정 전 MAPE
    healedMape: float                   # 교정 후 MAPE
    improvementPct: float               # 개선 비율
    corrections: List[Dict] = field(default_factory=list)
    healingLog: List[str] = field(default_factory=list)
    driftEvents: List[Dict] = field(default_factory=list)
    updatedPredictions: np.ndarray = field(default_factory=lambda: np.array([]))
    updatedLower: np.ndarray = field(default_factory=lambda: np.array([]))
    updatedUpper: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# SelfHealingForecast
# ---------------------------------------------------------------------------

class SelfHealingForecast:
    """
    자가 치유 예측 시스템

    예측이 생성된 후 실제 데이터가 순차적으로 도착하면,
    예측 오차를 실시간으로 모니터링하고 자동으로 교정한다.

    기능:
    1. 실시간 예측 오차 모니터링
    2. CUSUM + EWMA 기반 드리프트 감지
    3. Adaptive Conformal Prediction 기반 자동 교정
    4. 온라인 잔차 학습 + 예측 업데이트
    5. 자동 재학습 트리거 판단

    Parameters
    ----------
    predictions : np.ndarray
        원래 예측값 (길이 H)
    lower95 : np.ndarray
        원래 95% 하한 (길이 H)
    upper95 : np.ndarray
        원래 95% 상한 (길이 H)
    historicalData : np.ndarray
        학습에 사용된 과거 데이터
    period : int
        계절 주기 (기본값 7)
    healingMode : str
        교정 강도. 'conservative', 'adaptive', 'aggressive' 중 택 1

    Usage:
        >>> healer = SelfHealingForecast(original_forecast, lower, upper, data)
        >>> healer.observe(actual_values)      # 실제 데이터 도착
        >>> updated = healer.getUpdatedForecast()  # 교정된 예측
        >>> healer.getStatus()                 # 드리프트 감지, 경보, 건강 상태
    """

    _VALID_MODES = ('conservative', 'adaptive', 'aggressive')

    def __init__(
        self,
        predictions: np.ndarray,
        lower95: np.ndarray,
        upper95: np.ndarray,
        historicalData: np.ndarray,
        period: int = 7,
        healingMode: str = 'adaptive',
    ):
        # ------------------------------------------------------------------
        # 입력 검증
        # ------------------------------------------------------------------
        predictions = np.asarray(predictions, dtype=np.float64)
        lower95 = np.asarray(lower95, dtype=np.float64)
        upper95 = np.asarray(upper95, dtype=np.float64)
        historicalData = np.asarray(historicalData, dtype=np.float64)

        if predictions.ndim != 1 or lower95.ndim != 1 or upper95.ndim != 1:
            raise ValueError("predictions, lower95, upper95는 1차원 배열이어야 합니다.")
        if len(predictions) == 0:
            raise ValueError("predictions가 비어있습니다.")
        if len(predictions) != len(lower95) or len(predictions) != len(upper95):
            raise ValueError(
                "predictions, lower95, upper95의 길이가 같아야 합니다. "
                f"받은 길이: {len(predictions)}, {len(lower95)}, {len(upper95)}"
            )
        if historicalData.ndim != 1 or len(historicalData) < 2:
            raise ValueError("historicalData는 길이 2 이상의 1차원 배열이어야 합니다.")
        if period < 1:
            raise ValueError(f"period는 1 이상이어야 합니다. 받은 값: {period}")
        if healingMode not in self._VALID_MODES:
            raise ValueError(
                f"healingMode는 {self._VALID_MODES} 중 하나여야 합니다. "
                f"받은 값: '{healingMode}'"
            )

        # ------------------------------------------------------------------
        # 원래 예측 저장 (불변)
        # ------------------------------------------------------------------
        self.originalPredictions = predictions.copy()
        self.currentPredictions = predictions.copy()
        self.originalLower = lower95.copy()
        self.originalUpper = upper95.copy()
        self.currentLower = lower95.copy()
        self.currentUpper = upper95.copy()
        self.historicalData = historicalData.copy()
        self.period = period
        self.healingMode = healingMode
        self.totalSteps = len(predictions)

        # ------------------------------------------------------------------
        # 참조 통계량 (과거 데이터 기반)
        # ------------------------------------------------------------------
        self._referenceStd = float(np.std(historicalData))
        if self._referenceStd < 1e-10:
            # 데이터가 거의 상수인 경우 fallback
            self._referenceStd = float(np.mean(np.abs(historicalData))) * 0.01 + 1e-6
        self._referenceMean = float(np.mean(historicalData))

        # 초기 예측 구간 평균 너비 (참조용)
        self._originalMeanWidth = float(np.mean(upper95 - lower95))

        # ------------------------------------------------------------------
        # 관측 데이터 추적
        # ------------------------------------------------------------------
        self.observedValues: List[float] = []
        self.observedCount: int = 0

        # ------------------------------------------------------------------
        # 오차 추적
        # ------------------------------------------------------------------
        self.errors: List[float] = []            # actual - prediction (부호 있음)
        self.absErrors: List[float] = []         # |actual - prediction|
        self.signedBias: List[float] = []        # 누적 편향 추적용

        # ------------------------------------------------------------------
        # 드리프트 감지 (CUSUM + EWMA)
        # ------------------------------------------------------------------
        self.cusumPos: float = 0.0
        self.cusumNeg: float = 0.0
        self.ewmaError: float = 0.0
        self.ewmaLambda: float = 0.2
        self.driftDetected: bool = False
        self.driftDirection: Optional[str] = None
        self.driftMagnitude: float = 0.0
        self._driftEvents: List[Dict] = []

        # ------------------------------------------------------------------
        # 교정 이력
        # ------------------------------------------------------------------
        self.corrections: List[Dict] = []
        self.healingLog: List[str] = []

        # ------------------------------------------------------------------
        # 건강 상태
        # ------------------------------------------------------------------
        self.health: str = 'healthy'
        self.healthScore: float = 100.0
        self.refitRecommended: bool = False
        self.refitReason: Optional[str] = None
        self.refitRecommendedAt: Optional[int] = None

        # 초기 로그
        self.healingLog.append(
            f"[init] SelfHealingForecast 생성: "
            f"steps={self.totalSteps}, mode={healingMode}, "
            f"refStd={self._referenceStd:.4f}"
        )

    # ======================================================================
    # 공개 API
    # ======================================================================

    def observe(self, actuals: np.ndarray) -> HealingStatus:
        """
        실제 데이터를 관측하고 예측을 자동 업데이트

        새로 도착한 실제 관측값을 1개 또는 여러 개 입력하면,
        내부적으로 오차 분석 -> 드리프트 감지 -> 교정을 수행한다.

        Parameters
        ----------
        actuals : np.ndarray
            새로 도착한 실제 관측값. 스칼라 또는 1차원 배열.

        Returns
        -------
        HealingStatus
            현재 치유 상태 스냅샷

        Raises
        ------
        ValueError
            관측값이 예측 범위를 초과하는 경우
        """
        actuals = np.atleast_1d(np.asarray(actuals, dtype=np.float64)).ravel()

        if len(actuals) == 0:
            return self.getStatus()

        remaining = self.totalSteps - self.observedCount
        if len(actuals) > remaining:
            raise ValueError(
                f"관측값 {len(actuals)}개가 남은 예측 스텝 {remaining}개를 초과합니다."
            )

        correctionApplied = False

        for actual in actuals:
            idx = self.observedCount
            predicted = self.originalPredictions[idx]

            # 1. 오차 계산 (actual - predicted: 양수 = 과소예측)
            error = float(actual - predicted)
            absError = abs(error)

            self.observedValues.append(float(actual))
            self.errors.append(error)
            self.absErrors.append(absError)

            # 누적 편향 추적
            cumBias = float(np.mean(self.errors))
            self.signedBias.append(cumBias)

            # 2. CUSUM 업데이트
            self._updateCUSUM(error)

            # 3. EWMA 업데이트
            self._updateEWMA(error)

            self.observedCount += 1

            # 4. 건강 상태 평가
            self._evaluateHealth()

            # 5. 로그 기록
            self.healingLog.append(
                f"[step {idx}] actual={actual:.4f}, pred={predicted:.4f}, "
                f"error={error:.4f}, health={self.health}({self.healthScore:.1f})"
            )

        # 6. 교정 적용 (최소 2개 관측 이후)
        if self.observedCount >= 2:
            self._applyCorrection()
            correctionApplied = True

        # 7. 재학습 권장 여부 판단
        self._evaluateRefitNeed()

        return self._buildStatus(correctionApplied)

    def getUpdatedForecast(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        교정된 예측값, 하한, 상한 반환

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (predictions, lower95, upper95) - 각각 길이 H 배열.
            관측된 스텝은 실제값으로, 나머지는 교정된 예측으로 채워진다.
        """
        preds = self.currentPredictions.copy()
        lower = self.currentLower.copy()
        upper = self.currentUpper.copy()

        # 이미 관측된 스텝은 실제값으로 덮어쓰기
        for i, val in enumerate(self.observedValues):
            preds[i] = val
            lower[i] = val
            upper[i] = val

        return preds, lower, upper

    def getStatus(self) -> HealingStatus:
        """
        현재 상태 스냅샷 반환

        Returns
        -------
        HealingStatus
            현재 치유 상태
        """
        return self._buildStatus(correctionApplied=len(self.corrections) > 0)

    def getReport(self) -> HealingReport:
        """
        전체 치유 과정 보고서 생성

        Returns
        -------
        HealingReport
            치유 과정 요약, 교정 전후 비교, 드리프트 이벤트 등
        """
        # 교정 전 MAPE (원래 예측 vs 실제)
        originalMape = self._computeMape(
            self.originalPredictions[:self.observedCount],
            np.array(self.observedValues)
        )

        # 교정 후 MAPE (교정된 예측 vs 실제)
        # 단, 교정은 '이후' 스텝에 적용되므로 직접적 비교는 제한적.
        # 여기서는 현재까지의 교정 효과를 estimated로 계산
        healedMape = originalMape  # 기본값

        if len(self.corrections) > 0 and self.observedCount > 0:
            # 각 교정 스텝에서 corrected prediction과 actual 비교
            healedErrors = []
            for i, actual in enumerate(self.observedValues):
                # i번째 관측 직전의 currentPrediction을 사용
                # (교정이 이전 관측들 기반으로 i번째 스텝을 수정했을 수 있음)
                correctedPred = self.currentPredictions[i]
                if abs(actual) > 1e-10:
                    healedErrors.append(abs(correctedPred - actual) / abs(actual))
                else:
                    healedErrors.append(abs(correctedPred - actual))
            healedMape = float(np.mean(healedErrors) * 100) if healedErrors else originalMape

        # 개선율
        if originalMape > 1e-10:
            improvementPct = max(0.0, (originalMape - healedMape) / originalMape * 100)
        else:
            improvementPct = 0.0

        preds, lower, upper = self.getUpdatedForecast()

        return HealingReport(
            overallHealth=self.health,
            healthScore=self.healthScore,
            totalObserved=self.observedCount,
            totalCorrected=len(self.corrections),
            originalMape=originalMape,
            healedMape=healedMape,
            improvementPct=improvementPct,
            corrections=list(self.corrections),
            healingLog=list(self.healingLog),
            driftEvents=list(self._driftEvents),
            updatedPredictions=preds,
            updatedLower=lower,
            updatedUpper=upper,
        )

    def reset(self) -> None:
        """
        관측 데이터 및 교정 이력 초기화.
        원래 예측값은 유지한다.
        """
        self.currentPredictions = self.originalPredictions.copy()
        self.currentLower = self.originalLower.copy()
        self.currentUpper = self.originalUpper.copy()

        self.observedValues.clear()
        self.observedCount = 0
        self.errors.clear()
        self.absErrors.clear()
        self.signedBias.clear()

        self.cusumPos = 0.0
        self.cusumNeg = 0.0
        self.ewmaError = 0.0
        self.driftDetected = False
        self.driftDirection = None
        self.driftMagnitude = 0.0
        self._driftEvents.clear()

        self.corrections.clear()
        self.healingLog.clear()

        self.health = 'healthy'
        self.healthScore = 100.0
        self.refitRecommended = False
        self.refitReason = None
        self.refitRecommendedAt = None

        self.healingLog.append("[reset] SelfHealingForecast 상태 초기화 완료")

    # ======================================================================
    # 드리프트 감지
    # ======================================================================

    def _updateCUSUM(self, error: float) -> None:
        """
        양방향 CUSUM으로 체계적 편향 감지

        S_pos = max(0, S_pos + (error - k))
        S_neg = max(0, S_neg + (-error - k))

        k = 0.5 * sigma  (슬랙 파라미터: 작은 변동 무시)
        h = 5.0 * sigma  (임계값: 이를 넘으면 드리프트)

        error > 0 이 지속 -> cusumPos 증가 -> 예측이 과소 (downward_bias)
        error < 0 이 지속 -> cusumNeg 증가 -> 예측이 과대 (upward_bias)
        """
        k = 0.5 * self._referenceStd
        h = 5.0 * self._referenceStd

        self.cusumPos = max(0.0, self.cusumPos + error - k)
        self.cusumNeg = max(0.0, self.cusumNeg + (-error) - k)

        prevDrift = self.driftDetected

        if self.cusumPos > h:
            self.driftDetected = True
            self.driftDirection = 'downward_bias'   # 실제 > 예측
            self.driftMagnitude = self.cusumPos / max(self.observedCount, 1)
        elif self.cusumNeg > h:
            self.driftDetected = True
            self.driftDirection = 'upward_bias'     # 예측 > 실제
            self.driftMagnitude = self.cusumNeg / max(self.observedCount, 1)
        else:
            # 아직 임계값 미만이면 드리프트 해제 가능
            if self.cusumPos < h * 0.3 and self.cusumNeg < h * 0.3:
                self.driftDetected = False
                self.driftDirection = None
                self.driftMagnitude = 0.0

        # 새 드리프트 이벤트 기록
        if self.driftDetected and not prevDrift:
            event = {
                'step': self.observedCount,
                'direction': self.driftDirection,
                'magnitude': self.driftMagnitude,
                'cusumPos': self.cusumPos,
                'cusumNeg': self.cusumNeg,
            }
            self._driftEvents.append(event)
            self.healingLog.append(
                f"[drift] 드리프트 감지 at step {self.observedCount}: "
                f"{self.driftDirection}, magnitude={self.driftMagnitude:.4f}"
            )

    def _updateEWMA(self, error: float) -> None:
        """
        EWMA로 최근 오차 추세 추적

        E_t = lambda * error + (1 - lambda) * E_{t-1}

        EWMA는 최근 오차에 가중치를 두어 추세 변화를
        CUSUM보다 빠르게 감지한다.
        """
        self.ewmaError = (
            self.ewmaLambda * error
            + (1.0 - self.ewmaLambda) * self.ewmaError
        )

    # ======================================================================
    # 건강 상태 평가
    # ======================================================================

    def _evaluateHealth(self) -> None:
        """
        예측 건강 상태를 0-100 점수로 평가

        점수 기준:
        - 100-80: healthy  (오차 정상 범위)
        - 80-50:  degrading (오차 증가 추세 또는 경미한 편향)
        - 50-0:   critical  (체계적 편향 또는 큰 오차)

        드리프트 감지 + 교정 적용 후 오차가 줄면 'healed'로 전환
        """
        if self.observedCount == 0:
            self.health = 'healthy'
            self.healthScore = 100.0
            return

        score = 100.0

        # --- (1) MAE 기반 감점 ---
        mae = float(np.mean(self.absErrors))
        maeRatio = mae / max(self._referenceStd, 1e-10)
        # maeRatio가 1이면 오차가 참조 표준편차 수준 -> 적당
        # maeRatio > 2이면 큰 오차
        maeDeduction = min(40.0, maeRatio * 20.0)
        score -= maeDeduction

        # --- (2) 편향 기반 감점 ---
        meanBias = abs(float(np.mean(self.errors)))
        biasRatio = meanBias / max(self._referenceStd, 1e-10)
        biasDeduction = min(30.0, biasRatio * 15.0)
        score -= biasDeduction

        # --- (3) 오차 추세 감점 ---
        if len(self.absErrors) >= 3:
            recentErrors = self.absErrors[-min(5, len(self.absErrors)):]
            # 오차가 증가하는 추세면 감점
            if len(recentErrors) >= 3:
                xVals = np.arange(len(recentErrors), dtype=np.float64)
                slope = np.polyfit(xVals, recentErrors, 1)[0]
                if slope > 0:
                    trendDeduction = min(15.0, slope / max(self._referenceStd, 1e-10) * 10.0)
                    score -= trendDeduction

        # --- (4) EWMA 편향 감점 ---
        ewmaRatio = abs(self.ewmaError) / max(self._referenceStd, 1e-10)
        ewmaDeduction = min(15.0, ewmaRatio * 10.0)
        score -= ewmaDeduction

        # --- (5) 드리프트 감지 시 추가 감점 ---
        if self.driftDetected:
            score -= 10.0

        # 점수 범위 제한
        score = max(0.0, min(100.0, score))
        self.healthScore = score

        # 건강 상태 결정
        if score >= 80:
            self.health = 'healthy'
        elif score >= 50:
            self.health = 'degrading'
        else:
            self.health = 'critical'

        # 교정 후 오차가 개선되었으면 'healed'
        if len(self.corrections) > 0 and self.observedCount >= 3:
            recentAbsErrors = self.absErrors[-min(3, len(self.absErrors)):]
            earlyAbsErrors = self.absErrors[:min(3, len(self.absErrors))]
            if np.mean(recentAbsErrors) < np.mean(earlyAbsErrors) * 0.7:
                self.health = 'healed'

    def _evaluateRefitNeed(self) -> None:
        """
        재학습 권장 여부 판단

        재학습 권장 기준:
        - healthScore < 50 (5단계 이상 관측 후)
        - 드리프트 감지 + 교정으로도 개선 안 됨
        - 현재 MAPE > 초기 기대치 * 2
        """
        if self.refitRecommended:
            return  # 이미 권장됨

        if self.observedCount < 5:
            return  # 판단하기엔 데이터 부족

        reasons = []

        # 기준 1: 건강 점수 저조
        if self.healthScore < 50:
            reasons.append(f"건강점수 {self.healthScore:.1f} < 50")

        # 기준 2: 드리프트 + 교정 실패
        if self.driftDetected and len(self.corrections) >= 2:
            recentErrors = self.absErrors[-3:]
            if np.mean(recentErrors) > self._referenceStd * 2:
                reasons.append(
                    f"드리프트 감지 후 교정에도 오차 큼 "
                    f"(최근 MAE={np.mean(recentErrors):.4f})"
                )

        # 기준 3: MAPE 과다
        currentMape = self._computeCurrentMape()
        if currentMape > 30.0:  # 30% 이상
            reasons.append(f"MAPE {currentMape:.1f}% > 30%")

        if reasons:
            self.refitRecommended = True
            self.refitReason = "; ".join(reasons)
            self.refitRecommendedAt = self.observedCount
            self.healingLog.append(
                f"[refit] 재학습 권장 at step {self.observedCount}: {self.refitReason}"
            )

    # ======================================================================
    # 교정 알고리즘
    # ======================================================================

    def _applyCorrection(self) -> None:
        """
        관측된 오차 패턴을 기반으로 남은 예측 교정

        교정 전략:
        1. 편향 보정: 체계적 편향 감지 시 예측 시프트
        2. 추세 보정: 오차에 추세가 있으면 선형 보정
        3. 계절 보정: 오차에 주기적 패턴이 있으면 반영
        4. 변동성 보정: 신뢰구간 너비 조정
        """
        remainingSteps = self.totalSteps - self.observedCount
        if remainingSteps <= 0:
            return

        errors = np.array(self.errors, dtype=np.float64)

        # ------------------------------------------------------------------
        # 1. 편향 보정
        # ------------------------------------------------------------------
        meanError = float(np.mean(errors))
        biasCorrection = meanError  # 양수 = 과소예측 -> 위로 시프트

        # ------------------------------------------------------------------
        # 2. 추세 보정 (오차에 선형 추세가 있는 경우)
        # ------------------------------------------------------------------
        errorTrend = 0.0
        if len(errors) >= 3:
            xVals = np.arange(len(errors), dtype=np.float64)
            coeffs = np.polyfit(xVals, errors, 1)
            errorTrend = float(coeffs[0])

        # ------------------------------------------------------------------
        # 3. 계절 보정 (오차에 주기적 패턴이 있는 경우)
        # ------------------------------------------------------------------
        seasonalCorrection = np.zeros(remainingSteps)
        if len(errors) >= self.period * 2 and self.period > 1:
            # 오차의 계절 패턴 추출
            seasonalPattern = np.zeros(self.period)
            counts = np.zeros(self.period)
            for i, e in enumerate(errors):
                phase = i % self.period
                seasonalPattern[phase] += e
                counts[phase] += 1
            # 안전한 나눗셈
            mask = counts > 0
            seasonalPattern[mask] /= counts[mask]
            # 평균 제거 (편향은 biasCorrection에서 처리)
            seasonalPattern -= np.mean(seasonalPattern)

            for h in range(remainingSteps):
                futurePhase = (self.observedCount + h) % self.period
                seasonalCorrection[h] = seasonalPattern[futurePhase]

        # ------------------------------------------------------------------
        # 4. 교정 적용 (감쇠 적용 - 먼 미래일수록 불확실)
        # ------------------------------------------------------------------
        for h in range(remainingSteps):
            stepIdx = self.observedCount + h
            decay = self._getDecayFactor(h, remainingSteps)

            # 기본 교정: 편향 + 추세 외삽 + 계절
            trendComponent = errorTrend * (self.observedCount + h)
            correction = (biasCorrection + trendComponent + seasonalCorrection[h]) * decay

            # healingMode별 교정 강도 조절
            if self.healingMode == 'conservative':
                correction *= 0.5
            elif self.healingMode == 'aggressive':
                correction *= 1.5
            # 'adaptive'는 1.0 (기본)

            self.currentPredictions[stepIdx] = (
                self.originalPredictions[stepIdx] + correction
            )

        # ------------------------------------------------------------------
        # 5. 신뢰구간 업데이트
        # ------------------------------------------------------------------
        observedStd = float(np.std(errors)) if len(errors) > 1 else self._referenceStd
        widthRatio = observedStd / max(self._referenceStd, 1e-10)

        # 구간 너비는 넓어지기만 함 (보수적)
        widthRatio = max(widthRatio, 1.0)

        for h in range(remainingSteps):
            stepIdx = self.observedCount + h
            originalWidth = self.originalUpper[stepIdx] - self.originalLower[stepIdx]
            newWidth = originalWidth * widthRatio

            # 먼 미래일수록 추가 불확실성 부여
            horizonFactor = 1.0 + 0.05 * h
            newWidth *= horizonFactor

            center = self.currentPredictions[stepIdx]
            self.currentLower[stepIdx] = center - newWidth / 2.0
            self.currentUpper[stepIdx] = center + newWidth / 2.0

        # ------------------------------------------------------------------
        # 6. 교정 이력 기록
        # ------------------------------------------------------------------
        self.corrections.append({
            'step': self.observedCount,
            'biasCorrection': float(biasCorrection),
            'trendCorrection': float(errorTrend),
            'seasonalApplied': bool(np.any(seasonalCorrection != 0)),
            'widthRatio': float(widthRatio),
            'healingMode': self.healingMode,
        })

    def _getDecayFactor(self, horizonStep: int, totalRemaining: int) -> float:
        """
        미래 스텝에 대한 교정 감쇠 계수

        가까운 미래는 교정 효과가 크고,
        먼 미래는 교정 효과가 줄어든다 (지수 감쇠).

        Parameters
        ----------
        horizonStep : int
            교정 대상 스텝 (0부터 시작)
        totalRemaining : int
            남은 전체 스텝 수

        Returns
        -------
        float
            0.0 ~ 1.0 감쇠 계수
        """
        if totalRemaining <= 1:
            return 1.0

        # 지수 감쇠: exp(-rate * h)
        # rate는 전체 남은 스텝에 따라 조정
        # 목표: 마지막 스텝에서 약 0.3 수준
        rate = -np.log(0.3) / max(totalRemaining - 1, 1)
        decay = float(np.exp(-rate * horizonStep))

        # healingMode에 따른 감쇠 조정
        if self.healingMode == 'aggressive':
            # 감쇠를 느리게 (교정 효과 오래 유지)
            decay = decay ** 0.7
        elif self.healingMode == 'conservative':
            # 감쇠를 빠르게 (교정 효과 빠르게 소멸)
            decay = decay ** 1.5

        return max(0.0, min(1.0, decay))

    # ======================================================================
    # 유틸리티
    # ======================================================================

    def _computeMape(self, predicted: np.ndarray, actual: np.ndarray) -> float:
        """MAPE 계산 (0이 아닌 값만 사용)"""
        mask = np.abs(actual) > 1e-10
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

    def _computeCurrentMape(self) -> float:
        """현재까지 관측된 데이터 기준 MAPE"""
        if self.observedCount == 0:
            return 0.0
        predicted = self.originalPredictions[:self.observedCount]
        actual = np.array(self.observedValues)
        return self._computeMape(predicted, actual)

    def _computeCurrentMae(self) -> float:
        """현재까지 관측된 데이터 기준 MAE"""
        if self.observedCount == 0:
            return 0.0
        return float(np.mean(self.absErrors))

    def _buildStatus(self, correctionApplied: bool) -> HealingStatus:
        """HealingStatus 객체 생성"""
        remaining = self.totalSteps - self.observedCount
        biasEstimate = float(np.mean(self.errors)) if self.errors else 0.0
        currentMape = self._computeCurrentMape()
        currentMae = self._computeCurrentMae()

        # 사람이 읽을 수 있는 메시지 생성
        message = self._buildStatusMessage(
            currentMape, currentMae, biasEstimate, remaining
        )

        return HealingStatus(
            health=self.health,
            healthScore=self.healthScore,
            observedCount=self.observedCount,
            remainingSteps=remaining,
            driftDetected=self.driftDetected,
            driftDirection=self.driftDirection,
            driftMagnitude=self.driftMagnitude,
            correctionApplied=correctionApplied,
            biasEstimate=biasEstimate,
            mape=currentMape,
            mae=currentMae,
            refitRecommended=self.refitRecommended,
            refitReason=self.refitReason,
            message=message,
        )

    def _buildStatusMessage(
        self,
        mape: float,
        mae: float,
        bias: float,
        remaining: int,
    ) -> str:
        """사람이 읽을 수 있는 상태 메시지 생성"""
        parts = []

        # 건강 상태
        healthLabels = {
            'healthy': '정상',
            'degrading': '저하 중',
            'critical': '위험',
            'healed': '치유됨',
        }
        parts.append(f"상태: {healthLabels.get(self.health, self.health)} "
                      f"(점수 {self.healthScore:.0f}/100)")

        # 관측 진행률
        parts.append(f"관측: {self.observedCount}/{self.totalSteps}")

        # 오차 정보
        if self.observedCount > 0:
            parts.append(f"MAPE: {mape:.1f}%, MAE: {mae:.4f}")

        # 편향 정보
        if abs(bias) > self._referenceStd * 0.5:
            direction = "과소예측" if bias > 0 else "과대예측"
            parts.append(f"편향: {direction} ({bias:+.4f})")

        # 드리프트
        if self.driftDetected:
            driftLabels = {
                'upward_bias': '상향 편향',
                'downward_bias': '하향 편향',
            }
            label = driftLabels.get(self.driftDirection, self.driftDirection or '')
            parts.append(f"드리프트: {label} 감지")

        # 재학습 권장
        if self.refitRecommended:
            parts.append(f"재학습 권장: {self.refitReason}")

        return " | ".join(parts)

    # ======================================================================
    # 표현
    # ======================================================================

    def __repr__(self) -> str:
        return (
            f"SelfHealingForecast("
            f"steps={self.totalSteps}, "
            f"observed={self.observedCount}, "
            f"health='{self.health}', "
            f"score={self.healthScore:.1f}, "
            f"mode='{self.healingMode}')"
        )
