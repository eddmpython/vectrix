"""
Vectrix: Pure self-implemented time series forecasting engine.

Uses only numpy + scipy.optimize (+ optional numba).
No external forecasting libraries (statsforecast, statsmodels, prophet).

Features:
- Zero-Config: feed data, get optimal forecasts automatically
- Flat prediction prevention: 4-level defense system
- Periodic drop pattern detection (E009: 61.3% improvement)
- Multi-seasonality decomposition (E006: 57.8% improvement)
- 2.3x faster than statsforecast
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

warnings.filterwarnings('ignore')

from .types import (
    ForecastResult,
    DataCharacteristics,
    FlatRiskAssessment,
    ModelResult,
    FlatPredictionInfo,
    RiskLevel,
    MODEL_INFO
)
from .analyzer import AutoAnalyzer
from .flat_defense import FlatRiskDiagnostic, FlatPredictionDetector, FlatPredictionCorrector
from .models import AdaptiveModelSelector
from .models.ensemble import VariabilityPreservingEnsemble

# 자체 구현 엔진
from .engine import TurboCore, ETSModel, ARIMAModel, ThetaModel, SeasonalDecomposition, MSTL, AutoMSTL, PeriodicDropDetector
from .engine import NaiveModel, SeasonalNaiveModel, MeanModel, RandomWalkDrift, WindowAverage
from .engine import CESModel, AutoCES, CrostonClassic, CrostonSBA, CrostonTSB, AutoCroston, DynamicOptimizedTheta
from .engine import TBATS, AutoTBATS, GARCHModel, EGARCHModel, GJRGARCHModel
from .engine.ets import AutoETS
from .engine.arima import AutoARIMA
from .engine.theta import OptimizedTheta
from .engine.decomposition import MSTLDecomposition


class Vectrix:
    """
    Vectrix: Zero-config time series forecasting engine.

    Usage:
        >>> from vectrix import Vectrix
        >>> fx = Vectrix()
        >>> result = fx.forecast(df, dateCol='date', valueCol='sales', steps=30)
        >>> print(result.predictions)

    Dependencies: numpy, pandas, scipy (required), numba (optional)
    """

    VERSION = "3.0.0"

    # 자체 구현 모델 목록
    NATIVE_MODELS = {
        'auto_ets': {
            'name': 'AutoETS (Native)',
            'description': '자체 구현 자동 지수평활법',
            'class': AutoETS
        },
        'auto_arima': {
            'name': 'AutoARIMA (Native)',
            'description': '자체 구현 자동 ARIMA',
            'class': AutoARIMA
        },
        'theta': {
            'name': 'Theta (Native)',
            'description': '자체 구현 Theta 모델',
            'class': OptimizedTheta
        },
        'ets_aan': {
            'name': 'ETS(A,A,N)',
            'description': "자체 구현 Holt's Linear",
            'class': lambda period: ETSModel('A', 'A', 'N', period)
        },
        'ets_aaa': {
            'name': 'ETS(A,A,A)',
            'description': '자체 구현 Holt-Winters Additive',
            'class': lambda period: ETSModel('A', 'A', 'A', period)
        },
        'seasonal_naive': {
            'name': 'Seasonal Naive (Native)',
            'description': '자체 구현 계절성 나이브',
            'class': None  # 직접 구현
        },
        'mstl': {
            'name': 'MSTL (Native)',
            'description': '자체 구현 다중 계절 분해',
            'class': MSTLDecomposition
        },
        'auto_mstl': {
            'name': 'AutoMSTL (Native)',
            'description': '자동 다중 계절성 분해 + ARIMA (E006 57.8% 개선)',
            'class': AutoMSTL
        },
        'naive': {
            'name': 'Naive',
            'description': 'Random Walk — 마지막 값 반복',
            'class': NaiveModel
        },
        'mean': {
            'name': 'Mean',
            'description': '과거 평균값 예측',
            'class': MeanModel
        },
        'rwd': {
            'name': 'Random Walk with Drift',
            'description': '마지막 값 + 평균 추세',
            'class': RandomWalkDrift
        },
        'window_avg': {
            'name': 'Window Average',
            'description': '최근 윈도우 평균 예측',
            'class': WindowAverage
        },
        'auto_ces': {
            'name': 'AutoCES (Native)',
            'description': '복소수 지수평활법 자동 선택',
            'class': AutoCES
        },
        'croston': {
            'name': 'Croston (Auto)',
            'description': '간헐적 수요 예측 자동 선택',
            'class': AutoCroston
        },
        'dot': {
            'name': 'Dynamic Optimized Theta',
            'description': 'Theta+alpha+drift 동시 최적화',
            'class': DynamicOptimizedTheta
        },
        'tbats': {
            'name': 'TBATS (Native)',
            'description': 'Trigonometric 다중 계절성 모델',
            'class': AutoTBATS
        },
        'garch': {
            'name': 'GARCH(1,1)',
            'description': '조건부 분산 모델 (금융 변동성)',
            'class': GARCHModel
        },
        'egarch': {
            'name': 'EGARCH',
            'description': '비대칭 변동성 모델',
            'class': EGARCHModel
        },
        'gjr_garch': {
            'name': 'GJR-GARCH',
            'description': '임계 비대칭 GARCH',
            'class': GJRGARCHModel
        }
    }

    def __init__(
        self,
        locale: str = 'ko_KR',
        verbose: bool = False,
        nJobs: int = -1
    ):
        self.locale = locale
        self.verbose = verbose
        self.nJobs = nJobs

        # 컴포넌트
        self.analyzer = AutoAnalyzer()
        self.flatDiagnostic = FlatRiskDiagnostic()
        self.flatDetector = FlatPredictionDetector()
        self.flatCorrector = FlatPredictionCorrector()
        self.modelSelector = AdaptiveModelSelector()
        self.ensemble = VariabilityPreservingEnsemble()

        # 상태
        self.characteristics: Optional[DataCharacteristics] = None
        self.flatRisk: Optional[FlatRiskAssessment] = None
        self.modelResults: Dict[str, ModelResult] = {}
        self.dropDetector: Optional[PeriodicDropDetector] = None

        # 콜백
        self.onProgress: Optional[Callable] = None

        if verbose:
            from .engine.turbo import isNumbaAvailable
            numbaStatus = "✓ Numba 활성화" if isNumbaAvailable() else "✗ Numba 없음 (순수 Python)"
            print(f"[Vectrix v{self.VERSION}] {numbaStatus}")

    def setProgressCallback(self, callback: Callable):
        self.onProgress = callback

    def forecast(
        self,
        df: pd.DataFrame,
        dateCol: str,
        valueCol: str,
        steps: int = 30,
        trainRatio: float = 0.8
    ) -> ForecastResult:
        """
        시계열 예측 (완전 자체 구현)
        """
        try:
            # 1. 데이터 준비
            self._progress('데이터 준비 중...')
            workDf = self._prepareData(df, dateCol, valueCol)
            values = workDf[valueCol].values.astype(np.float64)
            n = len(values)

            if n < 10:
                return ForecastResult(
                    success=False,
                    error=f'데이터 부족: {n}개 (최소 10개 필요)'
                )

            # 2. 데이터 분석
            self._progress('데이터 특성 분석 중...')
            self.characteristics = self.analyzer.analyze(workDf, dateCol, valueCol)
            period = self.characteristics.period

            # 3. 정기 드롭 패턴 감지 (E009 결과: 61.3% 개선)
            self._progress('정기 드롭 패턴 감지 중...')
            self.dropDetector = PeriodicDropDetector(minDropRatio=0.8, minDropDuration=3)
            hasPeriodicDrop = self.dropDetector.detect(values)

            if hasPeriodicDrop and self.verbose:
                print(f"[드롭 감지] 주기: {self.dropDetector.dropPeriod}일, "
                      f"지속: {self.dropDetector.dropDuration}일, "
                      f"비율: {self.dropDetector.dropRatio:.2f}")

            # 4. 일직선 위험도 진단
            self._progress('일직선 예측 위험도 진단 중...')
            self.flatDiagnostic.period = period
            self.flatRisk = self.flatDiagnostic.diagnose(values, self.characteristics)

            if self.verbose:
                self._printRiskAssessment()

            # 5. 모델 선택
            self._progress('모델 선택 중...')
            selectedModels = self._selectNativeModels()

            if self.verbose:
                print(f"선택된 모델: {selectedModels}")

            # 6. 학습/테스트 분할
            splitIdx = int(n * trainRatio)
            trainData = values[:splitIdx]
            testData = values[splitIdx:]
            testSteps = len(testData)

            # 드롭 패턴 감지되면 학습 데이터에서 드롭 제거
            trainDataForModel = trainData
            trainDropDetector = None
            if hasPeriodicDrop:
                trainDropDetector = PeriodicDropDetector(minDropRatio=0.8, minDropDuration=3)
                trainDropDetector.detect(trainData)
                trainDropDetector.dropPeriod = self.dropDetector.dropPeriod
                trainDropDetector.dropDuration = self.dropDetector.dropDuration
                trainDropDetector.dropRatio = self.dropDetector.dropRatio
                trainDataForModel = trainDropDetector.removeDrops(trainData)
                if self.verbose:
                    print("[드롭 감지] 학습 데이터에서 드롭 구간 보간 처리")

            # 7. 모델 평가
            self._progress('모델 학습 중...')
            self.modelResults = self._evaluateNativeModels(
                selectedModels, trainDataForModel, testData, testSteps, period,
                applyDropPattern=hasPeriodicDrop, trainLength=len(trainData)
            )

            if not self.modelResults:
                return ForecastResult(
                    success=False,
                    error='모든 모델 평가 실패'
                )

            # 8. 최종 예측
            self._progress('예측 생성 중...')
            valuesForModel = self.dropDetector.removeDrops(values) if hasPeriodicDrop else values
            result = self._generateFinalPrediction(
                valuesForModel, steps, workDf, dateCol, period,
                applyDropPattern=hasPeriodicDrop, originalLength=n
            )

            self._progress('완료!')
            return result

        except Exception as e:
            import traceback
            if self.verbose:
                traceback.print_exc()
            return ForecastResult(success=False, error=str(e))

    def _selectNativeModels(self) -> List[str]:
        """자체 모델 선택"""
        riskLevel = self.flatRisk.riskLevel if self.flatRisk else RiskLevel.LOW
        n = self.characteristics.length if self.characteristics else 100
        period = self.characteristics.period if self.characteristics else 7
        hasMultiSeason = self.characteristics.hasMultipleSeasonality if self.characteristics else False
        seasonalStrength = self.characteristics.seasonalStrength if self.characteristics else 0.0

        # 다중 계절성 또는 강한 계절성이면 AutoMSTL 우선 (E006 결과: 57.8% 개선)
        if (hasMultiSeason or seasonalStrength > 0.4) and n >= 60:
            models = ['auto_mstl', 'auto_ets', 'theta']
        elif riskLevel in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            # 계절성 강제 모델 우선
            models = ['seasonal_naive', 'ets_aaa', 'theta']
        elif riskLevel == RiskLevel.MEDIUM:
            models = ['theta', 'auto_ets', 'ets_aaa', 'auto_arima']
        else:
            models = ['auto_ets', 'auto_arima', 'theta']

        # 데이터 길이로 필터
        if n < period * 2:
            models = [m for m in models if m not in ['ets_aaa', 'mstl', 'auto_mstl']]

        if n < 30:
            models = [m for m in models if m not in ['auto_arima']]

        if not models:
            models = ['theta', 'seasonal_naive']

        return models

    def _evaluateNativeModels(
        self,
        modelIds: List[str],
        trainData: np.ndarray,
        testData: np.ndarray,
        testSteps: int,
        period: int,
        applyDropPattern: bool = False,
        trainLength: int = 0
    ) -> Dict[str, ModelResult]:
        """자체 모델 평가"""
        results = {}
        totalModels = len(modelIds)
        self._fittedModels = {}  # 학습된 모델 캐시 (재학습 방지)

        for i, modelId in enumerate(modelIds):
            startTime = time.time()

            try:
                self._progress(f'{self.NATIVE_MODELS.get(modelId, {}).get("name", modelId)} 학습 중...')

                # 예측 (모델 캐시 포함)
                predictions, lower95, upper95, fittedModel = self._fitAndPredictNativeWithCache(
                    modelId, trainData, testSteps, period
                )
                if fittedModel is not None:
                    self._fittedModels[modelId] = fittedModel

                # E009: 드롭 패턴 재적용 (예측 구간에 드롭 발생 시에만)
                if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
                    predictions = self.dropDetector.applyDropPatternSmart(predictions, trainLength)
                    lower95 = self.dropDetector.applyDropPatternSmart(lower95, trainLength)
                    upper95 = self.dropDetector.applyDropPatternSmart(upper95, trainLength)

                # 일직선 감지
                flatInfo = self.flatDetector.detect(predictions, trainData)

                # 보정
                if flatInfo.isFlat:
                    predictions, flatInfo = self.flatCorrector.correct(
                        predictions, trainData, flatInfo, period
                    )

                # 평가
                mape = TurboCore.mape(testData, predictions[:len(testData)])
                rmse = TurboCore.rmse(testData, predictions[:len(testData)])
                mae = TurboCore.mae(testData, predictions[:len(testData)])

                if flatInfo.isFlat and not flatInfo.correctionApplied:
                    mape *= 1.5

                result = ModelResult(
                    modelId=modelId,
                    modelName=self.NATIVE_MODELS.get(modelId, {}).get('name', modelId),
                    predictions=predictions,
                    lower95=lower95,
                    upper95=upper95,
                    mape=mape,
                    rmse=rmse,
                    mae=mae,
                    flatInfo=flatInfo,
                    trainingTime=time.time() - startTime,
                    isValid=True
                )
                results[modelId] = result

                flatMark = "⚠️" if flatInfo.isFlat else "✓"
                self._progress(f'{result.modelName} 완료 ({i+1}/{totalModels})')

                if self.verbose:
                    print(f"  {flatMark} {modelId}: MAPE={mape:.2f}%")

            except Exception as e:
                if self.verbose:
                    print(f"  ✗ {modelId} 오류: {str(e)[:50]}")

        return results

    def _fitAndPredictNativeWithCache(
        self,
        modelId: str,
        trainData: np.ndarray,
        steps: int,
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """자체 모델로 학습 및 예측 (모델 객체도 반환)"""

        if modelId == 'auto_ets':
            model = AutoETS(period=period)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'auto_arima':
            model = AutoARIMA(maxP=3, maxD=2, maxQ=3)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'theta':
            model = OptimizedTheta(period=period)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'ets_aan':
            model = ETSModel('A', 'A', 'N', period)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'ets_aaa':
            model = ETSModel('A', 'A', 'A', period)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'seasonal_naive':
            pred, lo, hi = self._seasonalNaive(trainData, steps, period)
            return pred, lo, hi, None

        elif modelId == 'mstl':
            model = MSTLDecomposition(periods=[period])
            predictions = model.predict(trainData, steps)
            sigma = np.std(trainData[-30:])
            margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
            return predictions, predictions - margin, predictions + margin, model

        elif modelId == 'auto_mstl':
            model = AutoMSTL()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'tbats':
            model = AutoTBATS(periods=[period])
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'garch':
            model = GARCHModel()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'egarch':
            model = EGARCHModel()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'gjr_garch':
            model = GJRGARCHModel()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'auto_ces':
            model = AutoCES(period=period)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'croston':
            model = AutoCroston()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'dot':
            model = DynamicOptimizedTheta(period=period)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'naive':
            model = NaiveModel()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'mean':
            model = MeanModel()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'rwd':
            model = RandomWalkDrift()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'window_avg':
            model = WindowAverage(window=min(period, 30))
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        else:
            pred, lo, hi = self._seasonalNaive(trainData, steps, period)
            return pred, lo, hi, None

    def _fitAndPredictNative(
        self,
        modelId: str,
        trainData: np.ndarray,
        steps: int,
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """자체 모델로 학습 및 예측"""
        pred, lo, hi, _ = self._fitAndPredictNativeWithCache(
            modelId, trainData, steps, period
        )
        return pred, lo, hi

    def _seasonalNaive(
        self,
        values: np.ndarray,
        steps: int,
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """자체 구현 Seasonal Naive"""
        n = len(values)

        if n < period:
            period = max(1, n // 2)

        lastSeason = values[-period:]
        predictions = np.tile(lastSeason, steps // period + 1)[:steps]

        sigma = np.std(values[-min(30, n):])
        margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))

        return predictions, predictions - margin, predictions + margin

    def _refitModelOnFullData(
        self,
        modelId: str,
        allValues: np.ndarray,
        steps: int,
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """캐시된 모델의 파라미터를 재사용하여 전체 데이터로 빠르게 재학습"""
        cachedModel = getattr(self, '_fittedModels', {}).get(modelId)

        if cachedModel is None:
            # 캐시 없으면 새로 학습
            return self._fitAndPredictNative(modelId, allValues, steps, period)

        try:
            if modelId == 'auto_ets' and hasattr(cachedModel, 'bestModel') and cachedModel.bestModel is not None:
                # 최적 ETS 구조 재사용 (파라미터 최적화 건너뜀)
                bm = cachedModel.bestModel
                model = ETSModel(
                    errorType=bm.errorType, trendType=bm.trendType,
                    seasonalType=bm.seasonalType, period=bm.period, damped=bm.damped
                )
                model.alpha, model.beta, model.gamma, model.phi = bm.alpha, bm.beta, bm.gamma, bm.phi
                model._initializeState(allValues)
                model._fitWithParams(allValues)
                model.fitted = True
                return model.predict(steps)

            elif modelId == 'auto_arima' and hasattr(cachedModel, 'bestOrder') and cachedModel.bestOrder is not None:
                # 최적 ARIMA order 재사용
                model = ARIMAModel(order=cachedModel.bestOrder)
                model.fit(allValues)
                return model.predict(steps)

            elif modelId == 'theta' and hasattr(cachedModel, 'bestTheta'):
                # 최적 theta 재사용 (6개 시도 건너뜀)
                from .engine.theta import ThetaModel
                model = ThetaModel(theta=cachedModel.bestTheta, period=period)
                model.fit(allValues)
                return model.predict(steps)

            elif modelId in ('ets_aan', 'ets_aaa'):
                # 최적화된 파라미터 재사용
                bm = cachedModel
                model = ETSModel(
                    errorType=bm.errorType, trendType=bm.trendType,
                    seasonalType=bm.seasonalType, period=bm.period, damped=bm.damped
                )
                model.alpha, model.beta, model.gamma, model.phi = bm.alpha, bm.beta, bm.gamma, bm.phi
                model._initializeState(allValues)
                model._fitWithParams(allValues)
                model.fitted = True
                return model.predict(steps)

            elif modelId == 'auto_mstl':
                # MSTL: 감지된 주기 재사용
                if hasattr(cachedModel, 'detectedPeriods'):
                    from .engine.mstl import MSTL as MSTLEngine
                    model = MSTLEngine(periods=cachedModel.detectedPeriods, autoDetect=False)
                    model.fit(allValues)
                    return model.predict(steps)

            # 기타 모델이나 실패 시 새로 학습
            return self._fitAndPredictNative(modelId, allValues, steps, period)

        except Exception:
            return self._fitAndPredictNative(modelId, allValues, steps, period)

    def _generateFinalPrediction(
        self,
        allValues: np.ndarray,
        steps: int,
        df: pd.DataFrame,
        dateCol: str,
        period: int,
        applyDropPattern: bool = False,
        originalLength: int = 0
    ) -> ForecastResult:
        """최종 예측 생성 - 캐시된 모델 파라미터로 빠르게 전체 데이터 재학습"""
        warnings = list(self.flatRisk.warnings) if self.flatRisk else []

        if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
            warnings.append(
                f"정기 드롭 패턴 감지: {self.dropDetector.dropPeriod}일 주기, "
                f"{self.dropDetector.dropDuration}일 지속 (E009 적용)"
            )

        # 유효한 모델 필터링 (일직선 아닌 모델만)
        validModels = [
            mid for mid, res in self.modelResults.items()
            if res.isValid and (not res.flatInfo or not res.flatInfo.isFlat)
        ]

        # MAPE 기준 최적 모델 선택
        if validModels:
            bestModelId = min(
                validModels,
                key=lambda k: self.modelResults[k].mape
            )
        else:
            bestModelId = min(
                self.modelResults.keys(),
                key=lambda k: self.modelResults[k].mape
            )

        bestResult = self.modelResults[bestModelId]
        bestModelName = bestResult.modelName

        # 캐시된 파라미터로 빠르게 재학습 (최적화 건너뜀)
        predictions, lower95, upper95 = self._refitModelOnFullData(
            bestModelId, allValues, steps, period
        )

        # E009: 드롭 패턴 재적용
        if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
            predictions = self.dropDetector.applyDropPatternSmart(predictions, originalLength)
            lower95 = self.dropDetector.applyDropPatternSmart(lower95, originalLength)
            upper95 = self.dropDetector.applyDropPatternSmart(upper95, originalLength)

        # 앙상블 (캐시된 모델로 빠르게)
        if len(validModels) >= 2:
            try:
                modelPredictions = {}
                for mid in validModels[:3]:
                    pred, _, _ = self._refitModelOnFullData(mid, allValues, steps, period)
                    modelPredictions[mid] = pred

                # 가중 앙상블
                weights = []
                for mid in modelPredictions.keys():
                    mape = self.modelResults[mid].mape
                    weights.append(1.0 / (mape + 1e-6))
                weights = np.array(weights)
                weights = weights / weights.sum()

                ensemblePred = np.zeros(steps)
                for i, (mid, pred) in enumerate(modelPredictions.items()):
                    ensemblePred += weights[i] * pred

                origStd = np.std(allValues[-min(30, len(allValues)):])
                ensembleStd = np.std(ensemblePred)
                singleStd = np.std(predictions)

                if abs(ensembleStd - origStd) < abs(singleStd - origStd):
                    predictions = ensemblePred
                    bestModelId = 'ensemble'
                    bestModelName = '변동성 보존 앙상블 (Native)'

                    if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
                        predictions = self.dropDetector.applyDropPatternSmart(predictions, originalLength)

                    sigma = origStd
                    margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
                    lower95 = predictions - margin
                    upper95 = predictions + margin

                    if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
                        lower95 = self.dropDetector.applyDropPatternSmart(lower95, originalLength)
                        upper95 = self.dropDetector.applyDropPatternSmart(upper95, originalLength)
            except Exception:
                pass

        # 일직선 감지 및 보정
        dropApplied = False
        if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
            willOccur, _ = self.dropDetector.willDropOccurInPrediction(originalLength, steps)
            dropApplied = willOccur

        flatInfo = self.flatDetector.detect(predictions, allValues)
        if flatInfo.isFlat and not dropApplied:
            predictions, flatInfo = self.flatCorrector.correct(
                predictions, allValues, flatInfo, period
            )
            warnings.append(flatInfo.message)

        # 날짜 생성
        lastDate = pd.to_datetime(df[dateCol].iloc[-1])
        freq = self.characteristics.frequency.value if self.characteristics else 'D'
        futureDates = pd.date_range(start=lastDate + pd.Timedelta(days=1), periods=steps, freq=freq)
        dateStrings = [d.strftime('%Y-%m-%d') for d in futureDates]

        return ForecastResult(
            success=True,
            predictions=predictions,
            dates=dateStrings,
            lower95=lower95,
            upper95=upper95,
            bestModelId=bestModelId,
            bestModelName=bestModelName,
            allModelResults=self.modelResults,
            characteristics=self.characteristics,
            flatRisk=self.flatRisk,
            flatInfo=flatInfo,
            interpretation=self._generateInterpretation(),
            warnings=warnings
        )

    def _prepareData(self, df: pd.DataFrame, dateCol: str, valueCol: str) -> pd.DataFrame:
        """데이터 전처리"""
        workDf = df.copy()
        workDf[dateCol] = pd.to_datetime(workDf[dateCol])
        workDf = workDf.sort_values(dateCol).reset_index(drop=True)

        if workDf[valueCol].isna().any():
            workDf[valueCol] = workDf[valueCol].interpolate(method='linear')
            workDf[valueCol] = workDf[valueCol].ffill().bfill()

        return workDf

    def _generateInterpretation(self) -> Dict[str, Any]:
        """해석 생성"""
        c = self.characteristics
        return {
            'engine': 'Vectrix (100% 자체 구현)',
            'dataQuality': '양호' if c.predictabilityScore >= 60 else '보통' if c.predictabilityScore >= 40 else '주의 필요',
            'predictability': c.predictabilityScore,
            'dependencies': ['numpy', 'numba (optional)', 'scipy.optimize', 'pandas']
        }

    def _progress(self, message: str):
        if self.onProgress:
            self.onProgress(message)
        if self.verbose:
            print(f"[Vectrix] {message}")

    def _printRiskAssessment(self):
        r = self.flatRisk
        print("\n" + "=" * 50)
        print("일직선 예측 위험도 평가")
        print("=" * 50)
        print(f"위험 점수: {r.riskScore:.2f}")
        print(f"위험 수준: {r.riskLevel.value}")
        print(f"권장 모델: {r.recommendedModels}")
        print("=" * 50 + "\n")

    def analyze(
        self,
        df: pd.DataFrame,
        dateCol: str,
        valueCol: str
    ) -> Dict[str, Any]:
        """
        데이터 분석만 수행 (예측 없이)

        Returns
        -------
        Dict
            {'characteristics': DataCharacteristics, 'flatRisk': FlatRiskAssessment}
        """
        workDf = self._prepareData(df, dateCol, valueCol)
        values = workDf[valueCol].values.astype(np.float64)

        characteristics = self.analyzer.analyze(workDf, dateCol, valueCol)
        period = characteristics.period

        self.flatDiagnostic.period = period
        flatRisk = self.flatDiagnostic.diagnose(values, characteristics)

        return {
            'characteristics': characteristics,
            'flatRisk': flatRisk
        }

