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

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from .analyzer import AutoAnalyzer

# Self-implemented engines
from .engine import (
    ARIMAModel,
    AutoCES,
    AutoCroston,
    AutoMSTL,
    AutoTBATS,
    DynamicOptimizedTheta,
    EGARCHModel,
    ETSModel,
    GARCHModel,
    GJRGARCHModel,
    MeanModel,
    NaiveModel,
    PeriodicDropDetector,
    RandomWalkDrift,
    TurboCore,
    WindowAverage,
)
from .engine.arima import AutoARIMA
from .engine.decomposition import MSTLDecomposition
from .engine.dtsf import DynamicTimeScanForecaster
from .engine.esn import EchoStateForecaster
from .engine.ets import AutoETS
from .engine.fourTheta import AdaptiveThetaEnsemble
from .engine.theta import OptimizedTheta
from .flat_defense import FlatPredictionCorrector, FlatPredictionDetector, FlatRiskDiagnostic
from .models import AdaptiveModelSelector
from .models.ensemble import VariabilityPreservingEnsemble
from .types import DataCharacteristics, FlatRiskAssessment, ForecastResult, ModelResult, RiskLevel


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

    VERSION = "0.0.8"

    NATIVE_MODELS = {
        'auto_ets': {
            'name': 'AutoETS (Native)',
            'description': 'Self-implemented automatic exponential smoothing.',
            'class': AutoETS
        },
        'auto_arima': {
            'name': 'AutoARIMA (Native)',
            'description': 'Self-implemented automatic ARIMA.',
            'class': AutoARIMA
        },
        'theta': {
            'name': 'Theta (Native)',
            'description': 'Self-implemented Theta model.',
            'class': OptimizedTheta
        },
        'ets_aan': {
            'name': 'ETS(A,A,N)',
            'description': "Holt's Linear (additive trend, no season).",
            'class': lambda period: ETSModel('A', 'A', 'N', period)
        },
        'ets_aaa': {
            'name': 'ETS(A,A,A)',
            'description': 'Holt-Winters additive seasonality.',
            'class': lambda period: ETSModel('A', 'A', 'A', period)
        },
        'seasonal_naive': {
            'name': 'Seasonal Naive (Native)',
            'description': 'Seasonal naive baseline.',
            'class': None
        },
        'mstl': {
            'name': 'MSTL (Native)',
            'description': 'Multiple seasonal decomposition.',
            'class': MSTLDecomposition
        },
        'auto_mstl': {
            'name': 'AutoMSTL (Native)',
            'description': 'Auto multiple seasonality decomposition + ARIMA.',
            'class': AutoMSTL
        },
        'naive': {
            'name': 'Naive',
            'description': 'Random Walk — last value repetition.',
            'class': NaiveModel
        },
        'mean': {
            'name': 'Mean',
            'description': 'Historical mean forecast.',
            'class': MeanModel
        },
        'rwd': {
            'name': 'Random Walk with Drift',
            'description': 'Last value + average trend.',
            'class': RandomWalkDrift
        },
        'window_avg': {
            'name': 'Window Average',
            'description': 'Recent window average forecast.',
            'class': WindowAverage
        },
        'auto_ces': {
            'name': 'AutoCES (Native)',
            'description': 'Complex exponential smoothing auto selection.',
            'class': AutoCES
        },
        'croston': {
            'name': 'Croston (Auto)',
            'description': 'Intermittent demand auto selection.',
            'class': AutoCroston
        },
        'dot': {
            'name': 'Dynamic Optimized Theta',
            'description': 'Joint Theta+alpha+drift optimization.',
            'class': DynamicOptimizedTheta
        },
        'tbats': {
            'name': 'TBATS (Native)',
            'description': 'Trigonometric multiple seasonality model.',
            'class': AutoTBATS
        },
        'garch': {
            'name': 'GARCH(1,1)',
            'description': 'Conditional variance model (financial volatility).',
            'class': GARCHModel
        },
        'egarch': {
            'name': 'EGARCH',
            'description': 'Asymmetric volatility model.',
            'class': EGARCHModel
        },
        'gjr_garch': {
            'name': 'GJR-GARCH',
            'description': 'Threshold asymmetric GARCH.',
            'class': GJRGARCHModel
        },
        'four_theta': {
            'name': '4Theta Ensemble',
            'description': 'Weighted 4 theta line ensemble.',
            'class': AdaptiveThetaEnsemble
        },
        'esn': {
            'name': 'Echo State Network',
            'description': 'Reservoir Computing nonlinear forecasting.',
            'class': EchoStateForecaster
        },
        'dtsf': {
            'name': 'Dynamic Time Scan',
            'description': 'Non-parametric pattern matching forecasting.',
            'class': DynamicTimeScanForecaster
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

        # Components
        self.analyzer = AutoAnalyzer()
        self.flatDiagnostic = FlatRiskDiagnostic()
        self.flatDetector = FlatPredictionDetector()
        self.flatCorrector = FlatPredictionCorrector()
        self.modelSelector = AdaptiveModelSelector()
        self.ensemble = VariabilityPreservingEnsemble()

        # State
        self.characteristics: Optional[DataCharacteristics] = None
        self.flatRisk: Optional[FlatRiskAssessment] = None
        self.modelResults: Dict[str, ModelResult] = {}
        self.dropDetector: Optional[PeriodicDropDetector] = None

        # Callbacks
        self.onProgress: Optional[Callable] = None

        if verbose:
            from .engine.turbo import isNumbaAvailable
            numbaStatus = "✓ Numba enabled" if isNumbaAvailable() else "✗ Numba not found (pure Python)"
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
        Time series forecasting (fully self-implemented).
        """
        try:
            self._progress('Preparing data...')
            workDf = self._prepareData(df, dateCol, valueCol)
            values = workDf[valueCol].values.astype(np.float64)
            n = len(values)

            if n < 10:
                return ForecastResult(
                    success=False,
                    error=f'Insufficient data: {n} points (minimum 10 required)'
                )

            self._progress('Analyzing data characteristics...')
            self.characteristics = self.analyzer.analyze(workDf, dateCol, valueCol)
            period = self.characteristics.period

            self._progress('Detecting periodic drop patterns...')
            self.dropDetector = PeriodicDropDetector(minDropRatio=0.8, minDropDuration=3)
            hasPeriodicDrop = self.dropDetector.detect(values)

            if hasPeriodicDrop and self.verbose:
                print(f"[Drop detected] Period: {self.dropDetector.dropPeriod}, "
                      f"Duration: {self.dropDetector.dropDuration}, "
                      f"Ratio: {self.dropDetector.dropRatio:.2f}")

            self._progress('Diagnosing flat prediction risk...')
            self.flatDiagnostic.period = period
            self.flatRisk = self.flatDiagnostic.diagnose(values, self.characteristics)

            if self.verbose:
                self._printRiskAssessment()

            self._progress('Selecting models...')
            selectedModels = self._selectNativeModels()

            if self.verbose:
                print(f"Selected models: {selectedModels}")
            splitIdx = int(n * trainRatio)
            trainData = values[:splitIdx]
            testData = values[splitIdx:]
            testSteps = len(testData)

            # Remove drops from training data if periodic drop detected
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
                    print("[Drop detected] Drop sections interpolated in training data")

            self._progress('Training models...')
            self.modelResults = self._evaluateNativeModels(
                selectedModels, trainDataForModel, testData, testSteps, period,
                applyDropPattern=hasPeriodicDrop, trainLength=len(trainData)
            )

            if not self.modelResults:
                return ForecastResult(
                    success=False,
                    error='All model evaluations failed'
                )

            self._progress('Generating forecasts...')
            valuesForModel = self.dropDetector.removeDrops(values) if hasPeriodicDrop else values
            result = self._generateFinalPrediction(
                valuesForModel, steps, workDf, dateCol, period,
                applyDropPattern=hasPeriodicDrop, originalLength=n
            )

            self._progress('Done!')
            return result

        except Exception as e:
            import traceback
            if self.verbose:
                traceback.print_exc()
            return ForecastResult(success=False, error=str(e))

    def _selectNativeModels(self) -> List[str]:
        """Select native models based on data characteristics."""
        riskLevel = self.flatRisk.riskLevel if self.flatRisk else RiskLevel.LOW
        n = self.characteristics.length if self.characteristics else 100
        period = self.characteristics.period if self.characteristics else 7
        hasMultiSeason = self.characteristics.hasMultipleSeasonality if self.characteristics else False
        seasonalStrength = self.characteristics.seasonalStrength if self.characteristics else 0.0
        freq = self.characteristics.frequency.value if self.characteristics else 'D'

        if freq == 'H' and n >= 100:
            models = ['dot', 'auto_ces', 'dtsf', 'auto_mstl', 'esn']
        elif (hasMultiSeason or seasonalStrength > 0.4) and n >= 60:
            models = ['dot', 'auto_ces', 'four_theta', 'auto_mstl', 'dtsf']
        elif riskLevel in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
            models = ['dot', 'four_theta', 'seasonal_naive', 'ets_aaa', 'esn']
        elif riskLevel == RiskLevel.MEDIUM:
            models = ['dot', 'auto_ces', 'four_theta', 'esn', 'dtsf']
        else:
            models = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_arima']

        if n < 30:
            models = [m for m in models if m not in ['auto_arima', 'dtsf']]
        if n < period * 2:
            models = [m for m in models if m not in ['ets_aaa', 'mstl', 'auto_mstl']]

        if not models:
            models = ['dot', 'auto_ces']

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
        """Evaluate native models (parallel)."""
        results = {}
        totalModels = len(modelIds)
        self._fittedModels = {}

        def evaluateSingle(modelId):
            startTime = time.time()
            predictions, lower95, upper95, fittedModel = self._fitAndPredictNativeWithCache(
                modelId, trainData, testSteps, period
            )

            if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
                predictions = self.dropDetector.applyDropPatternSmart(predictions, trainLength)
                lower95 = self.dropDetector.applyDropPatternSmart(lower95, trainLength)
                upper95 = self.dropDetector.applyDropPatternSmart(upper95, trainLength)

            flatInfo = self.flatDetector.detect(predictions, trainData)

            if flatInfo.isFlat:
                predictions, flatInfo = self.flatCorrector.correct(
                    predictions, trainData, flatInfo, period
                )

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
            return modelId, result, fittedModel

        nWorkers = min(totalModels, 4) if self.nJobs != 1 else 1

        if nWorkers <= 1:
            for i, modelId in enumerate(modelIds):
                try:
                    self._progress(f'Training {self.NATIVE_MODELS.get(modelId, {}).get("name", modelId)}...')
                    mid, result, fittedModel = evaluateSingle(modelId)
                    results[mid] = result
                    if fittedModel is not None:
                        self._fittedModels[mid] = fittedModel
                    self._progress(f'{result.modelName} done ({i+1}/{totalModels})')
                    if self.verbose:
                        flatMark = "⚠️" if result.flatInfo and result.flatInfo.isFlat else "✓"
                        print(f"  {flatMark} {modelId}: MAPE={result.mape:.2f}%")
                except Exception as e:
                    if self.verbose:
                        print(f"  ✗ {modelId} error: {str(e)[:50]}")
        else:
            self._progress(f'Training {totalModels} models in parallel...')
            with ThreadPoolExecutor(max_workers=nWorkers) as executor:
                futureMap = {
                    executor.submit(evaluateSingle, mid): mid for mid in modelIds
                }
                completed = 0
                for future in as_completed(futureMap):
                    modelId = futureMap[future]
                    completed += 1
                    try:
                        mid, result, fittedModel = future.result()
                        results[mid] = result
                        if fittedModel is not None:
                            self._fittedModels[mid] = fittedModel
                        self._progress(f'{result.modelName} done ({completed}/{totalModels})')
                        if self.verbose:
                            flatMark = "⚠️" if result.flatInfo and result.flatInfo.isFlat else "✓"
                            print(f"  {flatMark} {modelId}: MAPE={result.mape:.2f}%")
                    except Exception as e:
                        if self.verbose:
                            print(f"  ✗ {modelId} error: {str(e)[:50]}")

        return results

    def _fitAndPredictNativeWithCache(
        self,
        modelId: str,
        trainData: np.ndarray,
        steps: int,
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Fit and predict with a native model (also returns the fitted model object)."""

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

        elif modelId == 'four_theta':
            model = AdaptiveThetaEnsemble(period=period)
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'esn':
            model = EchoStateForecaster()
            model.fit(trainData)
            pred, lo, hi = model.predict(steps)
            return pred, lo, hi, model

        elif modelId == 'dtsf':
            model = DynamicTimeScanForecaster()
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
        """Fit and predict with a native model."""
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
        """Self-implemented Seasonal Naive."""
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
        """Re-fit on full data using cached model parameters for speed."""
        cachedModel = getattr(self, '_fittedModels', {}).get(modelId)

        if cachedModel is None:
            # No cache, fit from scratch
            return self._fitAndPredictNative(modelId, allValues, steps, period)

        try:
            if modelId == 'auto_ets' and hasattr(cachedModel, 'bestModel') and cachedModel.bestModel is not None:
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
                # Reuse optimal ARIMA order
                model = ARIMAModel(order=cachedModel.bestOrder)
                model.fit(allValues)
                return model.predict(steps)

            elif modelId == 'theta' and hasattr(cachedModel, 'bestTheta'):
                # Reuse optimal theta value
                from .engine.theta import ThetaModel
                model = ThetaModel(theta=cachedModel.bestTheta, period=period)
                model.fit(allValues)
                return model.predict(steps)

            elif modelId in ('ets_aan', 'ets_aaa'):
                # Reuse optimized parameters
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
                # MSTL: reuse detected periods
                if hasattr(cachedModel, 'detectedPeriods'):
                    from .engine.mstl import MSTL as MSTLEngine
                    model = MSTLEngine(periods=cachedModel.detectedPeriods, autoDetect=False)
                    model.fit(allValues)
                    return model.predict(steps)

            # Other models or cache miss — train from scratch
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
        """Generate final prediction — fast refit on full data using cached model parameters."""
        warnings = list(self.flatRisk.warnings) if self.flatRisk else []

        if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
            warnings.append(
                f"Periodic drop pattern detected: {self.dropDetector.dropPeriod}-day cycle, "
                f"{self.dropDetector.dropDuration}-day duration (E009 applied)"
            )

        # Filter valid models (exclude flat predictions)
        validModels = [
            mid for mid, res in self.modelResults.items()
            if res.isValid and (not res.flatInfo or not res.flatInfo.isFlat)
        ]

        # Select best model by MAPE
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

        # Fast refit using cached parameters (skip optimization)
        predictions, lower95, upper95 = self._refitModelOnFullData(
            bestModelId, allValues, steps, period
        )

        # E009: reapply drop pattern
        if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
            predictions = self.dropDetector.applyDropPatternSmart(predictions, originalLength)
            lower95 = self.dropDetector.applyDropPatternSmart(lower95, originalLength)
            upper95 = self.dropDetector.applyDropPatternSmart(upper95, originalLength)

        for mid in validModels:
            try:
                fPred, fLo, fHi = self._refitModelOnFullData(mid, allValues, steps, period)
                self.modelResults[mid].predictions = fPred
                self.modelResults[mid].lower95 = fLo
                self.modelResults[mid].upper95 = fHi
            except Exception:
                pass

        if len(validModels) >= 2:
            try:
                modelPredictions = {}
                for mid in validModels[:3]:
                    modelPredictions[mid] = self.modelResults[mid].predictions

                # Weighted ensemble
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
                    bestModelName = 'Variability-Preserving Ensemble (Native)'

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

        # Flat prediction detection and correction
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

        # Generate future dates
        lastDate = pd.to_datetime(df[dateCol].iloc[-1])
        freq = self.characteristics.frequency.value if self.characteristics else 'D'
        _FREQ_MAP = {'M': 'ME', 'Q': 'QE', 'Y': 'YE', 'H': 'h'}
        pdFreq = _FREQ_MAP.get(freq, freq)
        futureDates = pd.date_range(start=lastDate + pd.Timedelta(days=1), periods=steps, freq=pdFreq)
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
        """Data preprocessing."""
        workDf = df.copy()
        workDf[dateCol] = pd.to_datetime(workDf[dateCol])
        workDf = workDf.sort_values(dateCol).reset_index(drop=True)

        if workDf[valueCol].isna().any():
            workDf[valueCol] = workDf[valueCol].interpolate(method='linear')
            workDf[valueCol] = workDf[valueCol].ffill().bfill()

        return workDf

    def _generateInterpretation(self) -> Dict[str, Any]:
        """Generate interpretation summary."""
        c = self.characteristics
        return {
            'engine': 'Vectrix (100% self-implemented)',
            'dataQuality': 'Good' if c.predictabilityScore >= 60 else 'Fair' if c.predictabilityScore >= 40 else 'Caution',
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
        print("Flat Prediction Risk Assessment")
        print("=" * 50)
        print(f"Risk Score: {r.riskScore:.2f}")
        print(f"Risk Level: {r.riskLevel.value}")
        print(f"Recommended Models: {r.recommendedModels}")
        print("=" * 50 + "\n")

    def analyze(
        self,
        df: pd.DataFrame,
        dateCol: str,
        valueCol: str
    ) -> Dict[str, Any]:
        """
        Run data analysis only (no forecasting).

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

