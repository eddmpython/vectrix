"""
Vectrix: Time series forecasting engine with built-in Rust acceleration.

Built on numpy + scipy + pandas with 25 Rust-accelerated hot loops.
No external forecasting libraries (statsforecast, statsmodels, prophet).

Features:
- Zero-Config: feed data, get optimal forecasts automatically
- Flat prediction prevention: 4-level defense system
- Periodic drop pattern detection (E009: 61.3% improvement)
- Multi-seasonality decomposition (E006: 57.8% improvement)
- Built-in Rust engine for 5-67x acceleration
"""

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', module='vectrix')

from .analyzer import AutoAnalyzer
from .engine import PeriodicDropDetector, TurboCore
from .engine.registry import createModel, getCoreModelIds, getModelSpec, listModelIds, selectModels
from .flat_defense import FlatPredictionCorrector, FlatPredictionDetector, FlatRiskDiagnostic
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

    VERSION = "0.0.16"

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
        self.ensemble = VariabilityPreservingEnsemble()

        # State
        self.characteristics: Optional[DataCharacteristics] = None
        self.flatRisk: Optional[FlatRiskAssessment] = None
        self.modelResults: Dict[str, ModelResult] = {}
        self._fittedModels: Dict[str, Any] = {}
        self.dropDetector: Optional[PeriodicDropDetector] = None

        # Callbacks
        self.onProgress: Optional[Callable] = None

        if verbose:
            from .engine.turbo import isNumbaAvailable
            numbaStatus = "✓ Numba enabled" if isNumbaAvailable() else "✗ Numba not found (fallback mode)"
            print(f"[Vectrix v{self.VERSION}] {numbaStatus}")

    def setProgressCallback(self, callback: Callable):
        self.onProgress = callback

    def forecast(
        self,
        df: pd.DataFrame,
        dateCol: str,
        valueCol: str,
        steps: int = 30,
        trainRatio: float = 0.8,
        models: Optional[List[str]] = None,
        ensembleMethod: Optional[str] = None,
        confidenceLevel: float = 0.95
    ) -> ForecastResult:
        """
        Time series forecasting (fully self-implemented).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        dateCol : str
            Date column name.
        valueCol : str
            Value column name.
        steps : int
            Forecast horizon (default: 30).
        trainRatio : float
            Train/test split ratio (default: 0.8).
        models : list of str, optional
            Model IDs to evaluate. None = auto-select based on data characteristics.
            Available: 'dot', 'auto_ets', 'auto_arima', 'auto_ces', 'four_theta',
            'auto_mstl', 'tbats', 'theta', 'dtsf', 'esn', 'garch', 'croston', etc.
        ensembleMethod : str, optional
            Ensemble strategy. None = auto (variability-preserving).
            'mean' = simple average, 'weighted' = MAPE-weighted (default behavior),
            'median' = median ensemble, 'best' = no ensemble (single best model).
        confidenceLevel : float
            Confidence interval level (default: 0.95). E.g., 0.90 for 90% CI.
        """
        self._confidenceLevel = confidenceLevel
        self._ensembleMethod = ensembleMethod

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

            if hasPeriodicDrop and self.dropDetector.dropPeriod:
                dp = self.dropDetector.dropPeriod
                if dp == period or (period > 1 and dp % period == 0 and dp <= period * 2):
                    hasPeriodicDrop = False

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
            if models is not None:
                validModelIds = set(listModelIds())
                invalidModels = [m for m in models if m not in validModelIds]
                if invalidModels:
                    return ForecastResult(
                        success=False,
                        error=f'Unknown model IDs: {invalidModels}. '
                              f'Available: {sorted(validModelIds)}'
                    )
                selectedModels = list(models)
            else:
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

        except (ValueError, TypeError, KeyError, IndexError) as e:
            if self.verbose:
                import traceback
                traceback.print_exc()
            return ForecastResult(success=False, error=str(e))
        except np.linalg.LinAlgError as e:
            return ForecastResult(success=False, error=f'Numerical instability: {e}')

    def _selectNativeModels(self) -> List[str]:
        """Select native models dynamically from registry based on data characteristics."""
        riskLevel = self.flatRisk.riskLevel if self.flatRisk else RiskLevel.LOW
        n = self.characteristics.length if self.characteristics else 100
        period = self.characteristics.period if self.characteristics else 7
        hasMultiSeason = self.characteristics.hasMultipleSeasonality if self.characteristics else False
        seasonalStrength = self.characteristics.seasonalStrength if self.characteristics else 0.0
        freq = self.characteristics.frequency.value if self.characteristics else 'D'
        highFlatRisk = riskLevel in [RiskLevel.CRITICAL, RiskLevel.HIGH]

        models = selectModels(
            n=n, period=period, freq=freq,
            seasonalStrength=seasonalStrength,
            hasMultiSeason=hasMultiSeason,
            highFlatRisk=highFlatRisk,
            maxModels=5,
        )

        if n < period * 2:
            models = [m for m in models if m not in ['ets_aaa', 'mstl', 'auto_mstl']]

        return models if models else ['dot', 'auto_ces']

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
            smape = TurboCore.smape(testData, predictions[:len(testData)])

            if flatInfo.isFlat and not flatInfo.correctionApplied:
                mape *= 1.5

            result = ModelResult(
                modelId=modelId,
                modelName=self._getModelName(modelId),
                predictions=predictions,
                lower95=lower95,
                upper95=upper95,
                mape=mape,
                rmse=rmse,
                mae=mae,
                smape=smape,
                flatInfo=flatInfo,
                trainingTime=time.time() - startTime,
                isValid=True
            )
            return modelId, result, fittedModel

        def storeResult(mid, result, fittedModel, idx):
            results[mid] = result
            if fittedModel is not None:
                self._fittedModels[mid] = fittedModel
            self._progress(f'{result.modelName} done ({idx}/{totalModels})')
            if self.verbose:
                flatMark = "⚠️" if result.flatInfo and result.flatInfo.isFlat else "✓"
                print(f"  {flatMark} {mid}: MAPE={result.mape:.2f}%")

        nWorkers = min(totalModels, 4) if self.nJobs != 1 else 1

        if nWorkers <= 1:
            for i, modelId in enumerate(modelIds):
                try:
                    self._progress(f'Training {self._getModelName(modelId)}...')
                    mid, result, fittedModel = evaluateSingle(modelId)
                    storeResult(mid, result, fittedModel, i + 1)
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
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
                        storeResult(mid, result, fittedModel, completed)
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                        if self.verbose:
                            print(f"  ✗ {modelId} error: {str(e)[:50]}")

        return results

    @staticmethod
    def _getModelName(modelId: str) -> str:
        """Get display name for a model ID from registry."""
        spec = getModelSpec(modelId)
        return spec.name if spec else modelId

    def _fitAndPredictNativeWithCache(
        self,
        modelId: str,
        trainData: np.ndarray,
        steps: int,
        period: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Fit and predict using the model registry."""
        model = createModel(modelId, period)
        if model is None:
            pred, lo, hi = self._seasonalNaive(trainData, steps, period)
            return pred, lo, hi, None

        model.fit(trainData)
        pred, lo, hi = model.predict(steps)
        return pred, lo, hi, model

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
            return self._fitAndPredictNative(modelId, allValues, steps, period)

        try:
            if hasattr(cachedModel, 'refit'):
                cachedModel.refit(allValues)
                return cachedModel.predict(steps)
            return self._fitAndPredictNative(modelId, allValues, steps, period)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
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
        from scipy.stats import norm
        confidenceLevel = getattr(self, '_confidenceLevel', 0.95)
        zScore = norm.ppf(1 - (1 - confidenceLevel) / 2)

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
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                pass

        ensembleMethod = getattr(self, '_ensembleMethod', None)
        if ensembleMethod == 'best':
            pass
        elif len(validModels) >= 2:
            try:
                coreModels = [m for m in getCoreModelIds() if m in validModels]
                ensemblePool = coreModels + [m for m in validModels if m not in coreModels]
                modelPredictions = {}
                for mid in ensemblePool[:3]:
                    modelPredictions[mid] = self.modelResults[mid].predictions

                if ensembleMethod == 'mean':
                    ensemblePred = np.mean(
                        [pred for pred in modelPredictions.values()], axis=0
                    )
                    useEnsemble = True
                elif ensembleMethod == 'median':
                    ensemblePred = np.median(
                        [pred for pred in modelPredictions.values()], axis=0
                    )
                    useEnsemble = True
                else:
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
                    useEnsemble = (
                        ensembleMethod == 'weighted' or
                        abs(ensembleStd - origStd) < abs(singleStd - origStd)
                    )

                if useEnsemble:
                    predictions = ensemblePred
                    bestModelId = 'ensemble'
                    ensembleLabel = (ensembleMethod or 'auto').capitalize()
                    bestModelName = f'{ensembleLabel} Ensemble (Native)'

                    if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
                        predictions = self.dropDetector.applyDropPatternSmart(predictions, originalLength)

                    sigma = np.std(allValues[-min(30, len(allValues)):])
                    margin = zScore * sigma * np.sqrt(np.arange(1, steps + 1))
                    lower95 = predictions - margin
                    upper95 = predictions + margin

                    if applyDropPattern and self.dropDetector and self.dropDetector.hasPeriodicDrop():
                        lower95 = self.dropDetector.applyDropPatternSmart(lower95, originalLength)
                        upper95 = self.dropDetector.applyDropPatternSmart(upper95, originalLength)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
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

        if abs(confidenceLevel - 0.95) > 1e-6:
            halfWidth = (upper95 - lower95) / 2.0
            scaledWidth = halfWidth * (zScore / 1.96)
            lower95 = predictions - scaledWidth
            upper95 = predictions + scaledWidth

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

