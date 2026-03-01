"""
Forecast DNA - Time Series Fingerprinting

Extracts 65+ statistical features from a time series to create a "DNA profile".
This enables:
1. Automatic optimal model recommendation (meta-learning)
2. Similar time series search (cosine similarity)
3. Forecast difficulty pre-assessment (0-100)
4. Unique time series identifier generation (8-char hex fingerprint)

Captures the intrinsic statistical structure of each time series to automate
model selection and search for similar patterns in large time series databases
in O(N) time.

Usage:
    >>> from vectrix.adaptive.dna import ForecastDNA
    >>> dna = ForecastDNA()
    >>> profile = dna.analyze(y, period=7)
    >>> print(profile.fingerprint)      # '4F2A9B1C'
    >>> print(profile.difficulty)        # 'medium'
    >>> print(profile.recommendedModels) # ['auto_ets', 'theta', 'auto_arima']
    >>> print(profile.category)          # 'seasonal'
    >>>
    >>> sim = dna.similarity(profile1, profile2)  # 0.0~1.0
    >>> dna.findSimilar(profile, database)         # similar time series search
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy import stats as scipyStats  # noqa: F401
    from scipy.fft import fft as scipyFft  # noqa: F401
    from scipy.signal import periodogram
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class DNAProfile:
    """
    Time Series DNA Profile

    Attributes
    ----------
    features : Dict[str, float]
        Dictionary of 65+ statistical features
    fingerprint : str
        8-char hex unique identifier (MD5 hash of feature vector)
    difficulty : str
        Forecast difficulty ('easy', 'medium', 'hard', 'very_hard')
    difficultyScore : float
        Forecast difficulty score (0-100)
    recommendedModels : List[str]
        DNA-based model recommendations (sorted by fitness, descending)
    category : str
        Time series category
        ('trending', 'seasonal', 'stationary', 'intermittent', 'volatile', 'complex')
    summary : str
        Natural language summary
    """
    features: Dict[str, float] = field(default_factory=dict)
    fingerprint: str = ""
    difficulty: str = "medium"
    difficultyScore: float = 50.0
    recommendedModels: List[str] = field(default_factory=list)
    category: str = "stationary"
    summary: str = ""


class ForecastDNA:
    """
    Time Series DNA Analyzer

    Extracts 65+ statistical features to create a DNAProfile.
    Each feature computation is independently wrapped in try-except,
    so a failure in one does not affect the rest.

    Usage:
        >>> dna = ForecastDNA()
        >>> profile = dna.analyze(y, period=7)
        >>> print(profile.fingerprint)      # '4F2A9B1C'
        >>> print(profile.difficulty)        # 'medium'
        >>> print(profile.recommendedModels) # ['auto_ets', 'theta', 'auto_arima']
        >>> print(profile.category)          # 'seasonal'
        >>>
        >>> sim = dna.similarity(profile1, profile2)  # 0.0~1.0
        >>> dna.findSimilar(profile, database)         # similar time series search
    """

    def analyze(self, y: np.ndarray, period: int = 1) -> DNAProfile:
        """
        Full DNA analysis

        Parameters
        ----------
        y : np.ndarray
            Input time series (1-dimensional)
        period : int
            Time series period (1 if non-seasonal)

        Returns
        -------
        DNAProfile
            Analysis result
        """
        yArr = np.asarray(y, dtype=np.float64).ravel()

        # Remove NaN/Inf
        validMask = np.isfinite(yArr)
        if not np.any(validMask):
            return DNAProfile(summary="No valid data available.")
        yClean = yArr[validMask]

        if len(yClean) < 4:
            return DNAProfile(
                features={'length': float(len(yClean))},
                summary="Data length is too short (fewer than 4 observations)."
            )

        features = self._extractFeatures(yClean, period)
        fingerprint = self._computeFingerprint(features)
        difficulty, difficultyScore = self._assessDifficulty(features)
        recommendedModels = self._recommendModels(features)
        category = self._categorize(features)
        summary = self._generateSummary(features, difficulty, category)

        return DNAProfile(
            features=features,
            fingerprint=fingerprint,
            difficulty=difficulty,
            difficultyScore=difficultyScore,
            recommendedModels=recommendedModels,
            category=category,
            summary=summary
        )

    # ==================================================================
    # Feature Extraction (65+ features)
    # ==================================================================

    def _extractFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """
        Extract 65+ statistical features

        By category:
        - Basic statistics (10)
        - Trend (5)
        - Seasonality (8)
        - Autocorrelation (8)
        - Nonlinearity (5)
        - Stability (5)
        - Stationarity (4)
        - Forecastability (5)
        - Intermittency (5)
        - Volatility (5)
        - Other (5)
        """
        features: Dict[str, float] = {}

        # Basic statistics (10)
        try:
            features.update(self._basicStats(y))
        except Exception:
            pass

        # Trend (5)
        try:
            features.update(self._trendFeatures(y))
        except Exception:
            pass

        # Seasonality (8)
        try:
            features.update(self._seasonalFeatures(y, period))
        except Exception:
            pass

        # Autocorrelation (8)
        try:
            features.update(self._autocorrelationFeatures(y))
        except Exception:
            pass

        # Nonlinearity (5)
        try:
            features.update(self._nonlinearityFeatures(y))
        except Exception:
            pass

        # Stability (5)
        try:
            features.update(self._stabilityFeatures(y))
        except Exception:
            pass

        # Stationarity (4)
        try:
            features.update(self._stationarityFeatures(y))
        except Exception:
            pass

        # Forecastability (5)
        try:
            features.update(self._forecastabilityFeatures(y, period))
        except Exception:
            pass

        # Intermittency (5)
        try:
            features.update(self._intermittencyFeatures(y))
        except Exception:
            pass

        # Volatility (5)
        try:
            features.update(self._volatilityFeatures(y))
        except Exception:
            pass

        # Other (5)
        try:
            features.update(self._otherFeatures(y))
        except Exception:
            pass

        return features

    # ------------------------------------------------------------------
    # 1. Basic Statistics (10 features)
    # ------------------------------------------------------------------

    def _basicStats(self, y: np.ndarray) -> Dict[str, float]:
        """
        Basic statistics

        length, mean, std, cv (coefficient of variation), skewness, kurtosis,
        min, max, iqr, range_to_mean_ratio
        """
        n = len(y)
        mean = float(np.mean(y))
        std = float(np.std(y, ddof=1)) if n > 1 else 0.0
        cv = abs(std / mean) if abs(mean) > 1e-10 else 0.0

        skew = 0.0
        kurt = 0.0
        if n > 2 and std > 1e-10:
            centered = (y - mean) / std
            skew = float(np.mean(centered ** 3))
        if n > 3 and std > 1e-10:
            centered = (y - mean) / std
            kurt = float(np.mean(centered ** 4) - 3.0)

        q25, q75 = float(np.percentile(y, 25)), float(np.percentile(y, 75))
        iqr = q75 - q25
        yMin, yMax = float(np.min(y)), float(np.max(y))
        rangeVal = yMax - yMin
        rangeToMean = rangeVal / abs(mean) if abs(mean) > 1e-10 else 0.0

        return {
            'length': float(n),
            'mean': mean,
            'std': std,
            'cv': cv,
            'skewness': skew,
            'kurtosis': kurt,
            'min': yMin,
            'max': yMax,
            'iqr': iqr,
            'rangeToMeanRatio': rangeToMean
        }

    # ------------------------------------------------------------------
    # 2. Trend (5 features)
    # ------------------------------------------------------------------

    def _trendFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Trend-related features

        trendStrength: STL decomposition-based trend strength (0~1)
        trendSlope: Linear regression slope (normalized)
        trendLinearity: R^2 (linear goodness of fit)
        trendCurvature: Quadratic coefficient (nonlinear trend)
        trendDirection: Direction (+1/0/-1)
        """
        n = len(y)
        x = np.arange(n, dtype=np.float64)

        # Linear regression
        xMean = np.mean(x)
        yMean = np.mean(y)
        xDev = x - xMean
        yDev = y - yMean
        sxx = np.sum(xDev ** 2)
        sxy = np.sum(xDev * yDev)

        slope = sxy / sxx if sxx > 1e-10 else 0.0
        intercept = yMean - slope * xMean
        fitted = intercept + slope * x
        residuals = y - fitted

        ssRes = np.sum(residuals ** 2)
        ssTot = np.sum(yDev ** 2)
        r2 = 1.0 - ssRes / ssTot if ssTot > 1e-10 else 0.0
        r2 = max(0.0, r2)

        # Normalized slope: relative to full range
        yRange = np.max(y) - np.min(y)
        normalizedSlope = slope * n / yRange if yRange > 1e-10 else 0.0

        # Quadratic fit (curvature)
        curvature = 0.0
        if n >= 5:
            try:
                coeffs = np.polyfit(x, y, 2)
                curvature = coeffs[0] * n ** 2 / yRange if yRange > 1e-10 else 0.0
            except Exception:
                pass

        # Trend strength: residual ratio (STL-like)
        residStd = float(np.std(residuals, ddof=1)) if n > 1 else 0.0
        yStd = float(np.std(y, ddof=1)) if n > 1 else 1.0
        trendStrength = max(0.0, 1.0 - residStd / yStd) if yStd > 1e-10 else 0.0

        direction = 1.0 if slope > 1e-10 else (-1.0 if slope < -1e-10 else 0.0)

        return {
            'trendStrength': float(np.clip(trendStrength, 0, 1)),
            'trendSlope': float(normalizedSlope),
            'trendLinearity': float(r2),
            'trendCurvature': float(np.clip(curvature, -10, 10)),
            'trendDirection': float(direction)
        }

    # ------------------------------------------------------------------
    # 3. Seasonality (8 features)
    # ------------------------------------------------------------------

    def _seasonalFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """
        Seasonality-related features

        seasonalStrength: Seasonal strength (0~1)
        seasonalPeakPeriod: Peak period from frequency analysis
        seasonalAmplitude: Seasonal amplitude (normalized)
        seasonalPhaseConsistency: Phase consistency
        seasonalHarmonicRatio: Harmonic ratio relative to fundamental frequency
        seasonalAutoCorr: Autocorrelation at seasonal lag
        seasonalAdjustedVariance: Residual variance ratio after seasonal adjustment
        multiSeasonalScore: Multiple seasonality score
        """
        n = len(y)

        # Defaults
        result = {
            'seasonalStrength': 0.0,
            'seasonalPeakPeriod': float(period),
            'seasonalAmplitude': 0.0,
            'seasonalPhaseConsistency': 0.0,
            'seasonalHarmonicRatio': 0.0,
            'seasonalAutoCorr': 0.0,
            'seasonalAdjustedVariance': 1.0,
            'multiSeasonalScore': 0.0
        }

        if period <= 1 or n < 2 * period:
            return result

        yStd = float(np.std(y, ddof=1))
        if yStd < 1e-10:
            return result

        # Seasonal mean profile
        nFullCycles = n // period
        if nFullCycles < 2:
            return result

        truncated = y[:nFullCycles * period]
        seasonalMatrix = truncated.reshape(nFullCycles, period)
        seasonalProfile = np.mean(seasonalMatrix, axis=0)
        seasonalProfileStd = float(np.std(seasonalProfile, ddof=1))

        # Seasonal strength: seasonal profile variation / total variation
        seasonalStrength = seasonalProfileStd / yStd
        seasonalStrength = float(np.clip(seasonalStrength, 0, 1))

        # Seasonal amplitude (normalized)
        amplitude = (float(np.max(seasonalProfile)) - float(np.min(seasonalProfile))) / yStd

        # Phase consistency: whether peak positions are consistent across cycles
        peakPositions = np.argmax(seasonalMatrix, axis=1)
        if len(peakPositions) > 1:
            # Circular statistics (circular mean)
            angles = 2.0 * np.pi * peakPositions / period
            meanCos = np.mean(np.cos(angles))
            meanSin = np.mean(np.sin(angles))
            phaseConsistency = float(np.sqrt(meanCos ** 2 + meanSin ** 2))
        else:
            phaseConsistency = 0.0

        # Autocorrelation at seasonal lag
        if n > period:
            acf = self._acf(y, maxLag=period)
            seasonalAutoCorr = float(acf[period]) if len(acf) > period else 0.0
        else:
            seasonalAutoCorr = 0.0

        # FFT-based period analysis
        peakPeriod = float(period)
        harmonicRatio = 0.0
        multiSeasonalScore = 0.0
        try:
            freqs, power = self._periodogram(y)
            if len(power) > 0:
                # Fundamental frequency of base period
                baseFundamental = 1.0 / period if period > 0 else 0.0

                # Peak period search
                sortedIdx = np.argsort(power)[::-1]
                if len(sortedIdx) > 0 and freqs[sortedIdx[0]] > 1e-10:
                    peakPeriod = float(1.0 / freqs[sortedIdx[0]])

                # Harmonic ratio
                if baseFundamental > 1e-10:
                    fundIdx = np.argmin(np.abs(freqs - baseFundamental))
                    fundPower = power[fundIdx] if fundIdx < len(power) else 0.0
                    # 2nd harmonic
                    harm2Idx = np.argmin(np.abs(freqs - 2 * baseFundamental))
                    harm2Power = power[harm2Idx] if harm2Idx < len(power) else 0.0
                    harmonicRatio = harm2Power / fundPower if fundPower > 1e-10 else 0.0

                # Multi-seasonality score: number of non-harmonic peaks among top 5
                topK = min(5, len(sortedIdx))
                topFreqs = freqs[sortedIdx[:topK]]
                if baseFundamental > 1e-10:
                    harmonicMask = np.array([
                        self._isHarmonic(f, baseFundamental) for f in topFreqs
                    ])
                    nonHarmonicCount = topK - int(np.sum(harmonicMask))
                    multiSeasonalScore = nonHarmonicCount / max(topK, 1)
        except Exception:
            pass

        # Residual variance ratio after seasonal adjustment
        seasonalComponent = np.tile(seasonalProfile, nFullCycles)[:n]
        if len(seasonalComponent) < n:
            seasonalComponent = np.concatenate([
                seasonalComponent,
                seasonalProfile[:n - len(seasonalComponent)]
            ])
        residual = y - seasonalComponent
        residStd = float(np.std(residual, ddof=1)) if n > 1 else 0.0
        adjVarianceRatio = (residStd / yStd) ** 2 if yStd > 1e-10 else 1.0

        result.update({
            'seasonalStrength': seasonalStrength,
            'seasonalPeakPeriod': peakPeriod,
            'seasonalAmplitude': float(np.clip(amplitude, 0, 10)),
            'seasonalPhaseConsistency': float(np.clip(phaseConsistency, 0, 1)),
            'seasonalHarmonicRatio': float(np.clip(harmonicRatio, 0, 10)),
            'seasonalAutoCorr': float(np.clip(seasonalAutoCorr, -1, 1)),
            'seasonalAdjustedVariance': float(np.clip(adjVarianceRatio, 0, 2)),
            'multiSeasonalScore': float(np.clip(multiSeasonalScore, 0, 1))
        })

        return result

    # ------------------------------------------------------------------
    # 4. Autocorrelation (8 features)
    # ------------------------------------------------------------------

    def _autocorrelationFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Autocorrelation-related features

        acf1~acf3: Autocorrelation at lag 1~3
        acfSum5: Sum of absolute autocorrelations at lag 1~5
        acfDecayRate: Autocorrelation decay rate
        pacfLag1: Partial autocorrelation at lag 1
        acfFirstZero: First lag where autocorrelation crosses zero
        ljungBoxStat: Ljung-Box statistic (autocorrelation significance)
        """
        n = len(y)
        maxLag = min(20, n // 3)
        if maxLag < 1:
            return {
                'acf1': 0.0, 'acf2': 0.0, 'acf3': 0.0,
                'acfSum5': 0.0, 'acfDecayRate': 0.0,
                'pacfLag1': 0.0, 'acfFirstZero': 0.0,
                'ljungBoxStat': 0.0
            }

        acf = self._acf(y, maxLag=maxLag)

        acf1 = float(acf[1]) if len(acf) > 1 else 0.0
        acf2 = float(acf[2]) if len(acf) > 2 else 0.0
        acf3 = float(acf[3]) if len(acf) > 3 else 0.0

        # Sum of absolute values at lag 1~5
        acfSum5 = float(np.sum(np.abs(acf[1:min(6, len(acf))])))

        # Decay rate: exponential decay model fit
        decayRate = 0.0
        absAcf = np.abs(acf[1:])
        if len(absAcf) >= 3 and absAcf[0] > 1e-10:
            try:
                logAcf = np.log(np.maximum(absAcf[:min(10, len(absAcf))], 1e-10))
                lags = np.arange(1, len(logAcf) + 1, dtype=np.float64)
                slope, _ = np.polyfit(lags, logAcf, 1)
                decayRate = -slope  # positive = faster decay
            except Exception:
                pass

        # PACF lag 1 (Yule-Walker approximation)
        pacfLag1 = acf1  # Simple approximation: PACF(1) = ACF(1)

        # First zero-crossing lag
        acfFirstZero = float(maxLag)
        for lag in range(1, len(acf)):
            if lag > 0 and acf[lag] * acf[lag - 1] < 0:
                acfFirstZero = float(lag)
                break

        # Ljung-Box statistic
        ljungBox = 0.0
        if n > 1:
            for k in range(1, min(11, len(acf))):
                ljungBox += acf[k] ** 2 / (n - k)
            ljungBox *= n * (n + 2)

        return {
            'acf1': float(np.clip(acf1, -1, 1)),
            'acf2': float(np.clip(acf2, -1, 1)),
            'acf3': float(np.clip(acf3, -1, 1)),
            'acfSum5': float(np.clip(acfSum5, 0, 5)),
            'acfDecayRate': float(np.clip(decayRate, -5, 5)),
            'pacfLag1': float(np.clip(pacfLag1, -1, 1)),
            'acfFirstZero': acfFirstZero,
            'ljungBoxStat': float(ljungBox)
        }

    # ------------------------------------------------------------------
    # 5. Nonlinearity (5 features)
    # ------------------------------------------------------------------

    def _nonlinearityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Nonlinearity-related features

        approximateEntropy: Approximate entropy (regularity measure)
        turningPointRate: Turning point rate (direction change frequency)
        thirdOrderAutoCorr: Third-order autocorrelation (nonlinear dependence)
        asymmetry: Rise/fall asymmetry
        nonlinearAutocorr: Nonlinear autocorrelation (ACF of y^2)
        """
        n = len(y)

        # Approximate entropy
        apen = self._approximateEntropy(y)

        # Turning point rate
        turningPoints = 0
        for i in range(1, n - 1):
            if (y[i] > y[i - 1] and y[i] > y[i + 1]) or \
               (y[i] < y[i - 1] and y[i] < y[i + 1]):
                turningPoints += 1
        turningPointRate = turningPoints / max(n - 2, 1)

        # Third-order autocorrelation: E[y(t) * y(t-1) * y(t-2)]
        thirdOrder = 0.0
        if n > 3:
            yc = y - np.mean(y)
            std = np.std(y, ddof=1)
            if std > 1e-10:
                yc = yc / std
                thirdOrder = float(np.mean(yc[2:] * yc[1:-1] * yc[:-2]))

        # Asymmetry: upward moves vs downward moves
        diffs = np.diff(y)
        if len(diffs) > 0:
            posSum = float(np.sum(diffs[diffs > 0]))
            negSum = float(np.abs(np.sum(diffs[diffs < 0])))
            totalMove = posSum + negSum
            asymmetry = (posSum - negSum) / totalMove if totalMove > 1e-10 else 0.0
        else:
            asymmetry = 0.0

        # Nonlinear autocorrelation: ACF(1) of y^2
        nonlinearAcf = 0.0
        if n > 3:
            ySq = (y - np.mean(y)) ** 2
            acfSq = self._acf(ySq, maxLag=1)
            nonlinearAcf = float(acfSq[1]) if len(acfSq) > 1 else 0.0

        return {
            'approximateEntropy': float(np.clip(apen, 0, 5)),
            'turningPointRate': float(np.clip(turningPointRate, 0, 1)),
            'thirdOrderAutoCorr': float(np.clip(thirdOrder, -5, 5)),
            'asymmetry': float(np.clip(asymmetry, -1, 1)),
            'nonlinearAutocorr': float(np.clip(nonlinearAcf, -1, 1))
        }

    # ------------------------------------------------------------------
    # 6. Stability (5 features)
    # ------------------------------------------------------------------

    def _stabilityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Stability-related features

        stabilityMean: Variation of segment means (lower = more stable)
        stabilityVariance: Variation of segment variances
        levelShiftCount: Number of level shifts
        levelShiftMagnitude: Maximum level shift magnitude
        structuralBreakScore: Structural break score
        """
        n = len(y)
        nSegments = max(min(n // 10, 10), 2)
        segLen = n // nSegments

        if segLen < 2:
            return {
                'stabilityMean': 0.0, 'stabilityVariance': 0.0,
                'levelShiftCount': 0.0, 'levelShiftMagnitude': 0.0,
                'structuralBreakScore': 0.0
            }

        segMeans = []
        segVars = []
        for i in range(nSegments):
            start = i * segLen
            end = start + segLen if i < nSegments - 1 else n
            seg = y[start:end]
            segMeans.append(float(np.mean(seg)))
            segVars.append(float(np.var(seg, ddof=1)) if len(seg) > 1 else 0.0)

        segMeans = np.array(segMeans)
        segVars = np.array(segVars)

        yStd = float(np.std(y, ddof=1)) if n > 1 else 1.0
        yVar = yStd ** 2

        # CV of segment means
        meanStab = float(np.std(segMeans, ddof=1)) / yStd if yStd > 1e-10 else 0.0

        # CV of segment variances
        meanVar = float(np.mean(segVars))
        varStab = float(np.std(segVars, ddof=1)) / meanVar if meanVar > 1e-10 else 0.0

        # Level shift detection
        meanDiffs = np.abs(np.diff(segMeans))
        threshold = 2.0 * yStd / np.sqrt(segLen) if yStd > 1e-10 else 0.0
        levelShiftCount = int(np.sum(meanDiffs > threshold))
        levelShiftMag = float(np.max(meanDiffs)) / yStd if yStd > 1e-10 else 0.0

        # Structural break score: simplified CUSUM-based calculation
        cusum = np.cumsum(y - np.mean(y))
        cusumRange = float(np.max(cusum) - np.min(cusum))
        structBreak = cusumRange / (yStd * np.sqrt(n)) if yStd > 1e-10 and n > 0 else 0.0

        return {
            'stabilityMean': float(np.clip(meanStab, 0, 5)),
            'stabilityVariance': float(np.clip(varStab, 0, 10)),
            'levelShiftCount': float(levelShiftCount),
            'levelShiftMagnitude': float(np.clip(levelShiftMag, 0, 10)),
            'structuralBreakScore': float(np.clip(structBreak, 0, 10))
        }

    # ------------------------------------------------------------------
    # 7. Stationarity (4 features)
    # ------------------------------------------------------------------

    def _stationarityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Stationarity-related features

        adfStatistic: ADF test statistic (simplified approximation)
        diffStationary: Stationarity improvement after first differencing
        hurstExponent: Hurst exponent (long memory)
        unitRootIndicator: Unit root indicator
        """
        n = len(y)

        # Hurst exponent
        hurst = self._hurstExponent(y)

        # ADF simplified approximation: estimated via AR(1) coefficient
        # y(t) = rho * y(t-1) + e(t), unit root implies rho -> 1
        adfStat = 0.0
        rho = 0.0
        if n > 3:
            yLag = y[:-1]
            yNow = y[1:]
            yLagMean = np.mean(yLag)
            denom = np.sum((yLag - yLagMean) ** 2)
            if denom > 1e-10:
                rho = float(np.sum((yLag - yLagMean) * (yNow - np.mean(yNow))) / denom)
                # ADF statistic = (rho - 1) / SE(rho) approximation
                residuals = yNow - rho * yLag
                se = float(np.std(residuals, ddof=1)) / np.sqrt(denom) if denom > 1e-10 else 1.0
                adfStat = (rho - 1.0) / se if se > 1e-10 else 0.0

        # Variance reduction rate after first differencing
        diffImprovement = 0.0
        if n > 2:
            origVar = float(np.var(y, ddof=1))
            diffVar = float(np.var(np.diff(y), ddof=1))
            if origVar > 1e-10:
                diffImprovement = 1.0 - diffVar / origVar
                # Positive means differencing helps (possible unit root)

        # Unit root indicator: closer rho is to 1, closer this is to 1
        unitRoot = max(0.0, 1.0 - abs(1.0 - abs(rho)) * 5.0) if abs(rho) > 0.5 else 0.0

        return {
            'adfStatistic': float(np.clip(adfStat, -20, 5)),
            'diffStationary': float(np.clip(diffImprovement, -2, 1)),
            'hurstExponent': float(np.clip(hurst, 0, 1)),
            'unitRootIndicator': float(np.clip(unitRoot, 0, 1))
        }

    # ------------------------------------------------------------------
    # 8. Forecastability (5 features)
    # ------------------------------------------------------------------

    def _forecastabilityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """
        Forecastability-related features

        spectralEntropy: Spectral entropy (frequency dispersion)
        forecastability: 1 - spectralEntropy (forecastability)
        signalToNoise: Signal-to-noise ratio
        sampleEntropy: Approximate sample entropy
        regularityIndex: Regularity index
        """
        n = len(y)

        # Spectral entropy
        specEntropy = self._spectralEntropy(y)
        forecastability = 1.0 - specEntropy

        # Signal-to-noise ratio
        # Trend + seasonality = signal, remainder = noise
        snr = 0.0
        yStd = float(np.std(y, ddof=1)) if n > 1 else 0.0
        if yStd > 1e-10 and n > 5:
            # Estimate signal using moving average
            windowSize = min(max(period, 3), n // 3)
            if windowSize >= 2:
                kernel = np.ones(windowSize) / windowSize
                smoothed = np.convolve(y, kernel, mode='valid')
                if len(smoothed) > 1:
                    noise = y[windowSize // 2:windowSize // 2 + len(smoothed)] - smoothed
                    signalPower = float(np.var(smoothed, ddof=1))
                    noisePower = float(np.var(noise, ddof=1))
                    if noisePower > 1e-10:
                        snr = signalPower / noisePower

        # Sample entropy (simplified)
        sampleEnt = self._approximateEntropy(y, m=2, r=0.2)

        # Regularity index: predictable component / total
        regularityIndex = max(0.0, 1.0 - sampleEnt / 3.0) if sampleEnt < 3.0 else 0.0

        return {
            'spectralEntropy': float(np.clip(specEntropy, 0, 1)),
            'forecastability': float(np.clip(forecastability, 0, 1)),
            'signalToNoise': float(np.clip(snr, 0, 100)),
            'sampleEntropy': float(np.clip(sampleEnt, 0, 5)),
            'regularityIndex': float(np.clip(regularityIndex, 0, 1))
        }

    # ------------------------------------------------------------------
    # 9. Intermittency (5 features)
    # ------------------------------------------------------------------

    def _intermittencyFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Intermittency-related features (important for demand forecasting)

        zeroRatio: Ratio of zero values
        adi: Average Demand Interval
        cv2: Squared coefficient of variation of non-zero demand
        intermittencyType: Intermittency type indicator
            0=continuous, 1=intermittent, 2=lumpy, 3=erratic
        demandDensity: Demand density (concentration of non-zero values)
        """
        n = len(y)
        threshold = np.max(np.abs(y)) * 0.001 if np.max(np.abs(y)) > 0 else 1e-10

        # Zero (near-zero) ratio
        zeroMask = np.abs(y) <= threshold
        zeroRatio = float(np.sum(zeroMask)) / n

        # Non-zero demand
        nonZero = y[~zeroMask]

        # ADI: average interval between demand occurrences
        adi = 1.0
        demandIndices = np.where(~zeroMask)[0]
        if len(demandIndices) > 1:
            intervals = np.diff(demandIndices)
            adi = float(np.mean(intervals))

        # CV^2: squared coefficient of variation of non-zero demand
        cv2 = 0.0
        if len(nonZero) > 1:
            nonZeroMean = float(np.mean(nonZero))
            nonZeroStd = float(np.std(nonZero, ddof=1))
            if abs(nonZeroMean) > 1e-10:
                cv2 = (nonZeroStd / nonZeroMean) ** 2

        # Intermittency type classification (Syntetos-Boylan)
        # ADI >= 1.32 and CV2 < 0.49 -> intermittent
        # ADI >= 1.32 and CV2 >= 0.49 -> lumpy
        # ADI < 1.32 and CV2 >= 0.49 -> erratic
        # ADI < 1.32 and CV2 < 0.49 -> continuous (smooth)
        if adi >= 1.32 and cv2 < 0.49:
            intermType = 1.0  # intermittent
        elif adi >= 1.32 and cv2 >= 0.49:
            intermType = 2.0  # lumpy
        elif adi < 1.32 and cv2 >= 0.49:
            intermType = 3.0  # erratic
        else:
            intermType = 0.0  # continuous

        # Demand density: concentration of non-zero values (Gini-like)
        demandDensity = 0.0
        if len(demandIndices) > 1 and n > 1:
            intervals = np.diff(demandIndices).astype(np.float64)
            intervalMean = float(np.mean(intervals))
            if intervalMean > 1e-10:
                intervalCv = float(np.std(intervals, ddof=1)) / intervalMean
                demandDensity = 1.0 / (1.0 + intervalCv)

        return {
            'zeroRatio': float(np.clip(zeroRatio, 0, 1)),
            'adi': float(np.clip(adi, 1, n)),
            'cv2': float(np.clip(cv2, 0, 100)),
            'intermittencyType': intermType,
            'demandDensity': float(np.clip(demandDensity, 0, 1))
        }

    # ------------------------------------------------------------------
    # 10. Volatility (5 features)
    # ------------------------------------------------------------------

    def _volatilityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Volatility-related features

        volatility: Standard deviation of log returns
        volatilityClustering: Volatility clustering (ARCH effect)
        garchEffect: Volatility autocorrelation (ACF of squared returns)
        extremeValueRatio: Extreme value ratio (exceeding 3-sigma)
        tailIndex: Tail thickness indicator (Hill estimator approximation)
        """
        n = len(y)

        # Volatility: difference-based
        diffs = np.diff(y)
        volatility = 0.0
        if len(diffs) > 1:
            yStd = float(np.std(y, ddof=1))
            diffStd = float(np.std(diffs, ddof=1))
            volatility = diffStd / yStd if yStd > 1e-10 else 0.0

        # Volatility clustering: autocorrelation of squared differences
        volClustering = 0.0
        garchEffect = 0.0
        if len(diffs) > 3:
            sqDiffs = diffs ** 2
            sqAcf = self._acf(sqDiffs, maxLag=min(5, len(sqDiffs) // 3))
            volClustering = float(sqAcf[1]) if len(sqAcf) > 1 else 0.0
            garchEffect = float(np.sum(np.abs(sqAcf[1:min(4, len(sqAcf))])))

        # Extreme value ratio
        extremeRatio = 0.0
        if n > 3:
            yMean = np.mean(y)
            yStd = float(np.std(y, ddof=1))
            if yStd > 1e-10:
                zScores = np.abs((y - yMean) / yStd)
                extremeRatio = float(np.sum(zScores > 3.0)) / n

        # Tail thickness: simplified Hill estimator
        tailIndex = 0.0
        if n > 10:
            absDiffs = np.sort(np.abs(diffs))[::-1]
            k = max(int(np.sqrt(len(absDiffs))), 2)
            topK = absDiffs[:k]
            if topK[-1] > 1e-10:
                logRatios = np.log(topK / topK[-1])
                tailIndex = float(np.mean(logRatios))

        return {
            'volatility': float(np.clip(volatility, 0, 10)),
            'volatilityClustering': float(np.clip(volClustering, -1, 1)),
            'garchEffect': float(np.clip(garchEffect, 0, 5)),
            'extremeValueRatio': float(np.clip(extremeRatio, 0, 1)),
            'tailIndex': float(np.clip(tailIndex, 0, 10))
        }

    # ------------------------------------------------------------------
    # 11. Other (5 features)
    # ------------------------------------------------------------------

    def _otherFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        Other features

        flatSpotRate: Ratio of consecutive identical value segments
        crossingRate: Mean-crossing frequency
        peakCount: Number of local peaks (normalized)
        longestRun: Longest increasing/decreasing run length (normalized)
        binEntropy: Binary entropy (increase/decrease pattern entropy)
        """
        n = len(y)

        # Consecutive identical value segments
        flatSpots = 0
        for i in range(1, n):
            if abs(y[i] - y[i - 1]) < 1e-10:
                flatSpots += 1
        flatSpotRate = flatSpots / max(n - 1, 1)

        # Mean-crossing frequency
        yMean = np.mean(y)
        crossings = 0
        for i in range(1, n):
            if (y[i] - yMean) * (y[i - 1] - yMean) < 0:
                crossings += 1
        crossingRate = crossings / max(n - 1, 1)

        # Local peak count
        peaks = 0
        for i in range(1, n - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
                peaks += 1
        peakCount = peaks / max(n - 2, 1)

        # Longest consecutive increasing/decreasing run
        longestRun = 0
        currentRun = 1
        if n > 1:
            direction = 1 if y[1] > y[0] else -1
            for i in range(2, n):
                newDir = 1 if y[i] > y[i - 1] else (-1 if y[i] < y[i - 1] else 0)
                if newDir == direction and newDir != 0:
                    currentRun += 1
                else:
                    longestRun = max(longestRun, currentRun)
                    currentRun = 1
                    direction = newDir
            longestRun = max(longestRun, currentRun)
        longestRunNorm = longestRun / n if n > 0 else 0.0

        # Binary entropy: Shannon entropy of increase/decrease patterns
        binEntropy = 0.0
        if n > 1:
            diffs = np.diff(y)
            posRate = float(np.sum(diffs > 0)) / len(diffs)
            negRate = 1.0 - posRate
            if 0 < posRate < 1:
                binEntropy = -(posRate * np.log2(posRate) + negRate * np.log2(negRate))

        return {
            'flatSpotRate': float(np.clip(flatSpotRate, 0, 1)),
            'crossingRate': float(np.clip(crossingRate, 0, 1)),
            'peakCount': float(np.clip(peakCount, 0, 1)),
            'longestRun': float(np.clip(longestRunNorm, 0, 1)),
            'binEntropy': float(np.clip(binEntropy, 0, 1))
        }

    # ==================================================================
    # Fingerprint
    # ==================================================================

    def _computeFingerprint(self, features: Dict[str, float]) -> str:
        """
        Hash-based 8-char hex identifier of the feature vector

        Serializes features in sorted key order, then takes the first 8 chars of MD5 hash.
        Identical time series produce identical fingerprints, but similar series may not
        produce similar fingerprints (hash property).
        """
        featureStr = '|'.join(
            f'{k}:{v:.6f}' for k, v in sorted(features.items())
        )
        return hashlib.md5(featureStr.encode()).hexdigest()[:8].upper()

    # ==================================================================
    # Difficulty Assessment
    # ==================================================================

    def _assessDifficulty(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Forecast difficulty assessment (0-100)

        Difficulty increasing factors:
        - High entropy -> +20
        - Low forecastability -> +20
        - High nonlinearity -> +15
        - Many level shifts (levelShiftCount) -> +10
        - High noise-to-signal -> +15
        - Intermittency -> +10
        - High volatility clustering -> +10

        Difficulty decreasing factors:
        - Strong trend -> -10
        - Strong seasonality -> -10
        - High autocorrelation -> -10
        """
        score = 50.0  # Default: medium

        # --- Increasing factors ---
        # Spectral entropy (higher = harder)
        specEnt = features.get('spectralEntropy', 0.5)
        score += (specEnt - 0.5) * 40.0  # 0.5->0, 1.0->+20

        # Low forecastability
        fc = features.get('forecastability', 0.5)
        score += (0.5 - fc) * 40.0  # 0.5->0, 0.0->+20

        # Nonlinearity (ApEn)
        apen = features.get('approximateEntropy', 0.5)
        score += min(apen * 7.5, 15.0)

        # Level shifts
        shifts = features.get('levelShiftCount', 0.0)
        score += min(shifts * 2.5, 10.0)

        # Signal-to-noise ratio (lower = harder)
        snr = features.get('signalToNoise', 1.0)
        snrPenalty = max(0.0, 15.0 - snr * 5.0)
        score += min(snrPenalty, 15.0)

        # Intermittency
        zeroRatio = features.get('zeroRatio', 0.0)
        score += zeroRatio * 10.0

        # Volatility clustering
        volCluster = features.get('volatilityClustering', 0.0)
        score += max(0.0, volCluster) * 10.0

        # --- Decreasing factors ---
        # Strong trend
        trendStr = features.get('trendStrength', 0.0)
        score -= trendStr * 10.0

        # Strong seasonality
        seasStr = features.get('seasonalStrength', 0.0)
        score -= seasStr * 10.0

        # High autocorrelation
        acf1 = abs(features.get('acf1', 0.0))
        score -= acf1 * 10.0

        # Clipping
        score = float(np.clip(score, 0, 100))

        # Grade
        if score < 25:
            level = 'easy'
        elif score < 50:
            level = 'medium'
        elif score < 75:
            level = 'hard'
        else:
            level = 'very_hard'

        return level, round(score, 1)

    # ==================================================================
    # Model Recommendation (Meta-Learning Rules)
    # ==================================================================

    def _recommendModels(self, features: Dict[str, float]) -> List[str]:
        """
        DNA feature-based model recommendation (meta-learning)

        Rule-based recommendation system. Each matching rule assigns scores
        to candidate models. Returns the top 5 models sorted by score (descending).

        Rules:
        - Strong trend + weak seasonality -> theta, dot, auto_arima
        - Strong seasonality + weak trend -> auto_ets, auto_ces, mstl
        - Strong trend + strong seasonality -> auto_ets, theta, auto_ces
        - High volatility -> garch, auto_ces, auto_arima
        - Intermittent demand -> croston, mean, naive
        - Stable -> auto_ces, auto_ets, mean
        - Nonlinear -> dot, auto_ces, auto_ets
        - Complex multiple seasonality -> tbats, mstl, auto_ces
        - High autocorrelation -> auto_arima, auto_ets, theta
        - Low autocorrelation -> auto_ces, mean, dot
        """
        scores: Dict[str, float] = {}

        trendStr = features.get('trendStrength', 0.0)
        seasStr = features.get('seasonalStrength', 0.0)
        volat = features.get('volatility', 0.0)
        zeroRatio = features.get('zeroRatio', 0.0)
        apen = features.get('approximateEntropy', 0.0)
        acf1 = abs(features.get('acf1', 0.0))
        multiSeas = features.get('multiSeasonalScore', 0.0)
        nonlinAcf = abs(features.get('nonlinearAutocorr', 0.0))
        volCluster = features.get('volatilityClustering', 0.0)
        fc = features.get('forecastability', 0.5)
        intermType = features.get('intermittencyType', 0.0)
        dataLen = features.get('length', 100.0)

        def _add(model: str, points: float) -> None:
            scores[model] = scores.get(model, 0.0) + points

        if trendStr > 0.5 and seasStr < 0.3:
            _add('theta', 3.0)
            _add('dot', 3.0)
            _add('auto_arima', 2.5)
            _add('auto_ces', 2.0)
            _add('rwd', 1.5)

        if seasStr > 0.5 and trendStr < 0.3:
            _add('auto_ets', 3.0)
            _add('auto_ces', 2.5)
            _add('seasonal_naive', 2.0)
            if dataLen >= 3 * 7:
                _add('mstl', 1.5)

        if trendStr > 0.4 and seasStr > 0.4:
            _add('auto_ets', 3.0)
            _add('theta', 2.5)
            _add('auto_ces', 2.5)
            _add('tbats', 1.5)
            if dataLen >= 3 * 7:
                _add('mstl', 1.0)

        if volat > 1.0 or volCluster > 0.3:
            _add('garch', 3.0)
            _add('auto_ces', 2.0)
            _add('auto_arima', 1.5)
            _add('window_avg', 1.0)

        if zeroRatio > 0.3 or intermType >= 1.0:
            _add('croston', 3.0)
            _add('mean', 1.5)
            _add('naive', 1.0)

        if fc > 0.7 and volat < 0.5 and trendStr < 0.2 and seasStr < 0.2:
            _add('auto_ces', 2.5)
            _add('auto_ets', 2.5)
            _add('mean', 2.0)
            _add('dot', 1.5)

        if apen > 1.0 or nonlinAcf > 0.3:
            _add('dot', 3.0)
            _add('auto_ces', 2.5)
            _add('auto_ets', 2.0)

        if multiSeas > 0.4 and dataLen >= 60:
            _add('tbats', 3.0)
            _add('mstl', 2.5)
            _add('auto_ces', 2.0)

        if acf1 > 0.7:
            _add('auto_arima', 2.5)
            _add('auto_ets', 2.0)
            _add('theta', 1.5)

        if acf1 < 0.2 and seasStr < 0.2:
            _add('auto_ces', 2.5)
            _add('mean', 2.0)
            _add('dot', 1.5)
            _add('window_avg', 1.0)

        if not scores:
            scores = {
                'auto_ets': 3.0,
                'auto_ces': 2.5,
                'theta': 2.5,
                'dot': 2.0,
                'auto_arima': 1.5
            }

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in ranked[:5]]

    # ==================================================================
    # Categorization
    # ==================================================================

    def _categorize(self, features: Dict[str, float]) -> str:
        """
        Time series category classification

        Categories:
        - 'trending': Trend-dominant
        - 'seasonal': Seasonality-dominant
        - 'stationary': Stable/stationary
        - 'intermittent': Intermittent demand
        - 'volatile': High volatility
        - 'complex': Complex (multiple characteristics mixed)
        """
        trendStr = features.get('trendStrength', 0.0)
        seasStr = features.get('seasonalStrength', 0.0)
        volat = features.get('volatility', 0.0)
        zeroRatio = features.get('zeroRatio', 0.0)
        apen = features.get('approximateEntropy', 0.0)
        acf1 = abs(features.get('acf1', 0.0))

        # Intermittency first
        if zeroRatio > 0.3:
            return 'intermittent'

        # High volatility
        if volat > 1.5 and trendStr < 0.3 and seasStr < 0.3:
            return 'volatile'

        # Complex
        complexScore = 0
        if trendStr > 0.3:
            complexScore += 1
        if seasStr > 0.3:
            complexScore += 1
        if volat > 0.8:
            complexScore += 1
        if apen > 1.0:
            complexScore += 1
        if complexScore >= 3:
            return 'complex'

        # Trend-dominant
        if trendStr > 0.5 and trendStr > seasStr:
            return 'trending'

        # Seasonality-dominant
        if seasStr > 0.4 and seasStr > trendStr:
            return 'seasonal'

        # Stationary
        return 'stationary'

    # ==================================================================
    # Summary Generation
    # ==================================================================

    def _generateSummary(
        self,
        features: Dict[str, float],
        difficulty: str,
        category: str
    ) -> str:
        """Generate natural language summary"""
        parts = []

        # Category
        catNames = {
            'trending': 'trending',
            'seasonal': 'seasonal',
            'stationary': 'stationary',
            'intermittent': 'intermittent',
            'volatile': 'volatile',
            'complex': 'complex'
        }
        catName = catNames.get(category, category)
        parts.append(f"This time series is of '{catName}' type.")

        # Trend
        trendStr = features.get('trendStrength', 0.0)
        trendDir = features.get('trendDirection', 0.0)
        if trendStr > 0.5:
            dirStr = 'upward' if trendDir > 0 else ('downward' if trendDir < 0 else 'flat')
            parts.append(f"A strong {dirStr} trend is observed (strength: {trendStr:.2f}).")
        elif trendStr > 0.2:
            parts.append(f"A weak trend is present (strength: {trendStr:.2f}).")

        # Seasonality
        seasStr = features.get('seasonalStrength', 0.0)
        if seasStr > 0.4:
            period = features.get('seasonalPeakPeriod', 0)
            parts.append(f"A clear seasonal pattern exists (strength: {seasStr:.2f}, period: {period:.0f}).")
        elif seasStr > 0.2:
            parts.append(f"Weak seasonality detected (strength: {seasStr:.2f}).")

        # Intermittency
        zeroRatio = features.get('zeroRatio', 0.0)
        if zeroRatio > 0.3:
            parts.append(f"{zeroRatio*100:.0f}% of data are zero values, indicating intermittent demand.")

        # Volatility
        volat = features.get('volatility', 0.0)
        if volat > 1.0:
            parts.append(f"High volatility observed (index: {volat:.2f}).")

        # Difficulty
        diffNames = {
            'easy': 'easy',
            'medium': 'medium',
            'hard': 'hard',
            'very_hard': 'very hard'
        }
        diffName = diffNames.get(difficulty, difficulty)
        score = features.get('difficultyScore', 50.0)
        parts.append(f"Forecast difficulty: {diffName}.")

        # Hurst
        hurst = features.get('hurstExponent', 0.5)
        if hurst > 0.6:
            parts.append(f"Long memory characteristics present (Hurst: {hurst:.2f}).")
        elif hurst < 0.4:
            parts.append(f"Mean-reverting tendency present (Hurst: {hurst:.2f}).")

        return ' '.join(parts)

    # ==================================================================
    # Similarity & Search
    # ==================================================================

    def similarity(self, profile1: DNAProfile, profile2: DNAProfile) -> float:
        """
        Cosine similarity between two DNA profiles (0.0~1.0)

        Constructs vectors using common feature keys, then computes cosine similarity.

        Parameters
        ----------
        profile1, profile2 : DNAProfile

        Returns
        -------
        float
            Cosine similarity (0.0~1.0, 1 = most similar)
        """
        commonKeys = sorted(
            set(profile1.features.keys()) & set(profile2.features.keys())
        )
        if not commonKeys:
            return 0.0

        vec1 = np.array([profile1.features[k] for k in commonKeys])
        vec2 = np.array([profile2.features[k] for k in commonKeys])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        cosine = float(np.dot(vec1, vec2) / (norm1 * norm2))
        # Map cosine similarity to 0~1 range (-1~1 -> 0~1)
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    def findSimilar(
        self,
        target: DNAProfile,
        database: List[DNAProfile],
        topK: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Search for most similar time series in a database

        Parameters
        ----------
        target : DNAProfile
            Target profile to search for
        database : List[DNAProfile]
            Database of profiles to search through
        topK : int
            Maximum number of results to return

        Returns
        -------
        List[Tuple[int, float]]
            [(index, similarity), ...] sorted by similarity descending
        """
        similarities = []
        for i, profile in enumerate(database):
            sim = self.similarity(target, profile)
            similarities.append((i, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topK]

    def analyzeBatch(
        self,
        seriesDict: Dict[str, np.ndarray],
        period: int = 1
    ) -> Dict[str, DNAProfile]:
        """
        Batch analysis of multiple time series

        Parameters
        ----------
        seriesDict : Dict[str, np.ndarray]
            {name: time series} dictionary
        period : int
            Common period

        Returns
        -------
        Dict[str, DNAProfile]
            {name: DNAProfile} result dictionary
        """
        results = {}
        for name, series in seriesDict.items():
            try:
                results[name] = self.analyze(np.asarray(series), period=period)
            except Exception as e:
                results[name] = DNAProfile(summary=f"Analysis failed: {str(e)}")
        return results

    # ==================================================================
    # Helper: ACF
    # ==================================================================

    def _acf(self, y: np.ndarray, maxLag: int) -> np.ndarray:
        """
        Autocorrelation Function (ACF)

        Computes normalized autocovariance from lag 0 to maxLag.
        """
        n = len(y)
        maxLag = min(maxLag, n - 1)
        if maxLag < 0:
            return np.array([1.0])

        yMean = np.mean(y)
        yDev = y - yMean
        var = np.sum(yDev ** 2)

        if var < 1e-10:
            result = np.zeros(maxLag + 1)
            result[0] = 1.0
            return result

        acfValues = np.zeros(maxLag + 1)
        acfValues[0] = 1.0
        for lag in range(1, maxLag + 1):
            cov = np.sum(yDev[:n - lag] * yDev[lag:])
            acfValues[lag] = cov / var

        return acfValues

    # ==================================================================
    # Helper: Periodogram
    # ==================================================================

    def _periodogram(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """FFT-based power spectral density estimation"""
        n = len(y)
        yDetrended = y - np.mean(y)

        if SCIPY_AVAILABLE:
            freqs, power = periodogram(yDetrended, fs=1.0)
        else:
            # numpy FFT fallback
            spectrum = np.fft.rfft(yDetrended)
            power = np.abs(spectrum) ** 2 / n
            freqs = np.fft.rfftfreq(n, d=1.0)

        # Remove DC component
        if len(freqs) > 1:
            return freqs[1:], power[1:]
        return freqs, power

    # ==================================================================
    # Helper: Harmonic check
    # ==================================================================

    def _isHarmonic(self, freq: float, fundamental: float, tolerance: float = 0.1) -> bool:
        """Check if frequency is an integer multiple (harmonic) of the fundamental"""
        if fundamental < 1e-10:
            return False
        ratio = freq / fundamental
        return abs(ratio - round(ratio)) < tolerance

    # ==================================================================
    # Helper: Hurst Exponent (R/S Analysis)
    # ==================================================================

    def _hurstExponent(self, y: np.ndarray) -> float:
        """
        Compute Hurst exponent via R/S analysis

        H > 0.5: Trend persistence (long memory)
        H = 0.5: Random walk
        H < 0.5: Mean reversion

        Parameters
        ----------
        y : np.ndarray

        Returns
        -------
        float
            Hurst exponent (0~1)
        """
        n = len(y)
        maxK = min(n // 2, 100)
        if maxK < 10:
            return 0.5

        rsValues = []
        ns = []

        step = max(1, (maxK - 10) // 20)
        for k in range(10, maxK + 1, step):
            rsList = []
            for start in range(0, n - k + 1, k):
                segment = y[start:start + k]
                mean = np.mean(segment)
                devs = np.cumsum(segment - mean)
                R = np.max(devs) - np.min(devs)
                S = np.std(segment, ddof=1)
                if S > 1e-10:
                    rsList.append(R / S)
            if rsList:
                rsValues.append(np.mean(rsList))
                ns.append(k)

        if len(rsValues) < 3:
            return 0.5

        logN = np.log(np.array(ns, dtype=np.float64))
        logRs = np.log(np.array(rsValues, dtype=np.float64))

        try:
            slope, _ = np.polyfit(logN, logRs, 1)
            return float(np.clip(slope, 0, 1))
        except Exception:
            return 0.5

    # ==================================================================
    # Helper: Approximate Entropy
    # ==================================================================

    def _approximateEntropy(
        self,
        y: np.ndarray,
        m: int = 2,
        r: float = 0.2
    ) -> float:
        """
        Approximate Entropy (ApEn)

        Measures regularity/complexity of a time series.
        Higher values indicate more complex/irregular patterns.

        Parameters
        ----------
        y : np.ndarray
        m : int
            Embedding dimension (default 2)
        r : float
            Tolerance (fraction of standard deviation, default 0.2)

        Returns
        -------
        float
            ApEn value (>= 0)
        """
        n = len(y)
        if n < m + 1:
            return 0.0

        yStd = float(np.std(y, ddof=1))
        tolerance = r * yStd if yStd > 1e-10 else r

        def _phi(dim: int) -> float:
            """Log matching probability of dim-dimensional patterns"""
            nPatterns = n - dim + 1
            if nPatterns <= 0:
                return 0.0

            # Construct patterns
            patterns = np.array([y[i:i + dim] for i in range(nPatterns)])

            counts = np.zeros(nPatterns)
            for i in range(nPatterns):
                # Match using Chebyshev distance
                dists = np.max(np.abs(patterns - patterns[i]), axis=1)
                counts[i] = np.sum(dists <= tolerance)

            # Including self-match (no bias correction, traditional ApEn)
            counts = counts / nPatterns
            counts = np.maximum(counts, 1e-10)  # prevent log(0)
            return float(np.mean(np.log(counts)))

        try:
            phi_m = _phi(m)
            phi_m1 = _phi(m + 1)
            return abs(phi_m - phi_m1)
        except Exception:
            return 0.0

    # ==================================================================
    # Helper: Spectral Entropy
    # ==================================================================

    def _spectralEntropy(self, y: np.ndarray) -> float:
        """
        Spectral Entropy

        Shannon entropy of the normalized power spectrum after FFT.
        Higher values indicate broad frequency distribution (complex/hard to predict).
        Lower values indicate energy concentrated in few frequencies (regular/easy to predict).

        Returns
        -------
        float
            Normalized spectral entropy (0~1)
        """
        n = len(y)
        if n < 4:
            return 0.5

        try:
            freqs, power = self._periodogram(y)
            if len(power) == 0:
                return 0.5

            # Normalize: convert to probability distribution
            totalPower = np.sum(power)
            if totalPower < 1e-10:
                return 0.5

            psd = power / totalPower
            psd = psd[psd > 1e-10]  # remove zeros

            # Shannon entropy
            entropy = -float(np.sum(psd * np.log(psd)))

            # Normalize by maximum entropy (uniform distribution)
            maxEntropy = np.log(len(psd))
            if maxEntropy < 1e-10:
                return 0.5

            return float(np.clip(entropy / maxEntropy, 0, 1))
        except Exception:
            return 0.5
