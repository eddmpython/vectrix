"""
Time Series Feature Extraction - Forecast DNA based

Extracts 65+ statistical features using pure numpy/scipy:
- Basic statistics (10): length, mean, std, min, max, median, skewness, kurtosis, cv, iqr
- Trend (5): trend_strength, trend_slope, trend_linearity, trend_curvature, trend_direction
- Seasonality (8): seasonality_strength, seasonal_period, seasonal_peak/trough, multiple seasonality, etc.
- Autocorrelation (8): ACF, PACF, Ljung-Box, ARCH-LM, etc.
- Nonlinearity (5): Hurst exponent, approximate/sample entropy, LZ complexity, nonlinearity
- Stability (5): stability, lumpiness, flat_spots, crossing_points, max_kl_shift
- Stationarity (4): diff_std_ratio, diff_mean_ratio, KPSS approx, PP approx
- Forecastability (5): forecastability, SNR, mean_absolute_change, etc.
- Intermittency (5): ADI, cv_squared, zero_proportion, etc.
- Volatility (5): GARCH alpha/beta, volatility_clustering, max_drawdown, etc.
- Miscellaneous (5): peaks, troughs, peak_to_trough_time, longest runs

Uses pure numpy/scipy only (pandas only in extractBatch)
"""

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from vectrix._core import (  # noqa: I001
        approximate_entropy as _rust_approximate_entropy,
        hurst_exponent as _rust_hurst_exponent,
        sample_entropy as _rust_sample_entropy,
    )
    _RUST_DNA = True
except ImportError:
    _RUST_DNA = False


class TSFeatureExtractor:
    """
    Time Series Feature Extractor

    Extracts 65+ statistical features to compose the time series "DNA".
    Comprehensive feature extraction at the Kats TSFeature level.

    Examples
    --------
    >>> extractor = TSFeatureExtractor()
    >>> y = np.random.randn(200)
    >>> features = extractor.extract(y, period=7)
    >>> print(len(features))  # 65+
    """

    def extract(self, y: np.ndarray, period: int = 1) -> Dict[str, float]:
        """
        Extract all features from a time series

        Parameters
        ----------
        y : np.ndarray
            Time series data (1-dimensional)
        period : int
            Seasonal period (1 if non-seasonal)

        Returns
        -------
        Dict[str, float]
            Feature name -> value dictionary (65+ entries)
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y = y[~np.isnan(y)]
        n = len(y)

        if n < 4:
            return self._emptyFeatures()

        features = {}

        # Basic statistics (10)
        features.update(self._basicStats(y))

        # Trend (5)
        features.update(self._trendFeatures(y))

        # Seasonality (8)
        features.update(self._seasonalityFeatures(y, period))

        # Autocorrelation (8)
        features.update(self._autocorrelationFeatures(y, period))

        # Nonlinearity (5)
        features.update(self._nonlinearityFeatures(y))

        # Stability (5)
        features.update(self._stabilityFeatures(y, period))

        # Stationarity (4)
        features.update(self._stationarityFeatures(y))

        # Forecastability (5)
        features.update(self._forecastabilityFeatures(y, period))

        # Intermittency (5)
        features.update(self._intermittencyFeatures(y))

        # Volatility (5)
        features.update(self._volatilityFeatures(y))

        # Miscellaneous (5)
        features.update(self._miscFeatures(y))

        return features

    def extractBatch(
        self,
        seriesDict: Dict[str, np.ndarray],
        period: int = 1
    ):
        """
        Batch-extract features from multiple time series and return a DataFrame

        Parameters
        ----------
        seriesDict : Dict[str, np.ndarray]
            Series name -> data dictionary
        period : int
            Seasonal period

        Returns
        -------
        pd.DataFrame
            Rows: series names, Columns: features
        """
        import pandas as pd

        results = {}
        for name, y in seriesDict.items():
            try:
                results[name] = self.extract(y, period)
            except Exception:
                results[name] = self._emptyFeatures()

        return pd.DataFrame(results).T

    def similarity(
        self,
        features1: Dict[str, float],
        features2: Dict[str, float]
    ) -> float:
        """
        Compute cosine similarity between two feature vectors

        Parameters
        ----------
        features1 : Dict[str, float]
            First feature dictionary
        features2 : Dict[str, float]
            Second feature dictionary

        Returns
        -------
        float
            Cosine similarity (-1 to 1)
        """
        # Use only common keys
        commonKeys = sorted(set(features1.keys()) & set(features2.keys()))
        if len(commonKeys) == 0:
            return 0.0

        v1 = np.array([features1[k] for k in commonKeys], dtype=np.float64)
        v2 = np.array([features2[k] for k in commonKeys], dtype=np.float64)

        # Handle NaN/Inf
        validMask = np.isfinite(v1) & np.isfinite(v2)
        v1 = v1[validMask]
        v2 = v2[validMask]

        if len(v1) == 0:
            return 0.0

        # Normalization (z-score)
        combined = np.column_stack([v1, v2])
        mu = np.mean(combined, axis=1, keepdims=True)
        sigma = np.std(combined, axis=1, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        v1n = (v1 - mu.ravel()) / sigma.ravel()
        v2n = (v2 - mu.ravel()) / sigma.ravel()

        # Cosine similarity
        dot = np.dot(v1n, v2n)
        norm1 = np.linalg.norm(v1n)
        norm2 = np.linalg.norm(v2n)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.clip(dot / (norm1 * norm2), -1.0, 1.0))

    def fingerprint(self, features: Dict[str, float]) -> str:
        """
        Generate a hash-based fingerprint of the feature dictionary (8-char hex)

        Time series with identical feature patterns will have the same fingerprint.

        Parameters
        ----------
        features : Dict[str, float]
            Feature dictionary

        Returns
        -------
        str
            8-character hex string
        """
        # Sort keys and quantize before hashing
        sortedKeys = sorted(features.keys())
        values = []
        for k in sortedKeys:
            v = features[k]
            if np.isfinite(v):
                # Quantize to 3 decimal places (ignore minor noise)
                values.append(f"{k}:{v:.3f}")
            else:
                values.append(f"{k}:nan")

        hashInput = "|".join(values).encode('utf-8')
        return hashlib.md5(hashInput).hexdigest()[:8]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Basic Statistics (10)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _basicStats(self, y: np.ndarray) -> Dict[str, float]:
        """Basic statistical features (10)"""
        n = len(y)
        mean = float(np.mean(y))
        std = float(np.std(y, ddof=1)) if n > 1 else 0.0

        try:
            from scipy.stats import kurtosis, skew
            skewness = float(skew(y, bias=False))
            kurt = float(kurtosis(y, bias=False, fisher=True))
        except ImportError:
            # Pure numpy fallback
            skewness = self._computeSkewness(y)
            kurt = self._computeKurtosis(y)

        q1, q3 = np.percentile(y, [25, 75])

        return {
            'length': float(n),
            'mean': mean,
            'std': std,
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'median': float(np.median(y)),
            'skewness': skewness,
            'kurtosis': kurt,
            'cv': std / abs(mean) if abs(mean) > 1e-10 else 0.0,
            'iqr': float(q3 - q1),
        }

    def _computeSkewness(self, y: np.ndarray) -> float:
        """Compute skewness using pure numpy"""
        n = len(y)
        if n < 3:
            return 0.0
        m = np.mean(y)
        s = np.std(y, ddof=1)
        if s < 1e-10:
            return 0.0
        return float(n / ((n - 1) * (n - 2)) * np.sum(((y - m) / s) ** 3))

    def _computeKurtosis(self, y: np.ndarray) -> float:
        """Compute kurtosis using pure numpy (Fisher, excess)"""
        n = len(y)
        if n < 4:
            return 0.0
        m = np.mean(y)
        s = np.std(y, ddof=1)
        if s < 1e-10:
            return 0.0
        m4 = np.mean((y - m) ** 4)
        return float(m4 / (s ** 4) - 3.0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Trend (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _trendFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """Trend-related features (5)"""
        n = len(y)
        x = np.arange(n, dtype=np.float64)

        # Linear regression
        slope, intercept = self._linearRegression(x, y)
        yHat = intercept + slope * x
        residual = y - yHat

        # Trend strength: 1 - Var(residual) / Var(y)
        varY = np.var(y)
        varResid = np.var(residual)
        trendStrength = max(0, 1 - varResid / max(varY, 1e-10))

        # Trend linearity: R^2
        sst = max(np.sum((y - np.mean(y)) ** 2), 1e-10)
        sse = np.sum(residual ** 2)
        linearity = max(0, 1 - sse / sst)

        # Curvature: quadratic coefficient
        curvature = 0.0
        if n >= 5:
            try:
                coeffs = np.polyfit(x, y, 2)
                curvature = float(coeffs[0])
            except Exception:
                pass

        # Trend direction: sign of slope (positive=up, negative=down, 0=flat)
        if abs(slope) < 1e-10 * max(np.std(y), 1e-10):
            direction = 0.0
        else:
            direction = 1.0 if slope > 0 else -1.0

        return {
            'trend_strength': float(trendStrength),
            'trend_slope': float(slope),
            'trend_linearity': float(linearity),
            'trend_curvature': float(curvature),
            'trend_direction': float(direction),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Seasonality (8)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _seasonalityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """Seasonality-related features (8)"""
        n = len(y)

        # FFT-based spectral analysis
        fft = np.fft.rfft(y - np.mean(y))
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n)

        # Normalized power spectrum
        totalPower = max(np.sum(power[1:]), 1e-10)  # Exclude DC
        normPower = power[1:] / totalPower

        # Dominant frequency
        if len(normPower) > 0:
            dominantIdx = np.argmax(normPower)
            dominantFreq = float(freqs[dominantIdx + 1])
        else:
            dominantFreq = 0.0

        # Spectral entropy
        spectralEntropy = self._spectralEntropy(normPower)

        # Seasonality strength (when period > 1)
        seasonalityStrength = 0.0
        seasonalPeak = 0.0
        seasonalTrough = 0.0
        seasonalAmplitude = 0.0
        hasMultipleSeasonality = 0.0
        detectedPeriod = float(period)

        if period > 1 and n >= 2 * period:
            # Estimate seasonality strength via seasonal decomposition
            trend = self._movingAverage(y, period)
            detrended = y - trend

            # Seasonal means
            seasonalMeans = np.zeros(period)
            for i in range(period):
                vals = detrended[i::period]
                seasonalMeans[i] = np.mean(vals)
            seasonalMeans -= np.mean(seasonalMeans)

            # Seasonality strength: 1 - Var(remainder) / Var(detrended)
            seasonal = np.tile(seasonalMeans, n // period + 1)[:n]
            remainder = detrended - seasonal
            varDetrended = max(np.var(detrended), 1e-10)
            varRemainder = np.var(remainder)
            seasonalityStrength = max(0, 1 - varRemainder / varDetrended)

            # Seasonal peak/trough position (relative position within cycle)
            seasonalPeak = float(np.argmax(seasonalMeans)) / period
            seasonalTrough = float(np.argmin(seasonalMeans)) / period
            seasonalAmplitude = float(np.max(seasonalMeans) - np.min(seasonalMeans))

            # Multiple seasonality detection (number of major peaks in spectrum)
            nPeaksSpectrum = self._countSpectralPeaks(normPower, threshold=0.05)
            hasMultipleSeasonality = 1.0 if nPeaksSpectrum > 1 else 0.0

        # Automatic period detection (FFT-based)
        if n > 4:
            try:
                # Estimate period from peak frequency
                if dominantFreq > 0:
                    autoPeriod = 1.0 / dominantFreq
                    if 2 <= autoPeriod <= n / 2:
                        detectedPeriod = autoPeriod
            except Exception:
                pass

        return {
            'seasonality_strength': float(seasonalityStrength),
            'seasonal_period': float(detectedPeriod),
            'seasonal_peak_position': float(seasonalPeak),
            'seasonal_trough_position': float(seasonalTrough),
            'has_multiple_seasonality': float(hasMultipleSeasonality),
            'dominant_frequency': float(dominantFreq),
            'spectral_entropy': float(spectralEntropy),
            'seasonal_amplitude': float(seasonalAmplitude),
        }

    def _spectralEntropy(self, normPower: np.ndarray) -> float:
        """Compute normalized spectral entropy"""
        p = normPower[normPower > 0]
        if len(p) == 0:
            return 0.0
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log2(p + 1e-20))
        maxEntropy = np.log2(len(p)) if len(p) > 1 else 1.0
        return float(entropy / max(maxEntropy, 1e-10))

    def _countSpectralPeaks(self, normPower: np.ndarray, threshold: float = 0.05) -> int:
        """Count the number of major peaks in the spectrum"""
        count = 0
        for i in range(1, len(normPower) - 1):
            if (normPower[i] > threshold and
                    normPower[i] > normPower[i - 1] and
                    normPower[i] >= normPower[i + 1]):
                count += 1
        return count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Autocorrelation (8)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _autocorrelationFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """Autocorrelation-related features (8)"""
        n = len(y)

        # ACF
        acf = self._acf(y, nlags=max(period * 2, 10))
        acfLag1 = acf[1] if len(acf) > 1 else 0.0
        acfLag2 = acf[2] if len(acf) > 2 else 0.0

        # PACF (Durbin-Levinson algorithm)
        pacf = self._pacf(y, nlags=max(period * 2, 10))
        pacfLag1 = pacf[1] if len(pacf) > 1 else 0.0
        pacfLag2 = pacf[2] if len(pacf) > 2 else 0.0

        # ACF decay rate
        acfDecayRate = self._acfDecayRate(acf)

        # Ljung-Box statistic
        ljungBox = self._ljungBoxStat(y, acf, nlags=min(10, n // 5))

        # ARCH-LM statistic
        archLm = self._archLmStat(y)

        # Turning points rate
        turningPointsRate = self._turningPointsRate(y)

        return {
            'acf_lag1': float(acfLag1),
            'acf_lag2': float(acfLag2),
            'pacf_lag1': float(pacfLag1),
            'pacf_lag2': float(pacfLag2),
            'acf_decay_rate': float(acfDecayRate),
            'ljung_box_stat': float(ljungBox),
            'arch_lm_stat': float(archLm),
            'turning_points_rate': float(turningPointsRate),
        }

    def _acf(self, y: np.ndarray, nlags: int = 20) -> np.ndarray:
        """Compute autocorrelation function"""
        n = len(y)
        nlags = min(nlags, n - 1)
        yDemeaned = y - np.mean(y)
        c0 = np.dot(yDemeaned, yDemeaned) / n
        if c0 < 1e-10:
            return np.zeros(nlags + 1)

        acf = np.zeros(nlags + 1)
        acf[0] = 1.0
        for k in range(1, nlags + 1):
            acf[k] = np.dot(yDemeaned[:n - k], yDemeaned[k:]) / (n * c0)

        return acf

    def _pacf(self, y: np.ndarray, nlags: int = 20) -> np.ndarray:
        """
        Compute partial autocorrelation function (Durbin-Levinson algorithm)
        """
        acf = self._acf(y, nlags)
        n = len(acf) - 1

        pacf = np.zeros(n + 1)
        pacf[0] = 1.0

        if n == 0:
            return pacf

        # Durbin-Levinson
        phi = np.zeros((n + 1, n + 1))
        phi[1, 1] = acf[1]
        pacf[1] = acf[1]

        for k in range(2, n + 1):
            num = acf[k]
            for j in range(1, k):
                num -= phi[k - 1, j] * acf[k - j]

            den = 1.0
            for j in range(1, k):
                den -= phi[k - 1, j] * acf[j]

            if abs(den) < 1e-10:
                break

            phi[k, k] = num / den
            pacf[k] = phi[k, k]

            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]

        return pacf

    def _acfDecayRate(self, acf: np.ndarray) -> float:
        """ACF decay rate (exponential decay fitting)"""
        nlags = len(acf) - 1
        if nlags < 2:
            return 0.0

        absAcf = np.abs(acf[1:])
        if np.all(absAcf < 1e-10):
            return 1.0  # Immediate decay

        # Linear regression after log transform
        valid = absAcf > 1e-10
        if np.sum(valid) < 2:
            return 0.5

        x = np.arange(1, nlags + 1)[valid]
        logAcf = np.log(absAcf[valid])

        slope, _ = self._linearRegression(x.astype(float), logAcf)
        return float(np.clip(-slope, 0, 2))

    def _ljungBoxStat(self, y: np.ndarray, acf: np.ndarray, nlags: int = 10) -> float:
        """Ljung-Box Q statistic"""
        n = len(y)
        nlags = min(nlags, len(acf) - 1, n - 1)
        if nlags < 1:
            return 0.0

        Q = 0.0
        for k in range(1, nlags + 1):
            Q += acf[k] ** 2 / (n - k)

        return float(n * (n + 2) * Q)

    def _archLmStat(self, y: np.ndarray) -> float:
        """ARCH-LM statistic (autocorrelation of squared residuals)"""
        n = len(y)
        if n < 10:
            return 0.0

        residuals = y - np.mean(y)
        residSq = residuals ** 2

        acfResidSq = self._acf(residSq, nlags=5)
        # LM stat ~ n * R^2 (regression of squared residuals)
        rSq = np.sum(acfResidSq[1:6] ** 2)
        return float(n * rSq)

    def _turningPointsRate(self, y: np.ndarray) -> float:
        """Turning points rate (proportion of direction changes)"""
        n = len(y)
        if n < 3:
            return 0.0

        diff = np.diff(y)
        signs = np.sign(diff)
        turningPoints = np.sum(signs[:-1] != signs[1:])

        return float(turningPoints / (n - 2))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Nonlinearity (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _nonlinearityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """Nonlinearity-related features (5)"""
        return {
            'hurst_exponent': self._hurstExponent(y),
            'approximate_entropy': self._approximateEntropy(y),
            'sample_entropy': self._sampleEntropy(y),
            'lempel_ziv_complexity': self._lempelZivComplexity(y),
            'nonlinearity': self._nonlinearityStat(y),
        }

    def _hurstExponent(self, y: np.ndarray, maxLag: int = 20) -> float:
        """
        Compute Hurst exponent (R/S analysis)

        H > 0.5: persistent (trending)
        H = 0.5: random walk
        H < 0.5: anti-persistent (mean-reverting)

        Vectorized: each lag's R/S computed with array slicing instead of Python loops.
        """
        n = len(y)
        if n < 20:
            return 0.5

        if _RUST_DNA:
            try:
                return float(np.clip(_rust_hurst_exponent(y, maxLag), 0, 1))
            except Exception:
                pass

        try:
            lags = range(2, min(maxLag + 1, n // 4))
            rsValues = []
            lagValues = []

            for lag in lags:
                nBlocks = (n - lag + 1) // lag
                if nBlocks < 1:
                    continue
                blocks = y[:nBlocks * lag].reshape(nBlocks, lag)
                means = blocks.mean(axis=1, keepdims=True)
                cumDev = np.cumsum(blocks - means, axis=1)
                R = cumDev.max(axis=1) - cumDev.min(axis=1)
                S = blocks.std(axis=1, ddof=1)
                valid = S > 1e-10
                if valid.any():
                    rsValues.append(np.mean(R[valid] / S[valid]))
                    lagValues.append(lag)

            if len(lagValues) < 2:
                return 0.5

            logLags = np.log(np.array(lagValues, dtype=float))
            logRS = np.log(np.array(rsValues, dtype=float))

            slope, _ = self._linearRegression(logLags, logRS)
            return float(np.clip(slope, 0, 1))
        except Exception:
            return 0.5

    def _approximateEntropy(self, y: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        """
        Approximate Entropy

        Measures pattern complexity. Higher values indicate more irregularity.
        Vectorized: O(n*m) memory, O(n^2*m) ops but all in numpy C loops.
        """
        n = len(y)
        if n < 10:
            return 0.0

        if r is None:
            r = 0.2 * np.std(y, ddof=1)
        if r < 1e-10:
            return 0.0

        if _RUST_DNA:
            try:
                return float(max(_rust_approximate_entropy(y, m, r), 0))
            except Exception:
                pass

        try:
            def phi(m_val):
                nT = n - m_val + 1
                templates = np.lib.stride_tricks.sliding_window_view(y, m_val)[:nT]
                dists = np.abs(templates[:, np.newaxis, :] - templates[np.newaxis, :, :]).max(axis=2)
                counts = (dists <= r).sum(axis=1) / nT
                return np.mean(np.log(counts + 1e-20))

            return float(max(phi(m) - phi(m + 1), 0))
        except Exception:
            return 0.0

    def _sampleEntropy(self, y: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        """
        Sample Entropy

        Bias-corrected version of approximate entropy.
        Vectorized: upper-triangle pairwise Chebyshev distance via numpy broadcasting.
        """
        n = len(y)
        if n < 10:
            return 0.0

        if r is None:
            r = 0.2 * np.std(y, ddof=1)
        if r < 1e-10:
            return 0.0

        if _RUST_DNA:
            try:
                result = _rust_sample_entropy(y, m, r)
                if result == 0.0 or np.isfinite(result):
                    return float(result)
            except Exception:
                pass

        try:
            def countMatches(m_val):
                nT = n - m_val
                templates = np.lib.stride_tricks.sliding_window_view(y, m_val)[:nT]
                dists = np.abs(templates[:, np.newaxis, :] - templates[np.newaxis, :, :]).max(axis=2)
                iu = np.triu_indices(nT, k=1)
                return int((dists[iu] <= r).sum())

            A = countMatches(m + 1)
            B = countMatches(m)

            if B == 0 or A == 0:
                return 0.0

            return float(-np.log(A / B))
        except Exception:
            return 0.0

    def _lempelZivComplexity(self, y: np.ndarray) -> float:
        """
        Lempel-Ziv Complexity

        Computes complexity via LZ76 algorithm after binarization.
        Normalization: c(n) / (n / log2(n))
        """
        n = len(y)
        if n < 4:
            return 0.0

        try:
            # Binarize (median threshold)
            binary = (y > np.median(y)).astype(int)
            binaryStr = ''.join(map(str, binary))

            # LZ76 complexity
            complexity = 1
            l = 1
            k = 1
            kMax = 1

            while l + k <= n:
                if binaryStr[l:l + k] in binaryStr[:l + k - 1]:
                    k += 1
                    if k > kMax:
                        kMax = k
                else:
                    complexity += 1
                    l += kMax if kMax > k else k
                    k = 1
                    kMax = 1

            # Normalization
            logN = np.log2(max(n, 2))
            normalized = complexity * logN / max(n, 1)

            return float(np.clip(normalized, 0, 5))
        except Exception:
            return 0.0

    def _nonlinearityStat(self, y: np.ndarray) -> float:
        """
        Nonlinearity statistic

        Approximates the third moment of residuals and BDS-like statistic.
        """
        n = len(y)
        if n < 10:
            return 0.0

        try:
            # Remove linear component (AR(1) residuals)
            yLag = y[:-1]
            yCurrent = y[1:]
            slope, intercept = self._linearRegression(yLag, yCurrent)
            residuals = yCurrent - (intercept + slope * yLag)

            # Nonlinearity: correlation between squared residuals and raw residuals
            residSq = residuals ** 2
            acfResidSq = self._acf(residSq, nlags=3)
            nonlinearity = np.mean(np.abs(acfResidSq[1:]))

            return float(nonlinearity)
        except Exception:
            return 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stability (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _stabilityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """Stability-related features (5)"""
        n = len(y)
        windowSize = max(period, 10) if period > 1 else max(n // 10, 5)

        return {
            'stability': self._stability(y, windowSize),
            'lumpiness': self._lumpiness(y, windowSize),
            'flat_spots': self._flatSpots(y),
            'crossing_points': self._crossingPoints(y),
            'max_kl_shift': self._maxKLShift(y, windowSize),
        }

    def _stability(self, y: np.ndarray, windowSize: int) -> float:
        """Stability: variance of moving window means"""
        n = len(y)
        if n < windowSize * 2:
            return 0.0

        nWindows = n // windowSize
        means = np.array([np.mean(y[i * windowSize:(i + 1) * windowSize])
                          for i in range(nWindows)])

        return float(np.var(means))

    def _lumpiness(self, y: np.ndarray, windowSize: int) -> float:
        """Lumpiness: variance of moving window variances"""
        n = len(y)
        if n < windowSize * 2:
            return 0.0

        nWindows = n // windowSize
        variances = np.array([np.var(y[i * windowSize:(i + 1) * windowSize])
                              for i in range(nWindows)])

        return float(np.var(variances))

    def _flatSpots(self, y: np.ndarray) -> float:
        """Flat spots: maximum run length of identical values"""
        n = len(y)
        if n < 2:
            return 0.0

        # Histogram bin based (10 bins)
        try:
            bins = np.histogram_bin_edges(y, bins=10)
            digitized = np.digitize(y, bins)
        except Exception:
            digitized = np.round(y, 2)

        maxRun = 1
        currentRun = 1
        for i in range(1, n):
            if digitized[i] == digitized[i - 1]:
                currentRun += 1
                if currentRun > maxRun:
                    maxRun = currentRun
            else:
                currentRun = 1

        return float(maxRun)

    def _crossingPoints(self, y: np.ndarray) -> float:
        """Crossing points: number of mean crossings (normalized)"""
        n = len(y)
        if n < 3:
            return 0.0

        mean = np.mean(y)
        aboveMean = y > mean
        crossings = np.sum(aboveMean[:-1] != aboveMean[1:])

        return float(crossings / (n - 1))

    def _maxKLShift(self, y: np.ndarray, windowSize: int) -> float:
        """Max KL divergence shift: maximum distributional difference between adjacent windows"""
        n = len(y)
        if n < windowSize * 3:
            return 0.0

        nWindows = n // windowSize
        if nWindows < 2:
            return 0.0

        maxKL = 0.0
        for i in range(nWindows - 1):
            w1 = y[i * windowSize:(i + 1) * windowSize]
            w2 = y[(i + 1) * windowSize:(i + 2) * windowSize]

            # Gaussian-approximated KL divergence
            m1, s1 = np.mean(w1), max(np.std(w1), 1e-10)
            m2, s2 = np.mean(w2), max(np.std(w2), 1e-10)

            kl = np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5
            maxKL = max(maxKL, abs(kl))

        return float(maxKL)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stationarity (4)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _stationarityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """Stationarity-related features (4)"""
        n = len(y)
        diff = np.diff(y)

        # Pre/post differencing ratios
        stdRatio = np.std(diff) / max(np.std(y), 1e-10) if n > 1 else 1.0
        meanRatio = abs(np.mean(diff)) / max(abs(np.mean(y)), 1e-10) if n > 1 else 0.0

        return {
            'diff_std_ratio': float(stdRatio),
            'diff_mean_ratio': float(np.clip(meanRatio, 0, 100)),
            'unitroot_kpss_approx': self._kpssApprox(y),
            'unitroot_pp_approx': self._ppApprox(y),
        }

    def _kpssApprox(self, y: np.ndarray) -> float:
        """
        Simplified KPSS statistic

        H0: level stationary (trend-stationary)
        Large values = non-stationary
        """
        n = len(y)
        if n < 10:
            return 0.0

        try:
            # Detrend
            x = np.arange(n, dtype=np.float64)
            slope, intercept = self._linearRegression(x, y)
            residuals = y - (intercept + slope * x)

            # Cumulative sum
            cumResid = np.cumsum(residuals)
            S2 = np.sum(cumResid ** 2) / (n ** 2)

            # Long-run variance estimation (Bartlett kernel)
            nlags = min(int(np.sqrt(n)), n // 3)
            gamma0 = np.sum(residuals ** 2) / n
            longRunVar = gamma0

            for k in range(1, nlags + 1):
                weight = 1.0 - k / (nlags + 1)
                gammaK = np.sum(residuals[:n - k] * residuals[k:]) / n
                longRunVar += 2 * weight * gammaK

            if longRunVar < 1e-10:
                return 0.0

            return float(S2 / longRunVar)
        except Exception:
            return 0.0

    def _ppApprox(self, y: np.ndarray) -> float:
        """
        Simplified Phillips-Perron statistic

        H0: unit root (non-stationary)
        Large negative = stationary
        """
        n = len(y)
        if n < 10:
            return 0.0

        try:
            yLag = y[:-1]
            yDiff = y[1:]

            slope, intercept = self._linearRegression(yLag, yDiff)
            rho = slope
            residuals = yDiff - (intercept + rho * yLag)

            sigma2 = np.sum(residuals ** 2) / (n - 2)

            # Long-run variance correction
            nlags = min(int(n ** (1 / 3)), n // 4)
            gamma0 = sigma2
            longRunVar = gamma0
            for k in range(1, nlags + 1):
                if k < len(residuals):
                    weight = 1.0 - k / (nlags + 1)
                    gammaK = np.sum(residuals[:len(residuals) - k] * residuals[k:]) / (n - 2)
                    longRunVar += 2 * weight * gammaK

            # PP t-stat approximation
            seLag = np.sqrt(sigma2 / max(np.sum((yLag - np.mean(yLag)) ** 2), 1e-10))
            tStat = (rho - 1.0) / max(seLag, 1e-10)

            # Correction term
            correction = (longRunVar - gamma0) / (2 * max(seLag, 1e-10) *
                         np.sqrt(max(np.sum((yLag - np.mean(yLag)) ** 2), 1e-10)))
            ppStat = tStat - correction

            return float(ppStat)
        except Exception:
            return 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Forecastability (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _forecastabilityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """Forecastability-related features (5)"""
        n = len(y)

        # Forecastability based on spectral entropy (1 - entropy)
        fft = np.fft.rfft(y - np.mean(y))
        power = np.abs(fft) ** 2
        totalPower = max(np.sum(power[1:]), 1e-10)
        normPower = power[1:] / totalPower
        spectralEntropy = self._spectralEntropy(normPower)
        forecastability = 1.0 - spectralEntropy

        # Signal-to-noise ratio
        signalPower = np.var(self._movingAverage(y, max(period, 3)))
        noisePower = max(np.var(y - self._movingAverage(y, max(period, 3))), 1e-10)
        snr = signalPower / noisePower

        # Mean absolute change
        meanAbsChange = float(np.mean(np.abs(np.diff(y)))) if n > 1 else 0.0

        # Mean second derivative
        if n >= 3:
            secondDiff = np.diff(y, n=2)
            meanSecondDeriv = float(np.mean(np.abs(secondDiff)))
        else:
            meanSecondDeriv = 0.0

        # Percentage of zeros
        pctZeros = float(np.sum(np.abs(y) < 1e-10) / n)

        return {
            'forecastability': float(forecastability),
            'signal_to_noise': float(snr),
            'mean_absolute_change': meanAbsChange,
            'mean_second_derivative': meanSecondDeriv,
            'percentage_zeros': pctZeros,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Intermittency (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _intermittencyFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """Intermittency-related features (5) - characterizes intermittent demand patterns"""
        n = len(y)
        zeroProportion = float(np.sum(np.abs(y) < 1e-10) / n)

        # Average Demand Interval (ADI)
        nonZeroIdx = np.where(np.abs(y) > 1e-10)[0]
        if len(nonZeroIdx) < 2:
            adi = float(n)
            demandIntervalCv = 0.0
        else:
            intervals = np.diff(nonZeroIdx)
            adi = float(np.mean(intervals))
            demandIntervalCv = float(np.std(intervals) / max(np.mean(intervals), 1e-10))

        # Non-zero demand size
        nonZeroValues = y[np.abs(y) > 1e-10]
        if len(nonZeroValues) > 0:
            demandSizeMean = float(np.mean(nonZeroValues))
            cvSquared = float((np.std(nonZeroValues) / max(np.mean(np.abs(nonZeroValues)), 1e-10)) ** 2)
        else:
            demandSizeMean = 0.0
            cvSquared = 0.0

        return {
            'adi': adi,
            'cv_squared': cvSquared,
            'zero_proportion': zeroProportion,
            'demand_size_mean': demandSizeMean,
            'demand_interval_cv': demandIntervalCv,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Volatility (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _volatilityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """Volatility-related features (5)"""
        n = len(y)

        # GARCH(1,1) parameter approximate estimation
        garchAlpha, garchBeta = self._estimateGarchParams(y)

        # Volatility clustering: ACF(1) of absolute returns
        absReturns = np.abs(np.diff(y))
        if len(absReturns) > 2:
            acfAbs = self._acf(absReturns, nlags=5)
            volClustering = float(acfAbs[1]) if len(acfAbs) > 1 else 0.0
        else:
            volClustering = 0.0

        # Max drawdown
        maxDrawdown = self._maxDrawdown(y)

        # Conditional heteroscedasticity (correlation between squared and lagged squared residuals)
        condHeterosced = self._conditionalHeteroscedasticity(y)

        return {
            'garch_alpha': float(garchAlpha),
            'garch_beta': float(garchBeta),
            'volatility_clustering': float(volClustering),
            'max_drawdown': float(maxDrawdown),
            'conditional_heteroscedasticity': float(condHeterosced),
        }

    def _estimateGarchParams(self, y: np.ndarray) -> Tuple[float, float]:
        """GARCH(1,1) parameter moment estimation (fast approximation)"""
        n = len(y)
        if n < 20:
            return 0.1, 0.8

        try:
            residuals = y - np.mean(y)
            residSq = residuals ** 2

            # Moment estimation based on autocorrelation
            acfSq = self._acf(residSq, nlags=5)
            rho1 = acfSq[1] if len(acfSq) > 1 else 0.0
            rho2 = acfSq[2] if len(acfSq) > 2 else 0.0

            # alpha + beta approximation
            persistence = max(0, min(rho1, 0.999))

            # alpha: estimated from the relation rho1 - beta * rho1 ~ rho2
            if abs(rho1) > 1e-10:
                beta = max(0, min((rho2 / rho1), 0.99))
            else:
                beta = 0.8

            alpha = max(0, min(persistence - beta, 0.5))

            return alpha, beta
        except Exception:
            return 0.1, 0.8

    def _maxDrawdown(self, y: np.ndarray) -> float:
        """Maximum drawdown (peak-to-trough)"""
        n = len(y)
        if n < 2:
            return 0.0

        cumMax = np.maximum.accumulate(y)
        drawdown = cumMax - y
        rangeY = max(np.max(y) - np.min(y), 1e-10)

        return float(np.max(drawdown) / rangeY)

    def _conditionalHeteroscedasticity(self, y: np.ndarray) -> float:
        """Measure conditional heteroscedasticity"""
        n = len(y)
        if n < 10:
            return 0.0

        try:
            residuals = y - np.mean(y)
            residSq = residuals ** 2

            # AR(1) R^2 of squared residuals
            xLag = residSq[:-1]
            yTarget = residSq[1:]
            slope, intercept = self._linearRegression(xLag, yTarget)
            yHat = intercept + slope * xLag
            ss_res = np.sum((yTarget - yHat) ** 2)
            ss_tot = np.sum((yTarget - np.mean(yTarget)) ** 2)

            rSquared = 1 - ss_res / max(ss_tot, 1e-10)
            return float(np.clip(rSquared, 0, 1))
        except Exception:
            return 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Miscellaneous (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _miscFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """Miscellaneous features (5)"""
        n = len(y)

        # Peak/trough detection
        nPeaks, nTroughs, peakToTroughTime = self._peakTroughStats(y)

        # Longest increasing/decreasing runs
        longestInc, longestDec = self._longestRuns(y)

        return {
            'n_peaks': float(nPeaks),
            'n_troughs': float(nTroughs),
            'peak_to_trough_time': float(peakToTroughTime),
            'longest_increasing_run': float(longestInc),
            'longest_decreasing_run': float(longestDec),
        }

    def _peakTroughStats(self, y: np.ndarray) -> Tuple[int, int, float]:
        """Peak/trough statistics"""
        n = len(y)
        if n < 3:
            return 0, 0, 0.0

        peaks = []
        troughs = []

        for i in range(1, n - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
                peaks.append(i)
            elif y[i] < y[i - 1] and y[i] < y[i + 1]:
                troughs.append(i)

        nPeaks = len(peaks)
        nTroughs = len(troughs)

        # Average time between peak and trough
        peakToTroughTime = 0.0
        if nPeaks > 0 and nTroughs > 0:
            allPoints = sorted([(p, 'peak') for p in peaks] + [(t, 'trough') for t in troughs])
            times = []
            for i in range(len(allPoints) - 1):
                if allPoints[i][1] != allPoints[i + 1][1]:
                    times.append(allPoints[i + 1][0] - allPoints[i][0])
            if len(times) > 0:
                peakToTroughTime = np.mean(times)

        return nPeaks, nTroughs, peakToTroughTime

    def _longestRuns(self, y: np.ndarray) -> Tuple[int, int]:
        """Longest increasing/decreasing consecutive runs"""
        n = len(y)
        if n < 2:
            return 0, 0

        diff = np.diff(y)
        increasing = diff > 0
        decreasing = diff < 0

        longestInc = self._longestTrueRun(increasing)
        longestDec = self._longestTrueRun(decreasing)

        return longestInc, longestDec

    def _longestTrueRun(self, boolArr: np.ndarray) -> int:
        """Maximum consecutive True run length in a boolean array"""
        if len(boolArr) == 0:
            return 0

        maxRun = 0
        currentRun = 0
        for val in boolArr:
            if val:
                currentRun += 1
                if currentRun > maxRun:
                    maxRun = currentRun
            else:
                currentRun = 0
        return maxRun

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Utilities
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _linearRegression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Simple linear regression (least squares)"""
        n = len(x)
        if n < 2:
            return 0.0, float(np.mean(y)) if len(y) > 0 else 0.0

        xMean = np.mean(x)
        yMean = np.mean(y)
        num = np.sum((x - xMean) * (y - yMean))
        den = np.sum((x - xMean) ** 2)

        if den < 1e-20:
            return 0.0, float(yMean)

        slope = float(num / den)
        intercept = float(yMean - slope * xMean)
        return slope, intercept

    def _movingAverage(self, y: np.ndarray, window: int) -> np.ndarray:
        """Simple moving average"""
        n = len(y)
        window = min(window, n)
        if window < 1:
            return y.copy()

        cumsum = np.cumsum(y)
        cumsum = np.insert(cumsum, 0, 0)
        ma = (cumsum[window:] - cumsum[:-window]) / window

        # Pad to match original length
        padLeft = window // 2
        padRight = n - len(ma) - padLeft
        if padRight < 0:
            padRight = 0
            padLeft = n - len(ma)

        result = np.concatenate([
            np.full(padLeft, ma[0]) if padLeft > 0 else np.array([]),
            ma,
            np.full(padRight, ma[-1]) if padRight > 0 else np.array([]),
        ])

        return result[:n]

    def _emptyFeatures(self) -> Dict[str, float]:
        """Empty feature dictionary (when data is insufficient)"""
        keys = [
            # Basic statistics
            'length', 'mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis', 'cv', 'iqr',
            # Trend
            'trend_strength', 'trend_slope', 'trend_linearity', 'trend_curvature', 'trend_direction',
            # Seasonality
            'seasonality_strength', 'seasonal_period', 'seasonal_peak_position',
            'seasonal_trough_position', 'has_multiple_seasonality', 'dominant_frequency',
            'spectral_entropy', 'seasonal_amplitude',
            # Autocorrelation
            'acf_lag1', 'acf_lag2', 'pacf_lag1', 'pacf_lag2', 'acf_decay_rate',
            'ljung_box_stat', 'arch_lm_stat', 'turning_points_rate',
            # Nonlinearity
            'hurst_exponent', 'approximate_entropy', 'sample_entropy',
            'lempel_ziv_complexity', 'nonlinearity',
            # Stability
            'stability', 'lumpiness', 'flat_spots', 'crossing_points', 'max_kl_shift',
            # Stationarity
            'diff_std_ratio', 'diff_mean_ratio', 'unitroot_kpss_approx', 'unitroot_pp_approx',
            # Forecastability
            'forecastability', 'signal_to_noise', 'mean_absolute_change',
            'mean_second_derivative', 'percentage_zeros',
            # Intermittency
            'adi', 'cv_squared', 'zero_proportion', 'demand_size_mean', 'demand_interval_cv',
            # Volatility
            'garch_alpha', 'garch_beta', 'volatility_clustering', 'max_drawdown',
            'conditional_heteroscedasticity',
            # Miscellaneous
            'n_peaks', 'n_troughs', 'peak_to_trough_time',
            'longest_increasing_run', 'longest_decreasing_run',
        ]
        return {k: 0.0 for k in keys}
