"""
시계열 특성 추출 (Time Series Feature Extraction) - Forecast DNA 기반

65+ 통계적 특성을 순수 numpy/scipy로 추출:
- 기본 통계 (10): length, mean, std, min, max, median, skewness, kurtosis, cv, iqr
- 추세 (5): trend_strength, trend_slope, trend_linearity, trend_curvature, trend_direction
- 계절성 (8): seasonality_strength, seasonal_period, seasonal_peak/trough, 다중계절성 등
- 자기상관 (8): ACF, PACF, Ljung-Box, ARCH-LM 등
- 비선형성 (5): Hurst exponent, approximate/sample entropy, LZ complexity, nonlinearity
- 안정성 (5): stability, lumpiness, flat_spots, crossing_points, max_kl_shift
- 정상성 (4): diff_std_ratio, diff_mean_ratio, KPSS 근사, PP 근사
- 예측 가능성 (5): forecastability, SNR, mean_absolute_change 등
- 간헐성 (5): ADI, cv_squared, zero_proportion 등
- 변동성 (5): GARCH alpha/beta, volatility_clustering, max_drawdown 등
- 기타 (5): peaks, troughs, peak_to_trough_time, longest runs

순수 numpy/scipy만 사용 (pandas는 extractBatch에서만)
"""

import numpy as np
import hashlib
from typing import Dict, List, Optional, Tuple


class TSFeatureExtractor:
    """
    시계열 특성 추출기

    65+ 통계적 특성을 추출하여 시계열의 "DNA"를 구성.
    Kats TSFeature 수준의 포괄적 특성 추출.

    Examples
    --------
    >>> extractor = TSFeatureExtractor()
    >>> y = np.random.randn(200)
    >>> features = extractor.extract(y, period=7)
    >>> print(len(features))  # 65+
    """

    def extract(self, y: np.ndarray, period: int = 1) -> Dict[str, float]:
        """
        시계열에서 모든 특성 추출

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터 (1차원)
        period : int
            계절 주기 (1이면 비계절)

        Returns
        -------
        Dict[str, float]
            특성 이름 -> 값 딕셔너리 (65+ 항목)
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        y = y[~np.isnan(y)]
        n = len(y)

        if n < 4:
            return self._emptyFeatures()

        features = {}

        # 기본 통계 (10)
        features.update(self._basicStats(y))

        # 추세 (5)
        features.update(self._trendFeatures(y))

        # 계절성 (8)
        features.update(self._seasonalityFeatures(y, period))

        # 자기상관 (8)
        features.update(self._autocorrelationFeatures(y, period))

        # 비선형성 (5)
        features.update(self._nonlinearityFeatures(y))

        # 안정성 (5)
        features.update(self._stabilityFeatures(y, period))

        # 정상성 (4)
        features.update(self._stationarityFeatures(y))

        # 예측 가능성 (5)
        features.update(self._forecastabilityFeatures(y, period))

        # 간헐성 (5)
        features.update(self._intermittencyFeatures(y))

        # 변동성 (5)
        features.update(self._volatilityFeatures(y))

        # 기타 (5)
        features.update(self._miscFeatures(y))

        return features

    def extractBatch(
        self,
        seriesDict: Dict[str, np.ndarray],
        period: int = 1
    ):
        """
        여러 시계열의 특성을 일괄 추출하여 DataFrame 반환

        Parameters
        ----------
        seriesDict : Dict[str, np.ndarray]
            시계열 이름 -> 데이터 딕셔너리
        period : int
            계절 주기

        Returns
        -------
        pd.DataFrame
            행: 시계열 이름, 열: 특성
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
        두 특성 벡터 간 코사인 유사도 계산

        Parameters
        ----------
        features1 : Dict[str, float]
            첫 번째 특성 딕셔너리
        features2 : Dict[str, float]
            두 번째 특성 딕셔너리

        Returns
        -------
        float
            코사인 유사도 (-1 ~ 1)
        """
        # 공통 키만 사용
        commonKeys = sorted(set(features1.keys()) & set(features2.keys()))
        if len(commonKeys) == 0:
            return 0.0

        v1 = np.array([features1[k] for k in commonKeys], dtype=np.float64)
        v2 = np.array([features2[k] for k in commonKeys], dtype=np.float64)

        # NaN/Inf 처리
        validMask = np.isfinite(v1) & np.isfinite(v2)
        v1 = v1[validMask]
        v2 = v2[validMask]

        if len(v1) == 0:
            return 0.0

        # 정규화 (z-score)
        combined = np.column_stack([v1, v2])
        mu = np.mean(combined, axis=1, keepdims=True)
        sigma = np.std(combined, axis=1, keepdims=True)
        sigma[sigma < 1e-10] = 1.0
        v1n = (v1 - mu.ravel()) / sigma.ravel()
        v2n = (v2 - mu.ravel()) / sigma.ravel()

        # 코사인 유사도
        dot = np.dot(v1n, v2n)
        norm1 = np.linalg.norm(v1n)
        norm2 = np.linalg.norm(v2n)

        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0

        return float(np.clip(dot / (norm1 * norm2), -1.0, 1.0))

    def fingerprint(self, features: Dict[str, float]) -> str:
        """
        특성 딕셔너리의 해시 기반 지문 생성 (8자 hex)

        동일한 특성 패턴을 가진 시계열은 동일한 지문을 가짐.

        Parameters
        ----------
        features : Dict[str, float]
            특성 딕셔너리

        Returns
        -------
        str
            8자 hex 문자열
        """
        # 키 정렬 후 양자화하여 해시 생성
        sortedKeys = sorted(features.keys())
        values = []
        for k in sortedKeys:
            v = features[k]
            if np.isfinite(v):
                # 소수점 3자리로 양자화 (약간의 노이즈 무시)
                values.append(f"{k}:{v:.3f}")
            else:
                values.append(f"{k}:nan")

        hashInput = "|".join(values).encode('utf-8')
        return hashlib.md5(hashInput).hexdigest()[:8]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 기본 통계 (10)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _basicStats(self, y: np.ndarray) -> Dict[str, float]:
        """기본 통계 특성 (10개)"""
        n = len(y)
        mean = float(np.mean(y))
        std = float(np.std(y, ddof=1)) if n > 1 else 0.0

        try:
            from scipy.stats import skew, kurtosis
            skewness = float(skew(y, bias=False))
            kurt = float(kurtosis(y, bias=False, fisher=True))
        except ImportError:
            # 순수 numpy fallback
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
        """순수 numpy 왜도 계산"""
        n = len(y)
        if n < 3:
            return 0.0
        m = np.mean(y)
        s = np.std(y, ddof=1)
        if s < 1e-10:
            return 0.0
        return float(n / ((n - 1) * (n - 2)) * np.sum(((y - m) / s) ** 3))

    def _computeKurtosis(self, y: np.ndarray) -> float:
        """순수 numpy 첨도 계산 (Fisher, excess)"""
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
    # 추세 (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _trendFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """추세 관련 특성 (5개)"""
        n = len(y)
        x = np.arange(n, dtype=np.float64)

        # 선형 회귀
        slope, intercept = self._linearRegression(x, y)
        yHat = intercept + slope * x
        residual = y - yHat

        # 추세 강도: 1 - Var(residual) / Var(y)
        varY = np.var(y)
        varResid = np.var(residual)
        trendStrength = max(0, 1 - varResid / max(varY, 1e-10))

        # 추세 선형성: R^2
        sst = max(np.sum((y - np.mean(y)) ** 2), 1e-10)
        sse = np.sum(residual ** 2)
        linearity = max(0, 1 - sse / sst)

        # 곡률: 2차항 계수
        curvature = 0.0
        if n >= 5:
            try:
                coeffs = np.polyfit(x, y, 2)
                curvature = float(coeffs[0])
            except Exception:
                pass

        # 추세 방향: slope의 부호 (양수=상승, 음수=하락, 0=횡보)
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
    # 계절성 (8)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _seasonalityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """계절성 관련 특성 (8개)"""
        n = len(y)

        # FFT 기반 스펙트럼 분석
        fft = np.fft.rfft(y - np.mean(y))
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(n)

        # 정규화 파워 스펙트럼
        totalPower = max(np.sum(power[1:]), 1e-10)  # DC 제외
        normPower = power[1:] / totalPower

        # 주 주파수 (dominant frequency)
        if len(normPower) > 0:
            dominantIdx = np.argmax(normPower)
            dominantFreq = float(freqs[dominantIdx + 1])
        else:
            dominantFreq = 0.0

        # 스펙트럼 엔트로피
        spectralEntropy = self._spectralEntropy(normPower)

        # 계절성 강도 (period가 1보다 큰 경우)
        seasonalityStrength = 0.0
        seasonalPeak = 0.0
        seasonalTrough = 0.0
        seasonalAmplitude = 0.0
        hasMultipleSeasonality = 0.0
        detectedPeriod = float(period)

        if period > 1 and n >= 2 * period:
            # 계절 분해로 계절성 강도 추정
            trend = self._movingAverage(y, period)
            detrended = y - trend

            # 계절 평균
            seasonalMeans = np.zeros(period)
            for i in range(period):
                vals = detrended[i::period]
                seasonalMeans[i] = np.mean(vals)
            seasonalMeans -= np.mean(seasonalMeans)

            # 계절성 강도: 1 - Var(remainder) / Var(detrended)
            seasonal = np.tile(seasonalMeans, n // period + 1)[:n]
            remainder = detrended - seasonal
            varDetrended = max(np.var(detrended), 1e-10)
            varRemainder = np.var(remainder)
            seasonalityStrength = max(0, 1 - varRemainder / varDetrended)

            # 계절 피크/트로프 위치 (주기 내 상대 위치)
            seasonalPeak = float(np.argmax(seasonalMeans)) / period
            seasonalTrough = float(np.argmin(seasonalMeans)) / period
            seasonalAmplitude = float(np.max(seasonalMeans) - np.min(seasonalMeans))

            # 다중 계절성 감지 (스펙트럼에서 주요 피크 수)
            nPeaksSpectrum = self._countSpectralPeaks(normPower, threshold=0.05)
            hasMultipleSeasonality = 1.0 if nPeaksSpectrum > 1 else 0.0

        # 자동 주기 감지 (FFT 기반)
        if n > 4:
            try:
                # 피크 주파수에서 주기 추정
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
        """정규화된 스펙트럼 엔트로피 계산"""
        p = normPower[normPower > 0]
        if len(p) == 0:
            return 0.0
        p = p / np.sum(p)
        entropy = -np.sum(p * np.log2(p + 1e-20))
        maxEntropy = np.log2(len(p)) if len(p) > 1 else 1.0
        return float(entropy / max(maxEntropy, 1e-10))

    def _countSpectralPeaks(self, normPower: np.ndarray, threshold: float = 0.05) -> int:
        """스펙트럼에서 주요 피크 수 계산"""
        count = 0
        for i in range(1, len(normPower) - 1):
            if (normPower[i] > threshold and
                    normPower[i] > normPower[i - 1] and
                    normPower[i] >= normPower[i + 1]):
                count += 1
        return count

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 자기상관 (8)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _autocorrelationFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """자기상관 관련 특성 (8개)"""
        n = len(y)

        # ACF
        acf = self._acf(y, nlags=max(period * 2, 10))
        acfLag1 = acf[1] if len(acf) > 1 else 0.0
        acfLag2 = acf[2] if len(acf) > 2 else 0.0

        # PACF (Durbin-Levinson 알고리즘)
        pacf = self._pacf(y, nlags=max(period * 2, 10))
        pacfLag1 = pacf[1] if len(pacf) > 1 else 0.0
        pacfLag2 = pacf[2] if len(pacf) > 2 else 0.0

        # ACF 감소율
        acfDecayRate = self._acfDecayRate(acf)

        # Ljung-Box 통계량
        ljungBox = self._ljungBoxStat(y, acf, nlags=min(10, n // 5))

        # ARCH-LM 통계량
        archLm = self._archLmStat(y)

        # 전환점 비율
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
        """자기상관함수 계산"""
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
        편자기상관함수 계산 (Durbin-Levinson 알고리즘)
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
        """ACF 감소율 (지수 감소 피팅)"""
        nlags = len(acf) - 1
        if nlags < 2:
            return 0.0

        absAcf = np.abs(acf[1:])
        if np.all(absAcf < 1e-10):
            return 1.0  # 즉시 감소

        # 로그 변환 후 선형 회귀
        valid = absAcf > 1e-10
        if np.sum(valid) < 2:
            return 0.5

        x = np.arange(1, nlags + 1)[valid]
        logAcf = np.log(absAcf[valid])

        slope, _ = self._linearRegression(x.astype(float), logAcf)
        return float(np.clip(-slope, 0, 2))

    def _ljungBoxStat(self, y: np.ndarray, acf: np.ndarray, nlags: int = 10) -> float:
        """Ljung-Box Q 통계량"""
        n = len(y)
        nlags = min(nlags, len(acf) - 1, n - 1)
        if nlags < 1:
            return 0.0

        Q = 0.0
        for k in range(1, nlags + 1):
            Q += acf[k] ** 2 / (n - k)

        return float(n * (n + 2) * Q)

    def _archLmStat(self, y: np.ndarray) -> float:
        """ARCH-LM 통계량 (잔차 제곱의 자기상관)"""
        n = len(y)
        if n < 10:
            return 0.0

        residuals = y - np.mean(y)
        residSq = residuals ** 2

        acfResidSq = self._acf(residSq, nlags=5)
        # LM stat ≈ n * R^2 (잔차 제곱의 회귀)
        rSq = np.sum(acfResidSq[1:6] ** 2)
        return float(n * rSq)

    def _turningPointsRate(self, y: np.ndarray) -> float:
        """전환점 비율 (방향이 바뀌는 점의 비율)"""
        n = len(y)
        if n < 3:
            return 0.0

        diff = np.diff(y)
        signs = np.sign(diff)
        turningPoints = np.sum(signs[:-1] != signs[1:])

        return float(turningPoints / (n - 2))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 비선형성 (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _nonlinearityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """비선형성 관련 특성 (5개)"""
        return {
            'hurst_exponent': self._hurstExponent(y),
            'approximate_entropy': self._approximateEntropy(y),
            'sample_entropy': self._sampleEntropy(y),
            'lempel_ziv_complexity': self._lempelZivComplexity(y),
            'nonlinearity': self._nonlinearityStat(y),
        }

    def _hurstExponent(self, y: np.ndarray, maxLag: int = 20) -> float:
        """
        Hurst exponent 계산 (R/S 분석)

        H > 0.5: 지속성 (trending)
        H = 0.5: 랜덤 워크
        H < 0.5: 반지속성 (mean-reverting)
        """
        n = len(y)
        if n < 20:
            return 0.5

        try:
            lags = range(2, min(maxLag + 1, n // 4))
            rsValues = []
            lagValues = []

            for lag in lags:
                subseries = [y[i:i + lag] for i in range(0, n - lag + 1, lag)]

                rsLag = []
                for sub in subseries:
                    if len(sub) < 2:
                        continue
                    mean = np.mean(sub)
                    cumDev = np.cumsum(sub - mean)
                    R = np.max(cumDev) - np.min(cumDev)
                    S = np.std(sub, ddof=1)
                    if S > 1e-10:
                        rsLag.append(R / S)

                if len(rsLag) > 0:
                    rsValues.append(np.mean(rsLag))
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
        근사 엔트로피 (Approximate Entropy)

        패턴 복잡도 측정. 높을수록 불규칙.
        """
        n = len(y)
        if n < 10:
            return 0.0

        if r is None:
            r = 0.2 * np.std(y, ddof=1)
        if r < 1e-10:
            return 0.0

        try:
            def phi(m_val):
                templates = np.array([y[i:i + m_val] for i in range(n - m_val + 1)])
                nTemplates = len(templates)
                counts = np.zeros(nTemplates)

                for i in range(nTemplates):
                    # 체비셰프 거리
                    dists = np.max(np.abs(templates - templates[i]), axis=1)
                    counts[i] = np.sum(dists <= r) / nTemplates

                return np.mean(np.log(counts + 1e-20))

            return float(max(phi(m) - phi(m + 1), 0))
        except Exception:
            return 0.0

    def _sampleEntropy(self, y: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
        """
        표본 엔트로피 (Sample Entropy)

        근사 엔트로피의 편향 보정 버전.
        """
        n = len(y)
        if n < 10:
            return 0.0

        if r is None:
            r = 0.2 * np.std(y, ddof=1)
        if r < 1e-10:
            return 0.0

        try:
            def countMatches(m_val):
                templates = np.array([y[i:i + m_val] for i in range(n - m_val)])
                nTemplates = len(templates)
                count = 0

                for i in range(nTemplates):
                    for j in range(i + 1, nTemplates):
                        if np.max(np.abs(templates[i] - templates[j])) <= r:
                            count += 1
                return count

            A = countMatches(m + 1)
            B = countMatches(m)

            if B == 0 or A == 0:
                return 0.0

            return float(-np.log(A / B))
        except Exception:
            return 0.0

    def _lempelZivComplexity(self, y: np.ndarray) -> float:
        """
        Lempel-Ziv 복잡도

        이진화 후 LZ76 알고리즘으로 복잡도 계산.
        정규화: c(n) / (n / log2(n))
        """
        n = len(y)
        if n < 4:
            return 0.0

        try:
            # 이진화 (중앙값 기준)
            binary = (y > np.median(y)).astype(int)
            binaryStr = ''.join(map(str, binary))

            # LZ76 복잡도
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

            # 정규화
            logN = np.log2(max(n, 2))
            normalized = complexity * logN / max(n, 1)

            return float(np.clip(normalized, 0, 5))
        except Exception:
            return 0.0

    def _nonlinearityStat(self, y: np.ndarray) -> float:
        """
        비선형성 통계량

        잔차의 3차 모멘트와 BDS-like 통계량 근사.
        """
        n = len(y)
        if n < 10:
            return 0.0

        try:
            # 선형 제거 (AR(1) 잔차)
            yLag = y[:-1]
            yCurrent = y[1:]
            slope, intercept = self._linearRegression(yLag, yCurrent)
            residuals = yCurrent - (intercept + slope * yLag)

            # 비선형성: 잔차 제곱과 원시 잔차의 상관
            residSq = residuals ** 2
            acfResidSq = self._acf(residSq, nlags=3)
            nonlinearity = np.mean(np.abs(acfResidSq[1:]))

            return float(nonlinearity)
        except Exception:
            return 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 안정성 (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _stabilityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """안정성 관련 특성 (5개)"""
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
        """안정성: 이동 윈도우 평균의 분산"""
        n = len(y)
        if n < windowSize * 2:
            return 0.0

        nWindows = n // windowSize
        means = np.array([np.mean(y[i * windowSize:(i + 1) * windowSize])
                          for i in range(nWindows)])

        return float(np.var(means))

    def _lumpiness(self, y: np.ndarray, windowSize: int) -> float:
        """덩어리성: 이동 윈도우 분산의 분산"""
        n = len(y)
        if n < windowSize * 2:
            return 0.0

        nWindows = n // windowSize
        variances = np.array([np.var(y[i * windowSize:(i + 1) * windowSize])
                              for i in range(nWindows)])

        return float(np.var(variances))

    def _flatSpots(self, y: np.ndarray) -> float:
        """플랫 스팟: 동일 값이 연속되는 최대 길이"""
        n = len(y)
        if n < 2:
            return 0.0

        # 히스토그램 빈 기반 (10개 빈)
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
        """교차점: 평균을 교차하는 횟수 (정규화)"""
        n = len(y)
        if n < 3:
            return 0.0

        mean = np.mean(y)
        aboveMean = y > mean
        crossings = np.sum(aboveMean[:-1] != aboveMean[1:])

        return float(crossings / (n - 1))

    def _maxKLShift(self, y: np.ndarray, windowSize: int) -> float:
        """최대 KL 다이버전스 이동: 인접 윈도우 분포 차이의 최대값"""
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

            # 가우시안 근사 KL 다이버전스
            m1, s1 = np.mean(w1), max(np.std(w1), 1e-10)
            m2, s2 = np.mean(w2), max(np.std(w2), 1e-10)

            kl = np.log(s2 / s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2 * s2 ** 2) - 0.5
            maxKL = max(maxKL, abs(kl))

        return float(maxKL)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 정상성 (4)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _stationarityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """정상성 관련 특성 (4개)"""
        n = len(y)
        diff = np.diff(y)

        # 차분 전후 비율
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
        간소화된 KPSS 통계량

        H0: 수준 정상 (trend-stationary)
        큰 값 = 비정상
        """
        n = len(y)
        if n < 10:
            return 0.0

        try:
            # 추세 제거
            x = np.arange(n, dtype=np.float64)
            slope, intercept = self._linearRegression(x, y)
            residuals = y - (intercept + slope * x)

            # 누적합
            cumResid = np.cumsum(residuals)
            S2 = np.sum(cumResid ** 2) / (n ** 2)

            # 장기 분산 추정 (Bartlett 커널)
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
        간소화된 Phillips-Perron 통계량

        H0: 단위근 (비정상)
        큰 음수 = 정상
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

            # 장기 분산 보정
            nlags = min(int(n ** (1 / 3)), n // 4)
            gamma0 = sigma2
            longRunVar = gamma0
            for k in range(1, nlags + 1):
                if k < len(residuals):
                    weight = 1.0 - k / (nlags + 1)
                    gammaK = np.sum(residuals[:len(residuals) - k] * residuals[k:]) / (n - 2)
                    longRunVar += 2 * weight * gammaK

            # PP t-stat 근사
            seLag = np.sqrt(sigma2 / max(np.sum((yLag - np.mean(yLag)) ** 2), 1e-10))
            tStat = (rho - 1.0) / max(seLag, 1e-10)

            # 보정항
            correction = (longRunVar - gamma0) / (2 * max(seLag, 1e-10) *
                         np.sqrt(max(np.sum((yLag - np.mean(yLag)) ** 2), 1e-10)))
            ppStat = tStat - correction

            return float(ppStat)
        except Exception:
            return 0.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 예측 가능성 (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _forecastabilityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """예측 가능성 관련 특성 (5개)"""
        n = len(y)

        # 스펙트럼 엔트로피 기반 예측 가능성 (1 - 엔트로피)
        fft = np.fft.rfft(y - np.mean(y))
        power = np.abs(fft) ** 2
        totalPower = max(np.sum(power[1:]), 1e-10)
        normPower = power[1:] / totalPower
        spectralEntropy = self._spectralEntropy(normPower)
        forecastability = 1.0 - spectralEntropy

        # 신호 대 잡음비
        signalPower = np.var(self._movingAverage(y, max(period, 3)))
        noisePower = max(np.var(y - self._movingAverage(y, max(period, 3))), 1e-10)
        snr = signalPower / noisePower

        # 평균 절대 변화
        meanAbsChange = float(np.mean(np.abs(np.diff(y)))) if n > 1 else 0.0

        # 평균 2차 도함수
        if n >= 3:
            secondDiff = np.diff(y, n=2)
            meanSecondDeriv = float(np.mean(np.abs(secondDiff)))
        else:
            meanSecondDeriv = 0.0

        # 영점 비율
        pctZeros = float(np.sum(np.abs(y) < 1e-10) / n)

        return {
            'forecastability': float(forecastability),
            'signal_to_noise': float(snr),
            'mean_absolute_change': meanAbsChange,
            'mean_second_derivative': meanSecondDeriv,
            'percentage_zeros': pctZeros,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 간헐성 (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _intermittencyFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """간헐성 관련 특성 (5개) - 간헐적 수요 패턴 특성화"""
        n = len(y)
        zeroProportion = float(np.sum(np.abs(y) < 1e-10) / n)

        # 수요 간격 (Average Demand Interval)
        nonZeroIdx = np.where(np.abs(y) > 1e-10)[0]
        if len(nonZeroIdx) < 2:
            adi = float(n)
            demandIntervalCv = 0.0
        else:
            intervals = np.diff(nonZeroIdx)
            adi = float(np.mean(intervals))
            demandIntervalCv = float(np.std(intervals) / max(np.mean(intervals), 1e-10))

        # 비영 수요 크기
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
    # 변동성 (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _volatilityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """변동성 관련 특성 (5개)"""
        n = len(y)

        # GARCH(1,1) 파라미터 근사 추정
        garchAlpha, garchBeta = self._estimateGarchParams(y)

        # 변동성 클러스터링: 절대값 수익률의 ACF(1)
        absReturns = np.abs(np.diff(y))
        if len(absReturns) > 2:
            acfAbs = self._acf(absReturns, nlags=5)
            volClustering = float(acfAbs[1]) if len(acfAbs) > 1 else 0.0
        else:
            volClustering = 0.0

        # 최대 낙폭 (max drawdown)
        maxDrawdown = self._maxDrawdown(y)

        # 조건부 이분산성 (잔차 제곱과 과거 잔차 제곱의 상관)
        condHeterosced = self._conditionalHeteroscedasticity(y)

        return {
            'garch_alpha': float(garchAlpha),
            'garch_beta': float(garchBeta),
            'volatility_clustering': float(volClustering),
            'max_drawdown': float(maxDrawdown),
            'conditional_heteroscedasticity': float(condHeterosced),
        }

    def _estimateGarchParams(self, y: np.ndarray) -> Tuple[float, float]:
        """GARCH(1,1) 파라미터 모멘트 추정 (빠른 근사)"""
        n = len(y)
        if n < 20:
            return 0.1, 0.8

        try:
            residuals = y - np.mean(y)
            residSq = residuals ** 2

            # 자기상관 기반 모멘트 추정
            acfSq = self._acf(residSq, nlags=5)
            rho1 = acfSq[1] if len(acfSq) > 1 else 0.0
            rho2 = acfSq[2] if len(acfSq) > 2 else 0.0

            # alpha + beta 근사
            persistence = max(0, min(rho1, 0.999))

            # alpha: rho1 - beta * rho1 ≈ rho2 관계에서 추정
            if abs(rho1) > 1e-10:
                beta = max(0, min((rho2 / rho1), 0.99))
            else:
                beta = 0.8

            alpha = max(0, min(persistence - beta, 0.5))

            return alpha, beta
        except Exception:
            return 0.1, 0.8

    def _maxDrawdown(self, y: np.ndarray) -> float:
        """최대 낙폭 (peak-to-trough)"""
        n = len(y)
        if n < 2:
            return 0.0

        cumMax = np.maximum.accumulate(y)
        drawdown = cumMax - y
        rangeY = max(np.max(y) - np.min(y), 1e-10)

        return float(np.max(drawdown) / rangeY)

    def _conditionalHeteroscedasticity(self, y: np.ndarray) -> float:
        """조건부 이분산성 측정"""
        n = len(y)
        if n < 10:
            return 0.0

        try:
            residuals = y - np.mean(y)
            residSq = residuals ** 2

            # 잔차 제곱의 AR(1) R^2
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
    # 기타 (5)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _miscFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """기타 특성 (5개)"""
        n = len(y)

        # 피크/트로프 감지
        nPeaks, nTroughs, peakToTroughTime = self._peakTroughStats(y)

        # 최장 증가/감소 구간
        longestInc, longestDec = self._longestRuns(y)

        return {
            'n_peaks': float(nPeaks),
            'n_troughs': float(nTroughs),
            'peak_to_trough_time': float(peakToTroughTime),
            'longest_increasing_run': float(longestInc),
            'longest_decreasing_run': float(longestDec),
        }

    def _peakTroughStats(self, y: np.ndarray) -> Tuple[int, int, float]:
        """피크/트로프 통계"""
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

        # 피크-트로프 간 평균 시간
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
        """최장 증가/감소 연속 구간"""
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
        """불리언 배열에서 True가 연속되는 최대 길이"""
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
    # 유틸리티
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _linearRegression(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """단순 선형 회귀 (최소제곱)"""
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
        """단순 이동평균"""
        n = len(y)
        window = min(window, n)
        if window < 1:
            return y.copy()

        cumsum = np.cumsum(y)
        cumsum = np.insert(cumsum, 0, 0)
        ma = (cumsum[window:] - cumsum[:-window]) / window

        # 패딩하여 원본 길이와 동일하게
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
        """빈 특성 딕셔너리 (데이터 부족 시)"""
        keys = [
            # 기본 통계
            'length', 'mean', 'std', 'min', 'max', 'median', 'skewness', 'kurtosis', 'cv', 'iqr',
            # 추세
            'trend_strength', 'trend_slope', 'trend_linearity', 'trend_curvature', 'trend_direction',
            # 계절성
            'seasonality_strength', 'seasonal_period', 'seasonal_peak_position',
            'seasonal_trough_position', 'has_multiple_seasonality', 'dominant_frequency',
            'spectral_entropy', 'seasonal_amplitude',
            # 자기상관
            'acf_lag1', 'acf_lag2', 'pacf_lag1', 'pacf_lag2', 'acf_decay_rate',
            'ljung_box_stat', 'arch_lm_stat', 'turning_points_rate',
            # 비선형성
            'hurst_exponent', 'approximate_entropy', 'sample_entropy',
            'lempel_ziv_complexity', 'nonlinearity',
            # 안정성
            'stability', 'lumpiness', 'flat_spots', 'crossing_points', 'max_kl_shift',
            # 정상성
            'diff_std_ratio', 'diff_mean_ratio', 'unitroot_kpss_approx', 'unitroot_pp_approx',
            # 예측 가능성
            'forecastability', 'signal_to_noise', 'mean_absolute_change',
            'mean_second_derivative', 'percentage_zeros',
            # 간헐성
            'adi', 'cv_squared', 'zero_proportion', 'demand_size_mean', 'demand_interval_cv',
            # 변동성
            'garch_alpha', 'garch_beta', 'volatility_clustering', 'max_drawdown',
            'conditional_heteroscedasticity',
            # 기타
            'n_peaks', 'n_troughs', 'peak_to_trough_time',
            'longest_increasing_run', 'longest_decreasing_run',
        ]
        return {k: 0.0 for k in keys}
