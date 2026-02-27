"""
Forecast DNA - 시계열 지문 (Time Series Fingerprinting)

시계열의 65+ 통계적 특성을 추출하여 "DNA 프로파일"을 생성.
이를 통해:
1. 최적 모델 자동 추천 (메타러닝)
2. 유사 시계열 검색 (코사인 유사도)
3. 예측 난이도 사전 평가 (0-100)
4. 시계열 고유 식별자 생성 (8자 hex fingerprint)

각 시계열의 본질적 통계 구조를 포착하여 모델 선택을 자동화하고,
대규모 시계열 데이터베이스에서 유사 패턴을 O(N) 시간에 검색.

Usage:
    >>> from forecastx.adaptive.dna import ForecastDNA
    >>> dna = ForecastDNA()
    >>> profile = dna.analyze(y, period=7)
    >>> print(profile.fingerprint)      # '4F2A9B1C'
    >>> print(profile.difficulty)        # 'medium'
    >>> print(profile.recommendedModels) # ['auto_ets', 'theta', 'auto_arima']
    >>> print(profile.category)          # 'seasonal'
    >>>
    >>> sim = dna.similarity(profile1, profile2)  # 0.0~1.0
    >>> dna.findSimilar(profile, database)         # 유사 시계열 검색
"""

import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    from scipy import stats as scipyStats
    from scipy.signal import periodogram
    from scipy.fft import fft as scipyFft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class DNAProfile:
    """
    시계열 DNA 프로파일

    Attributes
    ----------
    features : Dict[str, float]
        65+ 통계적 특성 딕셔너리
    fingerprint : str
        8자 hex 고유 식별자 (특성 벡터의 MD5 해시)
    difficulty : str
        예측 난이도 ('easy', 'medium', 'hard', 'very_hard')
    difficultyScore : float
        예측 난이도 점수 (0-100)
    recommendedModels : List[str]
        DNA 기반 모델 추천 (적합도 내림차순)
    category : str
        시계열 카테고리
        ('trending', 'seasonal', 'stationary', 'intermittent', 'volatile', 'complex')
    summary : str
        자연어 요약 (한국어)
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
    시계열 DNA 분석기

    65+ 통계적 특성을 추출하여 DNAProfile을 생성한다.
    각 특성 계산은 독립적으로 try-except 처리되어,
    하나가 실패해도 나머지는 정상 계산.

    Usage:
        >>> dna = ForecastDNA()
        >>> profile = dna.analyze(y, period=7)
        >>> print(profile.fingerprint)      # '4F2A9B1C'
        >>> print(profile.difficulty)        # 'medium'
        >>> print(profile.recommendedModels) # ['auto_ets', 'theta', 'auto_arima']
        >>> print(profile.category)          # 'seasonal'
        >>>
        >>> sim = dna.similarity(profile1, profile2)  # 0.0~1.0
        >>> dna.findSimilar(profile, database)         # 유사 시계열 검색
    """

    def analyze(self, y: np.ndarray, period: int = 1) -> DNAProfile:
        """
        전체 DNA 분석

        Parameters
        ----------
        y : np.ndarray
            입력 시계열 (1차원)
        period : int
            시계열 주기 (비계절성이면 1)

        Returns
        -------
        DNAProfile
            분석 결과
        """
        yArr = np.asarray(y, dtype=np.float64).ravel()

        # NaN/Inf 제거
        validMask = np.isfinite(yArr)
        if not np.any(validMask):
            return DNAProfile(summary="유효한 데이터가 없습니다.")
        yClean = yArr[validMask]

        if len(yClean) < 4:
            return DNAProfile(
                features={'length': float(len(yClean))},
                summary="데이터 길이가 너무 짧습니다 (4개 미만)."
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
        65+ 통계적 특성 추출

        카테고리별:
        - 기본 통계 (10)
        - 추세 (5)
        - 계절성 (8)
        - 자기상관 (8)
        - 비선형성 (5)
        - 안정성 (5)
        - 정상성 (4)
        - 예측 가능성 (5)
        - 간헐성 (5)
        - 변동성 (5)
        - 기타 (5)
        """
        features: Dict[str, float] = {}

        # 기본 통계 (10)
        try:
            features.update(self._basicStats(y))
        except Exception:
            pass

        # 추세 (5)
        try:
            features.update(self._trendFeatures(y))
        except Exception:
            pass

        # 계절성 (8)
        try:
            features.update(self._seasonalFeatures(y, period))
        except Exception:
            pass

        # 자기상관 (8)
        try:
            features.update(self._autocorrelationFeatures(y))
        except Exception:
            pass

        # 비선형성 (5)
        try:
            features.update(self._nonlinearityFeatures(y))
        except Exception:
            pass

        # 안정성 (5)
        try:
            features.update(self._stabilityFeatures(y))
        except Exception:
            pass

        # 정상성 (4)
        try:
            features.update(self._stationarityFeatures(y))
        except Exception:
            pass

        # 예측 가능성 (5)
        try:
            features.update(self._forecastabilityFeatures(y, period))
        except Exception:
            pass

        # 간헐성 (5)
        try:
            features.update(self._intermittencyFeatures(y))
        except Exception:
            pass

        # 변동성 (5)
        try:
            features.update(self._volatilityFeatures(y))
        except Exception:
            pass

        # 기타 (5)
        try:
            features.update(self._otherFeatures(y))
        except Exception:
            pass

        return features

    # ------------------------------------------------------------------
    # 1. 기본 통계 (10 features)
    # ------------------------------------------------------------------

    def _basicStats(self, y: np.ndarray) -> Dict[str, float]:
        """
        기본 통계량

        length, mean, std, cv (변동계수), skewness, kurtosis,
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
    # 2. 추세 (5 features)
    # ------------------------------------------------------------------

    def _trendFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        추세 관련 특성

        trendStrength: STL 분해 기반 추세 강도 (0~1)
        trendSlope: 선형 회귀 기울기 (정규화)
        trendLinearity: R^2 (선형 적합도)
        trendCurvature: 2차 계수 (비선형 추세)
        trendDirection: 방향 (+1/0/-1)
        """
        n = len(y)
        x = np.arange(n, dtype=np.float64)

        # 선형 회귀
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

        # 정규화 기울기: 전체 범위 대비
        yRange = np.max(y) - np.min(y)
        normalizedSlope = slope * n / yRange if yRange > 1e-10 else 0.0

        # 2차 적합 (곡률)
        curvature = 0.0
        if n >= 5:
            try:
                coeffs = np.polyfit(x, y, 2)
                curvature = coeffs[0] * n ** 2 / yRange if yRange > 1e-10 else 0.0
            except Exception:
                pass

        # 추세 강도: 잔차 대비 (STL-like)
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
    # 3. 계절성 (8 features)
    # ------------------------------------------------------------------

    def _seasonalFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """
        계절성 관련 특성

        seasonalStrength: 계절성 강도 (0~1)
        seasonalPeakPeriod: 주파수 분석 기반 주기
        seasonalAmplitude: 계절 진폭 (정규화)
        seasonalPhaseConsistency: 위상 일관성
        seasonalHarmonicRatio: 기본 주파수 대비 고조파 비율
        seasonalAutoCorr: 주기 간격 자기상관
        seasonalAdjustedVariance: 계절 조정 후 잔차 분산 비율
        multiSeasonalScore: 다중 계절성 점수
        """
        n = len(y)

        # 기본값
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

        # 계절 평균 프로파일
        nFullCycles = n // period
        if nFullCycles < 2:
            return result

        truncated = y[:nFullCycles * period]
        seasonalMatrix = truncated.reshape(nFullCycles, period)
        seasonalProfile = np.mean(seasonalMatrix, axis=0)
        seasonalProfileStd = float(np.std(seasonalProfile, ddof=1))

        # 계절 강도: 계절 프로파일 변동 / 전체 변동
        seasonalStrength = seasonalProfileStd / yStd
        seasonalStrength = float(np.clip(seasonalStrength, 0, 1))

        # 계절 진폭 (정규화)
        amplitude = (float(np.max(seasonalProfile)) - float(np.min(seasonalProfile))) / yStd

        # 위상 일관성: 각 사이클의 피크 위치가 같은지
        peakPositions = np.argmax(seasonalMatrix, axis=1)
        if len(peakPositions) > 1:
            # 원형 통계 (circular mean)
            angles = 2.0 * np.pi * peakPositions / period
            meanCos = np.mean(np.cos(angles))
            meanSin = np.mean(np.sin(angles))
            phaseConsistency = float(np.sqrt(meanCos ** 2 + meanSin ** 2))
        else:
            phaseConsistency = 0.0

        # 주기 간격 자기상관
        if n > period:
            acf = self._acf(y, maxLag=period)
            seasonalAutoCorr = float(acf[period]) if len(acf) > period else 0.0
        else:
            seasonalAutoCorr = 0.0

        # FFT 기반 주기 분석
        peakPeriod = float(period)
        harmonicRatio = 0.0
        multiSeasonalScore = 0.0
        try:
            freqs, power = self._periodogram(y)
            if len(power) > 0:
                # 기본 주기의 주파수
                baseFundamental = 1.0 / period if period > 0 else 0.0

                # 피크 주기 탐색
                sortedIdx = np.argsort(power)[::-1]
                if len(sortedIdx) > 0 and freqs[sortedIdx[0]] > 1e-10:
                    peakPeriod = float(1.0 / freqs[sortedIdx[0]])

                # 고조파 비율
                if baseFundamental > 1e-10:
                    fundIdx = np.argmin(np.abs(freqs - baseFundamental))
                    fundPower = power[fundIdx] if fundIdx < len(power) else 0.0
                    # 2차 고조파
                    harm2Idx = np.argmin(np.abs(freqs - 2 * baseFundamental))
                    harm2Power = power[harm2Idx] if harm2Idx < len(power) else 0.0
                    harmonicRatio = harm2Power / fundPower if fundPower > 1e-10 else 0.0

                # 다중 계절성 점수: 상위 5개 피크 중 비조화 피크 수
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

        # 계절 조정 후 잔차 분산 비율
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
    # 4. 자기상관 (8 features)
    # ------------------------------------------------------------------

    def _autocorrelationFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        자기상관 관련 특성

        acf1~acf3: lag 1~3 자기상관
        acfSum5: lag 1~5 절대 자기상관 합
        acfDecayRate: 자기상관 감쇠율
        pacfLag1: 편자기상관 lag 1
        acfFirstZero: 자기상관이 처음 0을 교차하는 lag
        ljungBoxStat: Ljung-Box 통계량 (자기상관 유의성)
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

        # lag 1~5 절대 합
        acfSum5 = float(np.sum(np.abs(acf[1:min(6, len(acf))])))

        # 감쇠율: 지수 감쇠 모델 적합
        decayRate = 0.0
        absAcf = np.abs(acf[1:])
        if len(absAcf) >= 3 and absAcf[0] > 1e-10:
            try:
                logAcf = np.log(np.maximum(absAcf[:min(10, len(absAcf))], 1e-10))
                lags = np.arange(1, len(logAcf) + 1, dtype=np.float64)
                slope, _ = np.polyfit(lags, logAcf, 1)
                decayRate = -slope  # 양수가 빠른 감쇠
            except Exception:
                pass

        # PACF lag 1 (Yule-Walker 근사)
        pacfLag1 = acf1  # 단순 근사: PACF(1) = ACF(1)

        # 처음 0 교차 lag
        acfFirstZero = float(maxLag)
        for lag in range(1, len(acf)):
            if lag > 0 and acf[lag] * acf[lag - 1] < 0:
                acfFirstZero = float(lag)
                break

        # Ljung-Box 통계량
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
    # 5. 비선형성 (5 features)
    # ------------------------------------------------------------------

    def _nonlinearityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        비선형성 관련 특성

        approximateEntropy: 근사 엔트로피 (규칙성 측정)
        turningPointRate: 전환점 비율 (방향 변경 빈도)
        thirdOrderAutoCorr: 3차 자기상관 (비선형 의존성)
        asymmetry: 상승/하강 비대칭성
        nonlinearAutocorr: 비선형 자기상관 (y^2의 ACF)
        """
        n = len(y)

        # 근사 엔트로피
        apen = self._approximateEntropy(y)

        # 전환점 비율
        turningPoints = 0
        for i in range(1, n - 1):
            if (y[i] > y[i - 1] and y[i] > y[i + 1]) or \
               (y[i] < y[i - 1] and y[i] < y[i + 1]):
                turningPoints += 1
        turningPointRate = turningPoints / max(n - 2, 1)

        # 3차 자기상관: E[y(t) * y(t-1) * y(t-2)]
        thirdOrder = 0.0
        if n > 3:
            yc = y - np.mean(y)
            std = np.std(y, ddof=1)
            if std > 1e-10:
                yc = yc / std
                thirdOrder = float(np.mean(yc[2:] * yc[1:-1] * yc[:-2]))

        # 비대칭성: 상승 이동 vs 하강 이동
        diffs = np.diff(y)
        if len(diffs) > 0:
            posSum = float(np.sum(diffs[diffs > 0]))
            negSum = float(np.abs(np.sum(diffs[diffs < 0])))
            totalMove = posSum + negSum
            asymmetry = (posSum - negSum) / totalMove if totalMove > 1e-10 else 0.0
        else:
            asymmetry = 0.0

        # 비선형 자기상관: y^2의 ACF(1)
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
    # 6. 안정성 (5 features)
    # ------------------------------------------------------------------

    def _stabilityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        안정성 관련 특성

        stabilityMean: 구간별 평균의 변동 (낮을수록 안정)
        stabilityVariance: 구간별 분산의 변동
        levelShiftCount: 수준 변동 수
        levelShiftMagnitude: 최대 수준 변동 크기
        structuralBreakScore: 구조적 변환 점수
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

        # 구간별 평균 변동의 변동계수
        meanStab = float(np.std(segMeans, ddof=1)) / yStd if yStd > 1e-10 else 0.0

        # 구간별 분산 변동의 변동계수
        meanVar = float(np.mean(segVars))
        varStab = float(np.std(segVars, ddof=1)) / meanVar if meanVar > 1e-10 else 0.0

        # 수준 변동 감지
        meanDiffs = np.abs(np.diff(segMeans))
        threshold = 2.0 * yStd / np.sqrt(segLen) if yStd > 1e-10 else 0.0
        levelShiftCount = int(np.sum(meanDiffs > threshold))
        levelShiftMag = float(np.max(meanDiffs)) / yStd if yStd > 1e-10 else 0.0

        # 구조적 변환 점수: CUSUM 기반 간이 계산
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
    # 7. 정상성 (4 features)
    # ------------------------------------------------------------------

    def _stationarityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        정상성 관련 특성

        adfStatistic: ADF 검정 통계량 (간이 근사)
        diffStationary: 1차 차분 후 정상성 개선도
        hurstExponent: Hurst 지수 (장기 기억)
        unitRootIndicator: 단위근 존재 지표
        """
        n = len(y)

        # Hurst exponent
        hurst = self._hurstExponent(y)

        # ADF 간이 근사: AR(1) 계수로 추정
        # y(t) = rho * y(t-1) + e(t), 단위근이면 rho -> 1
        adfStat = 0.0
        rho = 0.0
        if n > 3:
            yLag = y[:-1]
            yNow = y[1:]
            yLagMean = np.mean(yLag)
            denom = np.sum((yLag - yLagMean) ** 2)
            if denom > 1e-10:
                rho = float(np.sum((yLag - yLagMean) * (yNow - np.mean(yNow))) / denom)
                # ADF 통계량 = (rho - 1) / SE(rho) 근사
                residuals = yNow - rho * yLag
                se = float(np.std(residuals, ddof=1)) / np.sqrt(denom) if denom > 1e-10 else 1.0
                adfStat = (rho - 1.0) / se if se > 1e-10 else 0.0

        # 1차 차분 후 분산 감소율
        diffImprovement = 0.0
        if n > 2:
            origVar = float(np.var(y, ddof=1))
            diffVar = float(np.var(np.diff(y), ddof=1))
            if origVar > 1e-10:
                diffImprovement = 1.0 - diffVar / origVar
                # 양수면 차분이 도움됨 (단위근 존재 가능)

        # 단위근 지표: rho가 1에 가까울수록 1
        unitRoot = max(0.0, 1.0 - abs(1.0 - abs(rho)) * 5.0) if abs(rho) > 0.5 else 0.0

        return {
            'adfStatistic': float(np.clip(adfStat, -20, 5)),
            'diffStationary': float(np.clip(diffImprovement, -2, 1)),
            'hurstExponent': float(np.clip(hurst, 0, 1)),
            'unitRootIndicator': float(np.clip(unitRoot, 0, 1))
        }

    # ------------------------------------------------------------------
    # 8. 예측 가능성 (5 features)
    # ------------------------------------------------------------------

    def _forecastabilityFeatures(self, y: np.ndarray, period: int) -> Dict[str, float]:
        """
        예측 가능성 관련 특성

        spectralEntropy: 스펙트럼 엔트로피 (주파수 분산)
        forecastability: 1 - spectralEntropy (예측 가능성)
        signalToNoise: 신호 대 잡음비
        sampleEntropy: 근사 샘플 엔트로피
        regularityIndex: 규칙성 지수
        """
        n = len(y)

        # 스펙트럼 엔트로피
        specEntropy = self._spectralEntropy(y)
        forecastability = 1.0 - specEntropy

        # 신호 대 잡음비
        # 추세+계절 = 신호, 나머지 = 잡음
        snr = 0.0
        yStd = float(np.std(y, ddof=1)) if n > 1 else 0.0
        if yStd > 1e-10 and n > 5:
            # 이동평균으로 신호 추정
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

        # 샘플 엔트로피 (간소화)
        sampleEnt = self._approximateEntropy(y, m=2, r=0.2)

        # 규칙성 지수: 예측 가능 성분 / 전체
        regularityIndex = max(0.0, 1.0 - sampleEnt / 3.0) if sampleEnt < 3.0 else 0.0

        return {
            'spectralEntropy': float(np.clip(specEntropy, 0, 1)),
            'forecastability': float(np.clip(forecastability, 0, 1)),
            'signalToNoise': float(np.clip(snr, 0, 100)),
            'sampleEntropy': float(np.clip(sampleEnt, 0, 5)),
            'regularityIndex': float(np.clip(regularityIndex, 0, 1))
        }

    # ------------------------------------------------------------------
    # 9. 간헐성 (5 features)
    # ------------------------------------------------------------------

    def _intermittencyFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        간헐성 관련 특성 (수요 예측에 중요)

        zeroRatio: 0값 비율
        adi: Average Demand Interval (평균 수요 간격)
        cv2: 비영 수요의 변동계수 제곱
        intermittencyType: 간헐성 유형 지표
            0=continuous, 1=intermittent, 2=lumpy, 3=erratic
        demandDensity: 수요 밀도 (비영값 집중도)
        """
        n = len(y)
        threshold = np.max(np.abs(y)) * 0.001 if np.max(np.abs(y)) > 0 else 1e-10

        # 0값 (거의 0) 비율
        zeroMask = np.abs(y) <= threshold
        zeroRatio = float(np.sum(zeroMask)) / n

        # 비영 수요
        nonZero = y[~zeroMask]

        # ADI: 수요 발생 간 평균 간격
        adi = 1.0
        demandIndices = np.where(~zeroMask)[0]
        if len(demandIndices) > 1:
            intervals = np.diff(demandIndices)
            adi = float(np.mean(intervals))

        # CV^2: 비영 수요의 변동계수 제곱
        cv2 = 0.0
        if len(nonZero) > 1:
            nonZeroMean = float(np.mean(nonZero))
            nonZeroStd = float(np.std(nonZero, ddof=1))
            if abs(nonZeroMean) > 1e-10:
                cv2 = (nonZeroStd / nonZeroMean) ** 2

        # 간헐성 유형 분류 (Syntetos-Boylan)
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

        # 수요 밀도: 비영값의 집중도 (Gini-like)
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
    # 10. 변동성 (5 features)
    # ------------------------------------------------------------------

    def _volatilityFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        변동성 관련 특성

        volatility: 로그 수익률 표준편차
        volatilityClustering: 변동성 클러스터링 (ARCH 효과)
        garchEffect: 변동성 자기상관 (제곱 수익률 ACF)
        extremeValueRatio: 극단값 비율 (3-sigma 초과)
        tailIndex: 꼬리 두께 지표 (Hill estimator 근사)
        """
        n = len(y)

        # 변동성: 차분 기반
        diffs = np.diff(y)
        volatility = 0.0
        if len(diffs) > 1:
            yStd = float(np.std(y, ddof=1))
            diffStd = float(np.std(diffs, ddof=1))
            volatility = diffStd / yStd if yStd > 1e-10 else 0.0

        # 변동성 클러스터링: 제곱 차분의 자기상관
        volClustering = 0.0
        garchEffect = 0.0
        if len(diffs) > 3:
            sqDiffs = diffs ** 2
            sqAcf = self._acf(sqDiffs, maxLag=min(5, len(sqDiffs) // 3))
            volClustering = float(sqAcf[1]) if len(sqAcf) > 1 else 0.0
            garchEffect = float(np.sum(np.abs(sqAcf[1:min(4, len(sqAcf))])))

        # 극단값 비율
        extremeRatio = 0.0
        if n > 3:
            yMean = np.mean(y)
            yStd = float(np.std(y, ddof=1))
            if yStd > 1e-10:
                zScores = np.abs((y - yMean) / yStd)
                extremeRatio = float(np.sum(zScores > 3.0)) / n

        # 꼬리 두께: 간이 Hill estimator
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
    # 11. 기타 (5 features)
    # ------------------------------------------------------------------

    def _otherFeatures(self, y: np.ndarray) -> Dict[str, float]:
        """
        기타 특성

        flatSpotRate: 연속 동일값 구간 비율
        crossingRate: 평균선 교차 빈도
        peakCount: 로컬 피크 수 (정규화)
        longestRun: 최장 증가/감소 런 길이 (정규화)
        binEntropy: 이진 엔트로피 (증감 패턴 엔트로피)
        """
        n = len(y)

        # 연속 동일값 구간
        flatSpots = 0
        for i in range(1, n):
            if abs(y[i] - y[i - 1]) < 1e-10:
                flatSpots += 1
        flatSpotRate = flatSpots / max(n - 1, 1)

        # 평균선 교차 빈도
        yMean = np.mean(y)
        crossings = 0
        for i in range(1, n):
            if (y[i] - yMean) * (y[i - 1] - yMean) < 0:
                crossings += 1
        crossingRate = crossings / max(n - 1, 1)

        # 로컬 피크 수
        peaks = 0
        for i in range(1, n - 1):
            if y[i] > y[i - 1] and y[i] > y[i + 1]:
                peaks += 1
        peakCount = peaks / max(n - 2, 1)

        # 최장 연속 증가/감소 런
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

        # 이진 엔트로피: 증가/감소 패턴의 Shannon 엔트로피
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
        특성 벡터의 해시 기반 8자 hex 식별자

        특성을 정렬된 키 순서로 직렬화한 뒤 MD5 해시의 상위 8자.
        동일 시계열은 동일 지문, 유사 시계열도 유사 지문은 아님 (해시 특성).
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
        예측 난이도 평가 (0-100)

        난이도 증가 요소:
        - 높은 entropy -> +20
        - 낮은 forecastability -> +20
        - 높은 nonlinearity -> +15
        - 많은 변경점(levelShiftCount) -> +10
        - 높은 noise-to-signal -> +15
        - 간헐성 -> +10
        - 높은 변동성 클러스터링 -> +10

        난이도 감소 요소:
        - 강한 추세 -> -10
        - 강한 계절성 -> -10
        - 높은 자기상관 -> -10
        """
        score = 50.0  # 기본 중간

        # --- 증가 요소 ---
        # 스펙트럼 엔트로피 (높을수록 어렵)
        specEnt = features.get('spectralEntropy', 0.5)
        score += (specEnt - 0.5) * 40.0  # 0.5->0, 1.0->+20

        # 낮은 예측 가능성
        fc = features.get('forecastability', 0.5)
        score += (0.5 - fc) * 40.0  # 0.5->0, 0.0->+20

        # 비선형성 (ApEn)
        apen = features.get('approximateEntropy', 0.5)
        score += min(apen * 7.5, 15.0)

        # 수준 변동
        shifts = features.get('levelShiftCount', 0.0)
        score += min(shifts * 2.5, 10.0)

        # 신호 대 잡음비 (낮을수록 어렵)
        snr = features.get('signalToNoise', 1.0)
        snrPenalty = max(0.0, 15.0 - snr * 5.0)
        score += min(snrPenalty, 15.0)

        # 간헐성
        zeroRatio = features.get('zeroRatio', 0.0)
        score += zeroRatio * 10.0

        # 변동성 클러스터링
        volCluster = features.get('volatilityClustering', 0.0)
        score += max(0.0, volCluster) * 10.0

        # --- 감소 요소 ---
        # 강한 추세
        trendStr = features.get('trendStrength', 0.0)
        score -= trendStr * 10.0

        # 강한 계절성
        seasStr = features.get('seasonalStrength', 0.0)
        score -= seasStr * 10.0

        # 높은 자기상관
        acf1 = abs(features.get('acf1', 0.0))
        score -= acf1 * 10.0

        # 클리핑
        score = float(np.clip(score, 0, 100))

        # 등급
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
        DNA 특성 기반 모델 추천 (메타러닝)

        규칙 기반 추천 시스템. 각 규칙에 매칭되면 해당 모델들에 점수 부여.
        최종적으로 점수가 높은 순으로 정렬하여 상위 5개 반환.

        규칙:
        - 강한 추세 + 약한 계절성 -> theta, auto_arima, rwd
        - 강한 계절성 + 약한 추세 -> auto_ets, auto_mstl, seasonal_naive
        - 강한 추세 + 강한 계절성 -> auto_ets, auto_mstl, theta
        - 높은 변동성 -> garch, window_avg, auto_arima
        - 간헐적 수요 -> croston, mean, naive
        - 안정적 -> mean, naive, auto_ets
        - 비선형 -> auto_mstl, dot, auto_ces
        - 복잡한 계절성 -> auto_mstl, tbats, auto_ces
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

        def _add(model: str, points: float) -> None:
            scores[model] = scores.get(model, 0.0) + points

        # 규칙 1: 강한 추세 + 약한 계절성
        if trendStr > 0.5 and seasStr < 0.3:
            _add('theta', 3.0)
            _add('auto_arima', 2.5)
            _add('rwd', 2.0)
            _add('dot', 2.0)

        # 규칙 2: 강한 계절성 + 약한 추세
        if seasStr > 0.4 and trendStr < 0.3:
            _add('auto_ets', 3.0)
            _add('mstl', 2.5)
            _add('seasonal_naive', 2.0)
            _add('auto_ces', 1.5)

        # 규칙 3: 강한 추세 + 강한 계절성
        if trendStr > 0.4 and seasStr > 0.4:
            _add('auto_ets', 3.0)
            _add('mstl', 3.0)
            _add('theta', 2.0)
            _add('tbats', 1.5)

        # 규칙 4: 높은 변동성
        if volat > 1.0 or volCluster > 0.3:
            _add('garch', 3.0)
            _add('window_avg', 1.5)
            _add('auto_arima', 1.5)

        # 규칙 5: 간헐적 수요
        if zeroRatio > 0.3 or intermType >= 1.0:
            _add('croston', 3.0)
            _add('mean', 1.5)
            _add('naive', 1.0)

        # 규칙 6: 안정적 (높은 예측 가능성 + 낮은 변동성)
        if fc > 0.7 and volat < 0.5 and trendStr < 0.2 and seasStr < 0.2:
            _add('mean', 2.5)
            _add('naive', 2.0)
            _add('auto_ets', 2.0)

        # 규칙 7: 비선형
        if apen > 1.0 or nonlinAcf > 0.3:
            _add('mstl', 2.5)
            _add('dot', 2.5)
            _add('auto_ces', 2.0)

        # 규칙 8: 복잡한 다중 계절성
        if multiSeas > 0.3:
            _add('mstl', 3.0)
            _add('tbats', 3.0)
            _add('auto_ces', 2.0)

        # 규칙 9: 높은 자기상관 (지속성)
        if acf1 > 0.7:
            _add('auto_arima', 2.0)
            _add('auto_ets', 1.5)
            _add('theta', 1.0)

        # 규칙 10: 낮은 자기상관 (잡음)
        if acf1 < 0.2 and seasStr < 0.2:
            _add('mean', 2.0)
            _add('window_avg', 1.5)
            _add('naive', 1.0)

        # 기본 추천 (아무 규칙도 매칭되지 않을 때)
        if not scores:
            scores = {
                'auto_ets': 3.0,
                'theta': 2.5,
                'auto_arima': 2.0,
                'mstl': 1.5,
                'dot': 1.0
            }

        # 점수 내림차순 정렬, 상위 5개
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [model for model, _ in ranked[:5]]

    # ==================================================================
    # Categorization
    # ==================================================================

    def _categorize(self, features: Dict[str, float]) -> str:
        """
        시계열 카테고리 분류

        카테고리:
        - 'trending': 추세 지배적
        - 'seasonal': 계절성 지배적
        - 'stationary': 안정적/정상성
        - 'intermittent': 간헐적 수요
        - 'volatile': 높은 변동성
        - 'complex': 복합 (여러 특성 혼재)
        """
        trendStr = features.get('trendStrength', 0.0)
        seasStr = features.get('seasonalStrength', 0.0)
        volat = features.get('volatility', 0.0)
        zeroRatio = features.get('zeroRatio', 0.0)
        apen = features.get('approximateEntropy', 0.0)
        acf1 = abs(features.get('acf1', 0.0))

        # 간헐성 우선
        if zeroRatio > 0.3:
            return 'intermittent'

        # 높은 변동성
        if volat > 1.5 and trendStr < 0.3 and seasStr < 0.3:
            return 'volatile'

        # 복합
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

        # 추세 지배
        if trendStr > 0.5 and trendStr > seasStr:
            return 'trending'

        # 계절성 지배
        if seasStr > 0.4 and seasStr > trendStr:
            return 'seasonal'

        # 정상성
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
        """자연어 요약 생성 (한국어)"""
        parts = []

        # 카테고리
        catNames = {
            'trending': '추세형',
            'seasonal': '계절형',
            'stationary': '안정형',
            'intermittent': '간헐형',
            'volatile': '변동형',
            'complex': '복합형'
        }
        catName = catNames.get(category, category)
        parts.append(f"이 시계열은 '{catName}' 유형입니다.")

        # 추세
        trendStr = features.get('trendStrength', 0.0)
        trendDir = features.get('trendDirection', 0.0)
        if trendStr > 0.5:
            dirStr = '상승' if trendDir > 0 else ('하락' if trendDir < 0 else '수평')
            parts.append(f"강한 {dirStr} 추세가 관찰됩니다 (강도: {trendStr:.2f}).")
        elif trendStr > 0.2:
            parts.append(f"약한 추세가 있습니다 (강도: {trendStr:.2f}).")

        # 계절성
        seasStr = features.get('seasonalStrength', 0.0)
        if seasStr > 0.4:
            period = features.get('seasonalPeakPeriod', 0)
            parts.append(f"뚜렷한 계절 패턴이 있습니다 (강도: {seasStr:.2f}, 주기: {period:.0f}).")
        elif seasStr > 0.2:
            parts.append(f"약한 계절성이 감지됩니다 (강도: {seasStr:.2f}).")

        # 간헐성
        zeroRatio = features.get('zeroRatio', 0.0)
        if zeroRatio > 0.3:
            parts.append(f"데이터의 {zeroRatio*100:.0f}%가 영값으로, 간헐적 수요 패턴입니다.")

        # 변동성
        volat = features.get('volatility', 0.0)
        if volat > 1.0:
            parts.append(f"변동성이 높습니다 (지수: {volat:.2f}).")

        # 난이도
        diffNames = {
            'easy': '쉬움',
            'medium': '보통',
            'hard': '어려움',
            'very_hard': '매우 어려움'
        }
        diffName = diffNames.get(difficulty, difficulty)
        score = features.get('difficultyScore', 50.0)
        parts.append(f"예측 난이도: {diffName}.")

        # Hurst
        hurst = features.get('hurstExponent', 0.5)
        if hurst > 0.6:
            parts.append(f"장기 기억 특성이 있습니다 (Hurst: {hurst:.2f}).")
        elif hurst < 0.4:
            parts.append(f"평균 회귀 성향이 있습니다 (Hurst: {hurst:.2f}).")

        return ' '.join(parts)

    # ==================================================================
    # Similarity & Search
    # ==================================================================

    def similarity(self, profile1: DNAProfile, profile2: DNAProfile) -> float:
        """
        두 DNA 프로파일의 코사인 유사도 (0.0~1.0)

        공통 특성 키만 사용하여 벡터를 구성한 뒤 코사인 유사도 계산.

        Parameters
        ----------
        profile1, profile2 : DNAProfile

        Returns
        -------
        float
            코사인 유사도 (0.0~1.0, 1이 가장 유사)
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
        # 코사인 유사도를 0~1로 매핑 (-1~1 -> 0~1)
        return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))

    def findSimilar(
        self,
        target: DNAProfile,
        database: List[DNAProfile],
        topK: int = 5
    ) -> List[Tuple[int, float]]:
        """
        데이터베이스에서 가장 유사한 시계열 검색

        Parameters
        ----------
        target : DNAProfile
            검색 대상 프로파일
        database : List[DNAProfile]
            검색 대상 데이터베이스
        topK : int
            반환할 최대 결과 수

        Returns
        -------
        List[Tuple[int, float]]
            [(인덱스, 유사도), ...] 유사도 내림차순
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
        여러 시계열 일괄 분석

        Parameters
        ----------
        seriesDict : Dict[str, np.ndarray]
            {이름: 시계열} 딕셔너리
        period : int
            공통 주기

        Returns
        -------
        Dict[str, DNAProfile]
            {이름: DNAProfile} 결과 딕셔너리
        """
        results = {}
        for name, series in seriesDict.items():
            try:
                results[name] = self.analyze(np.asarray(series), period=period)
            except Exception as e:
                results[name] = DNAProfile(summary=f"분석 실패: {str(e)}")
        return results

    # ==================================================================
    # Helper: ACF
    # ==================================================================

    def _acf(self, y: np.ndarray, maxLag: int) -> np.ndarray:
        """
        자기상관함수 (Autocorrelation Function)

        정규화된 자기공분산을 lag 0~maxLag까지 계산.
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
        """FFT 기반 파워 스펙트럼 밀도 추정"""
        n = len(y)
        yDetrended = y - np.mean(y)

        if SCIPY_AVAILABLE:
            freqs, power = periodogram(yDetrended, fs=1.0)
        else:
            # numpy FFT fallback
            spectrum = np.fft.rfft(yDetrended)
            power = np.abs(spectrum) ** 2 / n
            freqs = np.fft.rfftfreq(n, d=1.0)

        # DC 성분 제거
        if len(freqs) > 1:
            return freqs[1:], power[1:]
        return freqs, power

    # ==================================================================
    # Helper: Harmonic check
    # ==================================================================

    def _isHarmonic(self, freq: float, fundamental: float, tolerance: float = 0.1) -> bool:
        """주파수가 기본 주파수의 정수배(고조파)인지 확인"""
        if fundamental < 1e-10:
            return False
        ratio = freq / fundamental
        return abs(ratio - round(ratio)) < tolerance

    # ==================================================================
    # Helper: Hurst Exponent (R/S Analysis)
    # ==================================================================

    def _hurstExponent(self, y: np.ndarray) -> float:
        """
        R/S 분석으로 Hurst exponent 계산

        H > 0.5: 추세 지속 (장기 기억)
        H = 0.5: 랜덤 워크
        H < 0.5: 평균 회귀

        Parameters
        ----------
        y : np.ndarray

        Returns
        -------
        float
            Hurst 지수 (0~1)
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

        시계열의 규칙성/복잡도 측정.
        값이 클수록 복잡/불규칙.

        Parameters
        ----------
        y : np.ndarray
        m : int
            임베딩 차원 (기본 2)
        r : float
            허용 오차 (표준편차의 비율, 기본 0.2)

        Returns
        -------
        float
            ApEn 값 (>= 0)
        """
        n = len(y)
        if n < m + 1:
            return 0.0

        yStd = float(np.std(y, ddof=1))
        tolerance = r * yStd if yStd > 1e-10 else r

        def _phi(dim: int) -> float:
            """dim 차원 패턴의 로그 매칭 확률"""
            nPatterns = n - dim + 1
            if nPatterns <= 0:
                return 0.0

            # 패턴 구성
            patterns = np.array([y[i:i + dim] for i in range(nPatterns)])

            counts = np.zeros(nPatterns)
            for i in range(nPatterns):
                # Chebyshev 거리로 매칭
                dists = np.max(np.abs(patterns - patterns[i]), axis=1)
                counts[i] = np.sum(dists <= tolerance)

            # 자기 자신 포함 (bias 보정 없음, 전통적 ApEn)
            counts = counts / nPatterns
            counts = np.maximum(counts, 1e-10)  # log(0) 방지
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

        FFT 후 정규화된 파워 스펙트럼의 Shannon entropy.
        값이 클수록 넓은 주파수 분포 (복잡/예측 어려움).
        값이 작을수록 에너지가 소수 주파수에 집중 (규칙/예측 쉬움).

        Returns
        -------
        float
            정규화 스펙트럼 엔트로피 (0~1)
        """
        n = len(y)
        if n < 4:
            return 0.5

        try:
            freqs, power = self._periodogram(y)
            if len(power) == 0:
                return 0.5

            # 정규화: 확률 분포로 변환
            totalPower = np.sum(power)
            if totalPower < 1e-10:
                return 0.5

            psd = power / totalPower
            psd = psd[psd > 1e-10]  # 0 제거

            # Shannon entropy
            entropy = -float(np.sum(psd * np.log(psd)))

            # 최대 엔트로피 (균등 분포)로 정규화
            maxEntropy = np.log(len(psd))
            if maxEntropy < 1e-10:
                return 0.5

            return float(np.clip(entropy / maxEntropy, 0, 1))
        except Exception:
            return 0.5
