"""
Anomaly Detection for Time Series

Multiple methods:
- Z-score: 표준편차 기반
- IQR: 사분위범위 기반
- Seasonal Residual: 계절 분해 잔차 기반
- Rolling: 이동 윈도우 기반
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AnomalyResult:
    """이상치 탐지 결과"""
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    scores: np.ndarray = field(default_factory=lambda: np.array([]))
    method: str = ""
    threshold: float = 0.0
    nAnomalies: int = 0
    anomalyRatio: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)


class AnomalyDetector:
    """
    시계열 이상치 탐지기

    Usage:
        >>> detector = AnomalyDetector()
        >>> result = detector.detect(y, method='zscore')
    """

    def detect(
        self,
        y: np.ndarray,
        method: str = 'auto',
        threshold: float = 3.0,
        period: int = 1
    ) -> AnomalyResult:
        """
        이상치 탐지

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        method : str
            'zscore', 'iqr', 'seasonal', 'rolling', 'auto'
        threshold : float
            탐지 임계값
        period : int
            계절 주기 (seasonal method 전용)
        """
        if method == 'auto':
            return self._autoDetect(y, threshold, period)
        elif method == 'zscore':
            return self._zscoreDetect(y, threshold)
        elif method == 'iqr':
            return self._iqrDetect(y, threshold)
        elif method == 'seasonal':
            return self._seasonalDetect(y, threshold, period)
        elif method == 'rolling':
            return self._rollingDetect(y, threshold)
        else:
            return self._zscoreDetect(y, threshold)

    def _autoDetect(self, y: np.ndarray, threshold: float, period: int) -> AnomalyResult:
        results = []

        results.append(self._zscoreDetect(y, threshold))
        results.append(self._iqrDetect(y, 1.5))

        if period > 1 and len(y) >= period * 2:
            results.append(self._seasonalDetect(y, threshold, period))

        allIndices = set()
        for r in results:
            allIndices.update(r.indices.tolist())

        voteCount = np.zeros(len(y))
        for r in results:
            for idx in r.indices:
                voteCount[idx] += 1

        consensusIdx = np.where(voteCount >= 2)[0]

        if len(consensusIdx) == 0:
            consensusIdx = np.where(voteCount >= 1)[0]

        scores = np.zeros(len(y))
        for r in results:
            scores += r.scores / max(len(results), 1)

        details = []
        for idx in consensusIdx:
            details.append({
                'index': int(idx),
                'value': float(y[idx]),
                'score': float(scores[idx]),
                'votes': int(voteCount[idx])
            })

        return AnomalyResult(
            indices=consensusIdx,
            scores=scores,
            method='auto (consensus)',
            threshold=threshold,
            nAnomalies=len(consensusIdx),
            anomalyRatio=len(consensusIdx) / len(y) if len(y) > 0 else 0,
            details=details
        )

    def _zscoreDetect(self, y: np.ndarray, threshold: float) -> AnomalyResult:
        mean = np.mean(y)
        std = np.std(y)
        if std < 1e-10:
            return AnomalyResult(method='zscore', threshold=threshold)

        scores = np.abs((y - mean) / std)
        indices = np.where(scores > threshold)[0]

        details = [{'index': int(i), 'value': float(y[i]), 'zscore': float(scores[i])}
                    for i in indices]

        return AnomalyResult(
            indices=indices, scores=scores, method='zscore', threshold=threshold,
            nAnomalies=len(indices), anomalyRatio=len(indices) / len(y), details=details
        )

    def _iqrDetect(self, y: np.ndarray, multiplier: float) -> AnomalyResult:
        q1 = np.percentile(y, 25)
        q3 = np.percentile(y, 75)
        iqr = q3 - q1

        if iqr < 1e-10:
            return AnomalyResult(method='iqr', threshold=multiplier)

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        scores = np.maximum(y - upper, lower - y) / iqr
        scores = np.maximum(scores, 0)
        indices = np.where((y < lower) | (y > upper))[0]

        details = [{'index': int(i), 'value': float(y[i]), 'iqr_score': float(scores[i])}
                    for i in indices]

        return AnomalyResult(
            indices=indices, scores=scores, method='iqr', threshold=multiplier,
            nAnomalies=len(indices), anomalyRatio=len(indices) / len(y), details=details
        )

    def _seasonalDetect(self, y: np.ndarray, threshold: float, period: int) -> AnomalyResult:
        n = len(y)
        seasonal = np.zeros(n)
        for i in range(period):
            vals = y[i::period]
            seasonal[i::period] = np.mean(vals)

        residuals = y - seasonal
        mean = np.mean(residuals)
        std = np.std(residuals)

        if std < 1e-10:
            return AnomalyResult(method='seasonal', threshold=threshold)

        scores = np.abs((residuals - mean) / std)
        indices = np.where(scores > threshold)[0]

        details = [{'index': int(i), 'value': float(y[i]), 'residual': float(residuals[i]),
                     'seasonal_zscore': float(scores[i])} for i in indices]

        return AnomalyResult(
            indices=indices, scores=scores, method='seasonal', threshold=threshold,
            nAnomalies=len(indices), anomalyRatio=len(indices) / len(y), details=details
        )

    def _rollingDetect(self, y: np.ndarray, threshold: float, window: int = 30) -> AnomalyResult:
        n = len(y)
        scores = np.zeros(n)

        for i in range(n):
            start = max(0, i - window)
            end = i + 1
            windowData = y[start:end]
            mean = np.mean(windowData)
            std = np.std(windowData)
            if std > 1e-10:
                scores[i] = abs(y[i] - mean) / std

        indices = np.where(scores > threshold)[0]

        details = [{'index': int(i), 'value': float(y[i]), 'rolling_zscore': float(scores[i])}
                    for i in indices]

        return AnomalyResult(
            indices=indices, scores=scores, method='rolling', threshold=threshold,
            nAnomalies=len(indices), anomalyRatio=len(indices) / len(y), details=details
        )
