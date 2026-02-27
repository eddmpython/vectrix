"""
변경점 자동 감지 (Changepoint Detection)

시계열 데이터에서 통계적 특성이 변하는 지점을 자동으로 감지:
- PELT (Pruned Exact Linear Time): 가우시안 로그 우도 기반, BIC/커스텀 페널티
- CUSUM (Cumulative Sum): 양방향 누적합 기반 감지
- BOCPD (Bayesian Online Changepoint Detection): 간소화된 베이지안 온라인 감지
- Auto: 여러 방법의 합의(consensus) 기반 자동 감지

순수 numpy/scipy만 사용
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ChangePointResult:
    """변경점 감지 결과"""
    indices: np.ndarray          # 변경점 위치
    nChangepoints: int           # 변경점 수
    confidence: np.ndarray       # 각 변경점의 신뢰도 (0~1)
    segments: List[Dict]         # 각 구간의 통계 (mean, std, trend)
    method: str                  # 사용된 방법


class ChangePointDetector:
    """
    시계열 변경점 감지기

    PELT, CUSUM, BOCPD, Auto 방법 지원.
    Auto 모드에서는 세 가지 방법의 합의를 통해 변경점을 결정.

    Examples
    --------
    >>> detector = ChangePointDetector()
    >>> y = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])
    >>> result = detector.detect(y, method='pelt')
    >>> print(result.indices)
    """

    def detect(
        self,
        y: np.ndarray,
        method: str = 'auto',
        minSize: int = 10,
        penalty: str = 'bic'
    ) -> ChangePointResult:
        """
        시계열에서 변경점 감지

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터 (1차원)
        method : str
            감지 방법 ('pelt', 'cusum', 'bocpd', 'auto')
        minSize : int
            최소 세그먼트 크기
        penalty : str or float
            페널티 유형 ('bic') 또는 직접 지정 값

        Returns
        -------
        ChangePointResult
            감지된 변경점 정보
        """
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        if n < 2 * minSize:
            return ChangePointResult(
                indices=np.array([], dtype=int),
                nChangepoints=0,
                confidence=np.array([], dtype=float),
                segments=[self._computeSegmentStats(y, 0, n)],
                method=method
            )

        try:
            if method == 'pelt':
                indices, confidence = self._detectPELT(y, minSize, penalty)
            elif method == 'cusum':
                indices, confidence = self._detectCUSUM(y, minSize)
            elif method == 'bocpd':
                indices, confidence = self._detectBOCPD(y, minSize)
            elif method == 'auto':
                indices, confidence = self._detectAuto(y, minSize, penalty)
            else:
                raise ValueError(f"알 수 없는 방법: {method}. 'pelt', 'cusum', 'bocpd', 'auto' 중 선택")
        except Exception:
            # graceful fallback: 변경점 없음 반환
            indices = np.array([], dtype=int)
            confidence = np.array([], dtype=float)

        # 세그먼트 통계 계산
        segments = self._computeAllSegmentStats(y, indices)

        return ChangePointResult(
            indices=indices,
            nChangepoints=len(indices),
            confidence=confidence,
            segments=segments,
            method=method
        )

    # ─── PELT (Pruned Exact Linear Time) ──────────────────────────────────

    def _detectPELT(
        self,
        y: np.ndarray,
        minSize: int,
        penalty: str = 'bic'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PELT 알고리즘으로 변경점 감지

        비용함수: 가우시안 로그 우도 (평균 + 분산 변화)
        페널티: BIC = log(n) 또는 사용자 지정
        """
        n = len(y)

        # 페널티 값 계산
        if isinstance(penalty, str) and penalty.lower() == 'bic':
            pen = np.log(n)
        elif isinstance(penalty, (int, float)):
            pen = float(penalty)
        else:
            pen = np.log(n)

        # 누적합 사전 계산 (비용 함수 O(1) 계산용)
        cumSum = np.zeros(n + 1)
        cumSumSq = np.zeros(n + 1)
        cumSum[1:] = np.cumsum(y)
        cumSumSq[1:] = np.cumsum(y ** 2)

        def cost(start: int, end: int) -> float:
            """구간 [start, end)의 가우시안 로그 우도 비용"""
            length = end - start
            if length <= 1:
                return 0.0
            s = cumSum[end] - cumSum[start]
            ss = cumSumSq[end] - cumSumSq[start]
            mean = s / length
            variance = ss / length - mean ** 2
            if variance <= 1e-12:
                return 0.0
            # -2 * 로그 우도 (상수항 제외)
            return length * (np.log(max(variance, 1e-20)) + 1.0)

        # DP 배열
        F = np.full(n + 1, np.inf)
        F[0] = -pen  # 시작점에 -pen (첫 세그먼트의 pen 상쇄)
        lastChange = np.zeros(n + 1, dtype=int)

        # PELT 가지치기를 위한 후보 집합
        candidates = [0]

        for tStar in range(minSize, n + 1):
            bestCost = np.inf
            bestIdx = 0
            newCandidates = []

            for t in candidates:
                if tStar - t < minSize:
                    newCandidates.append(t)
                    continue
                c = F[t] + cost(t, tStar) + pen
                if c < bestCost:
                    bestCost = c
                    bestIdx = t
                # PELT 가지치기: F[t] + cost(t, tStar) <= F[tStar] 이면 유지
                if F[t] + cost(t, tStar) <= bestCost:
                    newCandidates.append(t)

            F[tStar] = bestCost
            lastChange[tStar] = bestIdx
            newCandidates.append(tStar)
            candidates = newCandidates

        # 변경점 역추적
        changepoints = []
        idx = n
        while idx > 0:
            cp = lastChange[idx]
            if cp > 0:
                changepoints.append(cp)
            idx = cp

        changepoints = sorted(changepoints)
        indices = np.array(changepoints, dtype=int)

        # 신뢰도 계산 (전후 세그먼트의 통계적 차이 기반)
        confidence = self._computeConfidence(y, indices, minSize)

        return indices, confidence

    # ─── CUSUM (Cumulative Sum) ───────────────────────────────────────────

    def _detectCUSUM(
        self,
        y: np.ndarray,
        minSize: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        양방향 CUSUM 기반 변경점 감지

        순방향과 역방향 CUSUM을 계산하여 변경점 후보를 검출.
        threshold = 5 * sigma
        """
        n = len(y)
        mu = np.mean(y)
        sigma = max(np.std(y), 1e-10)
        threshold = 5.0 * sigma

        # 순방향 CUSUM (양의 방향)
        cusumPos = np.zeros(n)
        cusumNeg = np.zeros(n)
        for t in range(1, n):
            cusumPos[t] = max(0, cusumPos[t - 1] + (y[t] - mu))
            cusumNeg[t] = min(0, cusumNeg[t - 1] + (y[t] - mu))

        # 역방향 CUSUM
        cusumPosRev = np.zeros(n)
        cusumNegRev = np.zeros(n)
        for t in range(n - 2, -1, -1):
            cusumPosRev[t] = max(0, cusumPosRev[t + 1] + (y[t] - mu))
            cusumNegRev[t] = min(0, cusumNegRev[t + 1] + (y[t] - mu))

        # 통합 CUSUM 통계량
        cusumStat = np.abs(cusumPos) + np.abs(cusumNeg)
        cusumStatRev = np.abs(cusumPosRev) + np.abs(cusumNegRev)
        combinedStat = cusumStat + cusumStatRev

        # threshold 초과 지점에서 변경점 후보 탐색
        candidates = np.where(combinedStat > threshold)[0]
        if len(candidates) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        # 후보를 클러스터링하여 변경점 정제
        indices = self._clusterChangepoints(candidates, combinedStat, minSize)

        # 신뢰도 계산
        confidence = self._computeConfidence(y, indices, minSize)

        return indices, confidence

    # ─── BOCPD (Bayesian Online Changepoint Detection) ────────────────────

    def _detectBOCPD(
        self,
        y: np.ndarray,
        minSize: int,
        hazardLambda: float = 250.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        간소화된 베이지안 온라인 변경점 감지 (BOCPD)

        실행길이 분포를 계산하여 변경점을 추론.
        hazard function: H(r) = 1/lambda (일정한 위험률)

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        minSize : int
            최소 세그먼트 크기
        hazardLambda : float
            hazard function의 lambda 파라미터 (기대 실행길이)
        """
        n = len(y)
        hazard = 1.0 / hazardLambda

        # 실행길이 확률 행렬 (메모리 효율: 현재/이전만 유지)
        # R[t, r] = P(run length = r at time t)
        maxRunLen = n + 1

        # Student-t 예측 분포를 위한 충분 통계량
        # 각 가능한 실행길이에 대해 유지
        mu0 = np.mean(y)
        kappa0 = 1.0
        alpha0 = 1.0
        beta0 = np.var(y) if np.var(y) > 0 else 1.0

        # 현재 실행길이별 충분 통계량
        muN = np.full(maxRunLen, mu0)
        kappaN = np.full(maxRunLen, kappa0)
        alphaN = np.full(maxRunLen, alpha0)
        betaN = np.full(maxRunLen, beta0)

        # 실행길이 확률
        runLenProb = np.zeros(maxRunLen)
        runLenProb[0] = 1.0  # 초기: 실행길이 0에 확률 1

        # 변경점 확률 누적
        cpProb = np.zeros(n)

        for t in range(n):
            # 현재 데이터 포인트
            xt = y[t]

            # 각 실행길이에 대한 예측 확률 (Student-t)
            predProb = np.zeros(t + 1)
            for r in range(t + 1):
                predProb[r] = self._studentTPdf(
                    xt,
                    muN[r],
                    betaN[r] * (kappaN[r] + 1) / (alphaN[r] * kappaN[r]),
                    2.0 * alphaN[r]
                )

            # 성장 확률: P(r_{t+1} = r+1) = P(r_t = r) * pi(x_t | r_t = r) * (1-H)
            growthProb = runLenProb[:t + 1] * predProb * (1 - hazard)

            # 변경점 확률: P(r_{t+1} = 0) = sum P(r_t = r) * pi(x_t | r_t = r) * H
            cpMass = np.sum(runLenProb[:t + 1] * predProb * hazard)

            # 새 실행길이 확률
            newRunLenProb = np.zeros(maxRunLen)
            newRunLenProb[0] = cpMass
            newRunLenProb[1:t + 2] = growthProb

            # 정규화
            total = np.sum(newRunLenProb)
            if total > 0:
                newRunLenProb /= total

            runLenProb = newRunLenProb

            # 변경점 확률 = P(r_t = 0)
            cpProb[t] = runLenProb[0]

            # 충분 통계량 업데이트 (각 실행길이에 대해)
            newMuN = np.full(maxRunLen, mu0)
            newKappaN = np.full(maxRunLen, kappa0)
            newAlphaN = np.full(maxRunLen, alpha0)
            newBetaN = np.full(maxRunLen, beta0)

            for r in range(min(t + 1, maxRunLen - 1)):
                k = kappaN[r]
                m = muN[r]
                a = alphaN[r]
                b = betaN[r]

                newKappaN[r + 1] = k + 1
                newMuN[r + 1] = (k * m + xt) / (k + 1)
                newAlphaN[r + 1] = a + 0.5
                newBetaN[r + 1] = b + 0.5 * k * (xt - m) ** 2 / (k + 1)

            muN = newMuN
            kappaN = newKappaN
            alphaN = newAlphaN
            betaN = newBetaN

        # 변경점 확률에서 피크 추출
        indices = self._extractPeaks(cpProb, minSize, threshold=0.1)

        # 신뢰도는 변경점 확률에서 직접 추출
        if len(indices) > 0:
            confidence = np.clip(cpProb[indices], 0, 1)
        else:
            confidence = np.array([], dtype=float)

        return indices, confidence

    def _studentTPdf(
        self,
        x: float,
        mu: float,
        varScale: float,
        nu: float
    ) -> float:
        """Student-t 확률밀도함수"""
        try:
            from scipy.special import gammaln
            sigma2 = max(varScale, 1e-10)
            z = (x - mu) ** 2 / sigma2
            logp = (
                gammaln((nu + 1) / 2)
                - gammaln(nu / 2)
                - 0.5 * np.log(nu * np.pi * sigma2)
                - (nu + 1) / 2 * np.log(1 + z / nu)
            )
            return np.exp(logp)
        except Exception:
            # fallback: 가우시안 근사
            sigma = np.sqrt(max(varScale, 1e-10))
            return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    # ─── Auto (합의 기반) ─────────────────────────────────────────────────

    def _detectAuto(
        self,
        y: np.ndarray,
        minSize: int,
        penalty: str = 'bic'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        자동 변경점 감지: 여러 방법의 합의(consensus)

        세 가지 방법을 모두 실행하고, 최소 2개 이상의 방법에서
        근접한 위치에 변경점이 감지된 경우만 최종 변경점으로 채택.
        """
        n = len(y)

        # 각 방법으로 감지
        results = {}
        methods = ['pelt', 'cusum', 'bocpd']

        for m in methods:
            try:
                if m == 'pelt':
                    idx, conf = self._detectPELT(y, minSize, penalty)
                elif m == 'cusum':
                    idx, conf = self._detectCUSUM(y, minSize)
                else:
                    idx, conf = self._detectBOCPD(y, minSize)
                results[m] = idx
            except Exception:
                results[m] = np.array([], dtype=int)

        # 합의 기반 변경점 결정
        # 모든 감지된 변경점을 수집하고 근접한 것들을 그룹화
        allCandidates = []
        for m, idx in results.items():
            for cp in idx:
                allCandidates.append((cp, m))

        if len(allCandidates) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)

        allCandidates.sort(key=lambda x: x[0])

        # 근접 변경점 그룹화 (tolerance = minSize // 2)
        tolerance = max(minSize // 2, 3)
        groups = []
        currentGroup = [allCandidates[0]]

        for i in range(1, len(allCandidates)):
            if allCandidates[i][0] - currentGroup[-1][0] <= tolerance:
                currentGroup.append(allCandidates[i])
            else:
                groups.append(currentGroup)
                currentGroup = [allCandidates[i]]
        groups.append(currentGroup)

        # 최소 2개 방법에서 감지된 그룹만 채택
        consensusIndices = []
        consensusConfidence = []

        for group in groups:
            uniqueMethods = set(item[1] for item in group)
            if len(uniqueMethods) >= 2:
                # 그룹의 중앙값을 변경점으로 사용
                positions = [item[0] for item in group]
                cpIdx = int(np.median(positions))
                consensusIndices.append(cpIdx)
                # 신뢰도: 합의한 방법 수 / 전체 방법 수
                consensusConfidence.append(len(uniqueMethods) / len(methods))

        indices = np.array(consensusIndices, dtype=int)
        confidence = np.array(consensusConfidence, dtype=float)

        # minSize 조건 재확인
        if len(indices) > 0:
            indices, confidence = self._enforceMinSize(indices, confidence, n, minSize)

        return indices, confidence

    # ─── 유틸리티 ─────────────────────────────────────────────────────────

    def _clusterChangepoints(
        self,
        candidates: np.ndarray,
        stat: np.ndarray,
        minSize: int
    ) -> np.ndarray:
        """
        후보 변경점들을 클러스터링하여 정제

        근접한 후보를 묶고, 각 클러스터에서 통계량이 가장 높은 지점 선택.
        """
        if len(candidates) == 0:
            return np.array([], dtype=int)

        # 연속 후보를 클러스터로 묶기
        clusters = []
        currentCluster = [candidates[0]]

        for i in range(1, len(candidates)):
            if candidates[i] - candidates[i - 1] <= minSize // 2:
                currentCluster.append(candidates[i])
            else:
                clusters.append(currentCluster)
                currentCluster = [candidates[i]]
        clusters.append(currentCluster)

        # 각 클러스터에서 통계량 최대 지점 선택
        changepoints = []
        for cluster in clusters:
            clusterArr = np.array(cluster)
            bestIdx = clusterArr[np.argmax(stat[clusterArr])]
            changepoints.append(bestIdx)

        # minSize 간격 보장
        result = []
        for cp in sorted(changepoints):
            if len(result) == 0 or cp - result[-1] >= minSize:
                result.append(cp)

        return np.array(result, dtype=int)

    def _extractPeaks(
        self,
        prob: np.ndarray,
        minSize: int,
        threshold: float = 0.1
    ) -> np.ndarray:
        """확률 배열에서 피크(변경점 후보) 추출"""
        n = len(prob)
        peaks = []

        for i in range(1, n - 1):
            if prob[i] > threshold and prob[i] > prob[i - 1] and prob[i] >= prob[i + 1]:
                peaks.append(i)

        if len(peaks) == 0:
            return np.array([], dtype=int)

        # minSize 간격 보장 (가장 높은 확률 우선)
        peaks = sorted(peaks, key=lambda x: prob[x], reverse=True)
        selected = []

        for p in peaks:
            if all(abs(p - s) >= minSize for s in selected):
                selected.append(p)

        return np.array(sorted(selected), dtype=int)

    def _computeConfidence(
        self,
        y: np.ndarray,
        indices: np.ndarray,
        minSize: int
    ) -> np.ndarray:
        """
        변경점의 신뢰도 계산

        전후 세그먼트의 평균 차이를 전체 표준편차로 정규화.
        Welch t-test p-value 기반.
        """
        if len(indices) == 0:
            return np.array([], dtype=float)

        n = len(y)
        confidence = np.zeros(len(indices))
        globalStd = max(np.std(y), 1e-10)

        for i, cp in enumerate(indices):
            # 전 세그먼트
            start = indices[i - 1] if i > 0 else 0
            before = y[start:cp]

            # 후 세그먼트
            end = indices[i + 1] if i < len(indices) - 1 else n
            after = y[cp:end]

            if len(before) < 2 or len(after) < 2:
                confidence[i] = 0.0
                continue

            # Welch t-test 근사
            meanDiff = abs(np.mean(after) - np.mean(before))
            pooledSe = np.sqrt(
                np.var(before) / len(before) + np.var(after) / len(after)
            )

            if pooledSe < 1e-10:
                tStat = meanDiff / globalStd * np.sqrt(min(len(before), len(after)))
            else:
                tStat = meanDiff / pooledSe

            # t-stat을 0~1 신뢰도로 변환 (시그모이드 근사)
            confidence[i] = 1.0 - 2.0 / (1.0 + np.exp(0.5 * tStat))

        return np.clip(confidence, 0.0, 1.0)

    def _computeSegmentStats(
        self,
        y: np.ndarray,
        start: int,
        end: int
    ) -> Dict:
        """단일 세그먼트의 통계량 계산"""
        segment = y[start:end]
        n = len(segment)

        if n == 0:
            return {'start': start, 'end': end, 'mean': 0, 'std': 0, 'trend': 0}

        mean = float(np.mean(segment))
        std = float(np.std(segment))

        # 추세 (선형 회귀 기울기)
        if n >= 2:
            x = np.arange(n, dtype=np.float64)
            xMean = np.mean(x)
            yMean = mean
            num = np.sum((x - xMean) * (segment - yMean))
            den = np.sum((x - xMean) ** 2)
            trend = float(num / den) if den > 0 else 0.0
        else:
            trend = 0.0

        return {
            'start': int(start),
            'end': int(end),
            'length': int(n),
            'mean': mean,
            'std': std,
            'trend': trend
        }

    def _computeAllSegmentStats(
        self,
        y: np.ndarray,
        indices: np.ndarray
    ) -> List[Dict]:
        """모든 세그먼트의 통계량 계산"""
        n = len(y)
        segments = []

        # 시작점 포함
        boundaries = [0] + list(indices) + [n]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segments.append(self._computeSegmentStats(y, start, end))

        return segments

    def _enforceMinSize(
        self,
        indices: np.ndarray,
        confidence: np.ndarray,
        n: int,
        minSize: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """변경점 간 최소 크기 보장"""
        if len(indices) == 0:
            return indices, confidence

        # 신뢰도 순으로 정렬하여 높은 것 우선 유지
        order = np.argsort(-confidence)
        selected = []
        selectedConf = []

        for idx in order:
            cp = indices[idx]
            # minSize 간격 확인 (시작/끝/다른 변경점과)
            if cp < minSize or cp > n - minSize:
                continue
            if all(abs(cp - s) >= minSize for s in selected):
                selected.append(cp)
                selectedConf.append(confidence[idx])

        # 위치 순으로 재정렬
        if len(selected) > 0:
            sortOrder = np.argsort(selected)
            selected = np.array(selected, dtype=int)[sortOrder]
            selectedConf = np.array(selectedConf, dtype=float)[sortOrder]
        else:
            selected = np.array([], dtype=int)
            selectedConf = np.array([], dtype=float)

        return selected, selectedConf
