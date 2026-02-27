"""
Constraint-Aware Forecasting

비즈니스/물리적 제약 조건을 예측에 직접 반영하는 제약 인식 예측 후처리기.

기존 시계열 라이브러리가 예측 후 수동 클리핑만 지원하는 것과 달리,
제약 유형별 최적 교정 전략(Projection, Smooth, Redistribute)을 적용하고,
신뢰구간까지 일관성 있게 조정한다.

지원 제약:
- non_negative: 비음수 제약 (재고, 판매량 등)
- range: 최소/최대 범위 제약 (물리적 한계)
- sum_constraint: 윈도우 합계 제약 (예산, 용량)
- yoy_change: 전년동기대비 변동폭 제약 (비즈니스 규칙)
- monotone: 단조 증가/감소 제약 (PAVA 알고리즘)
- capacity: 로지스틱 성장 상한 (Prophet style)
- ratio: 연속 값 비율 제약 (성장률 제한)
- custom: 사용자 정의 함수

Usage:
    >>> from forecastx.adaptive.constraints import ConstraintAwareForecaster, Constraint
    >>> caf = ConstraintAwareForecaster()
    >>> result = caf.apply(predictions, lower95, upper95, constraints=[
    ...     Constraint('non_negative', {}),
    ...     Constraint('range', {'min': 100, 'max': 5000}),
    ...     Constraint('sum_constraint', {'window': 7, 'maxSum': 10000}),
    ...     Constraint('monotone', {'direction': 'increasing'}),
    ... ])
    >>> print(result.violationsBefore, '->', result.violationsAfter)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, Optional, Any


@dataclass
class Constraint:
    """
    예측 제약 조건

    Parameters
    ----------
    type : str
        제약 유형. 지원: 'non_negative', 'range', 'sum_constraint',
        'yoy_change', 'monotone', 'capacity', 'ratio', 'custom'
    params : Dict
        제약 파라미터. 유형별로 상이.

    Examples
    --------
    >>> Constraint('non_negative', {})
    >>> Constraint('range', {'min': 0, 'max': 1000})
    >>> Constraint('sum_constraint', {'window': 7, 'maxSum': 10000})
    >>> Constraint('yoy_change', {'maxPct': 30, 'historicalData': pastYear})
    >>> Constraint('monotone', {'direction': 'increasing'})
    >>> Constraint('capacity', {'capacity': 10000})
    >>> Constraint('ratio', {'minRatio': 0.8, 'maxRatio': 1.2})
    >>> Constraint('custom', {'fn': lambda pred, lo, hi: (np.round(pred), lo, hi)})
    """
    type: str
    params: Dict = field(default_factory=dict)


@dataclass
class ConstraintResult:
    """
    제약 적용 결과

    Attributes
    ----------
    predictions : np.ndarray
        제약 적용 후 예측값
    lower95 : np.ndarray
        제약 적용 후 하한 95% 신뢰구간
    upper95 : np.ndarray
        제약 적용 후 상한 95% 신뢰구간
    violationsBefore : int
        제약 적용 전 위반 수
    violationsAfter : int
        제약 적용 후 위반 수 (정상 시 0)
    constraintsApplied : List[str]
        적용된 제약 유형 목록
    adjustmentDetails : List[Dict]
        각 제약별 조정 상세 (변경된 인덱스, 조정량 등)
    """
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    upper95: np.ndarray = field(default_factory=lambda: np.array([]))
    violationsBefore: int = 0
    violationsAfter: int = 0
    constraintsApplied: List[str] = field(default_factory=list)
    adjustmentDetails: List[Dict] = field(default_factory=list)


# 제약 유형별 우선순위 (Hard constraints 먼저)
_HARD_CONSTRAINTS = {'non_negative', 'range', 'capacity'}
_SOFT_CONSTRAINTS = {'sum_constraint', 'yoy_change', 'monotone', 'ratio', 'custom'}


class ConstraintAwareForecaster:
    """
    제약 조건 인식 예측기

    제약 조건을 예측값과 신뢰구간에 일관되게 적용.
    Hard constraints (non_negative, range, capacity)를 먼저 처리한 뒤,
    Soft constraints (sum, yoy_change, monotone, ratio, custom)를 순서대로 적용.
    마지막으로 부드러운 전환(smoothing)을 수행하여 교정 경계의 인위적 불연속을 완화.

    교정 전략:
    - Projection: 제약 위반 시 가장 가까운 유효점으로 투영 (range, capacity)
    - Smooth: 위반 구간 전후를 부드럽게 연결 (monotone)
    - Redistribute: 합계 제약 시 초과분을 주변 시점에 비례 분배 (sum_constraint)

    Usage:
        >>> caf = ConstraintAwareForecaster()
        >>> result = caf.apply(predictions, lower95, upper95, constraints=[
        ...     Constraint('non_negative', {}),
        ...     Constraint('range', {'min': 100, 'max': 5000}),
        ...     Constraint('sum_constraint', {'window': 7, 'maxSum': 10000}),
        ...     Constraint('yoy_change', {'maxPct': 30, 'historicalData': pastYear}),
        ...     Constraint('monotone', {'direction': 'increasing'}),
        ... ])
    """

    def apply(
        self,
        predictions: np.ndarray,
        lower95: np.ndarray,
        upper95: np.ndarray,
        constraints: List[Constraint],
        smoothing: bool = True
    ) -> ConstraintResult:
        """
        제약 조건을 순서대로 적용

        적용 순서:
        1. hard constraints (non_negative, range, capacity) 먼저
        2. soft constraints (sum, yoy_change, monotone, ratio, custom) 나중에
        3. smoothing=True이면 부드러운 보간으로 마무리

        Parameters
        ----------
        predictions : np.ndarray
            원본 예측값
        lower95 : np.ndarray
            원본 하한 95% 신뢰구간
        upper95 : np.ndarray
            원본 상한 95% 신뢰구간
        constraints : List[Constraint]
            적용할 제약 조건 리스트
        smoothing : bool
            교정 후 부드러운 전환 적용 여부

        Returns
        -------
        ConstraintResult
            제약 적용 결과
        """
        pred = np.array(predictions, dtype=np.float64).copy()
        lo = np.array(lower95, dtype=np.float64).copy()
        hi = np.array(upper95, dtype=np.float64).copy()
        originalPred = pred.copy()

        # 적용 전 위반 수
        violationsBefore = self._countViolations(pred, constraints)

        # Hard/Soft 분리 후 순서대로 적용
        hardConstraints = [c for c in constraints if c.type in _HARD_CONSTRAINTS]
        softConstraints = [c for c in constraints if c.type in _SOFT_CONSTRAINTS]
        orderedConstraints = hardConstraints + softConstraints

        constraintsApplied: List[str] = []
        adjustmentDetails: List[Dict] = []

        for constraint in orderedConstraints:
            predBefore = pred.copy()
            try:
                pred, lo, hi, detail = self._applyOne(pred, lo, hi, constraint)
                constraintsApplied.append(constraint.type)
                nAdjusted = int(np.sum(np.abs(pred - predBefore) > 1e-10))
                detail['nAdjusted'] = nAdjusted
                adjustmentDetails.append(detail)
            except Exception as e:
                adjustmentDetails.append({
                    'constraint': constraint.type,
                    'error': str(e),
                    'nAdjusted': 0
                })

        # 신뢰구간 일관성: lo <= pred <= hi
        lo = np.minimum(lo, pred)
        hi = np.maximum(hi, pred)

        # 부드러운 전환
        if smoothing:
            adjustedMask = np.abs(pred - originalPred) > 1e-10
            if np.any(adjustedMask):
                pred = self._smoothTransitions(pred, originalPred, adjustedMask)
                # smoothing 후에도 hard constraints 재확인
                for constraint in hardConstraints:
                    try:
                        pred, lo, hi, _ = self._applyOne(pred, lo, hi, constraint)
                    except Exception:
                        pass
                lo = np.minimum(lo, pred)
                hi = np.maximum(hi, pred)

        # 적용 후 위반 수
        violationsAfter = self._countViolations(pred, constraints)

        return ConstraintResult(
            predictions=pred,
            lower95=lo,
            upper95=hi,
            violationsBefore=violationsBefore,
            violationsAfter=violationsAfter,
            constraintsApplied=constraintsApplied,
            adjustmentDetails=adjustmentDetails
        )

    def _applyOne(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        constraint: Constraint
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """단일 제약 적용 디스패처"""
        cType = constraint.type
        params = constraint.params

        if cType == 'non_negative':
            p, l, h = self._applyNonNegative(pred, lo, hi)
            detail = {'constraint': 'non_negative'}
        elif cType == 'range':
            minVal = params.get('min', -np.inf)
            maxVal = params.get('max', np.inf)
            p, l, h = self._applyRange(pred, lo, hi, minVal, maxVal)
            detail = {'constraint': 'range', 'min': minVal, 'max': maxVal}
        elif cType == 'sum_constraint':
            window = params.get('window', 7)
            maxSum = params.get('maxSum', np.inf)
            minSum = params.get('minSum', -np.inf)
            p, l, h = self._applySumConstraint(pred, lo, hi, window, maxSum, minSum)
            detail = {'constraint': 'sum_constraint', 'window': window,
                      'maxSum': maxSum, 'minSum': minSum}
        elif cType == 'yoy_change':
            maxPct = params.get('maxPct', 100.0)
            historicalData = params.get('historicalData', None)
            if historicalData is None:
                raise ValueError("yoy_change 제약에는 'historicalData' 파라미터가 필요합니다.")
            p, l, h = self._applyYoYChange(pred, lo, hi, maxPct,
                                            np.asarray(historicalData, dtype=np.float64))
            detail = {'constraint': 'yoy_change', 'maxPct': maxPct}
        elif cType == 'monotone':
            direction = params.get('direction', 'increasing')
            p, l, h = self._applyMonotone(pred, lo, hi, direction)
            detail = {'constraint': 'monotone', 'direction': direction}
        elif cType == 'capacity':
            capacity = params.get('capacity', np.inf)
            floor = params.get('floor', -np.inf)
            p, l, h = self._applyCapacity(pred, lo, hi, capacity, floor)
            detail = {'constraint': 'capacity', 'capacity': capacity, 'floor': floor}
        elif cType == 'ratio':
            minRatio = params.get('minRatio', 0.0)
            maxRatio = params.get('maxRatio', np.inf)
            p, l, h = self._applyRatio(pred, lo, hi, minRatio, maxRatio)
            detail = {'constraint': 'ratio', 'minRatio': minRatio, 'maxRatio': maxRatio}
        elif cType == 'custom':
            fn = params.get('fn', None)
            if fn is None:
                raise ValueError("custom 제약에는 'fn' 파라미터(callable)가 필요합니다.")
            p, l, h = self._applyCustom(pred, lo, hi, fn)
            detail = {'constraint': 'custom'}
        else:
            raise ValueError(f"지원하지 않는 제약 유형: {cType}")

        return p, l, h, detail

    # ------------------------------------------------------------------
    # Hard Constraints
    # ------------------------------------------------------------------

    def _applyNonNegative(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        비음수 제약: 음수 -> 0으로 클리핑

        예측값과 신뢰구간 하한 모두 0 이상으로 보정.
        """
        p = np.maximum(pred, 0.0)
        l = np.maximum(lo, 0.0)
        h = np.maximum(hi, 0.0)
        return p, l, h

    def _applyRange(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        minVal: float,
        maxVal: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        범위 제약: min <= 예측값 <= max

        가장 가까운 유효점으로 투영 (projection).
        """
        p = np.clip(pred, minVal, maxVal)
        l = np.clip(lo, minVal, maxVal)
        h = np.clip(hi, minVal, maxVal)
        return p, l, h

    def _applyCapacity(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        capacity: float,
        floor: float = -np.inf
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        로지스틱 성장 상한/하한 적용 (Prophet style)

        단순 클리핑이 아닌, 상한 근처에서 로지스틱 감쇠를 적용하여
        예측이 상한에 부드럽게 수렴하도록 한다.

        capacity에 가까울수록 (상위 10% 구간) 로지스틱 압축 적용:
        adjusted = floor + (capacity - floor) * sigmoid(scaled)
        """
        p = pred.copy()
        l = lo.copy()
        h = hi.copy()

        if capacity < np.inf:
            # 상한 근처에서 로지스틱 감쇠
            effectiveRange = capacity - floor if floor > -np.inf else capacity
            threshold = capacity - 0.1 * effectiveRange  # 상위 10%

            mask = p > threshold
            if np.any(mask):
                # 로지스틱 압축: cap 근처에서 부드럽게 포화
                overshoot = (p[mask] - threshold) / max(effectiveRange * 0.1, 1e-10)
                # sigmoid 매핑: 0..inf -> 0..1
                compressed = 1.0 / (1.0 + np.exp(-overshoot + 2.0))
                p[mask] = threshold + (capacity - threshold) * compressed

        if floor > -np.inf:
            # 하한 근처에서 로지스틱 감쇠 (대칭)
            effectiveRange = capacity - floor if capacity < np.inf else abs(floor) * 2
            threshold = floor + 0.1 * effectiveRange

            mask = p < threshold
            if np.any(mask):
                undershoot = (threshold - p[mask]) / max(effectiveRange * 0.1, 1e-10)
                compressed = 1.0 / (1.0 + np.exp(-undershoot + 2.0))
                p[mask] = threshold - (threshold - floor) * compressed

        # 최종 안전 클리핑
        if capacity < np.inf:
            p = np.minimum(p, capacity)
            h = np.minimum(h, capacity)
        if floor > -np.inf:
            p = np.maximum(p, floor)
            l = np.maximum(l, floor)

        l = np.clip(l, floor if floor > -np.inf else l.min(), capacity)
        h = np.clip(h, floor if floor > -np.inf else h.min(), capacity)

        return p, l, h

    # ------------------------------------------------------------------
    # Soft Constraints
    # ------------------------------------------------------------------

    def _applySumConstraint(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        window: int,
        maxSum: float,
        minSum: float = -np.inf
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        윈도우 합계 제약

        window 기간의 합계가 maxSum을 초과하면 해당 윈도우 내 값들을
        비례적으로 축소. minSum 미달 시 비례적으로 확대.

        비례 축소: 각 값의 상대적 크기를 보존하면서 합계를 맞춤.
        pred[i] *= (maxSum / currentSum) for i in window
        """
        p = pred.copy()
        l = lo.copy()
        h = hi.copy()
        n = len(p)

        for start in range(0, n - window + 1, window):
            end = min(start + window, n)
            windowSlice = slice(start, end)
            currentSum = np.sum(p[windowSlice])

            if currentSum > maxSum and currentSum > 1e-10:
                # 비례 축소
                ratio = maxSum / currentSum
                p[windowSlice] *= ratio
                l[windowSlice] *= ratio
                h[windowSlice] *= ratio
            elif currentSum < minSum and abs(currentSum) > 1e-10:
                # 비례 확대
                ratio = minSum / currentSum
                p[windowSlice] *= ratio
                l[windowSlice] *= ratio
                h[windowSlice] *= ratio
            elif currentSum < minSum and abs(currentSum) <= 1e-10:
                # 합이 0에 가까운데 minSum이 양수인 경우: 균등 분배
                windowLen = end - start
                fillVal = minSum / windowLen
                p[windowSlice] = fillVal
                l[windowSlice] = fillVal * 0.8
                h[windowSlice] = fillVal * 1.2

        # lo <= hi 일관성
        l, h = np.minimum(l, h), np.maximum(l, h)
        return p, l, h

    def _applyYoYChange(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        maxPct: float,
        historicalData: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        전년동기대비 변동폭 제약

        과거 동일 기간(historicalData)과 비교하여, 각 시점의 예측값이
        해당 시점의 전년 값 대비 maxPct% 이내가 되도록 제한.

        historicalData의 길이는 pred 길이 이상이어야 함.
        historicalData[-len(pred):]을 전년 동기로 사용.

        Parameters
        ----------
        maxPct : float
            최대 허용 변동률 (%)
        historicalData : np.ndarray
            전년 동기 데이터 (pred와 같은 길이 이상)
        """
        p = pred.copy()
        l = lo.copy()
        h = hi.copy()
        n = len(p)

        # historicalData에서 전년 동기 추출
        histLen = len(historicalData)
        if histLen < n:
            # 데이터 부족 시 가용한 만큼만 적용
            refData = np.full(n, np.nan)
            refData[:histLen] = historicalData
        else:
            refData = historicalData[-n:]

        maxRatio = maxPct / 100.0

        for i in range(n):
            ref = refData[i]
            if np.isnan(ref) or abs(ref) < 1e-10:
                continue

            # 허용 범위 계산
            allowedMin = ref * (1.0 - maxRatio)
            allowedMax = ref * (1.0 + maxRatio)

            # 음수 참조값 처리 (min/max 교환)
            if ref < 0:
                allowedMin, allowedMax = allowedMax, allowedMin

            if p[i] > allowedMax:
                adjustRatio = allowedMax / p[i] if abs(p[i]) > 1e-10 else 1.0
                p[i] = allowedMax
                l[i] *= adjustRatio
                h[i] *= adjustRatio
            elif p[i] < allowedMin:
                adjustRatio = allowedMin / p[i] if abs(p[i]) > 1e-10 else 1.0
                p[i] = allowedMin
                l[i] *= adjustRatio
                h[i] *= adjustRatio

        l, h = np.minimum(l, h), np.maximum(l, h)
        return p, l, h

    def _applyMonotone(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        direction: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        단조 제약 (isotonic regression via PAVA)

        direction='increasing': 단조 증가 제약 (y[i] <= y[i+1])
        direction='decreasing': 단조 감소 제약 (y[i] >= y[i+1])

        Pool Adjacent Violators (PAVA) 알고리즘으로 최소 제곱 의미의
        최적 단조 근사를 계산.
        """
        increasing = (direction == 'increasing')
        p = self._pava(pred, increasing=increasing)
        l = self._pava(lo, increasing=increasing)
        h = self._pava(hi, increasing=increasing)

        # 일관성 보장
        l = np.minimum(l, p)
        h = np.maximum(h, p)
        return p, l, h

    def _pava(self, y: np.ndarray, increasing: bool = True) -> np.ndarray:
        """
        Isotonic Regression via Pool Adjacent Violators Algorithm (PAVA)

        단조 증가 제약 하에 ||y - result||^2를 최소화하는 result를 찾는다.

        알고리즘:
        1. y를 순서대로 스캔
        2. y[i] < y[i-1]이면 위반 -> 두 값을 평균으로 pooling
        3. pooled 값이 이전과 위반이면 계속 pooling
        4. 모든 위반이 해소될 때까지 반복

        Parameters
        ----------
        y : np.ndarray
            입력 배열
        increasing : bool
            True면 단조 증가, False면 단조 감소

        Returns
        -------
        np.ndarray
            단조 제약을 만족하는 최적 근사
        """
        n = len(y)
        if n <= 1:
            return y.copy()

        result = y.astype(np.float64).copy()
        if not increasing:
            result = -result

        # 블록 기반 PAVA
        blocks: List[List[int]] = [[i] for i in range(n)]
        blockMeans: List[float] = [float(result[i]) for i in range(n)]

        merged = True
        while merged:
            merged = False
            newBlocks: List[List[int]] = [blocks[0]]
            newMeans: List[float] = [blockMeans[0]]

            for i in range(1, len(blocks)):
                if blockMeans[i] < newMeans[-1]:
                    # 위반 -> 합치기
                    combinedBlock = newBlocks[-1] + blocks[i]
                    combinedMean = float(np.mean(result[combinedBlock]))
                    newBlocks[-1] = combinedBlock
                    newMeans[-1] = combinedMean
                    merged = True
                else:
                    newBlocks.append(blocks[i])
                    newMeans.append(blockMeans[i])

            blocks = newBlocks
            blockMeans = newMeans

        # 결과 할당
        for block, mean in zip(blocks, blockMeans):
            for idx in block:
                result[idx] = mean

        if not increasing:
            result = -result

        return result

    def _applyRatio(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        minRatio: float,
        maxRatio: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        연속 값 비율 제약: minRatio <= pred[t+1]/pred[t] <= maxRatio

        t=0부터 순방향으로 순회하며, t+1의 값이 비율 범위를 벗어나면
        허용 범위 경계로 조정. 이전 조정이 다음에 전파됨.
        """
        p = pred.copy()
        l = lo.copy()
        h = hi.copy()
        n = len(p)

        for i in range(1, n):
            if abs(p[i - 1]) < 1e-10:
                continue

            currentRatio = p[i] / p[i - 1]

            if currentRatio > maxRatio:
                target = p[i - 1] * maxRatio
                if abs(p[i]) > 1e-10:
                    scale = target / p[i]
                    l[i] *= scale
                    h[i] *= scale
                p[i] = target
            elif currentRatio < minRatio:
                target = p[i - 1] * minRatio
                if abs(p[i]) > 1e-10:
                    scale = target / p[i]
                    l[i] *= scale
                    h[i] *= scale
                p[i] = target

        l, h = np.minimum(l, h), np.maximum(l, h)
        return p, l, h

    def _applyCustom(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        fn: Callable
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        사용자 정의 제약 함수 적용

        fn 시그니처: fn(pred, lo, hi) -> (pred, lo, hi)
        각각 np.ndarray.
        """
        p, l, h = fn(pred.copy(), lo.copy(), hi.copy())
        return np.asarray(p, dtype=np.float64), \
               np.asarray(l, dtype=np.float64), \
               np.asarray(h, dtype=np.float64)

    # ------------------------------------------------------------------
    # Smoothing & Utilities
    # ------------------------------------------------------------------

    def _smoothTransitions(
        self,
        pred: np.ndarray,
        originalPred: np.ndarray,
        adjustedMask: np.ndarray
    ) -> np.ndarray:
        """
        교정된 부분과 원본 사이 부드러운 전환

        조정된 지점의 경계에서 가우시안 가중 블렌딩을 적용하여
        인위적인 불연속을 완화.

        경계 탐지 후, 전환 구간 (반경 radius=3)에서:
        weight = exp(-dist^2 / (2 * sigma^2))
        blended = weight * adjusted + (1 - weight) * original

        Parameters
        ----------
        pred : np.ndarray
            교정된 예측값
        originalPred : np.ndarray
            교정 전 원본 예측값
        adjustedMask : np.ndarray (bool)
            교정이 적용된 시점의 마스크
        """
        result = pred.copy()
        n = len(pred)
        radius = min(3, n // 4)
        if radius < 1:
            return result

        sigma = max(radius / 2.0, 1.0)

        # 경계 탐지: adjustedMask의 변경 지점
        boundaries = []
        for i in range(1, n):
            if adjustedMask[i] != adjustedMask[i - 1]:
                boundaries.append(i)

        for boundary in boundaries:
            for offset in range(-radius, radius + 1):
                idx = boundary + offset
                if idx < 0 or idx >= n:
                    continue
                if adjustedMask[idx]:
                    # 조정된 지점: 경계에서 먼 쪽은 조정값 유지,
                    # 가까운 쪽은 원본과 블렌딩
                    dist = abs(offset)
                    weight = np.exp(-dist ** 2 / (2.0 * sigma ** 2))
                    # 조정된 포인트는 weight가 높을수록 원본 비중 증가 (경계 근처)
                    blend = weight * 0.3  # 최대 30%까지 원본 블렌딩
                    result[idx] = (1.0 - blend) * pred[idx] + blend * originalPred[idx]

        return result

    def _countViolations(
        self,
        pred: np.ndarray,
        constraints: List[Constraint]
    ) -> int:
        """
        제약 위반 수 계산

        모든 제약에 대해 위반하는 시점의 총 수를 반환.
        (동일 시점이 여러 제약을 위반하면 중복 카운트)
        """
        total = 0
        n = len(pred)

        for constraint in constraints:
            try:
                total += self._countOneViolation(pred, constraint)
            except Exception:
                pass

        return total

    def _countOneViolation(
        self,
        pred: np.ndarray,
        constraint: Constraint
    ) -> int:
        """단일 제약의 위반 수 계산"""
        cType = constraint.type
        params = constraint.params
        n = len(pred)

        if cType == 'non_negative':
            return int(np.sum(pred < -1e-10))

        elif cType == 'range':
            minVal = params.get('min', -np.inf)
            maxVal = params.get('max', np.inf)
            return int(np.sum((pred < minVal - 1e-10) | (pred > maxVal + 1e-10)))

        elif cType == 'capacity':
            capacity = params.get('capacity', np.inf)
            floor = params.get('floor', -np.inf)
            violations = 0
            if capacity < np.inf:
                violations += int(np.sum(pred > capacity + 1e-10))
            if floor > -np.inf:
                violations += int(np.sum(pred < floor - 1e-10))
            return violations

        elif cType == 'sum_constraint':
            window = params.get('window', 7)
            maxSum = params.get('maxSum', np.inf)
            minSum = params.get('minSum', -np.inf)
            violations = 0
            for start in range(0, n - window + 1, window):
                end = min(start + window, n)
                s = np.sum(pred[start:end])
                if s > maxSum + 1e-10:
                    violations += 1
                if s < minSum - 1e-10:
                    violations += 1
            return violations

        elif cType == 'yoy_change':
            maxPct = params.get('maxPct', 100.0)
            historicalData = params.get('historicalData', None)
            if historicalData is None:
                return 0
            histArr = np.asarray(historicalData, dtype=np.float64)
            histLen = len(histArr)
            violations = 0
            refData = histArr[-n:] if histLen >= n else histArr
            compLen = min(n, len(refData))
            maxRatio = maxPct / 100.0
            for i in range(compLen):
                ref = refData[i] if i < len(refData) else np.nan
                if np.isnan(ref) or abs(ref) < 1e-10:
                    continue
                pctChange = abs(pred[i] - ref) / abs(ref)
                if pctChange > maxRatio + 1e-10:
                    violations += 1
            return violations

        elif cType == 'monotone':
            direction = params.get('direction', 'increasing')
            violations = 0
            for i in range(1, n):
                if direction == 'increasing' and pred[i] < pred[i - 1] - 1e-10:
                    violations += 1
                elif direction == 'decreasing' and pred[i] > pred[i - 1] + 1e-10:
                    violations += 1
            return violations

        elif cType == 'ratio':
            minRatio = params.get('minRatio', 0.0)
            maxRatio = params.get('maxRatio', np.inf)
            violations = 0
            for i in range(1, n):
                if abs(pred[i - 1]) < 1e-10:
                    continue
                r = pred[i] / pred[i - 1]
                if r > maxRatio + 1e-10 or r < minRatio - 1e-10:
                    violations += 1
            return violations

        elif cType == 'custom':
            # custom 제약은 위반 수를 사전에 알기 어려움
            return 0

        return 0
