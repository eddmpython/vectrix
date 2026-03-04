"""
Constraint-Aware Forecasting

Constraint-aware forecast post-processor that directly incorporates
business/physical constraints into predictions.

Unlike other time series libraries that only support manual clipping after
forecasting, this module applies optimal correction strategies per constraint
type (Projection, Smooth, Redistribute) and consistently adjusts confidence
intervals as well.

Supported constraints:
- non_negative: Non-negativity constraint (inventory, sales volume, etc.)
- range: Min/max range constraint (physical limits)
- sum_constraint: Window sum constraint (budget, capacity)
- yoy_change: Year-over-year change limit constraint (business rules)
- monotone: Monotone increasing/decreasing constraint (PAVA algorithm)
- capacity: Logistic growth upper bound (Prophet style)
- ratio: Consecutive value ratio constraint (growth rate limit)
- custom: User-defined function

Usage:
    >>> from vectrix.adaptive.constraints import ConstraintAwareForecaster, Constraint
    >>> caf = ConstraintAwareForecaster()
    >>> result = caf.apply(predictions, lower95, upper95, constraints=[
    ...     Constraint('non_negative', {}),
    ...     Constraint('range', {'min': 100, 'max': 5000}),
    ...     Constraint('sum_constraint', {'window': 7, 'maxSum': 10000}),
    ...     Constraint('monotone', {'direction': 'increasing'}),
    ... ])
    >>> print(result.violationsBefore, '->', result.violationsAfter)
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class Constraint:
    """
    Forecast constraint definition.

    Parameters
    ----------
    type : str
        Constraint type. Supported: 'non_negative', 'range', 'sum_constraint',
        'yoy_change', 'monotone', 'capacity', 'ratio', 'custom'
    params : Dict
        Constraint parameters. Varies by type.

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
    Constraint application result.

    Attributes
    ----------
    predictions : np.ndarray
        Predictions after constraint application
    lower95 : np.ndarray
        Lower 95% confidence interval after constraint application
    upper95 : np.ndarray
        Upper 95% confidence interval after constraint application
    violationsBefore : int
        Number of violations before constraint application
    violationsAfter : int
        Number of violations after constraint application (should be 0)
    constraintsApplied : List[str]
        List of applied constraint types
    adjustmentDetails : List[Dict]
        Adjustment details per constraint (modified indices, adjustment amounts, etc.)
    """
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    lower95: np.ndarray = field(default_factory=lambda: np.array([]))
    upper95: np.ndarray = field(default_factory=lambda: np.array([]))
    violationsBefore: int = 0
    violationsAfter: int = 0
    constraintsApplied: List[str] = field(default_factory=list)
    adjustmentDetails: List[Dict] = field(default_factory=list)


# Priority by constraint type (hard constraints first)
_HARD_CONSTRAINTS = {'non_negative', 'range', 'capacity'}
_SOFT_CONSTRAINTS = {'sum_constraint', 'yoy_change', 'monotone', 'ratio', 'custom'}


class ConstraintAwareForecaster:
    """
    Constraint-aware forecaster.

    Consistently applies constraints to predictions and confidence intervals.
    Processes hard constraints (non_negative, range, capacity) first, then
    soft constraints (sum, yoy_change, monotone, ratio, custom) in order.
    Finally applies smooth transitions to mitigate artificial discontinuities
    at correction boundaries.

    Correction strategies:
    - Projection: Projects to nearest valid point on violation (range, capacity)
    - Smooth: Smoothly connects before/after violation intervals (monotone)
    - Redistribute: Proportionally distributes excess to neighboring points (sum_constraint)

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
        Apply constraints in order.

        Application order:
        1. Hard constraints (non_negative, range, capacity) first
        2. Soft constraints (sum, yoy_change, monotone, ratio, custom) second
        3. If smoothing=True, finish with smooth interpolation

        Parameters
        ----------
        predictions : np.ndarray
            Original predictions
        lower95 : np.ndarray
            Original lower 95% confidence interval
        upper95 : np.ndarray
            Original upper 95% confidence interval
        constraints : List[Constraint]
            List of constraints to apply
        smoothing : bool
            Whether to apply smooth transitions after correction

        Returns
        -------
        ConstraintResult
            Constraint application result
        """
        pred = np.array(predictions, dtype=np.float64).copy()
        lo = np.array(lower95, dtype=np.float64).copy()
        hi = np.array(upper95, dtype=np.float64).copy()
        originalPred = pred.copy()

        # Violation count before application
        violationsBefore = self._countViolations(pred, constraints)

        # Separate hard/soft and apply in order
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

        # Confidence interval consistency: lo <= pred <= hi
        lo = np.minimum(lo, pred)
        hi = np.maximum(hi, pred)

        # Smooth transitions
        if smoothing:
            adjustedMask = np.abs(pred - originalPred) > 1e-10
            if np.any(adjustedMask):
                pred = self._smoothTransitions(pred, originalPred, adjustedMask)
                # Re-verify hard constraints after smoothing
                for constraint in hardConstraints:
                    try:
                        pred, lo, hi, _ = self._applyOne(pred, lo, hi, constraint)
                    except Exception:
                        pass
                lo = np.minimum(lo, pred)
                hi = np.maximum(hi, pred)

        # Violation count after application
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
        """Single constraint application dispatcher."""
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
                raise ValueError("yoy_change constraint requires 'historicalData' parameter.")
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
                raise ValueError("custom constraint requires 'fn' parameter (callable).")
            p, l, h = self._applyCustom(pred, lo, hi, fn)
            detail = {'constraint': 'custom'}
        else:
            raise ValueError(f"Unsupported constraint type: {cType}")

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
        Non-negativity constraint: clip negatives to 0.

        Corrects both predictions and lower confidence interval to >= 0.
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
        Range constraint: min <= prediction <= max.

        Projects to nearest valid point (projection).
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
        Logistic growth upper/lower bound (Prophet style).

        Instead of simple clipping, applies logistic damping near the
        upper bound so predictions converge smoothly to the capacity.

        Near the capacity (top 10% range), logistic compression is applied:
        adjusted = floor + (capacity - floor) * sigmoid(scaled)
        """
        p = pred.copy()
        l = lo.copy()
        h = hi.copy()

        if capacity < np.inf:
            # Logistic damping near upper bound
            effectiveRange = capacity - floor if floor > -np.inf else capacity
            threshold = capacity - 0.1 * effectiveRange  # Top 10%

            mask = p > threshold
            if np.any(mask):
                # Logistic compression: smooth saturation near cap
                overshoot = (p[mask] - threshold) / max(effectiveRange * 0.1, 1e-10)
                # Sigmoid mapping: 0..inf -> 0..1
                compressed = 1.0 / (1.0 + np.exp(-overshoot + 2.0))
                p[mask] = threshold + (capacity - threshold) * compressed

        if floor > -np.inf:
            # Logistic damping near lower bound (symmetric)
            effectiveRange = capacity - floor if capacity < np.inf else abs(floor) * 2
            threshold = floor + 0.1 * effectiveRange

            mask = p < threshold
            if np.any(mask):
                undershoot = (threshold - p[mask]) / max(effectiveRange * 0.1, 1e-10)
                compressed = 1.0 / (1.0 + np.exp(-undershoot + 2.0))
                p[mask] = threshold - (threshold - floor) * compressed

        # Final safety clipping
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
        Window sum constraint.

        If the sum over a window period exceeds maxSum, values within that
        window are proportionally scaled down. If below minSum, they are
        proportionally scaled up.

        Proportional scaling preserves relative magnitudes while matching the sum.
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
                # Proportional scale down
                ratio = maxSum / currentSum
                p[windowSlice] *= ratio
                l[windowSlice] *= ratio
                h[windowSlice] *= ratio
            elif currentSum < minSum and abs(currentSum) > 1e-10:
                # Proportional scale up
                ratio = minSum / currentSum
                p[windowSlice] *= ratio
                l[windowSlice] *= ratio
                h[windowSlice] *= ratio
            elif currentSum < minSum and abs(currentSum) <= 1e-10:
                # Sum near zero but minSum is positive: distribute equally
                windowLen = end - start
                fillVal = minSum / windowLen
                p[windowSlice] = fillVal
                l[windowSlice] = fillVal * 0.8
                h[windowSlice] = fillVal * 1.2

        lower = np.minimum(l, h)
        upper = np.maximum(l, h)
        return p, lower, upper

    def _applyYoYChange(
        self,
        pred: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        maxPct: float,
        historicalData: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Year-over-year change limit constraint.

        Compares against the same period from the prior year (historicalData)
        and limits each point's prediction to within maxPct% of the
        corresponding historical value.

        historicalData length must be >= pred length.
        Uses historicalData[-len(pred):] as the year-ago reference.

        Parameters
        ----------
        maxPct : float
            Maximum allowed change rate (%)
        historicalData : np.ndarray
            Prior year data (length >= pred)
        """
        p = pred.copy()
        l = lo.copy()
        h = hi.copy()
        n = len(p)

        # Extract year-ago reference from historicalData
        histLen = len(historicalData)
        if histLen < n:
            # Insufficient data: apply only what is available
            refData = np.full(n, np.nan)
            refData[:histLen] = historicalData
        else:
            refData = historicalData[-n:]

        maxRatio = maxPct / 100.0

        for i in range(n):
            ref = refData[i]
            if np.isnan(ref) or abs(ref) < 1e-10:
                continue

            # Compute allowed range
            allowedMin = ref * (1.0 - maxRatio)
            allowedMax = ref * (1.0 + maxRatio)

            # Handle negative reference values (swap min/max)
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
        Monotone constraint (isotonic regression via PAVA).

        direction='increasing': Monotone increasing constraint (y[i] <= y[i+1])
        direction='decreasing': Monotone decreasing constraint (y[i] >= y[i+1])

        Uses the Pool Adjacent Violators (PAVA) algorithm to compute the
        optimal monotone approximation in the least squares sense.
        """
        increasing = (direction == 'increasing')
        p = self._pava(pred, increasing=increasing)
        l = self._pava(lo, increasing=increasing)
        h = self._pava(hi, increasing=increasing)

        # Ensure consistency
        l = np.minimum(l, p)
        h = np.maximum(h, p)
        return p, l, h

    def _pava(self, y: np.ndarray, increasing: bool = True) -> np.ndarray:
        """
        Isotonic Regression via Pool Adjacent Violators Algorithm (PAVA).

        Finds result that minimizes ||y - result||^2 subject to the
        monotone increasing constraint.

        Algorithm:
        1. Scan y sequentially
        2. If y[i] < y[i-1], violation -> pool both values to their mean
        3. If pooled value violates previous, continue pooling
        4. Repeat until all violations are resolved

        Parameters
        ----------
        y : np.ndarray
            Input array
        increasing : bool
            True for monotone increasing, False for monotone decreasing

        Returns
        -------
        np.ndarray
            Optimal approximation satisfying the monotone constraint
        """
        n = len(y)
        if n <= 1:
            return y.copy()

        result = y.astype(np.float64).copy()
        if not increasing:
            result = -result

        # Block-based PAVA
        blocks: List[List[int]] = [[i] for i in range(n)]
        blockMeans: List[float] = [float(result[i]) for i in range(n)]

        merged = True
        while merged:
            merged = False
            newBlocks: List[List[int]] = [blocks[0]]
            newMeans: List[float] = [blockMeans[0]]

            for i in range(1, len(blocks)):
                if blockMeans[i] < newMeans[-1]:
                    # Violation -> merge
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

        # Assign results
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
        Consecutive value ratio constraint: minRatio <= pred[t+1]/pred[t] <= maxRatio.

        Traverses forward from t=0. If t+1 exceeds the ratio range, adjusts
        to the allowed boundary. Prior adjustments propagate to subsequent values.
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
        Apply user-defined constraint function.

        fn signature: fn(pred, lo, hi) -> (pred, lo, hi)
        Each is np.ndarray.
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
        Smooth transition between corrected and original predictions.

        Applies Gaussian-weighted blending at the boundaries of adjusted
        regions to mitigate artificial discontinuities.

        After boundary detection, within transition zones (radius=3):
        weight = exp(-dist^2 / (2 * sigma^2))
        blended = weight * adjusted + (1 - weight) * original

        Parameters
        ----------
        pred : np.ndarray
            Corrected predictions
        originalPred : np.ndarray
            Original predictions before correction
        adjustedMask : np.ndarray (bool)
            Mask of time points where correction was applied
        """
        result = pred.copy()
        n = len(pred)
        radius = min(3, n // 4)
        if radius < 1:
            return result

        sigma = max(radius / 2.0, 1.0)

        # Boundary detection: change points in adjustedMask
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
                    # Adjusted point: retain adjusted value far from boundary,
                    # blend with original near boundary
                    dist = abs(offset)
                    weight = np.exp(-dist ** 2 / (2.0 * sigma ** 2))
                    # For adjusted points, higher weight means more original blending (near boundary)
                    blend = weight * 0.3  # Up to 30% original blending
                    result[idx] = (1.0 - blend) * pred[idx] + blend * originalPred[idx]

        return result

    def _countViolations(
        self,
        pred: np.ndarray,
        constraints: List[Constraint]
    ) -> int:
        """
        Count constraint violations.

        Returns the total number of time points violating any constraint.
        (If a single point violates multiple constraints, it is counted multiple times)
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
        """Count violations for a single constraint."""
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
            # Cannot predetermine violation count for custom constraints
            return 0

        return 0
