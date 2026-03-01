"""
Periodic Drop Pattern Detection and Handling

E009 experiment result: 61.3% improvement (5.87% -> 2.27%)

Use cases:
- Manufacturing plant scheduled maintenance (90-day cycle, 1-week shutdown)
- System maintenance windows
- Regularly recurring degradation patterns
"""

from typing import List, Optional, Tuple

import numpy as np


class PeriodicDropDetector:
    """
    Periodic drop pattern detection and handling

    Algorithm:
    1. Detect sharp decline segments relative to moving average
    2. Analyze periodicity of drop segments
    3. Replace drop segments with linear interpolation (for training)
    4. Reapply drop pattern to forecast values
    """

    def __init__(self, minDropRatio: float = 0.8, minDropDuration: int = 3):
        """
        Args:
            minDropRatio: Classified as drop when below this ratio relative to moving average (0.8 = 20% decline)
            minDropDuration: Minimum consecutive drop days
        """
        self.minDropRatio = minDropRatio
        self.minDropDuration = minDropDuration
        self.detectedDrops: List[Tuple[int, int]] = []
        self.dropPeriod: Optional[int] = None
        self.dropDuration: Optional[int] = None
        self.dropRatio: Optional[float] = None

    def detect(self, y: np.ndarray) -> bool:
        """
        Detect drop patterns

        Returns:
            True if periodic drop pattern detected
        """
        n = len(y)
        if n < 30:
            return False

        window = min(21, max(7, n // 20))

        ma = np.zeros(n)
        halfWin = window // 2
        for i in range(n):
            start = max(0, i - halfWin)
            end = min(n, i + halfWin + 1)
            ma[i] = np.mean(y[start:end])

        globalMean = np.mean(y)
        globalStd = np.std(y)
        threshold = globalMean - 1.5 * globalStd

        dropMask = (y < ma * self.minDropRatio) | (y < threshold)

        drops = []
        inDrop = False
        dropStart = 0

        for i in range(n):
            if dropMask[i] and not inDrop:
                inDrop = True
                dropStart = i
            elif not dropMask[i] and inDrop:
                inDrop = False
                if i - dropStart >= self.minDropDuration:
                    drops.append((dropStart, i))

        if inDrop and n - dropStart >= self.minDropDuration:
            drops.append((dropStart, n))

        self.detectedDrops = drops

        if len(drops) >= 2:
            intervals = [drops[i+1][0] - drops[i][0] for i in range(len(drops)-1)]
            if intervals:
                medianInterval = np.median(intervals)
                stdInterval = np.std(intervals)
                if stdInterval / medianInterval < 0.3:
                    self.dropPeriod = int(medianInterval)
                    durations = [end - start for start, end in drops]
                    self.dropDuration = int(np.median(durations))

                    dropValues = []
                    normalValues = []
                    for start, end in drops:
                        dropValues.extend(y[start:end])
                        normalStart = max(0, start - 7)
                        normalValues.extend(y[normalStart:start])

                    if normalValues and np.mean(normalValues) > 0:
                        self.dropRatio = np.mean(dropValues) / np.mean(normalValues)
                        return True

        return False

    def removeDrops(self, y: np.ndarray) -> np.ndarray:
        """Replace drop segments with linear interpolation"""
        result = y.copy()

        for start, end in self.detectedDrops:
            beforeVal = y[max(0, start-1)]
            afterIdx = min(len(y)-1, end)
            afterVal = y[afterIdx] if afterIdx < len(y) else beforeVal

            for i in range(start, min(end, len(y))):
                ratio = (i - start) / max(1, end - start)
                result[i] = beforeVal + ratio * (afterVal - beforeVal)

        return result

    def applyDropPattern(self, predictions: np.ndarray, startIdx: int) -> np.ndarray:
        """Apply drop pattern to forecast values"""
        if self.dropPeriod is None or self.dropRatio is None:
            return predictions

        result = predictions.copy()
        n = len(predictions)

        if self.detectedDrops:
            lastDropStart = self.detectedDrops[-1][0]
            nextDropOffset = self.dropPeriod - ((startIdx - lastDropStart) % self.dropPeriod)
        else:
            nextDropOffset = self.dropPeriod

        i = nextDropOffset
        while i < n:
            dropEnd = min(i + (self.dropDuration or 7), n)
            result[i:dropEnd] *= self.dropRatio
            i += self.dropPeriod

        return result

    def hasPeriodicDrop(self) -> bool:
        """Check if periodic drop pattern was detected"""
        return self.dropPeriod is not None and self.dropRatio is not None

    def willDropOccurInPrediction(self, trainLength: int, predSteps: int) -> Tuple[bool, int]:
        """
        Calculate whether a drop will occur in the forecast horizon

        Returns:
            (whether it occurs, first drop start offset)
        """
        if not self.hasPeriodicDrop() or not self.detectedDrops:
            return False, -1

        lastDropStart = self.detectedDrops[-1][0]
        nextDropStart = lastDropStart + self.dropPeriod

        predStart = trainLength
        predEnd = trainLength + predSteps

        while nextDropStart < predEnd:
            dropEnd = nextDropStart + (self.dropDuration or 7)
            if nextDropStart < predEnd and dropEnd > predStart:
                return True, nextDropStart - predStart
            nextDropStart += self.dropPeriod

        return False, -1

    def applyDropPatternSmart(self, predictions: np.ndarray, trainLength: int) -> np.ndarray:
        """
        Apply drop pattern only when a drop will occur in the forecast horizon

        Unlike applyDropPattern, first checks whether a drop occurs within the forecast period
        """
        willOccur, firstOffset = self.willDropOccurInPrediction(trainLength, len(predictions))

        if not willOccur:
            return predictions

        result = predictions.copy()
        n = len(predictions)

        i = firstOffset
        while i < n and i >= 0:
            dropEnd = min(i + (self.dropDuration or 7), n)
            if dropEnd > i:
                result[i:dropEnd] *= self.dropRatio
            i += self.dropPeriod

        return result
