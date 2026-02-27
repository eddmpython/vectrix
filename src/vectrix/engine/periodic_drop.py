"""
정기 드롭 패턴 감지 및 처리

E009 실험 결과: 61.3% 개선 (5.87% → 2.27%)

사용 사례:
- 제조 공장 정기 점검 (90일 주기 1주일 가동 중단)
- 시스템 유지보수 윈도우
- 정기적으로 반복되는 저하 패턴
"""

import numpy as np
from typing import List, Tuple, Optional


class PeriodicDropDetector:
    """
    정기 드롭 패턴 감지 및 처리

    알고리즘:
    1. 이동평균 대비 급락 구간 감지
    2. 드롭 구간의 주기성 분석
    3. 드롭 구간을 선형 보간으로 대체 (학습용)
    4. 예측값에 드롭 패턴 재적용
    """

    def __init__(self, minDropRatio: float = 0.8, minDropDuration: int = 3):
        """
        Args:
            minDropRatio: 이동평균 대비 이 비율 미만이면 드롭으로 판정 (0.8 = 20% 하락)
            minDropDuration: 최소 연속 드롭 일수
        """
        self.minDropRatio = minDropRatio
        self.minDropDuration = minDropDuration
        self.detectedDrops: List[Tuple[int, int]] = []
        self.dropPeriod: Optional[int] = None
        self.dropDuration: Optional[int] = None
        self.dropRatio: Optional[float] = None

    def detect(self, y: np.ndarray) -> bool:
        """
        드롭 패턴 감지

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
        """드롭 구간을 선형 보간으로 대체"""
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
        """예측값에 드롭 패턴 적용"""
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
        """정기 드롭 패턴이 감지되었는지 확인"""
        return self.dropPeriod is not None and self.dropRatio is not None

    def willDropOccurInPrediction(self, trainLength: int, predSteps: int) -> Tuple[bool, int]:
        """
        예측 구간에 드롭이 발생할지 계산

        Returns:
            (발생 여부, 첫 드롭 시작 offset)
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
        예측 구간에 드롭이 발생할 때만 패턴 적용

        기존 applyDropPattern과 달리, 예측 구간 내 드롭 발생 여부를 먼저 확인
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
