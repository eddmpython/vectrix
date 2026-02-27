"""
What-If Scenario Analysis

시나리오 시뮬레이션:
- 추세 변화: "추세가 10% 증가하면?"
- 계절성 변화: "계절 패턴이 사라지면?"
- 충격 이벤트: "30일차에 -20% 충격이 발생하면?"
- 수준 이동: "전체 수준이 5% 상승하면?"
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Scenario:
    """시나리오 정의"""
    name: str = ""
    trendChange: float = 0.0
    seasonalMultiplier: float = 1.0
    shockAt: Optional[int] = None
    shockMagnitude: float = 0.0
    shockDuration: int = 1
    levelShift: float = 0.0


@dataclass
class ScenarioResult:
    """시나리오 결과"""
    name: str = ""
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    baselinePredictions: np.ndarray = field(default_factory=lambda: np.array([]))
    difference: np.ndarray = field(default_factory=lambda: np.array([]))
    percentChange: np.ndarray = field(default_factory=lambda: np.array([]))
    impact: float = 0.0


class WhatIfAnalyzer:
    """
    What-If 시나리오 분석기

    Usage:
        >>> analyzer = WhatIfAnalyzer()
        >>> results = analyzer.analyze(
        ...     basePredictions,
        ...     scenarios=[
        ...         {'name': 'optimistic', 'trend_change': 0.1},
        ...         {'name': 'shock', 'shock_at': 10, 'shock_magnitude': -0.2}
        ...     ]
        ... )
    """

    def analyze(
        self,
        basePredictions: np.ndarray,
        historicalData: np.ndarray,
        scenarios: List[Dict[str, Any]],
        period: int = 7
    ) -> List[ScenarioResult]:
        """
        시나리오 분석

        Parameters
        ----------
        basePredictions : np.ndarray
            기본 예측
        historicalData : np.ndarray
            과거 데이터
        scenarios : List[Dict]
            시나리오 정의 리스트
        period : int
            계절 주기
        """
        results = []

        for scenarioDict in scenarios:
            scenario = self._parseScenario(scenarioDict)
            modified = self._applyScenario(basePredictions.copy(), historicalData, scenario, period)

            diff = modified - basePredictions
            pctChange = np.zeros_like(diff)
            mask = np.abs(basePredictions) > 1e-10
            pctChange[mask] = diff[mask] / basePredictions[mask] * 100

            results.append(ScenarioResult(
                name=scenario.name,
                predictions=modified,
                baselinePredictions=basePredictions.copy(),
                difference=diff,
                percentChange=pctChange,
                impact=float(np.mean(np.abs(pctChange)))
            ))

        return results

    def _parseScenario(self, d: Dict[str, Any]) -> Scenario:
        return Scenario(
            name=d.get('name', 'unnamed'),
            trendChange=d.get('trend_change', 0.0),
            seasonalMultiplier=d.get('seasonal_multiplier', 1.0),
            shockAt=d.get('shock_at', None),
            shockMagnitude=d.get('shock_magnitude', 0.0),
            shockDuration=d.get('shock_duration', 1),
            levelShift=d.get('level_shift', 0.0),
        )

    def _applyScenario(
        self,
        predictions: np.ndarray,
        historicalData: np.ndarray,
        scenario: Scenario,
        period: int
    ) -> np.ndarray:
        steps = len(predictions)

        if scenario.trendChange != 0:
            trendDelta = np.arange(1, steps + 1) * scenario.trendChange * np.std(historicalData[-30:])
            predictions += trendDelta

        if scenario.seasonalMultiplier != 1.0 and period > 1:
            mean = np.mean(predictions)
            seasonal = predictions - mean
            predictions = mean + seasonal * scenario.seasonalMultiplier

        if scenario.shockAt is not None and 0 <= scenario.shockAt < steps:
            shockStart = scenario.shockAt
            shockEnd = min(shockStart + scenario.shockDuration, steps)
            for i in range(shockStart, shockEnd):
                predictions[i] *= (1 + scenario.shockMagnitude)

            decay = 0.8
            for i in range(shockEnd, steps):
                factor = scenario.shockMagnitude * (decay ** (i - shockEnd + 1))
                predictions[i] *= (1 + factor)

        if scenario.levelShift != 0:
            predictions *= (1 + scenario.levelShift)

        return predictions

    def compareSummary(self, results: List[ScenarioResult], locale: str = 'ko') -> str:
        if not results:
            return "시나리오 없음"

        lines = ["시나리오 비교 분석:" if locale == 'ko' else "Scenario Comparison:"]

        for r in sorted(results, key=lambda x: -abs(x.impact)):
            if locale == 'ko':
                lines.append(
                    f"  [{r.name}] 평균 영향: {r.impact:.1f}%, "
                    f"최종값 변화: {r.percentChange[-1]:.1f}%"
                )
            else:
                lines.append(
                    f"  [{r.name}] Avg impact: {r.impact:.1f}%, "
                    f"Final change: {r.percentChange[-1]:.1f}%"
                )

        return '\n'.join(lines)
