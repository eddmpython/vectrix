"""
Hierarchical Reconciliation Methods

계층적 시계열 (예: 전국 → 지역 → 매장) 예측 조정.

S matrix: summing matrix (계층 구조 정의)
y_tilde = S @ P @ y_hat  (조정된 예측)
"""

from typing import Dict, List, Optional

import numpy as np


class BottomUp:
    """
    Bottom-Up Reconciliation

    하위 시계열 예측을 합산하여 상위 예측 생성.
    가장 단순하고 안전한 방법.
    """

    def reconcile(
        self,
        bottomForecasts: np.ndarray,
        summingMatrix: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
        bottomForecasts : np.ndarray
            하위 시계열 예측 [nBottom, steps]
        summingMatrix : np.ndarray
            합산 행렬 S [nTotal, nBottom]

        Returns
        -------
        np.ndarray
            조정된 전체 예측 [nTotal, steps]
        """
        return summingMatrix @ bottomForecasts


class TopDown:
    """
    Top-Down Reconciliation

    상위 예측을 비율에 따라 하위로 배분.

    Methods:
    - 'proportions': 과거 비율 기반
    - 'forecast_proportions': 예측 비율 기반
    """

    def __init__(self, method: str = 'proportions'):
        self.method = method

    def reconcile(
        self,
        topForecast: np.ndarray,
        proportions: np.ndarray,
        summingMatrix: np.ndarray
    ) -> np.ndarray:
        """
        Parameters
        ----------
        topForecast : np.ndarray
            상위 예측 [steps]
        proportions : np.ndarray
            배분 비율 [nBottom]
        summingMatrix : np.ndarray
            합산 행렬 S [nTotal, nBottom]

        Returns
        -------
        np.ndarray
            조정된 전체 예측 [nTotal, steps]
        """
        nBottom = len(proportions)
        steps = len(topForecast)

        bottomForecasts = np.zeros((nBottom, steps))
        for i in range(nBottom):
            bottomForecasts[i] = topForecast * proportions[i]

        return summingMatrix @ bottomForecasts

    @staticmethod
    def computeProportions(historicalBottom: np.ndarray) -> np.ndarray:
        """
        과거 데이터로 배분 비율 계산

        Parameters
        ----------
        historicalBottom : np.ndarray
            하위 과거 데이터 [nBottom, T]

        Returns
        -------
        np.ndarray
            비율 [nBottom]
        """
        totals = np.sum(historicalBottom, axis=1)
        grandTotal = np.sum(totals)
        if grandTotal > 0:
            return totals / grandTotal
        return np.ones(len(totals)) / len(totals)


class MinTrace:
    """
    MinTrace Reconciliation (Wickramasuriya et al., 2019)

    최소 분산 조정으로 계층 전체의 예측 오차를 최소화.

    y_tilde = S @ (S'W^{-1}S)^{-1} S'W^{-1} @ y_hat

    Methods:
    - 'ols': W = I (단위 행렬)
    - 'wls': W = diag(residual variances)
    """

    def __init__(self, method: str = 'ols'):
        self.method = method

    def reconcile(
        self,
        forecasts: np.ndarray,
        summingMatrix: np.ndarray,
        residuals: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Parameters
        ----------
        forecasts : np.ndarray
            전체 (조정 전) 예측 [nTotal, steps]
        summingMatrix : np.ndarray
            합산 행렬 S [nTotal, nBottom]
        residuals : np.ndarray, optional
            잔차 [nTotal, T] (WLS에 필요)

        Returns
        -------
        np.ndarray
            조정된 전체 예측 [nTotal, steps]
        """
        S = summingMatrix
        nTotal, nBottom = S.shape

        if self.method == 'wls' and residuals is not None:
            W = np.diag(np.var(residuals, axis=1) + 1e-10)
        else:
            W = np.eye(nTotal)

        try:
            WInv = np.linalg.inv(W)
            G = np.linalg.solve(S.T @ WInv @ S, S.T @ WInv)
            P = S @ G
            return P @ forecasts
        except np.linalg.LinAlgError:
            return forecasts

    @staticmethod
    def buildSummingMatrix(
        structure: Dict[str, List[str]]
    ) -> np.ndarray:
        """
        계층 구조에서 합산 행렬 S 생성

        Parameters
        ----------
        structure : Dict[str, List[str]]
            {상위노드: [하위노드들]}
            예: {'total': ['A', 'B'], 'A': ['A1', 'A2'], 'B': ['B1', 'B2']}

        Returns
        -------
        np.ndarray
            합산 행렬 S
        """
        allNodes = set()
        bottomNodes = set()
        parentNodes = set()

        for parent, children in structure.items():
            allNodes.add(parent)
            parentNodes.add(parent)
            for child in children:
                allNodes.add(child)

        for node in allNodes:
            if node not in parentNodes:
                bottomNodes.add(node)

        bottomList = sorted(bottomNodes)
        nBottom = len(bottomList)
        bottomIdx = {name: i for i, name in enumerate(bottomList)}

        allList = sorted(allNodes)
        nTotal = len(allList)

        S = np.zeros((nTotal, nBottom))

        for i, node in enumerate(allList):
            if node in bottomNodes:
                S[i, bottomIdx[node]] = 1.0
            else:
                descendantBottoms = _getDescendantBottoms(node, structure, bottomNodes)
                for b in descendantBottoms:
                    S[i, bottomIdx[b]] = 1.0

        return S


def _getDescendantBottoms(
    node: str,
    structure: Dict[str, List[str]],
    bottomNodes: set
) -> List[str]:
    if node in bottomNodes:
        return [node]

    result = []
    if node in structure:
        for child in structure[node]:
            result.extend(_getDescendantBottoms(child, structure, bottomNodes))
    return result
