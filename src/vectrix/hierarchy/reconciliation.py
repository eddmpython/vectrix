"""
Hierarchical Reconciliation Methods

Hierarchical time series (e.g., national -> regional -> store) forecast reconciliation.

S matrix: summing matrix (defines hierarchical structure)
y_tilde = S @ P @ y_hat  (reconciled forecasts)
"""

from typing import Dict, List, Optional

import numpy as np


class BottomUp:
    """
    Bottom-Up Reconciliation

    Aggregates bottom-level forecasts to produce upper-level forecasts.
    The simplest and safest method.
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
            Bottom-level forecasts [nBottom, steps]
        summingMatrix : np.ndarray
            Summing matrix S [nTotal, nBottom]

        Returns
        -------
        np.ndarray
            Reconciled forecasts [nTotal, steps]
        """
        return summingMatrix @ bottomForecasts


class TopDown:
    """
    Top-Down Reconciliation

    Distributes top-level forecasts to bottom levels by proportions.

    Methods:
    - 'proportions': Based on historical proportions
    - 'forecast_proportions': Based on forecast proportions
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
            Top-level forecast [steps]
        proportions : np.ndarray
            Distribution proportions [nBottom]
        summingMatrix : np.ndarray
            Summing matrix S [nTotal, nBottom]

        Returns
        -------
        np.ndarray
            Reconciled forecasts [nTotal, steps]
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
        Compute distribution proportions from historical data

        Parameters
        ----------
        historicalBottom : np.ndarray
            Historical bottom-level data [nBottom, T]

        Returns
        -------
        np.ndarray
            Proportions [nBottom]
        """
        totals = np.sum(historicalBottom, axis=1)
        grandTotal = np.sum(totals)
        if grandTotal > 0:
            return totals / grandTotal
        return np.ones(len(totals)) / len(totals)


class MinTrace:
    """
    MinTrace Reconciliation (Wickramasuriya et al., 2019)

    Minimizes forecast error variance across the entire hierarchy.

    y_tilde = S @ (S'W^{-1}S)^{-1} S'W^{-1} @ y_hat

    Methods:
    - 'ols': W = I (identity matrix)
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
            Base (unreconciled) forecasts [nTotal, steps]
        summingMatrix : np.ndarray
            Summing matrix S [nTotal, nBottom]
        residuals : np.ndarray, optional
            Residuals [nTotal, T] (required for WLS)

        Returns
        -------
        np.ndarray
            Reconciled forecasts [nTotal, steps]
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
        Build summing matrix S from hierarchical structure

        Parameters
        ----------
        structure : Dict[str, List[str]]
            {parent_node: [child_nodes]}
            e.g., {'total': ['A', 'B'], 'A': ['A1', 'A2'], 'B': ['B1', 'B2']}

        Returns
        -------
        np.ndarray
            Summing matrix S
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
