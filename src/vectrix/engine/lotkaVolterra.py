"""
Lotka-Volterra Ensemble Forecaster

Ecological competition dynamics applied to ensemble forecasting.
Each model acts as a species competing for weight (resource) allocation
based on fitness (accuracy). Residual correlation determines competition
vs symbiosis between model pairs.

Lotka-Volterra competition ODE:
    dN_i/dt = r_i * N_i * (1 - sum(alpha_ij * N_j) / K)

where N_i = weight of model i, r_i = fitness-based growth rate,
alpha_ij = competition coefficient, K = carrying capacity.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_EULER_STEPS = 150
EXTINCTION_THRESHOLD = 1e-6
INITIAL_WEIGHT_FLOOR = 0.01


def _buildDefaultModels() -> List[Tuple[str, Callable]]:
    from .arima import ARIMAModel
    from .baselines import NaiveModel, SeasonalNaiveModel
    from .ets import ETSModel
    from .theta import ThetaModel

    return [
        ("ETS", lambda: ETSModel()),
        ("ARIMA", lambda: ARIMAModel()),
        ("Theta", lambda: ThetaModel()),
        ("Naive", lambda: NaiveModel()),
        ("SeasNaive", lambda: SeasonalNaiveModel()),
    ]


def _computeRMSE(residuals: np.ndarray) -> float:
    if len(residuals) == 0:
        return np.inf
    return float(np.sqrt(np.mean(residuals ** 2)))


def _shannonDiversity(weights: np.ndarray) -> float:
    safeWeights = weights[weights > EXTINCTION_THRESHOLD]
    if len(safeWeights) == 0:
        return 0.0
    normalized = safeWeights / np.sum(safeWeights)
    return float(-np.sum(normalized * np.log(normalized + 1e-15)))


class LotkaVolterraEnsemble:
    """
    Ecological competition dynamics ensemble.
    Each model is a species competing for weight allocation
    based on predictive fitness via Lotka-Volterra ODEs.
    """

    def __init__(
        self,
        models: Optional[List[Tuple[str, Callable]]] = None,
        competitionRate: float = 0.1,
        growthRate: float = 0.5,
        carryingCapacity: float = 1.0,
        symbiosisThreshold: float = 0.3
    ):
        self._modelSpecs = models
        self.competitionRate = competitionRate
        self.growthRate = growthRate
        self.carryingCapacity = carryingCapacity
        self.symbiosisThreshold = symbiosisThreshold

        self._fittedModels: List[Any] = []
        self._modelNames: List[str] = []
        self._weights: np.ndarray = np.array([])
        self._fitnessList: np.ndarray = np.array([])
        self._competitionMatrix: np.ndarray = np.array([])
        self._residualMatrix: np.ndarray = np.array([])
        self._isFitted = False
        self._y: np.ndarray = np.array([])

    def fit(self, y: np.ndarray, period: int = 1) -> 'LotkaVolterraEnsemble':
        """Fit all sub-models, compute fitness, competition matrix, and solve LV ODE for weights."""
        self._y = y.copy()
        modelSpecs = self._modelSpecs if self._modelSpecs is not None else _buildDefaultModels()

        nModels = len(modelSpecs)
        self._fittedModels = []
        self._modelNames = []
        residualsList = []

        for name, factory in modelSpecs:
            try:
                model = factory()
                model.fit(y)
                pred, _, _ = model.predict(1)
                self._fittedModels.append(model)
                self._modelNames.append(name)

                if hasattr(model, 'residuals') and model.residuals is not None and len(model.residuals) > 0:
                    residualsList.append(model.residuals.copy())
                else:
                    fittedVals = np.zeros(len(y))
                    fittedVals[0] = y[0]
                    for t in range(1, len(y)):
                        fittedVals[t] = y[t - 1]
                    residualsList.append(y - fittedVals)
            except Exception:
                continue

        nAlive = len(self._fittedModels)
        if nAlive == 0:
            from .baselines import NaiveModel
            fallback = NaiveModel()
            fallback.fit(y)
            self._fittedModels = [fallback]
            self._modelNames = ["Naive"]
            self._weights = np.array([1.0])
            self._fitnessList = np.array([1.0])
            self._competitionMatrix = np.array([[0.0]])
            self._isFitted = True
            return self

        minLen = min(len(r) for r in residualsList)
        self._residualMatrix = np.column_stack([r[-minLen:] for r in residualsList])

        self._fitnessList = np.zeros(nAlive)
        for i in range(nAlive):
            rmse = _computeRMSE(self._residualMatrix[:, i])
            self._fitnessList[i] = 1.0 / (1.0 + rmse)

        self._competitionMatrix = self._buildCompetitionMatrix()

        self._weights = self._solveLotkaVolterraODE()

        self._isFitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Weighted ensemble prediction with variance-based confidence intervals."""
        if not self._isFitted:
            raise ValueError("Model not fitted.")

        nModels = len(self._fittedModels)
        allPreds = np.zeros((nModels, steps))
        allLower = np.zeros((nModels, steps))
        allUpper = np.zeros((nModels, steps))

        for i, model in enumerate(self._fittedModels):
            try:
                p, lo, up = model.predict(steps)
                allPreds[i] = p[:steps]
                allLower[i] = lo[:steps]
                allUpper[i] = up[:steps]
            except Exception:
                allPreds[i] = np.full(steps, self._y[-1])
                sigma = np.std(self._y) if len(self._y) > 1 else 1.0
                margin = 1.96 * sigma * np.sqrt(np.arange(1, steps + 1))
                allLower[i] = allPreds[i] - margin
                allUpper[i] = allPreds[i] + margin

        w = self._weights
        predictions = np.zeros(steps)
        for i in range(nModels):
            predictions += w[i] * allPreds[i]

        weightedVar = np.zeros(steps)
        for t in range(steps):
            meanPred = predictions[t]
            for i in range(nModels):
                weightedVar[t] += w[i] * (allPreds[i, t] - meanPred) ** 2

        intrinsicVar = np.zeros(steps)
        for i in range(nModels):
            modelRange = allUpper[i] - allLower[i]
            modelSigma = modelRange / (2 * 1.96)
            intrinsicVar += w[i] * (modelSigma ** 2)

        totalSigma = np.sqrt(weightedVar + intrinsicVar)
        lower95 = predictions - 1.96 * totalSigma
        upper95 = predictions + 1.96 * totalSigma

        return predictions, lower95, upper95

    def getEcosystemState(self) -> Dict[str, Any]:
        """Return ecosystem state: diversity, dominant species, extinct species, symbiosis/competition pairs."""
        if not self._isFitted:
            raise ValueError("Model not fitted.")

        nModels = len(self._fittedModels)
        diversity = _shannonDiversity(self._weights)

        dominantIdx = int(np.argmax(self._weights))
        dominantSpecies = self._modelNames[dominantIdx]

        extinctSpecies = []
        for i in range(nModels):
            if self._weights[i] < EXTINCTION_THRESHOLD:
                extinctSpecies.append(self._modelNames[i])

        symbiosisPairs = []
        competitionPairs = []
        for i in range(nModels):
            for j in range(i + 1, nModels):
                alpha = self._competitionMatrix[i, j]
                pair = (self._modelNames[i], self._modelNames[j])
                if alpha < 0:
                    symbiosisPairs.append(pair)
                elif alpha > self.competitionRate:
                    competitionPairs.append(pair)

        speciesState = {}
        for i in range(nModels):
            speciesState[self._modelNames[i]] = {
                "weight": float(self._weights[i]),
                "fitness": float(self._fitnessList[i]),
                "extinct": self._weights[i] < EXTINCTION_THRESHOLD,
            }

        return {
            "shannonDiversity": diversity,
            "dominantSpecies": dominantSpecies,
            "extinctSpecies": extinctSpecies,
            "symbiosisPairs": symbiosisPairs,
            "competitionPairs": competitionPairs,
            "speciesState": speciesState,
            "weights": self._weights.copy(),
            "competitionMatrix": self._competitionMatrix.copy(),
        }

    def _buildCompetitionMatrix(self) -> np.ndarray:
        nModels = self._residualMatrix.shape[1]

        stds = np.std(self._residualMatrix, axis=0)
        lowStdMask = stds < 1e-10

        corrMatrix = np.corrcoef(self._residualMatrix.T)
        corrMatrix = np.where(np.isnan(corrMatrix), 0.0, corrMatrix)
        absCorr = np.abs(corrMatrix)

        alpha = np.where(
            absCorr < self.symbiosisThreshold,
            -self.competitionRate * (self.symbiosisThreshold - absCorr),
            self.competitionRate * absCorr,
        )

        lowStdRows = np.where(lowStdMask)[0]
        for idx in lowStdRows:
            alpha[idx, :] = self.competitionRate
            alpha[:, idx] = self.competitionRate

        np.fill_diagonal(alpha, 1.0)

        return alpha

    def _solveLotkaVolterraODE(self) -> np.ndarray:
        nModels = len(self._fittedModels)
        K = self.carryingCapacity
        r = self.growthRate * self._fitnessList
        alpha = self._competitionMatrix

        N = np.full(nModels, K / nModels)
        N = np.maximum(N, INITIAL_WEIGHT_FLOOR)

        dt = 0.05
        for _ in range(DEFAULT_EULER_STEPS):
            competitionSums = alpha @ N
            dN = r * N * (1.0 - competitionSums / K)
            N = np.maximum(N + dt * dN, 0.0)

            totalN = np.sum(N)
            if totalN < 1e-10:
                N = np.full(nModels, K / nModels)

        totalN = np.sum(N)
        if totalN < 1e-10:
            weights = np.full(nModels, 1.0 / nModels)
        else:
            weights = N / totalN

        return weights
