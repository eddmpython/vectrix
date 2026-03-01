"""
Model Selection Tools

Variable selection and regularization parameter tuning:
- Stepwise Selection (Forward/Backward/Both)
- RidgeCV / LassoCV / ElasticNetCV
- Best Subset (exhaustive, small p only)

Pure numpy/scipy implementation (no sklearn dependency).
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StepwiseResult:
    """Stepwise selection result"""
    selectedFeatures: List[str]
    selectedIndices: List[int]
    selectionHistory: List[Dict]
    finalAIC: float = 0.0
    finalBIC: float = 0.0
    finalR2: float = 0.0

    def summary(self) -> str:
        """Result summary"""
        lines = []
        lines.append("=" * 50)
        lines.append("  Stepwise Selection Results")
        lines.append("=" * 50)
        lines.append(f"  Selected variables: {len(self.selectedFeatures)}")
        lines.append(f"  Selected features: {self.selectedFeatures}")
        lines.append(f"  Selected indices: {self.selectedIndices}")
        lines.append(f"  AIC: {self.finalAIC:.4f}")
        lines.append(f"  BIC: {self.finalBIC:.4f}")
        lines.append(f"  R^2: {self.finalR2:.4f}")
        lines.append("")
        lines.append("  Selection history:")
        for step in self.selectionHistory:
            action = step.get('action', '')
            variable = step.get('variable', '')
            criterion = step.get('criterionValue', 0)
            lines.append(f"    {action}: {variable} (criterion={criterion:.4f})")
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class RegularizationCVResult:
    """Regularization CV result"""
    bestAlpha: float = 0.0
    bestScore: float = 0.0
    bestL1Ratio: Optional[float] = None
    alphaPath: np.ndarray = field(default_factory=lambda: np.array([]))
    scorePath: np.ndarray = field(default_factory=lambda: np.array([]))
    coef: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept: float = 0.0

    def summary(self) -> str:
        """Result summary"""
        lines = []
        lines.append("=" * 50)
        lines.append("  Regularization CV Results")
        lines.append("=" * 50)
        lines.append(f"  Best alpha: {self.bestAlpha:.6f}")
        lines.append(f"  Best CV score (neg MSE): {self.bestScore:.6f}")
        if self.bestL1Ratio is not None:
            lines.append(f"  Best l1_ratio: {self.bestL1Ratio:.4f}")
        if len(self.coef) > 0:
            nNonzero = np.sum(np.abs(self.coef) > 1e-10)
            lines.append(f"  Non-zero coefficients: {nNonzero} / {len(self.coef)}")
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class BestSubsetResult:
    """Best Subset Selection result"""
    selectedIndices: List[int] = field(default_factory=list)
    selectedFeatures: List[str] = field(default_factory=list)
    bestCriterion: float = 0.0
    allResults: List[Dict] = field(default_factory=list)


class StepwiseSelector:
    """
    AIC/BIC-based Stepwise Variable Selection

    direction: 'forward', 'backward', 'both'
    criterion: 'aic', 'bic'

    Parameters
    ----------
    direction : str
        Selection direction: 'forward', 'backward', 'both' (default: 'both')
    criterion : str
        Information criterion: 'aic' or 'bic' (default: 'aic')
    maxFeatures : int, optional
        Maximum number of selected variables. None for no limit
    """

    def __init__(
        self,
        direction: str = 'both',
        criterion: str = 'aic',
        maxFeatures: Optional[int] = None
    ):
        if direction not in ('forward', 'backward', 'both'):
            raise ValueError(f"direction must be one of 'forward', 'backward', 'both': {direction}")
        if criterion not in ('aic', 'bic'):
            raise ValueError(f"criterion must be 'aic' or 'bic': {criterion}")

        self.direction = direction
        self.criterion = criterion
        self.maxFeatures = maxFeatures

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        featureNames: Optional[List[str]] = None
    ) -> StepwiseResult:
        """
        Perform stepwise variable selection

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable
        featureNames : List[str], optional
            Variable names. None auto-generates X0, X1, ...

        Returns
        -------
        StepwiseResult
        """
        n, p = X.shape

        if featureNames is None:
            featureNames = [f"X{i}" for i in range(p)]

        if self.direction == 'forward':
            return self._forward(X, y, featureNames, n, p)
        elif self.direction == 'backward':
            return self._backward(X, y, featureNames, n, p)
        else:
            return self._stepwiseBoth(X, y, featureNames, n, p)

    def _computeCriterion(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: List[int],
        n: int
    ) -> Tuple[float, float, float]:
        """
        Compute AIC, BIC, R^2 for selected variables

        Returns
        -------
        Tuple[aic, bic, r2]
        """
        if len(indices) == 0:
            # intercept-only model
            Xa = np.ones((n, 1))
        else:
            Xa = np.column_stack([np.ones(n), X[:, indices]])

        k = Xa.shape[1]  # Parameter count including intercept

        try:
            beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return np.inf, np.inf, 0.0

        residuals = y - Xa @ beta
        sse = np.sum(residuals ** 2)
        ssTot = np.sum((y - np.mean(y)) ** 2)

        if ssTot < 1e-15:
            return np.inf, np.inf, 0.0

        r2 = 1.0 - sse / ssTot

        # Log-likelihood (normal distribution assumption)
        if sse < 1e-15:
            sse = 1e-15
        logLik = -n / 2.0 * (np.log(2 * np.pi) + np.log(sse / n) + 1)

        aic = -2 * logLik + 2 * k
        bic = -2 * logLik + np.log(n) * k

        return aic, bic, r2

    def _getCriterionValue(self, aic: float, bic: float) -> float:
        """Return the selected criterion value"""
        return aic if self.criterion == 'aic' else bic

    def _forward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        featureNames: List[str],
        n: int,
        p: int
    ) -> StepwiseResult:
        """Forward selection"""
        selected = []
        remaining = list(range(p))
        history = []

        # Initial model (intercept only)
        aic0, bic0, r2_0 = self._computeCriterion(X, y, selected, n)
        bestVal = self._getCriterionValue(aic0, bic0)

        maxFeatures = self.maxFeatures or p

        while remaining and len(selected) < maxFeatures:
            bestCandidate = None
            bestCandidateVal = bestVal

            for j in remaining:
                candidateSet = selected + [j]
                aic, bic, r2 = self._computeCriterion(X, y, candidateSet, n)
                val = self._getCriterionValue(aic, bic)

                if val < bestCandidateVal:
                    bestCandidateVal = val
                    bestCandidate = j

            if bestCandidate is not None:
                selected.append(bestCandidate)
                remaining.remove(bestCandidate)
                bestVal = bestCandidateVal

                history.append({
                    'action': 'add',
                    'variable': featureNames[bestCandidate],
                    'variableIndex': bestCandidate,
                    'criterionValue': bestVal,
                })
            else:
                break

        # Final results
        aic, bic, r2 = self._computeCriterion(X, y, selected, n)

        return StepwiseResult(
            selectedFeatures=[featureNames[i] for i in selected],
            selectedIndices=selected,
            selectionHistory=history,
            finalAIC=aic,
            finalBIC=bic,
            finalR2=r2,
        )

    def _backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        featureNames: List[str],
        n: int,
        p: int
    ) -> StepwiseResult:
        """Backward elimination"""
        selected = list(range(p))
        history = []

        aic0, bic0, r2_0 = self._computeCriterion(X, y, selected, n)
        bestVal = self._getCriterionValue(aic0, bic0)

        while len(selected) > 0:
            bestCandidate = None
            bestCandidateVal = bestVal

            for j in selected:
                candidateSet = [s for s in selected if s != j]
                aic, bic, r2 = self._computeCriterion(X, y, candidateSet, n)
                val = self._getCriterionValue(aic, bic)

                if val < bestCandidateVal:
                    bestCandidateVal = val
                    bestCandidate = j

            if bestCandidate is not None:
                selected.remove(bestCandidate)
                bestVal = bestCandidateVal

                history.append({
                    'action': 'remove',
                    'variable': featureNames[bestCandidate],
                    'variableIndex': bestCandidate,
                    'criterionValue': bestVal,
                })
            else:
                break

        aic, bic, r2 = self._computeCriterion(X, y, selected, n)

        return StepwiseResult(
            selectedFeatures=[featureNames[i] for i in selected],
            selectedIndices=selected,
            selectionHistory=history,
            finalAIC=aic,
            finalBIC=bic,
            finalR2=r2,
        )

    def _stepwiseBoth(
        self,
        X: np.ndarray,
        y: np.ndarray,
        featureNames: List[str],
        n: int,
        p: int
    ) -> StepwiseResult:
        """Stepwise both (forward + backward at each step)"""
        selected = []
        history = []

        aic0, bic0, r2_0 = self._computeCriterion(X, y, selected, n)
        bestVal = self._getCriterionValue(aic0, bic0)

        maxFeatures = self.maxFeatures or p

        for step in range(2 * p):  # Maximum iteration limit
            improved = False

            # Forward step: search for variable to add
            remaining = [j for j in range(p) if j not in selected]
            bestAddCandidate = None
            bestAddVal = bestVal

            if len(selected) < maxFeatures:
                for j in remaining:
                    candidateSet = selected + [j]
                    aic, bic, r2 = self._computeCriterion(X, y, candidateSet, n)
                    val = self._getCriterionValue(aic, bic)

                    if val < bestAddVal:
                        bestAddVal = val
                        bestAddCandidate = j

            # Backward step: search for variable to remove
            bestRemoveCandidate = None
            bestRemoveVal = bestVal

            for j in selected:
                candidateSet = [s for s in selected if s != j]
                aic, bic, r2 = self._computeCriterion(X, y, candidateSet, n)
                val = self._getCriterionValue(aic, bic)

                if val < bestRemoveVal:
                    bestRemoveVal = val
                    bestRemoveCandidate = j

            # Choose better direction
            if bestAddCandidate is not None and bestRemoveCandidate is not None:
                if bestAddVal <= bestRemoveVal:
                    selected.append(bestAddCandidate)
                    bestVal = bestAddVal
                    history.append({
                        'action': 'add',
                        'variable': featureNames[bestAddCandidate],
                        'variableIndex': bestAddCandidate,
                        'criterionValue': bestVal,
                    })
                    improved = True
                else:
                    selected.remove(bestRemoveCandidate)
                    bestVal = bestRemoveVal
                    history.append({
                        'action': 'remove',
                        'variable': featureNames[bestRemoveCandidate],
                        'variableIndex': bestRemoveCandidate,
                        'criterionValue': bestVal,
                    })
                    improved = True
            elif bestAddCandidate is not None:
                selected.append(bestAddCandidate)
                bestVal = bestAddVal
                history.append({
                    'action': 'add',
                    'variable': featureNames[bestAddCandidate],
                    'variableIndex': bestAddCandidate,
                    'criterionValue': bestVal,
                })
                improved = True
            elif bestRemoveCandidate is not None:
                selected.remove(bestRemoveCandidate)
                bestVal = bestRemoveVal
                history.append({
                    'action': 'remove',
                    'variable': featureNames[bestRemoveCandidate],
                    'variableIndex': bestRemoveCandidate,
                    'criterionValue': bestVal,
                })
                improved = True

            if not improved:
                break

        aic, bic, r2 = self._computeCriterion(X, y, selected, n)

        return StepwiseResult(
            selectedFeatures=[featureNames[i] for i in selected],
            selectedIndices=selected,
            selectionHistory=history,
            finalAIC=aic,
            finalBIC=bic,
            finalR2=r2,
        )


class RegularizationCV:
    """
    K-fold CV for optimal alpha selection

    Supported models: Ridge, Lasso, ElasticNet

    Parameters
    ----------
    model : str
        Model type: 'ridge', 'lasso', 'elasticnet' (default: 'ridge')
    alphas : np.ndarray, optional
        Alpha values to search. None for auto-generation
    nFolds : int
        Number of cross-validation folds (default: 5)
    l1Ratios : list, optional
        ElasticNet only: l1_ratio values to search (default: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
    randomState : int, optional
        Random seed
    """

    def __init__(
        self,
        model: str = 'ridge',
        alphas: Optional[np.ndarray] = None,
        nFolds: int = 5,
        l1Ratios: Optional[List[float]] = None,
        randomState: Optional[int] = None
    ):
        if model not in ('ridge', 'lasso', 'elasticnet'):
            raise ValueError(f"model must be one of 'ridge', 'lasso', 'elasticnet': {model}")

        self.model = model
        self.alphas = alphas
        self.nFolds = nFolds
        self.l1Ratios = l1Ratios or [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        self.randomState = randomState

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegularizationCVResult:
        """
        Select optimal regularization parameter via K-fold CV

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable

        Returns
        -------
        RegularizationCVResult
        """
        n, p = X.shape

        # Auto-generate alpha grid
        alphas = self.alphas
        if alphas is None:
            alphas = np.logspace(-4, 4, 50)

        nFolds = min(self.nFolds, n)

        if self.model == 'elasticnet':
            return self._fitElasticNetCV(X, y, alphas, nFolds)
        elif self.model == 'lasso':
            return self._fitSingleModelCV(X, y, alphas, nFolds, modelType='lasso')
        else:
            return self._fitSingleModelCV(X, y, alphas, nFolds, modelType='ridge')

    def _fitSingleModelCV(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alphas: np.ndarray,
        nFolds: int,
        modelType: str
    ) -> RegularizationCVResult:
        """CV for Ridge or Lasso"""
        n, p = X.shape
        foldIndices = self._createFolds(n, nFolds)

        scores = np.zeros(len(alphas))

        for aIdx, alpha in enumerate(alphas):
            foldScores = []

            for fold in range(nFolds):
                trainIdx, valIdx = self._splitFold(foldIndices, fold)
                Xtrain, ytrain = X[trainIdx], y[trainIdx]
                Xval, yval = X[valIdx], y[valIdx]

                if len(ytrain) == 0 or len(yval) == 0:
                    continue

                if modelType == 'ridge':
                    coef, intercept = self._fitRidge(Xtrain, ytrain, alpha)
                else:
                    coef, intercept = self._fitLasso(Xtrain, ytrain, alpha)

                pred = Xval @ coef + intercept
                mse = np.mean((yval - pred) ** 2)
                foldScores.append(-mse)  # Sign flip (higher is better)

            if foldScores:
                scores[aIdx] = np.mean(foldScores)
            else:
                scores[aIdx] = -np.inf

        bestIdx = np.argmax(scores)
        bestAlpha = alphas[bestIdx]
        bestScore = scores[bestIdx]

        # Final training on full data
        if modelType == 'ridge':
            coef, intercept = self._fitRidge(X, y, bestAlpha)
        else:
            coef, intercept = self._fitLasso(X, y, bestAlpha)

        return RegularizationCVResult(
            bestAlpha=float(bestAlpha),
            bestScore=float(bestScore),
            alphaPath=alphas,
            scorePath=scores,
            coef=coef,
            intercept=intercept,
        )

    def _fitElasticNetCV(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alphas: np.ndarray,
        nFolds: int
    ) -> RegularizationCVResult:
        """ElasticNet CV (simultaneous alpha + l1Ratio search)"""
        n, p = X.shape
        foldIndices = self._createFolds(n, nFolds)

        bestOverallScore = -np.inf
        bestAlpha = alphas[0]
        bestL1Ratio = self.l1Ratios[0]
        allScores = np.zeros(len(alphas))  # Stored at best l1_ratio

        for l1Ratio in self.l1Ratios:
            scores = np.zeros(len(alphas))

            for aIdx, alpha in enumerate(alphas):
                foldScores = []

                for fold in range(nFolds):
                    trainIdx, valIdx = self._splitFold(foldIndices, fold)
                    Xtrain, ytrain = X[trainIdx], y[trainIdx]
                    Xval, yval = X[valIdx], y[valIdx]

                    if len(ytrain) == 0 or len(yval) == 0:
                        continue

                    coef, intercept = self._fitElasticNet(Xtrain, ytrain, alpha, l1Ratio)
                    pred = Xval @ coef + intercept
                    mse = np.mean((yval - pred) ** 2)
                    foldScores.append(-mse)

                if foldScores:
                    scores[aIdx] = np.mean(foldScores)
                else:
                    scores[aIdx] = -np.inf

            localBestIdx = np.argmax(scores)
            if scores[localBestIdx] > bestOverallScore:
                bestOverallScore = scores[localBestIdx]
                bestAlpha = alphas[localBestIdx]
                bestL1Ratio = l1Ratio
                allScores = scores.copy()

        # Final training on full data
        coef, intercept = self._fitElasticNet(X, y, bestAlpha, bestL1Ratio)

        return RegularizationCVResult(
            bestAlpha=float(bestAlpha),
            bestScore=float(bestOverallScore),
            bestL1Ratio=float(bestL1Ratio),
            alphaPath=alphas,
            scorePath=allScores,
            coef=coef,
            intercept=intercept,
        )

    # ----------------------------------------------------------------
    # Internal Model Fitting
    # ----------------------------------------------------------------

    def _fitRidge(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, float]:
        """Ridge fitting, returns (coef, intercept)"""
        n, p = X.shape
        xMean = np.mean(X, axis=0)
        yMean = np.mean(y)
        Xc = X - xMean
        yc = y - yMean

        I = np.eye(p)
        try:
            coef = np.linalg.solve(Xc.T @ Xc + alpha * I, Xc.T @ yc)
        except np.linalg.LinAlgError:
            coef = np.zeros(p)

        intercept = yMean - xMean @ coef
        return coef, intercept

    def _fitLasso(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        maxIter: int = 1000,
        tol: float = 1e-4
    ) -> Tuple[np.ndarray, float]:
        """Lasso fitting (coordinate descent), returns (coef, intercept)"""
        n, p = X.shape
        xMean = np.mean(X, axis=0)
        yMean = np.mean(y)
        Xc = X - xMean
        yc = y - yMean

        colNorms = np.sum(Xc ** 2, axis=0)
        beta = np.zeros(p)

        for iteration in range(maxIter):
            betaOld = beta.copy()

            for j in range(p):
                if colNorms[j] < 1e-10:
                    beta[j] = 0.0
                    continue

                residual = yc - Xc @ beta + Xc[:, j] * beta[j]
                rho = Xc[:, j] @ residual

                beta[j] = self._softThreshold(rho, alpha * n) / colNorms[j]

            if np.max(np.abs(beta - betaOld)) < tol:
                break

        intercept = yMean - xMean @ beta
        return beta, intercept

    def _fitElasticNet(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        l1Ratio: float,
        maxIter: int = 1000,
        tol: float = 1e-4
    ) -> Tuple[np.ndarray, float]:
        """ElasticNet fitting (coordinate descent), returns (coef, intercept)"""
        n, p = X.shape
        l1Penalty = alpha * l1Ratio
        l2Penalty = alpha * (1 - l1Ratio)

        xMean = np.mean(X, axis=0)
        yMean = np.mean(y)
        Xc = X - xMean
        yc = y - yMean

        colNorms = np.sum(Xc ** 2, axis=0)
        beta = np.zeros(p)

        for iteration in range(maxIter):
            betaOld = beta.copy()

            for j in range(p):
                denom = colNorms[j] + l2Penalty * n
                if denom < 1e-10:
                    beta[j] = 0.0
                    continue

                residual = yc - Xc @ beta + Xc[:, j] * beta[j]
                rho = Xc[:, j] @ residual

                beta[j] = self._softThreshold(rho, l1Penalty * n) / denom

            if np.max(np.abs(beta - betaOld)) < tol:
                break

        intercept = yMean - xMean @ beta
        return beta, intercept

    @staticmethod
    def _softThreshold(x: float, lam: float) -> float:
        """Soft thresholding operation"""
        if x > lam:
            return x - lam
        elif x < -lam:
            return x + lam
        return 0.0

    # ----------------------------------------------------------------
    # K-fold Utilities
    # ----------------------------------------------------------------

    def _createFolds(self, n: int, nFolds: int) -> np.ndarray:
        """Generate fold assignments for each observation"""
        rng = np.random.RandomState(self.randomState)
        indices = np.arange(n)
        rng.shuffle(indices)
        folds = np.zeros(n, dtype=int)
        foldSizes = np.full(nFolds, n // nFolds)
        foldSizes[:n % nFolds] += 1
        current = 0
        for fold, size in enumerate(foldSizes):
            folds[indices[current:current + size]] = fold
            current += size
        return folds

    def _splitFold(
        self,
        foldIndices: np.ndarray,
        fold: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Split train/val indices for a specific fold"""
        valMask = foldIndices == fold
        trainIdx = np.where(~valMask)[0]
        valIdx = np.where(valMask)[0]
        return trainIdx, valIdx


class BestSubsetSelector:
    """
    Best Subset Selection (exhaustive search)

    Search all variable combinations to find optimal subset.
    Practical only for small p (p <= 20 recommended).

    Parameters
    ----------
    criterion : str
        Information criterion: 'aic', 'bic', 'adjr2' (default: 'bic')
    maxSize : int, optional
        Maximum subset size. None defaults to p
    """

    def __init__(self, criterion: str = 'bic', maxSize: Optional[int] = None):
        if criterion not in ('aic', 'bic', 'adjr2'):
            raise ValueError(f"criterion must be one of 'aic', 'bic', 'adjr2': {criterion}")
        self.criterion = criterion
        self.maxSize = maxSize

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        featureNames: Optional[List[str]] = None
    ) -> BestSubsetResult:
        """
        Select optimal variable subset via exhaustive search

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            Design matrix
        y : np.ndarray, shape (n,)
            Response variable
        featureNames : List[str], optional
            Variable name list

        Returns
        -------
        BestSubsetResult
        """
        n, p = X.shape

        if featureNames is None:
            featureNames = [f"X{i}" for i in range(p)]

        maxSize = self.maxSize or p
        maxSize = min(maxSize, p)

        if p > 20:
            raise ValueError(
                f"Too many variables ({p}) (p <= 20 recommended). "
                "Use StepwiseSelector or RegularizationCV instead."
            )

        bestCriterion = np.inf if self.criterion != 'adjr2' else -np.inf
        bestIndices = []
        allResults = []

        yMean = np.mean(y)
        ssTot = np.sum((y - yMean) ** 2)

        for size in range(1, maxSize + 1):
            for combo in combinations(range(p), size):
                indices = list(combo)
                Xa = np.column_stack([np.ones(n), X[:, indices]])
                k = Xa.shape[1]

                try:
                    beta = np.linalg.lstsq(Xa, y, rcond=None)[0]
                except np.linalg.LinAlgError:
                    continue

                residuals = y - Xa @ beta
                sse = np.sum(residuals ** 2)

                if ssTot < 1e-15:
                    continue

                r2 = 1.0 - sse / ssTot
                adjR2 = 1.0 - (1.0 - r2) * (n - 1) / max(n - k, 1)

                if sse < 1e-15:
                    sse = 1e-15
                logLik = -n / 2.0 * (np.log(2 * np.pi) + np.log(sse / n) + 1)
                aic = -2 * logLik + 2 * k
                bic = -2 * logLik + np.log(n) * k

                resultEntry = {
                    'indices': indices,
                    'features': [featureNames[i] for i in indices],
                    'aic': aic,
                    'bic': bic,
                    'adjR2': adjR2,
                    'r2': r2,
                }
                allResults.append(resultEntry)

                if self.criterion == 'aic':
                    val = aic
                    isBetter = val < bestCriterion
                elif self.criterion == 'bic':
                    val = bic
                    isBetter = val < bestCriterion
                else:  # adjr2
                    val = adjR2
                    isBetter = val > bestCriterion

                if isBetter:
                    bestCriterion = val
                    bestIndices = indices

        return BestSubsetResult(
            selectedIndices=bestIndices,
            selectedFeatures=[featureNames[i] for i in bestIndices],
            bestCriterion=float(bestCriterion),
            allResults=allResults,
        )
