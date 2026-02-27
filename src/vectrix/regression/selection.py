"""
Model Selection Tools

변수 선택 및 정규화 파라미터 튜닝:
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
    """Stepwise 선택 결과"""
    selectedFeatures: List[str]
    selectedIndices: List[int]
    selectionHistory: List[Dict]
    finalAIC: float = 0.0
    finalBIC: float = 0.0
    finalR2: float = 0.0

    def summary(self) -> str:
        """결과 요약"""
        lines = []
        lines.append("=" * 50)
        lines.append("  Stepwise Selection 결과")
        lines.append("=" * 50)
        lines.append(f"  선택된 변수 수: {len(self.selectedFeatures)}")
        lines.append(f"  선택된 변수: {self.selectedFeatures}")
        lines.append(f"  선택된 인덱스: {self.selectedIndices}")
        lines.append(f"  AIC: {self.finalAIC:.4f}")
        lines.append(f"  BIC: {self.finalBIC:.4f}")
        lines.append(f"  R^2: {self.finalR2:.4f}")
        lines.append("")
        lines.append("  선택 이력:")
        for step in self.selectionHistory:
            action = step.get('action', '')
            variable = step.get('variable', '')
            criterion = step.get('criterionValue', 0)
            lines.append(f"    {action}: {variable} (criterion={criterion:.4f})")
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class RegularizationCVResult:
    """정규화 CV 결과"""
    bestAlpha: float = 0.0
    bestScore: float = 0.0
    bestL1Ratio: Optional[float] = None
    alphaPath: np.ndarray = field(default_factory=lambda: np.array([]))
    scorePath: np.ndarray = field(default_factory=lambda: np.array([]))
    coef: np.ndarray = field(default_factory=lambda: np.array([]))
    intercept: float = 0.0

    def summary(self) -> str:
        """결과 요약"""
        lines = []
        lines.append("=" * 50)
        lines.append("  Regularization CV 결과")
        lines.append("=" * 50)
        lines.append(f"  최적 alpha: {self.bestAlpha:.6f}")
        lines.append(f"  최적 CV score (neg MSE): {self.bestScore:.6f}")
        if self.bestL1Ratio is not None:
            lines.append(f"  최적 l1_ratio: {self.bestL1Ratio:.4f}")
        if len(self.coef) > 0:
            nNonzero = np.sum(np.abs(self.coef) > 1e-10)
            lines.append(f"  비영 계수 수: {nNonzero} / {len(self.coef)}")
        lines.append("=" * 50)
        return "\n".join(lines)


@dataclass
class BestSubsetResult:
    """Best Subset Selection 결과"""
    selectedIndices: List[int] = field(default_factory=list)
    selectedFeatures: List[str] = field(default_factory=list)
    bestCriterion: float = 0.0
    allResults: List[Dict] = field(default_factory=list)


class StepwiseSelector:
    """
    AIC/BIC 기반 Stepwise 변수 선택

    direction: 'forward', 'backward', 'both'
    criterion: 'aic', 'bic'

    Parameters
    ----------
    direction : str
        선택 방향: 'forward', 'backward', 'both' (기본값: 'both')
    criterion : str
        정보 기준: 'aic' 또는 'bic' (기본값: 'aic')
    maxFeatures : int, optional
        최대 선택 변수 수. None이면 제한 없음
    """

    def __init__(
        self,
        direction: str = 'both',
        criterion: str = 'aic',
        maxFeatures: Optional[int] = None
    ):
        if direction not in ('forward', 'backward', 'both'):
            raise ValueError(f"direction은 'forward', 'backward', 'both' 중 하나: {direction}")
        if criterion not in ('aic', 'bic'):
            raise ValueError(f"criterion은 'aic' 또는 'bic' 중 하나: {criterion}")

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
        Stepwise 변수 선택 수행

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수
        featureNames : List[str], optional
            변수명 목록. None이면 X0, X1, ... 자동 생성

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
        선택된 변수로 AIC, BIC, R^2 계산

        Returns
        -------
        Tuple[aic, bic, r2]
        """
        if len(indices) == 0:
            # intercept-only model
            Xa = np.ones((n, 1))
        else:
            Xa = np.column_stack([np.ones(n), X[:, indices]])

        k = Xa.shape[1]  # intercept 포함 파라미터 수

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

        # 로그 우도 (정규 분포 가정)
        if sse < 1e-15:
            sse = 1e-15
        logLik = -n / 2.0 * (np.log(2 * np.pi) + np.log(sse / n) + 1)

        aic = -2 * logLik + 2 * k
        bic = -2 * logLik + np.log(n) * k

        return aic, bic, r2

    def _getCriterionValue(self, aic: float, bic: float) -> float:
        """선택된 정보 기준 값 반환"""
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

        # 초기 모델 (intercept only)
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

        # 최종 결과
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

        for step in range(2 * p):  # 최대 반복 제한
            improved = False

            # Forward step: 추가할 변수 탐색
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

            # Backward step: 제거할 변수 탐색
            bestRemoveCandidate = None
            bestRemoveVal = bestVal

            for j in selected:
                candidateSet = [s for s in selected if s != j]
                aic, bic, r2 = self._computeCriterion(X, y, candidateSet, n)
                val = self._getCriterionValue(aic, bic)

                if val < bestRemoveVal:
                    bestRemoveVal = val
                    bestRemoveCandidate = j

            # 더 나은 방향 선택
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
    K-fold CV로 최적 alpha 선택

    지원 모델: Ridge, Lasso, ElasticNet

    Parameters
    ----------
    model : str
        모델 종류: 'ridge', 'lasso', 'elasticnet' (기본값: 'ridge')
    alphas : np.ndarray, optional
        탐색할 alpha 값들. None이면 자동 생성
    nFolds : int
        교차검증 폴드 수 (기본값: 5)
    l1Ratios : list, optional
        ElasticNet 전용: 탐색할 l1_ratio 값들 (기본값: [0.1, 0.5, 0.7, 0.9, 0.95, 0.99])
    randomState : int, optional
        난수 시드
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
            raise ValueError(f"model은 'ridge', 'lasso', 'elasticnet' 중 하나: {model}")

        self.model = model
        self.alphas = alphas
        self.nFolds = nFolds
        self.l1Ratios = l1Ratios or [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
        self.randomState = randomState

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegularizationCVResult:
        """
        K-fold CV로 최적 정규화 파라미터 선택

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수

        Returns
        -------
        RegularizationCVResult
        """
        n, p = X.shape

        # alpha 격자 자동 생성
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
        """Ridge 또는 Lasso에 대한 CV"""
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
                foldScores.append(-mse)  # 부호 반전 (높을수록 좋음)

            if foldScores:
                scores[aIdx] = np.mean(foldScores)
            else:
                scores[aIdx] = -np.inf

        bestIdx = np.argmax(scores)
        bestAlpha = alphas[bestIdx]
        bestScore = scores[bestIdx]

        # 전체 데이터로 최종 학습
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
        """ElasticNet CV (alpha + l1Ratio 동시 탐색)"""
        n, p = X.shape
        foldIndices = self._createFolds(n, nFolds)

        bestOverallScore = -np.inf
        bestAlpha = alphas[0]
        bestL1Ratio = self.l1Ratios[0]
        allScores = np.zeros(len(alphas))  # best l1_ratio 기준 저장

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

        # 전체 데이터로 최종 학습
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
    # 내부 모델 학습
    # ----------------------------------------------------------------

    def _fitRidge(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float
    ) -> Tuple[np.ndarray, float]:
        """Ridge 학습, (coef, intercept) 반환"""
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
        """Lasso 학습 (coordinate descent), (coef, intercept) 반환"""
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
        """ElasticNet 학습 (coordinate descent), (coef, intercept) 반환"""
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
        """Soft thresholding 연산"""
        if x > lam:
            return x - lam
        elif x < -lam:
            return x + lam
        return 0.0

    # ----------------------------------------------------------------
    # K-fold 유틸리티
    # ----------------------------------------------------------------

    def _createFolds(self, n: int, nFolds: int) -> np.ndarray:
        """각 관측값의 폴드 할당 생성"""
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
        """특정 폴드에 대한 train/val 인덱스 분할"""
        valMask = foldIndices == fold
        trainIdx = np.where(~valMask)[0]
        valIdx = np.where(valMask)[0]
        return trainIdx, valIdx


class BestSubsetSelector:
    """
    Best Subset Selection (완전 탐색)

    모든 변수 조합을 탐색하여 최적 조합 선택.
    p가 작을 때만 실용적 (p <= 20 권장).

    Parameters
    ----------
    criterion : str
        정보 기준: 'aic', 'bic', 'adjr2' (기본값: 'bic')
    maxSize : int, optional
        최대 부분집합 크기. None이면 p
    """

    def __init__(self, criterion: str = 'bic', maxSize: Optional[int] = None):
        if criterion not in ('aic', 'bic', 'adjr2'):
            raise ValueError(f"criterion은 'aic', 'bic', 'adjr2' 중 하나: {criterion}")
        self.criterion = criterion
        self.maxSize = maxSize

    def select(
        self,
        X: np.ndarray,
        y: np.ndarray,
        featureNames: Optional[List[str]] = None
    ) -> BestSubsetResult:
        """
        완전 탐색으로 최적 변수 부분집합 선택

        Parameters
        ----------
        X : np.ndarray, shape (n, p)
            설계 행렬
        y : np.ndarray, shape (n,)
            반응변수
        featureNames : List[str], optional
            변수명 목록

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
                f"변수가 {p}개로 너무 많습니다 (p <= 20 권장). "
                "StepwiseSelector 또는 RegularizationCV를 사용하세요."
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
