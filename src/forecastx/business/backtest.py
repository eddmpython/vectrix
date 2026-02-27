"""
Backtester

Walk-forward validation with business metrics.
Expanding or sliding window strategy.
"""

import numpy as np
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field

from ..engine.turbo import TurboCore


@dataclass
class BacktestFold:
    """백테스트 폴드 결과"""
    fold: int = 0
    trainSize: int = 0
    testSize: int = 0
    mape: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    smape: float = 0.0
    bias: float = 0.0
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    actuals: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BacktestResult:
    """백테스트 종합 결과"""
    nFolds: int = 0
    avgMAPE: float = 0.0
    avgRMSE: float = 0.0
    avgMAE: float = 0.0
    avgSMAPE: float = 0.0
    avgBias: float = 0.0
    mapeStd: float = 0.0
    folds: List[BacktestFold] = field(default_factory=list)
    bestFold: int = 0
    worstFold: int = 0


class Backtester:
    """
    시계열 백테스터

    Usage:
        >>> bt = Backtester(nFolds=5, horizon=30)
        >>> result = bt.run(y, model_factory)
    """

    def __init__(
        self,
        nFolds: int = 5,
        horizon: int = 30,
        strategy: str = 'expanding',
        minTrainSize: int = 50,
        stepSize: Optional[int] = None
    ):
        self.nFolds = nFolds
        self.horizon = horizon
        self.strategy = strategy
        self.minTrainSize = minTrainSize
        self.stepSize = stepSize

    def run(
        self,
        y: np.ndarray,
        modelFactory: Callable
    ) -> BacktestResult:
        """
        백테스트 실행

        Parameters
        ----------
        y : np.ndarray
            전체 시계열
        modelFactory : Callable
            모델 생성 함수 (fit/predict 메서드 필수)
        """
        n = len(y)
        needed = self.minTrainSize + self.horizon

        if n < needed:
            return BacktestResult()

        available = n - self.minTrainSize - self.horizon
        step = self.stepSize or max(1, available // max(self.nFolds - 1, 1))

        folds = []
        for i in range(self.nFolds):
            if self.strategy == 'sliding':
                trainEnd = self.minTrainSize + i * step
                trainStart = max(0, trainEnd - self.minTrainSize)
            else:
                trainEnd = self.minTrainSize + i * step
                trainStart = 0

            testEnd = min(trainEnd + self.horizon, n)

            if trainEnd >= n or testEnd <= trainEnd:
                break

            trainData = y[trainStart:trainEnd]
            testData = y[trainEnd:testEnd]
            testSteps = len(testData)

            try:
                model = modelFactory()
                model.fit(trainData)
                pred, _, _ = model.predict(testSteps)
                pred = pred[:len(testData)]

                mape = TurboCore.mape(testData, pred)
                rmse = TurboCore.rmse(testData, pred)
                mae = TurboCore.mae(testData, pred)
                smape = TurboCore.smape(testData, pred)
                bias = float(np.mean(pred - testData))

                folds.append(BacktestFold(
                    fold=i, trainSize=len(trainData), testSize=len(testData),
                    mape=mape, rmse=rmse, mae=mae, smape=smape, bias=bias,
                    predictions=pred, actuals=testData
                ))
            except Exception:
                folds.append(BacktestFold(
                    fold=i, trainSize=len(trainData), testSize=len(testData),
                    mape=np.inf, rmse=np.inf, mae=np.inf, smape=np.inf
                ))

        if not folds:
            return BacktestResult()

        validFolds = [f for f in folds if f.mape < np.inf]
        if not validFolds:
            return BacktestResult(nFolds=len(folds), folds=folds)

        mapes = [f.mape for f in validFolds]

        return BacktestResult(
            nFolds=len(folds),
            avgMAPE=float(np.mean(mapes)),
            avgRMSE=float(np.mean([f.rmse for f in validFolds])),
            avgMAE=float(np.mean([f.mae for f in validFolds])),
            avgSMAPE=float(np.mean([f.smape for f in validFolds])),
            avgBias=float(np.mean([f.bias for f in validFolds])),
            mapeStd=float(np.std(mapes)),
            folds=folds,
            bestFold=int(np.argmin(mapes)),
            worstFold=int(np.argmax(mapes))
        )

    def summary(self, result: BacktestResult, locale: str = 'ko') -> str:
        if locale == 'ko':
            lines = [
                f"백테스트 결과 ({result.nFolds}개 폴드)",
                f"  평균 MAPE: {result.avgMAPE:.2f}% (±{result.mapeStd:.2f}%)",
                f"  평균 RMSE: {result.avgRMSE:.2f}",
                f"  평균 MAE: {result.avgMAE:.2f}",
                f"  평균 편향: {result.avgBias:.2f}",
                f"  최고 폴드: #{result.bestFold} (MAPE {result.folds[result.bestFold].mape:.2f}%)" if result.folds else "",
                f"  최악 폴드: #{result.worstFold} (MAPE {result.folds[result.worstFold].mape:.2f}%)" if result.folds else "",
            ]
        else:
            lines = [
                f"Backtest Results ({result.nFolds} folds)",
                f"  Avg MAPE: {result.avgMAPE:.2f}% (±{result.mapeStd:.2f}%)",
                f"  Avg RMSE: {result.avgRMSE:.2f}",
                f"  Avg MAE: {result.avgMAE:.2f}",
                f"  Avg Bias: {result.avgBias:.2f}",
            ]

        return '\n'.join([l for l in lines if l])
