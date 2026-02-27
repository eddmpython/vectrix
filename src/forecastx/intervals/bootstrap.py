"""
Bootstrap Prediction Intervals

Residual bootstrap for prediction interval estimation.
"""

import numpy as np
from typing import Tuple, Callable


class BootstrapInterval:
    """
    Residual Bootstrap Prediction Intervals

    1. 모델 학습, 잔차 계산
    2. 잔차를 재표본하여 부트스트랩 미래 경로 생성
    3. 경로들의 분위수로 신뢰구간 계산
    """

    def __init__(
        self,
        nBoot: int = 100,
        coverageLevel: float = 0.95
    ):
        self.nBoot = nBoot
        self.coverageLevel = coverageLevel
        self.residuals = None

    def calibrate(
        self,
        y: np.ndarray,
        modelFactory: Callable,
        steps: int = 1
    ) -> 'BootstrapInterval':
        """
        Bootstrap calibration

        Parameters
        ----------
        y : np.ndarray
            시계열 데이터
        modelFactory : Callable
            모델 생성 팩토리
        steps : int
            예측 horizon
        """
        n = len(y)
        trainSize = int(n * 0.8)
        if trainSize < 10:
            self.residuals = np.random.randn(10) * np.std(y)
            return self

        trainData = y[:trainSize]
        testData = y[trainSize:]
        testSteps = len(testData)

        try:
            model = modelFactory()
            model.fit(trainData)
            pred, _, _ = model.predict(testSteps)
            self.residuals = testData - pred[:len(testData)]
        except Exception:
            self.residuals = np.random.randn(10) * np.std(y) * 0.1

        return self

    def predict(
        self,
        pointPredictions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate bootstrap intervals.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (lower, upper)
        """
        steps = len(pointPredictions)

        if self.residuals is None or len(self.residuals) == 0:
            sigma = np.std(pointPredictions) * 0.1 + 1.0
            return pointPredictions - 1.96 * sigma, pointPredictions + 1.96 * sigma

        bootPaths = np.zeros((self.nBoot, steps))

        for b in range(self.nBoot):
            bootResiduals = np.random.choice(self.residuals, size=steps, replace=True)
            cumResiduals = np.cumsum(bootResiduals) * np.sqrt(1.0 / np.arange(1, steps + 1))
            bootPaths[b] = pointPredictions + cumResiduals

        alpha = 1 - self.coverageLevel
        lower = np.percentile(bootPaths, alpha / 2 * 100, axis=0)
        upper = np.percentile(bootPaths, (1 - alpha / 2) * 100, axis=0)

        return lower, upper
