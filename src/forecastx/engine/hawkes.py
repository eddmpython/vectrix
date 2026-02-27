"""
Hawkes Process for Intermittent Demand Forecasting

호크스 과정 기반 간헐적 수요 예측.
수요 발생이 후속 수요를 촉발하는 자기흥분(self-exciting) 효과를 모델링.
기존 Croston/SBA 대비 군집적 수요 패턴에 강점.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson, geom


SUPPORTED_DISTRIBUTIONS = ('negbin', 'poisson', 'geometric')

DEFAULT_BASE_INTENSITY = 0.1
DEFAULT_EXCITATION_DECAY = 0.5
DEFAULT_EXCITATION_MAGNITUDE = 0.3
DEFAULT_N_SIM = 200
MIN_EVENTS_FOR_FIT = 3
STABILITY_MARGIN = 0.999


class HawkesIntermittentDemand:
    """
    호크스 과정 기반 간헐적 수요 예측.
    수요 발생이 후속 수요를 촉발하는 자기흥분(self-exciting) 효과를 모델링.
    기존 Croston/SBA 대비 군집적 수요 패턴에 강점.
    """

    def __init__(
        self,
        baseIntensity: float = DEFAULT_BASE_INTENSITY,
        excitationDecay: float = DEFAULT_EXCITATION_DECAY,
        excitationMagnitude: float = DEFAULT_EXCITATION_MAGNITUDE,
        demandDistribution: str = 'negbin',
    ):
        self.baseIntensity = baseIntensity
        self.excitationDecay = excitationDecay
        self.excitationMagnitude = excitationMagnitude
        self.demandDistribution = demandDistribution

        self._eventTimes: Optional[np.ndarray] = None
        self._eventSizes: Optional[np.ndarray] = None
        self._T: float = 0.0
        self._sizeParams: Dict = {}
        self._fitted: bool = False
        self._lastIntensity: float = baseIntensity
        self._y: Optional[np.ndarray] = None

    def fit(self, y: np.ndarray) -> 'HawkesIntermittentDemand':
        """MLE로 호크스 과정 파라미터와 수요 크기 분포를 동시 추정."""
        y = np.asarray(y, dtype=np.float64)
        self._y = y.copy()
        self._T = float(len(y))

        self._eventTimes = np.where(y > 0)[0].astype(np.float64)
        self._eventSizes = y[y > 0].copy()

        if len(self._eventTimes) < MIN_EVENTS_FOR_FIT:
            self._fitted = True
            self._fitSizeDistribution()
            return self

        self._optimizeHawkesParams()
        self._fitSizeDistribution()
        self._computeLastIntensity()

        self._fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Monte Carlo 시뮬레이션 기반 예측."""
        if not self._fitted:
            raise ValueError("Model not fitted.")

        simResults = self._simulateHawkes(steps, nSim=DEFAULT_N_SIM)

        predictions = np.median(simResults, axis=0)
        lower = np.percentile(simResults, 2.5, axis=0)
        upper = np.percentile(simResults, 97.5, axis=0)

        lower = np.maximum(lower, 0.0)

        return predictions, lower, upper

    def getIntensityProfile(self) -> Dict:
        """시계열 전체의 강도 함수 프로필을 반환."""
        if not self._fitted or self._eventTimes is None:
            return {
                'burstiness': 0.0,
                'selfExcitationRatio': 0.0,
                'meanInterArrival': 0.0,
                'baseIntensity': self.baseIntensity,
                'excitationDecay': self.excitationDecay,
                'excitationMagnitude': self.excitationMagnitude,
            }

        selfExcitationRatio = (
            self.excitationMagnitude / self.excitationDecay
            if self.excitationDecay > 0 else 0.0
        )

        if len(self._eventTimes) > 1:
            interArrivals = np.diff(self._eventTimes)
            meanInterArrival = float(np.mean(interArrivals))
            stdInterArrival = float(np.std(interArrivals))
            burstiness = (
                (stdInterArrival - meanInterArrival)
                / (stdInterArrival + meanInterArrival + 1e-10)
            )
        else:
            meanInterArrival = self._T
            burstiness = 0.0

        return {
            'burstiness': float(np.clip(burstiness, -1.0, 1.0)),
            'selfExcitationRatio': float(selfExcitationRatio),
            'meanInterArrival': float(meanInterArrival),
            'baseIntensity': float(self.baseIntensity),
            'excitationDecay': float(self.excitationDecay),
            'excitationMagnitude': float(self.excitationMagnitude),
        }

    def _logLikelihood(self, params: np.ndarray, eventTimes: np.ndarray, T: float) -> float:
        mu, alpha, beta = params[0], params[1], params[2]

        if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta * STABILITY_MARGIN:
            return 1e10

        n = len(eventTimes)
        logLik = 0.0

        for i in range(n):
            intensity = mu
            for j in range(i):
                dt = eventTimes[i] - eventTimes[j]
                intensity += alpha * np.exp(-beta * dt)
            if intensity > 0:
                logLik += np.log(intensity)
            else:
                return 1e10

        integral = mu * T
        for i in range(n):
            integral += (alpha / beta) * (1.0 - np.exp(-beta * (T - eventTimes[i])))

        logLik -= integral
        return -logLik

    def _optimizeHawkesParams(self) -> None:
        eventTimes = self._eventTimes
        T = self._T
        n = len(eventTimes)

        muInit = n / T if T > 0 else DEFAULT_BASE_INTENSITY
        alphaInit = muInit * 0.3
        betaInit = alphaInit / 0.5

        x0 = np.array([muInit, alphaInit, betaInit])

        bounds = [
            (1e-6, n / T * 5 if T > 0 else 10.0),
            (1e-6, 10.0),
            (1e-4, 20.0),
        ]

        try:
            result = minimize(
                self._logLikelihood,
                x0,
                args=(eventTimes, T),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-6},
            )
            if result.success or result.fun < self._logLikelihood(x0, eventTimes, T):
                self.baseIntensity = result.x[0]
                self.excitationMagnitude = result.x[1]
                self.excitationDecay = result.x[2]
        except Exception:
            pass

    def _fitSizeDistribution(self) -> None:
        if self._eventSizes is None or len(self._eventSizes) == 0:
            self._sizeParams = {
                'distribution': self.demandDistribution,
                'mean': 1.0,
                'r': 1.0,
                'p': 0.5,
                'lam': 1.0
            }
            return

        sizes = self._eventSizes
        meanSize = float(np.mean(sizes))
        varSize = float(np.var(sizes, ddof=1)) if len(sizes) > 1 else meanSize

        if self.demandDistribution == 'negbin':
            if varSize > meanSize and meanSize > 0:
                p = meanSize / varSize
                p = np.clip(p, 0.01, 0.99)
                r = meanSize * p / (1.0 - p)
                r = max(r, 0.1)
            else:
                r = max(meanSize, 0.1)
                p = 0.5
            self._sizeParams = {'distribution': 'negbin', 'r': float(r), 'p': float(p)}

        elif self.demandDistribution == 'poisson':
            lambdaParam = max(meanSize, 0.1)
            self._sizeParams = {'distribution': 'poisson', 'lambda': float(lambdaParam)}

        elif self.demandDistribution == 'geometric':
            geoP = 1.0 / max(meanSize, 1.0)
            geoP = np.clip(geoP, 0.01, 0.99)
            self._sizeParams = {'distribution': 'geometric', 'p': float(geoP)}

        else:
            self._sizeParams = {'distribution': 'empirical', 'mean': float(meanSize)}

    def _sampleDemandSize(self, rng: np.random.Generator) -> float:
        dist = self._sizeParams.get('distribution', 'empirical')

        if dist == 'negbin':
            r = self._sizeParams['r']
            p = self._sizeParams['p']
            return float(rng.negative_binomial(max(1, int(round(r))), p)) + 1.0

        if dist == 'poisson':
            return float(rng.poisson(self._sizeParams['lambda'])) + 1.0

        if dist == 'geometric':
            return float(rng.geometric(self._sizeParams['p']))

        return max(self._sizeParams.get('mean', 1.0), 1.0)

    def _computeLastIntensity(self) -> None:
        if self._eventTimes is None or len(self._eventTimes) == 0:
            self._lastIntensity = self.baseIntensity
            return

        intensity = self.baseIntensity
        T = self._T
        for ti in self._eventTimes:
            intensity += self.excitationMagnitude * np.exp(
                -self.excitationDecay * (T - ti)
            )
        self._lastIntensity = intensity

    def _simulateHawkes(self, steps: int, nSim: int = DEFAULT_N_SIM) -> np.ndarray:
        results = np.zeros((nSim, steps))
        rng = np.random.default_rng()

        mu = self.baseIntensity
        alpha = self.excitationMagnitude
        beta = self.excitationDecay

        for sim in range(nSim):
            currentIntensity = self._lastIntensity
            eventHistory = []

            for step in range(steps):
                upperBound = max(currentIntensity, mu) + alpha * len(eventHistory)
                upperBound = max(upperBound, 1e-6)

                demandAtStep = 0.0
                t = 0.0

                while t < 1.0:
                    dt = rng.exponential(1.0 / upperBound)
                    t += dt

                    if t >= 1.0:
                        break

                    lambdaT = mu
                    for evtTime in eventHistory:
                        lambdaT += alpha * np.exp(-beta * (step + t - evtTime))
                    lambdaT += alpha * np.exp(-beta * t) * (currentIntensity - mu)

                    lambdaT = max(lambdaT, 0.0)

                    if rng.uniform() < lambdaT / upperBound:
                        size = self._sampleDemandSize(rng)
                        demandAtStep += size
                        eventHistory.append(step + t)
                        upperBound = max(upperBound, lambdaT + alpha)

                results[sim, step] = demandAtStep

                decayFactor = np.exp(-beta)
                currentIntensity = mu + (currentIntensity - mu) * decayFactor
                for evtTime in eventHistory:
                    if step + 1.0 - evtTime < 20.0 / beta:
                        currentIntensity += alpha * np.exp(-beta * (step + 1.0 - evtTime))

                maxHistory = 50
                if len(eventHistory) > maxHistory:
                    eventHistory = eventHistory[-maxHistory:]

        return results
