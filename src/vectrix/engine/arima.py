"""
ARIMA Model Implementation

ARIMA(p, d, q) x (P, D, Q)[m]
- AR: Autoregressive
- I: Integrated (Differencing)
- MA: Moving Average

Implements various estimation methods including Yule-Walker, CSS, and MLE
"""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from vectrix_core import css_objective as _cssObjectiveRust
    from vectrix_core import seasonal_css_objective as _seasonalCSSObjectiveRust
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .turbo import TurboCore


@jit(nopython=True, cache=True)
def _cssObjectiveNumba(y: np.ndarray, arCoefs: np.ndarray, maCoefs: np.ndarray) -> float:
    """Numba-optimized CSS objective function"""
    n = len(y)
    p = len(arCoefs)
    q = len(maCoefs)
    maxLag = max(p, q, 1)

    residuals = np.zeros(n)
    css = 0.0

    for t in range(maxLag, n):
        pred = 0.0

        for i in range(p):
            pred += arCoefs[i] * y[t - i - 1]

        for j in range(min(q, t)):
            pred += maCoefs[j] * residuals[t - j - 1]

        residuals[t] = y[t] - pred
        css += residuals[t] * residuals[t]

    return css


@jit(nopython=True, cache=True)
def _seasonalCSSObjectiveNumba(
    y: np.ndarray,
    arCoefs: np.ndarray,
    maCoefs: np.ndarray,
    sarCoefs: np.ndarray,
    smaCoefs: np.ndarray,
    m: int
) -> float:
    """Numba-optimized seasonal ARIMA CSS objective function"""
    n = len(y)
    p = len(arCoefs)
    q = len(maCoefs)
    bigP = len(sarCoefs)
    bigQ = len(smaCoefs)
    maxLag = max(p, q, bigP * m, bigQ * m, 1)

    residuals = np.zeros(n)
    css = 0.0

    for t in range(maxLag, n):
        pred = 0.0

        for i in range(p):
            pred += arCoefs[i] * y[t - i - 1]

        for i in range(bigP):
            idx = t - (i + 1) * m
            if idx >= 0:
                pred += sarCoefs[i] * y[idx]

        for j in range(min(q, t)):
            pred += maCoefs[j] * residuals[t - j - 1]

        for j in range(bigQ):
            idx = t - (j + 1) * m
            if idx >= 0:
                pred += smaCoefs[j] * residuals[idx]

        residuals[t] = y[t] - pred
        css += residuals[t] * residuals[t]

    return css


def _checkStationarity(arCoefs: np.ndarray) -> np.ndarray:
    """Verify and correct stationarity of AR coefficients"""
    if len(arCoefs) == 0:
        return arCoefs

    polyCoefs = np.concatenate(([1.0], -arCoefs))
    roots = np.roots(polyCoefs)
    moduli = np.abs(roots)

    if np.all(moduli > 1.0):
        return arCoefs

    shrinkFactor = 0.99
    corrected = arCoefs.copy()
    for _ in range(50):
        polyCoefs = np.concatenate(([1.0], -corrected))
        roots = np.roots(polyCoefs)
        moduli = np.abs(roots)
        if np.all(moduli > 1.0):
            return corrected
        corrected = corrected * shrinkFactor
        shrinkFactor *= 0.99

    return corrected


def _checkInvertibility(maCoefs: np.ndarray) -> np.ndarray:
    """Verify and correct invertibility of MA coefficients"""
    if len(maCoefs) == 0:
        return maCoefs

    polyCoefs = np.concatenate(([1.0], maCoefs))
    roots = np.roots(polyCoefs)
    moduli = np.abs(roots)

    if np.all(moduli > 1.0):
        return maCoefs

    shrinkFactor = 0.99
    corrected = maCoefs.copy()
    for _ in range(50):
        polyCoefs = np.concatenate(([1.0], corrected))
        roots = np.roots(polyCoefs)
        moduli = np.abs(roots)
        if np.all(moduli > 1.0):
            return corrected
        corrected = corrected * shrinkFactor
        shrinkFactor *= 0.99

    return corrected


class ARIMAModel:
    """
    ARIMA Model Implementation

    ARIMA(p, d, q) with optional seasonal component
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonalOrder: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Parameters
        ----------
        order : Tuple[int, int, int]
            (p, d, q) - AR order, differencing order, MA order
        seasonalOrder : Tuple[int, int, int, int], optional
            (P, D, Q, m) - Seasonal AR, Seasonal diff, Seasonal MA, period
        """
        self.p, self.d, self.q = order

        if seasonalOrder:
            self.P, self.D, self.Q, self.m = seasonalOrder
        else:
            self.P, self.D, self.Q, self.m = 0, 0, 0, 1

        self.arCoefs = None
        self.maCoefs = None
        self.sarCoefs = None
        self.smaCoefs = None
        self.constant = 0.0

        self.origData = None
        self.diffData = None
        self.residuals = None
        self.fitted = False
        self.sigma2 = 1.0

    def fit(self, y: np.ndarray) -> 'ARIMAModel':
        """
        Fit the model

        Parameters
        ----------
        y : np.ndarray
            Time series data
        """
        self.origData = y.copy()
        n = len(y)

        diffed = self._difference(y)
        self.diffData = diffed

        if len(diffed) < max(self.p, self.q) + 5:
            self._simpleFit(diffed)
            self.fitted = True
            return self

        if self.p > 0:
            self.arCoefs = self._estimateAR(diffed, self.p)
        else:
            self.arCoefs = np.array([])

        if self.p > 0:
            arResiduals = self._computeARResiduals(diffed, self.arCoefs)
        else:
            arResiduals = diffed.copy()

        if self.q > 0:
            self.maCoefs = self._estimateMA(arResiduals, self.q)
        else:
            self.maCoefs = np.array([])

        if self.P > 0:
            self.sarCoefs = self._estimateSAR(diffed)
        else:
            self.sarCoefs = np.array([])

        if self.Q > 0:
            sarResiduals = self._computeSeasonalARResiduals(diffed)
            self.smaCoefs = self._estimateSMA(sarResiduals)
        else:
            self.smaCoefs = np.array([])

        self._optimizeCSS(diffed)

        self.arCoefs = _checkStationarity(self.arCoefs)
        self.maCoefs = _checkInvertibility(self.maCoefs)
        if self.P > 0:
            self.sarCoefs = _checkStationarity(self.sarCoefs)
        if self.Q > 0:
            self.smaCoefs = _checkInvertibility(self.smaCoefs)

        self.residuals = self._computeResiduals(diffed)
        self.sigma2 = np.var(self.residuals) if len(self.residuals) > 0 else 1.0

        self.fitted = True
        return self

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast

        Parameters
        ----------
        steps : int
            Forecast horizon

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (predictions, lower95, upper95)
        """
        if not self.fitted:
            raise ValueError("Model has not been fitted.")

        diffPred = self._predictDiff(steps)

        predictions = self._inverseDifference(diffPred)

        sigma = np.sqrt(self.sigma2)

        psiWeights = self._computePsiWeights(steps)

        variances = np.zeros(steps)
        cumPsi2 = 0.0
        for h in range(steps):
            if h < len(psiWeights):
                cumPsi2 += psiWeights[h] ** 2
            else:
                cumPsi2 += 1.0
            variances[h] = sigma ** 2 * cumPsi2

        margin = 1.96 * np.sqrt(variances)
        lower95 = predictions - margin
        upper95 = predictions + margin

        return predictions, lower95, upper95

    def _difference(self, y: np.ndarray) -> np.ndarray:
        """Apply differencing"""
        result = y.copy()

        for _ in range(self.d):
            result = TurboCore.diff(result, 1)

        for _ in range(self.D):
            if len(result) > self.m:
                result = TurboCore.seasonalDiff(result, self.m)

        return result

    def _inverseDifference(self, pred: np.ndarray) -> np.ndarray:
        """Inverse differencing"""
        result = pred.copy()
        orig = self.origData

        for _ in range(self.D):
            newResult = np.zeros(len(result))
            for i in range(len(result)):
                if i < self.m:
                    idx = len(orig) - self.m + i
                    if idx >= 0:
                        newResult[i] = orig[idx] + result[i]
                    else:
                        newResult[i] = result[i]
                else:
                    newResult[i] = newResult[i - self.m] + result[i]
            result = newResult

        for _ in range(self.d):
            lastVal = orig[-1] if len(orig) > 0 else 0
            result = TurboCore.integrate(result, lastVal, 1)[1:]

        return result

    def _estimateAR(self, y: np.ndarray, p: int) -> np.ndarray:
        """
        AR coefficient estimation (Yule-Walker)
        """
        acf = TurboCore.acf(y, p)

        if p == 0:
            return np.array([])

        R = np.zeros((p, p))
        r = np.zeros(p)

        for i in range(p):
            r[i] = acf[i + 1]
            for j in range(p):
                R[i, j] = acf[abs(i - j)]

        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            phi = np.zeros(p)
            phi[0] = acf[1] if len(acf) > 1 else 0

        return phi

    def _estimateMA(self, residuals: np.ndarray, q: int) -> np.ndarray:
        """
        MA coefficient estimation (Innovation Algorithm - Brockwell & Davis)

        Input: autocovariance gamma(0), gamma(1), ..., gamma(q)
        Output: theta = [theta_{q,0}, theta_{q,1}, ..., theta_{q,q-1}]
        """
        n = len(residuals)

        if n < q + 5:
            return np.zeros(q)

        acfVals = TurboCore.acf(residuals, q)
        gamma0 = np.var(residuals)

        if gamma0 < 1e-10:
            return np.zeros(q)

        gamma = np.zeros(q + 1)
        gamma[0] = gamma0
        for k in range(1, q + 1):
            gamma[k] = acfVals[k] * gamma0

        theta = np.zeros((q + 1, q + 1))
        v = np.zeros(q + 1)
        v[0] = gamma[0]

        for m in range(1, q + 1):
            for k in range(m):
                innerSum = 0.0
                for j in range(k):
                    innerSum += theta[k, k - 1 - j] * theta[m, m - 1 - j] * v[j]
                if abs(v[k]) > 1e-10:
                    theta[m, m - 1 - k] = (gamma[m - k] - innerSum) / v[k]

            vSum = 0.0
            for j in range(m):
                vSum += theta[m, m - 1 - j] ** 2 * v[j]
            v[m] = gamma[0] - vSum

            if v[m] < 1e-10:
                v[m] = 1e-10

        result = np.zeros(q)
        for i in range(q):
            result[i] = theta[q, i]

        return result

    def _estimateSAR(self, y: np.ndarray) -> np.ndarray:
        """
        Seasonal AR coefficient estimation (Yule-Walker on seasonal lags)
        """
        bigP = self.P
        m = self.m
        maxSeasonalLag = bigP * m

        if len(y) <= maxSeasonalLag + 5:
            return np.zeros(bigP)

        acf = TurboCore.acf(y, maxSeasonalLag)

        R = np.zeros((bigP, bigP))
        r = np.zeros(bigP)

        for i in range(bigP):
            r[i] = acf[(i + 1) * m]
            for j in range(bigP):
                lagDiff = abs(i - j) * m
                R[i, j] = acf[lagDiff]

        try:
            phi = np.linalg.solve(R, r)
        except np.linalg.LinAlgError:
            phi = np.zeros(bigP)
            if bigP > 0 and len(acf) > m:
                phi[0] = acf[m] * 0.5

        return phi

    def _estimateSMA(self, residuals: np.ndarray) -> np.ndarray:
        """
        Seasonal MA coefficient estimation (Innovation Algorithm on seasonal residuals)
        """
        bigQ = self.Q
        m = self.m
        maxSeasonalLag = bigQ * m

        if len(residuals) <= maxSeasonalLag + 5:
            return np.zeros(bigQ)

        acfVals = TurboCore.acf(residuals, maxSeasonalLag)
        gamma0 = np.var(residuals)

        if gamma0 < 1e-10:
            return np.zeros(bigQ)

        gamma = np.zeros(bigQ + 1)
        gamma[0] = gamma0
        for k in range(1, bigQ + 1):
            lagIdx = k * m
            if lagIdx < len(acfVals):
                gamma[k] = acfVals[lagIdx] * gamma0

        theta = np.zeros((bigQ + 1, bigQ + 1))
        v = np.zeros(bigQ + 1)
        v[0] = gamma[0]

        for mm in range(1, bigQ + 1):
            for k in range(mm):
                innerSum = 0.0
                for j in range(k):
                    innerSum += theta[k, k - 1 - j] * theta[mm, mm - 1 - j] * v[j]
                if abs(v[k]) > 1e-10:
                    theta[mm, mm - 1 - k] = (gamma[mm - k] - innerSum) / v[k]

            vSum = 0.0
            for j in range(mm):
                vSum += theta[mm, mm - 1 - j] ** 2 * v[j]
            v[mm] = gamma[0] - vSum

            if v[mm] < 1e-10:
                v[mm] = 1e-10

        result = np.zeros(bigQ)
        for i in range(bigQ):
            result[i] = theta[bigQ, i]

        return result

    def _computeSeasonalARResiduals(self, y: np.ndarray) -> np.ndarray:
        """Compute seasonal AR residuals"""
        n = len(y)
        bigP = self.P
        m = self.m
        p = len(self.arCoefs) if self.arCoefs is not None else 0
        sarCoefs = self.sarCoefs if self.sarCoefs is not None else np.array([])
        startIdx = max(p, bigP * m)

        residuals = np.zeros(n)
        for t in range(startIdx, n):
            pred = 0.0
            for i in range(p):
                pred += self.arCoefs[i] * y[t - i - 1]
            for i in range(len(sarCoefs)):
                idx = t - (i + 1) * m
                if idx >= 0:
                    pred += sarCoefs[i] * y[idx]
            residuals[t] = y[t] - pred

        return residuals[startIdx:]

    def _computeARResiduals(self, y: np.ndarray, arCoefs: np.ndarray) -> np.ndarray:
        n = len(y)
        p = len(arCoefs)

        if p == 0:
            return y.copy()

        yLags = np.column_stack([y[p - i - 1 : n - i - 1] for i in range(p)])
        predictions = yLags @ arCoefs
        return y[p:] - predictions

    def _computeResiduals(self, y: np.ndarray) -> np.ndarray:
        """Compute ARMA residuals"""
        n = len(y)
        p = len(self.arCoefs) if self.arCoefs is not None else 0
        q = len(self.maCoefs) if self.maCoefs is not None else 0
        bigP = len(self.sarCoefs) if self.sarCoefs is not None else 0
        bigQ = len(self.smaCoefs) if self.smaCoefs is not None else 0
        m = self.m

        maxLag = max(p, q, bigP * m, bigQ * m, 1)
        residuals = np.zeros(n)

        for t in range(maxLag, n):
            pred = self.constant

            for i in range(p):
                pred += self.arCoefs[i] * y[t - i - 1]

            for i in range(bigP):
                idx = t - (i + 1) * m
                if idx >= 0:
                    pred += self.sarCoefs[i] * y[idx]

            for j in range(min(q, t)):
                pred += self.maCoefs[j] * residuals[t - j - 1]

            for j in range(bigQ):
                idx = t - (j + 1) * m
                if idx >= 0:
                    pred += self.smaCoefs[j] * residuals[idx]

            residuals[t] = y[t] - pred

        return residuals[maxLag:]

    def _optimizeCSS(self, y: np.ndarray):
        """CSS (Conditional Sum of Squares) optimization - Numba accelerated"""
        p = len(self.arCoefs) if self.arCoefs is not None else 0
        q = len(self.maCoefs) if self.maCoefs is not None else 0
        bigP = len(self.sarCoefs) if self.sarCoefs is not None else 0
        bigQ = len(self.smaCoefs) if self.smaCoefs is not None else 0
        m = self.m

        totalParams = p + q + bigP + bigQ

        if totalParams == 0:
            return

        x0 = []
        if p > 0:
            x0.extend(self.arCoefs)
        if q > 0:
            x0.extend(self.maCoefs)
        if bigP > 0:
            x0.extend(self.sarCoefs)
        if bigQ > 0:
            x0.extend(self.smaCoefs)

        bounds = [(-0.99, 0.99)] * totalParams

        hasSeasonal = bigP + bigQ > 0

        if hasSeasonal:
            if RUST_AVAILABLE:
                def objective(params):
                    arC = np.asarray(params[:p], dtype=np.float64) if p > 0 else np.zeros(0, dtype=np.float64)
                    maC = np.asarray(params[p:p + q], dtype=np.float64) if q > 0 else np.zeros(0, dtype=np.float64)
                    sarC = np.asarray(params[p + q:p + q + bigP], dtype=np.float64) if bigP > 0 else np.zeros(0, dtype=np.float64)
                    smaC = np.asarray(params[p + q + bigP:], dtype=np.float64) if bigQ > 0 else np.zeros(0, dtype=np.float64)
                    return _seasonalCSSObjectiveRust(y, arC, maC, sarC, smaC, m)
            else:
                def objective(params):
                    arC = np.asarray(params[:p]) if p > 0 else np.zeros(0)
                    maC = np.asarray(params[p:p + q]) if q > 0 else np.zeros(0)
                    sarC = np.asarray(params[p + q:p + q + bigP]) if bigP > 0 else np.zeros(0)
                    smaC = np.asarray(params[p + q + bigP:]) if bigQ > 0 else np.zeros(0)
                    return _seasonalCSSObjectiveNumba(y, arC, maC, sarC, smaC, m)
        else:
            if RUST_AVAILABLE:
                def objective(params):
                    arC = np.asarray(params[:p], dtype=np.float64) if p > 0 else np.zeros(0, dtype=np.float64)
                    maC = np.asarray(params[p:], dtype=np.float64) if q > 0 else np.zeros(0, dtype=np.float64)
                    return _cssObjectiveRust(y, arC, maC)
            else:
                def objective(params):
                    arC = np.asarray(params[:p]) if p > 0 else np.zeros(0)
                    maC = np.asarray(params[p:]) if q > 0 else np.zeros(0)
                    return _cssObjectiveNumba(y, arC, maC)

        try:
            result = minimize(
                objective, x0, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 50, 'ftol': 1e-4}
            )
            idx = 0
            if p > 0:
                self.arCoefs = result.x[idx:idx + p]
                idx += p
            if q > 0:
                self.maCoefs = result.x[idx:idx + q]
                idx += q
            if bigP > 0:
                self.sarCoefs = result.x[idx:idx + bigP]
                idx += bigP
            if bigQ > 0:
                self.smaCoefs = result.x[idx:idx + bigQ]
        except Exception:
            pass

    def _predictDiff(self, steps: int) -> np.ndarray:
        """Forecast in differenced space"""
        y = self.diffData
        n = len(y)
        p = len(self.arCoefs) if self.arCoefs is not None else 0
        q = len(self.maCoefs) if self.maCoefs is not None else 0
        bigP = len(self.sarCoefs) if self.sarCoefs is not None else 0
        bigQ = len(self.smaCoefs) if self.smaCoefs is not None else 0
        m = self.m

        extended = np.concatenate([y, np.zeros(steps)])
        extResiduals = np.concatenate([
            self.residuals if self.residuals is not None else np.zeros(n),
            np.zeros(steps)
        ])

        for h in range(steps):
            t = n + h
            pred = self.constant

            for i in range(p):
                if t - i - 1 >= 0:
                    pred += self.arCoefs[i] * extended[t - i - 1]

            for i in range(bigP):
                idx = t - (i + 1) * m
                if idx >= 0:
                    pred += self.sarCoefs[i] * extended[idx]

            for j in range(q):
                if t - j - 1 >= 0 and t - j - 1 < len(self.residuals):
                    pred += self.maCoefs[j] * self.residuals[t - j - 1]

            for j in range(bigQ):
                idx = t - (j + 1) * m
                if idx >= 0 and idx < len(self.residuals):
                    pred += self.smaCoefs[j] * self.residuals[idx]

            extended[t] = pred

        return extended[n:]

    def _computePsiWeights(self, steps: int) -> np.ndarray:
        """Psi weights of the MA(infinity) representation"""
        p = len(self.arCoefs) if self.arCoefs is not None else 0
        q = len(self.maCoefs) if self.maCoefs is not None else 0

        psi = np.zeros(steps)
        psi[0] = 1.0

        for j in range(1, steps):
            val = 0.0

            for i in range(min(p, j)):
                val += self.arCoefs[i] * psi[j - i - 1]

            if j <= q:
                val += self.maCoefs[j - 1]

            psi[j] = val

        return psi

    def _simpleFit(self, y: np.ndarray):
        """Simple fit (when data is insufficient)"""
        self.arCoefs = np.array([0.5]) if self.p > 0 else np.array([])
        self.maCoefs = np.array([0.3]) if self.q > 0 else np.array([])
        self.sarCoefs = np.array([0.3]) if self.P > 0 else np.array([])
        self.smaCoefs = np.array([0.2]) if self.Q > 0 else np.array([])
        self.residuals = np.zeros(len(y))
        self.sigma2 = np.var(y) if len(y) > 0 else 1.0


class AutoARIMA:
    """
    Automatic ARIMA Model Selection

    Automatically selects the optimal (p, d, q) based on AICc
    Supports seasonal ARIMA
    """

    def __init__(
        self,
        maxP: int = 3,
        maxD: int = 2,
        maxQ: int = 3,
        seasonalPeriod: int = 7
    ):
        self.maxP = maxP
        self.maxD = maxD
        self.maxQ = maxQ
        self.seasonalPeriod = seasonalPeriod

        self.bestModel = None
        self.bestOrder = None
        self.bestSeasonalOrder = None
        self.bestAIC = np.inf

    def fit(self, y: np.ndarray) -> ARIMAModel:
        """
        Automatic optimal ARIMA selection

        Parameters
        ----------
        y : np.ndarray
            Time series data

        Returns
        -------
        ARIMAModel
            Optimal model
        """
        n = len(y)

        d = self._determineDifferencingOrder(y)

        priorityOrders = [
            (1, d, 1),
            (1, d, 0),
            (0, d, 1),
            (2, d, 1),
            (1, d, 2),
        ]

        modelsEvaluated = 0
        noImprovementCount = 0

        for p, dd, q in priorityOrders:
            if n < p + q + dd + 10:
                continue

            try:
                model = ARIMAModel(order=(p, dd, q))
                model.fit(y)
                aic = self._computeAICc(y, model, p, dd, q)
                modelsEvaluated += 1

                if aic < self.bestAIC:
                    self.bestAIC = aic
                    self.bestModel = model
                    self.bestOrder = (p, dd, q)
                    self.bestSeasonalOrder = None
                    noImprovementCount = 0
                else:
                    noImprovementCount += 1

                if modelsEvaluated >= 2 and noImprovementCount >= 1:
                    break

            except Exception:
                continue

        seasonal = self._hasSeasonality(y, self.seasonalPeriod)
        if seasonal:
            self._fitSeasonalCandidates(y, d)

        if self.bestModel is None:
            self.bestModel = ARIMAModel(order=(1, 1, 1))
            self.bestModel.fit(y)
            self.bestOrder = (1, 1, 1)
            self.bestSeasonalOrder = None

        return self.bestModel

    def _fitSeasonalCandidates(self, y: np.ndarray, d: int):
        """Evaluate seasonal ARIMA candidate models"""
        n = len(y)
        m = self.seasonalPeriod

        baseOrders = [
            (1, d, 1),
            (0, d, 1),
        ]

        seasonalOrders = [
            (0, 1, 1, m),
            (1, 1, 0, m),
        ]

        noImprovement = 0
        for p, dd, q in baseOrders:
            for bigP, bigD, bigQ, period in seasonalOrders:
                minRequired = p + q + dd + bigP * m + bigD * m + bigQ * m + 10
                if n < minRequired:
                    continue

                try:
                    model = ARIMAModel(
                        order=(p, dd, q),
                        seasonalOrder=(bigP, bigD, bigQ, period)
                    )
                    model.fit(y)

                    totalK = p + q + bigP + bigQ + 1
                    nEff = len(model.residuals) if model.residuals is not None else n - dd
                    if model.residuals is None or len(model.residuals) == 0:
                        continue

                    sse = np.sum(model.residuals ** 2)
                    sigma2 = sse / len(model.residuals)
                    if sigma2 <= 0:
                        continue

                    aic = nEff * np.log(sigma2) + 2 * totalK
                    if nEff - totalK - 1 > 0:
                        aicc = aic + 2 * totalK * (totalK + 1) / (nEff - totalK - 1)
                    else:
                        aicc = np.inf

                    if aicc < self.bestAIC:
                        self.bestAIC = aicc
                        self.bestModel = model
                        self.bestOrder = (p, dd, q)
                        self.bestSeasonalOrder = (bigP, bigD, bigQ, period)
                        noImprovement = 0
                    else:
                        noImprovement += 1

                    if noImprovement >= 2:
                        return

                except Exception:
                    continue

    def _hasSeasonality(self, y: np.ndarray, period: int) -> bool:
        """Check for seasonality"""
        if len(y) < period * 2:
            return False

        acf = TurboCore.acf(y, period + 1)
        return abs(acf[period]) > 0.3

    def _determineDifferencingOrder(self, y: np.ndarray) -> int:
        """Determine differencing order based on KPSS/ADF"""
        adfStat = TurboCore.adfStatistic(y)

        if adfStat < -2.86:
            return 0

        diffed = TurboCore.diff(y, 1)
        adfStat1 = TurboCore.adfStatistic(diffed)

        if adfStat1 < -2.86:
            return 1

        return min(2, self.maxD)

    def _computeAICc(
        self,
        y: np.ndarray,
        model: ARIMAModel,
        p: int, d: int, q: int
    ) -> float:
        """Compute AICc"""
        n = len(y) - d

        if model.residuals is None or len(model.residuals) == 0:
            return np.inf

        sse = np.sum(model.residuals ** 2)
        sigma2 = sse / len(model.residuals)

        if sigma2 <= 0:
            return np.inf

        k = p + q + 1

        aic = n * np.log(sigma2) + 2 * k

        if n - k - 1 > 0:
            aicc = aic + 2 * k * (k + 1) / (n - k - 1)
        else:
            aicc = np.inf

        return aicc

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forecast"""
        if self.bestModel is None:
            raise ValueError("Model has not been fitted.")
        return self.bestModel.predict(steps)
