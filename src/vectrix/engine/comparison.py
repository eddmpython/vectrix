"""
Model Comparison Tools

Statistical tests for comparing forecast models:
- Diebold-Mariano test: Compare predictive accuracy of two models
- Giacomini-White test: Conditional predictive ability test
- Forecast encompassing test: Harvey-Leybourne-Newbold
- Model comparison table: Multi-model comparison and ranking

References:
- Diebold & Mariano (1995)
- Giacomini & White (2006)
- Harvey, Leybourne & Newbold (1997)
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import t as t_dist


class ModelComparison:
    """Model comparison tools"""

    @staticmethod
    def dieboldMariano(
        errors1: np.ndarray,
        errors2: np.ndarray,
        horizon: int = 1,
        alternative: str = 'two_sided'
    ) -> Dict[str, Any]:
        """
        Diebold-Mariano test

        Tests whether the predictive accuracy of two forecast models is statistically significantly different

        Parameters
        ----------
        errors1 : np.ndarray
            Forecast errors from model 1
        errors2 : np.ndarray
            Forecast errors from model 2
        horizon : int
            Forecast horizon (Newey-West bandwidth = horizon - 1)
        alternative : str
            'two_sided', 'less', 'greater'

        Returns
        -------
        Dict[str, Any]
            statistic, pValue, conclusion
        """
        errors1 = np.asarray(errors1, dtype=np.float64)
        errors2 = np.asarray(errors2, dtype=np.float64)

        n = len(errors1)
        if n != len(errors2):
            raise ValueError("Error arrays must have the same length.")

        if n < 3:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Insufficient data (n < 3)'
            }

        dt = errors1 ** 2 - errors2 ** 2
        dBar = np.mean(dt)

        gamma0 = np.mean((dt - dBar) ** 2)

        bandwidth = max(horizon - 1, 0)
        hacVar = gamma0
        for lag in range(1, bandwidth + 1):
            gammaLag = np.mean((dt[lag:] - dBar) * (dt[:-lag] - dBar))
            weight = 1.0 - lag / (bandwidth + 1.0)
            hacVar += 2.0 * weight * gammaLag

        hacVar = hacVar / n

        if hacVar <= 0 or np.isnan(hacVar):
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Variance estimation failed (HAC variance <= 0)'
            }

        seDBar = np.sqrt(hacVar)
        dmStat = dBar / seDBar

        df = n - 1

        if alternative == 'less':
            pValue = t_dist.cdf(dmStat, df)
        elif alternative == 'greater':
            pValue = 1.0 - t_dist.cdf(dmStat, df)
        else:
            pValue = 2.0 * (1.0 - t_dist.cdf(np.abs(dmStat), df))

        if pValue < 0.01:
            significance = '***'
        elif pValue < 0.05:
            significance = '**'
        elif pValue < 0.10:
            significance = '*'
        else:
            significance = ''

        if pValue < 0.05:
            if dmStat > 0:
                conclusion = f'Model 2 significantly better (p={pValue:.4f}) {significance}'
            else:
                conclusion = f'Model 1 significantly better (p={pValue:.4f}) {significance}'
        else:
            conclusion = f'No significant difference (p={pValue:.4f}) {significance}'.strip()

        return {
            'statistic': float(dmStat),
            'pValue': float(pValue),
            'conclusion': conclusion
        }

    @staticmethod
    def modelComparisonTable(
        modelErrors: Dict[str, np.ndarray],
        modelNames: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Model comparison table

        Compare errors across multiple models and rank them

        Parameters
        ----------
        modelErrors : Dict[str, np.ndarray]
            Model name -> error array dictionary
        modelNames : List[str] or None
            Model ordering (None uses dictionary key order)

        Returns
        -------
        Dict[str, Any]
            rankings, metrics, pairwiseTests
        """
        if not modelErrors:
            return {'rankings': [], 'metrics': {}, 'pairwiseTests': {}}

        if modelNames is None:
            modelNames = list(modelErrors.keys())

        metrics = {}
        for name in modelNames:
            errors = np.asarray(modelErrors[name], dtype=np.float64)
            n = len(errors)

            if n == 0:
                metrics[name] = {
                    'mape': np.inf,
                    'rmse': np.inf,
                    'mae': np.inf,
                    'smape': np.inf
                }
                continue

            absErrors = np.abs(errors)
            mae = float(np.mean(absErrors))
            rmse = float(np.sqrt(np.mean(errors ** 2)))

            absY = np.abs(errors)
            nonzeroMask = absY > 1e-10
            if np.any(nonzeroMask):
                mape = float(np.mean(absErrors[nonzeroMask] / absY[nonzeroMask]) * 100)
            else:
                mape = 0.0

            denominator = absErrors + absErrors
            nonzeroDenom = denominator > 1e-10
            if np.any(nonzeroDenom):
                smape = float(np.mean(
                    2.0 * absErrors[nonzeroDenom] / denominator[nonzeroDenom]
                ) * 100)
            else:
                smape = 0.0

            metrics[name] = {
                'mape': mape,
                'rmse': rmse,
                'mae': mae,
                'smape': smape
            }

        rankings = sorted(modelNames, key=lambda nm: metrics[nm]['rmse'])

        pairwiseTests = {}
        for i in range(len(modelNames)):
            for j in range(i + 1, len(modelNames)):
                name1 = modelNames[i]
                name2 = modelNames[j]
                e1 = np.asarray(modelErrors[name1], dtype=np.float64)
                e2 = np.asarray(modelErrors[name2], dtype=np.float64)

                minLen = min(len(e1), len(e2))
                if minLen < 3:
                    pairwiseTests[f'{name1} vs {name2}'] = {
                        'statistic': 0.0,
                        'pValue': 1.0,
                        'conclusion': 'Insufficient data'
                    }
                    continue

                e1Trimmed = e1[:minLen]
                e2Trimmed = e2[:minLen]
                dmResult = ModelComparison.dieboldMariano(e1Trimmed, e2Trimmed)
                pairwiseTests[f'{name1} vs {name2}'] = dmResult

        return {
            'rankings': rankings,
            'metrics': metrics,
            'pairwiseTests': pairwiseTests
        }

    @staticmethod
    def giacominiWhite(
        errors1: np.ndarray,
        errors2: np.ndarray,
        instruments: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Giacomini-White conditional predictive ability test

        Parameters
        ----------
        errors1 : np.ndarray
            Forecast errors from model 1
        errors2 : np.ndarray
            Forecast errors from model 2
        instruments : np.ndarray or None
            Instrument variable matrix. If None, uses constant only (unconditional test)

        Returns
        -------
        Dict[str, Any]
            statistic, pValue, conclusion
        """
        errors1 = np.asarray(errors1, dtype=np.float64)
        errors2 = np.asarray(errors2, dtype=np.float64)

        n = len(errors1)
        if n != len(errors2):
            raise ValueError("Error arrays must have the same length.")

        if n < 5:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Insufficient data (n < 5)'
            }

        dt = errors1 ** 2 - errors2 ** 2

        if instruments is None:
            Z = np.ones((n, 1), dtype=np.float64)
        else:
            instruments = np.asarray(instruments, dtype=np.float64)
            if instruments.ndim == 1:
                instruments = instruments.reshape(-1, 1)
            if len(instruments) != n:
                raise ValueError("Instrument matrix row count must match error array length.")
            Z = np.column_stack([np.ones(n), instruments])

        q = Z.shape[1]

        if n <= q:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Insufficient data relative to number of instruments'
            }

        Zd = Z * dt.reshape(-1, 1)

        ZtZ = Z.T @ Z
        detZtZ = np.linalg.det(ZtZ)
        if np.abs(detZtZ) < 1e-12:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Instrument matrix is singular'
            }

        ZtZInv = np.linalg.inv(ZtZ)

        beta = ZtZInv @ (Z.T @ dt)

        residuals = dt - Z @ beta
        S = np.zeros((q, q))
        for t in range(n):
            zt = Z[t, :].reshape(-1, 1)
            S += (residuals[t] ** 2) * (zt @ zt.T)
        S = S / n

        detS = np.linalg.det(S)
        if np.abs(detS) < 1e-12:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Covariance matrix is singular'
            }

        SInv = np.linalg.inv(S)
        gBar = np.mean(Zd, axis=0)
        gwStat = float(n * gBar @ SInv @ gBar)

        gwStat = max(gwStat, 0.0)

        from scipy.stats import chi2
        pValue = float(1.0 - chi2.cdf(gwStat, q))

        if pValue < 0.01:
            significance = '***'
        elif pValue < 0.05:
            significance = '**'
        elif pValue < 0.10:
            significance = '*'
        else:
            significance = ''

        if pValue < 0.05:
            conclusion = f'Significant difference in conditional predictive ability (p={pValue:.4f}) {significance}'
        else:
            conclusion = f'No significant difference in conditional predictive ability (p={pValue:.4f}) {significance}'.strip()

        return {
            'statistic': gwStat,
            'pValue': pValue,
            'conclusion': conclusion
        }

    @staticmethod
    def forecastEncompassingTest(
        errors1: np.ndarray,
        errors2: np.ndarray
    ) -> Dict[str, Any]:
        """
        Harvey-Leybourne-Newbold forecast encompassing test

        Tests whether model 1 encompasses model 2.
        Null hypothesis: Model 1 encompasses model 2 (model 2 provides no additional information)

        Parameters
        ----------
        errors1 : np.ndarray
            Forecast errors from model 1
        errors2 : np.ndarray
            Forecast errors from model 2

        Returns
        -------
        Dict[str, Any]
            statistic, pValue, conclusion
        """
        errors1 = np.asarray(errors1, dtype=np.float64)
        errors2 = np.asarray(errors2, dtype=np.float64)

        n = len(errors1)
        if n != len(errors2):
            raise ValueError("Error arrays must have the same length.")

        if n < 4:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Insufficient data (n < 4)'
            }

        ct = errors1 * (errors1 - errors2)

        cBar = np.mean(ct)

        gamma0 = np.var(ct, ddof=0)
        if gamma0 < 1e-15:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': 'Variance near zero (identical predictions)'
            }

        seCBar = np.sqrt(gamma0 / n)
        hlnStat = cBar / seCBar

        df = n - 1
        pValue = float(1.0 - t_dist.cdf(hlnStat, df))

        if pValue < 0.01:
            significance = '***'
        elif pValue < 0.05:
            significance = '**'
        elif pValue < 0.10:
            significance = '*'
        else:
            significance = ''

        if pValue < 0.05:
            conclusion = (
                f'Model 1 does not encompass model 2 '
                f'(model 2 provides additional information, p={pValue:.4f}) {significance}'
            )
        else:
            conclusion = (
                f'Model 1 encompasses model 2 '
                f'(no additional contribution from model 2, p={pValue:.4f}) {significance}'.strip()
            )

        return {
            'statistic': float(hlnStat),
            'pValue': pValue,
            'conclusion': conclusion
        }
