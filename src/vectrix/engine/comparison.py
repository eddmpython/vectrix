"""
모델 비교 도구

예측 모델 간 통계적 비교를 위한 검정:
- Diebold-Mariano 검정: 두 모델의 예측 정확도 비교
- Giacomini-White 검정: 조건부 예측능력 검정
- 예측 포함 검정: Harvey-Leybourne-Newbold
- 모델 비교 테이블: 다중 모델 비교 및 순위

참조:
- Diebold & Mariano (1995)
- Giacomini & White (2006)
- Harvey, Leybourne & Newbold (1997)
"""

from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import t as t_dist


class ModelComparison:
    """모델 비교 도구"""

    @staticmethod
    def dieboldMariano(
        errors1: np.ndarray,
        errors2: np.ndarray,
        horizon: int = 1,
        alternative: str = 'two_sided'
    ) -> Dict[str, Any]:
        """
        Diebold-Mariano 검정

        두 예측 모델의 예측 정확도가 통계적으로 유의하게 다른지 검정

        Parameters
        ----------
        errors1 : np.ndarray
            모델 1의 예측 오차
        errors2 : np.ndarray
            모델 2의 예측 오차
        horizon : int
            예측 호라이즌 (Newey-West bandwidth = horizon - 1)
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
            raise ValueError("오차 배열의 길이가 동일해야 합니다.")

        if n < 3:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': '데이터 부족 (n < 3)'
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
                'conclusion': '분산 추정 불가 (HAC 분산 <= 0)'
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
                conclusion = f'모델 2가 유의하게 우수 (p={pValue:.4f}) {significance}'
            else:
                conclusion = f'모델 1이 유의하게 우수 (p={pValue:.4f}) {significance}'
        else:
            conclusion = f'유의한 차이 없음 (p={pValue:.4f}) {significance}'.strip()

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
        모델 비교 테이블

        여러 모델의 오차를 비교하고 순위를 매김

        Parameters
        ----------
        modelErrors : Dict[str, np.ndarray]
            모델명 -> 오차 배열 딕셔너리
        modelNames : List[str] or None
            모델 순서 지정 (None이면 딕셔너리 키 순서)

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
                        'conclusion': '데이터 부족'
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
        Giacomini-White 조건부 예측능력 검정

        Parameters
        ----------
        errors1 : np.ndarray
            모델 1의 예측 오차
        errors2 : np.ndarray
            모델 2의 예측 오차
        instruments : np.ndarray or None
            도구변수 행렬. None이면 상수항만 사용 (비조건부 검정)

        Returns
        -------
        Dict[str, Any]
            statistic, pValue, conclusion
        """
        errors1 = np.asarray(errors1, dtype=np.float64)
        errors2 = np.asarray(errors2, dtype=np.float64)

        n = len(errors1)
        if n != len(errors2):
            raise ValueError("오차 배열의 길이가 동일해야 합니다.")

        if n < 5:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': '데이터 부족 (n < 5)'
            }

        dt = errors1 ** 2 - errors2 ** 2

        if instruments is None:
            Z = np.ones((n, 1), dtype=np.float64)
        else:
            instruments = np.asarray(instruments, dtype=np.float64)
            if instruments.ndim == 1:
                instruments = instruments.reshape(-1, 1)
            if len(instruments) != n:
                raise ValueError("도구변수 행렬의 행 수가 오차 배열 길이와 동일해야 합니다.")
            Z = np.column_stack([np.ones(n), instruments])

        q = Z.shape[1]

        if n <= q:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': '도구변수 수 대비 데이터 부족'
            }

        Zd = Z * dt.reshape(-1, 1)

        ZtZ = Z.T @ Z
        detZtZ = np.linalg.det(ZtZ)
        if np.abs(detZtZ) < 1e-12:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': '도구변수 행렬 특이 (singular)'
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
                'conclusion': '공분산 행렬 특이 (singular)'
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
            conclusion = f'조건부 예측능력에 유의한 차이 (p={pValue:.4f}) {significance}'
        else:
            conclusion = f'조건부 예측능력에 유의한 차이 없음 (p={pValue:.4f}) {significance}'.strip()

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
        Harvey-Leybourne-Newbold 예측 포함 검정

        모델 1이 모델 2를 포함하는지 검정.
        귀무가설: 모델 1이 모델 2를 포함 (모델 2가 추가 정보 없음)

        Parameters
        ----------
        errors1 : np.ndarray
            모델 1의 예측 오차
        errors2 : np.ndarray
            모델 2의 예측 오차

        Returns
        -------
        Dict[str, Any]
            statistic, pValue, conclusion
        """
        errors1 = np.asarray(errors1, dtype=np.float64)
        errors2 = np.asarray(errors2, dtype=np.float64)

        n = len(errors1)
        if n != len(errors2):
            raise ValueError("오차 배열의 길이가 동일해야 합니다.")

        if n < 4:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': '데이터 부족 (n < 4)'
            }

        ct = errors1 * (errors1 - errors2)

        cBar = np.mean(ct)

        gamma0 = np.var(ct, ddof=0)
        if gamma0 < 1e-15:
            return {
                'statistic': 0.0,
                'pValue': 1.0,
                'conclusion': '분산이 0에 근접 (완전 동일 예측)'
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
                f'모델 1이 모델 2를 포함하지 않음 '
                f'(모델 2가 추가 정보 제공, p={pValue:.4f}) {significance}'
            )
        else:
            conclusion = (
                f'모델 1이 모델 2를 포함 '
                f'(모델 2의 추가 기여 없음, p={pValue:.4f}) {significance}'.strip()
            )

        return {
            'statistic': float(hlnStat),
            'pValue': pValue,
            'conclusion': conclusion
        }
