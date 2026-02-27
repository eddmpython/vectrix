"""
ForecastX Easy API

    >>> from forecastx.easy import forecast, analyze, regress, quick_report
    >>> result = forecast("sales.csv", steps=30)
    >>> result = forecast(df, date="date", value="sales", steps=30)
    >>> report = analyze(df, date="date", value="sales")
    >>> result = regress(y, X, summary=True)
    >>> result = regress(df, formula="sales ~ ads + price")
    >>> report = quick_report(df, date="date", value="sales")
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple



def _autoDetectColumns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    DataFrame에서 자동으로 날짜/값 컬럼 감지

    날짜: datetime 타입 또는 이름에 'date', 'time', 'dt', '날짜' 포함
    값: 첫 번째 숫자 컬럼 (날짜 제외)

    Parameters
    ----------
    df : pd.DataFrame
        입력 데이터프레임

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        (dateCol, valueCol) 튜플. 감지 실패 시 None.
    """
    dateCol = None
    valueCol = None

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dateCol = col
            break

    if dateCol is None:
        dateKeywords = ['date', 'time', 'dt', 'timestamp', '날짜', '일자', '일시']
        for col in df.columns:
            colLower = str(col).lower()
            if any(kw in colLower for kw in dateKeywords):
                try:
                    pd.to_datetime(df[col].head(5))
                    dateCol = col
                    break
                except (ValueError, TypeError):
                    continue

    for col in df.columns:
        if col == dateCol:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            valueCol = col
            break

    return dateCol, valueCol



def _prepareData(
    data: Union[str, pd.DataFrame, np.ndarray, list, tuple, pd.Series, dict],
    date: Optional[str],
    value: Optional[str]
) -> Tuple[pd.DataFrame, str, str]:
    """다양한 입력을 (DataFrame, dateCol, valueCol) 형태로 통일"""
    if isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            raise ValueError(
                f"파일을 찾을 수 없습니다: '{data}'\n"
                "파일 경로를 확인해주세요. 현재 디렉터리 기준 상대 경로 또는 절대 경로를 사용하세요."
            )
        except Exception as e:
            raise ValueError(
                f"CSV 파일을 읽는 중 오류가 발생했습니다: '{data}'\n"
                f"오류 내용: {e}\n"
                "올바른 CSV 파일인지 확인해주세요."
            )

    elif isinstance(data, (list, tuple, np.ndarray)):
        values = np.asarray(data, dtype=np.float64).ravel()
        if len(values) == 0:
            raise ValueError(
                "빈 데이터가 전달되었습니다.\n"
                "최소 10개 이상의 데이터를 입력해주세요."
            )
        nPoints = len(values)
        fakeDates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=nPoints, freq='D')
        df = pd.DataFrame({
            '_date': fakeDates,
            '_value': values
        })
        if date is None:
            date = '_date'
        if value is None:
            value = '_value'

    elif isinstance(data, pd.Series):
        values = data.values.astype(np.float64).ravel()
        if len(values) == 0:
            raise ValueError(
                "빈 Series가 전달되었습니다.\n"
                "최소 10개 이상의 데이터를 입력해주세요."
            )
        if isinstance(data.index, pd.DatetimeIndex):
            fakeDates = data.index
        else:
            nPoints = len(values)
            fakeDates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=nPoints, freq='D')
        df = pd.DataFrame({
            '_date': fakeDates,
            '_value': values
        })
        if date is None:
            date = '_date'
        if value is None:
            value = '_value'

    elif isinstance(data, dict):
        try:
            df = pd.DataFrame(data)
        except Exception as e:
            raise ValueError(
                f"dict를 DataFrame으로 변환할 수 없습니다.\n"
                f"오류 내용: {e}\n"
                "dict는 {'컬럼명': [값, ...]} 형태로 전달해주세요."
            )

    elif isinstance(data, pd.DataFrame):
        df = data.copy()

    else:
        raise ValueError(
            f"지원하지 않는 데이터 타입: {type(data).__name__}\n"
            "지원: str(CSV경로), DataFrame, Series, ndarray, list, tuple, dict"
        )

    if date is None or value is None:
        autoDate, autoValue = _autoDetectColumns(df)
        if date is None:
            date = autoDate
        if value is None:
            value = autoValue

    if date is None:
        raise ValueError(
            "날짜 컬럼을 자동으로 감지하지 못했습니다.\n"
            "date='컬럼명' 파라미터로 직접 지정해주세요.\n"
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    if value is None:
        raise ValueError(
            "값 컬럼을 자동으로 감지하지 못했습니다.\n"
            "value='컬럼명' 파라미터로 직접 지정해주세요.\n"
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    if date not in df.columns:
        raise ValueError(
            f"'{date}' 컬럼이 데이터에 없습니다.\n"
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    if value not in df.columns:
        raise ValueError(
            f"'{value}' 컬럼이 데이터에 없습니다.\n"
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df[date]):
        try:
            df[date] = pd.to_datetime(df[date])
        except Exception:
            raise ValueError(
                f"'{date}' 컬럼을 날짜로 변환할 수 없습니다.\n"
                "날짜 형식을 확인해주세요 (예: '2024-01-01', '2024/01/01')."
            )

    if not pd.api.types.is_numeric_dtype(df[value]):
        try:
            df[value] = pd.to_numeric(df[value], errors='coerce')
        except Exception:
            raise ValueError(
                f"'{value}' 컬럼을 숫자로 변환할 수 없습니다.\n"
                "숫자 데이터인지 확인해주세요."
            )

    return df, date, value



def _parseFormula(formula: str, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    "y ~ x1 + x2 + x3" 형식 파싱

    지원:
    - "y ~ x1 + x2"  기본
    - "y ~ ."         모든 다른 컬럼
    - "y ~ x1 * x2"  교호작용 (x1 + x2 + x1:x2)
    - "y ~ x1 + I(x2**2)"  다항식

    Parameters
    ----------
    formula : str
        R-style formula 문자열
    data : pd.DataFrame
        데이터프레임

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        (y, X, featureNames)

    Raises
    ------
    ValueError
        formula 형식이 잘못된 경우
    """
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError(
            f"formula 형식이 잘못되었습니다: '{formula}'\n"
            "올바른 형식: 'y ~ x1 + x2 + x3'\n"
            "  - 'y ~ .'  는 y 이외 모든 컬럼을 의미합니다."
        )

    yName = parts[0].strip()
    rhsRaw = parts[1].strip()

    if yName not in data.columns:
        raise ValueError(
            f"종속변수 '{yName}'이 데이터에 없습니다.\n"
            f"사용 가능한 컬럼: {list(data.columns)}"
        )

    y = data[yName].values.astype(np.float64)

    if rhsRaw.strip() == '.':
        xCols = [c for c in data.columns if c != yName and pd.api.types.is_numeric_dtype(data[c])]
        if not xCols:
            raise ValueError(
                f"'{yName}' 이외에 숫자형 컬럼이 없습니다.\n"
                f"사용 가능한 컬럼: {list(data.columns)}"
            )
        X = data[xCols].values.astype(np.float64)
        return y, X, xCols

    featureNames = []
    featureArrays = []

    terms = _splitTerms(rhsRaw)

    for term in terms:
        term = term.strip()
        if not term:
            continue

        if '*' in term:
            subTerms = [t.strip() for t in term.split('*')]
            if len(subTerms) == 2:
                t1, t2 = subTerms
                for t in [t1, t2]:
                    if t in data.columns:
                        featureNames.append(t)
                        featureArrays.append(data[t].values.astype(np.float64))

                if t1 in data.columns and t2 in data.columns:
                    interactionName = f"{t1}:{t2}"
                    featureNames.append(interactionName)
                    featureArrays.append(
                        data[t1].values.astype(np.float64) * data[t2].values.astype(np.float64)
                    )
            continue

        if ':' in term and 'I(' not in term:
            subTerms = [t.strip() for t in term.split(':')]
            if all(t in data.columns for t in subTerms):
                interactionName = ':'.join(subTerms)
                featureNames.append(interactionName)
                result = np.ones(len(data), dtype=np.float64)
                for t in subTerms:
                    result *= data[t].values.astype(np.float64)
                featureArrays.append(result)
            continue

        if term.startswith('I(') and term.endswith(')'):
            innerExpr = term[2:-1].strip()
            for op in ['**', '^']:
                if op in innerExpr:
                    varName, powerStr = innerExpr.split(op, 1)
                    varName = varName.strip()
                    try:
                        power = float(powerStr.strip())
                    except ValueError:
                        raise ValueError(
                            f"다항식 표현을 해석할 수 없습니다: '{term}'\n"
                            "올바른 형식: I(x**2) 또는 I(x^2)"
                        )
                    if varName in data.columns:
                        featureNames.append(f"I({varName}**{int(power)})")
                        featureArrays.append(
                            data[varName].values.astype(np.float64) ** power
                        )
                    else:
                        raise ValueError(
                            f"변수 '{varName}'이 데이터에 없습니다.\n"
                            f"사용 가능한 컬럼: {list(data.columns)}"
                        )
                    break
            else:
                if innerExpr in data.columns:
                    featureNames.append(term)
                    featureArrays.append(data[innerExpr].values.astype(np.float64))
            continue

        if term in data.columns:
            featureNames.append(term)
            featureArrays.append(data[term].values.astype(np.float64))
        else:
            raise ValueError(
                f"변수 '{term}'이 데이터에 없습니다.\n"
                f"사용 가능한 컬럼: {list(data.columns)}\n"
                "팁: 다항식은 I(x**2) 형태로, 교호작용은 x1 * x2 형태로 입력하세요."
            )

    if not featureArrays:
        raise ValueError(
            f"formula에서 유효한 독립변수를 찾지 못했습니다: '{formula}'\n"
            "최소 하나 이상의 독립변수를 지정해주세요."
        )

    X = np.column_stack(featureArrays)
    return y, X, featureNames


def _splitTerms(rhs: str) -> List[str]:
    """
    '+' 기준으로 항을 분리하되, 괄호 안의 '+'는 무시

    Parameters
    ----------
    rhs : str
        formula의 우변 문자열

    Returns
    -------
    List[str]
        분리된 항 목록
    """
    terms = []
    current = []
    depth = 0
    for ch in rhs:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == '+' and depth == 0:
            terms.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        terms.append(''.join(current))
    return terms



class EasyForecastResult:
    """
    예측 결과 래퍼 - 초보자 친화적

    Attributes
    ----------
    predictions : np.ndarray
        예측값 배열
    dates : list
        예측 날짜 문자열 목록
    lower : np.ndarray
        95% 하한 신뢰구간
    upper : np.ndarray
        95% 상한 신뢰구간
    model : str
        선택된 최적 모델 이름
    """

    def __init__(self, forecastResult, df=None, historicalValues=None):
        self.predictions = forecastResult.predictions
        self.dates = forecastResult.dates
        self.lower = forecastResult.lower95
        self.upper = forecastResult.upper95
        self.model = forecastResult.bestModelName
        self._raw = forecastResult
        self._historicalValues = historicalValues

    def summary(self) -> str:
        """
        간단한 텍스트 요약

        Returns
        -------
        str
            예측 결과 요약 문자열
        """
        nSteps = len(self.predictions)
        lines = []
        lines.append("=" * 50)
        lines.append("        ForecastX 예측 결과 요약")
        lines.append("=" * 50)
        lines.append(f"  선택 모델: {self.model}")
        lines.append(f"  예측 기간: {nSteps}일")

        if nSteps > 0:
            lines.append(f"  예측 시작: {self.dates[0]}")
            lines.append(f"  예측 종료: {self.dates[-1]}")
            lines.append(f"  예측 평균: {np.mean(self.predictions):.2f}")
            lines.append(f"  예측 최소: {np.min(self.predictions):.2f}")
            lines.append(f"  예측 최대: {np.max(self.predictions):.2f}")

        rawResult = self._raw
        if rawResult.warnings:
            lines.append("")
            lines.append("  [경고]")
            for w in rawResult.warnings:
                lines.append(f"    - {w}")

        if rawResult.allModelResults:
            lines.append("")
            lines.append("  [모델 비교]")
            sortedModels = sorted(
                rawResult.allModelResults.values(),
                key=lambda m: m.mape
            )
            for m in sortedModels[:5]:
                flatMark = " (일직선)" if m.flatInfo and m.flatInfo.isFlat else ""
                lines.append(f"    {m.modelName}: MAPE={m.mape:.2f}%{flatMark}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        예측을 DataFrame으로 변환

        Returns
        -------
        pd.DataFrame
            columns: date, prediction, lower95, upper95
        """
        return pd.DataFrame({
            'date': self.dates,
            'prediction': self.predictions,
            'lower95': self.lower,
            'upper95': self.upper
        })

    def __str__(self):
        n = len(self.predictions)
        lines = []
        lines.append(f"ForecastX Result: {n} steps | Model: {self.model}")
        separator = "\u2500" * 40
        lines.append(separator)
        lines.append(f"  {'Date':<16}{'Forecast':>10}  {'[95% CI]'}")

        def _fmtRow(i):
            d = self.dates[i] if self.dates is not None else f"t+{i+1}"
            pred = f"{self.predictions[i]:.2f}"
            lo = f"{self.lower[i]:.1f}"
            hi = f"{self.upper[i]:.1f}"
            return f"  {str(d):<16}{pred:>10}  [{lo}, {hi}]"

        if n <= 10:
            for i in range(n):
                lines.append(_fmtRow(i))
        else:
            for i in range(3):
                lines.append(_fmtRow(i))
            lines.append(f"  {'...':<16}{'...':>10}  ...")
            for i in range(n - 3, n):
                lines.append(_fmtRow(i))

        lines.append(separator)
        return "\n".join(lines)

    def __repr__(self):
        return f"ForecastResult(model='{self.model}', steps={len(self.predictions)})"

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return pd.DataFrame({
                'date': self.dates[idx] if self.dates is not None else None,
                'prediction': self.predictions[idx],
                'lower95': self.lower[idx],
                'upper95': self.upper[idx]
            })
        return {
            'date': self.dates[idx] if self.dates is not None else None,
            'prediction': self.predictions[idx],
            'lower95': self.lower[idx],
            'upper95': self.upper[idx]
        }

    def plot(self, figsize=(12, 5), showHistory=True, showCI=True, title=None):
        """matplotlib 시각화 (선택적 의존성)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "시각화에 matplotlib가 필요합니다.\n"
                "설치: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)

        if showHistory and self._historicalValues is not None:
            nHist = len(self._historicalValues)
            ax.plot(range(nHist), self._historicalValues,
                    color='#2c3e50', linewidth=1.5, label='실측')
        else:
            nHist = 0

        forecastX = range(nHist, nHist + len(self.predictions))
        ax.plot(forecastX, self.predictions,
                color='#e74c3c', linewidth=2, label=f'예측 ({self.model})')

        if showCI:
            ax.fill_between(forecastX, self.lower, self.upper,
                            color='#e74c3c', alpha=0.15, label='95% CI')

        if title is None:
            title = f'ForecastX \u2014 {self.model}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def to_csv(self, path, **kwargs):
        """결과를 CSV로 저장."""
        self.to_dataframe().to_csv(path, index=False, **kwargs)
        return self

    def to_json(self, path=None, **kwargs):
        """결과를 JSON 문자열로 반환하거나 파일로 저장."""
        import json
        data = {
            'model': self.model,
            'predictions': self.predictions.tolist(),
            'lower95': self.lower.tolist(),
            'upper95': self.upper.tolist(),
            'dates': list(self.dates) if self.dates is not None else None,
        }
        jsonStr = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(jsonStr)
        return jsonStr

    def save(self, path):
        """결과를 JSON으로 저장 (.to_json의 편의 래퍼)."""
        self.to_json(path)
        return self

    def describe(self):
        """pandas .describe() 스타일 요약."""
        return pd.DataFrame({
            'forecast': pd.Series(self.predictions).describe(),
            'lower95': pd.Series(self.lower).describe(),
            'upper95': pd.Series(self.upper).describe()
        })


class EasyAnalysisResult:
    """
    분석 결과 래퍼 - 초보자 친화적

    Attributes
    ----------
    dna : DNAProfile
        시계열 DNA 프로파일 (difficulty, category, recommendedModels 등)
    changepoints : list
        변경점 인덱스 목록
    anomalies : list
        이상치 인덱스 목록
    features : dict
        추출된 통계적 특성 딕셔너리
    characteristics : DataCharacteristics
        데이터 특성 분석 결과
    """

    def __init__(self, dna, changepoints, anomalies, features, characteristics):
        self.dna = dna
        self.changepoints = changepoints
        self.anomalies = anomalies
        self.features = features
        self.characteristics = characteristics

    def summary(self) -> str:
        """
        분석 결과 한국어 텍스트 요약

        Returns
        -------
        str
            분석 결과 요약 문자열
        """
        lines = []
        lines.append("=" * 55)
        lines.append("        ForecastX 시계열 분석 보고서")
        lines.append("=" * 55)

        if self.dna and self.dna.summary:
            lines.append("")
            lines.append("  [DNA 분석]")
            lines.append(f"    {self.dna.summary}")
            lines.append(f"    카테고리: {self.dna.category}")
            lines.append(f"    예측 난이도: {self.dna.difficulty} ({self.dna.difficultyScore:.1f}/100)")
            lines.append(f"    지문(fingerprint): {self.dna.fingerprint}")
            if self.dna.recommendedModels:
                lines.append(f"    추천 모델: {', '.join(self.dna.recommendedModels[:3])}")

        lines.append("")
        lines.append("  [변경점 감지]")
        if self.changepoints is not None and len(self.changepoints) > 0:
            lines.append(f"    발견된 변경점: {len(self.changepoints)}개")
            lines.append(f"    위치: {list(self.changepoints[:10])}")
        else:
            lines.append("    변경점이 발견되지 않았습니다.")

        lines.append("")
        lines.append("  [이상치 감지]")
        if self.anomalies is not None and len(self.anomalies) > 0:
            lines.append(f"    이상치: {len(self.anomalies)}개")
            lines.append(f"    위치: {list(self.anomalies[:10])}")
        else:
            lines.append("    이상치가 발견되지 않았습니다.")

        if self.characteristics:
            c = self.characteristics
            lines.append("")
            lines.append("  [데이터 특성]")
            lines.append(f"    데이터 길이: {c.length}개")
            lines.append(f"    주기: {c.period}")
            freqStr = c.frequency.value if hasattr(c.frequency, 'value') else str(c.frequency)
            lines.append(f"    빈도: {freqStr}")

            if c.hasTrend:
                lines.append(f"    추세: {c.trendDirection} (강도: {c.trendStrength:.2f})")
            else:
                lines.append("    추세: 없음")

            if c.hasSeasonality:
                lines.append(f"    계절성: 있음 (강도: {c.seasonalStrength:.2f})")
            else:
                lines.append("    계절성: 없음")

            lines.append(f"    예측 가능성: {c.predictabilityScore:.1f}/100")

        lines.append("=" * 55)
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        nCp = len(self.changepoints) if self.changepoints is not None else 0
        nAn = len(self.anomalies) if self.anomalies is not None else 0
        diff = self.dna.difficulty if self.dna else 'unknown'
        return f"AnalysisResult(difficulty='{diff}', changepoints={nCp}, anomalies={nAn})"


class EasyRegressionResult:
    """
    회귀분석 결과 래퍼 - 초보자 친화적

    Attributes
    ----------
    coefficients : np.ndarray
        회귀계수 배열 (절편 포함)
    pvalues : np.ndarray
        각 계수의 p-value
    r_squared : float
        결정계수 (R-squared)
    adj_r_squared : float
        수정 결정계수
    f_stat : float
        F-통계량
    """

    def __init__(self, result, diagnosticResult=None):
        self.coefficients = result.coefficients
        self.pvalues = result.pValues
        self.r_squared = result.rSquared
        self.adj_r_squared = result.adjustedRSquared
        self.f_stat = result.fStatistic
        self._result = result
        self._diagResult = diagnosticResult
        self._olsEngine = None

    def summary(self) -> str:
        """
        회귀분석 결과 요약 (statsmodels 스타일)

        Returns
        -------
        str
            포맷된 요약 문자열
        """
        return self._result.summary()

    def diagnose(self) -> str:
        """
        회귀 진단 실행 및 결과 출력

        VIF, 등분산성, 정규성, 자기상관, 영향점 분석을 모두 한 번에 수행.
        이미 진단을 실행한 경우 캐시된 결과를 반환합니다.

        Returns
        -------
        str
            진단 결과 요약 문자열
        """
        if self._diagResult is not None:
            return self._diagResult.summary()

        try:
            from .regression import RegressionDiagnostics

            diag = RegressionDiagnostics()
            result = self._result

            nObs = result.nObs
            nParams = result.nParams

            if self._olsEngine is not None and hasattr(self._olsEngine, '_Xa'):
                Xa = self._olsEngine._Xa
                X = Xa[:, 1:] if Xa.shape[1] > 1 else Xa
            else:
                return (
                    "진단을 실행하려면 회귀분석 시 내부 데이터가 필요합니다.\n"
                    "regress() 함수를 통해 분석한 결과에서만 diagnose()를 사용할 수 있습니다."
                )

            y = result.fittedValues + result.residuals
            self._diagResult = diag.diagnose(
                X=X,
                y=y,
                residuals=result.residuals,
                hatMatrix=result.hatMatrix,
                beta=result.coefficients,
                fittedValues=result.fittedValues
            )
            return self._diagResult.summary()

        except Exception as e:
            return f"진단 실행 중 오류 발생: {e}"

    def predict(self, X, interval: str = 'prediction', alpha: float = 0.05) -> pd.DataFrame:
        """
        새 데이터에 대한 예측 + 구간을 DataFrame으로 반환

        Parameters
        ----------
        X : array-like
            새로운 독립변수 데이터 (절편 미포함)
        interval : str
            'prediction' (예측구간) 또는 'confidence' (신뢰구간) 또는 'none'
        alpha : float
            유의수준 (기본값: 0.05 -> 95% 구간)

        Returns
        -------
        pd.DataFrame
            columns: prediction, lower, upper (interval이 'none'이면 prediction만)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._olsEngine is not None:
            yPred, lower, upper = self._olsEngine.predict(X, interval=interval, alpha=alpha)
        else:
            if X.shape[1] == len(self.coefficients) - 1:
                Xa = np.column_stack([np.ones(X.shape[0]), X])
                yPred = Xa @ self.coefficients
            elif X.shape[1] == len(self.coefficients):
                yPred = X @ self.coefficients
            else:
                raise ValueError(
                    f"입력 차원 불일치: X의 열 수({X.shape[1]})가 "
                    f"모델 변수 수({len(self.coefficients) - 1})와 맞지 않습니다."
                )
            lower, upper = None, None

        if lower is not None and upper is not None:
            return pd.DataFrame({
                'prediction': yPred,
                'lower': lower,
                'upper': upper
            })
        else:
            return pd.DataFrame({
                'prediction': yPred
            })

    def __repr__(self):
        return (
            f"RegressionResult(R2={self.r_squared:.4f}, "
            f"adjR2={self.adj_r_squared:.4f}, "
            f"nCoefs={len(self.coefficients)})"
        )



def forecast(
    data: Union[str, pd.DataFrame, np.ndarray, list, tuple, pd.Series, dict],
    date: str = None,
    value: str = None,
    steps: int = 30,
    frequency: str = 'auto',
    verbose: bool = False
) -> EasyForecastResult:
    """
    한 줄 예측

    Parameters
    ----------
    data : str, DataFrame, ndarray, or list
        - str: CSV 파일 경로
        - DataFrame: pandas DataFrame
        - ndarray/list: 값 배열 (날짜 자동 생성)
    date : str, optional
        날짜 컬럼명 (DataFrame일 때). None이면 자동 감지
    value : str, optional
        값 컬럼명. None이면 자동 감지 (숫자 컬럼)
    steps : int
        예측 기간 (기본값: 30)
    frequency : str
        'auto', 'D', 'W', 'M', 'H' 등 (현재 미사용, 자동 감지)
    verbose : bool
        진행 상황 출력

    Returns
    -------
    EasyForecastResult
        .predictions, .dates, .lower, .upper, .model, .summary(), .to_dataframe()

    Raises
    ------
    ValueError
        데이터를 해석할 수 없거나 필수 컬럼이 없는 경우

    Examples
    --------
    >>> result = forecast([100, 120, 130, 115, 140, 160, 150, 170, 180, 165], steps=5)
    >>> result = forecast("data.csv", steps=30)
    >>> result = forecast(df, date="date", value="sales", steps=14)
    """
    from .forecastx import ForecastX

    df, dateCol, valueCol = _prepareData(data, date, value)

    nRows = len(df)
    if nRows < 10:
        raise ValueError(
            f"데이터가 너무 짧습니다: {nRows}개 (최소 10개 필요)\n"
            "시계열 예측에는 최소 10개 이상의 데이터가 필요합니다."
        )

    fx = ForecastX(verbose=verbose)
    rawResult = fx.forecast(df, dateCol=dateCol, valueCol=valueCol, steps=steps)

    if not rawResult.success:
        errorMsg = rawResult.error or "알 수 없는 오류"
        raise RuntimeError(
            f"예측에 실패했습니다: {errorMsg}\n"
            "데이터를 확인하거나, verbose=True로 설정하여 자세한 정보를 확인해보세요."
        )

    histValues = df[valueCol].values.astype(np.float64)
    return EasyForecastResult(rawResult, df, historicalValues=histValues)



def analyze(
    data: Union[str, pd.DataFrame, np.ndarray, list, tuple, pd.Series, dict],
    date: str = None,
    value: str = None,
    period: int = None
) -> EasyAnalysisResult:
    """
    한 줄 시계열 분석 (DNA + 변경점 + 이상치 + 특성)

    Parameters
    ----------
    data : str, DataFrame, ndarray, or list
        입력 데이터 (forecast와 동일한 형식)
    date : str, optional
        날짜 컬럼명
    value : str, optional
        값 컬럼명
    period : int, optional
        계절 주기. None이면 자동 감지

    Returns
    -------
    EasyAnalysisResult
        .dna, .changepoints, .anomalies, .features, .characteristics, .summary()

    Examples
    --------
    >>> report = analyze(df, date="date", value="sales")
    >>> print(report.dna.difficulty)    # 'medium'
    >>> print(report.changepoints)      # [45, 120, 200]
    >>> print(report.summary())
    """
    from .forecastx import ForecastX
    from .adaptive.dna import ForecastDNA
    from .engine.changepoint import ChangePointDetector

    df, dateCol, valueCol = _prepareData(data, date, value)

    workDf = df.copy()
    workDf[dateCol] = pd.to_datetime(workDf[dateCol])
    workDf = workDf.sort_values(dateCol).reset_index(drop=True)
    if workDf[valueCol].isna().any():
        workDf[valueCol] = workDf[valueCol].interpolate(method='linear')
        workDf[valueCol] = workDf[valueCol].ffill().bfill()

    values = workDf[valueCol].values.astype(np.float64)

    fx = ForecastX(verbose=False)
    analysisDict = fx.analyze(workDf, dateCol=dateCol, valueCol=valueCol)
    characteristics = analysisDict.get('characteristics', None)

    detectedPeriod = period
    if detectedPeriod is None and characteristics is not None:
        detectedPeriod = characteristics.period
    if detectedPeriod is None or detectedPeriod < 1:
        detectedPeriod = 1

    dna = ForecastDNA()
    dnaProfile = dna.analyze(values, period=detectedPeriod)

    changepointIndices = np.array([])
    try:
        cpDetector = ChangePointDetector()
        cpResult = cpDetector.detect(values, method='auto')
        changepointIndices = cpResult.indices
    except Exception:
        pass

    anomalyIndices = np.array([])
    try:
        if len(values) > 3:
            valMean = np.mean(values)
            valStd = np.std(values, ddof=1)
            if valStd > 1e-10:
                zScores = np.abs((values - valMean) / valStd)
                anomalyIndices = np.where(zScores > 3.0)[0]
    except Exception:
        pass

    features = dnaProfile.features if dnaProfile else {}

    return EasyAnalysisResult(
        dna=dnaProfile,
        changepoints=changepointIndices,
        anomalies=anomalyIndices,
        features=features,
        characteristics=characteristics
    )



def regress(
    y: Union[np.ndarray, pd.Series, str] = None,
    X: Union[np.ndarray, pd.DataFrame] = None,
    data: pd.DataFrame = None,
    formula: str = None,
    method: str = 'ols',
    summary: bool = True
) -> EasyRegressionResult:
    """
    한 줄 회귀분석 (statsmodels 수준)

    두 가지 방식으로 사용할 수 있습니다:

    1. 직접 입력: regress(y, X)
    2. Formula 방식: regress(data=df, formula="sales ~ ads + price")

    Parameters
    ----------
    y : array-like or None
        종속변수 (직접 입력 방식)
    X : array-like or None
        독립변수 행렬 (직접 입력 방식)
    data : pd.DataFrame or None
        formula 방식에서 사용할 데이터프레임
    formula : str or None
        R-style formula 문자열 (예: "y ~ x1 + x2")
    method : str
        'ols', 'ridge', 'lasso', 'huber', 'quantile'
    summary : bool
        True이면 결과 summary를 자동 출력

    Returns
    -------
    EasyRegressionResult
        .coefficients, .pvalues, .r_squared, .summary(), .diagnose(), .predict()

    Raises
    ------
    ValueError
        입력 파라미터가 잘못된 경우

    Examples
    --------
    >>> result = regress(y, X)
    >>> result = regress(data=df, formula="sales ~ ads + price + season")
    >>> result.diagnose()   # VIF, 등분산성, 정규성 모두 한 번에
    """
    from .regression import OLSInference, RegressionDiagnostics
    from .regression import RidgeRegressor, LassoRegressor, HuberRegressor, QuantileRegressor

    featureNames = None
    olsEngine = None

    if formula is not None and data is not None:
        yArr, XArr, featureNames = _parseFormula(formula, data)

    elif y is not None and X is not None:
        yArr = np.asarray(y, dtype=np.float64).ravel()
        if isinstance(X, pd.DataFrame):
            featureNames = list(X.columns)
            XArr = X.values.astype(np.float64)
        else:
            XArr = np.asarray(X, dtype=np.float64)
            if XArr.ndim == 1:
                XArr = XArr.reshape(-1, 1)

    else:
        raise ValueError(
            "다음 중 하나의 방식으로 입력해주세요:\n"
            "  1. regress(y, X) - y와 X를 직접 입력\n"
            "  2. regress(data=df, formula='y ~ x1 + x2') - formula 방식\n"
            "\n"
            "예시:\n"
            "  result = regress(y_array, X_array)\n"
            "  result = regress(data=my_df, formula='sales ~ ads + price')"
        )

    if len(yArr) != XArr.shape[0]:
        raise ValueError(
            f"y와 X의 행 수가 일치하지 않습니다: y={len(yArr)}, X={XArr.shape[0]}\n"
            "y의 길이와 X의 행 수를 맞춰주세요."
        )

    if len(yArr) < XArr.shape[1] + 2:
        raise ValueError(
            f"데이터가 너무 적습니다: 관측치 {len(yArr)}개, 변수 {XArr.shape[1]}개\n"
            f"최소 {XArr.shape[1] + 2}개 이상의 관측치가 필요합니다."
        )

    validMask = np.isfinite(yArr)
    for j in range(XArr.shape[1]):
        validMask &= np.isfinite(XArr[:, j])

    nDropped = np.sum(~validMask)
    if nDropped > 0:
        yArr = yArr[validMask]
        XArr = XArr[validMask]
        if len(yArr) < XArr.shape[1] + 2:
            raise ValueError(
                f"결측치 제거 후 데이터가 부족합니다: {len(yArr)}개 남음\n"
                f"원본에서 {nDropped}개의 결측치가 제거되었습니다."
            )

    methodLower = method.lower()

    if methodLower == 'ols':
        olsEngine = OLSInference(fitIntercept=True)
        regressionResult = olsEngine.fit(XArr, yArr, featureNames=featureNames)

    elif methodLower == 'ridge':
        ridgeModel = RidgeRegressor(alpha=1.0, fitIntercept=True)
        ridgeModel.fit(XArr, yArr)
        olsEngine = OLSInference(fitIntercept=True)
        regressionResult = olsEngine.fit(XArr, yArr, featureNames=featureNames)

    elif methodLower == 'lasso':
        lassoModel = LassoRegressor(alpha=0.1, fitIntercept=True)
        lassoModel.fit(XArr, yArr)
        olsEngine = OLSInference(fitIntercept=True)
        regressionResult = olsEngine.fit(XArr, yArr, featureNames=featureNames)

    elif methodLower == 'huber':
        huberModel = HuberRegressor(fitIntercept=True)
        huberModel.fit(XArr, yArr)
        olsEngine = OLSInference(fitIntercept=True)
        regressionResult = olsEngine.fit(XArr, yArr, featureNames=featureNames)

    elif methodLower == 'quantile':
        quantModel = QuantileRegressor(quantile=0.5, fitIntercept=True)
        quantModel.fit(XArr, yArr)
        olsEngine = OLSInference(fitIntercept=True)
        regressionResult = olsEngine.fit(XArr, yArr, featureNames=featureNames)

    else:
        raise ValueError(
            f"지원하지 않는 method입니다: '{method}'\n"
            "다음 중 하나를 사용해주세요: 'ols', 'ridge', 'lasso', 'huber', 'quantile'"
        )

    easyResult = EasyRegressionResult(regressionResult)
    easyResult._olsEngine = olsEngine

    if summary:
        print(easyResult.summary())

    return easyResult



def quick_report(
    data: Union[str, pd.DataFrame, np.ndarray, list, tuple, pd.Series, dict],
    date: str = None,
    value: str = None,
    steps: int = 30
) -> Dict[str, Any]:
    """
    한 줄 종합 보고서 (분석 + 예측 + 진단)

    분석과 예측을 한꺼번에 실행하고, 결과를 딕셔너리로 반환합니다.

    Parameters
    ----------
    data : str, DataFrame, ndarray, or list
        입력 데이터
    date : str, optional
        날짜 컬럼명
    value : str, optional
        값 컬럼명
    steps : int
        예측 기간 (기본값: 30)

    Returns
    -------
    Dict with keys:
        'forecast' : EasyForecastResult
            예측 결과
        'analysis' : EasyAnalysisResult
            분석 결과 (DNA, 변경점, 이상치)
        'summary' : str
            텍스트 요약

    Examples
    --------
    >>> report = quick_report(df, date="date", value="sales")
    >>> print(report['summary'])
    >>> report['forecast'].to_dataframe()
    """
    analysisResult = analyze(data, date=date, value=value)

    forecastResult = forecast(data, date=date, value=value, steps=steps)

    summaryLines = []
    summaryLines.append("=" * 60)
    summaryLines.append("          ForecastX 종합 보고서")
    summaryLines.append("=" * 60)

    if analysisResult.dna:
        dna = analysisResult.dna
        catNames = {
            'trending': '추세형',
            'seasonal': '계절형',
            'stationary': '안정형',
            'intermittent': '간헐형',
            'volatile': '변동형',
            'complex': '복합형'
        }
        diffNames = {
            'easy': '쉬움',
            'medium': '보통',
            'hard': '어려움',
            'very_hard': '매우 어려움'
        }
        catName = catNames.get(dna.category, dna.category)
        diffName = diffNames.get(dna.difficulty, dna.difficulty)

        summaryLines.append("")
        summaryLines.append(f"  시계열 유형: {catName}")
        summaryLines.append(f"  예측 난이도: {diffName} ({dna.difficultyScore:.1f}/100)")

    if analysisResult.characteristics:
        c = analysisResult.characteristics
        summaryLines.append(f"  데이터 길이: {c.length}개")
        summaryLines.append(f"  감지된 주기: {c.period}")

    nCp = len(analysisResult.changepoints) if analysisResult.changepoints is not None else 0
    nAn = len(analysisResult.anomalies) if analysisResult.anomalies is not None else 0
    summaryLines.append(f"  변경점: {nCp}개")
    summaryLines.append(f"  이상치: {nAn}개")

    summaryLines.append("")
    summaryLines.append(f"  선택 모델: {forecastResult.model}")
    summaryLines.append(f"  예측 기간: {steps}일")
    if len(forecastResult.predictions) > 0:
        summaryLines.append(f"  예측 범위: {np.min(forecastResult.predictions):.2f} ~ "
                            f"{np.max(forecastResult.predictions):.2f}")

    if analysisResult.dna and analysisResult.dna.recommendedModels:
        summaryLines.append("")
        summaryLines.append(f"  DNA 추천 모델: {', '.join(analysisResult.dna.recommendedModels[:3])}")

    summaryLines.append("=" * 60)

    summaryText = "\n".join(summaryLines)

    print(summaryText)

    return {
        'forecast': forecastResult,
        'analysis': analysisResult,
        'summary': summaryText
    }



__all__ = [
    'forecast',
    'analyze',
    'regress',
    'quick_report',
    'EasyForecastResult',
    'EasyAnalysisResult',
    'EasyRegressionResult',
]
