"""
Vectrix Easy API

    >>> from vectrix.easy import forecast, analyze, regress, quick_report
    >>> result = forecast("sales.csv", steps=30)
    >>> result = forecast(df, date="date", value="sales", steps=30)
    >>> report = analyze(df, date="date", value="sales")
    >>> result = regress(y, X, summary=True)
    >>> result = regress(df, formula="sales ~ ads + price")
    >>> report = quick_report(df, date="date", value="sales")
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def _autoDetectColumns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect date and value columns from a DataFrame.

    Date: datetime dtype or column name containing 'date', 'time', 'dt'.
    Value: first numeric column (excluding date).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        (dateCol, valueCol) tuple. None if detection fails.
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
    """Normalize various input types into (DataFrame, dateCol, valueCol)."""
    if isinstance(data, str):
        try:
            df = pd.read_csv(data)
        except FileNotFoundError:
            raise ValueError(
                f"File not found: '{data}'\n"
                "Please check the file path. Use a relative or absolute path."
            )
        except Exception as e:
            raise ValueError(
                f"Error reading CSV file: '{data}'\n"
                f"Details: {e}\n"
                "Please verify the file is a valid CSV."
            )

    elif isinstance(data, (list, tuple, np.ndarray)):
        values = np.asarray(data, dtype=np.float64).ravel()
        if len(values) == 0:
            raise ValueError(
                "Empty data provided.\n"
                "Please provide at least 10 data points."
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
                "Empty Series provided.\n"
                "Please provide at least 10 data points."
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
                f"Cannot convert dict to DataFrame.\n"
                f"Details: {e}\n"
                "Dict should be in {{'column': [values, ...]}} format."
            )

    elif isinstance(data, pd.DataFrame):
        df = data.copy()

    else:
        raise ValueError(
            f"Unsupported data type: {type(data).__name__}\n"
            "Supported: str (CSV path), DataFrame, Series, ndarray, list, tuple, dict"
        )

    if date is None or value is None:
        autoDate, autoValue = _autoDetectColumns(df)
        if date is None:
            date = autoDate
        if value is None:
            value = autoValue

    if date is None:
        raise ValueError(
            "Could not auto-detect date column.\n"
            "Please specify with date='column_name'.\n"
            f"Available columns: {list(df.columns)}"
        )

    if value is None:
        raise ValueError(
            "Could not auto-detect value column.\n"
            "Please specify with value='column_name'.\n"
            f"Available columns: {list(df.columns)}"
        )

    if date not in df.columns:
        raise ValueError(
            f"Column '{date}' not found in data.\n"
            f"Available columns: {list(df.columns)}"
        )

    if value not in df.columns:
        raise ValueError(
            f"Column '{value}' not found in data.\n"
            f"Available columns: {list(df.columns)}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df[date]):
        try:
            df[date] = pd.to_datetime(df[date])
        except Exception:
            raise ValueError(
                f"Cannot convert column '{date}' to datetime.\n"
                "Please check the date format (e.g., '2024-01-01', '2024/01/01')."
            )

    if not pd.api.types.is_numeric_dtype(df[value]):
        try:
            df[value] = pd.to_numeric(df[value], errors='coerce')
        except Exception:
            raise ValueError(
                f"Cannot convert column '{value}' to numeric.\n"
                "Please verify it contains numeric data."
            )

    return df, date, value



def _parseFormula(formula: str, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Parse an R-style formula string "y ~ x1 + x2 + x3".

    Supported syntax:
    - "y ~ x1 + x2"        basic
    - "y ~ ."              all other columns
    - "y ~ x1 * x2"       interaction (x1 + x2 + x1:x2)
    - "y ~ x1 + I(x2**2)" polynomial

    Parameters
    ----------
    formula : str
        R-style formula string.
    data : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        (y, X, featureNames)

    Raises
    ------
    ValueError
        If the formula format is invalid.
    """
    parts = formula.split('~')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid formula format: '{formula}'\n"
            "Correct format: 'y ~ x1 + x2 + x3'\n"
            "  - 'y ~ .' means all columns except y."
        )

    yName = parts[0].strip()
    rhsRaw = parts[1].strip()

    if yName not in data.columns:
        raise ValueError(
            f"Dependent variable '{yName}' not found in data.\n"
            f"Available columns: {list(data.columns)}"
        )

    y = data[yName].values.astype(np.float64)

    if rhsRaw.strip() == '.':
        xCols = [c for c in data.columns if c != yName and pd.api.types.is_numeric_dtype(data[c])]
        if not xCols:
            raise ValueError(
                f"No numeric columns found except '{yName}'.\n"
                f"Available columns: {list(data.columns)}"
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
                            f"Cannot parse polynomial expression: '{term}'\n"
                            "Correct format: I(x**2) or I(x^2)"
                        )
                    if varName in data.columns:
                        featureNames.append(f"I({varName}**{int(power)})")
                        featureArrays.append(
                            data[varName].values.astype(np.float64) ** power
                        )
                    else:
                        raise ValueError(
                            f"Variable '{varName}' not found in data.\n"
                            f"Available columns: {list(data.columns)}"
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
                f"Variable '{term}' not found in data.\n"
                f"Available columns: {list(data.columns)}\n"
                "Tip: Use I(x**2) for polynomials and x1 * x2 for interactions."
            )

    if not featureArrays:
        raise ValueError(
            f"No valid independent variables found in formula: '{formula}'\n"
            "Please specify at least one independent variable."
        )

    X = np.column_stack(featureArrays)
    return y, X, featureNames


def _splitTerms(rhs: str) -> List[str]:
    """
    Split terms by '+' while ignoring '+' inside parentheses.

    Parameters
    ----------
    rhs : str
        Right-hand side of the formula string.

    Returns
    -------
    List[str]
        List of split terms.
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
    Beginner-friendly forecast result wrapper.

    Attributes
    ----------
    predictions : np.ndarray
        Forecast values array.
    dates : list
        List of forecast date strings.
    lower : np.ndarray
        95% lower confidence interval.
    upper : np.ndarray
        95% upper confidence interval.
    model : str
        Selected best model name.
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
        Generate a text summary of the forecast result.

        Returns
        -------
        str
            Formatted summary string.
        """
        nSteps = len(self.predictions)
        lines = []
        lines.append("=" * 50)
        lines.append("        Vectrix Forecast Summary")
        lines.append("=" * 50)
        lines.append(f"  Model: {self.model}")
        lines.append(f"  Horizon: {nSteps} steps")

        if nSteps > 0:
            lines.append(f"  Start: {self.dates[0]}")
            lines.append(f"  End: {self.dates[-1]}")
            lines.append(f"  Mean: {np.mean(self.predictions):.2f}")
            lines.append(f"  Min: {np.min(self.predictions):.2f}")
            lines.append(f"  Max: {np.max(self.predictions):.2f}")

        rawResult = self._raw
        if rawResult.warnings:
            lines.append("")
            lines.append("  [Warnings]")
            for w in rawResult.warnings:
                lines.append(f"    - {w}")

        if rawResult.allModelResults:
            lines.append("")
            lines.append("  [Model Comparison]")
            sortedModels = sorted(
                rawResult.allModelResults.values(),
                key=lambda m: m.mape
            )
            for m in sortedModels[:5]:
                flatMark = " (flat)" if m.flatInfo and m.flatInfo.isFlat else ""
                lines.append(f"    {m.modelName}: MAPE={m.mape:.2f}%{flatMark}")

        lines.append("=" * 50)
        return "\n".join(lines)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert forecast to a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: date, prediction, lower95, upper95.
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
        lines.append(f"Vectrix Result: {n} steps | Model: {self.model}")
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
        """Plot forecast with matplotlib (optional dependency)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting.\n"
                "Install: pip install matplotlib"
            )

        fig, ax = plt.subplots(figsize=figsize)

        if showHistory and self._historicalValues is not None:
            nHist = len(self._historicalValues)
            ax.plot(range(nHist), self._historicalValues,
                    color='#2c3e50', linewidth=1.5, label='Actual')
        else:
            nHist = 0

        forecastX = range(nHist, nHist + len(self.predictions))
        ax.plot(forecastX, self.predictions,
                color='#e74c3c', linewidth=2, label=f'Forecast ({self.model})')

        if showCI:
            ax.fill_between(forecastX, self.lower, self.upper,
                            color='#e74c3c', alpha=0.15, label='95% CI')

        if title is None:
            title = f'Vectrix \u2014 {self.model}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def to_csv(self, path, **kwargs):
        """Save results to a CSV file."""
        self.to_dataframe().to_csv(path, index=False, **kwargs)
        return self

    def to_json(self, path=None, **kwargs):
        """Return results as JSON string or save to file."""
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
        """Save results to JSON (convenience wrapper for .to_json)."""
        self.to_json(path)
        return self

    def describe(self):
        """Pandas .describe()-style summary statistics."""
        return pd.DataFrame({
            'forecast': pd.Series(self.predictions).describe(),
            'lower95': pd.Series(self.lower).describe(),
            'upper95': pd.Series(self.upper).describe()
        })


class EasyAnalysisResult:
    """
    Beginner-friendly analysis result wrapper.

    Attributes
    ----------
    dna : DNAProfile
        Time series DNA profile (difficulty, category, recommendedModels, etc.)
    changepoints : list
        List of changepoint indices.
    anomalies : list
        List of anomaly indices.
    features : dict
        Extracted statistical features dictionary.
    characteristics : DataCharacteristics
        Data characteristics analysis result.
    """

    def __init__(self, dna, changepoints, anomalies, features, characteristics):
        self.dna = dna
        self.changepoints = changepoints
        self.anomalies = anomalies
        self.features = features
        self.characteristics = characteristics

    def summary(self) -> str:
        """
        Generate a text summary of the analysis result.

        Returns
        -------
        str
            Formatted analysis summary string.
        """
        lines = []
        lines.append("=" * 55)
        lines.append("        Vectrix Time Series Analysis Report")
        lines.append("=" * 55)

        if self.dna and self.dna.summary:
            lines.append("")
            lines.append("  [DNA Analysis]")
            lines.append(f"    {self.dna.summary}")
            lines.append(f"    Category: {self.dna.category}")
            lines.append(f"    Forecast Difficulty: {self.dna.difficulty} ({self.dna.difficultyScore:.1f}/100)")
            lines.append(f"    Fingerprint: {self.dna.fingerprint}")
            if self.dna.recommendedModels:
                lines.append(f"    Recommended Models: {', '.join(self.dna.recommendedModels[:3])}")

        lines.append("")
        lines.append("  [Changepoint Detection]")
        if self.changepoints is not None and len(self.changepoints) > 0:
            lines.append(f"    Changepoints found: {len(self.changepoints)}")
            lines.append(f"    Locations: {list(self.changepoints[:10])}")
        else:
            lines.append("    No changepoints detected.")

        lines.append("")
        lines.append("  [Anomaly Detection]")
        if self.anomalies is not None and len(self.anomalies) > 0:
            lines.append(f"    Anomalies: {len(self.anomalies)}")
            lines.append(f"    Locations: {list(self.anomalies[:10])}")
        else:
            lines.append("    No anomalies detected.")

        if self.characteristics:
            c = self.characteristics
            lines.append("")
            lines.append("  [Data Characteristics]")
            lines.append(f"    Length: {c.length}")
            lines.append(f"    Period: {c.period}")
            freqStr = c.frequency.value if hasattr(c.frequency, 'value') else str(c.frequency)
            lines.append(f"    Frequency: {freqStr}")

            if c.hasTrend:
                lines.append(f"    Trend: {c.trendDirection} (strength: {c.trendStrength:.2f})")
            else:
                lines.append("    Trend: none")

            if c.hasSeasonality:
                lines.append(f"    Seasonality: present (strength: {c.seasonalStrength:.2f})")
            else:
                lines.append("    Seasonality: none")

            lines.append(f"    Predictability: {c.predictabilityScore:.1f}/100")

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
    Beginner-friendly regression result wrapper.

    Attributes
    ----------
    coefficients : np.ndarray
        Regression coefficients (including intercept).
    pvalues : np.ndarray
        P-values for each coefficient.
    r_squared : float
        R-squared (coefficient of determination).
    adj_r_squared : float
        Adjusted R-squared.
    f_stat : float
        F-statistic.
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
        Regression result summary (statsmodels-style).

        Returns
        -------
        str
            Formatted summary string.
        """
        return self._result.summary()

    def diagnose(self) -> str:
        """
        Run regression diagnostics and return results.

        Performs VIF, homoscedasticity, normality, autocorrelation, and
        influence analysis all at once. Returns cached results if already run.

        Returns
        -------
        str
            Diagnostic summary string.
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
                    "Internal data required for diagnostics.\n"
                    "diagnose() is only available on results from the regress() function."
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
            return f"Error during diagnostics: {e}"

    def predict(self, X, interval: str = 'prediction', alpha: float = 0.05) -> pd.DataFrame:
        """
        Predict on new data with optional intervals.

        Parameters
        ----------
        X : array-like
            New independent variable data (without intercept).
        interval : str
            'prediction', 'confidence', or 'none'.
        alpha : float
            Significance level (default: 0.05 for 95% intervals).

        Returns
        -------
        pd.DataFrame
            Columns: prediction, lower, upper (only prediction if interval='none').
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
                    f"Dimension mismatch: X has {X.shape[1]} columns but "
                    f"model expects {len(self.coefficients) - 1} features."
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
    One-line forecasting.

    Parameters
    ----------
    data : str, DataFrame, ndarray, or list
        - str: CSV file path
        - DataFrame: pandas DataFrame
        - ndarray/list: value array (dates auto-generated)
    date : str, optional
        Date column name (for DataFrame). Auto-detected if None.
    value : str, optional
        Value column name. Auto-detected if None.
    steps : int
        Forecast horizon (default: 30).
    frequency : str
        'auto', 'D', 'W', 'M', 'H', etc. (currently unused, auto-detected).
    verbose : bool
        Print progress messages.

    Returns
    -------
    EasyForecastResult
        .predictions, .dates, .lower, .upper, .model, .summary(), .to_dataframe()

    Raises
    ------
    ValueError
        If data cannot be parsed or required columns are missing.

    Examples
    --------
    >>> result = forecast([100, 120, 130, 115, 140, 160, 150, 170, 180, 165], steps=5)
    >>> result = forecast("data.csv", steps=30)
    >>> result = forecast(df, date="date", value="sales", steps=14)
    """
    from .vectrix import Vectrix

    df, dateCol, valueCol = _prepareData(data, date, value)

    nRows = len(df)
    if nRows < 10:
        raise ValueError(
            f"Data too short: {nRows} points (minimum 10 required).\n"
            "Time series forecasting requires at least 10 data points."
        )

    fx = Vectrix(verbose=verbose)
    rawResult = fx.forecast(df, dateCol=dateCol, valueCol=valueCol, steps=steps)

    if not rawResult.success:
        errorMsg = rawResult.error or "Unknown error"
        raise RuntimeError(
            f"Forecast failed: {errorMsg}\n"
            "Check your data or set verbose=True for details."
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
    One-line time series analysis (DNA + changepoints + anomalies + features).

    Parameters
    ----------
    data : str, DataFrame, ndarray, or list
        Input data (same formats as forecast).
    date : str, optional
        Date column name.
    value : str, optional
        Value column name.
    period : int, optional
        Seasonal period. Auto-detected if None.

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
    from .adaptive.dna import ForecastDNA
    from .engine.changepoint import ChangePointDetector
    from .vectrix import Vectrix

    df, dateCol, valueCol = _prepareData(data, date, value)

    workDf = df.copy()
    workDf[dateCol] = pd.to_datetime(workDf[dateCol])
    workDf = workDf.sort_values(dateCol).reset_index(drop=True)
    if workDf[valueCol].isna().any():
        workDf[valueCol] = workDf[valueCol].interpolate(method='linear')
        workDf[valueCol] = workDf[valueCol].ffill().bfill()

    values = workDf[valueCol].values.astype(np.float64)

    fx = Vectrix(verbose=False)
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
    One-line regression analysis (statsmodels-level).

    Two usage modes:

    1. Direct input: regress(y, X)
    2. Formula mode: regress(data=df, formula="sales ~ ads + price")

    Parameters
    ----------
    y : array-like or None
        Dependent variable (direct input mode).
    X : array-like or None
        Independent variable matrix (direct input mode).
    data : pd.DataFrame or None
        DataFrame for formula mode.
    formula : str or None
        R-style formula string (e.g., "y ~ x1 + x2").
    method : str
        'ols', 'ridge', 'lasso', 'huber', 'quantile'.
    summary : bool
        If True, auto-print summary.

    Returns
    -------
    EasyRegressionResult
        .coefficients, .pvalues, .r_squared, .summary(), .diagnose(), .predict()

    Raises
    ------
    ValueError
        If input parameters are invalid.

    Examples
    --------
    >>> result = regress(y, X)
    >>> result = regress(data=df, formula="sales ~ ads + price + season")
    >>> result.diagnose()   # VIF, homoscedasticity, normality all at once
    """
    from .regression import (
        HuberRegressor,
        LassoRegressor,
        OLSInference,
        QuantileRegressor,
        RidgeRegressor,
    )

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
            "Please use one of the following modes:\n"
            "  1. regress(y, X) - direct input\n"
            "  2. regress(data=df, formula='y ~ x1 + x2') - formula mode\n"
            "\n"
            "Examples:\n"
            "  result = regress(y_array, X_array)\n"
            "  result = regress(data=my_df, formula='sales ~ ads + price')"
        )

    if len(yArr) != XArr.shape[0]:
        raise ValueError(
            f"Row count mismatch: y={len(yArr)}, X={XArr.shape[0]}.\n"
            "y length and X row count must match."
        )

    if len(yArr) < XArr.shape[1] + 2:
        raise ValueError(
            f"Insufficient data: {len(yArr)} observations, {XArr.shape[1]} variables.\n"
            f"At least {XArr.shape[1] + 2} observations required."
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
                f"Insufficient data after removing missing values: {len(yArr)} remaining.\n"
                f"{nDropped} missing values were removed from the original data."
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
            f"Unsupported method: '{method}'.\n"
            "Supported: 'ols', 'ridge', 'lasso', 'huber', 'quantile'."
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
    One-line comprehensive report (analysis + forecast + diagnostics).

    Runs analysis and forecasting together and returns results as a dict.

    Parameters
    ----------
    data : str, DataFrame, ndarray, or list
        Input data.
    date : str, optional
        Date column name.
    value : str, optional
        Value column name.
    steps : int
        Forecast horizon (default: 30).

    Returns
    -------
    Dict with keys:
        'forecast' : EasyForecastResult
            Forecast result.
        'analysis' : EasyAnalysisResult
            Analysis result (DNA, changepoints, anomalies).
        'summary' : str
            Text summary.

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
    summaryLines.append("          Vectrix Comprehensive Report")
    summaryLines.append("=" * 60)

    if analysisResult.dna:
        dna = analysisResult.dna
        catName = dna.category
        diffName = dna.difficulty

        summaryLines.append("")
        summaryLines.append(f"  Series Type: {catName}")
        summaryLines.append(f"  Forecast Difficulty: {diffName} ({dna.difficultyScore:.1f}/100)")

    if analysisResult.characteristics:
        c = analysisResult.characteristics
        summaryLines.append(f"  Data Length: {c.length}")
        summaryLines.append(f"  Detected Period: {c.period}")

    nCp = len(analysisResult.changepoints) if analysisResult.changepoints is not None else 0
    nAn = len(analysisResult.anomalies) if analysisResult.anomalies is not None else 0
    summaryLines.append(f"  Changepoints: {nCp}")
    summaryLines.append(f"  Anomalies: {nAn}")

    summaryLines.append("")
    summaryLines.append(f"  Selected Model: {forecastResult.model}")
    summaryLines.append(f"  Forecast Horizon: {steps} steps")
    if len(forecastResult.predictions) > 0:
        summaryLines.append(f"  Forecast Range: {np.min(forecastResult.predictions):.2f} ~ "
                            f"{np.max(forecastResult.predictions):.2f}")

    if analysisResult.dna and analysisResult.dna.recommendedModels:
        summaryLines.append("")
        summaryLines.append(f"  DNA Recommended Models: {', '.join(analysisResult.dna.recommendedModels[:3])}")

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
