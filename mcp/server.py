"""
Vectrix MCP Server

Model Context Protocol server that exposes Vectrix forecasting
capabilities as tools for AI assistants (Claude Desktop, Claude Code, etc.).

Setup:
    pip install "mcp[cli]" vectrix

Usage with Claude Code:
    claude mcp add --transport stdio vectrix -- uv run python mcp/server.py

Usage with Claude Desktop (claude_desktop_config.json):
    {
        "mcpServers": {
            "vectrix": {
                "command": "uv",
                "args": ["run", "python", "/path/to/mcp/server.py"]
            }
        }
    }
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "Vectrix Forecasting",
    description="Time series forecasting, analysis, and regression tools powered by Vectrix",
)


@mcp.tool()
def forecast_timeseries(
    values: list[float],
    steps: int = 12,
    frequency: str = "auto",
) -> dict:
    """Forecast future values from a numeric time series.

    Args:
        values: Historical time series values as a list of floats.
        steps: Number of future periods to forecast (default: 12).
        frequency: Data frequency - 'D' (daily), 'W' (weekly), 'M' (monthly),
                   'Q' (quarterly), 'Y' (yearly), 'H' (hourly), or 'auto'.

    Returns:
        Dictionary with forecast results including predictions, confidence intervals,
        model name, and accuracy metrics.
    """
    import numpy as np
    import pandas as pd
    from vectrix import forecast

    dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    df = pd.DataFrame({"date": dates, "value": values})

    result = forecast(df, date="date", value="value", steps=steps, frequency=frequency)

    return {
        "predictions": [round(v, 4) for v in result.predictions.tolist()],
        "lower_95": [round(v, 4) for v in result.lower.tolist()],
        "upper_95": [round(v, 4) for v in result.upper.tolist()],
        "model": result.model,
        "mape": round(result.mape, 4),
        "rmse": round(result.rmse, 4),
        "mae": round(result.mae, 4),
        "smape": round(result.smape, 4),
        "steps": steps,
    }


@mcp.tool()
def forecast_csv(
    csv_path: str,
    date_column: str,
    value_column: str,
    steps: int = 12,
) -> dict:
    """Forecast from a CSV file on disk.

    Args:
        csv_path: Absolute path to the CSV file.
        date_column: Name of the date column.
        value_column: Name of the value column.
        steps: Number of future periods to forecast.

    Returns:
        Dictionary with forecast results.
    """
    import pandas as pd
    from vectrix import forecast

    df = pd.read_csv(csv_path, parse_dates=[date_column])
    result = forecast(df, date=date_column, value=value_column, steps=steps)

    return {
        "predictions": [round(v, 4) for v in result.predictions.tolist()],
        "lower_95": [round(v, 4) for v in result.lower.tolist()],
        "upper_95": [round(v, 4) for v in result.upper.tolist()],
        "model": result.model,
        "mape": round(result.mape, 4),
        "rmse": round(result.rmse, 4),
        "steps": steps,
        "summary": result.summary(),
    }


@mcp.tool()
def analyze_timeseries(values: list[float], period: int | None = None) -> dict:
    """Analyze a time series — detect patterns, seasonality, anomalies, and difficulty.

    Args:
        values: Historical time series values.
        period: Seasonal period (auto-detected if None). E.g., 12 for monthly, 7 for weekly.

    Returns:
        Dictionary with DNA profile, features, changepoints, and recommendations.
    """
    import numpy as np
    import pandas as pd
    from vectrix import analyze

    dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    df = pd.DataFrame({"date": dates, "value": values})

    result = analyze(df, date="date", value="value", period=period)

    features = {}
    if result.features:
        for k, v in result.features.items():
            if isinstance(v, (int, float)):
                features[k] = round(float(v), 4)
            else:
                features[k] = str(v)

    return {
        "difficulty": result.dna.difficulty if result.dna else "unknown",
        "category": result.dna.category if result.dna else "unknown",
        "recommended_models": result.dna.recommendedModels if result.dna else [],
        "changepoints": [int(c) for c in result.changepoints] if result.changepoints else [],
        "anomaly_count": len(result.anomalies) if result.anomalies else 0,
        "features": features,
        "summary": result.summary(),
    }


@mcp.tool()
def compare_models(
    values: list[float],
    steps: int = 12,
) -> dict:
    """Compare all available models on a time series and rank by accuracy.

    Args:
        values: Historical time series values.
        steps: Forecast horizon for evaluation.

    Returns:
        Dictionary with model comparison table sorted by sMAPE.
    """
    import pandas as pd
    from vectrix import compare

    dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    df = pd.DataFrame({"date": dates, "value": values})

    comparison = compare(df, date="date", value="value", steps=steps)

    return {
        "models": comparison.to_dict(orient="records"),
        "best_model": comparison.iloc[0]["model"] if len(comparison) > 0 else "unknown",
        "num_models_evaluated": len(comparison),
    }


@mcp.tool()
def run_regression(
    y: list[float],
    X: list[list[float]],
    feature_names: list[str] | None = None,
    method: str = "ols",
) -> dict:
    """Run regression analysis.

    Args:
        y: Dependent variable values.
        X: Independent variable matrix (list of rows, each row is a list of feature values).
        feature_names: Names for the features/columns. If None, uses x1, x2, etc.
        method: Regression method — 'ols', 'ridge', 'lasso', 'huber', 'quantile'.

    Returns:
        Dictionary with coefficients, p-values, R-squared, and diagnostics.
    """
    import numpy as np
    import pandas as pd
    from vectrix import regress

    X_arr = np.array(X)
    y_arr = np.array(y)

    if feature_names is None:
        feature_names = [f"x{i+1}" for i in range(X_arr.shape[1])]

    df = pd.DataFrame(X_arr, columns=feature_names)
    df["y"] = y_arr

    formula = "y ~ " + " + ".join(feature_names)
    result = regress(data=df, formula=formula, method=method, summary=False)

    return {
        "coefficients": {k: round(v, 6) for k, v in result.coefficients.to_dict().items()},
        "pvalues": {k: round(v, 6) for k, v in result.pvalues.to_dict().items()},
        "r_squared": round(result.r_squared, 6),
        "adj_r_squared": round(result.adj_r_squared, 6),
        "f_stat": round(result.f_stat, 4),
        "summary": result.summary(),
        "diagnostics": result.diagnose(),
    }


@mcp.tool()
def detect_anomalies(
    values: list[float],
    method: str = "auto",
    threshold: float = 3.0,
) -> dict:
    """Detect anomalies/outliers in a time series.

    Args:
        values: Time series values.
        method: Detection method — 'zscore', 'iqr', 'seasonal', 'rolling', or 'auto'.
        threshold: Detection threshold (lower = more sensitive). Default 3.0.

    Returns:
        Dictionary with anomaly indices, scores, and count.
    """
    import numpy as np
    from vectrix.business import AnomalyDetector

    detector = AnomalyDetector()
    y = np.array(values, dtype=np.float64)
    result = detector.detect(y, method=method, threshold=threshold)

    return {
        "anomaly_indices": result.indices.tolist(),
        "anomaly_scores": [round(s, 4) for s in result.scores[result.indices].tolist()],
        "num_anomalies": result.nAnomalies,
        "anomaly_ratio": round(result.anomalyRatio, 4),
        "method": result.method,
        "threshold": result.threshold,
    }


@mcp.tool()
def backtest_model(
    values: list[float],
    n_folds: int = 4,
    horizon: int = 12,
) -> dict:
    """Backtest forecasting accuracy using walk-forward validation.

    Args:
        values: Historical time series values (needs enough data for folds).
        n_folds: Number of cross-validation folds.
        horizon: Forecast horizon per fold.

    Returns:
        Dictionary with per-fold MAPE scores and overall metrics.
    """
    import numpy as np
    from vectrix.business import Backtester
    from vectrix.engine.ets import AutoETS

    bt = Backtester(nFolds=n_folds, horizon=horizon, strategy="expanding", minTrainSize=60)
    y = np.array(values, dtype=np.float64)
    result = bt.run(y, modelFactory=AutoETS)

    folds = []
    for fold in result.folds:
        folds.append({
            "fold": fold.fold,
            "train_size": fold.trainSize,
            "test_size": fold.testSize,
            "mape": round(fold.mape, 4),
        })

    return {
        "folds": folds,
        "summary": bt.summary(result),
    }


@mcp.tool()
def list_sample_datasets() -> dict:
    """List all built-in sample datasets available for testing.

    Returns:
        Dictionary with dataset names, descriptions, and metadata.
    """
    from vectrix import listSamples

    samples = listSamples()
    return {
        "datasets": samples.to_dict(orient="records"),
        "count": len(samples),
    }


@mcp.tool()
def load_sample_dataset(name: str) -> dict:
    """Load a built-in sample dataset.

    Args:
        name: Dataset name — 'airline', 'retail', 'stock', 'temperature',
              'energy', 'web', or 'intermittent'.

    Returns:
        Dictionary with the dataset as records and metadata.
    """
    from vectrix import loadSample

    df = loadSample(name)
    return {
        "columns": list(df.columns),
        "rows": len(df),
        "head": df.head(10).to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


@mcp.resource("vectrix://models")
def list_available_models() -> str:
    """List all available forecasting models in Vectrix."""
    return """Available models:
- AutoETS: Exponential Smoothing (30 combinations)
- AutoARIMA: Seasonal ARIMA with stepwise selection
- OptimizedTheta: Theta method
- DynamicOptimizedTheta (DOT): M4-validated, OWA 0.905
- AutoCES: Complex Exponential Smoothing, OWA 0.927
- AutoMSTL: Multi-Seasonal STL decomposition
- AutoTBATS: Trigonometric seasonality
- GARCHModel / EGARCHModel / GJRGARCHModel: Volatility
- AutoCroston: Intermittent demand
- AdaptiveThetaEnsemble (4Theta): M4 3rd place method
- DynamicTimeScanForecaster (DTSF): Pattern matching
- EchoStateForecaster (ESN): Reservoir computing
- Baselines: Naive, SeasonalNaive, Mean, RandomWalkDrift, WindowAverage"""


@mcp.resource("vectrix://api-reference")
def api_reference() -> str:
    """Quick API reference for Vectrix."""
    return """Quick Reference:
from vectrix import forecast, analyze, regress, compare, quick_report

# Forecast
result = forecast(df, date="date", value="sales", steps=12)
result.predictions, result.model, result.mape, result.compare()

# Analyze
analysis = analyze(df, date="date", value="sales")
analysis.dna.difficulty, analysis.dna.recommendedModels

# Regress
reg = regress(data=df, formula="y ~ x1 + x2")
reg.r_squared, reg.coefficients, reg.diagnose()

# Compare all models
comparison = compare(df, date="date", value="sales", steps=12)

# Sample data
from vectrix import loadSample
df = loadSample("airline")
"""


@mcp.prompt()
def forecast_workflow(data_description: str = "monthly sales data") -> str:
    """Generate a step-by-step forecast workflow prompt."""
    return f"""Analyze and forecast the following {data_description} using Vectrix:

1. First, use analyze_timeseries to understand the data characteristics
2. Review the difficulty level and recommended models
3. Use compare_models to find the best model
4. Use forecast_timeseries for the final forecast
5. Check anomalies with detect_anomalies if needed
6. Report: model name, accuracy (MAPE), predictions with confidence intervals"""


@mcp.prompt()
def regression_workflow(target: str = "sales", features: str = "ads, price") -> str:
    """Generate a regression analysis workflow prompt."""
    return f"""Perform regression analysis to predict {target} using {features}:

1. Use run_regression with the data
2. Check R-squared and adjusted R-squared
3. Review p-values for statistical significance
4. Check diagnostics for model assumptions
5. Report: significant predictors, effect sizes, model fit quality"""


if __name__ == "__main__":
    mcp.run(transport="stdio")
