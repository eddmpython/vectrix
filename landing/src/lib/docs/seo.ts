export const SITE_URL = 'https://eddmpython.github.io/vectrix';
export const SITE_NAME = 'Vectrix';
export const DEFAULT_DESCRIPTION = 'Zero-config time series forecasting for Python. 30+ models, one line of code, Rust turbo acceleration.';
export const OG_IMAGE = `${SITE_URL}/icon-512.png`;

export const pageDescriptions: Record<string, string> = {
	'/docs': 'Vectrix documentation — guides, API reference, tutorials, and benchmarks for time series forecasting in Python.',
	'/docs/getting-started/installation': 'Install Vectrix with pip or uv. Supports Python 3.10+ on Windows, macOS, and Linux. Optional Rust turbo acceleration.',
	'/docs/getting-started/quickstart': 'Get started with Vectrix in under a minute. Forecast, analyze, and visualize time series data with a single function call.',
	'/docs/guide/forecasting': 'Complete guide to time series forecasting with Vectrix. Easy API, model selection, confidence intervals, and advanced options.',
	'/docs/guide/analysis': 'Automatic time series analysis with DNA profiling. Detect patterns, seasonality, trends, anomalies, and data characteristics.',
	'/docs/guide/regression': 'Time series regression analysis with Vectrix. OLS, diagnostics, robust regression, and R-style formula support.',
	'/docs/guide/adaptive': 'Adaptive forecasting with regime detection, self-healing, constraint enforcement, and DNA-based model selection.',
	'/docs/guide/business': 'Business intelligence tools: anomaly detection, scenario analysis, backtesting, and forecast explanation.',
	'/docs/guide/pipeline': 'Build forecasting pipelines combining preprocessing, model fitting, and post-processing steps.',
	'/docs/guide/foundation': 'Use foundation models (TimesFM, Chronos, Moirai) with Vectrix for zero-shot time series forecasting.',
	'/docs/guide/multivariate': 'Multivariate time series forecasting with ARIMAX, VAR, and exogenous variable support.',
	'/docs/api/easy': 'Easy API reference — forecast(), analyze(), regress(), quick_report(). One function call for each task.',
	'/docs/api/vectrix': 'Vectrix class API reference. Full control over model fitting, forecasting, and configuration.',
	'/docs/api/adaptive': 'Adaptive forecasting API — RegimeDetector, SelfHealing, ConstraintEnforcer, and DNA profiling.',
	'/docs/api/business': 'Business intelligence API — AnomalyDetector, ScenarioAnalyzer, Backtester, ForecastExplainer.',
	'/docs/api/regression': 'Regression API — OLS, diagnostics, robust regression, time series regression, and formula interface.',
	'/docs/api/pipeline': 'Pipeline API — build and compose forecasting workflows with preprocessing and post-processing.',
	'/docs/api/foundation': 'Foundation model API — integrate TimesFM, Chronos, and Moirai models with Vectrix.',
	'/docs/api/types': 'Type definitions for Vectrix — ForecastResult, AnalysisResult, ModelInfo, and configuration types.',
	'/docs/tutorials/quickstart': 'Step-by-step tutorial: your first forecast with Vectrix. Learn forecast(), analyze(), and result visualization.',
	'/docs/tutorials/analyze': 'Tutorial: deep dive into time series analysis. DNA profiling, pattern detection, and automated insights.',
	'/docs/tutorials/regression': 'Tutorial: regression analysis with Vectrix. Build models, run diagnostics, and interpret results.',
	'/docs/tutorials/models': 'Tutorial: compare 30+ forecasting models. ETS, ARIMA, Theta, DOT, CES, and custom model selection.',
	'/docs/tutorials/adaptive': 'Tutorial: adaptive forecasting with regime detection, self-healing data, and constraint enforcement.',
	'/docs/tutorials/business': 'Tutorial: business intelligence with anomaly detection, scenario analysis, and backtesting.',
	'/docs/showcase/korean-economy': 'Showcase: forecasting Korean economic indicators (GDP, CPI, trade) with Vectrix.',
	'/docs/showcase/korean-regression': 'Showcase: regression analysis on Korean economic data with R-style formulas.',
	'/docs/showcase/model-comparison': 'Showcase: comparing 30+ forecasting models on real-world datasets. Side-by-side accuracy benchmarks.',
	'/docs/showcase/business-intelligence': 'Showcase: business intelligence pipeline with anomaly detection, scenarios, and automated reporting.',
	'/docs/benchmarks': 'M3 & M4 Competition benchmark results. Vectrix OWA scores vs Naive2 baseline across all frequencies.',
	'/docs/changelog': 'Vectrix changelog — version history, new features, bug fixes, and breaking changes.'
};

export function getDescription(path: string): string {
	const normalized = path.replace(/\/$/, '');
	return pageDescriptions[normalized] || DEFAULT_DESCRIPTION;
}

export function getCanonicalUrl(path: string): string {
	const normalized = path.replace(/\/$/, '');
	return `${SITE_URL}${normalized}`;
}
