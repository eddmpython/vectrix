---
name: vectrix-analyze
description: Run Vectrix time series analysis and DNA profiling. Use when the user asks to analyze patterns, detect anomalies, profile a time series, or understand data characteristics.
allowed-tools: Bash(uv *), Bash(python *), Read, Glob, Grep
---

# Vectrix Analysis & DNA Profiling

When the user wants to analyze time series data:

## DNA Analysis

```python
from vectrix import analyze

result = analyze(df, date="date", value="sales")
print(result.summary())
print(f"Difficulty: {result.dna.difficulty}")
print(f"Recommended: {result.dna.recommendedModels}")
```

### DNAProfile attributes
- `result.dna.difficulty` — "easy", "medium", "hard", "extreme"
- `result.dna.category` — data category string
- `result.dna.recommendedModels` — list of model names
- `result.dna.features` — dict of 65+ features
- `result.dna.fingerprint` — np.ndarray feature vector

### Key features (65+)
trendStrength, seasonalStrength, nonlinearAutocorr, volatilityClustering, hurstExponent, seasonalPeakPeriod, demandDensity, etc.

## Anomaly Detection

```python
from vectrix.business import AnomalyDetector

detector = AnomalyDetector()
result = detector.detect(y, method='auto', threshold=3.0)
# result.indices — anomaly positions
# result.scores — anomaly scores
# result.nAnomalies — count
```

Methods: 'zscore', 'iqr', 'seasonal', 'rolling', 'auto'

**IMPORTANT**: The parameter is `threshold=` NOT `sensitivity=`

## Regime Detection

```python
from vectrix.adaptive.regime import RegimeDetector

detector = RegimeDetector()
result = detector.detect(y)
print(f"Regimes: {len(result.regimeStats)}")
print(f"Current: {result.currentRegime}")
```

## Changepoint Detection

```python
from vectrix.engine.changepoint import ChangePointDetector

detector = ChangePointDetector()
result = detector.detect(y)
```

## Feature Extraction

```python
from vectrix.engine.features import TSFeatureExtractor

extractor = TSFeatureExtractor()
features = extractor.extract(y, period=12)  # dict of 40+ features
```

## Common Gotchas

- `analyze()` requires DataFrame + dateCol + valueCol (NOT raw ndarray)
- Access difficulty via `result.dna.difficulty` not `result.difficulty`
- `listSamples()` returns DataFrame not dict
- `loadSample()` value column name varies per dataset
