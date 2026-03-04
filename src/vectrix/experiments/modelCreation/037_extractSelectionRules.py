"""
==============================================================================
Experiment ID: modelCreation/037
Experiment: Extract Model Selection Rules from Oracle Data
==============================================================================

Purpose:
- E034 showed GBR meta_top1 = 0.873 (DOT 0.885 baseline)
- scikit-learn runtime dependency is undesirable
- Extract interpretable rules from oracle data → embed in _selectNativeModels()
- Also analyze: which features matter most for model selection
- Target: reproduce E034's 0.873 with pure numpy rules

Hypothesis:
1. A small set of rules (10-20) using top features can match GBR performance
2. Frequency + period + 3-5 key DNA features should suffice
3. Rules can be encoded as simple if/elif chains in numpy

Method:
1. Load E031 oracle data (8 models x 7273 series)
2. For each series: identify best safe model (oracle)
3. Train decision tree (max_depth=5) to learn oracle selection
4. Extract tree rules → convert to numpy if/elif
5. Validate: rule-based OWA vs GBR OWA vs DOT-only OWA
6. Also: analyze per-frequency best-model distributions

Results (to be filled after experiment):

Conclusion:

Experiment date: 2026-03-04
==============================================================================
"""

import os
import sys
import pickle
import warnings

warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

M4_GROUPS = {
    'Yearly':    {'horizon': 6,  'seasonality': 1},
    'Quarterly': {'horizon': 8,  'seasonality': 4},
    'Monthly':   {'horizon': 18, 'seasonality': 12},
    'Weekly':    {'horizon': 13, 'seasonality': 1},
    'Daily':     {'horizon': 14, 'seasonality': 1},
    'Hourly':    {'horizon': 48, 'seasonality': 24},
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), '_cache')
ALL_MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'auto_arima',
              'dtsf', 'esn', 'theta']
SAFE_MODELS = ['dot', 'auto_ces', 'four_theta', 'auto_ets', 'theta']

P = lambda *a, **kw: print(*a, **kw, flush=True)


def _smape(actual, predicted):
    return np.mean(2.0 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-10)) * 100


def _mase(trainY, actual, predicted, seasonality):
    n = len(trainY)
    m = max(seasonality, 1)
    if n <= m:
        naiveErrors = np.abs(np.diff(trainY))
    else:
        naiveErrors = np.abs(trainY[m:] - trainY[:-m])
    masep = np.mean(naiveErrors) if len(naiveErrors) > 0 else 1e-10
    if masep < 1e-10:
        masep = 1e-10
    return np.mean(np.abs(actual - predicted)) / masep


def _loadAllOracle():
    allOracle = {}
    for group in M4_GROUPS:
        cachePath = os.path.join(CACHE_DIR, f'oracle_{group}.pkl')
        with open(cachePath, 'rb') as f:
            allOracle[group] = pickle.load(f)
        P(f"  Loaded {group}: {len(allOracle[group])} series")
    return allOracle


def _analyzeOracleDistribution(allOracle):
    P(f"\n{'='*70}")
    P(f"  Phase 1: Oracle Best Model Distribution (Safe Pool)")
    P(f"{'='*70}")

    allBestModels = {g: [] for g in M4_GROUPS}
    allFeatures = []
    allBestIdx = []
    allGroupLabels = []

    groupEncodings = {g: i for i, g in enumerate(M4_GROUPS)}

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]
        counts = {m: 0 for m in SAFE_MODELS}

        for entry in oracleData:
            owas = entry['modelOwas']
            safeOwas = {m: owas.get(m, 5.0) for m in SAFE_MODELS}
            bestModel = min(safeOwas, key=safeOwas.get)
            counts[bestModel] += 1
            allBestModels[groupName].append(bestModel)

            features = entry['features']
            fNames = sorted([k for k in features.keys() if not k.startswith('_')])
            fVec = [features.get(fn, 0.0) for fn in fNames]
            fVec.append(features.get('_length', 0))
            fVec.append(features.get('_period', 0))
            fVec.append(features.get('_horizon', 0))
            fVec.append(groupEncodings[groupName])
            fVec = np.array(fVec, dtype=np.float64)
            fVec = np.where(np.isfinite(fVec), fVec, 0.0)

            allFeatures.append(fVec)
            allBestIdx.append(SAFE_MODELS.index(bestModel))
            allGroupLabels.append(groupName)

        total = len(oracleData)
        P(f"\n  {groupName} (n={total}):")
        for m in SAFE_MODELS:
            pct = counts[m] / total * 100
            bar = '#' * int(pct / 2)
            P(f"    {m:<15} {counts[m]:>5} ({pct:5.1f}%) {bar}")

    X = np.array(allFeatures)
    y = np.array(allBestIdx)

    firstEntry = list(allOracle.values())[0][0]
    fNames = sorted([k for k in firstEntry['features'].keys() if not k.startswith('_')])
    fNames += ['_length', '_period', '_horizon', '_groupIdx']

    return X, y, fNames, allGroupLabels, allBestModels


def _trainDecisionTree(X, y, fNames):
    P(f"\n{'='*70}")
    P(f"  Phase 2: Decision Tree Rule Extraction")
    P(f"{'='*70}")

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score

    depths = [3, 4, 5, 6, 7, 8, 10]
    bestDepth = 3
    bestScore = 0

    for depth in depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_leaf=20)
        scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
        meanScore = scores.mean()
        P(f"  depth={depth:>2}: accuracy={meanScore:.4f} (+/- {scores.std():.4f})")
        if meanScore > bestScore:
            bestScore = meanScore
            bestDepth = depth

    P(f"\n  Best depth: {bestDepth} (accuracy={bestScore:.4f})")

    dt = DecisionTreeClassifier(max_depth=bestDepth, random_state=42, min_samples_leaf=20)
    dt.fit(X, y)

    P(f"\n  Feature importances (top 15):")
    importances = dt.feature_importances_
    topIdx = np.argsort(importances)[::-1][:15]
    for i in topIdx:
        if importances[i] > 0.001:
            P(f"    {fNames[i]:<30} {importances[i]:.4f}")

    rules = export_text(dt, feature_names=fNames, max_depth=bestDepth)
    P(f"\n  Decision Tree Rules:")
    for line in rules.split('\n')[:60]:
        P(f"    {line}")
    if len(rules.split('\n')) > 60:
        P(f"    ... ({len(rules.split(chr(10)))} total lines)")

    return dt, bestDepth


def _evaluateStrategies(allOracle, X, y, dt, fNames, allGroupLabels):
    P(f"\n{'='*70}")
    P(f"  Phase 3: Evaluate Strategies")
    P(f"{'='*70}")

    dtPreds = dt.predict(X)

    strategies = ['oracle', 'dot_only', 'dt_pred', 'simple_rules']
    offset = 0
    groupResults = {}

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]
        horizon = info['horizon']
        seasonality = info['seasonality']
        nSeries = len(oracleData)

        stratSmapes = {s: [] for s in strategies}
        stratMases = {s: [] for s in strategies}

        for si, entry in enumerate(oracleData):
            gi = offset + si
            testY = entry['testY']
            trainY = entry['trainY']
            preds = entry['modelPreds']
            owas = entry['modelOwas']

            safePreds = {m: preds.get(m, np.full(horizon, np.mean(trainY))) for m in SAFE_MODELS}

            safeOwas = {m: owas.get(m, 5.0) for m in SAFE_MODELS}
            bestOracleModel = min(safeOwas, key=safeOwas.get)
            oraclePred = safePreds[bestOracleModel]
            stratSmapes['oracle'].append(_smape(testY, oraclePred))
            stratMases['oracle'].append(_mase(trainY, testY, oraclePred, seasonality))

            dotPred = safePreds['dot']
            stratSmapes['dot_only'].append(_smape(testY, dotPred))
            stratMases['dot_only'].append(_mase(trainY, testY, dotPred, seasonality))

            dtModelIdx = dtPreds[gi]
            dtModel = SAFE_MODELS[dtModelIdx]
            dtPred = safePreds[dtModel]
            stratSmapes['dt_pred'].append(_smape(testY, dtPred))
            stratMases['dt_pred'].append(_mase(trainY, testY, dtPred, seasonality))

            features = entry['features']
            simpleModel = _simpleRuleSelect(features, groupName)
            simplePred = safePreds[simpleModel]
            stratSmapes['simple_rules'].append(_smape(testY, simplePred))
            stratMases['simple_rules'].append(_mase(trainY, testY, simplePred, seasonality))

        offset += nSeries

        n2SmapeAvg = np.mean([e['n2Smape'] for e in oracleData])
        n2MaseAvg = np.mean([e['n2Mase'] for e in oracleData])

        groupOwa = {}
        P(f"\n  {groupName} (n={nSeries}):")
        for sName in strategies:
            sSmape = np.mean(stratSmapes[sName])
            sMase = np.mean(stratMases[sName])
            sOwa = 0.5 * (sSmape / n2SmapeAvg + sMase / n2MaseAvg)
            groupOwa[sName] = sOwa
            dotOwa = groupOwa.get('dot_only', 999)
            diff = sOwa - dotOwa
            marker = ' ***' if sName != 'oracle' and sName != 'dot_only' and diff < -0.001 else ''
            P(f"    {sName:<20} OWA={sOwa:.4f} ({diff:+.4f}){marker}")

        groupResults[groupName] = groupOwa

    P(f"\n{'='*70}")
    P(f"  OVERALL AVG OWA")
    P(f"{'='*70}")

    dotAvg = np.mean([groupResults[g]['dot_only'] for g in M4_GROUPS])
    for sName in strategies:
        vals = [groupResults[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if sName not in ('oracle', 'dot_only') and avg < dotAvg else ''
        P(f"  {sName:<20} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    return groupResults


def _simpleRuleSelect(features, groupName):
    length = features.get('_length', 100)
    period = features.get('_period', 1)
    horizon = features.get('_horizon', 6)

    trendStr = features.get('trendStrength', 0)
    seasStr = features.get('seasonalStrength', 0)
    acf1 = features.get('acf1', 0)
    cv = features.get('cv', 0)
    hurst = features.get('hurstExponent', 0.5)
    spectralEntropy = features.get('spectralEntropy', 0.5)
    volatilityClustering = features.get('volatilityClustering', 0)
    nonlinearAutocorr = features.get('nonlinearAutocorr', 0)
    trendSlope = features.get('trendSlope', 0)
    levelShiftCount = features.get('levelShiftCount', 0)

    if groupName == 'Yearly':
        return 'dot'

    if groupName == 'Hourly':
        if seasStr > 0.3:
            return 'auto_ces'
        return 'four_theta'

    if period >= 12 and seasStr > 0.5:
        return 'auto_ces'

    if trendStr > 0.7 and seasStr < 0.2:
        return 'dot'

    if cv > 1.0 and volatilityClustering > 0.3:
        return 'four_theta'

    if hurst > 0.7 and acf1 > 0.5:
        return 'dot'

    if spectralEntropy < 0.3:
        return 'auto_ces'

    if length > 200 and period >= 4:
        return 'auto_ces'

    return 'dot'


def _analyzePerGroupPatterns(allOracle):
    P(f"\n{'='*70}")
    P(f"  Phase 4: Per-Group Feature Analysis for Best Model")
    P(f"{'='*70}")

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]

        modelFeatures = {m: [] for m in SAFE_MODELS}

        for entry in oracleData:
            owas = entry['modelOwas']
            safeOwas = {m: owas.get(m, 5.0) for m in SAFE_MODELS}
            bestModel = min(safeOwas, key=safeOwas.get)
            features = entry['features']

            keyFeatures = {
                'trendStr': features.get('trendStrength', 0),
                'seasStr': features.get('seasonalStrength', 0),
                'acf1': features.get('acf1', 0),
                'cv': features.get('cv', 0),
                'hurst': features.get('hurstExponent', 0.5),
                'specEnt': features.get('spectralEntropy', 0.5),
                'volClust': features.get('volatilityClustering', 0),
                'length': features.get('_length', 100),
            }
            modelFeatures[bestModel].append(keyFeatures)

        P(f"\n  {groupName}:")
        for mName in SAFE_MODELS:
            if not modelFeatures[mName]:
                continue
            n = len(modelFeatures[mName])
            P(f"    {mName} (n={n}):")
            for fKey in ['trendStr', 'seasStr', 'acf1', 'cv', 'hurst', 'specEnt', 'volClust', 'length']:
                vals = [f[fKey] for f in modelFeatures[mName]]
                P(f"      {fKey:<12} mean={np.mean(vals):.3f} std={np.std(vals):.3f} "
                  f"[{np.percentile(vals, 25):.3f}, {np.percentile(vals, 75):.3f}]")


def _iterateRules(allOracle, allGroupLabels, iteration=0):
    P(f"\n{'='*70}")
    P(f"  Phase 5: Iterated Rule Refinement (Iteration {iteration})")
    P(f"{'='*70}")

    strategies = ['oracle', 'dot_only', 'refined_rules']
    groupResults = {}

    for groupName, oracleData in allOracle.items():
        info = M4_GROUPS[groupName]
        horizon = info['horizon']
        seasonality = info['seasonality']
        nSeries = len(oracleData)

        stratSmapes = {s: [] for s in strategies}
        stratMases = {s: [] for s in strategies}

        for entry in oracleData:
            testY = entry['testY']
            trainY = entry['trainY']
            preds = entry['modelPreds']
            owas = entry['modelOwas']
            features = entry['features']

            safePreds = {m: preds.get(m, np.full(horizon, np.mean(trainY))) for m in SAFE_MODELS}

            safeOwas = {m: owas.get(m, 5.0) for m in SAFE_MODELS}
            bestOracleModel = min(safeOwas, key=safeOwas.get)
            stratSmapes['oracle'].append(_smape(testY, safePreds[bestOracleModel]))
            stratMases['oracle'].append(_mase(trainY, testY, safePreds[bestOracleModel], seasonality))

            stratSmapes['dot_only'].append(_smape(testY, safePreds['dot']))
            stratMases['dot_only'].append(_mase(trainY, testY, safePreds['dot'], seasonality))

            selectedModel = _refinedRuleSelect(features, groupName)
            stratSmapes['refined_rules'].append(_smape(testY, safePreds[selectedModel]))
            stratMases['refined_rules'].append(_mase(trainY, testY, safePreds[selectedModel], seasonality))

        n2SmapeAvg = np.mean([e['n2Smape'] for e in oracleData])
        n2MaseAvg = np.mean([e['n2Mase'] for e in oracleData])

        groupOwa = {}
        P(f"\n  {groupName} (n={nSeries}):")
        for sName in strategies:
            sSmape = np.mean(stratSmapes[sName])
            sMase = np.mean(stratMases[sName])
            sOwa = 0.5 * (sSmape / n2SmapeAvg + sMase / n2MaseAvg)
            groupOwa[sName] = sOwa
            dotOwa = groupOwa.get('dot_only', 999)
            diff = sOwa - dotOwa
            marker = ' ***' if sName == 'refined_rules' and diff < -0.001 else ''
            P(f"    {sName:<20} OWA={sOwa:.4f} ({diff:+.4f}){marker}")

        groupResults[groupName] = groupOwa

    P(f"\n{'='*70}")
    P(f"  REFINED RULES OVERALL")
    P(f"{'='*70}")

    dotAvg = np.mean([groupResults[g]['dot_only'] for g in M4_GROUPS])
    for sName in strategies:
        vals = [groupResults[g][sName] for g in M4_GROUPS]
        avg = np.mean(vals)
        diff = avg - dotAvg
        detail = ' '.join([f"{list(M4_GROUPS.keys())[i][:2]}={vals[i]:.3f}" for i in range(len(vals))])
        marker = ' *** BETTER' if sName == 'refined_rules' and avg < dotAvg else ''
        P(f"  {sName:<20} AVG={avg:.4f} ({diff:+.4f}){marker}  [{detail}]")

    return groupResults


def _refinedRuleSelect(features, groupName):
    length = features.get('_length', 100)
    period = features.get('_period', 1)

    trendStr = features.get('trendStrength', 0)
    seasStr = features.get('seasonalStrength', 0)
    acf1 = features.get('acf1', 0)
    cv = features.get('cv', 0)
    hurst = features.get('hurstExponent', 0.5)
    specEnt = features.get('spectralEntropy', 0.5)
    volClust = features.get('volatilityClustering', 0)
    nonlinearAc = features.get('nonlinearAutocorr', 0)
    trendSlope = features.get('trendSlope', 0)
    levelShift = features.get('levelShiftCount', 0)
    flatSpot = features.get('flatSpotRate', 0)
    turningPt = features.get('turningPointRate', 0)
    demandDens = features.get('demandDensity', 1.0)
    seasPeak = features.get('seasonalPeakPeriod', 0)

    if groupName == 'Yearly':
        return 'dot'

    if groupName == 'Hourly':
        if seasStr > 0.4:
            return 'auto_ces'
        if trendStr > 0.5:
            return 'four_theta'
        return 'auto_ces'

    if groupName == 'Quarterly':
        if trendStr > 0.6 and seasStr < 0.3:
            return 'dot'
        if seasStr > 0.5:
            return 'auto_ces'
        if cv < 0.3:
            return 'auto_ets'
        return 'dot'

    if groupName == 'Monthly':
        if seasStr > 0.6:
            return 'auto_ces'
        if trendStr > 0.7:
            return 'dot'
        if specEnt < 0.3 and seasStr > 0.3:
            return 'auto_ces'
        if hurst > 0.7:
            return 'dot'
        return 'four_theta'

    if groupName == 'Weekly':
        if trendStr > 0.6:
            return 'dot'
        if seasStr > 0.3:
            return 'auto_ces'
        return 'dot'

    if groupName == 'Daily':
        if trendStr > 0.5:
            return 'dot'
        if seasStr > 0.3:
            return 'auto_ces'
        if volClust > 0.3:
            return 'four_theta'
        return 'dot'

    return 'dot'


if __name__ == '__main__':
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    P("=" * 70)
    P("E037: Extract Model Selection Rules from Oracle Data")
    P("=" * 70)

    allOracle = _loadAllOracle()

    X, y, fNames, allGroupLabels, allBestModels = _analyzeOracleDistribution(allOracle)

    dt, bestDepth = _trainDecisionTree(X, y, fNames)

    groupResults = _evaluateStrategies(allOracle, X, y, dt, fNames, allGroupLabels)

    _analyzePerGroupPatterns(allOracle)

    refinedResults = _iterateRules(allOracle, allGroupLabels)

    P(f"\n{'='*70}")
    P(f"  SUMMARY")
    P(f"{'='*70}")
    P(f"  DOT baseline:   {np.mean([groupResults[g]['dot_only'] for g in M4_GROUPS]):.4f}")
    P(f"  Oracle ceiling:  {np.mean([groupResults[g]['oracle'] for g in M4_GROUPS]):.4f}")
    P(f"  DT prediction:  {np.mean([groupResults[g]['dt_pred'] for g in M4_GROUPS]):.4f}")
    P(f"  Simple rules:   {np.mean([groupResults[g]['simple_rules'] for g in M4_GROUPS]):.4f}")
    P(f"  Refined rules:  {np.mean([refinedResults[g]['refined_rules'] for g in M4_GROUPS]):.4f}")
    P(f"  E034 GBR meta:  0.873 (reference)")
    P("=" * 70)
