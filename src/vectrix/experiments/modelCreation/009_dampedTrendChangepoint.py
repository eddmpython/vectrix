"""
==============================================================================
실험 ID: modelCreation/009
실험명: Damped Trend with Changepoint — 변화점 가중 감쇠 추세 모델
==============================================================================

목적:
- 변화점(changepoint) 감지 후 최근 레짐에 가중치를 부여하는 감쇠 ETS 모델
- ensembleEvolution/003 Regime-Switching 실패 교훈: "절단" 아닌 "가중"으로 변화점 적응
- 과거 데이터를 버리지 않되, 최근 레짐에 더 높은 가중치 → 안전한 적응
- 감쇠 추세(damped trend): 장기 예측에서 추세 과대 외삽 방지

가설:
1. regimeShift 데이터에서 기존 모델 대비 20%+ 개선
2. 감쇠 추세가 trending 데이터에서 표준 추세보다 안정적
3. 전체 평균 순위 상위 50%
4. 변화점 가중이 균등 가중보다 우수

방법:
1. DampedTrendChangepointForecaster 클래스 구현
   - CUSUM 기반 변화점 감지
   - 변화점 이후 데이터에 지수 가중치 부여 (soft weighting)
   - Damped Trend ETS: level + damped_trend (phi 감쇠 계수)
   - alpha, beta, phi 최적화 (grid search)
2. AdaptiveDTCForecaster: 변화점 감도/감쇠율 자동 최적화
3. 합성 데이터 12종 벤치마크

성공 기준:
- regimeShift에서 1위
- 전체 평균 순위 상위 50%

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (11개 데이터셋):
   - 4theta: 3.09, mstl: 3.73, dot: 4.09, auto_ces: 4.27
   - dtc: 4.82 (5위) ***
   - arima: 5.27, theta: 5.36, dtc_adaptive: 5.36 ***

2. Head-to-head 승률: 0/11 = 0.0%
   - 모든 데이터셋에서 기존 모델에 패배

3. regimeShift 결과 (가설 1 검증):
   - dtc: 5.93% (3위) — 4theta 5.74% 대비 3.2% 열위
   - 변화점 5개 감지 [108, 160, 202, 247, 291] — 감지 자체는 작동
   - 그러나 가중 결합이 4theta의 holdout sMAPE 가중보다 열위
   → 가설 1 기각

4. trending 결과 (가설 2 검증):
   - dtc_adaptive: 1.02% (3위) — arima 0.87% 대비 16.6% 열위
   - 감쇠 추세는 안정적이나, arima/auto_ces가 이미 충분히 우수
   → 가설 2 기각

5. 변화점 감지 분석:
   - CUSUM이 과민 반응 — retailSales에서 7개, stockPrice에서 6개 감지
   - 잡음에 의한 허위 변화점이 가중치를 왜곡
   - sensitivity 파라미터 조정으로 개선 가능하나 근본적 한계

결론: 기각
- 0/11 승률 — 어떤 데이터에서도 기존 모델을 이기지 못함
- 평균 순위 4.82 (5위) — 상위 50% 기준 미달
- 변화점 감지 자체는 작동하나, 가중 ETS가 4theta/dot 수준에 미달
- ensembleEvolution/003 교훈("가중이 절단보다 나음")은 맞으나, 가중 방식 자체의 한계 확인

==============================================================================
실험일: 2026-02-28
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class DampedTrendChangepointForecaster:

    def __init__(self, phi=0.95, changepointSensitivity=1.5, decayRate=0.05):
        self._phi = phi
        self._sensitivity = changepointSensitivity
        self._decayRate = decayRate
        self._level = 0.0
        self._trend = 0.0
        self._alpha = 0.3
        self._beta = 0.1
        self._residStd = 0.0
        self._n = 0

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64).copy()
        self._n = len(y)

        changepoints = self._detectChangepoints(y)

        weights = self._computeWeights(y, changepoints)

        bestSSE = float('inf')
        bestParams = (0.3, 0.1, 0.95)

        for alpha in [0.1, 0.2, 0.3, 0.5, 0.7]:
            for beta in [0.01, 0.05, 0.1, 0.2]:
                for phi in [0.8, 0.9, 0.95, 0.98]:
                    sse = self._weightedSSE(y, weights, alpha, beta, phi)
                    if sse < bestSSE:
                        bestSSE = sse
                        bestParams = (alpha, beta, phi)

        self._alpha, self._beta, self._phi = bestParams

        self._level = y[0]
        self._trend = (y[min(3, len(y) - 1)] - y[0]) / max(min(3, len(y) - 1), 1)

        residuals = []
        for t in range(1, len(y)):
            forecast = self._level + self._phi * self._trend
            error = y[t] - forecast
            residuals.append(error)

            newLevel = self._alpha * y[t] + (1.0 - self._alpha) * (self._level + self._phi * self._trend)
            self._trend = self._beta * (newLevel - self._level) + (1.0 - self._beta) * self._phi * self._trend
            self._level = newLevel

        if residuals:
            self._residStd = max(np.std(residuals), 1e-8)
        else:
            self._residStd = max(np.std(y) * 0.1, 1e-8)

        return self

    def predict(self, steps):
        predictions = np.zeros(steps)
        level = self._level
        trend = self._trend

        for h in range(steps):
            dampedTrend = trend * self._phi ** (h + 1)
            cumulativeDamped = trend * self._phi * (1.0 - self._phi ** (h + 1)) / max(1.0 - self._phi, 1e-10)
            predictions[h] = level + cumulativeDamped

        sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - 1.96 * sigma
        upper = predictions + 1.96 * sigma

        return predictions, lower, upper

    def _detectChangepoints(self, y):
        n = len(y)
        if n < 20:
            return []

        diff = np.diff(y)
        meanDiff = np.mean(diff)
        stdDiff = max(np.std(diff), 1e-8)

        cusum = np.zeros(n - 1)
        cumPos = 0.0
        cumNeg = 0.0
        threshold = self._sensitivity * stdDiff

        changepoints = []
        for i in range(len(diff)):
            cumPos = max(0, cumPos + diff[i] - meanDiff - threshold / 2)
            cumNeg = min(0, cumNeg + diff[i] - meanDiff + threshold / 2)
            cusum[i] = cumPos - cumNeg

            if cumPos > threshold or abs(cumNeg) > threshold:
                changepoints.append(i + 1)
                cumPos = 0.0
                cumNeg = 0.0

        filtered = []
        minGap = max(n // 10, 5)
        for cp in changepoints:
            if not filtered or cp - filtered[-1] >= minGap:
                filtered.append(cp)

        return filtered

    def _computeWeights(self, y, changepoints):
        n = len(y)
        weights = np.ones(n)

        if not changepoints:
            return weights

        lastCp = changepoints[-1]
        for i in range(n):
            if i < lastCp:
                dist = lastCp - i
                weights[i] = np.exp(-self._decayRate * dist)
            else:
                weights[i] = 1.0

        weights /= weights.sum()
        weights *= n

        return weights

    def _weightedSSE(self, y, weights, alpha, beta, phi):
        n = len(y)
        level = y[0]
        trend = (y[min(3, n - 1)] - y[0]) / max(min(3, n - 1), 1)
        sse = 0.0

        for t in range(1, n):
            forecast = level + phi * trend
            error = y[t] - forecast
            sse += weights[t] * error * error

            newLevel = alpha * y[t] + (1.0 - alpha) * (level + phi * trend)
            trend = beta * (newLevel - level) + (1.0 - beta) * phi * trend
            level = newLevel

        return sse


class AdaptiveDTCForecaster:

    def __init__(self, holdoutRatio=0.15):
        self._holdoutRatio = holdoutRatio
        self._bestModel = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64).copy()
        n = len(y)

        holdoutSize = max(1, int(n * self._holdoutRatio))
        holdoutSize = min(holdoutSize, n // 3)
        trainPart = y[:n - holdoutSize]
        valPart = y[n - holdoutSize:]

        configs = [
            {'changepointSensitivity': 1.0, 'decayRate': 0.02},
            {'changepointSensitivity': 1.0, 'decayRate': 0.1},
            {'changepointSensitivity': 1.5, 'decayRate': 0.05},
            {'changepointSensitivity': 2.0, 'decayRate': 0.03},
            {'changepointSensitivity': 2.0, 'decayRate': 0.1},
            {'changepointSensitivity': 3.0, 'decayRate': 0.01},
        ]

        bestSmape = float('inf')
        bestConfig = configs[0]

        for cfg in configs:
            model = DampedTrendChangepointForecaster(
                changepointSensitivity=cfg['changepointSensitivity'],
                decayRate=cfg['decayRate'],
            )
            model.fit(trainPart)
            pred, _, _ = model.predict(holdoutSize)
            pred = np.asarray(pred, dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                continue
            smape = np.mean(
                2.0 * np.abs(valPart - pred) / (np.abs(valPart) + np.abs(pred) + 1e-10)
            )
            if smape < bestSmape:
                bestSmape = smape
                bestConfig = cfg

        self._bestModel = DampedTrendChangepointForecaster(
            changepointSensitivity=bestConfig['changepointSensitivity'],
            decayRate=bestConfig['decayRate'],
        )
        self._bestModel.fit(y)
        return self

    def predict(self, steps):
        return self._bestModel.predict(steps)


def _buildModels():
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.mstl import AutoMSTL
    from vectrix.engine.theta import OptimizedTheta
    from vectrix.experiments.modelCreation.e034_adaptiveThetaEnsemble import AdaptiveThetaEnsemble

    return {
        "mstl": lambda: AutoMSTL(),
        "arima": lambda: AutoARIMA(),
        "theta": lambda: OptimizedTheta(),
        "auto_ces": lambda: AutoCES(),
        "dot": lambda: DynamicOptimizedTheta(),
        "4theta": lambda: AdaptiveThetaEnsemble(),
    }


def _generateHourlyMultiSeasonal(n=720, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    base = 500.0
    hourly = 80.0 * np.sin(2.0 * np.pi * t / 24.0)
    daily = 40.0 * np.sin(2.0 * np.pi * t / (24.0 * 7.0))
    noise = rng.normal(0, 15, n)
    return base + hourly + daily + noise


def _runExperiment():
    from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS

    print("=" * 70)
    print("E039: Damped Trend with Changepoint")
    print("=" * 70)

    datasets = {}
    for name, genFunc in ALL_GENERATORS.items():
        if name == "intermittentDemand":
            continue
        if name == "multiSeasonalRetail":
            df = genFunc(n=730, seed=42)
        elif name == "stockPrice":
            df = genFunc(n=252, seed=42)
        else:
            df = genFunc(n=365, seed=42)
        datasets[name] = df["value"].values

    datasets["hourlyMultiSeasonal"] = _generateHourlyMultiSeasonal(720, 42)

    existingModels = _buildModels()
    newModels = {
        "dtc": lambda: DampedTrendChangepointForecaster(),
        "dtc_adaptive": lambda: AdaptiveDTCForecaster(),
    }
    allModels = {**existingModels, **newModels}

    horizon = 14
    results = {}

    for dsName, values in datasets.items():
        n = len(values)
        if n < horizon + 30:
            continue

        trainEnd = n - horizon
        train = values[:trainEnd]
        actual = values[trainEnd:trainEnd + horizon]

        results[dsName] = {}

        for modelName, modelFactory in allModels.items():
            try:
                model = modelFactory()
                model.fit(train)
                pred, _, _ = model.predict(horizon)
                pred = np.asarray(pred[:horizon], dtype=np.float64)
                if not np.all(np.isfinite(pred)):
                    results[dsName][modelName] = float('inf')
                    continue
                mape = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
                results[dsName][modelName] = mape
            except Exception:
                results[dsName][modelName] = float('inf')

    print("\n" + "=" * 70)
    print("ANALYSIS 1: Rankings per Dataset")
    print("=" * 70)

    rankAccum = {m: [] for m in allModels}

    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        sortedModels = sorted(mapes.items(), key=lambda x: x[1])

        print(f"\n  [{dsName}]")
        for rank, (mName, mape) in enumerate(sortedModels, 1):
            marker = " ***" if mName.startswith("dtc") else ""
            mapeStr = f"{mape:.2f}" if mape < 1e6 else f"{mape:.0f}"
            print(f"    {rank}. {mName:20s} MAPE={mapeStr:>14s}%{marker}")

        for rank, (mName, _) in enumerate(sortedModels, 1):
            rankAccum[mName].append(rank)

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Average Rank")
    print("=" * 70)

    avgRanks = []
    for mName, ranks in rankAccum.items():
        if ranks:
            avgRanks.append((mName, np.mean(ranks), len(ranks)))
    avgRanks.sort(key=lambda x: x[1])

    for mName, avgRank, count in avgRanks:
        marker = " ***" if mName.startswith("dtc") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: DTC vs Best Existing")
    print("=" * 70)

    dtcWins = 0
    existWins = 0

    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        newMapes = {k: v for k, v in mapes.items() if k.startswith("dtc")}
        existMapes = {k: v for k, v in mapes.items() if not k.startswith("dtc")}

        if not newMapes or not existMapes:
            continue

        bestNew = min(newMapes, key=newMapes.get)
        bestExist = min(existMapes, key=existMapes.get)
        nVal = newMapes[bestNew]
        eVal = existMapes[bestExist]

        winner = "DTC" if nVal <= eVal else "EXIST"
        if winner == "DTC":
            dtcWins += 1
        else:
            existWins += 1

        improvement = (eVal - nVal) / max(eVal, 1e-8) * 100
        print(f"  {dsName:25s} | {bestNew:18s} {nVal:10.2f}% vs {bestExist:10s} {eVal:10.2f}% | {winner} ({improvement:+.1f}%)")

    total = dtcWins + existWins
    if total > 0:
        print(f"\n  DTC wins: {dtcWins}/{total} ({dtcWins/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Changepoint Detection Analysis")
    print("=" * 70)

    for dsName in ["regimeShift", "trending", "retailSales", "stockPrice"]:
        if dsName not in datasets:
            continue
        values = datasets[dsName]
        train = values[:len(values) - horizon]
        m = DampedTrendChangepointForecaster()
        cps = m._detectChangepoints(train)
        print(f"  {dsName:25s} | changepoints at: {cps}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if total > 0:
        print(f"\n    DTC win rate: {dtcWins}/{total} ({dtcWins/total*100:.1f}%)")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
