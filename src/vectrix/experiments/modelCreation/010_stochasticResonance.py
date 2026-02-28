"""
==============================================================================
실험 ID: modelCreation/010
실험명: Stochastic Resonance Forecaster — 이중 우물 퍼텐셜 레짐 예측 모델
==============================================================================

목적:
- 물리학의 확률 공명(Stochastic Resonance) 원리를 예측에 적용
- 이중 우물 퍼텐셜: 시계열을 2개 안정 상태(레짐) 사이의 전환으로 모델링
- 노이즈가 신호를 증폭하는 SR 효과를 활용
- 레짐 전환 확률 추정 → 각 레짐에서 독립 예측 → 확률 가중 결합

가설:
1. regimeShift 데이터에서 레짐 전환 패턴 포착 → 상위 3위
2. bimodal 분포 데이터에서 기존 모델 대비 우위
3. 단일 레짐 데이터(stationary, trending)에서는 열위하나 안전하게 작동
4. 전체 평균 순위 상위 50%

방법:
1. StochasticResonanceForecaster 클래스 구현
   - K-means(k=2)로 2-레짐 분할
   - 각 레짐에서 이동평균 기반 국소 추세 추출
   - 전환 확률 행렬 추정 (Markov chain)
   - 예측: 현재 레짐에서 시작, 전환 확률로 미래 레짐 가중 → 레짐별 예측 결합
2. AdaptiveSRForecaster: 레짐 수(2~4) 자동 최적화
3. 합성 데이터 12종 벤치마크

성공 기준:
- regimeShift에서 상위 3위
- 전체 평균 순위 상위 50%

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (11개 데이터셋):
   - 4theta: 3.09, mstl: 3.18, dot: 4.18, auto_ces: 4.45
   - sr_adaptive: 5.18 (5위) ***
   - arima: 5.27, theta: 5.27, sr: 5.36 ***

2. Head-to-head 승률: 1/11 = 9.1%
   - 유일한 승리: hourlyMultiSeasonal(18.1% 개선)

3. regimeShift 결과 (가설 1 검증):
   - sr: 8.44% (5위) — 4theta 5.74% 대비 47% 열위
   - 전환 확률: p00=0.99, p01=0.01 — 전환 확률이 너무 낮아 레짐 전환 예측 불가
   - 2-레짐 분할은 작동하나(c0=89.7, c1=150.3), Markov chain이 현재 레짐에 고착
   → 가설 1 기각

4. 레짐 분석:
   - volatile: c0=99.5, c1=100.5 (0.5 차이) — 2레짐이 의미 없는 분할
   - stockPrice: 전환 확률 p01=0.03, p10=0.04 — 매우 낮은 전환
   - 근본 문제: K-means가 시간 순서 무시, 연속 레짐 감지가 아닌 값 분포 분할

결론: 기각
- 1/11 승률, 평균 순위 5.18~5.36 — 하위 50%
- regimeShift에서도 5위 — 핵심 가설 실패
- K-means 기반 레짐 분할이 시계열 구조 무시
- Markov chain 전환 확률이 너무 낮아 실질적으로 단일 레짐 모델

==============================================================================
실험일: 2026-02-28
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class StochasticResonanceForecaster:

    def __init__(self, nRegimes=2, smoothWindow=None):
        self._nRegimes = nRegimes
        self._smoothWindow = smoothWindow
        self._y = None
        self._n = 0
        self._centroids = None
        self._labels = None
        self._transMatrix = None
        self._regimeModels = {}
        self._currentRegime = 0
        self._residStd = 0.0

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        self._n = len(self._y)

        if self._smoothWindow is None:
            self._smoothWindow = max(5, self._n // 20)

        self._centroids, self._labels = self._kmeansRegimes(self._y, self._nRegimes)

        self._transMatrix = self._estimateTransitions(self._labels)

        self._regimeModels = {}
        for r in range(self._nRegimes):
            regimeMask = self._labels == r
            regimeValues = self._y[regimeMask]
            if len(regimeValues) >= 3:
                self._regimeModels[r] = {
                    'mean': np.mean(regimeValues),
                    'trend': self._estimateLocalTrend(self._y, regimeMask),
                    'std': max(np.std(regimeValues), 1e-8),
                    'lastValue': self._y[np.where(regimeMask)[0][-1]] if np.any(regimeMask) else np.mean(regimeValues),
                }
            else:
                self._regimeModels[r] = {
                    'mean': self._centroids[r],
                    'trend': 0.0,
                    'std': max(np.std(self._y) * 0.1, 1e-8),
                    'lastValue': self._centroids[r],
                }

        self._currentRegime = self._labels[-1]

        smoothed = np.convolve(self._y, np.ones(self._smoothWindow) / self._smoothWindow, mode='valid')
        if len(smoothed) > 1:
            fitted = np.interp(np.arange(self._n), np.linspace(0, self._n - 1, len(smoothed)), smoothed)
        else:
            fitted = np.full(self._n, np.mean(self._y))
        residuals = self._y - fitted
        self._residStd = max(np.std(residuals), 1e-8)

        return self

    def predict(self, steps):
        predictions = np.zeros(steps)
        regimeProbs = np.zeros(self._nRegimes)
        regimeProbs[self._currentRegime] = 1.0

        for h in range(steps):
            regimeProbs = regimeProbs @ self._transMatrix

            pred = 0.0
            for r in range(self._nRegimes):
                model = self._regimeModels[r]
                regimePred = model['lastValue'] + model['trend'] * (h + 1)
                pred += regimeProbs[r] * regimePred
            predictions[h] = pred

        sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
        regimeSpread = np.std([m['mean'] for m in self._regimeModels.values()])
        sigma = np.sqrt(sigma ** 2 + regimeSpread ** 2 * 0.1)

        lower = predictions - 1.96 * sigma
        upper = predictions + 1.96 * sigma

        return predictions, lower, upper

    def _kmeansRegimes(self, y, k, maxIter=50):
        n = len(y)
        percentiles = np.linspace(0, 100, k + 2)[1:-1]
        centroids = np.percentile(y, percentiles)

        labels = np.zeros(n, dtype=int)
        for _ in range(maxIter):
            for i in range(n):
                dists = np.abs(y[i] - centroids)
                labels[i] = np.argmin(dists)

            newCentroids = np.zeros(k)
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    newCentroids[j] = np.mean(y[mask])
                else:
                    newCentroids[j] = centroids[j]

            if np.allclose(centroids, newCentroids):
                break
            centroids = newCentroids

        return centroids, labels

    def _estimateTransitions(self, labels):
        k = self._nRegimes
        trans = np.ones((k, k)) * 0.01
        for i in range(len(labels) - 1):
            trans[labels[i], labels[i + 1]] += 1

        for i in range(k):
            rowSum = trans[i].sum()
            if rowSum > 0:
                trans[i] /= rowSum

        return trans

    def _estimateLocalTrend(self, y, mask):
        indices = np.where(mask)[0]
        if len(indices) < 3:
            return 0.0

        lastN = min(20, len(indices))
        recentIdx = indices[-lastN:]
        recentVals = y[recentIdx]

        x = np.arange(len(recentVals), dtype=np.float64)
        xMean = np.mean(x)
        yMean = np.mean(recentVals)
        num = np.sum((x - xMean) * (recentVals - yMean))
        den = np.sum((x - xMean) ** 2)
        if den < 1e-10:
            return 0.0
        return num / den


class AdaptiveSRForecaster:

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
            {'nRegimes': 2, 'smoothWindow': max(5, n // 20)},
            {'nRegimes': 2, 'smoothWindow': max(5, n // 10)},
            {'nRegimes': 3, 'smoothWindow': max(5, n // 20)},
            {'nRegimes': 3, 'smoothWindow': max(5, n // 10)},
        ]

        bestSmape = float('inf')
        bestConfig = configs[0]

        for cfg in configs:
            model = StochasticResonanceForecaster(
                nRegimes=cfg['nRegimes'],
                smoothWindow=cfg['smoothWindow'],
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

        self._bestModel = StochasticResonanceForecaster(
            nRegimes=bestConfig['nRegimes'],
            smoothWindow=bestConfig['smoothWindow'],
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
    print("E040: Stochastic Resonance Forecaster")
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
        "sr": lambda: StochasticResonanceForecaster(),
        "sr_adaptive": lambda: AdaptiveSRForecaster(),
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
            marker = " ***" if mName.startswith("sr") else ""
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
        marker = " ***" if mName.startswith("sr") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: SR vs Best Existing")
    print("=" * 70)

    srWins = 0
    existWins = 0

    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        newMapes = {k: v for k, v in mapes.items() if k.startswith("sr")}
        existMapes = {k: v for k, v in mapes.items() if not k.startswith("sr")}

        if not newMapes or not existMapes:
            continue

        bestNew = min(newMapes, key=newMapes.get)
        bestExist = min(existMapes, key=existMapes.get)
        nVal = newMapes[bestNew]
        eVal = existMapes[bestExist]

        winner = "SR" if nVal <= eVal else "EXIST"
        if winner == "SR":
            srWins += 1
        else:
            existWins += 1

        improvement = (eVal - nVal) / max(eVal, 1e-8) * 100
        print(f"  {dsName:25s} | {bestNew:18s} {nVal:10.2f}% vs {bestExist:10s} {eVal:10.2f}% | {winner} ({improvement:+.1f}%)")

    total = srWins + existWins
    if total > 0:
        print(f"\n  SR wins: {srWins}/{total} ({srWins/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Regime Analysis")
    print("=" * 70)

    for dsName in ["regimeShift", "trending", "volatile", "stockPrice"]:
        if dsName not in datasets:
            continue
        values = datasets[dsName]
        train = values[:len(values) - horizon]
        m = StochasticResonanceForecaster()
        m.fit(train)
        centStr = ", ".join([f"c{i}={c:.1f}" for i, c in enumerate(m._centroids)])
        transStr = ", ".join([f"p{i}{j}={m._transMatrix[i,j]:.2f}" for i in range(m._nRegimes) for j in range(m._nRegimes)])
        print(f"  {dsName:25s} | {centStr}")
        print(f"  {'':25s} | transitions: {transStr}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if total > 0:
        print(f"\n    SR win rate: {srWins}/{total} ({srWins/total*100:.1f}%)")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
