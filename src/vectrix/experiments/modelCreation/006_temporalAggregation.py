"""
==============================================================================
실험 ID: modelCreation/006
실험명: Temporal Aggregation Forecaster — MAPA 다중 시간 집계 예측 모델
==============================================================================

목적:
- Multiple Aggregation Prediction Algorithm (MAPA)을 numpy로 재구현
- 시계열을 여러 집계 수준(1x, 2x, 3x, ... kx)으로 변환 후 각각 독립 예측
- 집계 수준별 ETS 분해 → 성분(level, trend, seasonal) 결합 → 최종 예측
- M3/M4 Competition에서 검증된 방법론
- 기존 모델과 근본적으로 다른 "시간 해상도" 관점

가설:
1. 계절성 데이터(retailSales, energyUsage)에서 평균 순위 상위 3위
2. 노이즈가 큰 데이터(volatile, stockPrice)에서 고집계 수준이 노이즈 필터링 효과
3. 전체 평균 순위 상위 50% (4.5 이내)
4. 기존 모델과 잔차 상관 0.6 이하 (다른 시간 해상도 관점)

방법:
1. TemporalAggregationForecaster 클래스 구현
   - 집계 수준 k=1,2,3,4,6,8,12 (원본 주기의 약수 + 범용 수준)
   - 각 수준에서 non-overlapping 평균으로 집계
   - 집계된 시계열에 ETS(A,N,N) 또는 Theta 적합
   - 성분별 결합: level은 중앙값, trend는 고집계 우선, seasonal은 저집계 우선
2. AdaptiveMAPAForecaster: holdout 기반 집계 수준 가중치 최적화
3. 합성 데이터 12종 벤치마크
4. 기존 6개 모델(mstl, arima, theta, ces, dot, 4theta) 대비 비교

성공 기준:
- 전체 평균 순위 상위 50% (4.5 이내)
- volatile/stockPrice에서 고집계 효과 확인

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (11개 데이터셋):
   - 4theta: 3.18 (1위)
   - mstl: 3.64 (2위)
   - dot: 4.09 (3위)
   - mapa: 4.27 (4위) ***
   - auto_ces: 4.45, arima: 5.27, theta: 5.55, mapa_adaptive: 5.55

2. Head-to-head 승률: 1/11 = 9.1%
   - 승리: hourlyMultiSeasonal만 (80.6% 개선!)
   - 패배: 나머지 10개 데이터셋

3. 핵심 발견 — hourlyMultiSeasonal:
   - mapa 2.26% (1위!) — 기존 최고 theta 11.69% 대비 80.6% 개선
   - 24시간 주기를 자동 감지, 집계 수준 [1,2,3,4,6,8,12,24]로 분해
   - 저수준(1x)은 계절성 포착, 고수준(24x)은 일간 추세 포착 → 결합 효과

4. 잔차 상관 분석 (가설 4 검증):
   - hourlyMultiSeasonal: 모든 기존 모델과 0.01~0.05 (거의 무상관!)
   - retailSales: mstl:-0.06, arima:0.15 / theta~dot: 1.00
   - energyUsage: mstl:-0.04, arima:0.17 / theta~4theta: 1.00
   → 부분 채택: hourly 데이터에서 잔차 직교성 우수, daily에서는 기존과 유사

5. 약점 분석:
   - retailSales: period=1로 감지 (실제 7) → 계절성 미활용
   - stockPrice: 10.61% (6위) — 비정상 데이터에 약함
   - mapa_adaptive가 mapa보다 대부분 열위 — 가중 결합이 오히려 악화
   - periodogram이 daily 데이터에서 주기 감지 실패 (001과 동일 문제)

6. 집계 수준 분석:
   - retailSales: period=1, 주기 미감지 → 집계만으로는 계절성 포착 불가
   - hourlyMultiSeasonal: period=24, 완벽 감지 → MAPA의 강점 발휘
   - volatile: period=1, 집계로 노이즈 감소 효과 미미 (0.41% vs dot 0.39%)

결론: 조건부 채택 — hourly/고빈도 전용
- 평균 순위 4.27 (4위) — 상위 50% 기준 충족
- hourlyMultiSeasonal에서 80.6% 개선, 잔차 상관 ~0 → 앙상블 핵심 후보
- 주기 감지 실패 시 효과 미미 → 주기를 명시적으로 지정하면 개선 가능
- mapa_adaptive 방식은 실패 — 단순 mapa가 더 우수

==============================================================================
실험일: 2026-02-28
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def _sesOptimize(y):
    n = len(y)
    if n < 3:
        return 0.3, y[-1]

    def _sse(alpha):
        level = y[0]
        sse = 0.0
        for t in range(1, n):
            error = y[t] - level
            sse += error * error
            level = alpha * y[t] + (1.0 - alpha) * level
        return sse

    result = minimize_scalar(_sse, bounds=(0.001, 0.999), method='bounded')
    alpha = result.x if result.success else 0.3

    level = y[0]
    for t in range(1, n):
        level = alpha * y[t] + (1.0 - alpha) * level
    return alpha, level


def _desOptimize(y):
    n = len(y)
    if n < 5:
        alpha, level = _sesOptimize(y)
        return alpha, 0.01, level, 0.0

    def _sse(params):
        alpha, beta = params[0], params[1]
        level = y[0]
        trend = (y[min(3, n - 1)] - y[0]) / min(3, n - 1)
        sse = 0.0
        for t in range(1, n):
            forecast = level + trend
            error = y[t] - forecast
            sse += error * error
            newLevel = alpha * y[t] + (1.0 - alpha) * (level + trend)
            trend = beta * (newLevel - level) + (1.0 - beta) * trend
            level = newLevel
        return sse

    bestSSE = float('inf')
    bestParams = (0.3, 0.01)
    for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for b in [0.01, 0.05, 0.1, 0.3]:
            sse = _sse([a, b])
            if sse < bestSSE:
                bestSSE = sse
                bestParams = (a, b)

    alpha, beta = bestParams
    level = y[0]
    trend = (y[min(3, n - 1)] - y[0]) / min(3, n - 1)
    for t in range(1, n):
        newLevel = alpha * y[t] + (1.0 - alpha) * (level + trend)
        trend = beta * (newLevel - level) + (1.0 - beta) * trend
        level = newLevel

    return alpha, beta, level, trend


class TemporalAggregationForecaster:

    def __init__(self, aggLevels=None, period=None):
        self._aggLevels = aggLevels
        self._period = period
        self._y = None
        self._n = 0
        self._components = {}
        self._combinedLevel = 0.0
        self._combinedTrend = 0.0
        self._seasonal = None
        self._residStd = 0.0

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        self._n = len(self._y)

        if self._period is None:
            self._period = self._detectPeriod(self._y)

        if self._aggLevels is None:
            self._aggLevels = self._defaultAggLevels(self._n, self._period)

        levels = []
        trends = []
        seasonals = {}

        for k in self._aggLevels:
            aggSeries = self._aggregate(self._y, k)
            if len(aggSeries) < 5:
                continue

            aggPeriod = max(1, self._period // k)

            if aggPeriod > 1 and len(aggSeries) >= aggPeriod * 3:
                deseason, seasonal = self._removeSeasonality(aggSeries, aggPeriod)
                seasonals[k] = seasonal
            else:
                deseason = aggSeries
                seasonals[k] = None

            alpha, beta, level, trend = _desOptimize(deseason)

            levels.append(level)
            trends.append(trend / k)

        if levels:
            self._combinedLevel = np.median(levels)
        else:
            self._combinedLevel = self._y[-1]

        if trends:
            self._combinedTrend = np.median(trends)
        else:
            self._combinedTrend = 0.0

        if self._period > 1 and 1 in seasonals and seasonals[1] is not None:
            self._seasonal = seasonals[1]
        else:
            self._seasonal = None

        fitted = np.zeros(self._n)
        for t in range(self._n):
            fitted[t] = self._combinedLevel + self._combinedTrend * (t - self._n + 1)
            if self._seasonal is not None:
                fitted[t] += self._seasonal[t % self._period]

        residuals = self._y - fitted
        self._residStd = max(np.std(residuals), 1e-8)

        self._components = {
            'levels': levels,
            'trends': trends,
            'seasonals': seasonals,
        }

        return self

    def predict(self, steps):
        predictions = np.zeros(steps)
        for h in range(steps):
            predictions[h] = self._combinedLevel + self._combinedTrend * (h + 1)
            if self._seasonal is not None:
                seasonIdx = (self._n + h) % self._period
                predictions[h] += self._seasonal[seasonIdx]

        sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - 1.96 * sigma
        upper = predictions + 1.96 * sigma

        return predictions, lower, upper

    def _aggregate(self, y, k):
        n = len(y)
        m = n // k
        if m < 1:
            return y.copy()
        trimmed = y[:m * k]
        return trimmed.reshape(m, k).mean(axis=1)

    def _removeSeasonality(self, y, period):
        n = len(y)
        seasonal = np.zeros(period)
        counts = np.zeros(period)

        trend = np.convolve(y, np.ones(period) / period, mode='valid')
        offset = (period - 1) // 2

        for i in range(len(trend)):
            idx = i + offset
            if idx < n:
                diff = y[idx] - trend[i]
                seasonal[idx % period] += diff
                counts[idx % period] += 1

        for i in range(period):
            if counts[i] > 0:
                seasonal[i] /= counts[i]

        seasonal -= np.mean(seasonal)

        deseasonalized = np.zeros(n)
        for i in range(n):
            deseasonalized[i] = y[i] - seasonal[i % period]

        return deseasonalized, seasonal

    def _detectPeriod(self, y):
        n = len(y)
        if n < 14:
            return 1
        from scipy.signal import periodogram as pg
        detrended = y - np.linspace(y[0], y[-1], n)
        freqs, power = pg(detrended)
        validMask = freqs > 0
        freqs = freqs[validMask]
        power = power[validMask]
        if len(freqs) == 0:
            return 1
        peakIdx = np.argmax(power)
        freq = freqs[peakIdx]
        if freq > 0:
            period = int(round(1.0 / freq))
            if 2 <= period <= n // 4:
                return period
        return 1

    def _defaultAggLevels(self, n, period):
        candidates = [1, 2, 3, 4, 6, 8, 12]
        if period > 1:
            for div in range(2, period + 1):
                if period % div == 0 and div not in candidates:
                    candidates.append(div)
            if period not in candidates:
                candidates.append(period)
        valid = [k for k in candidates if n // k >= 5]
        return sorted(set(valid))


class AdaptiveMAPAForecaster:

    def __init__(self, holdoutRatio=0.15, period=None):
        self._holdoutRatio = holdoutRatio
        self._period = period
        self._bestModel = None
        self._aggWeights = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64).copy()
        n = len(y)

        holdoutSize = max(1, int(n * self._holdoutRatio))
        holdoutSize = min(holdoutSize, n // 3)
        trainPart = y[:n - holdoutSize]
        valPart = y[n - holdoutSize:]

        base = TemporalAggregationForecaster(period=self._period)
        base.fit(trainPart)
        basePeriod = base._period
        aggLevels = base._defaultAggLevels(len(trainPart), basePeriod)

        levelContribs = []
        trendContribs = []
        smapes = []

        for k in aggLevels:
            single = TemporalAggregationForecaster(aggLevels=[k], period=basePeriod)
            single.fit(trainPart)
            pred, _, _ = single.predict(holdoutSize)
            pred = np.asarray(pred, dtype=np.float64)
            if not np.all(np.isfinite(pred)):
                continue
            smape = np.mean(
                2.0 * np.abs(valPart - pred) / (np.abs(valPart) + np.abs(pred) + 1e-10)
            )
            smapes.append(smape)
            levelContribs.append(single._combinedLevel)
            trendContribs.append(single._combinedTrend)

        if not smapes:
            self._bestModel = TemporalAggregationForecaster(period=self._period)
            self._bestModel.fit(y)
            return self

        smapes = np.array(smapes)
        invSmapes = 1.0 / np.maximum(smapes, 1e-10)
        weights = invSmapes / invSmapes.sum()
        self._aggWeights = weights

        self._bestModel = TemporalAggregationForecaster(period=self._period)
        self._bestModel.fit(y)

        self._bestModel._combinedLevel = np.average(levelContribs, weights=weights)
        self._bestModel._combinedTrend = np.average(trendContribs, weights=weights)

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
    print("E036: Temporal Aggregation Forecaster (MAPA)")
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
        "mapa": lambda: TemporalAggregationForecaster(),
        "mapa_adaptive": lambda: AdaptiveMAPAForecaster(),
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
            marker = " ***" if mName.startswith("mapa") else ""
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
        marker = " ***" if mName.startswith("mapa") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: MAPA vs Best Existing")
    print("=" * 70)

    mapaWins = 0
    existWins = 0

    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        newMapes = {k: v for k, v in mapes.items() if k.startswith("mapa")}
        existMapes = {k: v for k, v in mapes.items() if not k.startswith("mapa")}

        if not newMapes or not existMapes:
            continue

        bestNew = min(newMapes, key=newMapes.get)
        bestExist = min(existMapes, key=existMapes.get)
        nVal = newMapes[bestNew]
        eVal = existMapes[bestExist]

        winner = "MAPA" if nVal <= eVal else "EXIST"
        if winner == "MAPA":
            mapaWins += 1
        else:
            existWins += 1

        improvement = (eVal - nVal) / max(eVal, 1e-8) * 100
        print(f"  {dsName:25s} | {bestNew:18s} {nVal:10.2f}% vs {bestExist:10s} {eVal:10.2f}% | {winner} ({improvement:+.1f}%)")

    total = mapaWins + existWins
    if total > 0:
        print(f"\n  MAPA wins: {mapaWins}/{total} ({mapaWins/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Residual Correlation with Existing Models")
    print("=" * 70)

    corrDatasets = ["retailSales", "hourlyMultiSeasonal", "energyUsage", "trending"]
    existingNames = ["mstl", "arima", "theta", "auto_ces", "dot", "4theta"]

    for dsName in corrDatasets:
        if dsName not in datasets:
            continue
        values = datasets[dsName]
        n = len(values)
        trainEnd = n - horizon
        train = values[:trainEnd]
        actual = values[trainEnd:trainEnd + horizon]

        mapaModel = AdaptiveMAPAForecaster()
        mapaModel.fit(train)
        mapaPred, _, _ = mapaModel.predict(horizon)
        mapaResid = actual - np.asarray(mapaPred[:horizon], dtype=np.float64)

        corrStrs = []
        for eName in existingNames:
            factory = allModels.get(eName)
            if factory is None:
                continue
            eModel = factory()
            eModel.fit(train)
            ePred, _, _ = eModel.predict(horizon)
            eResid = actual - np.asarray(ePred[:horizon], dtype=np.float64)

            if np.std(mapaResid) > 1e-10 and np.std(eResid) > 1e-10:
                corr = np.corrcoef(mapaResid, eResid)[0, 1]
                corrStrs.append(f"{eName}:{corr:.2f}")

        print(f"  {dsName:25s} | mapa_adaptive vs {', '.join(corrStrs)}")

    print("\n" + "=" * 70)
    print("ANALYSIS 5: Aggregation Level Analysis")
    print("=" * 70)

    for dsName in ["retailSales", "trending", "volatile", "hourlyMultiSeasonal"]:
        if dsName not in datasets:
            continue
        values = datasets[dsName]
        train = values[:len(values) - horizon]
        m = TemporalAggregationForecaster()
        m.fit(train)
        print(f"  {dsName:25s} | period={m._period}, aggLevels={m._aggLevels}")
        print(f"  {'':25s} | level={m._combinedLevel:.2f}, trend={m._combinedTrend:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if total > 0:
        print(f"\n    MAPA win rate: {mapaWins}/{total} ({mapaWins/total*100:.1f}%)")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
