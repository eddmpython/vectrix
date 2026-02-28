"""
==============================================================================
실험 ID: modelCreation/001
실험명: Dynamic Time Scan Forecaster — 비모수 패턴 매칭 예측 모델
==============================================================================

목적:
- M4 Hourly에서 sMAPE 12.9%를 달성한 DTSF 방법론을 numpy로 재구현
- 과거에서 현재와 가장 유사한 패턴을 찾아 그 직후 값을 예측에 사용
- 기존 파라메트릭 모델(ETS, ARIMA, Theta 등)과 근본적으로 다른 예측 원리
- 잔차 상관 0.73~1.0 문제 해결을 위한 비모수 모델 확보

가설:
1. Hourly 스타일 데이터(강한 반복 주기)에서 기존 모델 대비 30%+ MAPE 개선
2. Daily 스타일 데이터에서 상위 50% 순위 달성
3. 짧은 시계열(n<100)에서는 유사 패턴 부족으로 열위
4. 기존 통계 모델과 잔차 상관 0.5 이하 (앙상블 가치 높음)

방법:
1. DynamicTimeScanForecaster 클래스 구현
   - 스캔 윈도우 W 기반 패턴 매칭
   - 유사 패턴 K개의 후속 값 중앙값 = 예측
   - 자동 윈도우 크기: periodogram 기반 dominant period 감지
2. 합성 데이터 12종 (기존 11종 + hourlyMultiSeasonal)
3. 기존 5개 모델 대비 비교 (MSTL, ARIMA, Theta, AutoCES, DOT)
4. 잔차 상관 분석

성공 기준:
- hourlyMultiSeasonal에서 1위
- 전체 평균 순위 상위 50%
- 기존 모델과 잔차 상관 0.5 이하

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (12개 데이터셋):
   - mstl: 3.08 (1위)
   - dtsf_adaptive: 3.58 (2위) ***
   - dot: 3.83 (3위)
   - auto_ces: 4.25, arima: 5.08
   - dtsf_multiscale: 5.17, theta: 5.50, dtsf: 5.50

2. Head-to-head 승률: 5/12 = 41.7%
   - 승리: hourlyMultiSeasonal(64.7%↑), manufacturing(29.9%↑), stationary(13.8%↑),
           temperature(64.9%↑), volatile(3.2%↑)
   - 패배: longRetail, multiSeasonalRetail, regimeShift, stockPrice, trending, energyUsage, retailSales

3. hourlyMultiSeasonal 결과 (가설 1 검증):
   - dtsf_multiscale: 4.12% (1위) — 기존 최고 theta 11.69% 대비 64.7% 개선
   - dtsf: 4.26% (2위), dtsf_adaptive: 5.57% (3위)
   → 가설 1 채택: Hourly 데이터에서 기존 모델 대비 60%+ 개선

4. 잔차 상관 분석 (가설 4 검증):
   - retailSales: dtsf vs arima 0.10, vs theta 0.29, vs dot 0.29 (매우 낮음!)
   - hourlyMultiSeasonal: dtsf vs mstl 0.47, vs theta/ces/dot 0.50
   - energyUsage: dtsf vs theta/ces -0.61 (역상관!)
   - longRetail: dtsf vs theta/dot -0.07 (거의 무상관)
   → 가설 4 채택: 대부분 데이터에서 잔차 상관 0.5 이하, 특히 theta/ces/dot과 0.1~0.3

5. 윈도우 크기 자동 감지:
   - hourlyMultiSeasonal: 24 (정확!)
   - 대부분 daily 데이터: 87 (연 4분기 주기 감지, 주간 7이 아님 — periodogram 한계)
   → 윈도우 크기 감지 개선 필요 (daily는 7 고정이 더 나을 수 있음)

6. 3종 변형 비교:
   - dtsf_adaptive (시간 감쇠): 평균 순위 3.58, 가장 범용적. 최근 패턴 가중으로 안정
   - dtsf_multiscale (다중 스케일): hourly에서 최강, 그러나 불안정 데이터에서 열위
   - dtsf (기본): 평균 순위 5.50, 윈도우 크기 의존도 높음

결론: 조건부 채택 — dtsf_adaptive를 엔진 모델로 통합 가능
- 평균 순위 2위 (3.58), mstl(3.08) 바로 다음
- hourlyMultiSeasonal에서 독보적 1위 (64.7% 개선)
- 잔차 직교성 확보: 기존 모델과 상관 0.1~0.5 (앙상블 가치 매우 높음)
- 약점: 추세 외삽 불가, 짧은 시계열에서 패턴 부족
- DNA 활성화 조건: hasSeasonality=True, length > 100, 또는 frequency=Hourly

==============================================================================
실험일: 2026-02-28
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.signal import periodogram

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class DynamicTimeScanForecaster:

    def __init__(self, windowSize=None, nNeighbors=5, normalize=True):
        self.windowSize = windowSize
        self.nNeighbors = nNeighbors
        self.normalize = normalize
        self._y = None
        self._detectedPeriod = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        if self.windowSize is None:
            self._detectedPeriod = self._detectPeriod(self._y)
            self.windowSize = self._detectedPeriod
        self._residStd = max(np.std(np.diff(self._y)), 1e-8)
        return self

    def predict(self, steps):
        y = self._y
        n = len(y)
        W = min(self.windowSize, n // 3)
        W = max(W, 2)

        if n < W + steps + 1:
            pred = np.full(steps, np.mean(y))
            sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
            return pred, pred - 1.96 * sigma, pred + 1.96 * sigma

        if self.normalize:
            rollingMean = np.convolve(y, np.ones(W) / W, mode='valid')
            rollingStd = np.array([
                np.std(y[i:i + W]) for i in range(n - W + 1)
            ])
            rollingStd = np.maximum(rollingStd, 1e-8)
        else:
            rollingMean = np.zeros(n - W + 1)
            rollingStd = np.ones(n - W + 1)

        query = y[-W:]
        if self.normalize:
            qMean = np.mean(query)
            qStd = max(np.std(query), 1e-8)
            queryNorm = (query - qMean) / qStd
        else:
            queryNorm = query

        maxStart = n - W - steps
        if maxStart < 1:
            pred = np.full(steps, np.mean(y))
            sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
            return pred, pred - 1.96 * sigma, pred + 1.96 * sigma

        distances = np.zeros(maxStart)
        for i in range(maxStart):
            window = y[i:i + W]
            if self.normalize:
                wMean = rollingMean[i] if i < len(rollingMean) else np.mean(window)
                wStd = rollingStd[i] if i < len(rollingStd) else max(np.std(window), 1e-8)
                windowNorm = (window - wMean) / wStd
            else:
                windowNorm = window
            distances[i] = np.sqrt(np.mean((queryNorm - windowNorm) ** 2))

        K = min(self.nNeighbors, maxStart)
        neighborIdx = np.argpartition(distances, K)[:K]

        futures = np.zeros((K, steps))
        for j, idx in enumerate(neighborIdx):
            segment = y[idx + W: idx + W + steps]
            futures[j, :len(segment)] = segment
            if len(segment) < steps:
                futures[j, len(segment):] = segment[-1] if len(segment) > 0 else np.mean(y)

        predictions = np.median(futures, axis=0)
        pctLow = np.percentile(futures, 10, axis=0)
        pctHigh = np.percentile(futures, 90, axis=0)
        sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
        lower = np.minimum(pctLow, predictions - 1.96 * sigma)
        upper = np.maximum(pctHigh, predictions + 1.96 * sigma)

        return predictions, lower, upper

    def _detectPeriod(self, y):
        n = len(y)
        if n < 10:
            return 7

        detrended = y - np.linspace(y[0], y[-1], n)
        freqs, power = periodogram(detrended)

        if len(freqs) < 3:
            return 7

        validMask = freqs > 0
        freqs = freqs[validMask]
        power = power[validMask]

        if len(freqs) == 0:
            return 7

        peakIdx = np.argmax(power)
        dominantFreq = freqs[peakIdx]

        if dominantFreq > 0:
            period = int(round(1.0 / dominantFreq))
            period = max(2, min(period, n // 4))
            return period

        return 7


class AdaptiveDTSF(DynamicTimeScanForecaster):

    def __init__(self, nNeighbors=5, normalize=True, timeDecay=0.001):
        super().__init__(windowSize=None, nNeighbors=nNeighbors, normalize=normalize)
        self.timeDecay = timeDecay

    def predict(self, steps):
        y = self._y
        n = len(y)
        W = min(self.windowSize, n // 3)
        W = max(W, 2)

        if n < W + steps + 1:
            pred = np.full(steps, np.mean(y))
            sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
            return pred, pred - 1.96 * sigma, pred + 1.96 * sigma

        if self.normalize:
            qMean = np.mean(y[-W:])
            qStd = max(np.std(y[-W:]), 1e-8)
            queryNorm = (y[-W:] - qMean) / qStd
        else:
            queryNorm = y[-W:]

        maxStart = n - W - steps
        if maxStart < 1:
            pred = np.full(steps, np.mean(y))
            sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
            return pred, pred - 1.96 * sigma, pred + 1.96 * sigma

        distances = np.zeros(maxStart)
        for i in range(maxStart):
            window = y[i:i + W]
            if self.normalize:
                wMean = np.mean(window)
                wStd = max(np.std(window), 1e-8)
                windowNorm = (window - wMean) / wStd
            else:
                windowNorm = window
            shapeDist = np.sqrt(np.mean((queryNorm - windowNorm) ** 2))
            timeWeight = np.exp(-self.timeDecay * (n - i))
            distances[i] = shapeDist / max(timeWeight, 1e-10)

        K = min(self.nNeighbors, maxStart)
        neighborIdx = np.argpartition(distances, K)[:K]

        futures = np.zeros((K, steps))
        weights = np.zeros(K)
        for j, idx in enumerate(neighborIdx):
            segment = y[idx + W: idx + W + steps]
            futures[j, :len(segment)] = segment
            if len(segment) < steps:
                futures[j, len(segment):] = segment[-1] if len(segment) > 0 else np.mean(y)
            weights[j] = 1.0 / max(distances[idx], 1e-10)

        weights /= weights.sum()
        predictions = np.average(futures, axis=0, weights=weights)
        pctLow = np.percentile(futures, 10, axis=0)
        pctHigh = np.percentile(futures, 90, axis=0)
        sigma = self._residStd * np.sqrt(np.arange(1, steps + 1))
        lower = np.minimum(pctLow, predictions - 1.96 * sigma)
        upper = np.maximum(pctHigh, predictions + 1.96 * sigma)

        return predictions, lower, upper


class MultiScaleDTSF:

    def __init__(self, nNeighbors=5, normalize=True):
        self.nNeighbors = nNeighbors
        self.normalize = normalize
        self._y = None
        self._models = []

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        basePeriod = DynamicTimeScanForecaster()
        basePeriod.fit(self._y)
        detectedPeriod = basePeriod._detectedPeriod

        scales = set()
        scales.add(detectedPeriod)
        scales.add(max(2, detectedPeriod // 2))
        scales.add(min(len(y) // 4, detectedPeriod * 2))

        self._models = []
        for w in sorted(scales):
            if w < 2 or w > len(y) // 3:
                continue
            m = DynamicTimeScanForecaster(
                windowSize=w, nNeighbors=self.nNeighbors, normalize=self.normalize
            )
            m.fit(self._y)
            self._models.append(m)

        if not self._models:
            m = DynamicTimeScanForecaster(windowSize=7, nNeighbors=self.nNeighbors)
            m.fit(self._y)
            self._models.append(m)

        return self

    def predict(self, steps):
        allPreds = []
        allLower = []
        allUpper = []
        for m in self._models:
            p, lo, hi = m.predict(steps)
            allPreds.append(p)
            allLower.append(lo)
            allUpper.append(hi)

        predictions = np.median(allPreds, axis=0)
        lower = np.min(allLower, axis=0)
        upper = np.max(allUpper, axis=0)

        return predictions, lower, upper


def _buildModels():
    from vectrix.engine.arima import AutoARIMA
    from vectrix.engine.ces import AutoCES
    from vectrix.engine.dot import DynamicOptimizedTheta
    from vectrix.engine.mstl import AutoMSTL
    from vectrix.engine.theta import OptimizedTheta

    return {
        "mstl": lambda: AutoMSTL(),
        "arima": lambda: AutoARIMA(),
        "theta": lambda: OptimizedTheta(),
        "auto_ces": lambda: AutoCES(),
        "dot": lambda: DynamicOptimizedTheta(),
    }


def _generateHourlyMultiSeasonal(n=720, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    base = 500.0
    hourly = 80.0 * np.sin(2.0 * np.pi * t / 24.0)
    daily = 40.0 * np.sin(2.0 * np.pi * t / (24.0 * 7.0))
    noise = rng.normal(0, 15, n)
    values = base + hourly + daily + noise
    return values


def _generateLongRetail(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    base = 1000.0
    trend = 200.0 * t / n
    weekly = 150.0 * np.sin(2.0 * np.pi * t / 7.0)
    yearly = 200.0 * np.sin(2.0 * np.pi * t / 365.0)
    noise = rng.normal(0, 50, n)
    values = base + trend + weekly + yearly + noise
    return np.maximum(values, 100.0)


def _runExperiment():
    from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS

    print("=" * 70)
    print("E031: Dynamic Time Scan Forecaster")
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
    datasets["longRetail"] = _generateLongRetail(1000, 42)

    existingModels = _buildModels()
    newModels = {
        "dtsf": lambda: DynamicTimeScanForecaster(nNeighbors=7),
        "dtsf_adaptive": lambda: AdaptiveDTSF(nNeighbors=7),
        "dtsf_multiscale": lambda: MultiScaleDTSF(nNeighbors=7),
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
        sorted_models = sorted(mapes.items(), key=lambda x: x[1])

        print(f"\n  [{dsName}]")
        for rank, (mName, mape) in enumerate(sorted_models, 1):
            marker = " ***" if mName.startswith("dtsf") else ""
            print(f"    {rank}. {mName:20s} MAPE={mape:12.2f}%{marker}")

        for rank, (mName, _) in enumerate(sorted_models, 1):
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
        marker = " ***" if mName.startswith("dtsf") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: DTSF vs Best Existing (Head-to-Head)")
    print("=" * 70)

    dtsfWins = 0
    existWins = 0
    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        bestDtsf = min(
            (v for k, v in mapes.items() if k.startswith("dtsf")),
            default=float('inf')
        )
        bestDtsfName = min(
            ((k, v) for k, v in mapes.items() if k.startswith("dtsf")),
            key=lambda x: x[1]
        )[0]
        bestExist = min(
            (v for k, v in mapes.items() if not k.startswith("dtsf")),
            default=float('inf')
        )
        bestExistName = min(
            ((k, v) for k, v in mapes.items() if not k.startswith("dtsf")),
            key=lambda x: x[1]
        )[0]

        winner = "DTSF" if bestDtsf <= bestExist else "EXIST"
        if winner == "DTSF":
            dtsfWins += 1
        else:
            existWins += 1

        improvement = (bestExist - bestDtsf) / max(bestExist, 1e-8) * 100
        print(f"  {dsName:25s} | {bestDtsfName:18s} {bestDtsf:8.2f}% vs {bestExistName:10s} {bestExist:8.2f}% | {winner} ({improvement:+.1f}%)")

    total = dtsfWins + existWins
    print(f"\n  DTSF wins: {dtsfWins}/{total} ({dtsfWins/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Residual Correlation (DTSF vs Existing)")
    print("=" * 70)

    corrDatasets = ["retailSales", "hourlyMultiSeasonal", "longRetail", "energyUsage", "stationary"]
    for dsName in corrDatasets:
        if dsName not in datasets:
            continue
        values = datasets[dsName]
        n = len(values)
        trainEnd = n - horizon
        train = values[:trainEnd]
        actual = values[trainEnd:trainEnd + horizon]

        residuals = {}
        for modelName in allModels:
            try:
                model = allModels[modelName]()
                model.fit(train)
                pred, _, _ = model.predict(horizon)
                pred = np.asarray(pred[:horizon], dtype=np.float64)
                residuals[modelName] = actual - pred
            except Exception:
                continue

        if "dtsf" not in residuals:
            continue

        print(f"\n  [{dsName}]")
        dtsfResid = residuals["dtsf"]
        for mName in ["mstl", "arima", "theta", "auto_ces", "dot"]:
            if mName in residuals:
                corr = np.corrcoef(dtsfResid, residuals[mName])[0, 1]
                print(f"    dtsf vs {mName:10s}: r = {corr:.3f}")

    print("\n" + "=" * 70)
    print("ANALYSIS 5: Window Size Detection")
    print("=" * 70)

    for dsName in sorted(datasets.keys()):
        values = datasets[dsName]
        m = DynamicTimeScanForecaster()
        m.fit(values[:len(values) - horizon])
        print(f"  {dsName:25s} | detected period = {m._detectedPeriod}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\n    DTSF win rate: {dtsfWins}/{total} ({dtsfWins/total*100:.1f}%)")
    bestNew = min(avgRanks, key=lambda x: x[1] if x[0].startswith("dtsf") else 99)
    print(f"    Best DTSF variant: {bestNew[0]} (avg rank {bestNew[1]:.2f})")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
