"""
==============================================================================
실험 ID: modelCreation/002
실험명: Koopman Mode Forecaster — 동적 모드 분해 기반 예측 모델
==============================================================================

목적:
- 동역학계 스펙트럼 이론(Koopman Operator)에서 영감받은 예측 모델
- DMD(Dynamic Mode Decomposition)로 시계열의 비선형 동역학을 선형 모드로 분해
- 각 모드의 고유값 = 감쇠율 + 진동주파수, 물리적 해석 가능
- FourierForecaster의 한계(감쇠 모델링 제한)를 극복

가설:
1. 감쇠하는 진동 패턴이 있는 데이터에서 기존 모델 대비 우위
2. 비선형 동역학 시계열(stockPrice, volatile)에서 FFT보다 우수
3. 안정 모드 필터링으로 예측 발산 방지
4. 기존 통계 모델과 다른 오류 구조 (잔차 직교성 확보)

방법:
1. KoopmanModeForecaster 클래스 구현
   - Takens 시간지연 임베딩 (1D → d차원 상태공간)
   - SVD 기반 축소 DMD
   - 안정 모드만 선택 (|λ| ≤ 1+ε)
   - 모드 기반 재귀 예측
2. 합성 데이터 12종 벤치마크
3. 기존 5개 모델 + DTSF 대비 비교

성공 기준:
- volatile/stockPrice에서 top-3 진입
- 전체 평균 순위 상위 50%

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (12개 데이터셋):
   - mstl: 3.08, auto_ces: 3.25, dot: 3.25, theta: 4.17
   - koopman: 4.67 (5위) ***
   - arima: 4.75, koopman_adaptive: 4.83

2. Head-to-head 승률: 3/12 = 25.0%
   - 승리: hourlyMultiSeasonal(30.2%↑), stockPrice(42.6%↑), temperature(22.3%↑)
   - stockPrice에서 독보적 1위: 3.35% (기존 최고 dot 5.84%, 42.6% 개선)

3. 가설 검증:
   - 가설 1 (감쇠 진동): dampedOscillation에서 4위(5.22%), auto_ces(3.89%)에 밀림 — 부분 기각
   - 가설 2 (비선형): stockPrice 1위(42.6%↑), volatile 5위(0.46% vs 0.39%) — 부분 채택
   - 가설 3 (안정 모드): 12개 중 발산 케이스 없음 — 채택

4. 모드 분석:
   - hourly: 15개 안정 모드, 감쇠율 0.55~0.98
   - 주기 감지가 시간지연 임베딩 기반이라 실제 주기(24)를 직접 포착하지 못함
   - 모드 주기가 2.0~4.2로 나타남 (임베딩 공간에서의 주기)

5. 약점:
   - manufacturing, regimeShift에서 크게 열위 (20~32% MAPE)
   - trending에서 46.95% (추세 외삽 약함)
   - dampedOscillation에서도 auto_ces에 밀림 (기대와 다름)

결론: 보류 — 특수 목적(stockPrice/hourly)으로 검토 가능
- stockPrice에서 42.6% 개선은 주목할 만함
- 그러나 범용성 낮음 (25% 승률)
- 001 DTSF(41.7%)보다 열위
- 임베딩 차원/랭크 자동 선택 개선 필요

==============================================================================
실험일: 2026-02-28
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class KoopmanModeForecaster:

    def __init__(self, embeddingDim=None, delay=1, rank=None, stabilityMargin=0.01):
        self.embeddingDim = embeddingDim
        self.delay = delay
        self.rank = rank
        self.stabilityMargin = stabilityMargin
        self._y = None
        self._eigenvalues = None
        self._modes = None
        self._amplitudes = None
        self._residualStd = None
        self._lastState = None
        self._nSteps = 0

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        n = len(self._y)

        if self.embeddingDim is None:
            self.embeddingDim = self._autoEmbeddingDim(n)

        d = self.embeddingDim
        tau = self.delay

        nEmbedded = n - (d - 1) * tau
        if nEmbedded < d + 2:
            d = max(2, n // 4)
            self.embeddingDim = d
            nEmbedded = n - (d - 1) * tau

        X = np.zeros((d, nEmbedded - 1))
        Y = np.zeros((d, nEmbedded - 1))
        for i in range(nEmbedded - 1):
            for j in range(d):
                X[j, i] = self._y[i + j * tau]
                Y[j, i] = self._y[i + 1 + j * tau]

        r = self.rank if self.rank else min(d, nEmbedded - 1, 15)
        r = min(r, min(X.shape))

        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        U_r = U[:, :r]
        S_r = S[:r]
        Vt_r = Vt[:r, :]

        S_inv = np.diag(1.0 / np.maximum(S_r, 1e-10))
        Atilde = U_r.T @ Y @ Vt_r.T @ S_inv

        eigenvalues, W = np.linalg.eig(Atilde)

        Phi = Y @ Vt_r.T @ S_inv @ W

        stableMask = np.abs(eigenvalues) <= (1.0 + self.stabilityMargin)
        if not np.any(stableMask):
            sortedIdx = np.argsort(np.abs(eigenvalues))
            stableMask[sortedIdx[:max(1, r // 2)]] = True

        self._eigenvalues = eigenvalues[stableMask]
        self._modes = Phi[:, stableMask]

        x0 = X[:, -1]
        self._amplitudes = np.linalg.lstsq(self._modes, x0, rcond=None)[0]

        reconstructed = np.zeros(nEmbedded - 1)
        for t in range(nEmbedded - 1):
            xRecon = np.zeros(d, dtype=np.complex128)
            for k in range(len(self._eigenvalues)):
                xRecon += self._amplitudes[k] * (self._eigenvalues[k] ** t) * self._modes[:, k]
            reconstructed[t] = np.real(xRecon[0])

        actualSeries = self._y[:nEmbedded - 1]
        residuals = actualSeries - reconstructed
        self._residualStd = max(np.std(residuals), 1e-8)
        self._nSteps = nEmbedded - 1

        return self

    def predict(self, steps):
        predictions = np.zeros(steps)
        nModes = len(self._eigenvalues)

        for h in range(steps):
            t = self._nSteps + h
            xPred = np.zeros(self.embeddingDim, dtype=np.complex128)
            for k in range(nModes):
                xPred += self._amplitudes[k] * (self._eigenvalues[k] ** t) * self._modes[:, k]
            predictions[h] = np.real(xPred[0])

        decayRates = np.abs(self._eigenvalues)
        avgDecay = np.mean(decayRates[decayRates < 1.0]) if np.any(decayRates < 1.0) else 0.95
        sigma = np.array([
            self._residualStd * np.sqrt(h + 1) / max(avgDecay ** (h + 1), 0.01)
            for h in range(steps)
        ])

        lower = predictions - 1.96 * sigma
        upper = predictions + 1.96 * sigma

        return predictions, lower, upper

    def _autoEmbeddingDim(self, n):
        if n > 500:
            return min(30, n // 20)
        if n > 200:
            return min(20, n // 15)
        if n > 50:
            return min(10, n // 8)
        return max(3, n // 10)


class AdaptiveKoopmanForecaster:

    def __init__(self, stabilityMargin=0.01):
        self.stabilityMargin = stabilityMargin
        self._bestModel = None
        self._y = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        n = len(self._y)

        valSize = min(14, n // 5)
        train = self._y[:n - valSize]
        val = self._y[n - valSize:]

        configs = []
        baseD = max(3, n // 20)
        for d in [max(3, baseD // 2), baseD, min(n // 4, baseD * 2)]:
            for r in [max(2, d // 3), max(2, d // 2), d]:
                configs.append((d, r))

        bestMape = float('inf')
        bestModel = None

        for d, r in configs:
            if d > len(train) // 3:
                continue
            try:
                m = KoopmanModeForecaster(
                    embeddingDim=d, rank=r, stabilityMargin=self.stabilityMargin
                )
                m.fit(train)
                pred, _, _ = m.predict(valSize)
                pred = np.real(pred)
                mape = np.mean(np.abs((val - pred) / np.maximum(np.abs(val), 1e-8)))
                if mape < bestMape and np.all(np.isfinite(pred)):
                    bestMape = mape
                    bestModel = (d, r)
            except Exception:
                continue

        if bestModel is None:
            bestModel = (max(3, n // 20), max(2, n // 40))

        self._bestModel = KoopmanModeForecaster(
            embeddingDim=bestModel[0], rank=bestModel[1],
            stabilityMargin=self.stabilityMargin
        )
        self._bestModel.fit(self._y)
        return self

    def predict(self, steps):
        return self._bestModel.predict(steps)


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
    return base + hourly + daily + noise


def _generateDampedOscillation(n=365, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    base = 100.0
    decay = np.exp(-0.005 * t)
    oscillation = 50.0 * decay * np.sin(2.0 * np.pi * t / 30.0)
    noise = rng.normal(0, 3, n)
    return base + oscillation + noise


def _runExperiment():
    from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS

    print("=" * 70)
    print("E032: Koopman Mode Forecaster")
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
    datasets["dampedOscillation"] = _generateDampedOscillation(365, 42)

    existingModels = _buildModels()
    newModels = {
        "koopman": lambda: KoopmanModeForecaster(),
        "koopman_adaptive": lambda: AdaptiveKoopmanForecaster(),
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
                pred = np.real(np.asarray(pred[:horizon], dtype=np.complex128))
                pred = pred.astype(np.float64)
                if not np.all(np.isfinite(pred)):
                    results[dsName][modelName] = float('inf')
                    continue
                mape = np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-8))) * 100
                results[dsName][modelName] = mape
            except Exception as exc:
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
            marker = " ***" if mName.startswith("koopman") else ""
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
        marker = " ***" if mName.startswith("koopman") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Koopman vs Best Existing (Head-to-Head)")
    print("=" * 70)

    koopWins = 0
    existWins = 0
    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        koopmanMapes = {k: v for k, v in mapes.items() if k.startswith("koopman")}
        existMapes = {k: v for k, v in mapes.items() if not k.startswith("koopman")}

        if not koopmanMapes or not existMapes:
            continue

        bestKoop = min(koopmanMapes, key=koopmanMapes.get)
        bestExist = min(existMapes, key=existMapes.get)

        kVal = koopmanMapes[bestKoop]
        eVal = existMapes[bestExist]

        winner = "KOOP" if kVal <= eVal else "EXIST"
        if winner == "KOOP":
            koopWins += 1
        else:
            existWins += 1

        improvement = (eVal - kVal) / max(eVal, 1e-8) * 100
        print(f"  {dsName:25s} | {bestKoop:18s} {kVal:10.2f}% vs {bestExist:10s} {eVal:10.2f}% | {winner} ({improvement:+.1f}%)")

    total = koopWins + existWins
    print(f"\n  Koopman wins: {koopWins}/{total} ({koopWins/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Mode Decomposition Analysis")
    print("=" * 70)

    for dsName in ["hourlyMultiSeasonal", "dampedOscillation", "retailSales", "volatile"]:
        if dsName not in datasets:
            continue
        values = datasets[dsName]
        train = values[:len(values) - horizon]
        m = KoopmanModeForecaster()
        m.fit(train)

        print(f"\n  [{dsName}]")
        print(f"    Embedding dim: {m.embeddingDim}, Stable modes: {len(m._eigenvalues)}")
        for k, ev in enumerate(m._eigenvalues[:5]):
            mag = np.abs(ev)
            freq = np.abs(np.angle(ev)) / (2 * np.pi)
            period = 1.0 / max(freq, 1e-10) if freq > 0.001 else float('inf')
            amp = np.abs(m._amplitudes[k])
            print(f"    Mode {k}: |λ|={mag:.4f}, freq={freq:.4f}, period={period:.1f}, amp={amp:.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\n    Koopman win rate: {koopWins}/{total} ({koopWins/total*100:.1f}%)")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
