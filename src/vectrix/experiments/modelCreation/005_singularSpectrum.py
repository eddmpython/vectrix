"""
==============================================================================
실험 ID: modelCreation/005
실험명: Singular Spectrum Forecaster — SVD 비모수 분해 예측 모델
==============================================================================

목적:
- Singular Spectrum Analysis(SSA)를 numpy로 재구현
- 궤적 행렬(trajectory matrix) → SVD → 자동 그룹화 → 대각 평균 → 재귀 예측
- 완전 비모수: 모델 가정 없이 데이터 자체가 구조를 드러냄
- 기존 파라메트릭 모델과 근본적으로 다른 분해 원리 → 앙상블 다양성 기여

가설:
1. 강한 주기성 데이터(retailSales, energyUsage)에서 상위 3위 이내
2. 비정상(trending, regimeShift) 데이터에서 적응형 윈도우가 고정 윈도우 대비 10%+ 개선
3. 기존 모델과 잔차 상관 0.5 이하 (SSA의 비모수 분해 특성)
4. 전체 평균 순위 상위 50%

방법:
1. SingularSpectrumForecaster 클래스 구현
   - 궤적 행렬 구성 (L×K Hankel matrix)
   - truncated SVD → 상위 r개 성분 선택 (자동 rank 결정)
   - 대각 평균(anti-diagonal averaging)으로 시계열 복원
   - 재귀 예측: 마지막 벡터 연장
2. AdaptiveSSAForecaster: 윈도우/rank 자동 최적화
3. 합성 데이터 12종 벤치마크
4. 기존 5개 모델 + 4Theta 대비 비교

성공 기준:
- 전체 평균 순위 상위 50% (3.5 이내)
- 계절성 데이터에서 최소 2개 1위

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (11개 데이터셋):
   - 4theta: 3.55 (1위)
   - mstl: 4.18 (2위)
   - dot: 4.55 (3위)
   - ssa: 4.82 (4위) ***
   - ssa_adaptive: 5.09 (5위) ***
   - auto_ces: 5.18, ssa_multigroup: 5.27, arima: 5.82, theta: 6.55

2. Head-to-head 승률: 2/11 = 18.2%
   - 승리: hourlyMultiSeasonal(44.3% 개선), stockPrice(31.5% 개선)
   - 패배: 나머지 9개 데이터셋

3. 강점 데이터:
   - hourlyMultiSeasonal: ssa 6.52% (1위!) — theta 11.69% 대비 44.3% 개선
   - stockPrice: ssa_multigroup 3.38% (1위!) — 4theta 4.94% 대비 31.5% 개선
   - stationary: ssa 1.86% (2위) — 4theta 1.47% 바로 뒤
   - volatile: ssa/ssa_adaptive 0.39% (2위 동률) — dot 0.39%와 동일

4. 약점 분석:
   - retailSales: ssa_adaptive 12.69% (7위) — mstl 3.34% 대비 3.8배 열위
   - trending: ssa_adaptive 5.71% (6위) — arima 0.87% 대비 6.6배 열위
   - autoRank에서 r=1만 선택 (85% 분산이 첫 성분에서 달성)
     → 주기성 성분 포착 부족, 실질적으로 단순 추세 모델로 전락

5. 잔차 상관 분석 (가설 3 검증):
   - retailSales: mstl:-0.06, arima:0.15 (좋음) / theta:1.00, ces:1.00, dot:1.00 (나쁨)
   - hourlyMultiSeasonal: 대부분 0.89~1.00 (나쁨)
   - energyUsage: mstl:-0.02, arima:0.17 (좋음) / 나머지 1.00 (나쁨)
   → 가설 3 기각: SSA 잔차가 기존 모델과 높은 상관 (r=1로 인한 단순 모델화)

6. SSA 성분 분석:
   - retailSales: L=87, r=1 — 1개 성분으로 85%+ 분산 설명
   - trending: L=87, r=1 — 동일
   - hourlyMultiSeasonal: L=48, r=1 — 동일
   → 근본 원인: autoRank threshold 85%가 너무 낮아 항상 r=1 선택

결론: 조건부 보류
- 평균 순위 4.82 (4위) — 중위 수준, 상위 50% 기준 미달
- hourlyMultiSeasonal, stockPrice에서 1위 — 특수 목적으로 가치
- 핵심 약점: autoRank r=1 → 계절성 포착 불가 → 추세만 예측
- 개선 방향: 최소 rank 3으로 강제하거나, 주기성 감지 기반 그룹화 필요
- 현재 상태로는 엔진 통합 불가, rank 개선 후 재실험 고려

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


class SingularSpectrumForecaster:

    def __init__(self, windowSize=None, nComponents=None):
        self._windowSize = windowSize
        self._nComponents = nComponents
        self._y = None
        self._n = 0
        self._L = 0
        self._r = 0
        self._U = None
        self._S = None
        self._V = None
        self._reconstructed = None
        self._lastVector = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        self._n = len(self._y)

        if self._windowSize is None:
            self._L = self._autoWindow(self._y)
        else:
            self._L = min(self._windowSize, self._n // 2)

        K = self._n - self._L + 1
        X = np.zeros((self._L, K))
        for i in range(K):
            X[:, i] = self._y[i:i + self._L]

        U, S, Vt = np.linalg.svd(X, full_matrices=False)

        if self._nComponents is None:
            self._r = self._autoRank(S)
        else:
            self._r = min(self._nComponents, len(S))

        self._U = U[:, :self._r]
        self._S = S[:self._r]
        self._V = Vt[:self._r, :].T

        Xr = self._U @ np.diag(self._S) @ Vt[:self._r, :]
        self._reconstructed = self._diagAverage(Xr, self._n)

        self._lastVector = self._reconstructed[-(self._L - 1):]

        return self

    def predict(self, steps):
        R = self._U[:-1, :]
        lastRow = self._U[-1, :]

        denom = 1.0 - np.sum(lastRow ** 2)
        if denom < 1e-10:
            denom = 1e-10
        coeff = lastRow @ R.T / denom

        extended = np.concatenate([self._reconstructed, np.zeros(steps)])
        for h in range(steps):
            startIdx = self._n + h - (self._L - 1)
            window = extended[startIdx:startIdx + self._L - 1]
            nextVal = coeff @ window
            extended[self._n + h] = nextVal

        predictions = extended[self._n:self._n + steps]

        residuals = self._y - self._reconstructed
        residStd = max(np.std(residuals), 1e-8)
        sigma = residStd * np.sqrt(np.arange(1, steps + 1))
        lower = predictions - 1.96 * sigma
        upper = predictions + 1.96 * sigma

        return predictions, lower, upper

    def _autoWindow(self, y):
        n = len(y)
        period = self._detectPeriod(y)
        if period > 1:
            L = min(period * 2, n // 2)
        else:
            L = max(n // 4, 10)
        L = max(L, 4)
        L = min(L, n // 2)
        return L

    def _detectPeriod(self, y):
        n = len(y)
        if n < 14:
            return 1
        detrended = y - np.linspace(y[0], y[-1], n)
        freqs, power = periodogram(detrended)
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

    def _autoRank(self, singularValues):
        total = np.sum(singularValues ** 2)
        if total < 1e-10:
            return 1
        cumVar = np.cumsum(singularValues ** 2) / total
        threshold = 0.85
        r = np.searchsorted(cumVar, threshold) + 1
        r = max(r, 1)
        r = min(r, len(singularValues))
        return r

    def _diagAverage(self, X, n):
        L, K = X.shape
        result = np.zeros(n)
        counts = np.zeros(n)
        for i in range(L):
            for j in range(K):
                result[i + j] += X[i, j]
                counts[i + j] += 1
        return result / np.maximum(counts, 1)


class AdaptiveSSAForecaster:

    def __init__(self, windowCandidates=None, rankCandidates=None, holdoutRatio=0.15):
        self._windowCandidates = windowCandidates
        self._rankCandidates = rankCandidates
        self._holdoutRatio = holdoutRatio
        self._bestModel = None
        self._bestWindow = None
        self._bestRank = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64).copy()
        n = len(y)

        period = self._detectPeriod(y)

        if self._windowCandidates is None:
            candidates = set()
            if period > 1:
                for mult in [1, 2, 3]:
                    w = period * mult
                    if 4 <= w <= n // 2:
                        candidates.add(w)
            for frac in [0.15, 0.25, 0.35]:
                w = max(4, int(n * frac))
                if w <= n // 2:
                    candidates.add(w)
            if not candidates:
                candidates.add(max(4, n // 4))
            windowCandidates = sorted(candidates)
        else:
            windowCandidates = self._windowCandidates

        if self._rankCandidates is None:
            rankCandidates = [None]
        else:
            rankCandidates = self._rankCandidates

        holdoutSize = max(1, int(n * self._holdoutRatio))
        holdoutSize = min(holdoutSize, n // 3)
        trainPart = y[:n - holdoutSize]
        valPart = y[n - holdoutSize:]

        bestSmape = float('inf')
        bestConfig = (windowCandidates[0], None)

        for w in windowCandidates:
            if w >= len(trainPart) // 2:
                continue
            for r in rankCandidates:
                model = SingularSpectrumForecaster(windowSize=w, nComponents=r)
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
                    bestConfig = (w, r)

        self._bestWindow, self._bestRank = bestConfig
        self._bestModel = SingularSpectrumForecaster(
            windowSize=self._bestWindow, nComponents=self._bestRank
        )
        self._bestModel.fit(y)
        return self

    def predict(self, steps):
        return self._bestModel.predict(steps)

    def _detectPeriod(self, y):
        n = len(y)
        if n < 14:
            return 1
        detrended = y - np.linspace(y[0], y[-1], n)
        freqs, power = periodogram(detrended)
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


class MultiGroupSSAForecaster:

    def __init__(self, windowSize=None, nTrend=None, nSeasonal=None):
        self._windowSize = windowSize
        self._nTrend = nTrend
        self._nSeasonal = nSeasonal
        self._y = None
        self._n = 0
        self._trendModel = None
        self._seasonModel = None
        self._trendRecon = None
        self._seasonRecon = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        self._n = len(self._y)

        base = SingularSpectrumForecaster(windowSize=self._windowSize)
        base.fit(self._y)

        L = base._L
        r = base._r
        U = base._U
        S = base._S
        V = base._V

        nTrend = self._nTrend if self._nTrend else max(1, r // 3)
        nSeasonal = self._nSeasonal if self._nSeasonal else max(1, r - nTrend)

        K = self._n - L + 1
        Vt_full = V.T

        trendX = U[:, :nTrend] @ np.diag(S[:nTrend]) @ Vt_full[:nTrend, :]
        self._trendRecon = base._diagAverage(trendX, self._n)

        endIdx = min(nTrend + nSeasonal, r)
        seasonX = U[:, nTrend:endIdx] @ np.diag(S[nTrend:endIdx]) @ Vt_full[nTrend:endIdx, :]
        self._seasonRecon = base._diagAverage(seasonX, self._n)

        self._trendModel = SingularSpectrumForecaster(windowSize=L, nComponents=nTrend)
        self._trendModel.fit(self._trendRecon)

        seasonPlusResid = self._y - self._trendRecon
        self._seasonModel = SingularSpectrumForecaster(windowSize=L, nComponents=nSeasonal)
        self._seasonModel.fit(seasonPlusResid)

        return self

    def predict(self, steps):
        trendPred, _, _ = self._trendModel.predict(steps)
        seasonPred, _, _ = self._seasonModel.predict(steps)
        combined = trendPred + seasonPred

        totalRecon = self._trendRecon + self._seasonRecon
        residuals = self._y - totalRecon
        residStd = max(np.std(residuals), 1e-8)
        sigma = residStd * np.sqrt(np.arange(1, steps + 1))
        lower = combined - 1.96 * sigma
        upper = combined + 1.96 * sigma

        return combined, lower, upper


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
    print("E035: Singular Spectrum Forecaster (SSA)")
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
        "ssa": lambda: SingularSpectrumForecaster(),
        "ssa_adaptive": lambda: AdaptiveSSAForecaster(),
        "ssa_multigroup": lambda: MultiGroupSSAForecaster(),
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
            except Exception as e:
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
            marker = " ***" if mName.startswith("ssa") else ""
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
        marker = " ***" if mName.startswith("ssa") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: SSA vs Best Existing")
    print("=" * 70)

    ssaWins = 0
    existWins = 0

    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        newMapes = {k: v for k, v in mapes.items() if k.startswith("ssa")}
        existMapes = {k: v for k, v in mapes.items() if not k.startswith("ssa")}

        if not newMapes or not existMapes:
            continue

        bestNew = min(newMapes, key=newMapes.get)
        bestExist = min(existMapes, key=existMapes.get)
        nVal = newMapes[bestNew]
        eVal = existMapes[bestExist]

        winner = "SSA" if nVal <= eVal else "EXIST"
        if winner == "SSA":
            ssaWins += 1
        else:
            existWins += 1

        improvement = (eVal - nVal) / max(eVal, 1e-8) * 100
        print(f"  {dsName:25s} | {bestNew:18s} {nVal:10.2f}% vs {bestExist:10s} {eVal:10.2f}% | {winner} ({improvement:+.1f}%)")

    total = ssaWins + existWins
    if total > 0:
        print(f"\n  SSA wins: {ssaWins}/{total} ({ssaWins/total*100:.1f}%)")

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

        ssaModel = AdaptiveSSAForecaster()
        ssaModel.fit(train)
        ssaPred, _, _ = ssaModel.predict(horizon)
        ssaResid = actual - np.asarray(ssaPred[:horizon], dtype=np.float64)

        corrStrs = []
        for eName in existingNames:
            if eName not in results.get(dsName, {}):
                continue
            factory = allModels.get(eName)
            if factory is None:
                continue
            eModel = factory()
            eModel.fit(train)
            ePred, _, _ = eModel.predict(horizon)
            eResid = actual - np.asarray(ePred[:horizon], dtype=np.float64)

            if np.std(ssaResid) > 1e-10 and np.std(eResid) > 1e-10:
                corr = np.corrcoef(ssaResid, eResid)[0, 1]
                corrStrs.append(f"{eName}:{corr:.2f}")

        print(f"  {dsName:25s} | ssa_adaptive vs {', '.join(corrStrs)}")

    print("\n" + "=" * 70)
    print("ANALYSIS 5: SSA Component Analysis")
    print("=" * 70)

    for dsName in ["retailSales", "trending", "hourlyMultiSeasonal"]:
        if dsName not in datasets:
            continue
        values = datasets[dsName]
        train = values[:len(values) - horizon]
        m = SingularSpectrumForecaster()
        m.fit(train)
        svRatio = m._S / m._S[0] if len(m._S) > 0 else []
        topN = min(6, len(svRatio))
        ratioStr = ", ".join([f"σ{i+1}={svRatio[i]:.3f}" for i in range(topN)])
        print(f"  {dsName:25s} | L={m._L}, r={m._r} | {ratioStr}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if total > 0:
        print(f"\n    SSA win rate: {ssaWins}/{total} ({ssaWins/total*100:.1f}%)")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
