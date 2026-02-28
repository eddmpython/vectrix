"""
==============================================================================
실험 ID: modelCreation/003
실험명: Wavelet Shrinkage Forecaster — 웨이블릿 축소 추정 예측 모델
==============================================================================

목적:
- Donoho-Johnstone 웨이블릿 축소 이론을 시계열 예측에 적용
- Haar DWT로 시간-주파수 동시 분해 → 스케일별 최적 디노이징 → 스케일별 외삽
- FFT의 한계(비정상성, Gibbs 현상)를 극복
- 구현 난이도가 가장 낮은 물리/신호처리 기반 모델

가설:
1. 비정상(non-stationary) 시계열에서 FourierForecaster보다 우수
2. 다중 스케일 데이터에서 기존 모델 대비 개선
3. 고노이즈 데이터에서 최적 디노이징으로 MAPE 개선
4. 구현 간결성: Haar DWT/IDWT 각 20줄 이내

방법:
1. WaveletShrinkageForecaster 클래스 구현
   - Haar DWT 직접 구현 (numpy 배열 연산)
   - 스케일별 soft thresholding (λ = σ√(2 ln n))
   - 근사 계수: 선형 외삽, 유의 상세 계수: 주기적 외삽
   - 역 DWT로 재조합
2. 합성 데이터 12종 벤치마크
3. 기존 5개 모델 대비 비교

성공 기준:
- 전체 평균 순위 상위 50%
- 비정상/다중스케일 데이터에서 top-3

==============================================================================
결과
==============================================================================

1. 전체 평균 순위 (12개 데이터셋):
   - mstl: 2.92, wavelet: 3.33 (2위) ***, dot: 3.42
   - auto_ces: 4.25, wavelet_adaptive: 4.33, theta: 4.75, arima: 5.00

2. Head-to-head 승률: 1/12 = 8.3% (temperature에서만 승리)
   - 대부분 데이터에서 중위권 (3~5위)
   - 1위 달성 없음 (mstl 또는 dot에 항상 밀림)

3. 주목할 결과:
   - multiSeasonalRetail: 3위 6.27% (arima 4.25% 다음, auto_ces 7.05% 앞)
   - stationary: 2위 2.03% (dot과 거의 동일)
   - trending: 2위 0.95% (arima 0.87% 바로 다음)
   - temperature: 1위 390365% (기존 최고 mstl 479607% 대비 18.6% 개선)

4. 가설 검증:
   - 가설 1 (비정상): nonStationary에서 5위 13.74% — 기각
   - 가설 2 (다중 스케일): multiSeasonalRetail 3위 — 부분 채택
   - 가설 3 (고노이즈): volatile 5위 0.88% — 기각 (dot 0.39%)
   - 가설 4 (구현 간결): Haar DWT/IDWT 각 15줄 — 채택

5. 약점:
   - wavelet과 wavelet_adaptive가 거의 동일 결과 (validation 무효)
   - 주기적 외삽이 단순 반복이라 계절 패턴 정밀도 낮음
   - DWT의 디노이징 효과는 있으나 외삽 전략이 빈약

결론: 기각 — 범용 예측 모델로 부적합
- 평균 순위는 준수(3.33)하나 head-to-head 1/12
- 디노이징 자체는 효과적이나 예측(외삽) 전략이 약함
- frequencyDomain/002 Spectral Denoising처럼 "전처리 도구"로서의 가치만 있음
- 독립 예측 모델이 아닌, 입력 정제 모듈로 전환 검토

==============================================================================
실험일: 2026-02-28
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


class WaveletShrinkageForecaster:

    def __init__(self, maxLevels=None, shrinkageType='soft', energyThreshold=0.9):
        self.maxLevels = maxLevels
        self.shrinkageType = shrinkageType
        self.energyThreshold = energyThreshold
        self._y = None
        self._coeffs = None
        self._sigMasks = None
        self._nLevels = 0
        self._padLen = 0
        self._residualStd = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        n = len(self._y)

        self._nLevels = self.maxLevels if self.maxLevels else min(5, int(np.log2(max(n, 4))) - 1)

        padLen = 1
        while padLen < n:
            padLen *= 2
        self._padLen = padLen
        padded = np.zeros(padLen)
        padded[:n] = self._y
        for i in range(n, padLen):
            padded[i] = self._y[-1]

        self._coeffs, self._lengths = self._haarDWT(padded, self._nLevels)

        self._sigMasks = []
        for j in range(len(self._coeffs)):
            if j == 0:
                self._sigMasks.append(True)
                continue
            detail = self._coeffs[j]
            if len(detail) == 0:
                self._sigMasks.append(False)
                continue

            sigma = np.median(np.abs(detail)) / 0.6745 if len(detail) > 0 else 0
            threshold = sigma * np.sqrt(2.0 * np.log(max(len(detail), 2)))

            if self.shrinkageType == 'soft':
                shrunk = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            else:
                shrunk = detail * (np.abs(detail) > threshold)

            energy = np.sum(shrunk ** 2)
            totalEnergy = np.sum(detail ** 2) + 1e-10
            isSignificant = energy / totalEnergy > (1.0 - self.energyThreshold)
            self._sigMasks.append(isSignificant)
            self._coeffs[j] = shrunk

        reconstructed = self._haarIDWT(self._coeffs, self._lengths)
        residuals = self._y - reconstructed[:n]
        self._residualStd = max(np.std(residuals), 1e-8)

        return self

    def predict(self, steps):
        n = len(self._y)
        totalNeeded = n + steps

        extLengths = []
        for origLen in self._lengths:
            ratio = totalNeeded / n
            extLengths.append(int(np.ceil(origLen * ratio)) + 2)

        extCoeffs = []
        for j in range(len(self._coeffs)):
            coeff = self._coeffs[j]
            targetLen = extLengths[j] if j < len(extLengths) else len(coeff) + steps
            extraNeeded = max(0, targetLen - len(coeff))

            if j == 0:
                ext = self._extrapolateTrend(coeff, extraNeeded)
            elif self._sigMasks[j] and extraNeeded > 0:
                ext = self._periodicExtend(coeff, extraNeeded)
            else:
                ext = np.concatenate([coeff, np.zeros(max(extraNeeded, 0))])
            extCoeffs.append(ext)

        prediction = self._haarIDWT(extCoeffs, extLengths)

        if len(prediction) < n + steps:
            pad = np.full(n + steps - len(prediction), prediction[-1] if len(prediction) > 0 else self._y[-1])
            prediction = np.concatenate([prediction, pad])

        pred = prediction[n:n + steps]

        sigma = self._residualStd * np.sqrt(np.arange(1, steps + 1))
        lower = pred - 1.96 * sigma
        upper = pred + 1.96 * sigma

        return pred, lower, upper

    def _haarDWT(self, signal, nLevels):
        current = signal.copy().astype(np.float64)
        details = []
        origLens = []

        for _ in range(nLevels):
            n = len(current)
            if n < 4:
                break
            origLens.append(n)
            halfN = n // 2
            trimN = halfN * 2

            approx = (current[:trimN:2] + current[1:trimN:2]) / np.sqrt(2)
            detail = (current[:trimN:2] - current[1:trimN:2]) / np.sqrt(2)

            details.append(detail)
            current = approx

        coeffs = [current] + details
        return coeffs, origLens

    def _haarIDWT(self, coeffs, origLens):
        current = coeffs[0].copy()
        nDetails = len(coeffs) - 1

        for j in range(nDetails):
            detail = coeffs[nDetails - j]
            n = min(len(current), len(detail))

            reconstructed = np.zeros(2 * n)
            reconstructed[0::2] = (current[:n] + detail[:n]) / np.sqrt(2)
            reconstructed[1::2] = (current[:n] - detail[:n]) / np.sqrt(2)

            if j < len(origLens):
                targetLen = origLens[len(origLens) - 1 - j]
                current = reconstructed[:targetLen]
            else:
                current = reconstructed

        return current

    def _extrapolateTrend(self, coeff, extraNeeded):
        n = len(coeff)
        if extraNeeded <= 0:
            return coeff
        if n < 2:
            return np.concatenate([coeff, np.full(extraNeeded, coeff[-1] if n > 0 else 0)])

        slope = (coeff[-1] - coeff[max(0, n - 10)]) / min(10, n - 1)
        extension = coeff[-1] + slope * np.arange(1, extraNeeded + 1)
        return np.concatenate([coeff, extension])

    def _periodicExtend(self, coeff, extraNeeded):
        n = len(coeff)
        if extraNeeded <= 0:
            return coeff
        if n < 2:
            return np.concatenate([coeff, np.zeros(extraNeeded)])

        period = min(n, max(2, n // 2))
        extension = np.zeros(extraNeeded)
        for i in range(extraNeeded):
            extension[i] = coeff[n - period + (i % period)]
        return np.concatenate([coeff, extension])


class AdaptiveWaveletForecaster:

    def __init__(self):
        self._bestModel = None
        self._y = None

    def fit(self, y):
        self._y = np.asarray(y, dtype=np.float64).copy()
        n = len(self._y)

        valSize = min(14, n // 5)
        train = self._y[:n - valSize]
        val = self._y[n - valSize:]

        configs = [
            {'shrinkageType': 'soft', 'energyThreshold': 0.85},
            {'shrinkageType': 'soft', 'energyThreshold': 0.90},
            {'shrinkageType': 'soft', 'energyThreshold': 0.95},
            {'shrinkageType': 'hard', 'energyThreshold': 0.85},
            {'shrinkageType': 'hard', 'energyThreshold': 0.90},
        ]

        bestMape = float('inf')
        bestConfig = configs[0]

        for cfg in configs:
            try:
                m = WaveletShrinkageForecaster(**cfg)
                m.fit(train)
                pred, _, _ = m.predict(valSize)
                if not np.all(np.isfinite(pred)):
                    continue
                mape = np.mean(np.abs((val - pred) / np.maximum(np.abs(val), 1e-8)))
                if mape < bestMape:
                    bestMape = mape
                    bestConfig = cfg
            except Exception:
                continue

        self._bestModel = WaveletShrinkageForecaster(**bestConfig)
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


def _generateNonStationary(n=365, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    base = 100.0
    trend1 = 50.0 * t / (n / 2) * (t < n / 2)
    trend2 = (50.0 - 30.0 * (t - n / 2) / (n / 2)) * (t >= n / 2)
    seasonal = 20.0 * np.sin(2.0 * np.pi * t / 30.0) * (1 + 0.5 * t / n)
    noise = rng.normal(0, 5, n)
    return base + trend1 + trend2 + seasonal + noise


def _runExperiment():
    from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS

    print("=" * 70)
    print("E033: Wavelet Shrinkage Forecaster")
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
    datasets["nonStationary"] = _generateNonStationary(365, 42)

    existingModels = _buildModels()
    newModels = {
        "wavelet": lambda: WaveletShrinkageForecaster(),
        "wavelet_adaptive": lambda: AdaptiveWaveletForecaster(),
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
            marker = " ***" if mName.startswith("wavelet") else ""
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
        marker = " ***" if mName.startswith("wavelet") else ""
        print(f"  {mName:20s} | Avg Rank = {avgRank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Wavelet vs Best Existing (Head-to-Head)")
    print("=" * 70)

    wavWins = 0
    existWins = 0
    for dsName in sorted(results.keys()):
        mapes = results[dsName]
        wavMapes = {k: v for k, v in mapes.items() if k.startswith("wavelet")}
        existMapes = {k: v for k, v in mapes.items() if not k.startswith("wavelet")}

        if not wavMapes or not existMapes:
            continue

        bestWav = min(wavMapes, key=wavMapes.get)
        bestExist = min(existMapes, key=existMapes.get)

        wVal = wavMapes[bestWav]
        eVal = existMapes[bestExist]

        winner = "WAV" if wVal <= eVal else "EXIST"
        if winner == "WAV":
            wavWins += 1
        else:
            existWins += 1

        improvement = (eVal - wVal) / max(eVal, 1e-8) * 100
        print(f"  {dsName:25s} | {bestWav:18s} {wVal:10.2f}% vs {bestExist:10s} {eVal:10.2f}% | {winner} ({improvement:+.1f}%)")

    total = wavWins + existWins
    print(f"\n  Wavelet wins: {wavWins}/{total} ({wavWins/total*100:.1f}%)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\n    Wavelet win rate: {wavWins}/{total} ({wavWins/total*100:.1f}%)")


if __name__ == '__main__':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    _runExperiment()
