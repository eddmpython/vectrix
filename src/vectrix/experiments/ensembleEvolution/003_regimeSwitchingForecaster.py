"""
==============================================================================
실험 ID: ensembleEvolution/003
실험명: Regime-Switching Forecaster — 레짐 전환 예측 모델 창조
==============================================================================

목적:
- 시계열에서 레짐(상태)을 자동 감지하고, 레짐별로 다른 모델을 적용
- 현재 vectrix에 changepoint 감지는 있으나 레짐별 예측 모델은 없음
- 각 레짐의 통계적 특성(평균, 분산, 추세)에 맞는 모델을 자동 선택

가설:
1. 레짐 변화가 있는 시계열에서 단일 모델보다 10%+ MAPE 개선
2. 정상적(stationary) 시계열에서는 단일 모델과 동등
3. 마지막 레짐 특성에 맞는 모델 선택이 전체 데이터 학습보다 유리

방법:
1. RegimeSwitchingForecaster 클래스 구현
   - CUSUM/변동점 감지로 레짐 경계 식별
   - 각 레짐의 특성 분석 (평균, 분산, 추세, 계절성)
   - 마지막 레짐 데이터로만 모델 학습 (전체 vs 부분 비교)
   - 레짐 특성에 맞는 모델 자동 선택
2. 합성 데이터 10종 + regimeShift 특화 데이터
3. 기존 모델 대비 비교

성공 기준:
- regimeShift 데이터에서 1위
- 전체 평균 순위 상위 50%

==============================================================================
결과
==============================================================================

1. 전체 평균 순위:
   - mstl: 3.08 (1위)
   - multi_regime: 3.17 (2위) ***
   - regime_switch: 3.50 (3위) ***
   - arima: 4.08, dot: 4.08, auto_ces: 4.75, theta: 5.33

2. Head-to-head 승률: 2/12 = 16.7%

3. Regime-shift 특화 데이터:
   - regimeShift: dot 5.85% > regime_switch 5.88% (2위)
   - multiRegime: mstl 3.50% > regime_switch 4.44% (4위)
   - suddenShift: mstl 4.55% > regime_switch 4.73% (2위)
   → 레짐 전환 데이터에서도 1위 달성 실패

4. multi_regime vs regime_switch:
   - multi_regime이 전반적으로 더 나음 (3.17 vs 3.50)
   - 특히 trending (1.07 vs 4.38), volatile (0.39 vs 0.93)에서 큰 차이
   - multi_regime은 validation으로 모델 선택, regime_switch는 규칙 기반

5. 1위 달성 케이스:
   - stationary: regime_switch/multi_regime 1.77% (theta 선택, 정작 theta 직접은 2.34%)
     → 마지막 레짐 데이터만 사용해 theta가 더 잘 동작

6. 모델 선택 패턴:
   - mstl이 계절성 데이터에서 압도적 (energyUsage, multiRegime, retailSales 등)
   - 레짐 모델이 mstl을 선택하면 mstl과 거의 동일하지만 약간 손실 (오버헤드)

결론: REJECTED
- 가설 1 기각: 레짐 전환 데이터에서도 단일 모델(mstl, dot)보다 못함
- 가설 3 부분 확인: stationary에서 마지막 레짐만 사용하니 theta 개선 (2.34→1.77)
- 근본 문제: 모델 선택이 정확해도 "마지막 레짐 데이터만" 쓰면 학습 데이터 감소
  → 레짐 경계가 애매한 경우 오히려 정보 손실
- multi_regime이 regime_switch보다 나은 이유: validation 기반 > 규칙 기반
- 차후 연구: "레짐별 데이터 가중치" 방식이 "데이터 절단"보다 유리할 수 있음

==============================================================================
실험일: 2026-02-28
"""

import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'
warnings.filterwarnings('ignore')

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from vectrix.engine.arima import ARIMAModel
from vectrix.engine.ces import AutoCES
from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.mstl import AutoMSTL
from vectrix.engine.theta import OptimizedTheta
from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS


class RegimeSwitchingForecaster:
    """
    Regime-Switching Forecaster

    변동점을 감지하여 마지막 레짐의 데이터만으로 모델 학습.
    레짐 특성에 따라 적절한 모델을 자동 선택.
    """

    def __init__(self, minRegimeLen: int = 30, sensitivity: float = 2.0):
        self.minRegimeLen = minRegimeLen
        self.sensitivity = sensitivity
        self.model = None
        self.regimeStart = 0
        self.regimeData = None
        self.fitted = False
        self.residuals = None
        self.selectedModel = None

    def _detectChangepoints(self, y: np.ndarray):
        """CUSUM 기반 변동점 감지."""
        n = len(y)
        if n < self.minRegimeLen * 2:
            return []

        changepoints = []
        globalMean = np.mean(y)
        cusum = np.zeros(n)
        cusumNeg = np.zeros(n)

        threshold = self.sensitivity * np.std(y)

        for i in range(1, n):
            cusum[i] = max(0, cusum[i-1] + (y[i] - globalMean) - threshold * 0.5)
            cusumNeg[i] = max(0, cusumNeg[i-1] - (y[i] - globalMean) - threshold * 0.5)

            if cusum[i] > threshold or cusumNeg[i] > threshold:
                if not changepoints or (i - changepoints[-1]) >= self.minRegimeLen:
                    changepoints.append(i)
                    cusum[i] = 0
                    cusumNeg[i] = 0
                    globalMean = np.mean(y[i:min(i + self.minRegimeLen, n)])

        return changepoints

    def _analyzeRegime(self, data: np.ndarray):
        """레짐의 통계적 특성 분석."""
        n = len(data)
        stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'cv': np.std(data) / (abs(np.mean(data)) + 1e-10),
            'length': n,
        }

        if n >= 5:
            t = np.arange(n, dtype=np.float64)
            slope = np.polyfit(t, data, 1)[0]
            stats['trendStrength'] = abs(slope * n) / (np.max(data) - np.min(data) + 1e-10)
        else:
            stats['trendStrength'] = 0

        if n >= 14:
            period = 7
            seasonalMeans = []
            for i in range(period):
                vals = data[i::period]
                if len(vals) > 0:
                    seasonalMeans.append(np.mean(vals))
            if seasonalMeans and np.var(data) > 1e-10:
                stats['seasonalStrength'] = np.var(seasonalMeans) / np.var(data)
            else:
                stats['seasonalStrength'] = 0
        else:
            stats['seasonalStrength'] = 0

        return stats

    def _selectModel(self, regimeStats: dict):
        """레짐 특성에 따라 최적 모델 선택."""
        if regimeStats['length'] < 20:
            return 'arima', lambda: ARIMAModel()

        if regimeStats['seasonalStrength'] > 0.2:
            return 'mstl', lambda: AutoMSTL()

        if regimeStats['trendStrength'] > 0.5:
            return 'theta', lambda: OptimizedTheta()

        if regimeStats['cv'] < 0.05:
            return 'arima', lambda: ARIMAModel()

        return 'dot', lambda: DynamicOptimizedTheta()

    def fit(self, y: np.ndarray) -> 'RegimeSwitchingForecaster':
        n = len(y)

        changepoints = self._detectChangepoints(y)

        if changepoints:
            lastCp = changepoints[-1]
            if n - lastCp >= self.minRegimeLen:
                self.regimeStart = lastCp
            elif len(changepoints) >= 2:
                self.regimeStart = changepoints[-2]
            else:
                self.regimeStart = 0
        else:
            self.regimeStart = 0

        self.regimeData = y[self.regimeStart:]

        if len(self.regimeData) < 10:
            self.regimeData = y[max(0, n - 60):]

        regimeStats = self._analyzeRegime(self.regimeData)
        self.selectedModel, modelFactory = self._selectModel(regimeStats)

        self.model = modelFactory()
        self.model.fit(self.regimeData)

        trainPred, _, _ = self.model.predict(len(self.regimeData))
        if len(trainPred) >= len(self.regimeData):
            self.residuals = self.regimeData - trainPred[:len(self.regimeData)]
        else:
            self.residuals = np.zeros(len(self.regimeData))

        self.fitted = True
        return self

    def predict(self, steps: int):
        if not self.fitted:
            raise ValueError("Model not fitted")

        pred, lower, upper = self.model.predict(steps)
        return pred[:steps], lower[:steps], upper[:steps]


class MultiRegimeForecaster:
    """
    Multi-Regime Forecaster

    여러 모델로 예측하고, 마지막 레짐에서 가장 잘 맞았던 모델을 선택.
    """

    def __init__(self, minRegimeLen: int = 30, sensitivity: float = 2.0):
        self.minRegimeLen = minRegimeLen
        self.sensitivity = sensitivity
        self.bestModel = None
        self.fitted = False
        self.residuals = None
        self.selectedModel = None

    def fit(self, y: np.ndarray) -> 'MultiRegimeForecaster':
        n = len(y)

        rsf = RegimeSwitchingForecaster(self.minRegimeLen, self.sensitivity)
        cps = rsf._detectChangepoints(y)

        if cps:
            lastCp = cps[-1]
            if n - lastCp >= self.minRegimeLen:
                regimeStart = lastCp
            else:
                regimeStart = max(0, n - 60)
        else:
            regimeStart = max(0, n - int(n * 0.3))

        regimeData = y[regimeStart:]
        if len(regimeData) < 20:
            regimeData = y[max(0, n - 60):]

        valSize = min(14, len(regimeData) // 3)
        if valSize < 3:
            valSize = min(len(regimeData) - 10, 7)
        if valSize < 2:
            self.bestModel = ARIMAModel()
            self.bestModel.fit(y)
            self.selectedModel = 'arima'
            self.residuals = np.zeros(len(y))
            self.fitted = True
            return self

        trainPart = regimeData[:len(regimeData) - valSize]
        valPart = regimeData[len(regimeData) - valSize:]

        candidates = {
            'arima': lambda: ARIMAModel(),
            'theta': lambda: OptimizedTheta(),
            'mstl': lambda: AutoMSTL(),
            'auto_ces': lambda: AutoCES(),
            'dot': lambda: DynamicOptimizedTheta(),
        }

        bestMape = np.inf
        bestName = 'arima'

        for name, factory in candidates.items():
            try:
                model = factory()
                model.fit(trainPart)
                pred, _, _ = model.predict(valSize)
                pred = pred[:len(valPart)]

                mask = np.abs(valPart) > 1e-8
                if mask.sum() > 0:
                    mape = np.mean(np.abs((valPart[mask] - pred[mask]) / valPart[mask]))
                else:
                    mape = np.inf

                if mape < bestMape:
                    bestMape = mape
                    bestName = name
            except Exception:
                continue

        self.bestModel = candidates[bestName]()
        self.bestModel.fit(regimeData)
        self.selectedModel = bestName

        trainPred, _, _ = self.bestModel.predict(len(regimeData))
        if len(trainPred) >= len(regimeData):
            self.residuals = regimeData - trainPred[:len(regimeData)]
        else:
            self.residuals = np.zeros(len(regimeData))

        self.fitted = True
        return self

    def predict(self, steps: int):
        if not self.fitted:
            raise ValueError("Model not fitted")

        pred, lower, upper = self.bestModel.predict(steps)
        return pred[:steps], lower[:steps], upper[:steps]


def _modelFactory(name: str):
    factories = {
        'arima': lambda: ARIMAModel(),
        'theta': lambda: OptimizedTheta(),
        'mstl': lambda: AutoMSTL(),
        'auto_ces': lambda: AutoCES(),
        'dot': lambda: DynamicOptimizedTheta(),
        'regime_switch': lambda: RegimeSwitchingForecaster(),
        'multi_regime': lambda: MultiRegimeForecaster(),
    }
    return factories.get(name, lambda: ARIMAModel())()


def _computeMetrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    mask = np.abs(actual) > 1e-8
    if mask.sum() == 0:
        return np.nan, np.nan
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mape, rmse


def _generateMultiRegime(n: int = 365, seed: int = 42):
    """3-4개 레짐이 있는 복잡한 시계열."""
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    cps = sorted(rng.choice(range(n // 5, 4 * n // 5), size=3, replace=False))

    values[:cps[0]] = 100 + 0.5 * np.arange(cps[0]) + rng.normal(0, 5, cps[0])
    values[cps[0]:cps[1]] = 200 + rng.normal(0, 20, cps[1] - cps[0])
    values[cps[1]:cps[2]] = 50 + 0.3 * np.arange(cps[2] - cps[1]) + rng.normal(0, 3, cps[2] - cps[1])
    values[cps[2]:] = 150 + 10 * np.sin(2 * np.pi * np.arange(n - cps[2]) / 7) + rng.normal(0, 8, n - cps[2])

    return values


def _generateSuddenShift(n: int = 365, seed: int = 42):
    """급격한 레벨 변화."""
    rng = np.random.default_rng(seed)
    cp = n * 3 // 4
    values = np.zeros(n)
    values[:cp] = 1000 + 50 * np.sin(2 * np.pi * np.arange(cp) / 7) + rng.normal(0, 30, cp)
    values[cp:] = 500 + 50 * np.sin(2 * np.pi * np.arange(n - cp) / 7) + rng.normal(0, 30, n - cp)
    return values


def main():
    print("=" * 70)
    print("E030: Regime-Switching Forecaster")
    print("=" * 70)

    datasets = {}
    for name, genFunc in ALL_GENERATORS.items():
        if name == 'intermittentDemand':
            continue
        if name == 'multiSeasonalRetail':
            df = genFunc(n=730, seed=42)
        elif name == 'stockPrice':
            df = genFunc(n=252, seed=42)
        else:
            df = genFunc(n=365, seed=42)
        datasets[name] = df['value'].values.astype(np.float64)

    datasets['multiRegime'] = _generateMultiRegime(365, seed=42)
    datasets['suddenShift'] = _generateSuddenShift(365, seed=42)

    allModels = ['arima', 'theta', 'mstl', 'auto_ces', 'dot',
                 'regime_switch', 'multi_regime']
    horizon = 14

    results = {}

    for dataName, values in datasets.items():
        results[dataName] = {}
        trainData = values[:len(values) - horizon]
        testData = values[len(values) - horizon:]

        for modelName in allModels:
            try:
                model = _modelFactory(modelName)
                model.fit(trainData)
                pred, _, _ = model.predict(horizon)
                mape, rmse = _computeMetrics(testData, pred[:len(testData)])
                results[dataName][modelName] = {'mape': mape, 'rmse': rmse}

                if modelName in ['regime_switch', 'multi_regime']:
                    selModel = getattr(model, 'selectedModel', 'unknown')
                    results[dataName][modelName]['selected'] = selModel
            except Exception as e:
                results[dataName][modelName] = {'mape': np.nan, 'rmse': np.nan, 'error': str(e)}

    print("\n" + "=" * 70)
    print("ANALYSIS 1: Rankings per Dataset")
    print("=" * 70)

    rankSums = {m: 0 for m in allModels}
    rankCounts = {m: 0 for m in allModels}

    for dataName in sorted(datasets.keys()):
        dataResults = results[dataName]
        validModels = [(m, dataResults[m]['mape']) for m in allModels
                       if not np.isnan(dataResults.get(m, {}).get('mape', np.nan))
                       and dataResults.get(m, {}).get('mape', np.inf) < 1e6]

        if not validModels:
            continue

        validModels.sort(key=lambda x: x[1])

        print(f"\n  [{dataName}]")
        for rank, (m, mape) in enumerate(validModels):
            marker = "***" if m in ['regime_switch', 'multi_regime'] else "   "
            sel = ""
            if m in ['regime_switch', 'multi_regime']:
                sel = f" (selected: {dataResults[m].get('selected', '?')})"
            print(f"    {rank+1}. {m:20s} MAPE={mape:10.2f}% {marker}{sel}")
            rankSums[m] += rank + 1
            rankCounts[m] += 1

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Average Rank")
    print("=" * 70)

    avgRanks = []
    for m in allModels:
        if rankCounts[m] > 0:
            avgRanks.append((m, rankSums[m] / rankCounts[m], rankCounts[m]))
    avgRanks.sort(key=lambda x: x[1])

    for m, rank, count in avgRanks:
        marker = " ***" if m in ['regime_switch', 'multi_regime'] else ""
        print(f"  {m:<20s} | Avg Rank = {rank:.2f} | n = {count}{marker}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Regime-shift datasets focus")
    print("=" * 70)

    regimeDatasets = ['regimeShift', 'multiRegime', 'suddenShift']

    for dataName in regimeDatasets:
        if dataName not in results:
            continue
        dataResults = results[dataName]
        validModels = [(m, dataResults[m]['mape']) for m in allModels
                       if not np.isnan(dataResults.get(m, {}).get('mape', np.nan))
                       and dataResults.get(m, {}).get('mape', np.inf) < 1e6]
        validModels.sort(key=lambda x: x[1])

        print(f"\n  [{dataName}]")
        for rank, (m, mape) in enumerate(validModels):
            print(f"    {rank+1}. {m:20s} MAPE={mape:10.2f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Model Selection by Regime")
    print("=" * 70)

    for dataName in sorted(datasets.keys()):
        for mName in ['regime_switch', 'multi_regime']:
            sel = results[dataName].get(mName, {}).get('selected', '?')
            mape = results[dataName].get(mName, {}).get('mape', np.nan)
            mapeStr = f"{mape:.2f}" if not np.isnan(mape) else "N/A"
            print(f"  {dataName:25s} | {mName:15s} selected: {sel:10s} | MAPE: {mapeStr}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    regimeWins = 0
    singleWins = 0
    for dataName in datasets:
        dataResults = results[dataName]
        singleMapes = [dataResults[m]['mape'] for m in ['arima', 'theta', 'mstl', 'auto_ces', 'dot']
                       if not np.isnan(dataResults.get(m, {}).get('mape', np.nan))
                       and dataResults.get(m, {}).get('mape', np.inf) < 1e6]
        regimeMapes = [dataResults[m]['mape'] for m in ['regime_switch', 'multi_regime']
                       if not np.isnan(dataResults.get(m, {}).get('mape', np.nan))
                       and dataResults.get(m, {}).get('mape', np.inf) < 1e6]

        if singleMapes and regimeMapes:
            if min(regimeMapes) < min(singleMapes) * 0.99:
                regimeWins += 1
            else:
                singleWins += 1

    print(f"""
    Regime model wins: {regimeWins}/{regimeWins + singleWins}
    Single model wins: {singleWins}/{regimeWins + singleWins}
    Win rate: {regimeWins / max(regimeWins + singleWins, 1) * 100:.1f}%
    """)


if __name__ == '__main__':
    main()
