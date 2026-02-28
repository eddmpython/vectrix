"""
==============================================================================
실험 ID: adaptiveInterval/003
실험명: GARCH 후처리 z-보정으로 Coverage 0.95 달성
==============================================================================

목적:
- 001-002에서 GARCH 후처리 구간이 Winkler 최적이나 Coverage 0.809 (under-coverage)
- z=1.96 → z 보정(2.0~2.5 범위)으로 Coverage 0.95 목표 달성
- 최적 z 탐색 + Coverage/Winkler 트레이드오프 분석

가설:
1. z를 2.2~2.3으로 올리면 Coverage가 0.95에 근접
2. z 보정 후에도 GARCH 구간이 고정 구간보다 Winkler 우수
3. 최적 z는 데이터 특성(변동성, 분산 이질성)에 따라 다름

방법:
1. 합성 데이터 10종 x 6개 모델 = 60건
2. 각 건에서 GARCH 후처리 구간을 z=1.5~3.0 (0.1 간격) 16개 수준으로 계산
3. Coverage, Width, Winkler Score 측정
4. 최적 z 탐색: Coverage >= 0.95이면서 Winkler 최소
5. 고정 구간 (z=1.96) 대비 개선 측정

성공 기준:
- 최적 z에서 Coverage 0.92~0.97 달성
- 최적 z GARCH Winkler < 고정 구간 Winkler
- 일관된 최적 z 범위 식별

==============================================================================
결과 (실험 후 작성)
==============================================================================

수치 (50건):
| 지표 | 값 |
|------|-----|
| z=1.5 Coverage | 0.924, Winkler 563 |
| z=1.6 Coverage | 0.931, Winkler 570 (GARCH wins: 29/50, 58%) |
| z=1.96 Coverage | ~0.946, Winkler ~596 |
| z=2.3 Coverage | 0.950, Winkler 626 |
| Fixed@z=1.96 Coverage | 0.940, Winkler 490 |

핵심 발견:
1. 가설 1 부분 채택: z=2.3에서 Coverage 0.950 달성 가능
2. 가설 2 기각: GARCH Winkler(570~626) > Fixed Winkler(482~505) — GARCH가 나쁨
3. 가설 3 채택: 데이터별 최적 z 범위 1.5~2.7로 분산
4. manufacturing에서 Coverage 0.714 — GARCH 조건부 분산이 주기적 드롭 포착 실패
5. z=1.5~1.6이 Coverage ≥ 0.93 최소 Winkler

001-002과의 차이 분석:
- 001-002: GARCH Winkler 1431 vs Fixed 2072 (GARCH 우위)
- 003: GARCH Winkler 570 vs Fixed 482 (Fixed 우위)
- 원인: 잔차 계산 방식 차이 + 데이터셋 차이 + horizon 효과
- 001-002은 rolling out-of-sample 잔차, 003은 in-sample 잔차

결론: 보류
- GARCH 후처리의 Winkler 우위가 재현되지 않음
- z 보정 자체는 작동하지만, 기본 GARCH 구간이 Fixed보다 나쁜 상황
- 근본 원인: GARCH(1,1) 파라미터(alpha=0.1, beta=0.85) 고정 → 데이터별 최적화 필요
- 또는 rolling CV 기반 잔차로 재실험 필요

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


def _fitGarch(residuals: np.ndarray, omega: float = None, alphaG: float = 0.1, betaG: float = 0.85):
    """잔차에 GARCH(1,1) 적합, 조건부 분산 시리즈 반환."""
    n = len(residuals)
    if omega is None:
        omega = np.var(residuals) * 0.05

    sigma2 = np.zeros(n)
    sigma2[0] = np.var(residuals)

    for t in range(1, n):
        sigma2[t] = omega + alphaG * residuals[t - 1] ** 2 + betaG * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-10)

    return sigma2


def _forecastGarchVariance(sigma2Last: float, residLast: float,
                            omega: float, alphaG: float, betaG: float, horizon: int):
    """GARCH 조건부 분산 h-step 예측."""
    forecastSigma2 = np.zeros(horizon)
    forecastSigma2[0] = omega + alphaG * residLast ** 2 + betaG * sigma2Last

    for h in range(1, horizon):
        forecastSigma2[h] = omega + (alphaG + betaG) * forecastSigma2[h - 1]
        forecastSigma2[h] = max(forecastSigma2[h], 1e-10)

    return forecastSigma2


def _computeWinkler(actual, lower, upper, alpha=0.05):
    """Winkler Score: 좁은 구간 + 커버리지 보상."""
    n = len(actual)
    scores = np.zeros(n)

    for i in range(n):
        width = upper[i] - lower[i]
        if actual[i] < lower[i]:
            scores[i] = width + (2.0 / alpha) * (lower[i] - actual[i])
        elif actual[i] > upper[i]:
            scores[i] = width + (2.0 / alpha) * (actual[i] - upper[i])
        else:
            scores[i] = width

    return np.mean(scores)


def _computeCoverage(actual, lower, upper):
    """실제 값이 구간 안에 들어가는 비율."""
    inside = np.sum((actual >= lower) & (actual <= upper))
    return inside / len(actual)


def _modelFactory(modelName: str):
    factories = {
        'arima': lambda: ARIMAModel(),
        'theta': lambda: OptimizedTheta(),
        'mstl': lambda: AutoMSTL(),
        'auto_ces': lambda: AutoCES(),
        'dot': lambda: DynamicOptimizedTheta(),
    }
    return factories.get(modelName, lambda: ARIMAModel())()


def _runSingleExperiment(data: np.ndarray, modelName: str, horizon: int):
    """단일 (데이터, 모델) 조합에서 z 스캔."""
    n = len(data)
    trainSize = n - horizon
    if trainSize < 30:
        return None

    trainData = data[:trainSize]
    testData = data[trainSize:]

    model = _modelFactory(modelName)
    model.fit(trainData)
    pred, _, _ = model.predict(horizon)
    pred = pred[:len(testData)]

    trainPredModel = _modelFactory(modelName)
    trainPredModel.fit(trainData)
    trainPred, _, _ = trainPredModel.predict(trainSize)

    if len(trainPred) >= trainSize:
        residuals = trainData - trainPred[:trainSize]
    else:
        residuals = trainData[len(trainPred):] - trainPred[-len(trainData[len(trainPred):]):]
        if len(residuals) < 10:
            residuals = trainData - np.mean(trainData)

    residStd = np.std(residuals)
    if residStd < 1e-10:
        return None

    omegaVal = np.var(residuals) * 0.05
    alphaG = 0.1
    betaG = 0.85

    sigma2 = _fitGarch(residuals, omega=omegaVal, alphaG=alphaG, betaG=betaG)
    forecastSigma2 = _forecastGarchVariance(
        sigma2[-1], residuals[-1], omegaVal, alphaG, betaG, horizon
    )

    zValues = np.arange(1.5, 3.05, 0.1)
    results = []

    for z in zValues:
        garchMargin = z * np.sqrt(forecastSigma2[:len(testData)])
        garchLower = pred - garchMargin
        garchUpper = pred + garchMargin

        garchCov = _computeCoverage(testData, garchLower, garchUpper)
        garchWinkler = _computeWinkler(testData, garchLower, garchUpper)
        garchWidth = np.mean(garchUpper - garchLower)

        fixedMargin = z * residStd
        fixedLower = pred - fixedMargin
        fixedUpper = pred + fixedMargin

        fixedCov = _computeCoverage(testData, fixedLower, fixedUpper)
        fixedWinkler = _computeWinkler(testData, fixedLower, fixedUpper)
        fixedWidth = np.mean(fixedUpper - fixedLower)

        results.append({
            'z': z,
            'garchCov': garchCov,
            'garchWinkler': garchWinkler,
            'garchWidth': garchWidth,
            'fixedCov': fixedCov,
            'fixedWinkler': fixedWinkler,
            'fixedWidth': fixedWidth,
        })

    return {
        'modelName': modelName,
        'zResults': results,
        'residStd': residStd,
        'garchVolRatio': np.std(np.sqrt(forecastSigma2)) / np.mean(np.sqrt(forecastSigma2)) if np.mean(np.sqrt(forecastSigma2)) > 0 else 0,
    }


def main():
    print("=" * 70)
    print("E027: GARCH z-Correction for Coverage 0.95")
    print("=" * 70)

    datasets = {}
    for name, genFunc in ALL_GENERATORS.items():
        if name == 'intermittentDemand':
            continue
        if name == 'multiSeasonalRetail':
            datasets[name] = genFunc(n=730, seed=42)
        elif name == 'stockPrice':
            datasets[name] = genFunc(n=252, seed=42)
        else:
            datasets[name] = genFunc(n=365, seed=42)

    modelNames = ['arima', 'theta', 'mstl', 'auto_ces', 'dot']
    horizon = 14

    allExperiments = []

    for dataName, df in datasets.items():
        values = df['value'].values.astype(np.float64)

        for modelName in modelNames:
            try:
                result = _runSingleExperiment(values, modelName, horizon)
                if result is None:
                    continue
                result['dataName'] = dataName
                allExperiments.append(result)
            except Exception as e:
                print(f"  [!] {dataName}/{modelName}: {e}")

    if not allExperiments:
        print("No results!")
        return

    print(f"\n--- {len(allExperiments)} experiments completed ---\n")

    print("=" * 70)
    print("ANALYSIS 1: Coverage vs z (averaged across all experiments)")
    print("=" * 70)

    zValues = np.arange(1.5, 3.05, 0.1)

    print(f"\n  {'z':>5s} | {'GARCH Cov':>10s} | {'Fixed Cov':>10s} | {'GARCH Winkler':>14s} | {'Fixed Winkler':>14s} | {'GARCH wins':>10s}")
    print("  " + "-" * 75)

    bestZ = None
    bestWinkler = float('inf')

    for zIdx, z in enumerate(zValues):
        garchCovs = []
        fixedCovs = []
        garchWinklers = []
        fixedWinklers = []

        for exp in allExperiments:
            if zIdx < len(exp['zResults']):
                r = exp['zResults'][zIdx]
                garchCovs.append(r['garchCov'])
                fixedCovs.append(r['fixedCov'])
                garchWinklers.append(r['garchWinkler'])
                fixedWinklers.append(r['fixedWinkler'])

        if not garchCovs:
            continue

        avgGarchCov = np.mean(garchCovs)
        avgFixedCov = np.mean(fixedCovs)
        avgGarchWinkler = np.mean(garchWinklers)
        avgFixedWinkler = np.mean(fixedWinklers)
        garchWins = sum(1 for gw, fw in zip(garchWinklers, fixedWinklers) if gw < fw)

        marker = " <--" if 0.93 <= avgGarchCov <= 0.97 else ""
        print(f"  {z:5.1f} | {avgGarchCov:10.3f} | {avgFixedCov:10.3f} | "
              f"{avgGarchWinkler:14.1f} | {avgFixedWinkler:14.1f} | "
              f"{garchWins}/{len(garchWinklers)}{marker}")

        if avgGarchCov >= 0.93 and avgGarchWinkler < bestWinkler:
            bestZ = z
            bestWinkler = avgGarchWinkler

    if bestZ:
        print(f"\n  >>> Best z = {bestZ:.1f} (Coverage >= 0.93, min Winkler = {bestWinkler:.1f})")

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Optimal z per dataset")
    print("=" * 70)

    datasetOptimalZ = {}
    for exp in allExperiments:
        dataName = exp['dataName']
        if dataName not in datasetOptimalZ:
            datasetOptimalZ[dataName] = {'zResults': [], 'bestZ': None, 'bestWinkler': float('inf')}

        for r in exp['zResults']:
            datasetOptimalZ[dataName]['zResults'].append(r)

    for dataName in sorted(datasetOptimalZ.keys()):
        dResults = datasetOptimalZ[dataName]['zResults']
        for z in zValues:
            zSpecific = [r for r in dResults if abs(r['z'] - z) < 0.01]
            if not zSpecific:
                continue
            avgCov = np.mean([r['garchCov'] for r in zSpecific])
            avgWinkler = np.mean([r['garchWinkler'] for r in zSpecific])
            if avgCov >= 0.90 and avgWinkler < datasetOptimalZ[dataName]['bestWinkler']:
                datasetOptimalZ[dataName]['bestZ'] = z
                datasetOptimalZ[dataName]['bestWinkler'] = avgWinkler

        bz = datasetOptimalZ[dataName].get('bestZ', 'N/A')
        bw = datasetOptimalZ[dataName].get('bestWinkler', 'N/A')
        zSpecific196 = [r for r in dResults if abs(r['z'] - 1.96) < 0.06]
        cov196 = np.mean([r['garchCov'] for r in zSpecific196]) if zSpecific196 else 'N/A'
        bwStr = f"{bw:.1f}" if isinstance(bw, float) else str(bw)
        covStr = f"{cov196:.3f}" if isinstance(cov196, float) else str(cov196)
        print(f"  {dataName:25s} | optimal z = {bz} | Winkler = {bwStr} | Coverage@1.96 = {covStr}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: GARCH vs Fixed at optimal z")
    print("=" * 70)

    if bestZ:
        zIdx = int(round((bestZ - 1.5) / 0.1))
        garchWins = 0
        total = 0

        for exp in allExperiments:
            if zIdx < len(exp['zResults']):
                r = exp['zResults'][zIdx]
                total += 1
                if r['garchWinkler'] < r['fixedWinkler']:
                    garchWins += 1

        print(f"\n  At z = {bestZ:.1f}:")
        print(f"    GARCH wins: {garchWins}/{total} ({garchWins/total*100:.1f}%)")

        avgGarchW = np.mean([exp['zResults'][zIdx]['garchWinkler'] for exp in allExperiments if zIdx < len(exp['zResults'])])
        avgFixedW = np.mean([exp['zResults'][zIdx]['fixedWinkler'] for exp in allExperiments if zIdx < len(exp['zResults'])])
        avgGarchC = np.mean([exp['zResults'][zIdx]['garchCov'] for exp in allExperiments if zIdx < len(exp['zResults'])])
        avgFixedC = np.mean([exp['zResults'][zIdx]['fixedCov'] for exp in allExperiments if zIdx < len(exp['zResults'])])

        print(f"    GARCH: Winkler = {avgGarchW:.1f}, Coverage = {avgGarchC:.3f}")
        print(f"    Fixed: Winkler = {avgFixedW:.1f}, Coverage = {avgFixedC:.3f}")
        print(f"    Winkler improvement: {(avgFixedW - avgGarchW) / avgFixedW * 100:.1f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Model-level optimal z")
    print("=" * 70)

    for modelName in modelNames:
        modelExps = [e for e in allExperiments if e['modelName'] == modelName]
        if not modelExps:
            continue

        modelBestZ = None
        modelBestWinkler = float('inf')

        for zIdx, z in enumerate(zValues):
            covs = []
            winklers = []
            for exp in modelExps:
                if zIdx < len(exp['zResults']):
                    covs.append(exp['zResults'][zIdx]['garchCov'])
                    winklers.append(exp['zResults'][zIdx]['garchWinkler'])

            if covs and np.mean(covs) >= 0.93 and np.mean(winklers) < modelBestWinkler:
                modelBestZ = z
                modelBestWinkler = np.mean(winklers)

        if modelBestZ:
            print(f"  {modelName:10s} | optimal z = {modelBestZ:.1f} | Winkler = {modelBestWinkler:.1f}")
        else:
            print(f"  {modelName:10s} | no z achieves Coverage >= 0.93")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    optimalZValues = [datasetOptimalZ[d].get('bestZ') for d in datasetOptimalZ if datasetOptimalZ[d].get('bestZ')]
    if optimalZValues:
        print(f"\n  Optimal z range: {min(optimalZValues):.1f} ~ {max(optimalZValues):.1f}")
        print(f"  Median optimal z: {np.median(optimalZValues):.1f}")
        print(f"  Global best z: {bestZ:.1f}" if bestZ else "  No global best z found")


if __name__ == '__main__':
    main()
