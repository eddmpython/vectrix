"""
==============================================================================
실험 ID: frequencyDomain/002
실험명: Spectral Denoising (주파수 도메인 잡음 제거)
==============================================================================

목적:
- FFT로 시계열에서 고주파 노이즈를 제거한 뒤 모델 학습 → 예측 정확도 개선
- 001에서 "잔차 보정"이 과적합 → 반대 방향: 입력 데이터 정제
- Spectral Threshold: 유의미한 주기 성분만 보존, 나머지 제거

가설:
1. 고주파 노이즈 제거 후 모델 학습 시 MAPE 5%+ 개선
2. 노이즈 비율이 높은 시계열(volatile, stationary)에서 효과 최대
3. 강한 계절성 시계열에서는 효과 미미 (이미 신호가 강함)
4. 최적 필터 차단 주파수가 존재 (에너지 80~95% 보존)

방법:
1. 원본 시계열에 FFT 적용
2. 에너지 임계값(70%, 80%, 90%, 95%)으로 주파수 성분 필터링
3. 역FFT로 denoised 시계열 복원
4. 원본 vs denoised로 각각 모델 학습 → 동일 horizon 예측
5. MAPE/RMSE 비교
6. 최적 에너지 임계값 탐색

성공 기준:
- Denoised 모델 승률 > 55%
- 평균 MAPE 개선 > 3%
- 최적 에너지 임계값 식별

==============================================================================
결과 (실험 후 작성)
==============================================================================

수치 (200건, 유효 195건):
| 지표 | 값 |
|------|-----|
| Denoised 승률 | 80/195 (41.0%) |
| Adaptive 승률 | 89/195 (45.6%) |
| Avg MAPE improvement | +19.1% (temperature 이상치 포함) |

카테고리별 승률:
| 카테고리 | Denoised 승률 | Adaptive 승률 | 비고 |
|----------|---------------|---------------|------|
| Seasonal | 35/60 (58.3%) | 36/60 (60.0%) | multiSeasonalRetail 5/5 전승 |
| Noisy/Volatile | 20/60 (33.3%) | 24/60 (40.0%) | volatile 5/5 전승 |
| Trending | 11/40 (27.5%) | 4/40 (10.0%) | 추세 정보 손실 → 악화 |
| Other | 14/35 (40.0%) | 25/35 (71.4%) | regimeShift에서 강함 |

모델별 (adaptive 승률):
| 모델 | Adaptive 승률 | 비고 |
|------|---------------|------|
| auto_ces | 69.2% | 최대 수혜 |
| dot | 59.0% | - |
| mstl | 41.0% | 이미 계절성 포착 |
| theta | 38.5% | - |
| arima | 20.5% | 최소 수혜 |

임계값: 70%는 과도, 80~95% 유사. 80%가 최적 절충.

핵심 발견:
1. 가설 1 부분 채택: Seasonal + Volatile에서 MAPE 개선 확인
2. 가설 2 부분 기각: volatile은 전승이나 stationary는 효과 없음
3. 가설 3 기각: 계절성 데이터에서 오히려 가장 큰 효과 (58.3%)
4. 가설 4 채택: 80% 에너지 보존이 최적
5. Trending 데이터에서 심각한 악화 — 추세 정보 손실
6. Adaptive denoising이 fixed보다 안정적 (45.6% vs 41.0%)

결론: 조건부 채택
- 계절성 + 변동성 데이터 한정으로 적용 가능
- DNA 특성으로 "계절성 강도"를 판단하여 조건부 적용
- Trending/Stationary 데이터에서는 비활성화 필수
- auto_ces, dot 모델에서 특히 효과적
- 추세 보존 로직 추가 필요 (현재 linear detrend만)

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
from vectrix.engine.ets import ETSModel
from vectrix.engine.mstl import AutoMSTL
from vectrix.engine.theta import OptimizedTheta
from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS


def _spectralDenoise(data: np.ndarray, energyThreshold: float = 0.9) -> np.ndarray:
    """FFT 기반 spectral denoising. 에너지 임계값까지의 주파수 성분만 보존."""
    n = len(data)
    if n < 10:
        return data.copy()

    trend = np.linspace(data[0], data[-1], n)
    detrended = data - trend

    fft = np.fft.rfft(detrended)
    magnitudes = np.abs(fft)
    power = magnitudes ** 2
    totalPower = np.sum(power)

    if totalPower < 1e-10:
        return data.copy()

    sortedIdx = np.argsort(power)[::-1]
    cumulativePower = 0.0
    keepMask = np.zeros(len(fft), dtype=bool)

    for idx in sortedIdx:
        keepMask[idx] = True
        cumulativePower += power[idx]
        if cumulativePower / totalPower >= energyThreshold:
            break

    filteredFft = fft * keepMask
    denoised = np.fft.irfft(filteredFft, n=n)

    return denoised + trend


def _adaptiveSpectralDenoise(data: np.ndarray) -> np.ndarray:
    """적응형 denoising: 노이즈 수준에 따라 임계값 자동 결정."""
    n = len(data)
    if n < 20:
        return data.copy()

    trend = np.linspace(data[0], data[-1], n)
    detrended = data - trend

    fft = np.fft.rfft(detrended)
    magnitudes = np.abs(fft)
    power = magnitudes ** 2

    if np.sum(power) < 1e-10:
        return data.copy()

    nFreqs = len(magnitudes)
    highFreqPower = np.sum(power[nFreqs * 3 // 4:])
    totalPower = np.sum(power)
    noiseRatio = highFreqPower / totalPower if totalPower > 0 else 0

    if noiseRatio > 0.3:
        threshold = 0.80
    elif noiseRatio > 0.1:
        threshold = 0.90
    else:
        threshold = 0.97

    return _spectralDenoise(data, energyThreshold=threshold)


def _modelFactory(modelName: str):
    """모델 팩토리."""
    factories = {
        'ets': lambda: ETSModel(),
        'arima': lambda: ARIMAModel(),
        'theta': lambda: OptimizedTheta(),
        'mstl': lambda: AutoMSTL(),
        'auto_ces': lambda: AutoCES(),
        'dot': lambda: DynamicOptimizedTheta(),
    }
    return factories.get(modelName, lambda: ETSModel())()


def _computeMetrics(actual, predicted):
    """MAPE, RMSE 계산."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    mask = np.abs(actual) > 1e-8
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mape, rmse


def _runSingleExperiment(data: np.ndarray, modelName: str, horizon: int, energyThreshold: float):
    """단일 실험: 원본 vs denoised 모델."""
    n = len(data)
    trainSize = n - horizon
    if trainSize < 30:
        return None

    trainOrig = data[:trainSize]
    testData = data[trainSize:]

    modelOrig = _modelFactory(modelName)
    modelOrig.fit(trainOrig)
    predOrig, _, _ = modelOrig.predict(horizon)
    predOrig = predOrig[:len(testData)]
    mapeOrig, rmseOrig = _computeMetrics(testData, predOrig)

    trainDenoised = _spectralDenoise(trainOrig, energyThreshold)

    modelDen = _modelFactory(modelName)
    modelDen.fit(trainDenoised)
    predDen, _, _ = modelDen.predict(horizon)
    predDen = predDen[:len(testData)]
    mapeDen, rmseDen = _computeMetrics(testData, predDen)

    trainAdaptive = _adaptiveSpectralDenoise(trainOrig)
    modelAdapt = _modelFactory(modelName)
    modelAdapt.fit(trainAdaptive)
    predAdapt, _, _ = modelAdapt.predict(horizon)
    predAdapt = predAdapt[:len(testData)]
    mapeAdapt, rmseAdapt = _computeMetrics(testData, predAdapt)

    fft = np.fft.rfft(trainOrig - np.linspace(trainOrig[0], trainOrig[-1], trainSize))
    power = np.abs(fft) ** 2
    nFreqs = len(power)
    highFreqPower = np.sum(power[nFreqs * 3 // 4:])
    totalPower = np.sum(power)
    noiseRatio = highFreqPower / totalPower if totalPower > 0 else 0

    return {
        'modelName': modelName,
        'energyThreshold': energyThreshold,
        'mapeOrig': mapeOrig,
        'rmseOrig': rmseOrig,
        'mapeDen': mapeDen,
        'rmseDen': rmseDen,
        'mapeAdapt': mapeAdapt,
        'rmseAdapt': rmseAdapt,
        'noiseRatio': noiseRatio,
    }


def main():
    print("=" * 70)
    print("E024: Spectral Denoising")
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
    thresholds = [0.70, 0.80, 0.90, 0.95]
    horizon = 14

    allResults = []

    print(f"\n--- {len(datasets)} datasets x {len(modelNames)} models x {len(thresholds)} thresholds ---\n")

    for dataName, df in datasets.items():
        values = df['value'].values.astype(np.float64)
        print(f"\n  [{dataName}]")

        for modelName in modelNames:
            for threshold in thresholds:
                try:
                    result = _runSingleExperiment(values, modelName, horizon, threshold)
                    if result is None:
                        continue
                    result['dataName'] = dataName
                    allResults.append(result)
                except Exception as e:
                    print(f"    [!] {modelName}/{threshold}: {e}")

            bestResult = None
            for r in allResults:
                if r['dataName'] == dataName and r['modelName'] == modelName:
                    if bestResult is None or r['mapeDen'] < bestResult['mapeDen']:
                        bestResult = r

            if bestResult:
                improved = bestResult['mapeDen'] < bestResult['mapeOrig']
                adaptImproved = bestResult['mapeAdapt'] < bestResult['mapeOrig']
                m1 = "+" if improved else "-"
                m2 = "+" if adaptImproved else "-"
                print(f"    [{m1}] {modelName:10s} | "
                      f"MAPE: {bestResult['mapeOrig']:8.2f} -> best_den: {bestResult['mapeDen']:8.2f} "
                      f"(adapt: [{m2}] {bestResult['mapeAdapt']:8.2f}) | "
                      f"noise: {bestResult['noiseRatio']:.3f}")

    if not allResults:
        print("\nNo results collected!")
        return

    print("\n" + "=" * 70)
    print("ANALYSIS 1: Overall Denoising Impact")
    print("=" * 70)

    nTotal = len(allResults)
    nImprovedDen = sum(1 for r in allResults if not np.isnan(r['mapeDen']) and not np.isnan(r['mapeOrig']) and r['mapeDen'] < r['mapeOrig'])
    nImprovedAdapt = sum(1 for r in allResults if not np.isnan(r['mapeAdapt']) and not np.isnan(r['mapeOrig']) and r['mapeAdapt'] < r['mapeOrig'])

    validResults = [r for r in allResults if not np.isnan(r['mapeOrig']) and not np.isnan(r['mapeDen'])
                    and r['mapeOrig'] < 1e6 and r['mapeDen'] < 1e6]

    if validResults:
        avgOrig = np.mean([r['mapeOrig'] for r in validResults])
        avgDen = np.mean([r['mapeDen'] for r in validResults])
        avgAdapt = np.mean([r['mapeAdapt'] for r in validResults if not np.isnan(r['mapeAdapt']) and r['mapeAdapt'] < 1e6])

        print(f"\nTotal experiments: {nTotal} ({len(validResults)} valid)")
        print(f"Denoised wins:  {nImprovedDen}/{nTotal} ({nImprovedDen/nTotal*100:.1f}%)")
        print(f"Adaptive wins:  {nImprovedAdapt}/{nTotal} ({nImprovedAdapt/nTotal*100:.1f}%)")
        print(f"\nAvg MAPE - Original:  {avgOrig:.2f}%")
        print(f"Avg MAPE - Denoised:  {avgDen:.2f}%")
        print(f"Avg MAPE - Adaptive:  {avgAdapt:.2f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS 2: By Energy Threshold")
    print("=" * 70)

    for threshold in thresholds:
        tResults = [r for r in validResults if r['energyThreshold'] == threshold]
        if not tResults:
            continue

        wins = sum(1 for r in tResults if r['mapeDen'] < r['mapeOrig'])
        avgOrig = np.mean([r['mapeOrig'] for r in tResults])
        avgDen = np.mean([r['mapeDen'] for r in tResults])
        improvement = (avgOrig - avgDen) / avgOrig * 100

        print(f"  Threshold {threshold:.0%}: wins = {wins}/{len(tResults)} ({wins/len(tResults)*100:.1f}%), "
              f"MAPE: {avgOrig:.2f} -> {avgDen:.2f} ({improvement:+.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: By Model")
    print("=" * 70)

    for modelName in modelNames:
        modelResults = [r for r in validResults if r['modelName'] == modelName]
        if not modelResults:
            continue

        wins = sum(1 for r in modelResults if r['mapeDen'] < r['mapeOrig'])
        adaptWins = sum(1 for r in modelResults if not np.isnan(r['mapeAdapt']) and r['mapeAdapt'] < r['mapeOrig'])
        avgOrig = np.mean([r['mapeOrig'] for r in modelResults])
        avgDen = np.mean([r['mapeDen'] for r in modelResults])

        print(f"  {modelName:10s} | den wins: {wins}/{len(modelResults)} ({wins/len(modelResults)*100:.1f}%) | "
              f"adapt wins: {adaptWins}/{len(modelResults)} ({adaptWins/len(modelResults)*100:.1f}%) | "
              f"MAPE: {avgOrig:.2f} -> {avgDen:.2f}")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: By Noise Level")
    print("=" * 70)

    noiseRatios = [r['noiseRatio'] for r in validResults]
    if noiseRatios:
        medianNoise = np.median(noiseRatios)

        highNoise = [r for r in validResults if r['noiseRatio'] >= medianNoise]
        lowNoise = [r for r in validResults if r['noiseRatio'] < medianNoise]

        for label, subset in [("High noise (>= median)", highNoise), ("Low noise (< median)", lowNoise)]:
            if not subset:
                continue
            wins = sum(1 for r in subset if r['mapeDen'] < r['mapeOrig'])
            avgOrig = np.mean([r['mapeOrig'] for r in subset])
            avgDen = np.mean([r['mapeDen'] for r in subset])
            print(f"  {label}: wins = {wins}/{len(subset)} ({wins/len(subset)*100:.1f}%), "
                  f"MAPE: {avgOrig:.2f} -> {avgDen:.2f}")

    print("\n" + "=" * 70)
    print("ANALYSIS 5: By Dataset Category")
    print("=" * 70)

    seasonalData = ['retailSales', 'energyUsage', 'multiSeasonalRetail']
    noisyData = ['volatile', 'stationary', 'stockPrice']
    trendData = ['trending', 'manufacturing']
    otherData = ['temperature', 'regimeShift']

    categories = [
        ("Seasonal", seasonalData),
        ("Noisy/Volatile", noisyData),
        ("Trending", trendData),
        ("Other", otherData),
    ]

    for catName, dataList in categories:
        catResults = [r for r in validResults if r['dataName'] in dataList]
        if not catResults:
            continue

        wins = sum(1 for r in catResults if r['mapeDen'] < r['mapeOrig'])
        adaptWins = sum(1 for r in catResults if not np.isnan(r['mapeAdapt']) and r['mapeAdapt'] < r['mapeOrig'])
        avgOrig = np.mean([r['mapeOrig'] for r in catResults])
        avgDen = np.mean([r['mapeDen'] for r in catResults])

        print(f"  {catName:20s} | den wins: {wins}/{len(catResults)} ({wins/len(catResults)*100:.1f}%) | "
              f"adapt wins: {adaptWins}/{len(catResults)} ({adaptWins/len(catResults)*100:.1f}%) | "
              f"MAPE: {avgOrig:.2f} -> {avgDen:.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if validResults:
        totalWins = sum(1 for r in validResults if r['mapeDen'] < r['mapeOrig'])
        totalAdaptWins = sum(1 for r in validResults if not np.isnan(r['mapeAdapt']) and r['mapeAdapt'] < r['mapeOrig'])
        avgOrig = np.mean([r['mapeOrig'] for r in validResults])
        avgDen = np.mean([r['mapeDen'] for r in validResults])

        print(f"""
    - Denoised win rate: {totalWins}/{len(validResults)} ({totalWins/len(validResults)*100:.1f}%)
    - Adaptive win rate: {totalAdaptWins}/{len(validResults)} ({totalAdaptWins/len(validResults)*100:.1f}%)
    - Avg MAPE: {avgOrig:.2f} -> {avgDen:.2f} ({(avgOrig-avgDen)/avgOrig*100:+.1f}%)
        """)


if __name__ == '__main__':
    main()
