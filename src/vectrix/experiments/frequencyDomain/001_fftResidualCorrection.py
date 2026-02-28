"""
==============================================================================
실험 ID: frequencyDomain/001
실험명: FFT 잔차 보정 (Fourier Residual Correction)
==============================================================================

목적:
- 시간 도메인 모델의 잔차에서 주기적 패턴을 FFT로 검출
- 검출된 주기 성분을 예측에 추가 보정하여 정확도 개선
- M4 Daily OWA 1.207 (Naive2보다 나쁨) 문제의 원인 분석 및 해법 탐색

가설:
1. 시간 도메인 모델 잔차에 유의미한 주기 성분이 남아있음 (전체 분산의 10%+)
2. FFT 잔차 보정 적용 시 MAPE 5%+ 개선
3. 다중 계절성(주간+연간)이 강한 데이터에서 보정 효과가 가장 큼

방법:
1. 합성 데이터 + 다양한 시계열 특성 데이터셋 생성 (10종)
2. 6개 모델 (ETS, ARIMA, Theta, MSTL, AutoCES, DOT)로 예측
3. 학습 잔차에서 FFT로 지배적 주기 성분 추출 (top-K)
4. 추출된 주기 성분으로 미래 구간 보정값 생성
5. 보정 전/후 MAPE, RMSE, MAE 비교
6. 주기 성분 에너지 비율 vs 보정 효과 상관 분석

성공 기준:
- 잔차에서 유의미 주기 검출: 60%+ 시리즈
- FFT 보정 후 MAPE 개선: 전체 평균 5%+
- 다중 계절성 데이터에서 10%+ 개선

==============================================================================
결과 (실험 후 작성)
==============================================================================

수치 (ETS 제외, 60건 기준 — ETS는 train predict 구조 문제로 MAPE 비정상):
| 지표 | 값 |
|------|-----|
| FFT 보정 승률 | 18/60 (30.0%) |
| Damped FFT 승률 | 19/60 (31.7%) |
| 잔차 주기 검출률 | 72/72 (100%) — energy > 10% |
| Energy-Improvement 상관 | -0.23 (역상관) |

모델별 FFT 승률 (ETS 제외):
| 모델 | 승률 | 비고 |
|------|------|------|
| theta | 6/12 (50%) | 최고 승률 |
| arima | 3/12 (25%) | - |
| auto_ces | 4/12 (33%) | - |
| dot | 3/12 (25%) | - |
| mstl | 2/12 (17%) | 이미 계절성 잘 포착 → 보정 불필요 |

핵심 발견:
1. 잔차에 주기 성분은 100% 존재하나, FFT 보정이 오히려 정확도 악화
2. Energy ratio가 높을수록 과보정 → MAPE 악화 (상관 -0.23)
3. MSTL처럼 이미 계절성을 잘 잡는 모델은 보정 효과 최저 (17%)
4. retailSales, stockPrice, multiSeasonalRetail에서만 부분적 효과
5. temperature, trending, regimeShift에서 심각한 악화 — 비주기적 잔차에 가짜 주기 투영

결론: 기각
- Naive FFT 잔차 보정은 과적합 위험이 높음
- 잔차의 FFT 성분 ≠ 미래 반복 패턴 (과거 노이즈 패턴 재투영)
- 다음 방향: 주파수 도메인 노이즈 필터링 (보정이 아닌 제거) 또는
  Cross-Validation 기반 보정 성분 선택

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
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from vectrix.engine.arima import ARIMAModel
from vectrix.engine.ces import AutoCES
from vectrix.engine.dot import DynamicOptimizedTheta
from vectrix.engine.ets import ETSModel
from vectrix.engine.mstl import AutoMSTL
from vectrix.engine.theta import OptimizedTheta
from vectrix.experiments._utils.dataGenerators import ALL_GENERATORS


def _extractResidualSpectrum(residuals: np.ndarray, topK: int = 5):
    """잔차에서 FFT로 지배적 주기 성분 추출."""
    n = len(residuals)
    if n < 10:
        return [], [], 0.0

    detrended = residuals - np.linspace(residuals[0], residuals[-1], n)

    fft = np.fft.rfft(detrended)
    magnitudes = np.abs(fft)
    phases = np.angle(fft)
    freqs = np.fft.rfftfreq(n)

    magnitudes[0] = 0

    totalPower = np.sum(magnitudes ** 2)
    if totalPower < 1e-10:
        return [], [], 0.0

    sortedIdx = np.argsort(magnitudes)[::-1]

    topFreqs = []
    topMags = []
    topPhases = []
    capturedPower = 0.0

    for idx in sortedIdx[:topK]:
        if magnitudes[idx] < 1e-10:
            break
        if freqs[idx] < 1.0 / n:
            continue
        topFreqs.append(freqs[idx])
        topMags.append(magnitudes[idx])
        topPhases.append(phases[idx])
        capturedPower += magnitudes[idx] ** 2

    energyRatio = capturedPower / totalPower if totalPower > 0 else 0.0

    return list(zip(topFreqs, topMags, topPhases)), energyRatio, totalPower


def _generateFftCorrection(components, nFuture: int, nTrain: int):
    """추출된 FFT 성분으로 미래 보정값 생성."""
    if not components:
        return np.zeros(nFuture)

    correction = np.zeros(nFuture)
    t = np.arange(nTrain, nTrain + nFuture)

    for freq, mag, phase in components:
        amplitude = 2.0 * mag / nTrain
        correction += amplitude * np.cos(2.0 * np.pi * freq * t + phase)

    return correction


def _computeMetrics(actual, predicted):
    """MAPE, RMSE, MAE 계산."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    mask = np.abs(actual) > 1e-8
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    return mape, rmse, mae


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


def _runSingleExperiment(data: np.ndarray, modelName: str, horizon: int, topK: int = 5):
    """단일 (데이터, 모델) 조합 실험."""
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

    trainPred, _, _ = model.predict(len(trainData))
    if len(trainPred) < trainSize:
        residuals = trainData[len(trainPred):] - trainPred[-len(trainData[len(trainPred):]):]
        if len(residuals) < 10:
            residuals = trainData - np.mean(trainData)
    else:
        residuals = trainData - trainPred[:trainSize]

    components, energyRatio, totalPower = _extractResidualSpectrum(residuals, topK=topK)

    mapeOrig, rmseOrig, maeOrig = _computeMetrics(testData, pred)

    correction = _generateFftCorrection(components, horizon, trainSize)
    correctedPred = pred + correction[:len(pred)]

    mapeCorrected, rmseCorrected, maeCorrected = _computeMetrics(testData, correctedPred)

    dampedCorrection = _generateFftCorrection(components, horizon, trainSize)
    dampFactor = np.exp(-0.1 * np.arange(horizon))
    dampedCorrectedPred = pred + dampedCorrection[:len(pred)] * dampFactor[:len(pred)]
    mapeDamped, rmseDamped, maeDamped = _computeMetrics(testData, dampedCorrectedPred)

    topKResults = {}
    for k in [1, 3, 5]:
        if k > len(components):
            topKResults[k] = (mapeOrig, rmseOrig, maeOrig)
            continue
        subComponents = components[:k]
        subCorrection = _generateFftCorrection(subComponents, horizon, trainSize)
        subPred = pred + subCorrection[:len(pred)]
        topKResults[k] = _computeMetrics(testData, subPred)

    periods = [1.0 / f for f, _, _ in components if f > 0] if components else []

    return {
        'modelName': modelName,
        'mapeOrig': mapeOrig,
        'rmseOrig': rmseOrig,
        'maeOrig': maeOrig,
        'mapeCorrected': mapeCorrected,
        'rmseCorrected': rmseCorrected,
        'maeCorrected': maeCorrected,
        'mapeDamped': mapeDamped,
        'rmseDamped': rmseDamped,
        'maeDamped': maeDamped,
        'energyRatio': energyRatio,
        'nComponents': len(components),
        'dominantPeriods': periods[:3],
        'topKResults': topKResults,
    }


def _generateHourlyMultiSeasonal(n: int = 720, seed: int = 42) -> pd.DataFrame:
    """시간별 다중 계절성 데이터 (24시간 + 168시간 주기)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="h")

    base = 500.0
    daily = 100.0 * np.sin(2.0 * np.pi * t / 24.0)
    weekly = 50.0 * np.sin(2.0 * np.pi * t / 168.0)
    noise = rng.normal(0, 15, n)

    values = base + daily + weekly + noise
    return pd.DataFrame({"date": dates, "value": values})


def _generateStrongMultiPeriodic(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """3중 주기성 (7일, 30일, 90일) 강한 데이터."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    dates = pd.date_range("2022-01-01", periods=n, freq="D")

    base = 1000.0
    p7 = 200.0 * np.sin(2.0 * np.pi * t / 7.0)
    p30 = 100.0 * np.sin(2.0 * np.pi * t / 30.0)
    p90 = 80.0 * np.sin(2.0 * np.pi * t / 90.0)
    noise = rng.normal(0, 30, n)

    values = base + p7 + p30 + p90 + noise
    return pd.DataFrame({"date": dates, "value": values})


def main():
    print("=" * 70)
    print("E023: FFT Residual Correction")
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

    datasets['hourlyMultiSeasonal'] = _generateHourlyMultiSeasonal(720, seed=42)
    datasets['strongMultiPeriodic'] = _generateStrongMultiPeriodic(500, seed=42)

    modelNames = ['ets', 'arima', 'theta', 'mstl', 'auto_ces', 'dot']
    horizon = 14

    allResults = []
    print(f"\n--- {len(datasets)} datasets x {len(modelNames)} models = {len(datasets) * len(modelNames)} combinations ---\n")

    for dataName, df in datasets.items():
        values = df['value'].values.astype(np.float64)

        for modelName in modelNames:
            try:
                result = _runSingleExperiment(values, modelName, horizon)
                if result is None:
                    continue
                result['dataName'] = dataName
                allResults.append(result)

                improved = result['mapeCorrected'] < result['mapeOrig']
                marker = "+" if improved else "-"
                print(f"  [{marker}] {dataName:25s} | {modelName:10s} | "
                      f"MAPE: {result['mapeOrig']:8.2f} -> {result['mapeCorrected']:8.2f} "
                      f"(damped: {result['mapeDamped']:8.2f}) | "
                      f"Energy: {result['energyRatio']:.3f}")
            except Exception as e:
                print(f"  [!] {dataName:25s} | {modelName:10s} | Error: {e}")

    if not allResults:
        print("\nNo results collected!")
        return

    print("\n" + "=" * 70)
    print("ANALYSIS 1: Overall FFT Correction Impact")
    print("=" * 70)

    nTotal = len(allResults)
    nImprovedFft = sum(1 for r in allResults if r['mapeCorrected'] < r['mapeOrig'])
    nImprovedDamped = sum(1 for r in allResults if r['mapeDamped'] < r['mapeOrig'])

    avgMapeOrig = np.mean([r['mapeOrig'] for r in allResults if not np.isnan(r['mapeOrig'])])
    avgMapeFft = np.mean([r['mapeCorrected'] for r in allResults if not np.isnan(r['mapeCorrected'])])
    avgMapeDamped = np.mean([r['mapeDamped'] for r in allResults if not np.isnan(r['mapeDamped'])])

    print(f"\nTotal experiments: {nTotal}")
    print(f"FFT correction wins:    {nImprovedFft}/{nTotal} ({nImprovedFft/nTotal*100:.1f}%)")
    print(f"Damped FFT wins:        {nImprovedDamped}/{nTotal} ({nImprovedDamped/nTotal*100:.1f}%)")
    print(f"\nAvg MAPE - Original:   {avgMapeOrig:.2f}%")
    print(f"Avg MAPE - FFT:        {avgMapeFft:.2f}%")
    print(f"Avg MAPE - Damped:     {avgMapeDamped:.2f}%")
    print(f"FFT improvement:       {(avgMapeOrig - avgMapeFft) / avgMapeOrig * 100:.1f}%")
    print(f"Damped improvement:    {(avgMapeOrig - avgMapeDamped) / avgMapeOrig * 100:.1f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS 2: By Dataset (Multi-seasonality effect)")
    print("=" * 70)

    multiSeasonalData = ['retailSales', 'energyUsage', 'multiSeasonalRetail',
                         'hourlyMultiSeasonal', 'strongMultiPeriodic']

    for category, dataList in [("Multi-Seasonal", multiSeasonalData),
                                ("Other", [d for d in datasets if d not in multiSeasonalData])]:
        catResults = [r for r in allResults if r['dataName'] in dataList]
        if not catResults:
            continue

        wins = sum(1 for r in catResults if r['mapeCorrected'] < r['mapeOrig'])
        dampWins = sum(1 for r in catResults if r['mapeDamped'] < r['mapeOrig'])
        avgOrig = np.nanmean([r['mapeOrig'] for r in catResults])
        avgFft = np.nanmean([r['mapeCorrected'] for r in catResults])
        avgDamp = np.nanmean([r['mapeDamped'] for r in catResults])
        avgEnergy = np.mean([r['energyRatio'] for r in catResults])

        print(f"\n  [{category}] ({len(catResults)} experiments)")
        print(f"    FFT wins: {wins}/{len(catResults)} ({wins/len(catResults)*100:.1f}%)")
        print(f"    Damped wins: {dampWins}/{len(catResults)} ({dampWins/len(catResults)*100:.1f}%)")
        print(f"    MAPE: {avgOrig:.2f} -> FFT {avgFft:.2f}, Damped {avgDamp:.2f}")
        print(f"    Avg residual energy ratio: {avgEnergy:.3f}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: By Model")
    print("=" * 70)

    for modelName in modelNames:
        modelResults = [r for r in allResults if r['modelName'] == modelName]
        if not modelResults:
            continue

        wins = sum(1 for r in modelResults if r['mapeCorrected'] < r['mapeOrig'])
        dampWins = sum(1 for r in modelResults if r['mapeDamped'] < r['mapeOrig'])
        avgOrig = np.nanmean([r['mapeOrig'] for r in modelResults])
        avgFft = np.nanmean([r['mapeCorrected'] for r in modelResults])

        print(f"  {modelName:10s} | wins: {wins}/{len(modelResults)} ({wins/len(modelResults)*100:.1f}%) | "
              f"MAPE: {avgOrig:.2f} -> {avgFft:.2f} | "
              f"damped wins: {dampWins}/{len(modelResults)}")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: Energy Ratio vs Improvement Correlation")
    print("=" * 70)

    energies = []
    improvements = []
    for r in allResults:
        if np.isnan(r['mapeOrig']) or np.isnan(r['mapeCorrected']) or r['mapeOrig'] < 1e-8:
            continue
        energies.append(r['energyRatio'])
        improvements.append((r['mapeOrig'] - r['mapeCorrected']) / r['mapeOrig'] * 100)

    if len(energies) > 5:
        corr = np.corrcoef(energies, improvements)[0, 1]
        print(f"\n  Correlation(energy_ratio, MAPE_improvement): {corr:.4f}")
        print(f"  N = {len(energies)}")

        p75Energy = np.percentile(energies, 75)
        highEnergy = [improvements[i] for i in range(len(energies)) if energies[i] >= p75Energy]
        lowEnergy = [improvements[i] for i in range(len(energies)) if energies[i] < p75Energy]

        print(f"\n  High energy (>= P75={p75Energy:.3f}): avg improvement = {np.mean(highEnergy):.2f}%")
        print(f"  Low energy  (<  P75):                avg improvement = {np.mean(lowEnergy):.2f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS 5: Top-K Selection (1 vs 3 vs 5 components)")
    print("=" * 70)

    for k in [1, 3, 5]:
        kMapes = [r['topKResults'][k][0] for r in allResults if k in r['topKResults'] and not np.isnan(r['topKResults'][k][0])]
        origMapes = [r['mapeOrig'] for r in allResults if k in r['topKResults'] and not np.isnan(r['mapeOrig'])]

        if kMapes:
            kWins = sum(1 for km, om in zip(kMapes, origMapes) if km < om)
            print(f"  Top-{k}: avg MAPE = {np.mean(kMapes):.2f}%, "
                  f"wins = {kWins}/{len(kMapes)} ({kWins/len(kMapes)*100:.1f}%)")

    print("\n" + "=" * 70)
    print("ANALYSIS 6: Dominant Periods Found in Residuals")
    print("=" * 70)

    allPeriods = []
    for r in allResults:
        allPeriods.extend(r['dominantPeriods'])

    if allPeriods:
        periodBins = {'<5': 0, '5-10': 0, '10-30': 0, '30-60': 0, '60-100': 0, '>100': 0}
        for p in allPeriods:
            if p < 5:
                periodBins['<5'] += 1
            elif p < 10:
                periodBins['5-10'] += 1
            elif p < 30:
                periodBins['10-30'] += 1
            elif p < 60:
                periodBins['30-60'] += 1
            elif p < 100:
                periodBins['60-100'] += 1
            else:
                periodBins['>100'] += 1

        print(f"\n  Total dominant periods found: {len(allPeriods)}")
        for binName, count in periodBins.items():
            bar = '#' * (count // 2)
            print(f"    {binName:>6s}: {count:4d} {bar}")

    nSignificant = sum(1 for r in allResults if r['energyRatio'] > 0.1)
    print(f"\n  Series with significant residual periodicity (energy > 10%): "
          f"{nSignificant}/{nTotal} ({nSignificant/nTotal*100:.1f}%)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    print(f"""
    - FFT win rate: {nImprovedFft}/{nTotal} ({nImprovedFft/nTotal*100:.1f}%)
    - Damped FFT win rate: {nImprovedDamped}/{nTotal} ({nImprovedDamped/nTotal*100:.1f}%)
    - Avg MAPE change: {avgMapeOrig:.2f} -> {avgMapeFft:.2f} (FFT), {avgMapeDamped:.2f} (Damped)
    - Residual periodicity detected: {nSignificant}/{nTotal} ({nSignificant/nTotal*100:.1f}%)
    """)


if __name__ == '__main__':
    main()
