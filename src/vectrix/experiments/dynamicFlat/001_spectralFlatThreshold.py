"""
==============================================================================
실험 ID: dynamicFlat/001
실험명: 스펙트럼 기반 동적 Flat Defense 임계값
==============================================================================

목적:
- 현재 FlatPredictionDetector의 고정 임계값(horizontalThreshold=0.01 등)이
  모든 시계열에 동일 적용 → 변동성 높은 시계열에서 오탐, 낮은 시계열에서 미탐
- FFT 스펙트럼으로 시계열의 "실제 변동성 수준"을 파악한 뒤 임계값을 동적 설정
- 오탐(False Positive)과 미탐(False Negative) 비율 개선 검증

가설:
1. 고정 임계값은 계절성 강한 시계열에서 오탐률 > 20%
2. 동적 임계값으로 오탐률 50%+ 감소
3. 스펙트럼 에너지 집중도가 높은 시계열은 더 관대한 임계값이 적절
4. 변동성 낮은 시계열은 더 엄격한 임계값이 필요

방법:
1. 합성 데이터 10종으로 "정상 예측"과 "인위적 flat 예측" 생성
2. 현재 고정 임계값으로 감지 → 오탐/미탐 측정 (baseline)
3. FFT 스펙트럼 분석으로 동적 임계값 계산
4. 동적 임계값으로 감지 → 오탐/미탐 측정 (개선)
5. F1 Score 비교

성공 기준:
- 동적 임계값 F1 > 고정 임계값 F1
- 오탐률 50%+ 감소

==============================================================================
결과 (실험 후 작성)
==============================================================================

수치 (80건: flat 30건 + normal 50건):
| 지표 | Fixed | Dynamic |
|------|-------|---------|
| TP | 30 | 30 |
| FP | 44 | 45 |
| TN | 6 | 5 |
| FN | 0 | 0 |
| Precision | 0.405 | 0.400 |
| Recall | 1.000 | 1.000 |
| F1 | 0.577 | 0.571 |
| FP Rate | 0.880 | 0.900 |

핵심 발견:
1. 가설 1 채택 (더 심각): FP Rate 88% — 거의 모든 정상 예측을 flat으로 오감지
2. 가설 2 기각: 동적 임계값이 오히려 FP Rate 2.3% 증가
3. 근본 원인: 문제는 임계값이 아니라 판단 기준 자체
   - 14-step 예측에서 대부분 모델이 추세만 반영, 계절 변동 미약
   - std(pred)/std(original) < 0.01이 정상 예측에서도 발생
   - 원본 365일 std vs 예측 14일 std는 본질적으로 비교 불가능
4. 스펙트럼 동적 임계값의 범위 0.004~0.05 → 정상 예측 std 비율이 이 범위보다 훨씬 아래

결론: 기각 — 근본적 재설계 필요
- Flat defense의 std 비율 기반 판단은 구조적 결함
- 대안 1: 예측 horizon에 비례하는 기대 std 계산 (계절 주기 * horizon / period)
- 대안 2: 예측값의 차분(diff) 패턴 분석 (주기성 존재 여부)
- 대안 3: 예측값과 naive seasonal 예측의 상관으로 판단
- 002에서 "horizon-aware 판단 기준" 실험

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
from vectrix.flat_defense.detector import FlatPredictionDetector


def _spectralAnalysis(data: np.ndarray):
    """FFT 기반 스펙트럼 특성 분석."""
    n = len(data)
    if n < 10:
        return {'energyConcentration': 0.5, 'dominantPeriod': 7, 'spectralEntropy': 1.0}

    detrended = data - np.linspace(data[0], data[-1], n)
    fft = np.fft.rfft(detrended)
    power = np.abs(fft) ** 2
    power[0] = 0
    totalPower = np.sum(power)

    if totalPower < 1e-10:
        return {'energyConcentration': 0.0, 'dominantPeriod': n, 'spectralEntropy': 0.0}

    normPower = power / totalPower
    entropy = -np.sum(normPower[normPower > 0] * np.log2(normPower[normPower > 0]))
    maxEntropy = np.log2(len(power))
    normalizedEntropy = entropy / maxEntropy if maxEntropy > 0 else 0

    sortedPower = np.sort(power)[::-1]
    top3Power = np.sum(sortedPower[:3])
    energyConcentration = top3Power / totalPower

    freqs = np.fft.rfftfreq(n)
    dominantIdx = np.argmax(power)
    dominantPeriod = 1.0 / freqs[dominantIdx] if freqs[dominantIdx] > 0 else n

    return {
        'energyConcentration': energyConcentration,
        'dominantPeriod': dominantPeriod,
        'spectralEntropy': normalizedEntropy,
    }


def _dynamicThresholds(spectralInfo: dict, dataStd: float, dataMean: float):
    """스펙트럼 분석 결과로 동적 임계값 계산."""
    ec = spectralInfo['energyConcentration']
    se = spectralInfo['spectralEntropy']
    cv = dataStd / abs(dataMean) if abs(dataMean) > 1e-10 else 0

    if ec > 0.7:
        horizontalThreshold = 0.05
    elif ec > 0.4:
        horizontalThreshold = 0.02
    else:
        horizontalThreshold = 0.008

    if cv < 0.03:
        horizontalThreshold *= 0.5

    if se > 0.8:
        varianceThreshold = 0.001
    elif se > 0.5:
        varianceThreshold = 0.0005
    else:
        varianceThreshold = 0.0001

    diagonalThreshold = 1e-8 * (1 + ec * 10)

    return {
        'horizontalThreshold': horizontalThreshold,
        'diagonalThreshold': diagonalThreshold,
        'varianceThreshold': varianceThreshold,
    }


def _generateFlatPrediction(data: np.ndarray, horizon: int, flatType: str):
    """인위적 flat 예측 생성."""
    lastVal = data[-1]

    if flatType == 'horizontal':
        return np.full(horizon, lastVal)
    elif flatType == 'diagonal':
        slope = (data[-1] - data[-10]) / 10 if len(data) >= 10 else 0
        return lastVal + slope * np.arange(1, horizon + 1)
    elif flatType == 'mean_reversion':
        meanVal = np.mean(data[-30:]) if len(data) >= 30 else np.mean(data)
        return np.linspace(lastVal, meanVal, horizon)
    return np.full(horizon, lastVal)


def _generateNormalPrediction(data: np.ndarray, modelName: str, horizon: int):
    """정상 모델 예측 생성."""
    factories = {
        'arima': lambda: ARIMAModel(),
        'theta': lambda: OptimizedTheta(),
        'mstl': lambda: AutoMSTL(),
        'auto_ces': lambda: AutoCES(),
        'dot': lambda: DynamicOptimizedTheta(),
    }

    model = factories.get(modelName, lambda: ARIMAModel())()
    model.fit(data)
    pred, _, _ = model.predict(horizon)
    return pred[:horizon]


def main():
    print("=" * 70)
    print("E021: Spectral-based Dynamic Flat Defense Threshold")
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

    horizon = 14
    modelNames = ['arima', 'theta', 'mstl', 'auto_ces', 'dot']
    flatTypes = ['horizontal', 'diagonal', 'mean_reversion']

    fixedDetector = FlatPredictionDetector(
        horizontalThreshold=0.01,
        diagonalThreshold=1e-8,
        varianceThreshold=0.0001
    )

    fixedTP = 0
    fixedFP = 0
    fixedTN = 0
    fixedFN = 0
    dynTP = 0
    dynFP = 0
    dynTN = 0
    dynFN = 0

    detailResults = []

    print("\n--- Phase 1: Flat prediction detection (expected: FLAT) ---\n")

    for dataName, df in datasets.items():
        values = df['value'].values.astype(np.float64)
        spectralInfo = _spectralAnalysis(values)
        dynThresh = _dynamicThresholds(spectralInfo, np.std(values), np.mean(values))

        dynamicDetector = FlatPredictionDetector(
            horizontalThreshold=dynThresh['horizontalThreshold'],
            diagonalThreshold=dynThresh['diagonalThreshold'],
            varianceThreshold=dynThresh['varianceThreshold'],
        )

        for flatType in flatTypes:
            flatPred = _generateFlatPrediction(values, horizon, flatType)

            fixedResult = fixedDetector.detect(flatPred, values)
            dynResult = dynamicDetector.detect(flatPred, values)

            if fixedResult.isFlat:
                fixedTP += 1
            else:
                fixedFN += 1

            if dynResult.isFlat:
                dynTP += 1
            else:
                dynFN += 1

            fM = "TP" if fixedResult.isFlat else "FN"
            dM = "TP" if dynResult.isFlat else "FN"

            print(f"  {dataName:25s} | {flatType:15s} | Fixed: {fM} | Dynamic: {dM} | "
                  f"ec={spectralInfo['energyConcentration']:.3f} hT={dynThresh['horizontalThreshold']:.4f}")

            detailResults.append({
                'dataName': dataName,
                'predType': f'flat_{flatType}',
                'isActuallyFlat': True,
                'fixedDetected': fixedResult.isFlat,
                'dynDetected': dynResult.isFlat,
                'spectralInfo': spectralInfo,
                'dynThresh': dynThresh,
            })

    print("\n--- Phase 2: Normal prediction detection (expected: NOT FLAT) ---\n")

    for dataName, df in datasets.items():
        values = df['value'].values.astype(np.float64)
        spectralInfo = _spectralAnalysis(values)
        dynThresh = _dynamicThresholds(spectralInfo, np.std(values), np.mean(values))

        dynamicDetector = FlatPredictionDetector(
            horizontalThreshold=dynThresh['horizontalThreshold'],
            diagonalThreshold=dynThresh['diagonalThreshold'],
            varianceThreshold=dynThresh['varianceThreshold'],
        )

        for modelName in modelNames:
            try:
                normalPred = _generateNormalPrediction(values, modelName, horizon)

                fixedResult = fixedDetector.detect(normalPred, values)
                dynResult = dynamicDetector.detect(normalPred, values)

                if fixedResult.isFlat:
                    fixedFP += 1
                else:
                    fixedTN += 1

                if dynResult.isFlat:
                    dynFP += 1
                else:
                    dynTN += 1

                fM = "FP" if fixedResult.isFlat else "TN"
                dM = "FP" if dynResult.isFlat else "TN"

                if fixedResult.isFlat or dynResult.isFlat:
                    print(f"  {dataName:25s} | {modelName:10s} | Fixed: {fM} | Dynamic: {dM} | "
                          f"hT={dynThresh['horizontalThreshold']:.4f}")

                detailResults.append({
                    'dataName': dataName,
                    'predType': f'normal_{modelName}',
                    'isActuallyFlat': False,
                    'fixedDetected': fixedResult.isFlat,
                    'dynDetected': dynResult.isFlat,
                    'spectralInfo': spectralInfo,
                    'dynThresh': dynThresh,
                })

            except Exception as e:
                print(f"  [!] {dataName}/{modelName}: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS 1: Confusion Matrix Comparison")
    print("=" * 70)

    print("\n  Fixed Detector:")
    print(f"    TP={fixedTP:3d}  FP={fixedFP:3d}")
    print(f"    FN={fixedFN:3d}  TN={fixedTN:3d}")

    fixedPrecision = fixedTP / (fixedTP + fixedFP) if (fixedTP + fixedFP) > 0 else 0
    fixedRecall = fixedTP / (fixedTP + fixedFN) if (fixedTP + fixedFN) > 0 else 0
    fixedF1 = 2 * fixedPrecision * fixedRecall / (fixedPrecision + fixedRecall) if (fixedPrecision + fixedRecall) > 0 else 0
    fixedFPR = fixedFP / (fixedFP + fixedTN) if (fixedFP + fixedTN) > 0 else 0

    print(f"    Precision: {fixedPrecision:.3f}")
    print(f"    Recall:    {fixedRecall:.3f}")
    print(f"    F1:        {fixedF1:.3f}")
    print(f"    FP Rate:   {fixedFPR:.3f}")

    print("\n  Dynamic Detector:")
    print(f"    TP={dynTP:3d}  FP={dynFP:3d}")
    print(f"    FN={dynFN:3d}  TN={dynTN:3d}")

    dynPrecision = dynTP / (dynTP + dynFP) if (dynTP + dynFP) > 0 else 0
    dynRecall = dynTP / (dynTP + dynFN) if (dynTP + dynFN) > 0 else 0
    dynF1 = 2 * dynPrecision * dynRecall / (dynPrecision + dynRecall) if (dynPrecision + dynRecall) > 0 else 0
    dynFPR = dynFP / (dynFP + dynTN) if (dynFP + dynTN) > 0 else 0

    print(f"    Precision: {dynPrecision:.3f}")
    print(f"    Recall:    {dynRecall:.3f}")
    print(f"    F1:        {dynF1:.3f}")
    print(f"    FP Rate:   {dynFPR:.3f}")

    print("\n  Improvement:")
    print(f"    F1: {fixedF1:.3f} -> {dynF1:.3f} ({(dynF1 - fixedF1) / max(fixedF1, 0.001) * 100:+.1f}%)")
    print(f"    FP Rate: {fixedFPR:.3f} -> {dynFPR:.3f} ({(fixedFPR - dynFPR) / max(fixedFPR, 0.001) * 100:+.1f}% reduction)")

    print("\n" + "=" * 70)
    print("ANALYSIS 2: By Data Type")
    print("=" * 70)

    seasonalData = ['retailSales', 'energyUsage', 'multiSeasonalRetail']
    volatileData = ['volatile', 'stockPrice']
    stableData = ['stationary', 'trending']
    otherData = ['temperature', 'manufacturing', 'regimeShift']

    categories = [
        ("Seasonal", seasonalData),
        ("Volatile", volatileData),
        ("Stable", stableData),
        ("Other", otherData),
    ]

    for catName, dataList in categories:
        catResults = [r for r in detailResults if r['dataName'] in dataList]

        flatResults = [r for r in catResults if r['isActuallyFlat']]
        normalResults = [r for r in catResults if not r['isActuallyFlat']]

        fTP = sum(1 for r in flatResults if r['fixedDetected'])
        fFN = sum(1 for r in flatResults if not r['fixedDetected'])
        fFP = sum(1 for r in normalResults if r['fixedDetected'])
        fTN = sum(1 for r in normalResults if not r['fixedDetected'])

        dTP = sum(1 for r in flatResults if r['dynDetected'])
        dFN = sum(1 for r in flatResults if not r['dynDetected'])
        dFP = sum(1 for r in normalResults if r['dynDetected'])
        dTN = sum(1 for r in normalResults if not r['dynDetected'])

        fPrec = fTP / (fTP + fFP) if (fTP + fFP) > 0 else 0
        fRec = fTP / (fTP + fFN) if (fTP + fFN) > 0 else 0
        fF = 2 * fPrec * fRec / (fPrec + fRec) if (fPrec + fRec) > 0 else 0

        dPrec = dTP / (dTP + dFP) if (dTP + dFP) > 0 else 0
        dRec = dTP / (dTP + dFN) if (dTP + dFN) > 0 else 0
        dF = 2 * dPrec * dRec / (dPrec + dRec) if (dPrec + dRec) > 0 else 0

        avgEc = np.mean([r['spectralInfo']['energyConcentration'] for r in catResults])

        print(f"\n  [{catName}] (ec={avgEc:.3f})")
        print(f"    Fixed:   F1={fF:.3f}  Prec={fPrec:.3f}  Rec={fRec:.3f}  FP={fFP}")
        print(f"    Dynamic: F1={dF:.3f}  Prec={dPrec:.3f}  Rec={dRec:.3f}  FP={dFP}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Dynamic Threshold Distribution")
    print("=" * 70)

    for dataName in sorted(datasets.keys()):
        sample = [r for r in detailResults if r['dataName'] == dataName]
        if sample:
            si = sample[0]['spectralInfo']
            dt = sample[0]['dynThresh']
            print(f"  {dataName:25s} | ec={si['energyConcentration']:.3f} se={si['spectralEntropy']:.3f} | "
                  f"hT={dt['horizontalThreshold']:.4f} vT={dt['varianceThreshold']:.6f} "
                  f"dT={dt['diagonalThreshold']:.2e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
    Fixed:   F1={fixedF1:.3f}, FP Rate={fixedFPR:.3f}
    Dynamic: F1={dynF1:.3f}, FP Rate={dynFPR:.3f}
    F1 improvement: {(dynF1 - fixedF1) / max(fixedF1, 0.001) * 100:+.1f}%
    FP Rate reduction: {(fixedFPR - dynFPR) / max(fixedFPR, 0.001) * 100:+.1f}%
    """)


if __name__ == '__main__':
    main()
