"""
==============================================================================
실험 ID: dynamicFlat/002
실험명: Horizon-Aware Flat Detection (예측 길이 인식 판단 기준)
==============================================================================

목적:
- 001에서 발견: FP Rate 88% — 현재 flat detector가 거의 모든 정상 예측을 flat으로 오감지
- 근본 원인: 365일 원본 std vs 14일 예측 std를 직접 비교하는 구조적 결함
- 새로운 판단 기준 설계: horizon 길이에 맞는 기대 변동성을 계산하여 비교

가설:
1. 원본 데이터에서 "같은 길이"의 창(window)을 추출하면 기대 std를 정확히 추정
2. Seasonal Naive와의 상관으로 "주기성 존재 여부" 판단 가능
3. 새로운 기준으로 FP Rate < 20%, Recall > 90% 달성

방법:
1. 판단 기준 A: Window-based std ratio (원본에서 horizon 길이 창의 평균 std와 비교)
2. 판단 기준 B: Seasonal correlation (예측과 seasonal naive의 상관)
3. 판단 기준 C: Diff entropy (예측 차분의 엔트로피가 극도로 낮으면 flat)
4. 판단 기준 D: Combined (A+B+C 앙상블)
5. 각 기준의 TP/FP/TN/FN, F1 비교

성공 기준:
- F1 > 0.8 (현재 0.577)
- FP Rate < 20% (현재 88%)
- Recall > 90%

==============================================================================
결과 (실험 후 작성)
==============================================================================

수치 (90건: flat 40건 + normal 50건):
| 방법 | TP | FP | TN | FN | Prec | Rec | F1 | FPR |
|------|-----|-----|-----|-----|------|------|------|------|
| fixed(현재) | 40 | 44 | 6 | 0 | 0.476 | 1.000 | 0.645 | 0.880 |
| windowBased | 29 | 44 | 6 | 11 | 0.397 | 0.725 | 0.513 | 0.880 |
| seasonalCorr | 33 | 40 | 10 | 7 | 0.452 | 0.825 | 0.584 | 0.800 |
| diffEntropy | 10 | 20 | 30 | 30 | 0.333 | 0.250 | 0.286 | 0.400 |
| combined | 22 | 40 | 10 | 18 | 0.355 | 0.550 | 0.431 | 0.800 |

핵심 발견:
1. 가설 1 부분 채택: windowBased가 Recall 0.725로 합리적이나 FPR 변화 없음
2. 가설 2 부분 채택: seasonalCorr FPR 0.800 (0.880 대비 소폭 개선)
3. 가설 3 기각: 최고 F1 = 0.645(fixed), FPR < 20% 미달성
4. diffEntropy만 FPR 0.400 달성했으나 Recall 0.250으로 실용성 부족
5. 근본 문제: 14일 정상 예측도 원본 대비 변동성이 매우 작음
   - theta, dot의 정상 예측도 거의 직선에 가까움 (추세만 반영)
   - mstl만 계절성 반영하여 TN으로 분류되는 경우 있음

구조적 인사이트:
- Flat detection의 진짜 질문: "이 예측이 flat이라서 나쁜가?"
- 현재 구현: "이 예측이 flat인가?" — 이 질문 자체가 잘못됨
- 대부분 시계열 모델의 장기 예측은 본질적으로 flat에 수렴
- Flat ≠ Bad: stationary 데이터에서 flat 예측은 오히려 정확
- 진정한 해결: flat 감지가 아니라 "이 예측이 과소 변동인가" 판단 필요
  → 학습 데이터의 계절 패턴 강도 vs 예측의 계절 패턴 강도 비교

결론: 보류 — 패러다임 전환 필요
- 단순 임계값 조정이나 기준 교체로는 해결 불가
- "Flat Defense" 개념 자체를 "Seasonal Pattern Preservation" 관점으로 재정의 필요
- 학습 데이터에서 계절 패턴을 추출하고, 예측에 동일 수준의 패턴이 보존되는지 확인하는
  "Pattern Fidelity Score" 접근이 더 적절할 수 있음

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


def _windowBasedDetection(predictions: np.ndarray, originalData: np.ndarray, threshold: float = 0.3):
    """기준 A: 원본에서 같은 길이의 창을 추출하여 기대 std 계산."""
    horizon = len(predictions)
    n = len(originalData)

    if n < horizon * 2:
        return False

    windowStds = []
    for i in range(0, n - horizon, max(1, horizon // 2)):
        windowStds.append(np.std(originalData[i:i + horizon]))

    if not windowStds:
        return False

    expectedStd = np.median(windowStds)
    predStd = np.std(predictions)

    if expectedStd < 1e-10:
        return predStd < 1e-10

    return (predStd / expectedStd) < threshold


def _seasonalCorrelationDetection(predictions: np.ndarray, originalData: np.ndarray, period: int = 7):
    """기준 B: Seasonal Naive와의 상관으로 주기성 판단."""
    horizon = len(predictions)
    n = len(originalData)

    if n < period:
        return False

    seasonalNaive = np.zeros(horizon)
    for h in range(horizon):
        idx = n - period + (h % period)
        if idx >= 0:
            seasonalNaive[h] = originalData[idx]
        else:
            seasonalNaive[h] = originalData[h % n]

    predDiff = np.diff(predictions)
    snDiff = np.diff(seasonalNaive)

    if np.std(predDiff) < 1e-10 and np.std(snDiff) > 1e-10:
        return True

    if np.std(predDiff) < 1e-10:
        return np.std(snDiff) < 1e-10

    corr = np.corrcoef(predDiff, snDiff)[0, 1] if len(predDiff) > 2 else 0
    return abs(corr) < 0.1 and np.std(predDiff) / (np.std(snDiff) + 1e-10) < 0.1


def _diffEntropyDetection(predictions: np.ndarray, threshold: float = 0.5):
    """기준 C: 예측 차분의 다양성으로 flat 판단."""
    if len(predictions) < 4:
        return False

    diffs = np.diff(predictions)

    if np.max(np.abs(diffs)) < 1e-10:
        return True

    absDiffs = np.abs(diffs)
    normalized = absDiffs / (np.sum(absDiffs) + 1e-10)
    entropy = -np.sum(normalized[normalized > 0] * np.log2(normalized[normalized > 0] + 1e-15))
    maxEntropy = np.log2(len(diffs))
    normalizedEntropy = entropy / maxEntropy if maxEntropy > 0 else 0

    uniqueRatio = len(np.unique(np.round(diffs, 6))) / len(diffs)

    return normalizedEntropy < threshold and uniqueRatio < 0.3


def _combinedDetection(predictions: np.ndarray, originalData: np.ndarray, period: int = 7):
    """기준 D: A+B+C 앙상블 (2/3 이상 flat이면 flat)."""
    aResult = _windowBasedDetection(predictions, originalData)
    bResult = _seasonalCorrelationDetection(predictions, originalData, period)
    cResult = _diffEntropyDetection(predictions)

    votes = sum([aResult, bResult, cResult])
    return votes >= 2


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
    elif flatType == 'near_flat':
        rng = np.random.default_rng(42)
        noise = rng.normal(0, np.std(data) * 0.001, horizon)
        return np.full(horizon, lastVal) + noise
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
    print("E022: Horizon-Aware Flat Detection")
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
    flatTypes = ['horizontal', 'diagonal', 'mean_reversion', 'near_flat']

    methods = {
        'fixed': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'windowBased': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'seasonalCorr': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'diffEntropy': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'combined': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
    }

    fixedDetector = FlatPredictionDetector(
        horizontalThreshold=0.01,
        diagonalThreshold=1e-8,
        varianceThreshold=0.0001
    )

    print("\n--- Phase 1: Flat predictions (expected: DETECT) ---\n")

    for dataName, df in datasets.items():
        values = df['value'].values.astype(np.float64)

        for flatType in flatTypes:
            flatPred = _generateFlatPrediction(values, horizon, flatType)

            fixedR = fixedDetector.detect(flatPred, values)
            windowR = _windowBasedDetection(flatPred, values)
            seasonR = _seasonalCorrelationDetection(flatPred, values)
            diffR = _diffEntropyDetection(flatPred)
            combR = _combinedDetection(flatPred, values)

            results = {
                'fixed': fixedR.isFlat,
                'windowBased': windowR,
                'seasonalCorr': seasonR,
                'diffEntropy': diffR,
                'combined': combR,
            }

            for method, detected in results.items():
                if detected:
                    methods[method]['TP'] += 1
                else:
                    methods[method]['FN'] += 1

            marks = " ".join(f"{'T' if v else 'F'}" for v in results.values())
            print(f"  {dataName:25s} | {flatType:15s} | {marks}")

    print("\n--- Phase 2: Normal predictions (expected: NOT DETECT) ---\n")

    for dataName, df in datasets.items():
        values = df['value'].values.astype(np.float64)

        for modelName in modelNames:
            try:
                normalPred = _generateNormalPrediction(values, modelName, horizon)

                fixedR = fixedDetector.detect(normalPred, values)
                windowR = _windowBasedDetection(normalPred, values)
                seasonR = _seasonalCorrelationDetection(normalPred, values)
                diffR = _diffEntropyDetection(normalPred)
                combR = _combinedDetection(normalPred, values)

                results = {
                    'fixed': fixedR.isFlat,
                    'windowBased': windowR,
                    'seasonalCorr': seasonR,
                    'diffEntropy': diffR,
                    'combined': combR,
                }

                for method, detected in results.items():
                    if detected:
                        methods[method]['FP'] += 1
                    else:
                        methods[method]['TN'] += 1

                anyFP = any(results.values())
                if anyFP:
                    marks = " ".join(f"{'F' if v else 'T'}" for v in results.values())
                    print(f"  {dataName:25s} | {modelName:10s} | {marks}")

            except Exception as e:
                print(f"  [!] {dataName}/{modelName}: {e}")

    print("\n" + "=" * 70)
    print("ANALYSIS: Method Comparison")
    print("=" * 70)

    print(f"\n  {'Method':<15s} | {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} | "
          f"{'Prec':>6s} {'Rec':>6s} {'F1':>6s} | {'FPR':>6s}")
    print("  " + "-" * 70)

    for method, counts in methods.items():
        tp, fp, tn, fn = counts['TP'], counts['FP'], counts['TN'], counts['FN']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        print(f"  {method:<15s} | {tp:4d} {fp:4d} {tn:4d} {fn:4d} | "
              f"{prec:6.3f} {rec:6.3f} {f1:6.3f} | {fpr:6.3f}")

    bestMethod = max(methods.keys(), key=lambda m: (
        2 * (methods[m]['TP'] / max(methods[m]['TP'] + methods[m]['FP'], 1)) *
        (methods[m]['TP'] / max(methods[m]['TP'] + methods[m]['FN'], 1)) /
        max(
            (methods[m]['TP'] / max(methods[m]['TP'] + methods[m]['FP'], 1)) +
            (methods[m]['TP'] / max(methods[m]['TP'] + methods[m]['FN'], 1)),
            0.001
        )
    ))

    bestCounts = methods[bestMethod]
    bestPrec = bestCounts['TP'] / max(bestCounts['TP'] + bestCounts['FP'], 1)
    bestRec = bestCounts['TP'] / max(bestCounts['TP'] + bestCounts['FN'], 1)
    bestF1 = 2 * bestPrec * bestRec / max(bestPrec + bestRec, 0.001)
    bestFPR = bestCounts['FP'] / max(bestCounts['FP'] + bestCounts['TN'], 1)

    fixedCounts = methods['fixed']
    fixedPrec = fixedCounts['TP'] / max(fixedCounts['TP'] + fixedCounts['FP'], 1)
    fixedRec = fixedCounts['TP'] / max(fixedCounts['TP'] + fixedCounts['FN'], 1)
    fixedF1 = 2 * fixedPrec * fixedRec / max(fixedPrec + fixedRec, 0.001)
    fixedFPR = fixedCounts['FP'] / max(fixedCounts['FP'] + fixedCounts['TN'], 1)

    print(f"\n  Best method: {bestMethod}")
    print(f"    F1:  {fixedF1:.3f} -> {bestF1:.3f} ({(bestF1-fixedF1)/max(fixedF1,0.001)*100:+.1f}%)")
    print(f"    FPR: {fixedFPR:.3f} -> {bestFPR:.3f} ({(fixedFPR-bestFPR)/max(fixedFPR,0.001)*100:+.1f}% reduction)")


if __name__ == '__main__':
    main()
