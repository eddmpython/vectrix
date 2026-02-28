"""
==============================================================================
실험 ID: ensembleEvolution/002
실험명: Residual Boosting Forecaster — 잔차 부스팅 모델 창조
==============================================================================

목적:
- 001에서 발견: 통계 모델 잔차 상관 0.73~1.0 → 동질적 앙상블 불가
- 새로운 접근: 앙상블이 아닌 "부스팅" — 1차 모델 잔차를 2차 모델이 학습
- Gradient Boosting의 시계열 버전: 이전 단계의 실패를 다음 단계가 보정

가설:
1. 2단계 부스팅(1차 + 잔차 모델)이 단일 모델보다 MAPE 5%+ 개선
2. 이질적 모델 조합 (ETS + Fourier, ARIMA + Theta)이 동질적 조합보다 우수
3. 3단계 이상은 과적합 → 2단계가 최적

방법:
1. ResidualBoostingForecaster 클래스 구현
   - Stage 1: 기본 모델 학습 → 잔차 계산
   - Stage 2: 잔차 전용 모델 학습 → 잔차의 잔차 계산
   - (옵션) Stage 3: 잔차의 잔차 전용 모델
   - 최종 예측 = Stage1 + Stage2 + Stage3
2. 조합 실험: 6가지 (1차, 2차) 조합
3. 기존 단일 모델과 MAPE/RMSE 비교

성공 기준:
- 부스팅 모델이 단일 최적 모델보다 MAPE 3%+ 개선
- 최소 5/13 데이터셋에서 1위

==============================================================================
결과 (실험 후 작성)
==============================================================================

수치 (10개 데이터셋):
| 지표 | 값 |
|------|-----|
| Boost wins vs Single | 1/10 (10%) — volatile에서만 |
| Best combo | arima+theta (s=0.5), dot+arima (s=0.5/1.0) 각 2회 |
| 2-stage median MAPE | 11.68% |
| 3-stage median MAPE | 12.29% |
| Shrinkage 0.5 median | 9.10% (최적) |

핵심 발견:
1. 가설 1 기각: 부스팅이 단일 모델보다 거의 항상 나쁨 (1/10 승)
2. 가설 2 검증 불가: 모든 조합이 열위
3. 가설 3 부분 채택: 3-stage(12.29%) > 2-stage(11.68%) — 3단계가 약간 나쁨
4. Shrinkage 0.5가 1.0보다 안정적이나 여전히 단일 모델 열위

근본 원인 분석:
- Stage 1의 in-sample fitted 값과 out-of-sample predict 값이 다름
- 모델.predict(n)은 미래 n스텝 예측이지 학습 데이터 fitted가 아님
- 잔차가 "학습 데이터 내 오차"가 아니라 "자기 자신에 대한 예측 오차"
- 이 잔차에 2차 모델을 학습해도 미래 예측 오차를 보정하지 못함
- 근본적으로 fitted 값 추출 기능이 없는 것이 문제

결론: 기각
- 통계 모델의 in-sample fitted 추출 인터페이스 부재로 올바른 부스팅 불가
- 모든 vectrix 모델이 predict()만 제공, fitted() 미제공
- 부스팅을 제대로 하려면 fittedValues() 인터페이스 추가 필요
- ML 모델(LightGBM 등) 잔차 부스팅이 더 자연스러운 접근

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


class ResidualBoostingForecaster:
    """
    Residual Boosting Forecaster

    Gradient Boosting의 시계열 버전.
    Stage 1이 예측한 잔차를 Stage 2가 학습하여 보정.
    """

    def __init__(self, stage1Factory, stage2Factory, stage3Factory=None,
                 shrinkage: float = 1.0):
        self.stage1Factory = stage1Factory
        self.stage2Factory = stage2Factory
        self.stage3Factory = stage3Factory
        self.shrinkage = shrinkage

        self.stage1Model = None
        self.stage2Model = None
        self.stage3Model = None
        self.fitted = False
        self.residuals = None

    def fit(self, y: np.ndarray) -> 'ResidualBoostingForecaster':
        n = len(y)

        self.stage1Model = self.stage1Factory()
        self.stage1Model.fit(y)

        stage1Pred, _, _ = self.stage1Model.predict(n)
        if len(stage1Pred) < n:
            padded = np.full(n, stage1Pred[-1] if len(stage1Pred) > 0 else np.mean(y))
            padded[:len(stage1Pred)] = stage1Pred
            stage1Pred = padded

        residual1 = y - stage1Pred[:n]

        if np.std(residual1) < 1e-10:
            self.fitted = True
            self.residuals = residual1
            return self

        self.stage2Model = self.stage2Factory()
        self.stage2Model.fit(residual1)

        if self.stage3Factory is not None:
            stage2Pred, _, _ = self.stage2Model.predict(n)
            if len(stage2Pred) < n:
                padded = np.full(n, stage2Pred[-1] if len(stage2Pred) > 0 else 0)
                padded[:len(stage2Pred)] = stage2Pred
                stage2Pred = padded

            residual2 = residual1 - stage2Pred[:n] * self.shrinkage

            if np.std(residual2) > 1e-10:
                self.stage3Model = self.stage3Factory()
                self.stage3Model.fit(residual2)

        fittedTotal = stage1Pred[:n]
        if self.stage2Model:
            s2p, _, _ = self.stage2Model.predict(n)
            if len(s2p) >= n:
                fittedTotal += s2p[:n] * self.shrinkage
        if self.stage3Model:
            s3p, _, _ = self.stage3Model.predict(n)
            if len(s3p) >= n:
                fittedTotal += s3p[:n] * self.shrinkage ** 2

        self.residuals = y - fittedTotal
        self.fitted = True
        return self

    def predict(self, steps: int):
        if not self.fitted:
            raise ValueError("Model not fitted")

        pred1, _, _ = self.stage1Model.predict(steps)
        pred1 = pred1[:steps]
        total = pred1.copy()

        if self.stage2Model:
            pred2, _, _ = self.stage2Model.predict(steps)
            pred2 = pred2[:steps]
            total += pred2 * self.shrinkage

        if self.stage3Model:
            pred3, _, _ = self.stage3Model.predict(steps)
            pred3 = pred3[:steps]
            total += pred3 * self.shrinkage ** 2

        residStd = np.std(self.residuals) if self.residuals is not None else 1.0
        margin = 1.96 * residStd * np.sqrt(1 + np.arange(steps) * 0.05)
        lower = total - margin
        upper = total + margin

        return total, lower, upper


def _modelFactory(name: str):
    factories = {
        'ets': lambda: ETSModel(),
        'arima': lambda: ARIMAModel(),
        'theta': lambda: OptimizedTheta(),
        'mstl': lambda: AutoMSTL(),
        'auto_ces': lambda: AutoCES(),
        'dot': lambda: DynamicOptimizedTheta(),
    }
    return factories.get(name, lambda: ETSModel())()


def _computeMetrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    mask = np.abs(actual) > 1e-8
    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mape, rmse


def main():
    print("=" * 70)
    print("E029: Residual Boosting Forecaster")
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

    horizon = 14

    boostingCombos = {
        'mstl+theta': (lambda: AutoMSTL(), lambda: OptimizedTheta()),
        'mstl+arima': (lambda: AutoMSTL(), lambda: ARIMAModel()),
        'arima+theta': (lambda: ARIMAModel(), lambda: OptimizedTheta()),
        'theta+ces': (lambda: OptimizedTheta(), lambda: AutoCES()),
        'dot+arima': (lambda: DynamicOptimizedTheta(), lambda: ARIMAModel()),
        'ets+dot': (lambda: ETSModel(), lambda: DynamicOptimizedTheta()),
    }

    boostingCombos3 = {
        'mstl+theta+arima': (lambda: AutoMSTL(), lambda: OptimizedTheta(), lambda: ARIMAModel()),
        'arima+theta+ces': (lambda: ARIMAModel(), lambda: OptimizedTheta(), lambda: AutoCES()),
    }

    shrinkageValues = [0.5, 0.8, 1.0]

    singleModels = ['mstl', 'arima', 'theta', 'auto_ces', 'dot']

    results = {}

    print("\n--- Running benchmarks ---\n")

    for dataName, values in datasets.items():
        results[dataName] = {}
        trainData = values[:len(values) - horizon]
        testData = values[len(values) - horizon:]

        for modelName in singleModels:
            try:
                model = _modelFactory(modelName)
                model.fit(trainData)
                pred, _, _ = model.predict(horizon)
                mape, rmse = _computeMetrics(testData, pred[:len(testData)])
                results[dataName][modelName] = {'mape': mape, 'rmse': rmse}
            except Exception:
                results[dataName][modelName] = {'mape': np.nan, 'rmse': np.nan}

        for comboName, (f1, f2) in boostingCombos.items():
            for shrink in shrinkageValues:
                fullName = f"boost_{comboName}_s{shrink}"
                try:
                    model = ResidualBoostingForecaster(f1, f2, shrinkage=shrink)
                    model.fit(trainData)
                    pred, _, _ = model.predict(horizon)
                    mape, rmse = _computeMetrics(testData, pred[:len(testData)])
                    results[dataName][fullName] = {'mape': mape, 'rmse': rmse}
                except Exception:
                    results[dataName][fullName] = {'mape': np.nan, 'rmse': np.nan}

        for comboName, (f1, f2, f3) in boostingCombos3.items():
            fullName = f"boost3_{comboName}"
            try:
                model = ResidualBoostingForecaster(f1, f2, f3, shrinkage=0.8)
                model.fit(trainData)
                pred, _, _ = model.predict(horizon)
                mape, rmse = _computeMetrics(testData, pred[:len(testData)])
                results[dataName][fullName] = {'mape': mape, 'rmse': rmse}
            except Exception:
                results[dataName][fullName] = {'mape': np.nan, 'rmse': np.nan}

    print("\n" + "=" * 70)
    print("ANALYSIS 1: Best Boosting vs Best Single per Dataset")
    print("=" * 70)

    boostWins = 0
    singleWins = 0

    for dataName in sorted(datasets.keys()):
        dataResults = results[dataName]

        bestSingle = min(
            [(m, dataResults[m]['mape']) for m in singleModels
             if not np.isnan(dataResults.get(m, {}).get('mape', np.nan))
             and dataResults.get(m, {}).get('mape', np.inf) < 1e6],
            key=lambda x: x[1],
            default=('none', np.inf)
        )

        boostModels = [m for m in dataResults if m.startswith('boost')]
        bestBoost = min(
            [(m, dataResults[m]['mape']) for m in boostModels
             if not np.isnan(dataResults.get(m, {}).get('mape', np.nan))
             and dataResults.get(m, {}).get('mape', np.inf) < 1e6],
            key=lambda x: x[1],
            default=('none', np.inf)
        )

        if bestBoost[1] < bestSingle[1] * 0.99:
            marker = "BOOST WINS"
            boostWins += 1
        else:
            marker = "single wins"
            singleWins += 1

        improvement = (bestSingle[1] - bestBoost[1]) / bestSingle[1] * 100 if bestSingle[1] > 0 else 0

        print(f"  {dataName:25s} | Single: {bestSingle[0]:10s} {bestSingle[1]:8.2f}% | "
              f"Boost: {bestBoost[0]:30s} {bestBoost[1]:8.2f}% | {improvement:+.1f}% | {marker}")

    print(f"\n  Boost wins:  {boostWins}/{boostWins + singleWins}")
    print(f"  Single wins: {singleWins}/{boostWins + singleWins}")

    print("\n" + "=" * 70)
    print("ANALYSIS 2: Best Boosting Combination")
    print("=" * 70)

    comboWins = {}
    for dataName in datasets:
        dataResults = results[dataName]
        boostModels = [(m, dataResults[m]['mape']) for m in dataResults
                       if m.startswith('boost') and not np.isnan(dataResults[m].get('mape', np.nan))
                       and dataResults[m].get('mape', np.inf) < 1e6]

        if boostModels:
            bestCombo = min(boostModels, key=lambda x: x[1])[0]
            comboWins[bestCombo] = comboWins.get(bestCombo, 0) + 1

    sortedCombos = sorted(comboWins.items(), key=lambda x: -x[1])
    for combo, wins in sortedCombos[:10]:
        print(f"  {combo:40s} | wins = {wins}")

    print("\n" + "=" * 70)
    print("ANALYSIS 3: Shrinkage Effect")
    print("=" * 70)

    for shrink in shrinkageValues:
        shrinkMapes = []
        for dataName in datasets:
            for comboName in boostingCombos:
                fullName = f"boost_{comboName}_s{shrink}"
                mape = results[dataName].get(fullName, {}).get('mape', np.nan)
                if not np.isnan(mape) and mape < 1e6:
                    shrinkMapes.append(mape)

        if shrinkMapes:
            print(f"  Shrinkage {shrink}: avg MAPE = {np.mean(shrinkMapes):.2f}%, "
                  f"median = {np.median(shrinkMapes):.2f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS 4: 2-Stage vs 3-Stage")
    print("=" * 70)

    stage2Mapes = []
    stage3Mapes = []

    for dataName in datasets:
        for fullName in results[dataName]:
            mape = results[dataName][fullName].get('mape', np.nan)
            if np.isnan(mape) or mape > 1e6:
                continue
            if fullName.startswith('boost3_'):
                stage3Mapes.append(mape)
            elif fullName.startswith('boost_'):
                stage2Mapes.append(mape)

    if stage2Mapes:
        print(f"  2-Stage: avg MAPE = {np.mean(stage2Mapes):.2f}%, "
              f"median = {np.median(stage2Mapes):.2f}%")
    if stage3Mapes:
        print(f"  3-Stage: avg MAPE = {np.mean(stage3Mapes):.2f}%, "
              f"median = {np.median(stage3Mapes):.2f}%")

    print("\n" + "=" * 70)
    print("ANALYSIS 5: Full Rankings (top models per dataset)")
    print("=" * 70)

    for dataName in sorted(datasets.keys()):
        allModelsData = [(m, results[dataName][m]['mape']) for m in results[dataName]
                         if not np.isnan(results[dataName][m].get('mape', np.nan))
                         and results[dataName][m].get('mape', np.inf) < 1e6]
        allModelsData.sort(key=lambda x: x[1])

        print(f"\n  [{dataName}] top-5:")
        for rank, (m, mape) in enumerate(allModelsData[:5]):
            marker = "***" if m.startswith('boost') else "   "
            print(f"    {rank+1}. {m:35s} MAPE={mape:8.2f}% {marker}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"""
    Boost wins: {boostWins}/{boostWins + singleWins} ({boostWins/(boostWins+singleWins)*100:.1f}%)
    Best combos: {sortedCombos[:3] if sortedCombos else 'None'}
    """)


if __name__ == '__main__':
    main()
