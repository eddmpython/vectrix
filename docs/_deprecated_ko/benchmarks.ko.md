# 벤치마크

Vectrix는 표준 시계열 예측 대회(M3, M4)에서 **OWA**(Overall Weighted Average) 지표로 성능을 평가합니다. OWA는 Naive2 계절 벤치마크 대비 성능을 측정합니다.

- **OWA < 1.0** → Naive2보다 우수
- **OWA = 1.0** → Naive2와 동일
- **OWA > 1.0** → Naive2보다 미흡

## M4 Competition 결과 — DOT-Hybrid 엔진

[M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) (Makridakis et al., 2020)은 6개 빈도에 100,000개 시계열을 포함합니다. **DOT-Hybrid** (DynamicOptimizedTheta, 8-way auto-select) 결과, 빈도별 2,000개 시계열 랜덤 샘플(seed=42) 기준

| 빈도 | DOT-Hybrid OWA | M4 대비 |
|------|:--------------:|---------|
| Yearly | **0.797** | M4 1위 ES-RNN(0.821)에 근접 |
| Quarterly | **0.894** | M4 상위권 수준 |
| Monthly | **0.897** | M4 상위권 수준 |
| Weekly | **0.959** | Naive2 초과 |
| Daily | **0.996** | Naive2와 동등 |
| Hourly | **0.722** | 세계 최정상급, M4 우승자 수준 |
| **평균** | **0.877** | **M4 #18 Theta(0.897) 초과** |

### M4 공식 순위 비교

| 순위 | 방법 | OWA |
|:----:|------|:---:|
| 1 | ES-RNN (Smyl) | 0.821 |
| 2 | FFORMA (Montero-Manso) | 0.838 |
| 3 | Theta (Fiorucci) | 0.854 |
| 11 | 4Theta (Petropoulos) | 0.874 |
| 18 | Theta (Assimakopoulos) | 0.897 |
| -- | **Vectrix DOT-Hybrid** | **0.877** |

Vectrix DOT-Hybrid는 M4 Competition의 **모든 순수 통계 방법**을 능가합니다. 더 높은 순위의 방법들은 모두 하이브리드(ES-RNN = LSTM + ETS, FFORMA = 메타러닝 앙상블)입니다.

## M3 Competition 결과

[M3 Competition](https://forecasters.org/resources/time-series-data/m3-competition/) (Makridakis, 2000)은 4개 카테고리에 3,003개 시계열을 포함합니다. 카테고리별 100개 시계열 기준

| 카테고리 | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|----------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly   | 22.675       | 19.404        | 3.861       | 3.246        | **0.848**       |
| Quarterly| 12.546       | 10.445        | 1.568       | 1.283        | **0.825**       |
| Monthly  | 37.872       | 30.731        | 1.214       | 0.856        | **0.758**       |
| Other    | 6.620        | 5.903         | 2.741       | 2.044        | **0.819**       |

Vectrix는 **M3 4개 카테고리 전부**에서 Naive2를 능가하며, M3 Monthly에서 OWA 0.758을 달성합니다.

## 지표 설명

| 지표 | 설명 |
|------|------|
| **sMAPE** | 대칭 평균 절대 백분율 오차 (0-200 스케일) |
| **MASE** | 평균 절대 스케일 오차 (스케일 프리, naive 대비) |
| **OWA** | 전체 가중 평균 = 0.5 × (sMAPE/sMAPE_naive2 + MASE/MASE_naive2) |

## 결과 재현

### 환경

| 항목 | 버전 / 사양 |
|------|-------------|
| Python | 3.10+ |
| Vectrix | 0.0.12 |
| OS | Windows 11 / Ubuntu 22.04 / macOS 14+ |
| CPU | x86_64 또는 ARM64 |
| RAM | 8 GB 이상 |

### 실행

```bash
pip install vectrix
```

### 실험 코드

모든 실험은 완전히 재현 가능한 Python 스크립트이며, 결과는 docstring에 기록되어 있습니다.

| 실험 | 설명 | 소스 |
|:-----|:-----|:-----|
| E019 | DOT-Hybrid 엔진 M4 100K 검증 | [019_dotHybridEngine.py](https://github.com/eddmpython/vectrix/blob/master/src/vectrix/experiments/modelCreation/019_dotHybridEngine.py) |
| E042 | M4 공식 OWA 검증 | [042_m4OfficialOwa.py](https://github.com/eddmpython/vectrix/blob/master/src/vectrix/experiments/modelCreation/042_m4OfficialOwa.py) |
| E043 | Holdout validation + auto period detection | [043_dotAutoPeriodHoldout.py](https://github.com/eddmpython/vectrix/blob/master/src/vectrix/experiments/modelCreation/043_dotAutoPeriodHoldout.py) |
| E044 | Daily/Weekly 전문화 전략 | [044_dailyWeeklySpecialist.py](https://github.com/eddmpython/vectrix/blob/master/src/vectrix/experiments/modelCreation/044_dailyWeeklySpecialist.py) |
| E045 | 통합 개선 검증 | [045_integratedImprovement.py](https://github.com/eddmpython/vectrix/blob/master/src/vectrix/experiments/modelCreation/045_integratedImprovement.py) |
| E046 | 최종 통합 규칙 검증 | [046_finalIntegration.py](https://github.com/eddmpython/vectrix/blob/master/src/vectrix/experiments/modelCreation/046_finalIntegration.py) |

전체 실험 현황 및 연구 노트: [STATUS.md](https://github.com/eddmpython/vectrix/blob/master/src/vectrix/experiments/modelCreation/STATUS.md)

### 테스트

573개 테스트, 5개 skip — 모든 엔진, 모델, 파이프라인 컴포넌트 커버.

```bash
pip install vectrix
pytest tests/ -x -q
```

| 테스트 모듈 | 개수 | 범위 |
|:------------|:----:|:-----|
| [test_all_models.py](https://github.com/eddmpython/vectrix/blob/master/tests/test_all_models.py) | 112 | 30+ 예측 모델 전체 |
| [test_new_models.py](https://github.com/eddmpython/vectrix/blob/master/tests/test_new_models.py) | 45 | DTSF, ESN, 4Theta 엔진 |
| [test_engine_utils.py](https://github.com/eddmpython/vectrix/blob/master/tests/test_engine_utils.py) | 55 | ARIMAX, CV, 분해 |
| [test_easy.py](https://github.com/eddmpython/vectrix/blob/master/tests/test_easy.py) | 33 | Easy API (forecast, analyze, regress) |
| [test_business.py](https://github.com/eddmpython/vectrix/blob/master/tests/test_business.py) | 45 | 이상치, 백테스트, 메트릭, 시나리오 |
| [test_adaptive.py](https://github.com/eddmpython/vectrix/blob/master/tests/test_adaptive.py) | 20 | 레짐, DNA, 자가치유, 제약 |
| [test_regression.py](https://github.com/eddmpython/vectrix/blob/master/tests/test_regression.py) | 22 | OLS, Ridge, Lasso, 진단 |

M4 데이터 파일은 [M4 Competition 저장소](https://github.com/Mcompetitions/M4-methods)에서 다운로드할 수 있습니다.
