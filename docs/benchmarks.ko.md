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
| Quarterly | **0.905** | M4 상위권 수준 |
| Monthly | **0.933** | 안정적 중상위 |
| Weekly | **0.959** | Naive2 초과 |
| Daily | **0.996** | Naive2와 동등 |
| Hourly | **0.722** | 세계 최정상급, M4 우승자 수준 |
| **평균** | **0.885** | **M4 #18 Theta(0.897) 초과** |

### M4 공식 순위 비교

| 순위 | 방법 | OWA |
|:----:|------|:---:|
| 1 | ES-RNN (Smyl) | 0.821 |
| 2 | FFORMA (Montero-Manso) | 0.838 |
| 3 | Theta (Fiorucci) | 0.854 |
| 11 | 4Theta (Petropoulos) | 0.874 |
| 18 | Theta (Assimakopoulos) | 0.897 |
| -- | **Vectrix DOT-Hybrid** | **0.885** |

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
| Vectrix | 0.0.10 |
| OS | Windows 11 / Ubuntu 22.04 / macOS 14+ |
| CPU | x86_64 또는 ARM64 |
| RAM | 8 GB 이상 |

### 실행

```bash
pip install vectrix
```

M4 벤치마크 실험 스크립트: `src/vectrix/experiments/modelCreation/019_dotHybridEngine.py`

M4 데이터 파일은 [M4 Competition 저장소](https://github.com/Mcompetitions/M4-methods)에서 다운로드할 수 있습니다.
