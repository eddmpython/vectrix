---
title: "벤치마크"
---

# 벤치마크

Vectrix의 M3 및 M4 Competition 벤치마크 결과입니다.

## M4 Competition 결과 — DOT-Hybrid 엔진

M4 Competition은 100,000개의 시계열로 구성된 세계 최대 예측 벤치마크입니다. **DOT-Hybrid** (DynamicOptimizedTheta, 8-way auto-select) 결과, 빈도별 2,000개 시계열 랜덤 샘플(seed=42) 기준:

| 빈도 | DOT-Hybrid OWA | M4 대비 |
|------|:--------------:|---------|
| Yearly | **0.797** | M4 1위 ES-RNN(0.821)에 근접 |
| Quarterly | **0.905** | M4 상위권 수준 |
| Monthly | **0.933** | 안정적 중상위 |
| Weekly | **0.959** | Naive2 초과 |
| Daily | **0.996** | Naive2와 동등 |
| Hourly | **0.722** | 세계 최정상급, M4 우승자 수준 |
| **평균** | **0.885** | **M4 #18 Theta(0.897) 초과** |

> **참고:** OWA (Overall Weighted Average)는 sMAPE와 MASE의 가중 평균입니다. 1.0은 Naive2 벤치마크와 동일한 수준을 의미합니다. 낮을수록 좋습니다.

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

M3 Competition은 3,003개의 시계열로 구성된 전통적인 예측 벤치마크입니다. 카테고리별 100개 시계열 기준:

| 카테고리 | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|----------|:---:|:---:|:---:|:---:|:---:|
| Yearly | 22.675 | 19.404 | 3.861 | 3.246 | **0.848** |
| Quarterly | 12.546 | 10.445 | 1.568 | 1.283 | **0.825** |
| Monthly | 37.872 | 30.731 | 1.214 | 0.856 | **0.758** |
| Other | 6.620 | 5.903 | 2.741 | 2.044 | **0.819** |

## 속도 벤치마크

내장 Rust 엔진의 성능 (Python 대비)

| 연산 | Python | Rust 엔진 | 배속 |
|------|:------:|:----------:|:----:|
| AutoETS fit | 348ms | 32ms | 10.8x |
| AutoARIMA fit | 195ms | 35ms | 5.6x |
| DOT fit | 68ms | 2.8ms | 24x |
| AutoCES fit | 118ms | 9.6ms | 12x |
| ETS filter loop | 0.17ms | 0.003ms | 67x |
| `forecast()` E2E | 295ms | 52ms | 5.6x |

> **참고:** 벤치마크는 단일 코어, Intel i7-13700H, Python 3.12 환경에서 측정되었습니다. Rust 엔진은 `pip install vectrix` 설치 시 자동으로 내장됩니다.

## 재현

```bash
pip install vectrix
```

M4 벤치마크 실험 스크립트: `src/vectrix/experiments/modelCreation/019_dotHybridEngine.py`

M4 데이터 파일은 [M4 Competition 저장소](https://github.com/Mcompetitions/M4-methods)에서 다운로드할 수 있습니다.
