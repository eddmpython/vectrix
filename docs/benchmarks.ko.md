# 벤치마크

Vectrix는 표준 시계열 예측 대회(M3, M4)에서 **OWA**(Overall Weighted Average) 지표로 성능을 평가합니다. OWA는 Naive2 계절 벤치마크 대비 성능을 측정합니다.

- **OWA < 1.0** → Naive2보다 우수
- **OWA = 1.0** → Naive2와 동일
- **OWA > 1.0** → Naive2보다 미흡

## M3 Competition 결과

[M3 Competition](https://forecasters.org/resources/time-series-data/m3-competition/) (Makridakis, 2000)은 4개 카테고리에 3,003개 시계열을 포함합니다. 카테고리별 100개 시계열 기준:

| 카테고리 | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|----------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly   | 22.675       | 24.305        | 3.861       | 4.006        | 1.055           |
| Quarterly| 12.546       | 11.316        | 1.568       | 1.366        | **0.887**       |
| Monthly  | 37.872       | 28.461        | 1.214       | 0.751        | **0.685**       |
| Other    | 6.620        | 6.161         | 2.741       | 2.209        | **0.868**       |

Vectrix는 **M3 4개 카테고리 중 3개**에서 Naive2를 능가하며, M3 Monthly에서 OWA 0.685를 달성합니다.

## M4 Competition 결과

[M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) (Makridakis et al., 2020)은 6개 빈도에 100,000개 시계열을 포함합니다. 빈도별 100개 시계열 기준:

| 빈도     | Naive2 sMAPE | Vectrix sMAPE | Naive2 MASE | Vectrix MASE | **Vectrix OWA** |
|----------|:------------:|:-------------:|:-----------:|:------------:|:---------------:|
| Yearly   | 13.493       | 12.092        | 4.369       | 3.827        | **0.886**       |
| Quarterly| 3.714        | 3.329         | 1.244       | 1.052        | **0.871**       |
| Monthly  | 8.943        | 9.386         | 0.923       | 0.999        | 1.066           |
| Weekly   | 10.534       | 7.785         | 0.857       | 0.554        | **0.693**       |
| Daily    | 2.652        | 3.196         | 1.122       | 1.361        | 1.209           |
| Hourly   | 6.814        | 6.524         | 0.987       | 0.973        | **0.972**       |

Vectrix는 **M4 6개 빈도 중 4개**에서 Naive2를 능가하며, M4 Weekly에서 OWA 0.693을 달성합니다.

## 지표 설명

| 지표 | 설명 |
|------|------|
| **sMAPE** | 대칭 평균 절대 백분율 오차 (0-200 스케일) |
| **MASE** | 평균 절대 스케일 오차 (스케일 프리, naive 대비) |
| **OWA** | 전체 가중 평균 = 0.5 × (sMAPE/sMAPE_naive2 + MASE/MASE_naive2) |

## 벤치마크 실행

```bash
# M3 Competition
python benchmarks/runM3.py --cat M3Month --n 100
python benchmarks/runM3.py --all --n 50

# M4 Competition
python benchmarks/runM4.py --freq Monthly --n 100
python benchmarks/runM4.py --all --n 50
```

### 사용 가능한 카테고리

**M3**: `M3Year`, `M3Quart`, `M3Month`, `M3Other`

**M4**: `Yearly`, `Quarterly`, `Monthly`, `Weekly`, `Daily`, `Hourly`

## 결과 재현

```bash
git clone https://github.com/eddmpython/vectrix.git
cd vectrix
pip install -e .
python benchmarks/runM3.py --all --n 100
python benchmarks/runM4.py --all --n 100
```

결과는 `benchmarks/m3Results.csv`와 `benchmarks/m4Results.csv`에 저장됩니다.
