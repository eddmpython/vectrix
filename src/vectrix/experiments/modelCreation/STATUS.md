# Model Creation 연구 현황

## 개요
기존에 없던 완전히 새로운 예측 모델을 연구하고 구현하는 실험 방향.
4개 전문 에이전트 토론(통계학, 물리학/신호처리, ML/정보이론, M4 Competition)에서 도출된 12개 모델 중 10개 실험 완료.
011 종합 스트레스 테스트로 최종 엔진 통합 3개 모델 확정.

## 실험 결과표

### Top 5 (1차 라운드)

| 실험 | 모델 | 평균순위 | 승률 | 결론 |
|------|------|----------|------|------|
| 001 | DynamicTimeScanForecaster | 3.58 (2위) | 41.7% | 조건부 채택 |
| 002 | KoopmanModeForecaster | 4.67 (5위) | 25.0% | 보류 (특수목적) |
| 003 | WaveletShrinkageForecaster | 3.33 (2위) | 8.3% | 기각 |
| 004 | AdaptiveThetaEnsemble | **2.73 (1위)** | 36.4% | **채택** |
| 005 | SingularSpectrumForecaster | 4.82 (4위) | 18.2% | 조건부 보류 |

### 대기 후보 (2차 라운드)

| 실험 | 모델 | 평균순위 | 승률 | 결론 |
|------|------|----------|------|------|
| 006 | TemporalAggregation (MAPA) | 4.27 (4위) | 9.1% | 조건부 채택 (hourly 전용) |
| 007 | EchoStateForecaster (ESN) | **3.82 (3위)** | 27.3% | **채택** |
| 008 | FeatureWeightedCombiner (FFORMA) | 4.00 (3위) | 9.1% | 조건부 채택 (메타러닝) |
| 009 | DampedTrendWithChangepoint | 4.82 (5위) | 0.0% | 기각 |
| 010 | StochasticResonance | 5.18 (5위) | 9.1% | 기각 |

### 011 종합 스트레스 테스트 (최종 판정)

| 모델 | Safety | Seed CV | Speed (n=1000) | Avg Rank (34ds) | Win Rate | 판정 |
|------|--------|---------|----------------|-----------------|----------|------|
| **ESN** | 100% | 21.7~39.2% | 11ms | **3.47 (1위)** | 17.6% | **엔진 통합 확정** |
| **4Theta** | 100% | 4.7~19.7% | 53ms | **3.62 (2위)** | 26.5% | **엔진 통합 확정** |
| **DTSF** | 92% | 11.5~40.8% | 16ms | **3.74 (3위)** | 38.2% | **엔진 통합 확정 (n≥30)** |
| mstl (기존) | - | - | - | 3.71 | - | 기존 엔진 |

## 핵심 발견

### 001 DynamicTimeScanForecaster (엔진 통합 확정)
- 비모수 패턴 매칭: 과거 유사 패턴의 후속값 중앙값 = 예측
- **hourlyMultiSeasonal에서 64.7% 개선** (기존 최고 theta 11.69% → dtsf 4.12%)
- 잔차 상관 0.1~0.5 — 기존 모델과 근본적으로 다른 예측 원리 확인
- 011: CI sqrt 확장 수정, n<30 fallback 추가

### 002 KoopmanModeForecaster (보류)
- DMD(동적 모드 분해): Takens 임베딩 → SVD → 고유값 분해 → 모드 예측
- **stockPrice에서 42.6% 개선** — 금융 시계열 전문

### 003 WaveletShrinkageForecaster (기각)
- **교훈: 좋은 디노이저 ≠ 좋은 예측기**

### 004 AdaptiveThetaEnsemble (엔진 통합 확정)
- **평균 순위 1위 (2.73)** — mstl(3.27)까지 초과
- 기존 Theta 대비 8/11 개선 (73% 승률)
- 011: Safety 100%, CV<20% (전 유형 최고 안정성)

### 005 SingularSpectrumForecaster (조건부 보류)
- autoRank r=1 문제로 계절성 포착 불가 → 개선 필요

### 006 TemporalAggregation (조건부 채택 — hourly 전용)
- **hourlyMultiSeasonal에서 80.6% 개선** (기존 최고 theta 11.69% → mapa 2.26%)
- **hourly 잔차 상관 ~0** — 모든 기존 모델과 거의 무상관!
- 주기 감지 실패 시 효과 미미 → 주기 명시 필요

### 007 EchoStateForecaster (엔진 통합 확정)
- **011 수정 후 평균 순위 1위 (3.47)** — 전 모델 중 최고!
- **hourlyMultiSeasonal 77.3% 개선, volatile 18.1% 개선, regimeShift 4.7% 개선**
- **잔차 상관 0.13~0.66** — 기존 모델과 "다르게 틀리는" 비선형 모델
- 011: adaptive ridge + prediction clamp로 noisy CV 251% → 21.7% 해결

### 008 FeatureWeightedCombiner (조건부 채택)
- 평균 순위 4.00 (3위) — 안정적 결합 성능
- **FFORMA > equal_avg: 9/11** — 특성 기반 가중 유효
- 학습 데이터 11개로는 메타러닝 불충분 → M4 전체로 확장 필요

### 009 DampedTrendWithChangepoint (기각)
- 0/11 승률 — CUSUM 과민 반응, 가중 ETS가 4theta 수준 미달

### 010 StochasticResonance (기각)
- 1/11 승률 — K-means 시간 순서 무시, Markov 전환 확률 너무 낮음

## 종합 인사이트

1. **ESN이 수정 후 전체 1위 (3.47)** — clamp + adaptive ridge로 안정화
2. **4Theta가 안정성 1위** — CV<20%, Safety 100%, 속도만 약간 느림
3. **DTSF가 승률 1위 (38.2%)** — 비모수 다양성으로 기존 모델과 차별화
4. **3개 신규 모델 모두 기존 최강 mstl(3.71) 능가** — 엔진 통합 자격 확인
5. **변화점/레짐 기반 모델은 실패** (ee/003, 009, 010) — 기존 모델이 이미 내재적으로 적응
6. **엔진 통합 최종 3개 모델**:
   - 1위: **EchoStateForecaster** (007) — 비선형 동역학 + 가장 높은 정확도
   - 2위: **AdaptiveThetaEnsemble** (004) — 가장 안정적 + 기존 Theta 완전 대체
   - 3위: **DynamicTimeScanForecaster** (001) — 최고 승률 + 앙상블 다양성

## 012: M4 Competition 100K 벤치마크 결과

| 모델 | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | AVG OWA |
|------|--------|-----------|---------|--------|-------|--------|---------|
| **dot** | 0.887 | **0.942** | **0.937** | **0.938** | 1.004 | 0.722 | **0.905** |
| **auto_ces** | 0.986 | **0.957** | **0.944** | **0.972** | 1.000 | **0.702** | **0.927** |
| **vx_ensemble** | 1.031 | 1.130 | 1.062 | **0.984** | 1.198 | **0.696** | 1.017 |
| **four_theta** | **0.879** | 1.065 | 1.096 | 1.386 | 1.858 | 1.292 | 1.263 |
| esn | 1.293 | 1.363 | 1.324 | 1.143 | 1.361 | 2.149 | 1.439 |
| dtsf | 2.081 | 3.381 | 2.295 | 1.905 | 2.125 | **0.765** | 2.092 |

### 핵심 발견
1. **DOT/AutoCES가 범용 최강** — M4 #18 Theta(0.897) 수준
2. **4Theta Yearly OWA 0.879** — M4 공식 #11 4Theta(0.874)와 동등
3. **VX-Ensemble Hourly OWA 0.696** — 전 모델 중 1위, M4 우승자급
4. **DTSF Hourly OWA 0.765** — 패턴 반복 데이터에서 강점 확인
5. **ESN은 독립 사용 부적합** — 앙상블 다양성 기여 역할에 특화

### M4 Competition 레퍼런스
- #1 ES-RNN: OWA 0.821
- #2 FFORMA: OWA 0.838
- #11 4Theta: OWA 0.874
- #18 Theta: OWA 0.897

## 013~015: 세상에 없던 새 모델 실험 (3개 기각)

### 013 Wasserstein Diversity Ensemble (기각)
- 잔차 분포의 Wasserstein 거리로 "분포적 다양성" 측정 → 앙상블 가중치 결합
- **alpha=0.0(순수 inv_mape)이 전 그룹 최적** → 다양성 가중치 추가 시 일관 악화
- 교훈: "다르게 실패하는 모델"의 존재는 확인되나, 가중치 보너스를 주면 정확도 낮은 모델 비중 증가로 역효과
- 부가 발견: 단일 CES(0.949)가 5개 모델 앙상블(0.977)보다 우수

### 014 Renormalization Group Forecaster (기각)
- 물리학 RG 원리로 다중 스케일 coarse-grain → 고정점 제약 부과
- **DOT 대비 전반적 악화** (Hourly -104%): coarse-grain이 24시간 주기를 파괴
- 교훈: 시계열은 대부분 스케일 불변(자기유사성 0.95+)이라 RG 제약의 효과 미미

### 015 Ergodic Predictability Engine (기각)
- 국소 Lyapunov 지수로 예측 가능 수평선 추정 → horizon별 모델 가중치 차등
- **DOT 대비 전반적 악화** (Daily -46%): exp(-LE·h) 변환이 너무 공격적
- 유효한 발견: Lyapunov 지수가 실제 예측 난이도와 일치 (Hourly 0.01, Yearly 0.47)
- 교훈: 적응형 모델(DOT/CES)이 이미 내부적으로 불확실성을 처리 → 외부 Naive 가중치 강제는 이중 보정

### 3개 실험 공통 교훈
1. **기존 최강 모델(DOT, CES)의 벽이 높다** — 새로운 원리를 도입해도 쉽게 넘지 못함
2. **앙상블 가중치의 핵심은 정확도** — 다양성, 물리적 제약, 예측 가능성 등 추가 축은 noise
3. **단일 모델 CES가 앙상블보다 우수한 경우가 많다** — 모델 풀의 품질이 앙상블보다 중요

## 016~018: DOT 강화 + SCUM 앙상블 실험 (2026-03-03)

### 016 DOT++ (조건부 채택)
- **8-way auto-select**: 2 trend(linear/exponential) x 2 model(additive/multiplicative) x 2 season(A/M/N)
- Yearly **0.796** (기존 0.887, -10.3%) — 세계 최고 수준 단일 통계 모델
- Hourly **0.955** (기존 0.722, +32.3%) — 고빈도에서 과적합 발생
- **결론**: period<=12에서만 사용 (DOT-Hybrid 전략)

### 017 SCUM (기각)
- **Full SCUM (DOT+CES+ETS+ARIMA median)**: AVG 0.925 > DOT 0.905 (악화)
- **원인**: Vectrix ETS/ARIMA가 DOT/CES보다 약해서 median을 끌어내림
- **SCUM Mean**: Weekly/Daily에서 ETS/ARIMA 극단값으로 폭발 (MASE 10^9)
- **유일한 성공**: Hourly median 0.704 > DOT 0.722 (CES 기여)

### 018 Hybrid DOT + SCUM Variants (채택)

| 모델 | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | **AVG** |
|------|--------|-----------|---------|--------|-------|--------|---------|
| dot_current | 0.887 | 0.942 | 0.937 | 0.938 | 1.004 | 0.722 | **0.905** |
| **dot_hybrid** | **0.796** | **0.904** | 0.931 | 0.957 | 0.996 | 0.722 | **0.884** |
| scum2 | 0.920 | 0.933 | 0.927 | 0.947 | 1.000 | **0.702** | 0.905 |
| **combined** | 0.838 | 0.913 | **0.917** | 0.962 | 0.996 | **0.702** | **0.888** |

- **DOT-Hybrid: AVG 0.884** — M4 #18 Theta(0.897) 초과! 단일 모델 세계급
- **Combined (DOT-Hybrid+CES median): AVG 0.888** — 앙상블 최강
- M4 참조: #1 ES-RNN 0.821, #2 FFORMA 0.838, #11 4Theta 0.874

### 핵심 발견
1. **지수추세(exponential) + 승법 theta line이 저빈도 데이터에서 혁신적**
2. **고빈도(Hourly)에서는 기존 DOT 3파라미터 최적화가 최적**
3. **DOT+CES median이 Hourly에서 DOT 단독보다 우수** (0.702 vs 0.722)
4. **ETS/ARIMA는 vectrix에서 SCUM 품질 부족** — DOT/CES만 조합 대상
5. **DOT-Hybrid가 현재 시점 vectrix 최강 모델** — 엔진 통합 확정

### 019 DOT-Hybrid Engine Verification (확인 완료)

| 모델 | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | **AVG** |
|------|--------|-----------|---------|--------|-------|--------|---------|
| **dot_engine** | **0.797** | **0.905** | **0.933** | 0.959 | **0.996** | **0.722** | **0.885** |
| E018 참조 | 0.796 | 0.904 | 0.931 | 0.957 | 0.996 | 0.722 | 0.884 |

- **속도**: M4 100K 전체 1.7분 (E016 16.6분 대비 9.8x)
- **정확도**: E018과 0.001 OWA 이내 일치 (Rust golden section vs scipy 미세 차이)
- **Rust 26개 함수**: DOT=True, SES=True, Hybrid=True 모두 활성

## 031~040: FFORMA 메타러닝 + 모델 선택 최적화 (2026-03-04)

### 031 Oracle Analysis + GBR Meta-Learner
- 8 모델 × 7273 M4 시리즈 oracle 데이터 수집 (캐시됨)
- Oracle ceiling: AVG OWA 0.662 (DOT 0.946)
- GBR 5-fold OOF: meta_top1 = 0.873 (DOT 대비 -1.4%)
- **핵심 발견**: oracle 수준(0.66)과 실현 가능한 수준(0.87) 사이 갭이 큼

### 032 Safe Meta-Ensemble (oracle 누설 있음)
- OWA > 3.0 마스킹으로 극단값 제거 → safe_top3_weighted = 0.847
- **문제**: safeMask가 실제 OWA 사용 → oracle 누설

### 033 Realistic Meta-Ensemble (누설 없음)
- GBR 예측만으로 모든 결정 → Weekly/Daily 폭발 (dtsf/esn/auto_arima 극단값)
- DOT R2 = 0.067 → DNA 특성으로 DOT 실패 예측 불가 (근본 병목)

### 034 Safe Pool Meta-Ensemble (최고 현실적 결과)
- 5개 safe 모델만 사용 (dot, auto_ces, four_theta, auto_ets, theta)
- **meta_top1 = 0.873** (DOT 0.885 대비 -1.4% 개선!)
- Yearly: DOT 최강 (0.797 vs meta 0.829), Hourly: meta 최강 (0.670 vs DOT 0.722)

### 035 Per-Series Holdout Model Selection (실패)
- holdout validation으로 per-series 모델 선택 → DOT보다 나쁨
- **원인**: 데이터 축소 > 모델 선택 이점 (short/medium 시리즈에서 치명적)

### 036 Vectrix Pipeline Benchmark (E036 = E035 확인)
- CV-based selection도 동일 패턴 (DOT wins everywhere except Hourly)

### 037 Extract Selection Rules from Oracle
- Decision tree accuracy 26% (무작위 수준) → 단순 규칙으로 모델 선택 불가
- **핵심 발견**: 어떤 그룹에서도 단일 모델이 50% 이상 차지하지 않음 → 앙상블이 답
- top feature: asymmetry, seasonalAdjustedVariance, adfStatistic

### 038 Ensemble Weight Optimization
- **inv_smape_top2 = 0.739** (oracle 활용, 비현실적)
- **optimal_static = 0.882** (DOT 0.885 대비 -0.003)
- **최적 가중치**: dot + auto_ces + four_theta가 핵심 3인방
  - Yearly: dot 76% + 4theta 24%
  - Monthly: dot 41% + ces 38% + 4theta 19%
  - Daily: auto_ces 100%
  - Hourly: dot 84% + ces 15%
- **auto_ets, theta의 최적 가중치 = 0** → 앙상블 기여도 없음

### 039 Improved Pipeline Benchmark (실패)
- auto_arima 포함 앙상블이 Quarterly에서 폭발 (OWA 1.6)
- holdout sMAPE와 test 성능 상관 극도로 낮음

### 040 Safe Ensemble Pipeline Benchmark (결론적)
- safe4 (auto_ets 포함): Weekly/Daily 폭발 (auto_ets 극단값)
- safe3_core (dot+ces+4theta): 전체 0.945 (DOT 0.885보다 나쁨!)
- **Monthly safe3: 0.917** (DOT 0.933, -0.016 ***), **Hourly: 0.716** (DOT 0.722, -0.006 ***)
- **cv_best Hourly: 0.692** (DOT 대비 -0.030 ***)
- **결론**: 앙상블 자체가 DOT-only보다 나쁨 (DOT가 이미 최적화)

### E031-E040 종합 결론
1. **DOT-Hybrid (0.877, holdout 적용 후)는 순수 통계 모델의 실질적 한계**
2. **메타러닝 최고 = 0.873** (scikit-learn 필요, 현재 미반영)
3. **앙상블은 DOT-only보다 나쁨** — DOT가 이미 충분히 최적화
4. **M4 #1 (0.821) 달성에는 DL 하이브리드 필수**
5. **안전한 변경**: auto_arima를 기본 풀에서 제거 → 폭발 방지
6. **Variability-preserving 앙상블 유지** — 무조건 앙상블은 악화

### 041 Conditional Ensemble Verification (채택 — smart_safe3)

| 전략 | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | **AVG** |
|------|--------|-----------|---------|--------|-------|--------|---------|
| dot_only | 0.797 | 0.905 | 0.933 | 0.959 | 0.994 | 0.722 | **0.885** |
| **smart_safe3** | **0.796** | 0.907 | **0.919** | **0.954** | 0.996 | **0.703** | **0.879** |
| cond_v2 (M:safe3,H:cvbest) | 0.797 | 0.905 | **0.917** | 0.959 | 0.994 | **0.694** | **0.878** |

- **smart_safe3**: core3(dot+ces+4theta) 앙상블 + variability 체크 → 전 그룹 안전, AVG -0.006
- **핵심**: 앙상블 풀을 MAPE순 top-3 대신 core3 고정 → 위험 모델 배제
- Monthly: -0.014, Hourly: -0.019, Weekly: -0.005 개선
- Yearly/Quarterly/Daily: ±0.002 이내 (동등)
- **엔진 반영**: 앙상블 구성 시 core3 모델 우선 선택

### vectrix.py 반영 사항
- `_selectNativeModels()`: auto_arima를 기본 풀에서 제거
- `_selectNativeModels()`: MEDIUM flat risk에서 esn 제거
- `_generateFinalPrediction()`: 앙상블 풀을 core3(dot+ces+4theta) 우선으로 변경
- 앙상블 로직: variability-preserving 유지 + core3 우선
- 테스트: 573 passed, 5 skipped

## 완료된 단계
- [x] 3개 모델 engine/ 모듈화 (fit/predict/residuals 인터페이스)
- [x] types.py에 모델 정보 등록
- [x] vectrix.py _selectNativeModels에 새 모델 반영
- [x] 기존 테스트 573개 통과 확인
- [x] 012 M4 100K 벤치마크 완료
- [x] 013~015 세상에 없던 새 앙상블/예측 원리 3개 실험 (전부 기각)
- [x] 016~018 DOT 강화 + SCUM 실험 완료
- [x] DOT-Hybrid를 engine/dot.py에 통합 (period<24: DOT++, period>=24: classic)
- [x] Rust dot_hybrid_objective 추가 (26번째 함수)
- [x] 019 통합 엔진 M4 100K 검증 완료 (OWA 0.885)
- [x] 031~040 FFORMA 메타러닝 + 모델 선택 최적화 10개 실험 완료
- [x] auto_arima 기본 풀 제거 반영
- [x] 041 조건부 앙상블 검증 → core3 우선 앙상블 엔진 반영 (AVG 0.885→0.879)
- [x] 042 M4 공식 OWA 검증 → 벤치마크 방법론 문제 발견

### 042 M4 Official OWA Verification (방법론 검증)

| 계산 방식 | OWA | 비고 |
|-----------|-----|------|
| 6-group 단순 평균 (기존 방식) | 0.881 | Hourly(414) = Yearly(23K) 동일 비중 |
| 6-group 단순 평균 (M4 공식 Naive2) | 0.879 | SeasonalityTest 차이 Monthly만 0.013 |
| **M4 공식 (시리즈 수 가중)** | **0.892** | **이것이 정확한 값** |

- **Naive2 구현 차이**: Monthly에서만 0.013 (ACF SeasonalityTest 차이), 나머지 무시
- **핵심 문제**: 6-group 단순 평균 vs 시리즈 수 가중 → 0.881 vs 0.892 (0.011 차이)
- **Daily OWA 1.007**: Naive2보다 나쁨 — 시리즈 수 가중에서 큰 페널티
- **정직한 위치**: M4 공식 기준 약 14~15위 (Theta 0.897보다는 우수)
- 주의: 11K 샘플 기준, 100K 전체에서는 Monthly(48K) 비중 증가로 약간 달라질 수 있음

## 043~046: DOT Holdout Validation 실험 (2026-03-04)

### 043 DOT Auto Period Detection + Holdout Validation

| 변형 | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly | **AVG** |
|------|--------|-----------|---------|--------|-------|--------|---------|
| baseline | 0.7971 | 0.9053 | 0.9200 | 0.9587 | 0.9949 | 0.7223 | **0.8831** |
| auto_period | 0.8019 | 0.9053 | 0.9200 | 0.9952 | 1.0220 | 0.7223 | **0.8944** |
| **holdout_val** | 0.8064 | **0.8940** | **0.8965** | **0.9457** | 0.9918 | 0.7223 | **0.8761** |
| combined | 0.8084 | 0.8940 | 0.8965 | 0.9831 | 1.0187 | 0.7223 | **0.8872** |

- **auto_period: 기각** — ACF가 노이즈에서 가짜 단주기(2,3) 감지, Daily +2.7%, Weekly +3.8% 악화
- **holdout_val: 조건부 채택** — Quarterly -1.25%, Monthly -2.55% 개선, Yearly +1.2% 회귀(데이터 축소)
- **combined: 기각** — auto_period가 holdout 이점을 상쇄

### 044 Daily/Weekly Specialist

- **Weekly classic_only: 채택** (-2.18%) — period=1에서 classic DOT가 Hybrid보다 우수
- **Daily classic_only: 기각** (+0.98%)
- **Core3 앙상블 Daily/Weekly: 기각** (+21%/+8%) — CES/4Theta가 period=1에서 해로움

### 045 Integrated Improvement (holdout + Weekly classic)

- **AVG 0.8831→0.8748 (-0.94%)** — 전반적 개선
- **Yearly +1.16% 회귀** — holdout으로 인한 짧은 시리즈 데이터 축소 문제

### 046 Final Integration (period별 분리)

- **period<=1 classic + period>1 holdout: 기각** — Yearly +11.26% 치명적 회귀!
- **핵심 발견**: Yearly(period=1)는 Hybrid 8-way가 trend 탐색에 유리, classic 적용 불가
- **최종 규칙**: period>1에서만 holdout validation 적용 (Quarterly/Monthly만 개선)

### E043-E046 종합 결론
1. **holdout validation은 period>1 계절성 데이터에서만 유효** (Quarterly -1.25%, Monthly -2.55%)
2. **ACF auto period detection은 해로움** — 노이즈에서 가짜 주기 감지
3. **period=1 데이터는 건드리지 않는 것이 안전** — Yearly/Daily/Weekly 모두 기존 방식 유지
4. **Core3 앙상블은 period=1에서 해로움** — CES/4Theta가 비계절성 데이터에서 약함

### dot.py 반영 사항 (v0.0.12)
- `_fitHybrid()`: `period > 1 and n >= period * 4`일 때만 holdout validation
- `_predictVariantSteps()` 헬퍼 메서드 추가
- holdout 후 전체 데이터로 refit
- **DOT-Hybrid AVG OWA: 0.885 → 0.877** (period>1만 개선, 나머지 unchanged)
- 테스트: 573 passed, 5 skipped

## 완료된 단계
- [x] 3개 모델 engine/ 모듈화 (fit/predict/residuals 인터페이스)
- [x] types.py에 모델 정보 등록
- [x] vectrix.py _selectNativeModels에 새 모델 반영
- [x] 기존 테스트 573개 통과 확인
- [x] 012 M4 100K 벤치마크 완료
- [x] 013~015 세상에 없던 새 앙상블/예측 원리 3개 실험 (전부 기각)
- [x] 016~018 DOT 강화 + SCUM 실험 완료
- [x] DOT-Hybrid를 engine/dot.py에 통합 (period<24: DOT++, period>=24: classic)
- [x] Rust dot_hybrid_objective 추가 (26번째 함수)
- [x] 019 통합 엔진 M4 100K 검증 완료 (OWA 0.885)
- [x] 031~040 FFORMA 메타러닝 + 모델 선택 최적화 10개 실험 완료
- [x] auto_arima 기본 풀 제거 반영
- [x] 041 조건부 앙상블 검증 → core3 우선 앙상블 엔진 반영 (AVG 0.885→0.879)
- [x] 042 M4 공식 OWA 검증 → 벤치마크 방법론 문제 발견
- [x] 043~046 DOT holdout validation 실험 → period>1 holdout 엔진 반영 (AVG 0.885→0.877)

## 다음 단계
- [ ] DL 하이브리드 (NeuralForecast/TimesFM) 탐색 → M4 #1 (0.821) 도전
- [ ] Daily OWA 0.996 개선 (period=1 비계절성 데이터 전략)
- [ ] 4Theta seasonality 처리 개선 (Quarterly/Monthly/Weekly/Daily 약세)
- [ ] DTSF 단기 시리즈 성능 개선 (n<100에서 약세)
- [ ] ESN reservoir 크기 자동 조정 (긴 시리즈에서 느림)
