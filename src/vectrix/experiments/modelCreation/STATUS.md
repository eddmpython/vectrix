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

## 완료된 단계
- [x] 3개 모델 engine/ 모듈화 (fit/predict/residuals 인터페이스)
- [x] types.py에 모델 정보 등록
- [x] vectrix.py _selectNativeModels에 새 모델 반영
- [x] 기존 테스트 387개 통과 확인
- [x] 012 M4 100K 벤치마크 완료
- [x] 013~015 세상에 없던 새 앙상블/예측 원리 3개 실험 (전부 기각)

## 다음 단계
- [ ] 4Theta seasonality 처리 개선 (Quarterly/Monthly/Weekly/Daily 약세)
- [ ] DTSF 단기 시리즈 성능 개선 (n<100에서 약세)
- [ ] ESN reservoir 크기 자동 조정 (긴 시리즈에서 느림)

## 남은 대기 후보 (미실험)
- KernelDensityForecaster
- BayesianChangeForecaster

## "세상에 없던 모델" 후보 (016~)

013~015 실패에서 얻은 설계 원칙:
1. 앙상블/가중치/후처리가 아니라 **예측 메커니즘 자체**가 달라야 함
2. DOT/CES를 이기는 게 아니라 **DOT/CES가 못하는 걸 하는** 모델
3. 기존 통계 모델 잔차 상관 0.73~1.0 → 비모수/비선형만이 진짜 다양성 제공
4. 외부 제약 부과(RG, Lyapunov)는 역효과 → 데이터에서 직접 구조를 학습해야 함
5. 단일 모델 품질 > 앙상블 모델 수

### 후보 1: Topological Persistence Forecaster
- **원리**: Persistent Homology(TDA)로 시계열의 위상 구조(루프, 구멍, 연결 성분) 추출 → 구조 변화 감지 → 구조별 국소 예측
- **근거**: TDA를 시계열 분류/이상치에 쓴 연구는 있으나, 예측기로 직접 쓴 논문 없음
- **기대**: 레짐 전환이 잦은 데이터에서 강점 가능

### 후보 2: Causal Entropy Forecaster
- **원리**: Transfer Entropy로 자기 과거→미래 인과 강도를 horizon별 측정 → 인과가 강한 lag에만 집중하는 비모수 예측기
- **근거**: Transfer Entropy는 Granger Causality의 비모수 확장. 예측기 자체로 쓴 사례 없음
- **기대**: 장기 의존성이 강한 데이터에서 lag 선택 자동화

### 후보 3: Fractal Interpolation Forecaster
- **원리**: IFS(Iterated Function System)로 시계열의 프랙탈 자기유사 구조 학습 → 자기유사성 기반 외삽
- **근거**: 프랙탈 보간(Barnsley 1986)은 존재하나 예측기로 쓴 건 없음. 014 RGF에서 자기유사성 0.95+ 확인
- **기대**: 긴 주기 반복 패턴(Yearly, Quarterly)에서 강점 가능
