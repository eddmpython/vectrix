---
title: "변경 로그"
---

# 변경 로그

Vectrix의 모든 주요 변경 사항을 기록합니다.

이 형식은 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)를 기반으로 하며, [Semantic Versioning](https://semver.org/spec/v2.0.0.html)을 준수합니다.

## [0.0.8] - 2026-03-03

내장 Rust 엔진 릴리즈 — Rust 가속을 전체 엔진으로 확장 (13 → 25개 함수), 모든 wheel에 컴파일. `[turbo]` 없이 `pip install vectrix`만으로 Rust 엔진이 포함됩니다. Polars처럼.

### 추가

**Rust 엔진 확장 (13 → 25개 함수)**
- GARCH: `garch_filter`, `egarch_filter`, `gjr_garch_filter`
- TBATS: `tbats_filter`
- DTSF: `dtsf_distances`, `dtsf_fit_residuals` (O(n²) 패턴 매칭 — 최대 속도 향상)
- MSTL: `mstl_extract_seasonal`, `mstl_moving_average`
- Croston: `croston_tsb_filter`
- ESN: `esn_reservoir_update`
- 4Theta: `four_theta_fitted`, `four_theta_deseasonalize`

**CI/CD: macOS x86_64 wheel 추가**
- `macos-13` 빌드 타겟 추가 — 4개 플랫폼 빌드 (Linux, macOS ARM, macOS x86, Windows)

### 변경

- 전체 문서 업데이트: "선택적 Rust turbo" → "내장 Rust 엔진"
- 모든 설치 가이드에서 `[turbo]` 제거
- 랜딩 페이지 전면 재작성: Hero, Features, Install, Performance 섹션

---

## [0.0.7] - 2026-03-02

llms.txt를 통한 즉각적인 AI 이해, MCP 서버를 통한 도구 사용, Claude Code 스킬을 통한 워크플로우 자동화를 포함한 AI 통합 릴리즈.

### 추가

**llms.txt / llms-full.txt**
- `llms.txt`: [llms.txt 표준](https://llmstxt.org/)을 따르는 구조화된 프로젝트 개요 -- 문서 링크, 빠른 시작, API 섹션
- `llms-full.txt`: 완전한 API 레퍼런스 (모든 클래스, 메서드, 파라미터, 반환 타입, 흔한 실수) -- AI가 한 번 읽으면 전체 라이브러리를 이해
- GitHub Pages 루트에 배포: `eddmpython.github.io/vectrix/llms.txt`, `llms-full.txt`
- PyPI 패키지에 포함하여 로컬 접근 가능

**MCP 서버 (Model Context Protocol)**
- 10개 도구: `forecast_timeseries`, `forecast_csv`, `analyze_timeseries`, `compare_models`, `run_regression`, `detect_anomalies`, `backtest_model`, `list_sample_datasets`, `load_sample_dataset`
- 2개 리소스: `vectrix://models`, `vectrix://api-reference`
- 2개 프롬프트: `forecast_workflow`, `regression_workflow`
- Claude Desktop, Claude Code 및 모든 MCP 클라이언트와 호환
- 설정: `pip install "vectrix[mcp]"` + `claude mcp add`

**Claude Code 스킬 (3개)**
- `vectrix-forecast`: 시계열 예측 워크플로우 (전체 API 레퍼런스 포함)
- `vectrix-analyze`: DNA 프로파일링, 이상치 감지, 레짐 분석
- `vectrix-regress`: R 스타일 회귀분석, 진단, 변수 선택
- 프로젝트 디렉토리에서 자동 로드, `/vectrix-forecast` 등으로 호출

### 변경

- `pyproject.toml`: `mcp` 선택적 의존성 추가, llms.txt/mcp wheel 포함 설정
- `docs.yml`: llms.txt, llms-full.txt를 GitHub Pages 배포 디렉토리에 복사
- README.md / README_KR.md: "AI Integration" 섹션 추가 (llms.txt, MCP, Skills)

[0.0.7]: https://github.com/eddmpython/vectrix/compare/v0.0.6...v0.0.7

## [0.0.6] - 2026-03-02

튜토리얼, 쇼케이스, EasyForecastResult 개선, SvelteKit 랜딩 + MkDocs GitHub Pages 통합 배포를 포함한 문서 및 배포 릴리즈.

### 추가

**EasyForecastResult 개선**
- `compare()`: sMAPE, MAPE, RMSE, MAE 지표를 포함한 모델 비교 테이블
- `all_forecasts()`: 수동 분석을 위한 모든 유효 모델 예측값 DataFrame
- 정확도 속성: EasyForecastResult에 `.mape`, `.rmse`, `.mae`, `.smape` 빠른 접근
- `Vectrix._refitAllModels()`: compare/all_forecasts를 위해 모든 유효 모델 재적합

**튜토리얼 (마크다운, 6개 주제 x 2개 언어 = 12개 파일)**
- 01_quickstart: 한 줄 예측, 결과 확인, 시각화
- 02_analyze: DNA 프로파일링, 특성 지문, 변화점 감지
- 03_regression: R 스타일 수식 회귀, 진단, 강건 방법
- 04_models: 30+ 모델 카탈로그, 수동 선택, 비교 워크플로우
- 05_adaptive: 레짐 감지, 자가 치유, 제약, Forecast DNA
- 06_business: 이상치 감지, 시나리오 분석, 백테스팅, 비즈니스 지표

**쇼케이스 (marimo 인터랙티브 노트북)**
- 03_modelComparison: 30+ 모델 비교 및 DNA 분석
- 04_businessIntelligence: 이상치 감지, 시나리오, 백테스팅
- GitHub Pages용 .md 페이지 (8개 파일)

### 변경

**통합 GitHub Pages 배포**
- SvelteKit 랜딩 페이지가 루트(`/vectrix/`)에서 제공
- MkDocs 문서가 `/vectrix/docs/`에서 제공
- `docs.yml` 워크플로우가 SvelteKit + MkDocs를 모두 빌드하고 단일 배포로 병합
- 모든 랜딩 페이지 링크가 `/vectrix/docs/` 경로를 가리키도록 업데이트
- SvelteKit `paths.base`가 `BASE_PATH` 환경 변수로 설정

**문서 네비게이션**
- mkdocs.yml nav에 튜토리얼 및 쇼케이스 하위 페이지 추가
- showcase/index, tutorials/index에 콘텐츠 설명 업데이트
- README.md, README_KR.md 업데이트: 573 테스트, compare API, 신규 모델, 문서 링크

[0.0.6]: https://github.com/eddmpython/vectrix/compare/v0.0.5...v0.0.6

## [0.0.5] - 2026-03-02

DOT, CES, 4Theta 모델에 Rust turbo 가속을 확장한 성능 릴리즈 (vectrix-core 0.2.0).

### 변경

**Rust Turbo 모드 확장 (vectrix-core 0.2.0)**
- DOT (Dynamic Optimized Theta): 68ms -> 2.8ms (24배 빠름) -- `dot_objective`, `dot_residuals` Rust 핫 패스
- AutoCES (Complex Exponential Smoothing): 118ms -> 9.6ms (12배 빠름) -- `ces_nonseasonal_sse`, `ces_seasonal_sse` Rust 핫 패스
- 4Theta (Adaptive Theta Ensemble): 63ms -> 5.6ms (11배 빠름) -- 기존 `ses_sse`/`ses_filter` Rust 함수 연결
- 총 13개 Rust 가속 함수 (기존 9개에서 4개 추가)
- 3단계 폴백 유지: Rust > Numba JIT > 순수 Python
- 모든 함수가 Python 레퍼런스 구현과 비트 단위 동일한 결과 생성

[0.0.5]: https://github.com/eddmpython/vectrix/compare/v0.0.4...v0.0.5

## [0.0.4] - 2026-03-02

전체 영문 docstring 전환, 573개 테스트(+186), DOT/CES 기본 후보 추가를 포함한 품질 및 국제화 릴리즈.

### 변경

**영문 Docstring 전환**
- 60개 이상 소스 모듈 전체에서 한국어->영어 전환 완료
- 모든 docstring, 오류 메시지, 주석, 사용자 대면 문자열이 영어로 변환
- API Reference 문서 (mkdocstrings)가 영어로 올바르게 렌더링
- 한국어 DataFrame 자동 감지 키워드(`'날짜', '일자', '일시'`)는 유지

**모델 선택 개선**
- DOT (Dynamic Optimized Theta)와 AutoCES가 기본 모델 후보에 포함
- M4 검증: DOT OWA 0.905 (#18 수준), AutoCES OWA 0.927 -- 범용 최강 모델
- Hourly 데이터: 다중 계절 패턴 포착을 위해 DTSF + MSTL 우선 적용
- 대체 모델이 four_theta/esn에서 dot/auto_ces로 업그레이드

### 추가

**테스트 커버리지 확장 (387 -> 573, +48%)**
- `test_new_models.py`: DTSF, ESN, 4Theta 45개 테스트 (패턴 매칭, 비선형, M4 스타일 holdout)
- `test_business.py`: 이상치 감지, 백테스팅, 지표, what-if, 리포트, HTML 리포트 45개 테스트
- `test_infrastructure.py`: flat defense, 계층 조정, 배치, 영속성, TSFrame, AutoAnalyzer 43개 테스트
- `test_engine_utils.py`: ARIMAX, 교차 검증, 분해, 진단, 주기적 drop, 비교, 결측치 보간 53개 테스트

### 수정

- 영문 오류 메시지에 맞게 테스트 assertion 업데이트 (pipeline, holiday names)
- FlatPredictionType enum 주석 번역

[0.0.4]: https://github.com/eddmpython/vectrix/compare/v0.0.3...v0.0.4

## [0.0.3] - 2026-02-28

Rust 가속 코어 루프 (vectrix-core), 내장 샘플 데이터셋, pandas 2.x 호환성 수정을 포함한 성능 릴리즈.

### 추가

**Rust Turbo 모드 (vectrix-core)**
- PyO3 + maturin을 통한 핵심 예측 핫 루프의 네이티브 Rust 확장
- 9개 가속 함수: `ets_filter`, `ets_loglik`, `css_objective`, `seasonal_css_objective`, `ses_sse`, `ses_filter`, `theta_decompose`, `arima_css`, `batch_ets_filter`
- 3단계 폴백: Rust > Numba JIT > 순수 Python -- 투명하게 동작, 코드 변경 불필요
- Linux (manylinux), macOS (x86 + ARM), Windows (x86_64), Python 3.10-3.13용 사전 빌드 wheel
- `pip install "vectrix[turbo]"`로 설치 -- 사용자에게 Rust 컴파일러 불필요
- `core-v*` 태그 시 자동 wheel 빌드 GitHub Actions CI 워크플로우 (`publish-core.yml`)

**내장 샘플 데이터셋**
- 빠른 테스트를 위한 7개 결정적 샘플 데이터셋: `airline` (144 월간), `retail` (730 일간), `stock` (252 영업일), `temperature` (1095 일간), `energy` (720 시간), `web` (180 일간), `intermittent` (365 일간)
- `loadSample(name)`: 샘플 데이터셋을 DataFrame으로 로드
- `listSamples()`: 메타데이터와 함께 모든 사용 가능한 데이터셋 목록 표시
- 41개 테스트로 모든 데이터셋 커버

### 변경

**성능 개선**
- AutoETS: 348ms -> 32ms (Rust turbo로 10.8배 빠름)
- AutoARIMA: 195ms -> 35ms (5.6배 빠름)
- Theta: 1.3ms -> 0.16ms (8.1배 빠름)
- `forecast()` 엔드투엔드: 295ms -> 52ms (5.6배 빠름)
- ETS filter 핫 루프: 0.17ms -> 0.003ms (67배 빠름)
- ARIMA CSS objective: 0.19ms -> 0.001ms (157배 빠름)

### 수정

- pandas 2.x 빈도 지원 중단 대응: `"M"` -> `"ME"`, `"Q"` -> `"QE"`, `"Y"` -> `"YE"`, `"H"` -> `"h"`

### 변경 (문서)

- i18n 플러그인을 사용한 완전한 이중 언어 문서 사이트 (영어/한국어)
- 설치 가이드에 Rust Turbo 모드 섹션 추가
- README에 turbo 벤치마크, 샘플 데이터셋, 비교표 업데이트
- 387개 테스트 (346개에서 증가), 5개 건너뜀 (선택적 의존성 가드)

[0.0.3]: https://github.com/eddmpython/vectrix/compare/v0.0.2...v0.0.3

## [0.0.2] - 2026-02-28

Foundation Model 래퍼, 딥러닝 모델, 다변량 예측, 확률적 분포, 다국가 휴일, 파이프라인 시스템을 포함한 기능 확장 릴리즈.

### 추가

**Foundation Model 래퍼 (선택적)**
- `ChronosForecaster`: Amazon Chronos-2 제로샷 예측 래퍼 (배치 예측 및 분위수 출력)
- `TimesFMForecaster`: Google TimesFM 2.5 래퍼 (공변량 지원 및 다중 호라이즌 예측)
- 선택적 의존성 그룹: `foundation` (torch + chronos-forecasting), `neural` (neuralforecast)

**딥러닝 모델 래퍼 (선택적)**
- `NeuralForecaster`: NBEATS, NHITS, TFT 아키텍처를 지원하는 NeuralForecast 래퍼
- 편의 클래스: `NBEATSForecaster`, `NHITSForecaster`, `TFTForecaster`
- 자동 numpy / NeuralForecast DataFrame 상호 변환

**확률적 예측 분포**
- `ForecastDistribution`: 파라메트릭 분포 예측 (Gaussian, Student-t, Log-Normal)
- `DistributionFitter`: AIC 비교를 통한 자동 분포 선택
- `empiricalCRPS`: 닫힌 형태 Gaussian CRPS + 기타 분포용 Monte Carlo CRPS
- 전체 분포 API: quantile, interval, sample, pdf, crps 메서드

**다변량 모델**
- `VARModel`: 자동 래그 선택(AIC/BIC)과 Granger 인과성 검정을 포함한 벡터 자기회귀
- `VECMModel`: Johansen 스타일 공적분 순위 추정을 포함한 벡터 오차 수정 모델

**다국가 휴일 지원**
- 미국 휴일: 고정 4개 (새해, 독립기념일, 재향군인의 날, 크리스마스) + 이동 6개 (MLK, 대통령의 날, 현충일, 노동절, 콜럼버스 데이, 추수감사절)
- 일본 휴일: 고정 13개 국경일
- 중국 휴일: 고정 5개 (원단, 노동절, 국경절)
- `getHolidays(year)`: KR/US/JP/CN 통합 휴일 조회
- `adjustForecast()`: 추정된 이벤트 효과를 점 예측에 적용
- `_nthWeekdayOfMonth()`: 이동 휴일 날짜 계산 헬퍼

**파이프라인 시스템**
- `ForecastPipeline`: sklearn 스타일 순차 체이닝 (예측값에 자동 역변환 포함)
- 8개 내장 변환기: `Differencer`, `LogTransformer`, `BoxCoxTransformer`, `Scaler` (zscore/minmax), `Deseasonalizer`, `Detrend`, `OutlierClipper`, `MissingValueImputer`
- 이름 기반 단계 접근, 파라미터 중첩 (`step__param`), repr 표시

### 변경

**속도 개선**
- `Vectrix._evaluateNativeModels`에서 `ThreadPoolExecutor`를 통한 모델 평가 병렬화
- M3/M4 벤치마크 러너에서 교차 검증 후보 루프 병렬화
- M3 Monthly 벤치마크 ~13% 빠름 (11.22s -> 9.70s)

**테스트 커버리지**
- 346개 테스트 (275개에서 증가), 5개 건너뜀 (선택적 의존성 가드)

[0.0.2]: https://github.com/eddmpython/vectrix/compare/v0.0.1...v0.0.2

## [0.0.1] - 2026-02-27

순수 NumPy + SciPy로 구축된 제로 설정 시계열 예측 라이브러리 Vectrix의 최초 공개 릴리즈.

### 추가

**핵심 예측 엔진 (30+ 모델)**
- AutoETS: AICc 모델 선택을 포함한 30가지 Error x Trend x Seasonal 조합 (Hyndman-Khandakar 단계적 알고리즘)
- AutoARIMA: 단계적 차수 선택을 포함한 계절 ARIMA, CSS objective 함수
- Theta / Dynamic Optimized Theta (DOT): 원본 Theta 방법 + M3 Competition 우승 방법론
- AutoCES: 복소 지수 평활 (Svetunkov 2023)
- AutoTBATS: 복잡한 다중 계절 시계열을 위한 삼각함수 계절성
- GARCH / EGARCH / GJR-GARCH: 비대칭 효과를 포함한 조건부 변동성 모델링
- Croston Classic / SBA / TSB / AutoCroston: 간헐적 및 덩어리 수요 예측
- Logistic Growth: 사용자 정의 용량 제약이 있는 Prophet 스타일 포화 추세
- AutoMSTL: ARIMA 잔차 예측을 포함한 다중 계절 STL 분해
- 기준선 모델: Naive, Seasonal Naive, Mean, Random Walk with Drift, Window Average

**독창적 방법**
- Lotka-Volterra Ensemble: 적응형 모델 가중치를 위한 생태학적 경쟁 동역학
- Phase Transition Forecaster: 레짐 전환 예측을 위한 임계 둔화 감지
- Adversarial Stress Tester: 예측 견고성 분석을 위한 5가지 교란 연산자 (spike, dropout, drift, noise, swap)
- Hawkes Intermittent Demand: 군집화된 수요 패턴을 위한 자기 흥분 점 과정
- Entropic Confidence Scorer: Shannon 엔트로피 기반 예측 불확실성 정량화

**적응형 지능**
- 레짐 감지: 순수 numpy Hidden Markov Model 구현 (Baum-Welch + Viterbi)
- 자가 치유 예측: CUSUM + EWMA 드리프트 감지와 Conformal Prediction 교정
- 제약 인식 예측: 8가지 비즈니스 제약 유형 (non-negative, range, capacity, YoY change, sum, monotone, ratio, custom)
- Forecast DNA: 메타러닝 모델 추천과 유사도 검색을 포함한 65+ 시계열 특성 지문
- Flat Defense: 평탄 예측 실패 방지를 위한 4단계 시스템 (진단, 감지, 교정, 예방)

**Easy API**
- `forecast()`: 자동 모델 선택을 포함한 단일 호출 예측, str/DataFrame/Series/ndarray/list/tuple/dict 지원
- `analyze()`: 시계열 DNA 프로파일링, 변화점 감지, 이상치 식별
- `regress()`: R 스타일 수식 회귀 (`y ~ x1 + x2`) + 완전한 진단
- `quick_report()`: 통합 분석 + 예측 리포트 생성
- 풍부한 결과 객체: `.plot()`, `.to_csv()`, `.to_json()`, `.to_dataframe()`, `.summary()`, `.describe()`

**회귀분석 & 진단**
- 5가지 회귀 방법: OLS, Ridge, Lasso, Huber, Quantile
- R 스타일 수식 인터페이스: `regress(data=df, formula="sales ~ ads + price")`
- 완전한 진단 도구: Durbin-Watson, Breusch-Pagan, VIF, Jarque-Bera 정규성 검정
- 변수 선택: 단계적 (전진/후진), 정규화 CV, 최적 부분 집합
- 시계열 회귀: Newey-West HAC, Cochrane-Orcutt, Prais-Winsten, Granger 인과성

**비즈니스 인텔리전스**
- 자동화된 이상치 식별과 자연어 설명을 포함한 이상치 감지
- What-if 분석: 파라미터 교란을 통한 시나리오 기반 예측 시뮬레이션
- 백테스팅: MAE, RMSE, MAPE, SMAPE 지표를 포함한 롤링 오리진 교차 검증
- 계층 조정: Bottom-up, Top-down, MinTrace 최적 (Wickramasuriya 2019)
- 예측 구간: Conformal Prediction + Bootstrap 방법

**인프라**
- ThreadPoolExecutor 병렬화를 포함한 배치 예측 API
- 모델 영속성: `.fxm` 바이너리 형식 (save/load/info 유틸리티)
- TSFrame: 빈도 감지를 포함한 시계열 DataFrame 래퍼
- Global model: 관련 시계열을 위한 교차 시리즈 학습
- Numba JIT 가속 (선택적 의존성)
- 모든 모델, 엣지 케이스, 통합 시나리오를 커버하는 275개 테스트
- GitHub Actions CI: 매트릭스 테스팅 (Python 3.10-3.13, Ubuntu + Windows)
- GitHub Actions를 통한 PyPI trusted publisher 배포

[0.0.1]: https://github.com/eddmpython/vectrix/releases/tag/v0.0.1
