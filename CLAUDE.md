# Vectrix 프로젝트 규칙

## 프로젝트 목표 (2026-03-04 확정)

### 목표 1: 세계 1등 시계열 예측 정확도
- M4 Competition OWA 기준 #1 (0.821) 달성
- 현재: M4 공식 기준 OWA 0.892 (약 14~15위)
- Daily OWA 1.007 (Naive2보다 나쁨) — 최우선 개선 대상
- DL 하이브리드, 메타러닝, 앙상블 전략 모두 탐색

### 목표 2: 사용자 편의와 교육
- 시계열 예측을 처음 접하는 사람도 이해할 수 있는 교육 콘텐츠
- 블로그를 통해 기초 용어/개념부터 고급 기법까지 체계적으로 설명
- SEO 최적화로 검색 유입 극대화 — 퍼나를 수 있는 콘텐츠
- "시계열 예측 = Vectrix" 포지셔닝

## [최우선] API 스팩 기반 개발

- **`API_SPEC.md`가 Single Source of Truth** — 모든 코드, 문서, 노트북, 예제는 이 파일을 참조
- **API 변경 시 `API_SPEC.md`를 먼저 수정**, 그다음 코드 변경, 그다음 문서/노트북 반영
- **노트북/예제 코드 작성 시 반드시 `API_SPEC.md` 읽은 후 작성** — 기억에 의존하지 말 것
- **노트북 작성 후 반드시 실행 검증** (`uv run jupyter nbconvert --execute` 또는 셀별 검증)
- 존재하지 않는 메서드/속성을 상상으로 쓰지 말 것 (예: `.table()`, `dna.trendStrength`)

## 필수: 메모리 먼저 읽기

**작업 시작 전 반드시 메모리 파일을 읽을 것:**
- `C:\Users\MSI\.claude\projects\c--Users-MSI-OneDrive-Desktop-sideProject-vectrix\memory\MEMORY.md`

## 언어

- 사용자와의 대화는 **한국어**로 답변

## 네이밍 규칙

- 파일/폴더/함수/변수: **camelCase**
- 클래스: **PascalCase**
- 상수: **UPPER_CASE**
- snake_case 금지 (단, easy.py 공개 API는 예외: forecast(), analyze(), regress())

## 절대 규칙

- **UTF-8 인코딩 필수**: 모든 파일 작업, Python 실행 시 UTF-8 명시
- **파일 삭제 금지**: rm, delete, git rm 금지. 제거 시 `_backup/` 또는 `_deprecated/`로 이동
- 코드에 인라인 주석 금지
- try-except 남발 금지
- 불필요한 설명/부연 금지

## README 업데이트 정책

- **기능 추가/변경/개선 작업 후 README.md와 README_KR.md를 반드시 업데이트**
- 새로운 기능, API 변경, 모델 추가, 도구 추가 등 모든 의미 있는 변경사항을 반영
- 설명은 **최대한 자세하게** 작성 — 한 줄 요약 금지, 기능의 목적/사용법/예시를 포함
- README는 프로젝트의 얼굴이므로 항상 최신 상태 유지

## [최우선] Git 커밋 정책 - AI 흔적 완전 금지

- **커밋 메시지에 Claude, Opus, Anthropic, AI, noreply@anthropic.com 절대 금지**
- **Co-Authored-By 트레일러 절대 추가 금지 (어떤 형태로든)**
- **코드, 주석, docstring, 파일 어디에도 AI 작성 흔적 금지**
- **GitHub contributor에 AI가 표시되면 안 됨**
- 커밋 메시지 형식: `날짜 + 작업내용` (예: `2026-02-27 Regime-Aware Adaptive Forecasting`)
- 패키지 관리: **uv** 사용 (pip 직접 사용 금지)
- **커밋 & 푸시 시 항상 PyPI 배포까지 완료할 것** (태그 생성 → push → GitHub Actions 자동 배포)

## Marimo 노트북 규칙

- **marimo 파일 작성/수정 시 반드시 marimo skills(`.agents/skills/marimo-notebook/`)를 참조할 것**
- 첫 셀에 `import marimo as mo` 필수 + return에 `mo` 포함
- PEP 723 dependencies 블록에 실제 import하는 패키지 모두 명시
- `uvx marimo check <파일>`로 정적 분석 통과 확인
- 차트는 **plotly** 사용 (matplotlib 금지)

## 문서 (docs/) 규칙

### 튜토리얼 (`docs/tutorials/`)
- **Markdown(.md) 파일로 작성** — MkDocs가 GitHub Pages에서 직접 렌더링
- marimo 사용하지 않음
- 한영 이중 파일: `01_quickstart.md` (영문), `01_quickstart.ko.md` (한국어)
- 모든 코드 예제는 복사-붙여넣기로 바로 실행 가능하게 작성
- 출력 예시를 코드 블록 아래에 포함
- 설명은 충실하게 — 초보자도 따라할 수 있는 수준

### 블로그 (`docs/blog/`)
- **Markdown(.md) 파일로 작성** — MkDocs Material blog plugin 사용
- **영문 단일 파일**: `posts/XXX_slug.md` — 한국어 번역은 브라우저/MkDocs 번역 기능으로 대체
- 번호는 발행 순서: `001_`, `002_`, ... (camelCase slug)
- **SEO 필수 요소**: 모든 포스트에 meta description, og:tags, 구조화된 헤딩(H1>H2>H3)
- **대상 독자**: 예측/데이터분석을 처음 접하는 사람 ~ 중급 실무자
- **콘텐츠 원칙**:
  - "예측(Forecasting)" 전체를 포괄 — 시계열에만 국한하지 않음
  - 기초 용어/개념부터 시작 — 전문 지식 전제 금지
  - 실행 가능한 코드 예제 필수 — 복사-붙여넣기로 바로 동작
  - 시각적 설명 (수식보다 직관, 그래프, 비유 우선)
  - 각 포스트는 독립적으로 읽을 수 있어야 함 (시리즈 의존성 최소화)
  - "왜 이것이 중요한가"를 반드시 설명
- **카테고리**: fundamentals(기초), howto(실습), deepdive(심층), benchmark(벤치마크)
- **SEO 전략**: 롱테일 키워드 타겟 (예: "예측이란 무엇인가", "수요 예측 방법", "Python 예측 라이브러리 비교", "ARIMA vs ETS")

### 공통
- `index.md` (영문) / `index.ko.md` (한국어) 가 진입점
- i18n: 영문이 default, 한국어는 `.ko.md` 접미사
- 모든 설명은 충실하게 작성 — 한 줄 설명 금지, 맥락/의미/사용법을 상세히

## 랜딩 페이지 (SvelteKit)

- 경로: `landing/`
- 스택: **SvelteKit 5 + Tailwind CSS v4 + shadcn-svelte (수동 구성)**
- 컴포넌트: `landing/src/lib/components/ui/` (Button, Badge, Card)
- 섹션: `landing/src/lib/components/sections/` (Hero, Features, Performance 등)
- 폰트: **Pretendard Variable** (한글) + **Inter** (영문) + **JetBrains Mono** (코드)
- 빌드: `@sveltejs/adapter-static` → `landing/build/`에 정적 HTML 출력
- Git 포함: 소스코드 전체. **빌드 결과물(build/, .svelte-kit/, node_modules/)만 .gitignore**

## 프로젝트 구조

```
src/vectrix/
├── vectrix.py               # 메인 Vectrix 클래스
├── easy.py                    # 초보자 API (forecast, analyze, regress, quick_report)
├── types.py                   # 데이터 타입 + 모델 정보
├── engine/                    # 핵심 예측 엔진 (ETS, ARIMA, Theta, MSTL, TBATS, GARCH, CES, Croston, DOT)
├── adaptive/                  # 적응형 예측 (레짐, 자가치유, 제약, DNA)
├── regression/                # 완전한 회귀분석 (OLS, 진단, 강건, 시계열회귀)
├── business/                  # 비즈니스 인텔리전스 (이상치, 설명, 시나리오, 백테스트)
├── flat_defense/              # 평탄 예측 방어 (4단계 시스템)
├── models/                    # 모델 선택 + 앙상블
├── analyzer/                  # 자동 데이터 분석
├── global_model/              # 글로벌 예측 (크로스러닝)
├── hierarchy/                 # 계층적 조정 (BottomUp, TopDown, MinTrace)
├── intervals/                 # 예측 구간 (Conformal, Bootstrap)
├── ml/                        # 선택적 ML (LightGBM, XGBoost, sklearn)
├── tsframe/                   # 시계열 DataFrame 래퍼
└── tests/                     # 106개 테스트
```

## 버전 릴리즈 정책

- **Semantic Versioning** 준수: `MAJOR.MINOR.PATCH`
- **0.0.1부터 시작**, 큰 단위로 올리지 않는다
- PATCH (0.0.x): 버그 수정, 오타, 성능 개선
- MINOR (0.x.0): 새 기능 추가, 하위 호환 유지
- MAJOR (x.0.0): 하위 호환 깨지는 변경 (신중하게, 충분한 사유 필요)
- 버전 점프 금지 (0.0.1 → 0.0.2 → 0.0.3 순서대로)

### CHANGELOG.md 필수 작성
- 모든 릴리즈에 상세 변경 내역 기록
- 형식: `## [버전] - 날짜` 헤더
- 카테고리: Added, Changed, Fixed, Removed, Deprecated
- 각 항목에 변경 사유와 영향 범위 명시
- Breaking Change는 별도 섹션으로 강조

### GitHub Release 필수 작성
- **릴리즈 노트에 변경사항 상세 기록** (CHANGELOG 내용 기반)
- 릴리즈 제목: `v버전` (예: `v0.0.1`)
- 본문: CHANGELOG의 해당 버전 섹션 전체 포함
- 빌드된 `.whl`, `.tar.gz` 파일 첨부

### 릴리즈 절차
1. CHANGELOG.md 작성
2. pyproject.toml 버전 업데이트
3. 커밋 & master에 push
4. 태그 생성 (`v0.0.x`) & push → Actions가 PyPI 배포 + GitHub Release 자동 생성
5. 자동 생성 실패 시 수동으로 GitHub Release 작성

## 기술 스택

- Python 3.10+, NumPy, Pandas, SciPy
- Numba (선택적 JIT 가속)
- PyPI 패키지명: `vectrix`
- GitHub: `eddmpython/vectrix`

## 실험 (Experiments) 규칙

### 폴더 구조
- 경로: `src/vectrix/experiments/`
- 기존 e001~e011: `legacy/` 서브폴더에 보관
- 새 실험: 연구 방향별 서브폴더 (camelCase)
  - `metaSelection/` — DNA 메타러닝 모델 선택
  - `adaptiveInterval/` — 적응형 예측 구간
  - `ensembleEvolution/` — 차세대 앙상블
  - `dynamicFlat/` — 동적 Flat Defense
  - `frequencyDomain/` — 주파수 도메인 예측
  - `modelCreation/` — 신규 예측 모델 창조
- 공통 유틸리티: `_utils/` 서브폴더 (데이터 생성, 메트릭)

### 파일 규칙
- **네이밍**: `XXX_camelCaseFeature.py` (폴더별 001부터 시작)
- **번호 폴더별 독립**: 각 폴더 안에서 001, 002, 003 순서대로 부여
- **독립 실행 필수**: 각 파일은 `if __name__ == '__main__':` 블록으로 직접 실행 가능
- **결과는 docstring에 기록**: 별도 결과 파일 생성 금지
- **실패한 실험도 기록**: 가설이 기각되면 "왜 실패했는지" 결론에 명시

### Docstring 구조 (필수)

```
실험 ID: 폴더명/XXX
실험명: 제목

목적:
- 해결하려는 문제, 기대 인사이트

가설:
1. 정량적 기준 포함한 가설

방법:
1. 구체적 절차, 사용 데이터, 비교 대상, 측정 지표

결과 (실험 후 작성):
- 수치 테이블

결론:
- 채택/기각 판단, 모듈화 여부

실험일: YYYY-MM-DD
```

### 벤치마크 기준
- **M4 Competition**: 주요 벤치마크, OWA 기준
- **최소 조건**: Naive2 대비 OWA < 1.0
- **최소 데이터**: 50개 시리즈 이상 (5개 미만은 예비 실험 표기)

### 실험 → 모듈화 전환
1. 실험 성공 → 핵심 로직을 `src/vectrix/[해당모듈]/`로 추출
2. `tests/`에 테스트 추가
3. `__init__.py`에 export
4. CHANGELOG.md에 기록

## 모델 창조 연구 (Model Creation Research)

### 배경
4개 전문 에이전트(통계이론, 신호처리/물리학, ML/정보이론, M4 Competition) 토론을 통해 12개 신규 모델 후보를 도출. 교차 분석으로 우선순위 확정.

### 핵심 원칙 (4개 에이전트 수렴 결론)
1. **"다르게 틀리는" 모델 필요** — 잔차 상관 0.73~1.0 해결이 최우선
2. **비모수 방법론이 돌파구** — DTSF, SSA, KernelDensity 모두 비모수
3. **변화점 적응은 "절단" 아닌 "가중"** — ensembleEvolution/003 실패 교훈 반영
4. **M4 Hourly/Daily가 최대 개선 지점** — 4개 에이전트 만장일치

### 구현 우선순위 (Top 5)

| 순위 | 실험 | 모델 | 출처 | 핵심 원리 |
|------|------|------|------|----------|
| 1 | mc/001 | DynamicTimeScanForecaster | M4 에이전트 | 비모수 패턴 매칭, Hourly sMAPE 12.9% (벤치마크 22.1%) |
| 2 | mc/002 | KoopmanModeForecaster | 물리학 에이전트 | DMD(동적 모드 분해), 비선형→선형 리프팅, 감쇠 진동 |
| 3 | mc/003 | WaveletShrinkageForecaster | 물리학 에이전트 | DWT + Donoho-Johnstone 축소, 시간-주파수 동시 분해 |
| 4 | mc/004 | AdaptiveThetaEnsemble | M4 에이전트 | M4 3위 4Theta 재현, 다중 theta line 가중 조합 |
| 5 | mc/005 | SingularSpectrumForecaster | 통계 에이전트 | SVD 비모수 분해, 주기 자동 발견, 재귀 예측 |

### 대기 후보 (Top 5 완료 후)

| 순위 | 모델 | 출처 | 핵심 |
|------|------|------|------|
| 6 | EchoStateForecaster | ML 에이전트 | Reservoir Computing, 비선형 잔차 직교 |
| 7 | DampedTrendWithChangepoint | 통계 에이전트 | 변경점 가중 감쇠 ETS, 안전 설계 |
| 8 | KernelDensityForecaster | ML 에이전트 | 시간지연 임베딩 + 커널 가중 이웃 |
| 9 | BayesianChangeForecaster | ML 에이전트 | BOCPD + 혼합 예측, 다봉 불확실성 |
| 10 | TemporalAggregation | 통계 에이전트 | MAPA 다중 시간 집계, M3/M4 검증 |
| 11 | FeatureWeightedCombiner | M4 에이전트 | FFORMA M4 2위, 특성 기반 메타러닝 |
| 12 | StochasticResonance | 물리학 에이전트 | 이중 우물 퍼텐셜, 레짐 전환 예측 |

### 각 모델 상세

**mc/001 DynamicTimeScanForecaster (DTSF)**
- 알고리즘: 마지막 W개 값과 유사한 과거 패턴 K개 탐색 → 패턴 직후 값들의 중앙값 = 예측
- 구현: numpy sliding_window_view + 거리 벡터화, 구현 난이도 하~중
- 기대: Hourly OWA 0.75~0.85 (현재 1.006), Daily OWA 0.85~0.95 (현재 1.207)
- 리스크: 짧은 시계열(n<100)에서 유사 패턴 부족

**mc/002 KoopmanModeForecaster (DMD)**
- 알고리즘: Takens 임베딩 → SVD 기반 DMD → 쿠프만 고유값/모드 추출 → 모드 기반 예측
- 각 고유값 λ = |λ|·exp(iθ): |λ|=감쇠율, θ=진동주파수
- 구현: numpy.linalg.svd + eig, 구현 난이도 중
- 기대: 비선형 진동/감쇠 데이터에서 기존 모델 대비 우위
- 리스크: 임베딩 차원 선택 민감, 불안정 모드 필터링 필요

**mc/003 WaveletShrinkageForecaster**
- 알고리즘: Haar DWT → 스케일별 소프트 축소(λ=σ√(2ln n)) → 스케일별 독립 외삽 → IDWT
- 핵심: 시간-주파수 동시 분해, Gibbs 현상 없음 (FFT의 약점 보완)
- 구현: numpy 배열 연산만으로 Haar DWT 구현, 구현 난이도 하
- 기대: 비정상/다중스케일 데이터에서 FourierForecaster 보완
- 리스크: 2의 거듭제곱 제약 (패딩 필요)

**mc/004 AdaptiveThetaEnsemble (4Theta)**
- 알고리즘: 4개 theta line(θ=0,1,2,3) × 최적 SES → holdout sMAPE 역수 가중 결합
- M4 3위 방법론의 numpy 재현
- 구현: 기존 ThetaModel 인프라 재활용, 구현 난이도 중
- 기대: Daily OWA 0.95~1.00, Monthly/Yearly 강세
- 리스크: 짧은 시계열에서 4-way validation 데이터 부족

**mc/005 SingularSpectrumForecaster (SSA)**
- 알고리즘: 궤적 행렬 구성 → SVD → 자동 그룹화 → Hankel 대각 평균 → 재귀 예측
- 완전 비모수: 어떤 모델 가정도 없이 데이터가 구조를 드러냄
- 구현: numpy.linalg.svd + 대각 평균, 구현 난이도 중~상
- 기대: Hourly 다중 주기/비정현파 패턴 자동 발견
- 리스크: 윈도우 길이 L 선택, 자동 그룹화 품질

### 실험 폴더
- modelCreation/001~019: `src/vectrix/experiments/modelCreation/`
- 각 연구 방향별 폴더에 독립 번호 체계

## 새로운 접근법 연구 (Novel Approaches, 2026-03-03~)

### 배경
기존 모든 모델(ETS, ARIMA, Theta, CES 등)은 "시계열 = 트렌드 + 계절성 + 잔차" 분해,
"과거 패턴 반복" 가정, "단일 시계열만 관찰"이라는 공통 프레임에 갇혀 있음.
이 프레임을 근본적으로 깨는 5가지 방향을 탐색.

### 핵심 원칙
1. **기존 통계 모델과 근본적으로 다른 원리** — 잔차 상관 0 기대
2. **모델-프리** — 분해 가정 없이 데이터 자체에서 예측
3. **구현 단순성** — numpy + 표준 라이브러리만으로 구현
4. E013~E015 교훈: 복잡할수록 실패. 극도의 단순함 추구

### 구현 우선순위 (Top 5)

| 순위 | 실험 | 모델 | 핵심 원리 |
|------|------|------|----------|
| 1 | mc/020 | CompressionForecaster | 예측=압축. zlib 압축률이 가장 높은 연속이 최적 예측 |
| 2 | mc/021 | TopologicalForecaster | TDA persistent homology로 궤적의 위상 구조 기반 예측 |
| 3 | mc/022 | GravitationalForecaster | 과거 값이 미래에 중력 행사, N-body 평형점 = 예측 |
| 4 | mc/023 | EvolutionaryForecaster | 유전 알고리즘으로 예측 후보 자연선택 |
| 5 | mc/024 | CausalEntropyForecaster | MaxEnt 원리: 인과 제약 하 엔트로피 최대화 = 예측 |

### 각 모델 상세

**mc/020 CompressionForecaster**
- 알고리즘: 시계열 → 바이트 직렬화 → 후보값 연결 → zlib 압축률 비교 → 최소 압축 크기 = 최적 예측
- Kolmogorov complexity의 실용적 근사
- 완전 모델-프리: 어떤 통계 가정도 없음
- 구현: numpy + zlib, 난이도 하
- 기대: 기존 모델과 잔차 상관 ~0, 앙상블 다양성 극대화

**mc/021 TopologicalForecaster**
- 알고리즘: 시간지연 임베딩 → 고차원 궤적 → persistent homology → Betti numbers → 형태 유사 구간 탐색 → 예측
- 값이 아닌 "궤적의 기하학적 구조"로 예측
- 구현: numpy + scipy, 난이도 중
- 기대: 비선형/혼란 시계열에서 우위

**mc/022 GravitationalForecaster**
- 알고리즘: 각 과거 값 = 질량 입자, 시간 감쇠 + 패턴 유사도로 인력 결정, 평형점 = 예측
- N-body simulation의 시계열 적용
- 구현: numpy, 난이도 하~중
- 기대: 자연스러운 불확실성 구간 (인력 분포 = 예측 분포)

**mc/023 EvolutionaryForecaster**
- 알고리즘: 무작위 후보 생성 → 과거 일관성 기준 선택 → 교배/돌연변이 → 반복 → 생존자 평균
- Genetic Algorithm 기반 예측
- 구현: numpy, 난이도 하
- 기대: 모델 구조 불필요, 적합성 함수만 정의

**mc/024 CausalEntropyForecaster**
- 알고리즘: 과거 패턴에서 조건부 제약 추출 → 제약 하 MaxEnt 분포 → 기댓값 = 예측
- Jaynes의 최대 엔트로피 원리
- 구현: numpy + scipy.optimize, 난이도 중
- 기대: "가장 놀랍지 않은" 예측 = 과적합 방지

## 아키텍처 설계 사상 (2026-03-04 확정)

### [최우선] 결합도 최소화 — "1곳 수정 = 1곳만 수정"

**과거 문제**: 모델 추가 시 5곳 수정 필요 (NATIVE_MODELS, _fitAndPredictNativeWithCache, _refitModelOnFullData, _selectNativeModels, MODEL_INFO) → 스파게티 코드, 연쇄 버그

**해결: 레지스트리 패턴 (engine/registry.py)**

```
engine/registry.py  ← 모든 모델 메타데이터의 Single Source of Truth
  │
  ├── ModelSpec: modelId, name, description, factory, needsPeriod, minData, flatResistance, bestFor
  ├── getRegistry() → Dict[str, ModelSpec]
  ├── createModel(modelId, period) → model instance
  └── getModelInfo() → backward-compatible MODEL_INFO dict
```

**새 모델 추가 절차 (1곳만 수정)**:
1. `engine/newmodel.py` 작성 (fit/predict 인터페이스)
2. `engine/registry.py`에 ModelSpec 1개 추가
3. 끝. vectrix.py, types.py, easy.py, docs 자동 반영

**절대 금지**:
- vectrix.py에 모델별 if/elif 분기 추가
- types.py에 모델 메타데이터 중복
- easy.py에 모델 로직 중복 구현

### 모듈 역할 분리 — 단방향 의존성

```
easy.py → vectrix.py → engine/registry.py → engine/*.py
   │                         │
   └── types.py ←────────────┘
```

- **easy.py**: 사용자 인터페이스 전용. 데이터 파싱 + Vectrix 래핑. 자체 로직 없음
- **vectrix.py**: 오케스트레이터. 분석/모델선택/학습/앙상블 조율. 모델 생성은 registry 위임
- **engine/registry.py**: 모델 메타데이터 중앙 저장소. 모든 모델 정보의 유일한 출처
- **engine/*.py**: 순수 예측 엔진. 외부 의존성 없음. fit(y) → predict(steps) 계약만 준수
- **types.py**: 데이터 타입 정의만. 모델 메타데이터 보관 금지 (registry.py로 이전)

### 엔진 인터페이스 계약

모든 예측 모델은 이 계약을 반드시 준수:

```python
class ForecastModel:
    def fit(self, y: np.ndarray) -> self:
        """학습. y는 1D float64 배열."""

    def predict(self, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """예측. (predictions, lower95, upper95) 반환."""
```

- `fit()`은 `self` 반환 (체이닝 가능)
- `predict()`는 항상 3-tuple 반환
- 생성자에서 `period` 파라미터는 선택적 (registry.ModelSpec.needsPeriod로 관리)
- 학습 전 predict() 호출 시 ValueError raise

### 확장/유지보수 원칙

#### API 설계 원칙
1. **Progressive Disclosure**: Level 1(제로설정) → Level 2(가이드 제어) → Level 3(엔진 직접)
2. **새 파라미터 추가 시 반드시 기본값 제공** — `forecast(data, steps=12)`는 영원히 동작해야 함
3. **Easy API 파라미터는 Vectrix 클래스의 기능을 투과** — easy.py가 vectrix.py를 래핑, 중복 구현 금지
4. **파라미터 네이밍**: Easy API는 snake_case 허용 (models, ensemble, confidence), 내부는 camelCase

#### 엔진 확장 원칙
1. **새 모델 = registry.py에 ModelSpec 1개 추가** — 다른 파일 수정 불필요
2. **M4 100K 벤치마크 통과 필수** — OWA < 1.0 (Naive2 대비) 확인 후 통합
3. **잔차 다양성 우선** — 기존 모델과 잔차 상관 < 0.5인 모델이 앙상블에 가치 있음
4. **engine/__init__.py에서 export** + tests/ 테스트 추가

#### 속도 확장 원칙
1. **핫 루프 식별 → Rust 이전** — profiling으로 병목 확인 후 rust/src/lib.rs에 추가
2. **Python 오버헤드 최소화** — 모델 선택/CV 로직의 불필요한 복사/변환 제거
3. **벤치마크 측정 필수** — 변경 전후 `forecast()` 전체 latency 비교

#### 정확도 확장 원칙
1. **앙상블 전략이 단일 모델보다 중요** — DNA 기반 가중치, 잔차 다양성 활용
2. **빈도별 전략 분리** — Yearly/Quarterly는 Theta계열, Hourly/Daily는 다중 계절성 특화
3. **실험 → 검증 → 통합 파이프라인** — experiments/에서 실험, M4로 검증, engine/으로 통합

#### 문서/마케팅 원칙
1. **모든 주장에 벤치마크 수치 첨부** — "빠르다"가 아닌 "5.6x faster (295ms → 52ms)"
2. **블로그는 교육 중심** — 기초부터 설명, Vectrix 홍보는 자연스럽게 녹여냄
3. **비교 표는 공정하게** — 경쟁사의 장점도 인정, 우리가 약한 부분도 투명하게 공개

## 약점 및 개선 필요사항 (2026-03-03 업데이트)

### [긴급] 정확도 — 가장 큰 문제
- **M4 Daily OWA 1.207** — Naive2보다 20.7% 나쁨. 이대로면 벤치마크 공개가 역효과
- **M4 Hourly OWA 1.006** — Naive2와 동일. 30+ 모델이 무의미
- **목표**: Daily OWA < 0.95, Hourly OWA < 0.85
- **원인**: 고빈도 다중 계절성(일/주/연) 처리 부족, 노이즈 높은 데이터에서 CV 과적합
- statsforecast는 Daily 0.92, Hourly 0.78 수준 — 이것이 현실적 목표

### [중요] 속도 — forecast() 전체 기준
- Rust 67x는 ETS filter 단일 루프 기준. forecast() 전체는 48ms
- statsforecast AutoETS는 0.3ms (Numba JIT) — 100배 차이
- Rust 내장했지만 Polars급 속도를 주장하려면 더 많은 핫 루프 이전 필요
- SES, ARIMA, Theta 등 핫 루프 외에 Python 오버헤드(모델 선택, CV, 앙상블) 최적화 필요

### [중요] 킬러 유스케이스 부재
- "30+ 모델"은 기본 능력이지 차별점이 아님
- VX-Ensemble Hourly OWA 0.696이 유일한 세계급 성과
- Forecast DNA + Flat Defense가 독자적이지만 마케팅에 활용 부족
- 한 가지를 세계 최고로 만들어야 함 — "이 문제에는 Vectrix" 포지셔닝

### [중요] 커뮤니티/인지도
- PyPI 다운로드 수 거의 0
- GitHub stars 미미
- 혼자 개발 (bus factor 1)
- 기업 지원 경쟁사(Nixtla, Meta, Unit8)와 규모 싸움은 불가능
- 블로그/Reddit/Kaggle/HackerNews 노출 전략 필요

### 업계 대비 기능 격차 (우선순위순)
1. **M4 Daily/Hourly 정확도** — 벤치마크 신뢰도 직결 (난이도: 중)
2. **속도 최적화** — forecast() 전체 파이프라인 Rust 이전 (난이도: 중)
3. **Foundation Model** — TimesFM 2.5, Chronos-2, Moirai-2 래퍼 (이미 있지만 깊이 부족)
4. **다변량 VAR/VECM** — ARIMAX만 존재 (난이도: 중)
5. **이벤트/휴일 효과** — Prophet 스타일 캘린더 모델링 부재 (난이도: 하)

### 강점 (유지해야 할 것)
- 적응형 예측 (레짐, 자가치유, DNA) — 독자적 차별화
- 비즈니스 인텔리전스 (시나리오, 백테스트, 이상치) — 실무 특화
- 3개 의존성 + 내장 Rust 엔진 — 설치 최간편
- AI 에이전트 통합 (llms.txt, MCP, Skills) — 업계 최선두
- M4 Hourly VX-Ensemble OWA 0.696 — 세계급

## 아이덴티티 원칙 (2026-03-04 확정)

- **"Pure Python" 표현 사용 금지** — Rust 엔진이 내장된 패키지
- **30+ 모델은 기본 능력** — 차별점으로 내세우지 않음
- **Rust는 투명** — Polars처럼 사용자가 의식하지 않아도 빠름
- **Python 문법, Rust 속도** — 이것이 정체성
- **Progressive Disclosure** — 초보자는 제로설정, 전문가는 완전 제어. 같은 함수, 같은 패키지
- **벤치마크 기반 정직성** — 약한 부분(Daily/Hourly)도 투명하게 공개. 수치로 증명

## 개선 로드맵 (2026-03-03~)

### [완료] Rust 엔진 내장 (v0.0.8)
- 25개 함수 Rust 가속, 4개 플랫폼 wheel, `pip install vectrix` = Rust 포함
- "optional turbo" → "built-in Rust engine" 메시징 전환 완료

### [다음] M4 Daily/Hourly 정확도 개선
- Daily OWA 1.207 → 0.95 이하, Hourly OWA 1.006 → 0.85 이하
- MSTL 고주파 다중 계절성 최적화 (일/주/연 3중 계절)
- 다중 계절 모델(TBATS, MSTL) 앙상블 전략 재설계

### [대기] forecast() 전체 파이프라인 속도
- 현재 48ms → 목표 5ms 이하
- 모델 선택/CV/앙상블 로직의 Python 오버헤드 Rust 이전
- Numba 의존성 제거 고려 (Rust로 완전 대체)

### [대기] 킬러 유스케이스 확립
- VX-Ensemble Hourly를 중심으로 "Hourly/고빈도 예측 = Vectrix" 포지셔닝
- 또는 "Zero-config + AI-native forecasting" 포지셔닝

### [대기] 커뮤니티 성장
- 기술 블로그 포스트 (M4 벤치마크 재현, Rust 가속 여정)
- Reddit r/MachineLearning, r/datascience 포스팅
- Kaggle 노트북 예제
