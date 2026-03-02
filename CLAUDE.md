# Vectrix 프로젝트 규칙

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

### 쇼케이스 (`docs/showcase/`)
- **marimo .py 파일로 작성** + marimo skills 사용 + `marimo check` 검증
- 한영 이중 파일: `en/`, `ko/` 서브폴더
- GitHub Pages에서는 **쇼케이스 전용 .md 페이지에 코드+설명+출력 예시를 Markdown으로 재구성**
- `showcase/index.md`에서 각 쇼케이스 설명 + 코드 하이라이트 + marimo 실행 안내

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
- modelCreation/001~012: `src/vectrix/experiments/modelCreation/`
- 각 연구 방향별 폴더에 독립 번호 체계

## 약점 분석 (2026-02-28 기준)

### 벤치마크 정확도
- **M4 Daily OWA 1.207** — Naive2보다 20.7% 나쁨 (sMAPE 3.254 vs 2.652)
- **M4 Hourly OWA 1.006** — Naive2와 동일 수준 (패배)
- 원인: 고빈도 다중 계절성(일/주/연) 처리 부족, 노이즈 비율 높은 데이터에서 CV 과적합

### 업계 대비 기능 격차 (우선순위순)
1. **M4 Daily/Hourly 정확도** — 벤치마크 신뢰도 직결 (난이도: 중)
2. **Foundation Model 미지원** — TimesFM 2.5, Chronos-2, Moirai-2 등 제로샷 예측이 업계 최대 트렌드 (난이도: 상)
3. **속도** — StatsForecast 대비 10~100x 느림, Spark/Dask/Ray 분산 미지원 (난이도: 중)
4. **딥러닝 모델 전무** — NBEATS, NHITS, TFT 등 M4 우승 모델 부재 (난이도: 상)
5. **확률적 예측 부족** — 파라메트릭 분포 출력, 분위수 예측, CRPS 메트릭 없음 (난이도: 중)
6. **다변량 VAR/VECM** — ARIMAX만 존재, VAR/VECM/다변량 GARCH 없음 (난이도: 중)
7. **이벤트/휴일 효과** — Prophet 스타일 캘린더 기반 모델링 부재 (난이도: 하)
8. **파이프라인 시스템** — 전처리→모델→후처리 조합 시스템 없음 (난이도: 중)

### 경쟁사 대비 강점 (유지해야 할 것)
- 적응형 예측 (레짐, 자가치유, DNA) — 독자적 차별화
- 비즈니스 인텔리전스 (시나리오, 백테스트, 이상치) — 실무 특화
- 순수 NumPy/SciPy — 의존성 최소, 설치 간편
- Conformal/Bootstrap 구간 — 이미 구현됨
- 계층적 조정 (BottomUp, TopDown, MinTrace) — 이미 구현됨

## 개선 로드맵 (2026-02-28~)

### [진행중] Rust 확장 모듈 (vectrix-core)
- **목표**: AutoETS fit 348ms → 5ms 이하, 전체 forecast() 0.6s → 0.05s
- **방법**: PyO3 + maturin, ETS state update / ARIMA likelihood / Theta decomposition 핫 패스 Rust 포팅
- **배포**: GitHub Actions에서 manylinux/macOS/Windows wheel pre-build
- **Python fallback**: `try: from vectrix_core import ... except ImportError: pure Python`
- **설치**: `pip install vectrix` (순수 Python) / `pip install vectrix[fast]` (Rust 가속)

### [대기] M4 Daily/Hourly 정확도 개선
- Daily OWA 1.207 → 0.95 이하, Hourly OWA 1.006 → 0.85 이하
- MSTL 고주파 다중 계절성 최적화 (일/주/연 3중 계절)
- M4 전체 데이터셋 벤치마크 실행 & 공개

### [대기] 전체 데이터셋 벤치마크
- M4 100,000 시리즈 전체 벤치마크
- statsforecast/Darts/Prophet 대비 속도 + 정확도 비교표

### [대기] 영문 docstring 전환
- 소스코드 docstring 한국어 → 영어 전환
- API 문서 사이트에서 영문 페이지에 한글이 노출되는 문제 해결

### [대기] 테스트 커버리지 강화
- 현재 387개 → 500개 이상 목표
- 엣지 케이스, 수치 안정성 테스트 추가
