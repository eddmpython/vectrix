# experienceIndex — 경험 역인덱스 기반 예측 전략 학습

## 정체성

파운데이션 모델은 "예측 자체"를 학습한다. 우리는 "예측 전략"을 학습한다.
대량의 시리즈에 대해 실제로 예측을 수행하고, 그 경험을 역인덱스로 축적한다.
새 시리즈가 들어오면 비슷한 경험을 검색하여 전략을 가져온다.

## 핵심 아이디어

### 검색 엔진의 역인덱스에서 착안

정방향: 시리즈 → DNA 특성 (분석)
역방향: DNA → 경험 목록 (조회)

### dataProfiling과의 차이

- dataProfiling: DNA → 모델/전처리를 "매핑 함수"로 학습 (Ridge, 규칙). 실패함
- experienceIndex: DNA를 키로 "경험 자체"를 저장하고 "검색"한다. 매핑 함수 없음

### 왜 작동하는가 (E002에서 확인)

1. **DNA 65차원이 결정적** — 4차원 간이 DNA는 시리즈를 구분 못 함. 65차원은 가능
2. **검색이 매핑보다 낫다** — Ridge(전역 함수)는 실패했지만 kNN(국소 검색)은 성공
3. **경험이 축적될수록 개선** — 500개 +3.1% → 4500개 +4.6%, 학습 곡선 미포화
4. **선택 > 블렌딩** — 경험 기반 "모델 선택"은 작동, "가중 블렌딩"은 역효과

## 경험 DB 저장 구조 (설계)

```
data/experience/
├── schema.json          ← DNA 특성 목록, 모델 목록, 버전
├── monthly.jsonl        ← Monthly 경험 로그 (append-only, git 추적)
├── yearly.jsonl         ← Yearly
├── quarterly.jsonl
├── weekly.jsonl
├── daily.jsonl
├── hourly.jsonl
└── _build/
    └── monthly.parquet  ← 빌드된 검색용 파일 (.gitignore)
```

- `.jsonl`은 git에 올린다 — append-only, diff 가능
- `.parquet`은 로컬 빌드 — 검색 속도용, `.gitignore`
- `schema.json`으로 스키마 변경 추적
- 빈도별 분리 — 같은 빈도 안에서만 매칭

## 실험 로드맵

### Phase A: 기초 검증 (E001~E002) ✅

| 번호 | 실험명 | 질문 |
|------|--------|------|
| 001 | bucketConsistency | 간이 DNA 4d로 가능한가? |
| 002 | fullDnaExperience | 실제 DNA 65d + 5000개로 가능한가? |

### Phase B: 스케일과 구조 (E003~E005)

| 번호 | 실험명 | 질문 |
|------|--------|------|
| 003 | scaleAndDomain | 48K 포화점 어디? 도메인 분리가 필요한가? |
| 004 | crossFrequency | Yearly/Quarterly에서도 작동하는가? |
| 005 | candidateExpansion | 후보 8→16+개 확장이 도움되는가? |

### Phase C: 최적화 (E006~E007)

| 번호 | 실험명 | 질문 |
|------|--------|------|
| 006 | dnaImportance | 65d 중 어떤 특성이 가장 중요한가? |
| 007 | kOptimization | 최적 k값이 DB 크기에 비례하는가? |

### Phase D: 엔진 통합 (E008~)

| 번호 | 실험명 | 질문 |
|------|--------|------|
| 008 | productionPipeline | JSONL 저장 + 로드 + 검색 파이프라인 |
| 009 | engineIntegration | forecast()에 경험 DB 통합 시 전체 OWA |

## 현황

| 실험 | 상태 | 결과 |
|------|------|------|
| 001 | 완료 | 간이 DNA 4d 실패. 버킷 purity 0.348, kNN 전부 악화(-0.3~-2.8%). DNA 차원 부족 |
| 002 | 완료 | **돌파!** 65d DNA kNN(k=50) +3.72%, 학습곡선 500→4500개 +3.1%→+4.6% 단조증가 |
| 003 | 진행중 | M4 Monthly 48K 전체, 학습 곡선 포화점 + 도메인 분석 |
| 004 | 대기 | |
| 005 | 대기 | |
| 006 | 대기 | |
| 007 | 대기 | |

## 핵심 발견

### E001
- 간이 DNA 4차원(seasonality, trend, acf1, lengthRatio)으로는 시리즈 구분 불가
- 버킷 purity 최대 0.348 (8후보 중 35%만 일치)
- kNN 전부 악화: k=5 -2.84%, k=50 -0.31%
- 버킷 역인덱스: ±0.00% (DOT와 동일)
- **교훈**: DNA 차원이 부족하면 검색도 안 된다

### E002 ★★★
- **65d DNA + 5000개 경험으로 kNN +3.72% 달성!**
- dataProfiling 15개 실험 최고(rolling holdout +0.96%)의 **4배**
- kNN(k=50) OWA 0.8279, Oracle gap 17.1% 캡처
- 학습 곡선 단조 증가: 500개 +3.1% → 4500개 +4.6% (미포화)
- 경험 "선택"은 성공(+4.6%), 경험 "블렌딩"은 실패(-10.6%)
- **E001→E002의 반전**: 4d에서 전부 악화 → 65d에서 전부 개선. DNA 차원이 핵심
