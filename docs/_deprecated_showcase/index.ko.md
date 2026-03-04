# 쇼케이스

**공개 데이터**를 활용한 실전 예측 예제.
각 쇼케이스에는 상세 가이드와 로컬에서 실행 가능한 인터랙티브 marimo 노트북이 있습니다.

## 쇼케이스 목록

| # | 주제 | 가이드 | 인터랙티브 |
|---|------|--------|------------|
| 01 | [한국 경제 예측](01_koreanEconomy.md) | FRED 경제 지표 | `marimo run docs/showcase/ko/01_koreanEconomy.py` |
| 02 | [한국 데이터 회귀분석](02_koreanRegression.md) | 자전거 대여 + 거시경제 | `marimo run docs/showcase/ko/02_koreanRegression.py` |
| 03 | [모델 비교](03_modelComparison.md) | 30+ 모델 병렬 비교 | `marimo run docs/showcase/ko/03_modelComparison.py` |
| 04 | [비즈니스 인텔리전스](04_businessIntelligence.md) | 이상치, 시나리오, 백테스팅 | `marimo run docs/showcase/ko/04_businessIntelligence.py` |

## 인터랙티브 실행 방법

쇼케이스는 [marimo](https://marimo.io) — 리액티브 Python 노트북으로 제작되었습니다.

```bash
pip install vectrix pandas numpy marimo
marimo run docs/showcase/ko/01_koreanEconomy.py
```

또는 위 가이드를 클릭하여 이 사이트에서 직접 코드와 설명을 확인하세요.

## 쇼케이스 설명

### 01 — 한국 경제 데이터 예측

FRED 데이터를 활용한 한국 경제 지표 예측:

- **원/달러 환율** — 1981년부터 월간 데이터, 12개월 예측
- **KOSPI 주가지수** — 월간 주가지수, 12개월 예측
- **소비자물가지수 (CPI)** — 인플레이션 추적, 12개월 예측
- **다중 지표 DNA 분석** — 난이도와 특성 비교

### 02 — 한국 실제 데이터 회귀분석

한국 데이터셋 회귀분석:

- **서울 자전거 대여량** — 8,760개 시간별 관측치, 기상 요인으로 예측 (UCI ML Repository)
- **한국 거시경제 회귀** — FRED 지표를 활용한 환율 결정요인

### 03 — 모델 비교 & 적응형 지능

30+ 예측 모델 전체 비교:

- **DNA 분석** — 데이터 난이도, 카테고리, 핑거프린트
- **모델 순위** — 전체 모델의 MAPE, RMSE, MAE, sMAPE
- **전체 예측 DataFrame** — 모든 모델의 예측값 병렬 확인
- **빠른 비교** — `compare()` 함수로 한 줄 비교

### 04 — 비즈니스 인텔리전스

End-to-end 비즈니스 워크플로우:

- **이상치 탐지** — 비정상 데이터 포인트 발견
- **What-If 시나리오** — 성장, 경기침체, 공급 충격 분석
- **백테스팅** — Walk-forward 모델 검증
- **비즈니스 지표** — MAPE, RMSE, MAE, 편향, 추적 신호

## 데이터 출처

| 출처 | URL | 인증 필요 |
|------|-----|:---------:|
| FRED | `fred.stlouisfed.org` | 불필요 |
| UCI ML Repository | `archive.ics.uci.edu` | 불필요 |

!!! note "주의사항"
    이 분석은 교육 목적이며, 실제 투자나 사업 결정에 사용하지 마세요.
