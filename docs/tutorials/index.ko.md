# 튜토리얼

인터랙티브 [marimo](https://marimo.io) 노트북. 로컬에서 실행하면 완전한 대화형 환경을 사용할 수 있습니다.

## 실행 방법

```bash
pip install "vectrix[tutorials]"
marimo run docs/tutorials/ko/01_quickstart.py
```

## 튜토리얼 목록

| # | 주제 | English | 한국어 |
|---|------|---------|--------|
| 01 | 빠른 시작 | `en/01_quickstart.py` | `ko/01_quickstart.py` |
| 02 | 분석 & DNA | `en/02_analyze.py` | `ko/02_analyze.py` |
| 03 | 회귀분석 | `en/03_regression.py` | `ko/03_regression.py` |
| 04 | 30+ 모델 | `en/04_models.py` | `ko/04_models.py` |
| 05 | 적응형 지능 | `en/05_adaptive.py` | `ko/05_adaptive.py` |
| 06 | 비즈니스 인텔리전스 | `en/06_business.py` | `ko/06_business.py` |

## 튜토리얼 설명

### 01 — 빠른 시작
리스트, DataFrame, CSV에서 예측. `.predictions`, `.plot()`, `.to_csv()` 결과 탐색.

### 02 — 분석 & DNA
시계열 자동 프로파일링: 난이도, 카테고리, 핑거프린트, 변화점, 이상치.

### 03 — 회귀분석
R 스타일 수식 회귀. OLS, Ridge, Lasso, Huber, Quantile. 진단 포함.

### 04 — 30+ 모델
`Vectrix` 클래스 직접 사용. 모든 모델 비교, Flat Defense 시스템 이해.

### 05 — 적응형 지능
레짐 감지, Forecast DNA, 자가 치유 예측, 비즈니스 제약 조건.

### 06 — 비즈니스 인텔리전스
이상치 탐지, What-if 시나리오, 백테스팅, 비즈니스 지표.
