---
hide:
  - navigation
---

# Vectrix

<div style="text-align: center; margin: 2em 0;">
<p style="font-size: 1.4em; font-weight: 300; color: var(--md-default-fg-color--light);">
설정 없는 Python 시계열 예측
</p>
<p style="font-size: 1.1em;">
<strong>30+ 모델</strong> · <strong>의존성 3개</strong> · <strong>코드 1줄</strong>
</p>
</div>

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
```

이것이 전부입니다. 모델 자동 선택, 평탄 예측 방어, 신뢰구간, 시각화 — 한 줄에 모두 포함.

[:material-download: 설치](getting-started/installation.md){ .md-button .md-button--primary }
[:material-rocket-launch: 빠른 시작](getting-started/quickstart.md){ .md-button }
[:material-github: GitHub](https://github.com/eddmpython/vectrix){ .md-button }

---

## 왜 Vectrix?

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } **제로 설정**

    ---

    데이터만 넣으면 예측이 나옵니다. 하이퍼파라미터 튜닝, 수동 모델 선택, 설정 파일 불필요.

-   :material-language-python:{ .lg .middle } **순수 Python**

    ---

    `numpy` + `scipy` + `pandas`만 사용. 컴파일된 확장 없음, 모든 플랫폼에서 설치 가능.

-   :material-shield-check:{ .lg .middle } **평탄 예측 방어**

    ---

    고유한 4단계 방어 시스템으로 예측 실패의 가장 흔한 원인인 상수/평탄 예측을 방지.

-   :material-dna:{ .lg .middle } **Forecast DNA**

    ---

    65+ 특성 핑거프린팅. 데이터의 난이도, 최적 모델, 다른 시계열과의 유사도를 파악.

-   :material-brain:{ .lg .middle } **적응형 지능**

    ---

    레짐 감지, 자가 치유 예측, 8가지 비즈니스 제약 유형. 현실에 적응하는 예측.

-   :material-chart-bell-curve-cumulative:{ .lg .middle } **확률적 예측**

    ---

    파라메트릭 분포(가우시안, Student-t, 로그정규), 분위수 예측, CRPS 스코어링.

</div>

---

## 기능 비교

| 기능 | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| 제로 설정 예측 | :white_check_mark: | :white_check_mark: | :x: | :x: |
| 순수 Python | :white_check_mark: | :x: | :x: | :x: |
| 30+ 통계 모델 | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| 평탄 예측 방어 | :white_check_mark: | :x: | :x: | :x: |
| Forecast DNA | :white_check_mark: | :x: | :x: | :x: |
| 비즈니스 제약 (8종) | :white_check_mark: | :x: | :x: | :x: |
| R 스타일 회귀 | :white_check_mark: | :x: | :x: | :x: |
| Foundation Model 래퍼 | :white_check_mark: | :x: | :x: | :white_check_mark: |
| 파이프라인 시스템 | :white_check_mark: | :x: | :x: | :white_check_mark: |
| VAR / VECM 다변량 | :white_check_mark: | :x: | :x: | :white_check_mark: |

---

## 주요 기능

### :material-chart-line: [예측](guide/forecasting.md)
30+ 모델 자동 선택 — ETS, ARIMA, Theta, MSTL, TBATS, GARCH, Croston 등. 모두 순수 NumPy.

### :material-dna: [분석 & DNA](guide/analysis.md)
65+ 특성 자동 핑거프린팅, 난이도 점수화, 최적 모델 추천.

### :material-function-variant: [회귀분석](guide/regression.md)
R 스타일 수식 인터페이스: `regress(df, "sales ~ ads + price")` — OLS, Ridge, Lasso, Huber, Quantile.

### :material-brain: [적응형 지능](guide/adaptive.md)
레짐 감지(HMM), 자가 치유 예측(CUSUM + EWMA), 8가지 비즈니스 제약 유형.

### :material-briefcase: [비즈니스 인텔리전스](guide/business.md)
이상치 탐지, What-if 시나리오, 백테스팅, 실전 지표(WAPE, MASE, 바이어스).

### :material-pipe: [파이프라인](guide/pipeline.md)
sklearn 스타일 `ForecastPipeline` — 변환기(로그, 스케일, 탈계절화)를 체이닝하고 예측 시 자동 역변환.

### :material-robot: [Foundation Models](guide/foundation.md)
Amazon Chronos-2, Google TimesFM 2.5 래퍼 — 사전 훈련 모델로 제로샷 예측.

### :material-chart-multiple: [다변량](guide/multivariate.md)
VAR 자동 시차 선택, 그랜저 인과성 검정, 공적분 시계열용 VECM.

---

## 벤치마크

M3/M4 예측 대회 결과 (OWA < 1.0 = Naive2 능가):

| 대회 | Yearly | Quarterly | Monthly | Weekly | Daily | Hourly |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| **M3** | **0.848** | **0.825** | **0.758** | — | — | **0.819** |
| **M4** | **0.974** | **0.797** | **0.987** | **0.737** | 1.207 | 1.006 |

[:material-chart-bar: 전체 벤치마크 결과](benchmarks.md)

---

## 설치

=== "Core"

    ```bash
    pip install vectrix
    ```
    NumPy + SciPy + Pandas만. Python 3.10+.

=== "Numba"

    ```bash
    pip install "vectrix[numba]"
    ```
    핵심 알고리즘 JIT 컴파일로 2-5배 가속.

=== "ML"

    ```bash
    pip install "vectrix[ml]"
    ```
    LightGBM, XGBoost, scikit-learn 앙상블 모델 추가.

=== "Foundation Models"

    ```bash
    pip install "vectrix[foundation]"
    ```
    Amazon Chronos-2, Google TimesFM 2.5 래퍼.

=== "전체"

    ```bash
    pip install "vectrix[all]"
    ```
    모든 선택적 의존성 포함.

---

<div style="text-align: center; margin: 2em 0; color: var(--md-default-fg-color--light);">
<a href="https://github.com/eddmpython/vectrix/blob/master/LICENSE">MIT 라이선스</a> — 개인 및 상업 프로젝트에서 자유롭게 사용.
</div>
