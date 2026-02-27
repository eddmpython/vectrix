# Vectrix

**순수 Python 시계열 예측 엔진**

30+ 모델 · 3개 의존성 · 코드 1줄

---

## 빠른 시작

```bash
pip install vectrix
```

```python
from vectrix import forecast

result = forecast("sales.csv", steps=12)
print(result)
result.plot()
```

한 줄이면 모델 선택, 일직선 예측 방지, 신뢰구간까지 포함된 예측이 완성됩니다.

---

## 왜 Vectrix?

| 차원 | Vectrix | statsforecast | Prophet | Darts |
|:--|:--:|:--:|:--:|:--:|
| **Zero-config** | :white_check_mark: | :white_check_mark: | :x: | :x: |
| **순수 Python** | :white_check_mark: | :x: | :x: | :x: |
| **30+ 모델** | :white_check_mark: | :white_check_mark: | :x: | :white_check_mark: |
| **평탄 예측 방어** | :white_check_mark: | :x: | :x: | :x: |
| **스트레스 테스트** | :white_check_mark: | :x: | :x: | :x: |
| **Forecast DNA** | :white_check_mark: | :x: | :x: | :x: |
| **제약 조건 (8종)** | :white_check_mark: | :x: | :x: | :x: |
| **R 스타일 회귀** | :white_check_mark: | :x: | :x: | :x: |

**벡터 3개.** `numpy` · `scipy` · `pandas` — 그것이 전체 궤도입니다.

---

## 주요 기능

### :material-chart-line: [예측](guide/forecasting.md)
30+ 모델 자동 선택. ETS, ARIMA, Theta, MSTL, TBATS, GARCH 등.

### :material-dna: [분석 & DNA](guide/analysis.md)
시계열 자동 핑거프린팅, 난이도 점수, 최적 모델 추천.

### :material-function-variant: [회귀분석](guide/regression.md)
R 스타일 수식 인터페이스. OLS, Ridge, Lasso, Huber, Quantile + 진단.

### :material-brain: [적응형 지능](guide/adaptive.md)
레짐 감지, 자가 치유, 비즈니스 제약 조건, Forecast DNA.

### :material-briefcase: [비즈니스 인텔리전스](guide/business.md)
이상치 탐지, What-if 분석, 백테스팅, 비즈니스 지표.

---

## 설치

```bash
pip install vectrix                # 핵심 (numpy + scipy + pandas)
pip install "vectrix[numba]"       # + Numba JIT (2-5배 가속)
pip install "vectrix[ml]"          # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[all]"         # 전체
```

---

## 라이선스

[MIT](https://github.com/eddmpython/vectrix/blob/master/LICENSE) — 개인 및 상업 프로젝트에서 자유롭게 사용 가능합니다.
