<p align="center">
  <h1 align="center">Vectrix</h1>
  <p align="center">
    <strong>설정 없이 바로 사용하는 시계열 예측 라이브러리</strong>
  </p>
  <p align="center">
    <a href="#설치">설치</a> &middot;
    <a href="#빠른-시작">빠른 시작</a> &middot;
    <a href="#주요-기능">주요 기능</a> &middot;
    <a href="#api-레퍼런스">API</a> &middot;
    <a href="README.md">English</a>
  </p>
</p>

---

Vectrix는 순수 NumPy + SciPy로 처음부터 구현한 시계열 예측 라이브러리입니다. statsforecast, statsmodels, prophet 없이 — 데이터만 넣으면 최적 예측과 신뢰구간을 제공합니다.

```python
from vectrix import Vectrix

fx = Vectrix()
result = fx.forecast(df, dateCol="date", valueCol="sales", steps=30)
print(result.predictions)
```

한 줄이면 모델 선택, 일직선 예측 방지, 신뢰구간까지 포함된 예측이 완성됩니다.

## 왜 Vectrix?

| 라이브러리 | 문제점 |
|-----------|-------|
| **statsforecast** | 무거운 의존성, C 컴파일 문제 |
| **Prophet** | pystan/cmdstan 필요, 200MB+ 설치 |
| **statsmodels** | 수동 모델 선택, 자동 파이프라인 없음 |
| **NeuralProphet** | PyTorch 필요, GPU 지향 |
| **darts** | 과도한 기능, 느린 시작 |

Vectrix는 **순수 Python, 의존성 3개, MIT 라이선스**로 프로덕션 수준의 예측을 제공합니다.

## 주요 기능

- **Zero-Config** — DataFrame만 넣으면 자동 최적 예측. 파라미터 튜닝 불필요
- **자체 구현 엔진** — AutoETS, AutoARIMA, Theta, MSTL 전부 NumPy로 직접 구현
- **일직선 예측 방지** — 4단계 방어 시스템으로 일직선 예측 방지
- **다중 계절성** — 여러 계절 패턴 자동 감지 및 분해
- **정기 드롭 감지** — 반복되는 하락 패턴 (휴일, 점검 기간) 자동 처리
- **신뢰구간** — 예측 기간에 따라 넓어지는 95% 예측 구간
- **변동성 보존 앙상블** — 원본 데이터의 변동성을 보존하면서 모델 결합
- **Numba 가속** — 선택적 Numba JIT으로 대용량 데이터에서 2-5배 속도 향상

## 설치

### uv 사용 (권장)

```bash
uv init my-forecast && cd my-forecast
uv add vectrix

# Numba 가속 포함
uv add "vectrix[numba]"
```

### pip 사용

```bash
pip install vectrix

# Numba 포함
pip install "vectrix[numba]"
```

## 빠른 시작

### 1. 기본 예측

```python
import pandas as pd
from vectrix import Vectrix

df = pd.read_csv("sales.csv")

fx = Vectrix()
result = fx.forecast(df, dateCol="date", valueCol="sales", steps=30)

if result.success:
    print(f"선택 모델: {result.bestModelName}")
    print(f"향후 30일: {result.predictions}")
    print(f"경고: {result.warnings}")
```

### 2. 데이터 분석만

예측 없이 데이터 특성만 분석:

```python
fx = Vectrix()
analysis = fx.analyze(df, dateCol="date", valueCol="sales")

chars = analysis["characteristics"]
print(f"주기: {chars.period}일")
print(f"추세: {chars.trendDirection} (강도={chars.trendStrength:.2f})")
print(f"계절성: {chars.seasonalStrength:.2f}")
print(f"예측 가능성: {chars.predictabilityScore:.0f}/100")
```

### 3. 상세 로그 모드

```python
fx = Vectrix(verbose=True)
result = fx.forecast(df, dateCol="date", valueCol="value", steps=30)
```

## 작동 원리

4단계 파이프라인으로 일반적인 예측 실패를 방지합니다:

1. **AutoAnalyzer** — 주기, 추세, 계절성, 정상성, 변동성 자동 감지
2. **FlatRiskDiagnostic** — 일직선 예측 위험도 평가 및 모델 선택
3. **모델 학습** — 자체 구현 모델 풀에서 학습 및 평가 (AutoETS, AutoARIMA, Theta, MSTL, Seasonal Naive)
4. **앙상블** — 변동성 보존 가중 결합

## API 레퍼런스

### `Vectrix(verbose=False)`

| 매개변수 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `verbose` | bool | `False` | 상세 로그 출력 |
| `nJobs` | int | `-1` | 병렬 처리 수 (-1 = 모든 코어) |

### `fx.forecast(df, dateCol, valueCol, steps=30, trainRatio=0.8)`

`ForecastResult` 반환:
- `success` — 예측 성공 여부
- `predictions` — 예측값 (numpy array)
- `dates` — 예측 날짜
- `lower95`, `upper95` — 95% 신뢰구간
- `bestModelName` — 선택된 모델명
- `characteristics` — 데이터 특성
- `warnings` — 경고 메시지

### `fx.analyze(df, dateCol, valueCol)`

`characteristics`와 `flatRisk`를 포함한 dict 반환.

## 의존성

| 패키지 | 필수 | 용도 |
|-------|------|------|
| numpy | O | 핵심 연산 |
| pandas | O | 데이터 처리 |
| scipy | O | 파라미터 최적화 |
| numba | X | JIT 가속 (2-5배) |

## 라이선스

MIT License
