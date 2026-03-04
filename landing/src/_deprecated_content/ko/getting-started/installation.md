---
title: 설치
---

# 설치

Vectrix는 **Rust 엔진이 내장**되어 있습니다 — Polars처럼 Rust 확장이 wheel에 컴파일되어 있어 별도 설치가 필요 없습니다. `pip install`만 하면 바로 빠릅니다.

## 요구 사항

- **Python 3.10 이상** (3.11+ 권장, 최적 성능)
- **OS:** Windows, macOS, Linux — Apple Silicon 포함 모든 플랫폼 지원

## 설치

원하는 패키지 매니저를 선택하세요:

**pip** (가장 일반적)

```bash
pip install vectrix
```

**uv** (가장 빠름)

```bash
uv add vectrix
```

**conda / mamba** (PyPI 경유)

```bash
pip install vectrix
```

이 명령은 3개의 핵심 의존성(NumPy, pandas, SciPy)과 **Rust 엔진**을 함께 설치합니다. C 컴파일러, CUDA, 무거운 프레임워크가 필요 없습니다.

## 선택적 추가 기능

Vectrix는 모듈식 설계를 따릅니다 — 필요한 것만 설치하세요:

```bash
pip install vectrix                  # 30+ 모델 + Rust 엔진 내장
pip install "vectrix[ml]"            # + LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"    # + Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[all]"           # 전체
```

| Extra | 추가되는 기능 | 사용 시점 |
|:------|:-------------|:---------|
| `ml` | LightGBM, XGBoost, scikit-learn | 머신러닝 모델 후보 추가 |
| `foundation` | Chronos-2, TimesFM 2.5 | 제로샷 파운데이션 모델 예측 |
| `neural` | NeuralForecast (N-BEATS, N-HiTS, TFT) | 딥러닝 모델 |

## 내장 Rust 엔진

25개 핵심 예측 핫 루프가 Rust로 가속되어 모든 wheel에 컴파일됩니다. 사전 빌드 바이너리 wheel이 모든 주요 플랫폼에 제공됩니다:

- **Linux** x86_64 (manylinux)
- **macOS** Apple Silicon (ARM64) + x86_64
- **Windows** x86_64
- **Python** 3.10, 3.11, 3.12, 3.13

Rust 컴파일러 불필요. 가속은 완전히 투명합니다 — 코드 변경 없이 설치만 하면 자동으로 빨라집니다. Vectrix는 import 시점에 Rust 엔진을 자동 감지합니다.

| 구성요소 | Python Only | Rust 적용 | 속도 향상 |
|:---------|:-----------|:----------|:---------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| DOT fit | 240ms | **10ms** | **24x** |
| ETS filter (핫 루프) | 0.17ms | **0.003ms** | **67x** |

## 설치 확인

설치 후 정상 작동을 확인하세요:

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

Rust 엔진이 로드되었는지 확인:

```python
import vectrix
print(vectrix.TURBO_AVAILABLE)  # True면 Rust 엔진 활성화
```

## 핵심 의존성

Vectrix는 경량화를 지향합니다. 3개의 패키지만 필요합니다 — 모두 널리 사용되고 잘 관리되는 과학 Python 라이브러리입니다:

| 패키지 | 최소 버전 | 용도 |
|--------|----------|------|
| numpy | >= 1.24.0 | 배열 연산 및 선형대수 |
| pandas | >= 2.0.0 | 시계열 데이터 처리 |
| scipy | >= 1.10.0 | 통계 분포 및 최적화 |
