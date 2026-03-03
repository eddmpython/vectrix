---
title: 설치
---

# 설치

## 요구 사항

- Python 3.10+
- OS: Windows, macOS, Linux

## 설치

**pip**

```bash
pip install vectrix
```

**uv**

```bash
uv add vectrix
```

## 선택적 추가 기능

```bash
pip install "vectrix[turbo]"       # Rust 가속 (5-10배 속도 향상, Rust 컴파일러 불필요)
pip install "vectrix[numba]"       # Numba JIT 가속 (2-5배 속도 향상)
pip install "vectrix[ml]"          # LightGBM, XGBoost, scikit-learn
pip install "vectrix[foundation]"  # Amazon Chronos-2, Google TimesFM 2.5
pip install "vectrix[tutorials]"   # 인터랙티브 marimo 튜토리얼
pip install "vectrix[all]"         # 전체
```

## Rust Turbo Mode

`turbo` 옵션은 Rust로 컴파일된 네이티브 확장 `vectrix-core`를 설치합니다. 핵심 예측 루프가 5-10배 빨라집니다. 사전 빌드 wheel 제공

- Linux (x86_64, manylinux)
- macOS (x86_64 + Apple Silicon ARM)
- Windows (x86_64)
- Python 3.10, 3.11, 3.12, 3.13

Rust 컴파일러 불필요. 코드 변경 없이 설치만 하면 자동으로 빨라집니다.

| 구성요소 | Turbo 없음 | Turbo 적용 | 속도 향상 |
|:---------|:----------|:----------|:---------|
| `forecast()` 200pts | 295ms | **52ms** | **5.6x** |
| AutoETS fit | 348ms | **32ms** | **10.8x** |
| AutoARIMA fit | 195ms | **35ms** | **5.6x** |

## 설치 확인

```python
import vectrix
from vectrix import forecast

result = forecast([100, 120, 130, 115, 140], steps=3)
print(result.predictions)
```

## 핵심 의존성

Vectrix는 3개의 필수 의존성만 있습니다

| 패키지 | 최소 버전 |
|--------|----------|
| numpy | >= 1.24.0 |
| pandas | >= 2.0.0 |
| scipy | >= 1.10.0 |
